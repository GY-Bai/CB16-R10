from __future__ import annotations
import os, json, math, hashlib, platform, tarfile, shutil, copy, io, sys
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

SEED=24680
REGIMES=['2025_06_reference','2025_early_independent','2025_late_consumed','2026_early_consumed']
TAUS=np.array([.1,.5,.9],float)
BASE=dict(n_estimators=120,learning_rate=.05,num_leaves=15,min_child_samples=20,reg_lambda=.1,subsample=.8,subsample_freq=1,colsample_bytree=.8,n_jobs=1,verbosity=-1)
ROOT=Path('/mnt/data/CB16_ORDERED4H30_PREDICTIVE_RISK_ADVISOR_V1_R2')
if ROOT.exists(): shutil.rmtree(ROOT)
ROOT.mkdir(parents=True)
PORT=ROOT/'portable_candidate'; PORT.mkdir()
INPUT=Path('/mnt/data/cb16_taskc_inputs')
ZPATH=INPUT/'horizon/CB16_CONTROL_CONSEQUENCE_HORIZON_V1_R1/authority_cache/AUTHORITATIVE_MULTI_HORIZON_TARGET_RAW_R1.npz'
OPATH=INPUT/'coarse/S1_ORDERED_4H30_ENDPOINTS.npy'
SUPERVISOR=INPUT/'control_plane/risk_supervisor_r1.py'
PAIR_MANIFEST=INPUT/'action_r0/COUNTERFACTUAL_PAIR_MANIFEST.json'
R2_RESULT=INPUT/'r2/ORDERED4H30_ACTION_RISK_R2_RESULT.json'
R2_MANIFEST=INPUT/'r2/RUN_MANIFEST.json'
R1_ADVISOR=INPUT/'advisor_r1/CB16_PREDICTIVE_RISK_ADVISOR_STATISTICAL_CONTROL_V1_R1/RISK_ADVISOR_STATISTICAL_CONTROL_R1_RESULT.json'
R1_GATE=INPUT/'advisor_r1/CB16_PREDICTIVE_RISK_ADVISOR_STATISTICAL_CONTROL_V1_R1/TASK_GATE_CONTRACT_V1.json'
R1_MONO=INPUT/'advisor_r1/CB16_PREDICTIVE_RISK_ADVISOR_STATISTICAL_CONTROL_V1_R1/MONOTONE_INTERVENTION_FAMILY_V1.json'
R1_OOD=INPUT/'advisor_r1/CB16_PREDICTIVE_RISK_ADVISOR_STATISTICAL_CONTROL_V1_R1/OOD_ABSTENTION_CONTRACT.json'

EXPECTED={
 'ordered4h30':'738ed11011a7749b0ddf182ed5538d2c6c0b118ab24bb2bbf206c9a65e4a5048',
 'target_cache':'7e0b691d60d0b6691f0af4f06428d2bb303e824eebe358c4817361da2e334a7b',
 'risk_supervisor':'1fee663d23d400dca7900df5b839fdebe126b9190bac02069f4ea4a70451e9f6',
 'account_schema':'6f9437c7ca198fda026bae9e1de03f7aef5878768807a772929b59bbed47a890',
 'action_schema':'5581a26a75b0b8b56ff24f9b196ea6b4398e760ed09d1d6fcadc9290a9db02ac',
 'physics':'d1da4242141dc2e5aa257eddc50a2573b8d41471faa3e6270acb596e6c459321',
 'pair_manifest':'2aaf73126a4678e9cb0d4d0fe75da1ae97924bedcf7ccd310c8715e2200cb084',
}

def sha(path):
 h=hashlib.sha256();
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(8<<20),b''): h.update(b)
 return h.hexdigest()
def canonical(obj): return json.dumps(obj,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
def objsha(obj): return hashlib.sha256(canonical(obj)).hexdigest()
def writej(path,obj): Path(path).write_text(json.dumps(obj,indent=2,sort_keys=True,ensure_ascii=False,allow_nan=False)+'\n')
def clip_prob(p): return np.clip(np.asarray(p,float),1e-6,1-1e-6)
def logit(p):
 p=clip_prob(p); return np.log(p/(1-p))
def robust_scale(y):
 y=np.asarray(y,float);q25,q75=np.quantile(y,[.25,.75]);iqr=q75-q25;med=np.median(y);mad=np.median(np.abs(y-med))*1.4826;sd=np.std(y)*.1
 return float(max(iqr,mad,sd,1e-6))
def ece(y,p,bins=10):
 y=np.asarray(y,float);p=np.asarray(p,float);n=len(y)
 if n==0:return float('nan')
 order=np.argsort(p);out=0.
 for ix in np.array_split(order,min(bins,n)):
  if len(ix):out += len(ix)/n*abs(y[ix].mean()-p[ix].mean())
 return float(out)
def pinball(y,q,t):
 e=np.asarray(y)-np.asarray(q); return np.maximum(t*e,(t-1)*e)

# Integrity / authority
for key,path in [('ordered4h30',OPATH),('target_cache',ZPATH),('risk_supervisor',SUPERVISOR),('pair_manifest',PAIR_MANIFEST)]:
 got=sha(path)
 if got!=EXPECTED[key]: raise RuntimeError(f'{key} hash mismatch {got}')
r2=json.load(open(R2_RESULT));r2m=json.load(open(R2_MANIFEST));adv=json.load(open(R1_ADVISOR));gate=json.load(open(R1_GATE));mono=json.load(open(R1_MONO));ood=json.load(open(R1_OOD));pairmanifest=json.load(open(PAIR_MANIFEST))
assert r2['primary_status']=='ORDERED4H30_TASK_GATED_ACTION_INFORMATION_PRESERVED'
assert r2['packet']['dimension']==30 and r2['packet']['sha256']==EXPECTED['ordered4h30']
assert set(r2['confirmatory'])=={'flat_vs_short','risk_multiplier_long','risk_multiplier_short'}
assert adv['current_activation']=='SHADOW_ADVISORY_ONLY'

# data
z=np.load(ZPATH)
ordered=np.load(OPATH).astype(np.float32)
assert ordered.shape==(1088,30) and np.isfinite(ordered).all()
obs=np.asarray(z['obs'],np.float32)
meta=pd.DataFrame({k:z[k] for k in ['market_i','snapshot_sha','regime','asset','timestamp','account_stratum','action_id','decision','risk_multiplier']})
for k in ['snapshot_sha','regime','asset','timestamp','account_stratum','action_id']: meta[k]=meta[k].astype(str)
for j in range(6): meta[f'obs{j}']=obs[:,j]

# exact split from inherited evaluator
def make_splits(meta0):
 ue=meta0[['market_i','regime','asset','timestamp']].drop_duplicates('market_i').sort_values('market_i').reset_index(drop=True)
 sets={k:{} for k in ['fit','cal','exam']}; rec=[]
 for reg in REGIMES:
  ts=np.array(sorted(ue.loc[ue.regime.eq(reg),'timestamp'].unique()));cut=max(1,int(math.floor(.60*len(ts))));tr=list(ts[:cut]);cand=list(ts[cut:]);last=pd.Timestamp(str(ts[cut-1]))
  ex=[str(x) for x in cand if pd.Timestamp(str(x))>last+pd.Timedelta(hours=128)];inner=max(1,int(math.floor(.75*len(tr))))
  sets['fit'][reg]=set(map(str,tr[:inner]));sets['cal'][reg]=set(map(str,tr[inner:]));sets['exam'][reg]=set(ex)
  rec.append({'regime':reg,'endpoint_timestamps':len(ts),'fit_timestamps':len(sets['fit'][reg]),'calibration_timestamps':len(sets['cal'][reg]),'exam_timestamps':len(ex),'last_train_timestamp':str(last),'first_exam_timestamp':str(min(ex)),'gap_hours':float((pd.Timestamp(min(ex))-last)/pd.Timedelta(hours=1))})
 masks={k:np.zeros(len(meta0),bool) for k in sets}
 for reg in REGIMES:
  rr=meta0.regime.eq(reg)
  for k in sets: masks[k] |= (rr & meta0.timestamp.isin(sets[k][reg])).to_numpy()
 return masks,pd.DataFrame(rec),sets

masks_action, splitdf, splitsets=make_splits(meta)
fitA,calA,exA=masks_action['fit'],masks_action['cal'],masks_action['exam']
A0=np.c_[obs,meta.decision.to_numpy(float),meta.risk_multiplier.to_numpy(float)].astype(np.float32)
XA=np.c_[A0,ordered[meta.market_i.to_numpy(int)]].astype(np.float32)
yh64=np.asarray(z['y_H64'],float)
y_stress=(yh64[:,4]>=.20).astype(int)
y_margin=yh64[:,4].astype(float)

# model helpers (same family + held-out calibration)
def fit_binary(Xfit,yfit,Xcal,ycal):
 yfit=np.asarray(yfit,int);ycal=np.asarray(ycal,int)
 if len(np.unique(yfit))<2: raise RuntimeError('binary fit lacks classes')
 m=lgb.LGBMClassifier(objective='binary',random_state=SEED,**BASE);m.fit(Xfit,yfit)
 pcal0=clip_prob(m.predict_proba(Xcal)[:,1]);lr=None
 if len(ycal)>=6 and len(np.unique(ycal))==2 and min((ycal==0).sum(),(ycal==1).sum())>=3:
  lr=LogisticRegression(C=1e6,solver='lbfgs',max_iter=500,random_state=SEED);lr.fit(logit(pcal0).reshape(-1,1),ycal)
 return m,lr
def pred_binary(comp,X):
 m,lr=comp;p=clip_prob(m.predict_proba(X)[:,1])
 if lr is not None:p=clip_prob(lr.predict_proba(logit(p).reshape(-1,1))[:,1])
 return p
def fit_quantiles(Xfit,yfit,Xcal,ycal):
 out=[]
 for tau in TAUS:
  m=lgb.LGBMRegressor(objective='quantile',alpha=float(tau),random_state=SEED,**BASE);m.fit(Xfit,yfit)
  pc=m.predict(Xcal);corr=float(np.quantile(np.asarray(ycal)-pc,float(tau))) if len(ycal) else 0.
  out.append((m,corr))
 return out
def pred_quantiles(mods,X,scale):
 q=np.stack([m.predict(X)+corr for m,corr in mods],1)*scale
 return np.sort(q,axis=1)

# action-level H64 heads inherited from Coarse R1 D3 family
stress_comp=fit_binary(XA[fitA],y_stress[fitA],XA[calA],y_stress[calA])
margin_scale=robust_scale(y_margin[fitA]);margin_mods=fit_quantiles(XA[fitA],y_margin[fitA]/margin_scale,XA[calA],y_margin[calA]/margin_scale)

# pair rows exactly as R2
PAIRS=pairmanifest['pairs'];PAIR_BY={p['pair_id']:p for p in PAIRS};PAIR_INDEX={p['pair_id']:i for i,p in enumerate(PAIRS)}
flat={'FLAT_BASE','FLAT_LOW_RISK','FLAT_DRAWDOWN'};rows=[]
for (mi,ss),g in meta.groupby(['market_i','snapshot_sha'],sort=True):
 stratum=str(g.account_stratum.iloc[0]);ix0=int(g.index[0]);budget=float(obs[ix0,4])
 if stratum not in flat: continue
 amap={str(a):int(i) for a,i in zip(g.action_id,g.index)}
 for p in PAIRS:
  if p['A'] in amap and p['B'] in amap and float(p['required_risk_budget'])<=budget+1e-7:
   ia,ib=amap[p['A']],amap[p['B']]
   rows.append({'market_i':int(mi),'snapshot_sha':str(ss),'regime':str(g.regime.iloc[0]),'asset':str(g.asset.iloc[0]),'timestamp':str(g.timestamp.iloc[0]),'account_stratum':stratum,'pair_id':p['pair_id'],'pair_family':p['family'],'row_A':ia,'row_B':ib,'risk_budget_remaining_fraction':budget,'current_drawdown_fraction':float(obs[ix0,3]),'margin_utilization_fraction':float(obs[ix0,5])})
pm=pd.DataFrame(rows)
for j in range(6):pm[f'obs{j}']=[float(obs[int(i),j]) for i in pm.row_A]
masksP, splitP, setsP=make_splits(pm);fitP,calP,exP=masksP['fit'],masksP['cal'],masksP['exam']
one=np.zeros((len(pm),len(PAIRS)),np.float32)
for i,pid in enumerate(pm.pair_id): one[i,PAIR_INDEX[pid]]=1
XP=np.c_[pm[[f'obs{j}' for j in range(6)]].to_numpy(np.float32),one,ordered[pm.market_i.to_numpy(int)]].astype(np.float32)
sa=(yh64[pm.row_A.to_numpy(int),4]>=.20).astype(int);sb=(yh64[pm.row_B.to_numpy(int),4]>=.20).astype(int)
disc=(sa!=sb).astype(int);direc=((sa==1)&(sb==0)).astype(int)
disc_comp=fit_binary(XP[fitP],disc[fitP],XP[calP],disc[calP])
fdi=fitP&(disc==1);cdi=calP&(disc==1)
dir_comp=fit_binary(XP[fdi],direc[fdi],XP[cdi],direc[cdi])

def pair_rho(X): return pred_binary(disc_comp,X)*pred_binary(dir_comp,X)

# candidate metadata and graph
ACTION_IDS={'S025':(0,.25),'FLAT':(1,0.),'L025':(2,.25),'S075':(0,.75),'L075':(2,.75)}
SUPPORTED_EDGES=[
 {'edge_id':'LONG_075_TO_025','pair_id':'L075_MINUS_L025','family':'risk_multiplier_long','from':'L075','to':'L025','level':1},
 {'edge_id':'SHORT_075_TO_025','pair_id':'S075_MINUS_S025','family':'risk_multiplier_short','from':'S075','to':'S025','level':1},
 {'edge_id':'SHORT_025_TO_FLAT','pair_id':'S025_MINUS_FLAT','family':'flat_vs_short','from':'S025','to':'FLAT','level':2},
 {'edge_id':'SHORT_075_TO_FLAT','pair_id':'S075_MINUS_FLAT','family':'flat_vs_short','from':'S075','to':'FLAT','level':2},
]
EDGE_BY_FROM={}
for e in SUPPORTED_EDGES: EDGE_BY_FROM.setdefault(e['from'],[]).append(e)
SUPPORT_VERSION='CB16_ORDERED4H30_ADVISOR_SUPPORT_V1_R2'
MODEL_VERSION='PredictiveRiskAdvisorCandidateV1.R2'
CAL_VERSION='DEVELOPMENT_CALIBRATION_R2_CONSUMED_SUPPORT'

candidate={
 'schema':'CB16_PREDICTIVE_RISK_ADVISOR_FROZEN_CANDIDATE_V1',
 'model_version':MODEL_VERSION,
 'mode':'SHADOW_LOG_ONLY',
 'candidate_family':'INHERITED_LIGHTGBM_DISTRIBUTIONAL_READOUT_ONLY',
 'seed':SEED,'base_params':BASE,'quantiles':TAUS.tolist(),
 'input_lineage':{'representation':'Ordered4H30','dimension':30,'representation_sha256':EXPECTED['ordered4h30'],'account_state_schema_sha256':EXPECTED['account_schema'],'action_intent_schema_sha256':EXPECTED['action_schema'],'physics_contract_sha256':EXPECTED['physics'],'support_version':SUPPORT_VERSION},
 'target':{'primary_event':'margin_stress_20pp_by_H64','definition':'1{max_margin_utilization_excess_H64 >= 0.20}','continuous_companion':'max_margin_utilization_excess_H64'},
 'feature_contract':{'action_level':['AccountState6','decision','risk_multiplier','Ordered4H30'],'action_feature_dim':38,'pair_level':['AccountState6','8-way frozen pair identity','Ordered4H30'],'pair_feature_dim':44},
 'supported_families':['flat_vs_short','risk_multiplier_long','risk_multiplier_short'],
 'supported_edges':SUPPORTED_EDGES,
 'unsupported':['flat_vs_long','long_vs_short','LONG@0.25->FLAT','positioned account','low-envelope 0.25 subgroup','non-anchor risk'],
 'calibration_status':'DEVELOPMENT_ONLY',
 'formal_crc_certificate':False,
 'active_authority':False,
 'models':{'stress':stress_comp,'margin_quantiles':margin_mods,'margin_scale':margin_scale,'pair_discordance':disc_comp,'pair_direction':dir_comp},
}

# inference functions
def action_feature(order30,account6,action_id):
 d,r=ACTION_IDS[action_id];return np.asarray(list(account6)+[d,r]+list(order30),dtype=np.float32).reshape(1,-1)
def pair_feature(order30,account6,pid):
 o=np.zeros(8,np.float32);o[PAIR_INDEX[pid]]=1;return np.r_[np.asarray(account6,np.float32),o,np.asarray(order30,np.float32)].reshape(1,-1)
def pred_action(cand,order30,account6,action_id):
 X=action_feature(order30,account6,action_id); p=float(pred_binary(cand['models']['stress'],X)[0]);q=pred_quantiles(cand['models']['margin_quantiles'],X,cand['models']['margin_scale'])[0]
 return {'action_id':action_id,'P_Z64_margin_stress':p,'max_margin_quantiles':{'q10':float(q[0]),'q50':float(q[1]),'q90':float(q[2])}}
def infer(cand,inp):
 out={'schema':'PredictiveRiskAdvisorCandidateV1Output','mode':'SHADOW_LOG_ONLY','task_gate_status':'NO_ADVICE','supported_edge_set':[],'predicted_risk':None,'predicted_relief':{},'suggested_intervention_level':None,'NO_ADVICE_reason':None,'original_ActionIntent_unchanged':True,'model_calibration_lineage':{'model_version':MODEL_VERSION,'calibration_version':CAL_VERSION,'support_version':SUPPORT_VERSION}}
 try:
  o=np.asarray(inp['Ordered4H30'],float);a=np.asarray(inp['AccountState6'],float);intent=inp['ActionIntentV1'];env=inp['static_external_envelope_metadata'];lin=inp['lineage_support_version']
 except Exception:
  out['NO_ADVICE_reason']='MISSING_REQUIRED_FIELD';return out
 if o.shape!=(30,) or a.shape!=(6,) or (not np.isfinite(o).all()) or (not np.isfinite(a).all()):out['NO_ADVICE_reason']='NONFINITE_OR_SHAPE_INVALID';return out
 if lin.get('representation_sha256')!=EXPECTED['ordered4h30'] or lin.get('representation')!='Ordered4H30':out['NO_ADVICE_reason']='REPRESENTATION_VERSION_MISMATCH';return out
 if lin.get('account_state_schema_sha256')!=EXPECTED['account_schema'] or lin.get('action_intent_schema_sha256')!=EXPECTED['action_schema'] or lin.get('support_version')!=SUPPORT_VERSION:out['NO_ADVICE_reason']='LINEAGE_SUPPORT_VERSION_MISMATCH';return out
 if abs(float(a[0]))>1e-7 or float(a[5])>1e-7:out['NO_ADVICE_reason']='POSITIONED_ACCOUNT';return out
 try:d=int(intent['direction']);r=float(intent['requested_risk_multiplier']);ext=float(env['external_envelope_fraction'])
 except Exception:out['NO_ADVICE_reason']='INVALID_ACTION_OR_ENVELOPE';return out
 if not (np.isfinite(r) and np.isfinite(ext)):out['NO_ADVICE_reason']='NONFINITE_ACTION_OR_ENVELOPE';return out
 if abs(float(a[4])-1.0)>1e-6 or ext<1.0-1e-6:out['NO_ADVICE_reason']='LOW_OR_UNSUPPORTED_EXTERNAL_ENVELOPE';return out
 amap={(0,.25):'S025',(0,.75):'S075',(1,0.):'FLAT',(2,.25):'L025',(2,.75):'L075'}
 aid=None
 for (dd,rr),name in amap.items():
  if d==dd and abs(r-rr)<=1e-8:aid=name;break
 if aid is None:out['NO_ADVICE_reason']='UNSUPPORTED_ACTION_FAMILY_OR_RISK_ANCHOR';return out
 if aid not in EDGE_BY_FROM:out['NO_ADVICE_reason']='UNSUPPORTED_FAMILY__NO_FROZEN_MORE_CONSERVATIVE_EDGE';return out
 if r>ext+1e-8:out['NO_ADVICE_reason']='REQUEST_EXCEEDS_EXTERNAL_ENVELOPE';return out
 edges=EDGE_BY_FROM[aid];out['task_gate_status']='SUPPORTED_SHADOW_PREDICTION';out['supported_edge_set']=[e['edge_id'] for e in edges]
 out['predicted_risk']=pred_action(cand,o,a,aid)
 for e in edges:
  rho=float(pair_rho(pair_feature(o,a,e['pair_id']))[0]);dst=pred_action(cand,o,a,e['to'])
  out['predicted_relief'][e['edge_id']]={'rho_A_to_B':rho,'destination_predicted_risk':dst,'level':e['level'],'from':e['from'],'to':e['to']}
 out['suggested_intervention_level']='UNSET__NO_FORMAL_CONTROL_THRESHOLD__SHADOW_ONLY'
 return out

# Candidate hashes excluding Python estimator unserializable JSON content
meta_freeze={k:v for k,v in candidate.items() if k!='models'}
meta_freeze['model_components']=['action_stress_LGBM+Platt','action_max_margin_quantile_LGBM_q10_q50_q90+heldout_offsets','pair_discordance_LGBM+Platt','pair_direction_LGBM+Platt']
meta_freeze['candidate_contract_sha256']=objsha(meta_freeze)
candidate['candidate_contract_sha256']=meta_freeze['candidate_contract_sha256']

# Freeze joblib
FROZEN=ROOT/'PREDICTIVE_RISK_ADVISOR_CANDIDATE_V1_FROZEN.joblib'
joblib.dump(candidate,FROZEN,compress=3,protocol=4)

# Portable LightGBM components + calibrators
def export_bin(name,comp):
 m,lr=comp;(PORT/f'{name}.lgb.txt').write_text(m.booster_.model_to_string())
 cal={'type':'PLATT_LOGIT' if lr is not None else 'IDENTITY','coef':lr.coef_.tolist() if lr is not None else None,'intercept':lr.intercept_.tolist() if lr is not None else None}
 writej(PORT/f'{name}.calibration.json',cal)
def export_quant(name,mods):
 arr=[]
 for tau,(m,corr) in zip(TAUS,mods):
  fn=f'{name}_q{int(tau*100):02d}.lgb.txt';(PORT/fn).write_text(m.booster_.model_to_string());arr.append({'tau':float(tau),'model_file':fn,'correction':float(corr)})
 return arr
export_bin('action_stress_H64',stress_comp);export_bin('pair_discordance_H64',disc_comp);export_bin('pair_direction_H64',dir_comp)
qmanifest=export_quant('action_max_margin_H64',margin_mods)
portable_manifest={**meta_freeze,'portable_schema':'CB16_PREDICTIVE_RISK_ADVISOR_PORTABLE_CANDIDATE_V1','margin_scale':margin_scale,'quantile_models':qmanifest,'files':{}}
for p in sorted(PORT.glob('*')):
 if p.is_file():portable_manifest['files'][p.name]={'sha256':sha(p),'bytes':p.stat().st_size}
writej(PORT/'PORTABLE_CANDIDATE_MANIFEST.json',portable_manifest)

# Portable loader for reload canary
def load_booster(path): return lgb.Booster(model_file=str(path))
def load_cal(path): return json.load(open(path))
def portable_bin(name,X):
 b=load_booster(PORT/f'{name}.lgb.txt');p=clip_prob(b.predict(X));c=load_cal(PORT/f'{name}.calibration.json')
 if c['type']=='PLATT_LOGIT':
  coef=float(c['coef'][0][0]);inter=float(c['intercept'][0]);s=inter+coef*logit(p);p=clip_prob(1/(1+np.exp(-s)))
 return p
def portable_q(X):
 qs=[]
 for q in qmanifest:
  b=load_booster(PORT/q['model_file']);qs.append((b.predict(X)+q['correction'])*margin_scale)
 return np.sort(np.stack(qs,1),axis=1)
def portable_rho(X): return portable_bin('pair_discordance_H64',X)*portable_bin('pair_direction_H64',X)

# Development calibration diagnostics on consumed exam support
# Restrict to runtime-admissible envelope/account state; include all action endpoints needed by supported edges.
runtime_action_ids={'L075','L025','S075','S025','FLAT'}
eligA=exA & meta.account_stratum.isin(['FLAT_BASE','FLAT_DRAWDOWN']).to_numpy() & meta.action_id.isin(runtime_action_ids).to_numpy() & (np.abs(obs[:,0])<=1e-7) & (np.abs(obs[:,5])<=1e-7) & (np.abs(obs[:,4]-1.0)<=1e-6)
p_ex=pred_binary(stress_comp,XA[eligA]);y_ex=y_stress[eligA];q_ex=pred_quantiles(margin_mods,XA[eligA],margin_scale);ym_ex=y_margin[eligA]
calrows=[]
calrows.append({'scope':'ACTION_LEVEL_H64_RUNTIME_SUPPORT','metric':'event_ECE_equal_mass','value':ece(y_ex,p_ex),'target':'margin_stress_20pp_by_H64','n_rows':int(eligA.sum()),'status_basis':'DEVELOPMENT_ONLY'})
calrows.append({'scope':'ACTION_LEVEL_H64_RUNTIME_SUPPORT','metric':'event_Brier','value':float(brier_score_loss(y_ex,p_ex)),'target':'margin_stress_20pp_by_H64','n_rows':int(eligA.sum()),'status_basis':'DEVELOPMENT_ONLY'})
calrows.append({'scope':'ACTION_LEVEL_H64_RUNTIME_SUPPORT','metric':'event_log_loss','value':float(log_loss(y_ex,p_ex,labels=[0,1])),'target':'margin_stress_20pp_by_H64','n_rows':int(eligA.sum()),'status_basis':'DEVELOPMENT_ONLY'})
covs=[float(np.mean(ym_ex<=q_ex[:,j])) for j in range(3)];qmae=float(np.mean(np.abs(np.asarray(covs)-TAUS)))
for tau,cv in zip(TAUS,covs):calrows.append({'scope':'ACTION_LEVEL_H64_RUNTIME_SUPPORT','metric':f'quantile_coverage_q{int(tau*100):02d}','value':cv,'target':'max_margin_utilization_excess_H64','n_rows':int(eligA.sum()),'status_basis':'DEVELOPMENT_ONLY'})
calrows.append({'scope':'ACTION_LEVEL_H64_RUNTIME_SUPPORT','metric':'mean_abs_quantile_coverage_error','value':qmae,'target':'max_margin_utilization_excess_H64','n_rows':int(eligA.sum()),'status_basis':'DEVELOPMENT_ONLY'})
calrows.append({'scope':'ACTION_LEVEL_H64_RUNTIME_SUPPORT','metric':'interval80_coverage','value':float(np.mean((ym_ex>=q_ex[:,0])&(ym_ex<=q_ex[:,2]))),'target':'max_margin_utilization_excess_H64','n_rows':int(eligA.sum()),'status_basis':'DEVELOPMENT_ONLY'})

# Pair relief results and calibration on four concrete supported edges
support_pair_ids={e['pair_id'] for e in SUPPORTED_EDGES}
pp=pred_binary(disc_comp,XP[exP]);pr=pred_binary(dir_comp,XP[exP]);rho=pp*pr
pmex=pm.loc[exP].reset_index(drop=True);saex=sa[exP];sbex=sb[exP];real_rel=((saex==1)&(sbex==0)).astype(int)
pairrows=[]
for e in SUPPORTED_EDGES:
 ix=pmex.pair_id.eq(e['pair_id']).to_numpy();yy=real_rel[ix];rr=rho[ix]
 # action-level model monotonic diagnostic on same pair rows
 srcrows=pm.loc[exP,'row_A'].to_numpy(int)[ix];dstrows=pm.loc[exP,'row_B'].to_numpy(int)[ix]
 psrc=pred_binary(stress_comp,XA[srcrows]);pdst=pred_binary(stress_comp,XA[dstrows])
 qsrc=pred_quantiles(margin_mods,XA[srcrows],margin_scale)[:,1];qdst=pred_quantiles(margin_mods,XA[dstrows],margin_scale)[:,1]
 pairrows.append({'edge_id':e['edge_id'],'pair_id':e['pair_id'],'family':e['family'],'from':e['from'],'to':e['to'],'level':e['level'],'exam_rows':int(ix.sum()),'endpoint_groups':int(pmex.loc[ix,'market_i'].nunique()),'regimes_with_support':int(pmex.loc[ix,'regime'].nunique()),'realized_A_ONLY_rate':float(yy.mean()),'mean_predicted_rho':float(rr.mean()),'rho_ECE_equal_mass':ece(yy,rr),'rho_Brier':float(brier_score_loss(yy,rr)),'Pstress_A_ge_B_fraction':float(np.mean(psrc>=pdst)),'q50_max_margin_A_ge_B_fraction':float(np.mean(qsrc>=qdst)),'development_only':True})
 calrows.append({'scope':e['edge_id'],'metric':'pairwise_relief_rho_ECE_equal_mass','value':ece(yy,rr),'target':'rho(A->B|I_t)','n_rows':int(ix.sum()),'status_basis':'DEVELOPMENT_ONLY'})
pairdf=pd.DataFrame(pairrows);pairdf.to_csv(ROOT/'PAIRWISE_RELIEF_RESULTS.csv',index=False)
caldf=pd.DataFrame(calrows);caldf.to_csv(ROOT/'DEVELOPMENT_CALIBRATION_RESULTS.csv',index=False)

# Diagnostic calibration gates inherited in spirit from R1/R2, never formal qualification gates
event_ece=float(caldf.loc[caldf.metric.eq('event_ECE_equal_mass'),'value'].iloc[0]);pair_ece_max=float(pairdf.rho_ECE_equal_mass.max())
calibration_ok=bool(event_ece<=0.10 and qmae<=0.08 and pair_ece_max<=0.10)

# Shadow exhaustive/local counterfactual replay: no nominal policy distribution, no frequency interpretation.
# One row per eligible endpoint/snapshot/nominal supported A, reporting a fixed diagnostic lambda grid only.
LGRID=[0.,.25,.5,.75,1.]
replay=[]
for (mi,ss),g in pmex[pmex.pair_id.isin(support_pair_ids)].groupby(['market_i','snapshot_sha'],sort=True):
 account=[float(g.iloc[0][f'obs{j}']) for j in range(6)];o=ordered[int(mi)]
 for aid in ['L075','S075','S025']:
  edges=EDGE_BY_FROM.get(aid,[]);available=[]
  for e in edges:
   gg=g[g.pair_id.eq(e['pair_id'])]
   if gg.empty: continue
   r=float(pair_rho(pair_feature(o,account,e['pair_id']))[0]);available.append((e,r))
  if not available:continue
  base=pred_action(candidate,o,account,aid);row={'market_i':int(mi),'snapshot_sha':str(ss),'regime':str(g.regime.iloc[0]),'asset':str(g.asset.iloc[0]),'nominal_action':aid,'P_Z64_nominal':base['P_Z64_margin_stress'],'q50_max_margin_nominal':base['max_margin_quantiles']['q50'],'edge_rhos':json.dumps({e['edge_id']:r for e,r in available},sort_keys=True),'development_only':True,'not_deployment_frequency_estimate':True}
  prev=-1;mono_ok=True
  for lam in LGRID:
   if lam==0:level=0;chosen='PASS'
   else:
    tau=1-lam;qual=[(e,r) for e,r in available if r>=tau]
    if qual:
     maxlev=max(e['level'] for e,r in qual);choices=[(e,r) for e,r in qual if e['level']==maxlev];e,r=max(choices,key=lambda x:(x[1],x[0]['edge_id']));level=maxlev;chosen=e['edge_id']
    else:level=0;chosen='PASS'
   row[f'level_lambda_{str(lam).replace(".","p")}']=level;row[f'choice_lambda_{str(lam).replace(".","p")}']=chosen
   if level<prev:mono_ok=False
   prev=level
  row['lambda_path_monotone']=mono_ok;replay.append(row)
replaydf=pd.DataFrame(replay);replaydf.to_csv(ROOT/'SHADOW_COUNTERFACTUAL_REPLAY.csv',index=False)
monotone_replay_ok=bool(replaydf.lambda_path_monotone.all())

# Runtime schema
schema={
 '$schema':'https://json-schema.org/draft/2020-12/schema','title':'PredictiveRiskAdvisorCandidateV1','type':'object','additionalProperties':False,
 'required':['Ordered4H30','AccountState6','ActionIntentV1','static_external_envelope_metadata','lineage_support_version'],
 'properties':{
  'Ordered4H30':{'type':'array','minItems':30,'maxItems':30,'items':{'type':'number'},'x-lineage-sha256':EXPECTED['ordered4h30']},
  'AccountState6':{'type':'array','minItems':6,'maxItems':6,'items':{'type':'number'},'x-field-order':['signed_exposure_fraction_of_contract_cap','entry_price_log_ratio','remaining_holding_fraction','current_drawdown_fraction','risk_budget_remaining_fraction','margin_utilization_fraction'],'x-schema-sha256':EXPECTED['account_schema']},
  'ActionIntentV1':{'type':'object','required':['direction','requested_risk_multiplier'],'properties':{'direction':{'enum':[0,1,2]},'requested_risk_multiplier':{'type':'number','minimum':0,'maximum':1}}},
  'static_external_envelope_metadata':{'type':'object','required':['external_envelope_fraction'],'properties':{'external_envelope_fraction':{'type':'number','minimum':0,'maximum':1}}},
  'lineage_support_version':{'type':'object','required':['representation','representation_sha256','account_state_schema_sha256','action_intent_schema_sha256','support_version'],'properties':{'representation':{'const':'Ordered4H30'},'representation_sha256':{'const':EXPECTED['ordered4h30']},'account_state_schema_sha256':{'const':EXPECTED['account_schema']},'action_intent_schema_sha256':{'const':EXPECTED['action_schema']},'support_version':{'const':SUPPORT_VERSION}}}
 },
 'x-output-contract':{'task_gate_status':['SUPPORTED_SHADOW_PREDICTION','NO_ADVICE'],'supported_edge_set':'array','predicted_risk':'P(Z64(a)=1|current info)+conditional max-margin q10/q50/q90','predicted_relief':'rho(A->B|I_t)','suggested_intervention_level':'UNSET until formal control threshold is prospectively frozen','NO_ADVICE_reason':'nullable reason code','model_calibration_lineage':'required'},
 'x-shadow-only':True,'x-original-action-intent-mutation':False
}
writej(ROOT/'PREDICTIVE_RISK_ADVISOR_CANDIDATE_V1_SCHEMA.json',schema)
writej(ROOT/'PREDICTIVE_RISK_ADVISOR_CANDIDATE_V1_FROZEN_METADATA.json',meta_freeze)

# OOD/NO_ADVICE canaries
baseidx=int(np.flatnonzero(eligA & meta.action_id.eq('S075').to_numpy())[0]);base_o=ordered[int(meta.loc[baseidx,'market_i'])].tolist();base_a=obs[baseidx].astype(float).tolist()
base_in={'Ordered4H30':base_o,'AccountState6':base_a,'ActionIntentV1':{'schema':'ActionIntentV1','direction':0,'requested_risk_multiplier':.75},'static_external_envelope_metadata':{'external_envelope_fraction':1.0},'lineage_support_version':{'representation':'Ordered4H30','representation_sha256':EXPECTED['ordered4h30'],'account_state_schema_sha256':EXPECTED['account_schema'],'action_intent_schema_sha256':EXPECTED['action_schema'],'support_version':SUPPORT_VERSION}}
canaries=[]
def can(name,mut,expect):
 x=copy.deepcopy(base_in);mut(x);y=infer(candidate,x);canaries.append({'name':name,'expected_reason':expect,'observed_status':y['task_gate_status'],'observed_reason':y['NO_ADVICE_reason'],'pass':y['task_gate_status']=='NO_ADVICE' and y['NO_ADVICE_reason']==expect})
can('representation_version_mismatch',lambda x:x['lineage_support_version'].__setitem__('representation_sha256','0'*64),'REPRESENTATION_VERSION_MISMATCH')
can('unsupported_family_LONG025',lambda x:x.__setitem__('ActionIntentV1',{'schema':'ActionIntentV1','direction':2,'requested_risk_multiplier':.25}),'UNSUPPORTED_FAMILY__NO_FROZEN_MORE_CONSERVATIVE_EDGE')
def posmut(x):x['AccountState6'][0]=.1;x['AccountState6'][5]=.1
can('positioned_account',posmut,'POSITIONED_ACCOUNT')
def nonfin(x):x['Ordered4H30'][0]=float('nan')
can('nonfinite_representation',nonfin,'NONFINITE_OR_SHAPE_INVALID')
def lowenv(x):x['AccountState6'][4]=.25;x['static_external_envelope_metadata']['external_envelope_fraction']=.25
can('low_envelope',lowenv,'LOW_OR_UNSUPPORTED_EXTERNAL_ENVELOPE')
# supported sanity
san=infer(candidate,base_in);canaries.append({'name':'supported_short075_shadow_prediction','expected_reason':None,'observed_status':san['task_gate_status'],'observed_reason':san['NO_ADVICE_reason'],'pass':san['task_gate_status']=='SUPPORTED_SHADOW_PREDICTION' and set(san['supported_edge_set'])=={'SHORT_075_TO_025','SHORT_075_TO_FLAT'}})
writej(ROOT/'OOD_NO_ADVICE_CANARIES.json',{'schema':'CB16_ORDERED4H30_ADVISOR_OOD_NO_ADVICE_CANARIES_V1','all_pass':all(x['pass'] for x in canaries),'cases':canaries})

# Hard non-override canaries
orig=copy.deepcopy(base_in['ActionIntentV1']);orig_bytes=canonical(orig);_out=infer(candidate,base_in);after_bytes=canonical(base_in['ActionIntentV1'])
edge_nonincrease=[]
for e in SUPPORTED_EDGES:
 fr=ACTION_IDS[e['from']][1];to=ACTION_IDS[e['to']][1];edge_nonincrease.append({'edge_id':e['edge_id'],'from_risk':fr,'to_risk':to,'pass':to<=fr+1e-12})
hard={'schema':'CB16_ORDERED4H30_ADVISOR_HARD_NON_OVERRIDE_CANARIES_V1','risk_supervisor_sha256_expected':EXPECTED['risk_supervisor'],'risk_supervisor_sha256_observed':sha(SUPERVISOR),'risk_supervisor_byte_identical_unchanged':sha(SUPERVISOR)==EXPECTED['risk_supervisor'],'original_action_intent_bytes_unchanged':orig_bytes==after_bytes,'advisor_runtime_has_no_supervisor_write_path':True,'advisor_output_is_shadow_log_only':True,'edge_nonincrease':edge_nonincrease,'no_LONG_to_FLAT_edge':all(not(e['from'].startswith('L') and e['to']=='FLAT') for e in SUPPORTED_EDGES),'no_LONG_SHORT_flip':all(not({e['from'][0],e['to'][0]}=={'L','S'}) for e in SUPPORTED_EDGES),'all_pass':False}
hard['all_pass']=bool(hard['risk_supervisor_byte_identical_unchanged'] and hard['original_action_intent_bytes_unchanged'] and all(x['pass'] for x in edge_nonincrease) and hard['no_LONG_to_FLAT_edge'] and hard['no_LONG_SHORT_flip'])
writej(ROOT/'HARD_NON_OVERRIDE_CANARIES.json',hard)

# Reload exact canary: frozen joblib and portable object
reloaded=joblib.load(FROZEN)
test_idx=np.flatnonzero(eligA)[:64];Xtest=XA[test_idx]
p_mem=pred_binary(stress_comp,Xtest);p_fro=pred_binary(reloaded['models']['stress'],Xtest);q_mem=pred_quantiles(margin_mods,Xtest,margin_scale);q_fro=pred_quantiles(reloaded['models']['margin_quantiles'],Xtest,reloaded['models']['margin_scale']);p_port=portable_bin('action_stress_H64',Xtest);q_port=portable_q(Xtest)
# pair test
ptidx=np.flatnonzero(exP & pm.pair_id.isin(support_pair_ids).to_numpy())[:64];Xpt=XP[ptidx];r_mem=pair_rho(Xpt);r_fro=pred_binary(reloaded['models']['pair_discordance'],Xpt)*pred_binary(reloaded['models']['pair_direction'],Xpt);r_port=portable_rho(Xpt)
reload_can={'schema':'CB16_ORDERED4H30_ADVISOR_RELOAD_CANARY_V1','frozen_candidate_sha256':sha(FROZEN),'frozen_action_probability_max_abs_diff':float(np.max(np.abs(p_mem-p_fro))),'frozen_quantile_max_abs_diff':float(np.max(np.abs(q_mem-q_fro))),'frozen_pair_rho_max_abs_diff':float(np.max(np.abs(r_mem-r_fro))),'portable_action_probability_max_abs_diff':float(np.max(np.abs(p_mem-p_port))),'portable_quantile_max_abs_diff':float(np.max(np.abs(q_mem-q_port))),'portable_pair_rho_max_abs_diff':float(np.max(np.abs(r_mem-r_port))),'exact_reload_pass':bool(np.array_equal(p_mem,p_fro) and np.array_equal(q_mem,q_fro) and np.array_equal(r_mem,r_fro)),'portable_tolerance_pass':bool(np.max(np.abs(p_mem-p_port))<=1e-12 and np.max(np.abs(q_mem-q_port))<=1e-12 and np.max(np.abs(r_mem-r_port))<=1e-12)}
reload_can['all_pass']=reload_can['exact_reload_pass'] and reload_can['portable_tolerance_pass'];writej(ROOT/'RELOAD_CANARY.json',reload_can)

# Portable tar
PORT_TAR=ROOT/'PREDICTIVE_RISK_ADVISOR_CANDIDATE_V1_PORTABLE.tar.gz'
with tarfile.open(PORT_TAR,'w:gz') as t:
 for p in sorted(PORT.iterdir()):t.add(p,arcname=p.name)

# Result / handoff / report
all_canary=bool(json.load(open(ROOT/'OOD_NO_ADVICE_CANARIES.json'))['all_pass'] and hard['all_pass'] and reload_can['all_pass'] and monotone_replay_ok)
status='ORDERED4H30_ADVISOR_SHADOW_CANDIDATE_FROZEN' if (calibration_ok and all_canary) else ('ADVISOR_PREDICTOR_CALIBRATION_INADEQUATE' if all_canary else 'ORDERED4H30_ADVISOR_BINDING_FAIL')
result={'schema':'CB16_ORDERED4H30_PREDICTIVE_RISK_ADVISOR_R2_RESULT_V1','primary_status':status,'candidate':'PredictiveRiskAdvisorCandidateV1','authority_mode':'SHADOW_ADVISORY_ONLY','runtime_representation':{'name':'Ordered4H30','dimension':30,'sha256':EXPECTED['ordered4h30'],'Raw24_runtime_input':False},'predictor_family':'exact inherited LightGBM distributional readout family; no architecture tournament','outputs_bound':['P(Z64(a)=1|current info)','conditional max-margin q10/q50/q90','rho(A->B|I_t)','support validity'],'primary_event':'margin_stress_20pp_by_H64','supported_edges':SUPPORTED_EDGES,'flat_vs_long_admitted':False,'LONG_to_FLAT_supported':False,'development_calibration':{'event_ece':event_ece,'max_margin_qcov_mae':qmae,'pair_rho_ece_max':pair_ece_max,'diagnostic_gates':{'event_ece_max':.10,'quantile_coverage_mae_max':.08,'pair_rho_ece_max':.10},'pass':calibration_ok,'formal_certificate':False},'shadow_replay':{'rows':len(replaydf),'lambda_path_monotonicity_pass':monotone_replay_ok,'intervention_frequency_is_deployment_estimate':False},'canaries':{'all_pass':all_canary,'reload':reload_can['all_pass'],'OOD_NO_ADVICE':json.load(open(ROOT/'OOD_NO_ADVICE_CANARIES.json'))['all_pass'],'hard_non_override':hard['all_pass']},'claims_not_made':['QUALIFIED','ACTIVE','FORMALLY_SAFE','formal CRC certificate','deployment intervention frequency'],'fresh_prospective_cohort_opened':False,'central_brain_trained':False,'original_ActionIntent_modified':False,'frozen_RiskSupervisor_modified':False}
writej(ROOT/'ORDERED4H30_ADVISOR_R2_RESULT.json',result)

handoff={'schema':'CB16_ORDERED4H30_ADVISOR_R2_NEXT_PROSPECTIVE_HANDOFF_V1','source_status':status,'candidate_object':'PREDICTIVE_RISK_ADVISOR_CANDIDATE_V1_FROZEN.joblib','portable_object':'PREDICTIVE_RISK_ADVISOR_CANDIDATE_V1_PORTABLE.tar.gz','runtime_representation':'Ordered4H30 only','allowed_next_phase':'independent prospective qualification/statistical-control adjudication using frozen candidate and frozen task gates','required_before_any_active_use':['fresh prospective support under separately authorized budget','time-series/non-exchangeability theorem-family adjudication','prospective calibration/control threshold freeze','false-veto and missed-high-risk qualification gates','hard Supervisor remains final authority'],'carry_forward_supported_edges':[e['edge_id'] for e in SUPPORTED_EDGES],'do_not_carry_forward':['flat_vs_long','LONG->FLAT','long_vs_short','Raw24 as runtime input','development calibration as formal guarantee'],'Task_B_parallel_result_auto_import_forbidden':True,'formal_crc_certificate':False,'deployment_authorized':False}
writej(ROOT/'NEXT_PROSPECTIVE_HANDOFF.json',handoff)

manifest={'schema':'CB16_ORDERED4H30_PREDICTIVE_RISK_ADVISOR_R2_RUN_MANIFEST_V1','run_id':'CB16_ORDERED4H30_PREDICTIVE_RISK_ADVISOR_V1_R2_20260902','created_utc':datetime.now(timezone.utc).isoformat(),'authority_paths':['/CB16_ORDERED4H30_ACTION_CONDITIONAL_RISK_V1/R2/','/CB16_COARSE_TEMPORAL_RISK_INTERFACE_V1/R1/','/CB16_PREDICTIVE_RISK_ADVISOR_STATISTICAL_CONTROL_V1/R1/','/CB16_PREDICTIVE_RISK_ADVISOR_ARCHITECTURE_V1/R0/','/CB16_ACCOUNT_PHYSICS_STATE_V1/R0/','/CB16_TRAINING_CONTROL_PLANE_V1/R1/'],'input_hashes':{'Ordered4H30':sha(OPATH),'target_cache':sha(ZPATH),'pair_manifest':sha(PAIR_MANIFEST),'risk_supervisor_r1.py':sha(SUPERVISOR)},'data_boundary':{'fresh_market_access':False,'fresh_prospective_cohort':False,'new_future_simulation':False,'Raw24_runtime_input':False,'central_brain_training':False,'active_deployment':False,'prospective_qualification':False},'execution':{'model_family':'inherited LightGBM distributional readout','architecture_candidates':1,'H64_only_runtime_heads':True,'action_fit_rows':int(fitA.sum()),'action_calibration_rows':int(calA.sum()),'action_exam_rows':int(exA.sum()),'pair_fit_rows':int(fitP.sum()),'pair_calibration_rows':int(calP.sum()),'pair_exam_rows':int(exP.sum()),'market_endpoints':int(meta.market_i.nunique()),'supported_concrete_edges':len(SUPPORTED_EDGES),'shadow_replay_rows':len(replaydf)},'environment':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'lightgbm':lgb.__version__,'sklearn':__import__('sklearn').__version__,'joblib':joblib.__version__},'final_status':status}
writej(ROOT/'RUN_MANIFEST.json',manifest)

report=f'''# CB16 Ordered4H30 Predictive Risk Advisor Shadow Candidate R2\n\n## Final status\n\n**`{status}`**\n\n本轮只冻结一个 **shadow-only** predictor/runtime candidate。没有打开 fresh cohort，没有 prospective qualification，没有 active intervention，也没有训练 Central Brain。\n\n## Frozen binding\n\n- Runtime market input: **Ordered4H30 / 30 floats only** (`{EXPECTED['ordered4h30']}`).\n- Raw24: scientific ceiling/reference only; **not a runtime input**.\n- Predictor family: inherited LightGBM distributional readout family (`n_estimators=120`, `num_leaves=15`, fixed seed {SEED}); no architecture tournament.\n- Primary event: `Z64 = 1{{max_margin_utilization_excess_H64 >= 0.20}}`.\n- Continuous companion: conditional `max_margin_utilization_excess_H64` q10/q50/q90.\n- Pairwise relief: `rho(A->B|I_t)=P(Z64(A)=1, Z64(B)=0 | I_t)` from the inherited two-stage discordance/direction evaluator.\n\n## Supported concrete edges\n\n| Edge | Family | Level |\n|---|---|---:|\n| LONG .75 -> .25 | risk_multiplier_long | 1 |\n| SHORT .75 -> .25 | risk_multiplier_short | 1 |\n| SHORT .25 -> FLAT | flat_vs_short | 2 |\n| SHORT .75 -> FLAT | flat_vs_short | 2 |\n\n`flat_vs_long`, `LONG->FLAT`, and `long_vs_short` remain unsupported. Parallel Task B cannot alter this R2 object.\n\n## Development-only calibration diagnostics\n\n- Action event ECE: **{event_ece:.6f}** (diagnostic ceiling 0.10).\n- Max-margin q10/q50/q90 mean absolute coverage error: **{qmae:.6f}** (diagnostic ceiling 0.08).\n- Max supported-edge rho ECE: **{pair_ece_max:.6f}** (diagnostic ceiling 0.10).\n- Diagnostic calibration pass: **{calibration_ok}**.\n\nThese values are **DEVELOPMENT_ONLY**. They are not a CRC certificate and do not support `QUALIFIED`, `ACTIVE`, or `FORMALLY_SAFE`.\n\n## Runtime behavior\n\n`PredictiveRiskAdvisorCandidateV1` validates representation/account/action/support lineage before inference. Supported rows receive risk and relief predictions in the shadow log. The candidate deliberately leaves `suggested_intervention_level` unset because R1 did not freeze a formally calibrated operating threshold/lambda. The monotone R1 intervention graph is replayed only on a diagnostic lambda grid to verify ordering. Original `ActionIntentV1` is never mutated.\n\nNO_ADVICE is deterministic for representation mismatch, unsupported action family/risk anchor, positioned account, nonfinite input, low/unsupported envelope, or lineage/support mismatch.\n\n## Canaries\n\n- Reload exact: **{reload_can['exact_reload_pass']}**; portable reload tolerance: **{reload_can['portable_tolerance_pass']}**.\n- OOD/NO_ADVICE canaries: **{json.load(open(ROOT/'OOD_NO_ADVICE_CANARIES.json'))['all_pass']}**.\n- Hard non-override canaries: **{hard['all_pass']}**.\n- Frozen Risk Supervisor source SHA unchanged: **{hard['risk_supervisor_byte_identical_unchanged']}**.\n- Shadow lambda-path monotonicity: **{monotone_replay_ok}**.\n\n## Stopping condition\n\nThis bundle is a serializable/replayable **shadow predictor candidate**, not a safety qualification. The next lawful step is a separately authorized prospective statistical-control qualification using this frozen object; no active intervention is opened here.\n'''
(ROOT/'ORDERED4H30_ADVISOR_R2_REPORT.md').write_text(report)

# Copy runtime implementation source as reproducibility artifact
shutil.copy2(Path(__file__),ROOT/'build_and_runtime_candidate_r2.py')

# SHA payload excluding SHA itself, durable record, final bundle and sidecars
files=[p for p in ROOT.iterdir() if p.is_file() and p.name not in {'SHA256SUMS.txt','DURABLE_FREEZE_RECORD.json'}]
with open(ROOT/'SHA256SUMS.txt','w') as f:
 for p in sorted(files,key=lambda x:x.name): f.write(f'{sha(p)}  {p.name}\n')
# top bundle (durable record written after Library roundtrip)
BUNDLE=Path('/mnt/data/CB16_ORDERED4H30_PREDICTIVE_RISK_ADVISOR_V1_R2.tar.gz')
with tarfile.open(BUNDLE,'w:gz') as t:
 for p in sorted(ROOT.iterdir(),key=lambda x:x.name):
  if p.name=='DURABLE_FREEZE_RECORD.json':continue
  t.add(p,arcname=p.name)
Path(str(BUNDLE)+'.sha256').write_text(f'{sha(BUNDLE)}  {BUNDLE.name}\n')
print(json.dumps({'status':status,'root':str(ROOT),'bundle':str(BUNDLE),'bundle_sha256':sha(BUNDLE),'calibration_ok':calibration_ok,'event_ece':event_ece,'qmae':qmae,'pair_ece_max':pair_ece_max,'canaries':all_canary,'replay_rows':len(replaydf)},indent=2))
