from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np, torch

class SharedKronosSensoryBus:
    def __init__(self, asset_root: str|Path, reducer_path: str|Path):
        root=Path(asset_root)
        source=next((root/'source').glob('Kronos-*')) if (root/'source').exists() else next(root.glob('**/source/Kronos-*'))
        hf=root/'hf' if (root/'hf').exists() else next(root.glob('**/hf'))
        sys.path.insert(0,str(source))
        from model.kronos import KronosTokenizer,Kronos
        self.tok=KronosTokenizer.from_pretrained(str(hf/'tokenizer_base'),local_files_only=True).eval()
        self.model=Kronos.from_pretrained(str(hf/'kronos_small'),local_files_only=True).eval()
        self.r=np.load(reducer_path)
    @staticmethod
    def stamps_from_datetime(index):
        import pandas as pd
        t=pd.DatetimeIndex(index)
        return np.stack([t.minute,t.hour,t.weekday,t.day,t.month],axis=1).astype(np.float32)
    @staticmethod
    def _prep(W):
        W=np.asarray(W,dtype=np.float32)
        if W.ndim==2: W=W[None]
        o=W[:,:,:4];v=W[:,:,4:5];amt=v*o.mean(2,keepdims=True)
        x=np.concatenate([o,v,amt],2).astype(np.float32);m=x.mean(1,keepdims=True);s=x.std(1,keepdims=True)
        return np.clip((x-m)/(s+1e-5),-5,5)
    def _d45(self,W,stamp):
        X=self._prep(W);T=np.asarray(stamp,dtype=np.float32)
        if T.ndim==2:T=T[None]
        with torch.inference_mode():
            x=torch.from_numpy(X);st=torch.from_numpy(T);z=self.tok.embed(x)
            for L in self.tok.encoder:z=L(z)
            z=self.tok.quant_embed(z);_,_,ids=self.tok.tokenizer(z,half=True,collect_metrics=False)
            h=self.model.embedding([ids[0],ids[1]])+self.model.time_emb(st);h=self.model.token_drop(h);h4=None
            for li,L in enumerate(self.model.transformer,1):
                h=L(h)
                if li==4:h4=h[:,-1].clone()
                if li==5:return (h[:,-1]-h4).cpu().numpy().astype(np.float32)
        raise RuntimeError('layer5 not reached')
    def _reduce(self,Z,scale):
        Z=np.asarray(Z,dtype=np.float64);S=(Z-self.r[f'{scale}_sc_mean'])/self.r[f'{scale}_sc_scale'];return ((S-self.r[f'{scale}_pca_mean'])@self.r[f'{scale}_pca_components'].T).astype(np.float32)
    def encode_micro(self,window_60x5,stamp_60x5):return self._reduce(self._d45(window_60x5,stamp_60x5),'micro')
    def encode_macro(self,window_32x5,stamp_32x5):return self._reduce(self._d45(window_32x5,stamp_32x5),'macro')
    def encode(self,micro_60x5,micro_stamp,macro_32x5,macro_stamp):
        mi=self.encode_micro(micro_60x5,micro_stamp);ma=self.encode_macro(macro_32x5,macro_stamp);return {'micro_operator24':mi,'macro_operator24':ma,'operator_bus48':np.concatenate([mi,ma],axis=1)}
