from __future__ import annotations

"""Durable leased work queue for the continuous CB16 data factory.

Exactly-once *identity* is provided by immutable JobSpec hashes. Execution is at-least-once
under crash recovery, so every stage command must publish outputs atomically/idempotently.
This is compatible with the existing CB16 immutable receipts and CAS semantics.
"""

import dataclasses, hashlib, json, sqlite3, time, uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


def canonical_hash(obj:Any)->str:
    if dataclasses.is_dataclass(obj):obj=asdict(obj)
    return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class FactoryJobSpecR8:
    job_id: str
    stage: str
    command: tuple[str,...]
    scientific_identity: Mapping[str,str]
    resource_class: str = 'CPU_IO'
    priority: int = 100
    max_attempts: int = 3
    retry_delay_seconds: float = 30.0
    timeout_seconds: float = 86400.0
    expected_outputs: tuple[str,...] = ()

    def validate(self):
        if not self.job_id or not self.stage or not self.command:raise ValueError('job id/stage/command required')
        if self.resource_class not in {'CPU_IO','CPU_HEAVY','GPU','MAINTENANCE','TRANSFER'}:raise ValueError('resource class')
        if self.max_attempts<=0 or self.timeout_seconds<=0 or self.retry_delay_seconds<0:raise ValueError('retry/timeout')
    @property
    def content_hash(self)->str:self.validate();return canonical_hash(self)


@dataclass(frozen=True)
class ClaimedJobR8:
    spec: FactoryJobSpecR8
    lease_token: str
    lease_owner: str
    lease_expires_at: float
    attempt: int


class FactoryQueueR8:
    def __init__(self,path:str|Path):
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
        self.db=sqlite3.connect(self.path,isolation_level=None,timeout=30)
        self.db.execute('PRAGMA journal_mode=WAL');self.db.execute('PRAGMA synchronous=FULL');self.db.execute('PRAGMA busy_timeout=30000')
        self.db.executescript('''
        CREATE TABLE IF NOT EXISTS jobs(
            job_id TEXT PRIMARY KEY,
            spec_hash TEXT NOT NULL,
            spec_json BLOB NOT NULL,
            stage TEXT NOT NULL,
            resource_class TEXT NOT NULL,
            priority INTEGER NOT NULL,
            state TEXT NOT NULL,
            attempts INTEGER NOT NULL DEFAULT 0,
            available_at REAL NOT NULL,
            lease_token TEXT,
            lease_owner TEXT,
            lease_expires_at REAL,
            last_error TEXT,
            result_json BLOB,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_jobs_claim ON jobs(state,available_at,priority,created_at);
        CREATE TABLE IF NOT EXISTS dependencies(
            job_id TEXT NOT NULL,
            depends_on TEXT NOT NULL,
            PRIMARY KEY(job_id,depends_on),
            FOREIGN KEY(job_id) REFERENCES jobs(job_id),
            FOREIGN KEY(depends_on) REFERENCES jobs(job_id)
        );
        CREATE TABLE IF NOT EXISTS events(
            seq INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id TEXT NOT NULL,
            event TEXT NOT NULL,
            payload_json BLOB,
            created_at REAL NOT NULL
        );
        ''')

    def close(self):self.db.close()

    def _event(self,job_id,event,payload=None):
        self.db.execute('INSERT INTO events(job_id,event,payload_json,created_at) VALUES(?,?,?,?)',(job_id,event,None if payload is None else json.dumps(payload,sort_keys=True),time.time()))

    def enqueue(self,spec:FactoryJobSpecR8,*,dependencies:Sequence[str]=())->bool:
        spec.validate();now=time.time();self.db.execute('BEGIN IMMEDIATE')
        try:
            old=self.db.execute('SELECT spec_hash FROM jobs WHERE job_id=?',(spec.job_id,)).fetchone()
            if old is not None:
                if old[0]!=spec.content_hash:raise RuntimeError('FACTORY_JOB_IDENTITY_CONFLICT:'+spec.job_id)
                self.db.execute('COMMIT');return False
            for dep in dependencies:
                if self.db.execute('SELECT 1 FROM jobs WHERE job_id=?',(dep,)).fetchone() is None:raise RuntimeError('FACTORY_DEPENDENCY_NOT_ENQUEUED:'+dep)
            self.db.execute('''INSERT INTO jobs(job_id,spec_hash,spec_json,stage,resource_class,priority,state,attempts,available_at,created_at,updated_at)
                               VALUES(?,?,?,?,?,?,'PENDING',0,?,?,?)''',(spec.job_id,spec.content_hash,json.dumps(asdict(spec),sort_keys=True),spec.stage,spec.resource_class,spec.priority,now,now,now))
            for dep in dependencies:self.db.execute('INSERT INTO dependencies VALUES(?,?)',(spec.job_id,dep))
            self._event(spec.job_id,'ENQUEUED',{'dependencies':list(dependencies)})
            self.db.execute('COMMIT');return True
        except Exception:self.db.execute('ROLLBACK');raise

    def recover_expired(self,*,now:float|None=None)->int:
        now=time.time() if now is None else now;self.db.execute('BEGIN IMMEDIATE')
        try:
            rows=self.db.execute("SELECT job_id,attempts,spec_json FROM jobs WHERE state='RUNNING' AND lease_expires_at<?",(now,)).fetchall();n=0
            for job_id,attempts,spec_json in rows:
                spec=FactoryJobSpecR8(**{**json.loads(spec_json),'command':tuple(json.loads(spec_json)['command']),'expected_outputs':tuple(json.loads(spec_json).get('expected_outputs',()))})
                state='FAILED' if attempts>=spec.max_attempts else 'PENDING';available=now if state=='FAILED' else now+spec.retry_delay_seconds
                self.db.execute('''UPDATE jobs SET state=?,available_at=?,lease_token=NULL,lease_owner=NULL,lease_expires_at=NULL,last_error='LEASE_EXPIRED',updated_at=? WHERE job_id=?''',(state,available,now,job_id));self._event(job_id,'LEASE_EXPIRED',{'new_state':state});n+=1
            self.db.execute('COMMIT');return n
        except Exception:self.db.execute('ROLLBACK');raise

    @staticmethod
    def _spec_from_json(text:str)->FactoryJobSpecR8:
        x=json.loads(text);x['command']=tuple(x['command']);x['expected_outputs']=tuple(x.get('expected_outputs',()));return FactoryJobSpecR8(**x)

    def claim(self,*,owner:str,lease_seconds:float=300.0,allowed_resource_classes:Sequence[str]|None=None)->ClaimedJobR8|None:
        if lease_seconds<=0:raise ValueError('lease seconds')
        self.recover_expired();now=time.time();classes=tuple(allowed_resource_classes or ('CPU_IO','CPU_HEAVY','GPU','MAINTENANCE','TRANSFER'))
        self.db.execute('BEGIN IMMEDIATE')
        try:
            q='''SELECT j.job_id,j.spec_json,j.attempts FROM jobs j
                 WHERE j.state='PENDING' AND j.available_at<=? AND j.resource_class IN (%s)
                 AND NOT EXISTS(SELECT 1 FROM dependencies d JOIN jobs p ON p.job_id=d.depends_on WHERE d.job_id=j.job_id AND p.state!='SUCCEEDED')
                 ORDER BY j.priority ASC,j.created_at ASC LIMIT 1''' % ','.join('?'*len(classes))
            row=self.db.execute(q,(now,*classes)).fetchone()
            if row is None:self.db.execute('COMMIT');return None
            job_id,spec_json,attempts=row;token=uuid.uuid4().hex;expiry=now+lease_seconds;attempt=int(attempts)+1
            changed=self.db.execute("UPDATE jobs SET state='RUNNING',attempts=?,lease_token=?,lease_owner=?,lease_expires_at=?,updated_at=? WHERE job_id=? AND state='PENDING'",(attempt,token,owner,expiry,now,job_id)).rowcount
            if changed!=1:raise RuntimeError('FACTORY_CLAIM_RACE')
            self._event(job_id,'CLAIMED',{'owner':owner,'attempt':attempt,'lease_expires_at':expiry});self.db.execute('COMMIT')
            return ClaimedJobR8(self._spec_from_json(spec_json),token,owner,expiry,attempt)
        except Exception:self.db.execute('ROLLBACK');raise

    def heartbeat(self,claim:ClaimedJobR8,*,lease_seconds:float=300.0)->float:
        now=time.time();expiry=now+lease_seconds;changed=self.db.execute("UPDATE jobs SET lease_expires_at=?,updated_at=? WHERE job_id=? AND state='RUNNING' AND lease_token=? AND lease_owner=?",(expiry,now,claim.spec.job_id,claim.lease_token,claim.lease_owner)).rowcount
        if changed!=1:
            raise RuntimeError('FACTORY_HEARTBEAT_LOST_LEASE')
        return expiry

    def succeed(self,claim:ClaimedJobR8,result:Mapping[str,Any]):
        now=time.time();self.db.execute('BEGIN IMMEDIATE')
        try:
            changed=self.db.execute("UPDATE jobs SET state='SUCCEEDED',result_json=?,lease_token=NULL,lease_owner=NULL,lease_expires_at=NULL,updated_at=? WHERE job_id=? AND state='RUNNING' AND lease_token=?",(json.dumps(dict(result),sort_keys=True),now,claim.spec.job_id,claim.lease_token)).rowcount
            if changed!=1:raise RuntimeError('FACTORY_SUCCESS_LOST_LEASE')
            self._event(claim.spec.job_id,'SUCCEEDED',dict(result));self.db.execute('COMMIT')
        except Exception:self.db.execute('ROLLBACK');raise

    def fail(self,claim:ClaimedJobR8,error:str,*,retryable:bool=True):
        now=time.time();state='PENDING' if retryable and claim.attempt<claim.spec.max_attempts else 'FAILED';available=now+claim.spec.retry_delay_seconds if state=='PENDING' else now
        self.db.execute('BEGIN IMMEDIATE')
        try:
            changed=self.db.execute("UPDATE jobs SET state=?,available_at=?,last_error=?,lease_token=NULL,lease_owner=NULL,lease_expires_at=NULL,updated_at=? WHERE job_id=? AND state='RUNNING' AND lease_token=?",(state,available,error[:10000],now,claim.spec.job_id,claim.lease_token)).rowcount
            if changed!=1:raise RuntimeError('FACTORY_FAIL_LOST_LEASE')
            self._event(claim.spec.job_id,'FAILED_ATTEMPT',{'retryable':retryable,'new_state':state,'error':error[:2000]});self.db.execute('COMMIT')
        except Exception:self.db.execute('ROLLBACK');raise

    def counts(self)->dict[str,int]:
        return {r[0]:int(r[1]) for r in self.db.execute('SELECT state,COUNT(*) FROM jobs GROUP BY state')}

    def get(self,job_id:str)->dict[str,Any]|None:
        row=self.db.execute('SELECT spec_hash,spec_json,state,attempts,available_at,lease_owner,lease_expires_at,last_error,result_json FROM jobs WHERE job_id=?',(job_id,)).fetchone()
        if row is None:return None
        return {'spec_hash':row[0],'spec':json.loads(row[1]),'state':row[2],'attempts':row[3],'available_at':row[4],'lease_owner':row[5],'lease_expires_at':row[6],'last_error':row[7],'result':None if row[8] is None else json.loads(row[8])}
