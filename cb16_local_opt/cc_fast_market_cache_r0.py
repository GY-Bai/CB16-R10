from __future__ import annotations

from dataclasses import dataclass
import hashlib
from multiprocessing import shared_memory
import numpy as np

CACHE_SCHEMA = "CB16_R11_CC_FAST_MARKET_CACHE_V1"
LAYOUT_VERSION = "CC_FAST_CONTIGUOUS_F64_V1"

@dataclass(frozen=True)
class MarketCacheKey:
    market_source_lineage: str
    visible_window_identity: str
    preprocessing_version: str
    frozen_organ_identity: str
    normalizer_identity: str
    output_dtype_layout_version: str = LAYOUT_VERSION
    def validate(self):
        for name,value in self.__dict__.items():
            if not isinstance(value,str) or not value: raise ValueError(f"{name.upper()}_EMPTY")
        return self
    @property
    def key_sha256(self):
        self.validate()
        return hashlib.sha256("\0".join((self.market_source_lineage,self.visible_window_identity,self.preprocessing_version,self.frozen_organ_identity,self.normalizer_identity,self.output_dtype_layout_version)).encode()).hexdigest()

@dataclass(frozen=True)
class SharedMarketDescriptor:
    schema_version:str
    cache_key_sha256:str
    shm_name:str
    shape:tuple[int,...]
    dtype:str
    nbytes:int
    content_sha256:str
    def validate(self):
        if self.schema_version!=CACHE_SCHEMA: raise ValueError("SHARED_MARKET_SCHEMA_MISMATCH")
        if not self.shm_name or self.nbytes<=0: raise ValueError("SHARED_MARKET_DESCRIPTOR_INVALID")
        return self

class SharedMarketOwner:
    def __init__(self,key:MarketCacheKey,array:np.ndarray):
        key.validate(); contiguous=np.ascontiguousarray(array)
        if contiguous.size==0: raise ValueError("MARKET_ARRAY_EMPTY")
        if contiguous.dtype.hasobject: raise ValueError("OBJECT_DTYPE_FORBIDDEN")
        self.key=key; self._shm=shared_memory.SharedMemory(create=True,size=contiguous.nbytes)
        self._array=np.ndarray(contiguous.shape,dtype=contiguous.dtype,buffer=self._shm.buf); self._array[...]=contiguous; self._array.setflags(write=False)
        self.descriptor=SharedMarketDescriptor(CACHE_SCHEMA,key.key_sha256,self._shm.name,tuple(contiguous.shape),contiguous.dtype.str,contiguous.nbytes,hashlib.sha256(contiguous.tobytes(order="C")).hexdigest())
    @property
    def array(self): return self._array
    def close(self,*,unlink=True):
        try: self._shm.close()
        finally:
            if unlink:
                try: self._shm.unlink()
                except FileNotFoundError: pass
    def __enter__(self): return self
    def __exit__(self,exc_type,exc,tb): self.close()

class SharedMarketView:
    def __init__(self,descriptor:SharedMarketDescriptor):
        descriptor.validate(); self.descriptor=descriptor; self._shm=shared_memory.SharedMemory(name=descriptor.shm_name,create=False)
        self.array=np.ndarray(descriptor.shape,dtype=np.dtype(descriptor.dtype),buffer=self._shm.buf); self.array.setflags(write=False)
        if self.array.nbytes!=descriptor.nbytes: self.close(); raise ValueError("SHARED_MARKET_SIZE_MISMATCH")
    def verify_content(self): return hashlib.sha256(self.array.tobytes(order="C")).hexdigest()==self.descriptor.content_sha256
    def close(self): self._shm.close()
    def __enter__(self): return self
    def __exit__(self,exc_type,exc,tb): self.close()
