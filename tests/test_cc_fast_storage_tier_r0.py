from pathlib import Path
import pytest
from cb16_local_opt.cc_fast_storage_tier_r0 import *
def test_archive_verifies_before_reclaim(tmp_path):
    ssd=tmp_path/"ssd"; hdd=tmp_path/"hdd"; ssd.mkdir(); f=ssd/"x"; f.write_bytes(b"abc"*100); m=StorageTierManager(ssd_root=str(ssd),hdd_root=str(hdd),config=TierConfig(1024,1024,4096)); r=m.archive(f,reclaim_hot=True); assert r.checksum_verified and r.source_reclaimed; assert not f.exists(); assert Path(r.archive_path).read_bytes()==b"abc"*100
def test_capacity_fail_keeps_hot_copy(tmp_path):
    ssd=tmp_path/"ssd"; hdd=tmp_path/"hdd"; ssd.mkdir(); f=ssd/"x"; f.write_bytes(b"abc"); m=StorageTierManager(ssd_root=str(ssd),hdd_root=str(hdd),config=TierConfig(1,10,2))
    with pytest.raises(RuntimeError): m.archive(f)
    assert f.exists()
