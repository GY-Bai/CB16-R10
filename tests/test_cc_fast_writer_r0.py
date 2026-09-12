from pathlib import Path
from cb16_local_opt.cc_fast_fact_queue_r0 import encode_fact
from cb16_local_opt.cc_fast_writer_r0 import *
def test_contiguous_writer_preserves_order_checksum_and_durability(tmp_path):
    facts=[encode_fact({"i":i},semantic_id=f"id-{i}",terminal_or_failure=(i==2)) for i in range(3)]; w=ContiguousChunkWriter(tmp_path); r=w.write_chunk(facts,chunk_id="c0"); assert r.fact_ids==("id-0","id-1","id-2"); assert r.durable; assert w.verify(r); lines=Path(r.path).read_text().splitlines(); assert '"semantic_id":"id-0"' in lines[0]; assert '"semantic_id":"id-2"' in lines[2]
