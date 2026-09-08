# CB16 R11 Stage-4 S4F — Canonical Persistent State Roots & Storage Ownership R0

## Authority boundary

S4F defines storage ownership and persistent root identity only. It does **not** make a scientific decision, open the final holdout, download market data, mutate frozen market/raw authority, grant Permission, acquire a runtime lease, mint/admit Evidence, create a Challenger, promote a Champion, advance a generation, or reinterpret replay as Evidence.

Highest rule: **SEMANTIC CONTRACTS ARE AUTHORITY; LEGACY PYTHON IMPLEMENTATION IS NOT AUTHORITY.**

Gatework base: `0f18e08ec7250b9b4e45c62803c25be966834390`  
Semantic Freeze Git blob: `3c401a0a350984381912f7860181e3e96eb8d7cf`  
Scientific status remains: `DISTRIBUTIONAL_MARKET_INFORMATION_NOT_QUALIFIED__TRUE_WORSE_THAN_SHUFFLE`.

## Why a new root contract is required

Accepted R11 engineering stores already contain useful durability mechanisms. `EvidenceStoreR11` separates metadata from payload roots, rejects overlap with configured read-only source roots, uses sealed payload segments, and has active-tail recovery. `CheckpointStoreR11` uses content-addressed semantic checkpoint objects and refuses to silently trust an unindexed object. `EventJournalR11` uses transactional metadata and generation seals.

Those are engineering inputs, not a canonical-authority grant. S4F therefore does not repurpose a legacy writable directory merely because current Python writes there. A non-empty unmarked root fails closed. Final integration must explicitly adopt the Stage-4 root provider and wire already-qualified domain stores beneath it.

## Canonical root classes

### SSD / control plane

The control root contains namespaces for:

- `authoritative_journal_metadata`
- `checkpoint_metadata`
- `runtime_lease_fencing_state`
- `hot_indexes`
- `recovery_metadata`
- `adoption_control_metadata`
- `object_seals`

The root receives a Stage-4 role marker only when empty. The marker binds canonical path, root role, Gatework base, Semantic Freeze blob, frozen-raw path/identity, scientific status, and deterministic root identity. A non-empty root without that marker is never auto-adopted.

S4F does not define the domain transactions inside those namespaces. Journal, checkpoint, lease, recovery and adoption semantics remain owned by their existing contracts or later Stage-4 integration providers.

### HDD / immutable data plane

The data root has three content-addressed classes:

- `evidence_payloads`
- `trace_replay_payloads`
- `cold_content_artifacts`

Storage identity is `SHA256(exact payload bytes)`. Canonical location is `<class>/sha256/<first-two-hex>/<sha256>.blob`. A matching control-plane storage seal binds class, digest, byte count and data-root identity.

The S4F seal is **storage integrity only**. It explicitly records `scientific_evidence_admission=false` and `generation_advancement=false`. Putting bytes in `evidence_payloads` is therefore not Evidence mint/admission. Replay bytes never become new Evidence.

Publication deliberately has a durability boundary between payload publication and seal publication. A crash in that gap creates an orphan. Startup detects it and fails closed; recovery does not synthesize the missing seal or silently adopt the orphan.

### Frozen market/raw authority

The frozen raw root is external authority and must preexist. S4F never creates, repairs, downloads into, truncates, moves, seals or mutates it. It must be disjoint from canonical writable roots and cannot be a symlink.

The implementation performs no write probe against frozen data. It checks the frozen root directory has no write permission bits. Final deployment should prefer an OS read-only mount, which is stronger than directory mode bits alone.

## Fail-closed startup invariants

Startup verifies:

1. control, data and frozen roots are pairwise disjoint;
2. frozen root preexists, is a directory, is not a symlink and is read-only at the root boundary;
3. control/data role markers exactly match pinned identities;
4. expected namespace directories have the correct type;
5. symlinks anywhere under canonical control/data roots are rejected;
6. `.stage4-partial` files are rejected;
7. every immutable payload has exactly one matching storage seal;
8. every storage seal has the corresponding payload;
9. payload SHA-256, size, class and canonical location match the seal;
10. unexpected/non-canonical payload or seal names fail closed.

No startup path repairs or upgrades an ambiguous object into authority.

## Machine-readable ownership contract

`authority/rearchitecture_r11/CB16_R11_STAGE4_STATE_ROOT_CONTRACT_V1.json` defines every required object class with owner, read/write mode, mutability, creation authority, sealing authority, recovery behavior, orphan behavior, torn-write policy, content identity and startup verification.

Critical separation rules:

- `runtime_lease_fencing_state` is only storage home for the later lease provider; S4F does not implement fencing authority.
- `adoption_control_metadata` is only storage home for the adoption provider; S4F does not implement authority adoption.
- `hot_indexes` are reconstructible caches and cannot manufacture missing authority.
- checkpoint tensor semantic identity remains owned by the checkpoint contract; S4F does not redefine it as raw file hash identity.
- journal/event scientific meaning remains owned by journal/evidence contracts.

## Hostile cases covered

Tests cover clean initialization and deterministic restart reconstruction, swapped/wrong root roles, refusal to auto-adopt a non-empty legacy root, frozen root becoming writable, payload-without-seal orphan, seal-without-payload, post-seal content tamper, relative path escape, symlink traversal, attempted control-root placement inside frozen raw authority, partial/torn file and root-marker corruption.

Ambiguous/incomplete states fail closed and remain in place for diagnosis; S4F does not silently repair them.

## Integration notes

Final Stage-4 integration should make the S4F provider an early startup gate before opening production writers. S4B can consume the verified root receipt before `RUNNING`. S4D can own `runtime_lease_fencing_state`. S4C can own `adoption_control_metadata`. Existing R11 journal/checkpoint/evidence stores can be routed into corresponding namespaces only after authoritative-write call sites are protected by final lifecycle, fencing and legacy-retirement guards.

S4F by itself does not qualify canonical cutover and must not be interpreted as `R11_CANONICAL_AUTHORITY_CUTOVER_QUALIFIED`.

## Qualification commands

```text
python -m pytest -q tests/test_stage4_gatework_r11.py tests/test_stage4_state_roots_r11.py
python -m cb16_local_opt.rearchitecture_authority_r11 static --repo-root .
python scripts/check_r11_stage4_task_receipt.py --receipt authority/rearchitecture_r11/stage4_receipts/S4F_RECEIPT_V1.json --repo-root .
```
