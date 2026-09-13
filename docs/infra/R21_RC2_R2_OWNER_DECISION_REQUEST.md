# R21 RC2 R2 — SSD FAST_HOT owner decision request

Status: **PENDING_OWNER_APPROVAL**  
Task: R2  
Inventory: `authority/infra/R21_RC2_R2_SSD_CAPACITY_INVENTORY_V1.json`  
Proposal: `authority/infra/R21_RC2_R2_FAST_HOT_PROVISION_PROPOSAL_V1.json`

R2 is read-only. No cleanup, deletion, migration, mount creation, partition
change or Docker reconfiguration has been performed. R3 remains blocked until
the owner selects and approves one concrete option.

## Current SSD state

- SSD `/dev/sdb` (`CT500MX500SSD1`, non-rotational) contains the root LVM.
- Root LV `/dev/mapper/ubuntu--vg-ubuntu--lv` on `/`:
  - total `453,853,200,384` bytes
  - used `400,243,847,168` bytes (93%)
  - free `34,029,350,912` bytes (31.69 GiB)
- Largest root consumers: `/home` 330.6 GB, `/usr` 18.3 GB, `/var` 8.3 GB.
- HDD `/dev/sda1` on `/data`: 647.30 GiB free; intended cold/migration target.
- Proposed FAST_HOT abstraction: host `/srv/cb16_fast_hot` to container
  `/cb16/fast_hot`, owned by `1001:1001`, never allowed to fall back to HDD.

## Options

| Option | Action | Frees | FAST_HOT quota | Reserve left | Risk |
|---|---|---|---|---|---|
| **A** | No cleanup; use existing free space | 0 | 24.0 GB (22.35 GiB) | 10.0 GB (9.34 GiB) | HIGH |
| **B** | Safe reclaim: uv/pip/conda caches, journal to 512 MiB, crash/apt/disabled snaps | 35.68 GB (33.23 GiB) | 48.0 GB (44.70 GiB) | 21.71 GB (20.21 GiB) | MEDIUM |
| **C** | Migrate selected cold media/audio to `/data` with checksum verification | 92.08 GB (85.76 GiB) | 96.0 GB (89.41 GiB) | 30.11 GB (28.04 GiB) | MEDIUM-HIGH |
| **D** | B + C combined | 127.76 GB (118.98 GiB) | 120.0 GB (111.76 GiB) | 41.79 GB (38.92 GiB) | MEDIUM-HIGH |
| **E** | Add a new SSD/NVMe for FAST_HOT | no SSD cleanup | depends on device (e.g. 500 GB) | root unchanged | LOW-DATA / HIGH-SCHEDULE |

### Option B candidate detail

- `/home/bgy/miniforge3/pkgs` — 28.34 GB
- `/home/bgy/.cache/uv` — 1.66 GB
- `/home/bgy/.cache/pip` — 0.10 GB
- `/var/log/journal` — keep 512 MiB, reclaim 3.92 GB
- `/var/crash` — 0.78 GB (review first)
- `/var/cache/apt` — 0.25 GB
- disabled snap revisions — 0.63 GB

Not included by default: HuggingFace cache (40.88 GB), personal photos/audio,
`Codes_and_Datas`, `m3-infra`, `stage1-nvidia-ref`, Python environments, and
the 32 GiB `/swapfile`.

### Option C migration detail (default set)

| Path | Size |
|---|---|
| `/home/bgy/audiotransfer` | 49,327,542,272 |
| `/home/bgy/photo_timeline` | 20,411,514,880 |
| `/home/bgy/recovered_photos` | 15,399,612,416 |
| `/home/bgy/photo_packs` | 3,504,091,136 |
| `/home/bgy/photo_packs.tar.zst` | 3,437,293,568 |

Migration rule: copy to `/data/cb16_hdd/home_archive`, verify SHA256, then
remove the SSD source. No source deletion before verification.

## Recommendation

- Default: **Option B** — lowest-risk meaningful headroom.
- If R4/R5 later show 44.7 GiB FAST_HOT is insufficient, open **Option D**.
- Option A is not recommended because the root stays near 93% full.
- Option E is the clean long-term answer but not immediate.
- Moving or shrinking `/swapfile` is explicitly **not recommended** and not
  part of any option.

## Owner decision needed before R3

Please approve one of:

```text
OPTION: <A|B|C|D|E>
FAST_HOT_QUOTA_BYTES: <number>
RESERVE_FLOOR_BYTES: <number>
SELECTED_CANDIDATE_IDS: <list, only if B/C/D>
MIGRATION_DESTINATION: /data/cb16_hdd/home_archive   (if C/D)
MIGRATION_INTEGRITY: COPY_THEN_SHA256_VERIFY_THEN_REMOVE_SOURCE   (if C/D)
AUTHORIZE_R3_HOST_CHANGE_PLAN: yes|no
```

An approval record should be produced as
`CB16_R21_RC2_R2_OWNER_APPROVAL_V1` bound to the selected bytes and paths.

Until that record exists, R3 must not execute any host/container change.

## Related R1 finding

R1 recorded `R1-DIV-001`: the live runner env points
`CB16_PROVISION_ENV` to `/run/secrets/cb16-provision.env`, which does not
exist; the actual mounted file is `/run/secrets/provision.env`. This is
`CONTRACT_MISMATCH` and remains a separate Sol/owner decision. It does not
change the SSD options above.
