from __future__ import annotations

import os
import pwd
from dataclasses import dataclass
from pathlib import Path


CANONICAL_R104_ROOT = Path('/data/cb16_hdd/cb16_runtime/R10_4')
CANONICAL_R104_WRITER = 'cb16-ci'


@dataclass(frozen=True)
class R104WriterContract:
    run_root: Path
    effective_user: str
    canonical: bool
    authorized: bool


def _effective_username() -> str:
    return pwd.getpwuid(os.geteuid()).pw_name


def inspect_r104_writer_contract(
    run_root: str | Path,
    *,
    effective_user: str | None = None,
) -> R104WriterContract:
    root = Path(run_root).resolve()
    user = effective_user or _effective_username()
    canonical = root == CANONICAL_R104_ROOT.resolve()
    authorized = (not canonical) or user == CANONICAL_R104_WRITER
    return R104WriterContract(
        run_root=root,
        effective_user=user,
        canonical=canonical,
        authorized=authorized,
    )


def enforce_r104_writer_contract(
    run_root: str | Path,
    *,
    effective_user: str | None = None,
) -> R104WriterContract:
    contract = inspect_r104_writer_contract(
        run_root,
        effective_user=effective_user,
    )
    if not contract.authorized:
        raise RuntimeError(
            'R10_4_CANONICAL_WRITER_REQUIRED:'
            f'{CANONICAL_R104_WRITER}:'
            f'actual={contract.effective_user}:'
            f'run_root={contract.run_root}:'
            'use_an_explicit_noncanonical_--run-root_for_manual_debug'
        )
    return contract
