#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROVISION = ROOT / "provision" / "scripts"
if str(PROVISION) not in sys.path:
    sys.path.insert(0, str(PROVISION))
if str(ROOT / "ci") not in sys.path:
    sys.path.insert(0, str(ROOT / "ci"))

import provision_python as pp
import resolve_verified_r104_python as vr


def _run(cmd: list[str], env: dict[str, str]) -> tuple[int, float, str]:
    t0=time.monotonic()
    p=subprocess.run(cmd,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,check=False)
    return p.returncode,time.monotonic()-t0,p.stdout[-4000:]


def _version_map(report: dict) -> dict[str,str|None]:
    return {row["name"].lower():row["version"] for row in report["packages"]}


def main() -> int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",required=True,type=Path)
    ap.add_argument("--out",required=True,type=Path)
    args=ap.parse_args()

    if not shutil.which("uv"):
        raise SystemExit("UV_NOT_INSTALLED")

    manifest=json.loads((ROOT/"provision/environments/r104.json").read_text())
    reqs=[ROOT/p for p in manifest["requirements"]]
    env=pp._install_env(pp.load_provision_env())
    route=pp._validate_host_env_policy(manifest["python"],reqs,env)
    env["UV_NO_CACHE"]="1"

    args.root.mkdir(parents=True,exist_ok=True)
    venv=args.root/"venv"
    if venv.exists(): shutil.rmtree(venv)

    rc_venv,sec_venv,_=_run(["uv","venv",str(venv)],env)
    py=venv/"bin/python"
    cmd=pp._uv_install_command(py,reqs)
    pp._assert_no_channel_args(cmd)
    rc_install,sec_install,install_tail=_run(cmd,env)

    rc_check=99; sec_check=0.0; rc_cuda=99; sec_cuda=0.0
    new_versions=None; canonical_versions=None; exact_match=False
    if rc_install==0:
        rc_check,sec_check,_=_run(["uv","pip","check","--python",str(py)],env)
        if rc_check==0:
            new_versions=pp._direct_version_report(py,reqs)
            canonical_versions=vr.direct_version_report(vr.EXPECTED_VENV/"bin/python")
            exact_match=_version_map(new_versions)==_version_map(canonical_versions)
            cuda_code=(
                "import torch,numpy,pandas,lightgbm,pytest;"
                "assert torch.__version__=='2.8.0+cu126',torch.__version__;"
                "assert torch.cuda.is_available(),'CUDA_UNAVAILABLE';"
                "print(torch.__version__,torch.version.cuda,torch.cuda.get_device_name(0))"
            )
            rc_cuda,sec_cuda,_=_run([str(py),"-c",cuda_code],env)

    status=(rc_venv==0 and rc_install==0 and rc_check==0 and rc_cuda==0 and exact_match)
    result={
        "schema":"CB16_UV_HOST_ENV_DOWNLOAD_QUALIFICATION_V1",
        "status":"PASS" if status else "FAIL",
        "uv_version":subprocess.run(["uv","--version"],text=True,capture_output=True).stdout.strip(),
        "host_route_summary":route,
        "uv_no_cache":True,
        "network_download_exercised":rc_install==0,
        "repository_selected_channel_arguments":False,
        "install_command_has_url_or_channel_flag":any(x in {"--index-url","--extra-index-url","--find-links","-i"} or "://" in x for x in cmd),
        "venv_rc":rc_venv,
        "install_rc":rc_install,
        "dependency_check_rc":rc_check,
        "cuda_import_canary_rc":rc_cuda,
        "venv_seconds":round(sec_venv,3),
        "install_seconds":round(sec_install,3),
        "dependency_check_seconds":round(sec_check,3),
        "cuda_import_seconds":round(sec_cuda,3),
        "direct_package_versions_new":new_versions,
        "direct_package_versions_canonical":canonical_versions,
        "direct_package_versions_exact_match":exact_match,
        "package_hash_validation_used":False,
        "install_failure_tail_redacted_to_non_url_lines":[
            line for line in install_tail.splitlines()[-20:] if "://" not in line
        ] if rc_install else [],
        "scientific_semantics_changed":False,
        "final_holdout_2025_09_accessed":False,
    }
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps({k:result[k] for k in (
        "status","uv_version","network_download_exercised","repository_selected_channel_arguments",
        "install_command_has_url_or_channel_flag","install_seconds","direct_package_versions_exact_match",
        "package_hash_validation_used")},indent=2,sort_keys=True))
    return 0 if status else 1


if __name__=="__main__":
    raise SystemExit(main())
