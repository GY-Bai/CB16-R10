from __future__ import annotations

import json
from pathlib import Path

import pytest

from cb16_local_opt.r10_ci_reporting import append_scientific_summary, expected_pass_status, scientific_summary, status_is_pass
from cb16_local_opt.r10_host_bindings import host_binding


def test_host_binding_process_env_wins(monkeypatch):
    monkeypatch.setenv('CB16_R10_DATA_ROOT','/synthetic/process/data')
    assert host_binding('CB16_R10_DATA_ROOT','/default') == '/synthetic/process/data'


def test_host_binding_worker_file(monkeypatch,tmp_path: Path):
    monkeypatch.delenv('CB16_R10_R102_ROOT',raising=False)
    monkeypatch.setenv('CB16_CI_WORKER_ROOT',str(tmp_path))
    (tmp_path/'provision.env').write_text('CB16_R10_R102_ROOT=/synthetic/ssd/R10_2\n')
    assert host_binding('CB16_R10_R102_ROOT','/default') == '/synthetic/ssd/R10_2'


def test_host_binding_rejects_unlisted_key():
    with pytest.raises(ValueError,match='R10_HOST_BINDING_NOT_ALLOWED'):
        host_binding('CB16_SECRET_TOKEN','x')


def _result(status: str):
    return {
        'profile_name':'R10_2_5GEN_QUALIFICATION',
        'final_status':status,
        'mechanistic_pipeline_pass':status.endswith('_PASS'),
        'attempts_requested':5,'attempts_completed':5,'promotions':2,'rejections':3,
        'final_champion_semantic_sha256':'a'*64,
        'final_holdout_2025_09_accessed':False,
        'profitability_claimed':False,'market_alpha_claimed':False,
        'teacher_evidence_summary':{'train':{'admitted_dependence_groups':40},'validation':{'admitted_dependence_groups':9}},
        'scientific_controls_status':'TRUE_MARKET_NOT_BETTER_THAN_SHUFFLE',
        'controls':{'F3_minus_F2':-0.01},
        'integrity':{'frozen_authority_unchanged':True,'real_evidence_gradients_connected':True},
        'return_bundle':{'path':'/SECRET/HOST/PATH/bundle.tar.gz','sha256':'b'*64,'size':1234},
    }


def test_exact_status_is_ci_authority():
    good=_result(expected_pass_status('R10_2'))
    bad=_result('R10_2_REAL_HISTORICAL_LEARNING_PIPELINE_NOT_READY')
    assert status_is_pass(good,'R10_2') is True
    assert status_is_pass(bad,'R10_2') is False
    assert expected_pass_status('R10_3') == 'R10_3_10ASSET_20GEN_EXPANSION_PASS'
    assert expected_pass_status('R10_4') == 'R10_4_LONG_100GEN_RESEARCH_PASS'


def test_scientific_summary_is_sanitized(monkeypatch,tmp_path: Path):
    monkeypatch.setenv('CI_OUT',str(tmp_path))
    report=tmp_path/'REPORT.md'; report.write_text('# report\n')
    summary=append_scientific_summary(_result(expected_pass_status('R10_2')),'R10_2')
    raw=json.dumps(summary,sort_keys=True)
    report_raw=report.read_text()
    assert '/SECRET/HOST/PATH' not in raw
    assert '/SECRET/HOST/PATH' not in report_raw
    assert summary['return_bundle'] == {'sha256':'b'*64,'size':1234}
    assert summary['status_driving_pass'] is True
    assert summary['teacher_train_admitted_dependence_groups'] == 40
    assert summary['final_holdout_2025_09_accessed'] is False
