"""评测脚本测试：指标计算、分组召回率、MRR，以及一次完整的端到端评测跑通。"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def run_eval_module():
    """把 eval/run_eval.py 当成模块加载（它不是包的一部分）。"""
    path = PROJECT_ROOT / "eval" / "run_eval.py"
    spec = importlib.util.spec_from_file_location("finrag_run_eval", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["finrag_run_eval"] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# 指标函数
# ---------------------------------------------------------------------------
def test_hit_rate_all_groups_missed(run_eval_module):
    assert run_eval_module.hit_rate([["A"], ["B"]], ["C"]) == 0.0


def test_hit_rate_single_group_any_member(run_eval_module):
    """同一答案分布在多份资料里时，命中任一即算该组命中。"""
    assert run_eval_module.hit_rate([["A", "B", "C"]], ["B"]) == 1.0


def test_hit_rate_partial_groups(run_eval_module):
    assert run_eval_module.hit_rate([["A"], ["B"]], ["A", "Z"]) == 0.5


def test_hit_rate_empty_expectation(run_eval_module):
    assert run_eval_module.hit_rate([], ["A"]) == 1.0


def test_recall_at_1(run_eval_module):
    assert run_eval_module.recall_at_1([["A"]], ["A", "B"]) == 1.0
    assert run_eval_module.recall_at_1([["A"]], ["B", "A"]) == 0.0
    assert run_eval_module.recall_at_1([["A"]], []) == 0.0


def test_reciprocal_rank(run_eval_module):
    assert run_eval_module.reciprocal_rank([["A"]], ["A"]) == 1.0
    assert run_eval_module.reciprocal_rank([["A"]], ["X", "A"]) == 0.5
    assert run_eval_module.reciprocal_rank([["A"]], ["X", "Y"]) == 0.0


def test_load_cases_shape(run_eval_module):
    payload = run_eval_module.load_cases()
    assert payload["cases"]
    assert payload["refusal_cases"]
    assert payload["faq_cases"]
    for case in payload["cases"]:
        assert case["id"] and case["question"]
        assert case["expect_source_groups"]


def test_summarize_computes_lift(run_eval_module):
    rows = [
        {"hit_rate": 1.0, "dense_hit_rate": 0.5, "bm25_hit_rate": 1.0, "traceable": True,
         "coverage": 1.0, "numbers_ok": True, "support_rate": 1.0, "latency_ms": 10.0,
         "recall_at_1": 1.0, "top1_hit": True, "dense_top1_hit": False, "mrr": 1.0, "dense_mrr": 0.5},
        {"hit_rate": 1.0, "dense_hit_rate": 0.5, "bm25_hit_rate": 1.0, "traceable": True,
         "coverage": 1.0, "numbers_ok": True, "support_rate": 1.0, "latency_ms": 20.0,
         "recall_at_1": 1.0, "top1_hit": True, "dense_top1_hit": False, "mrr": 1.0, "dense_mrr": 0.5},
    ]
    metrics = run_eval_module.summarize(rows)
    assert metrics["retrieval_hit_rate"] == 1.0
    assert metrics["dense_only_hit_rate"] == 0.5
    assert metrics["complex_question_retrieval_lift_pct"] == pytest.approx(100.0)
    assert metrics["top1_accuracy"] == 1.0
    assert metrics["mrr"] == 1.0
    assert metrics["avg_latency_ms"] == pytest.approx(15.0)


# ---------------------------------------------------------------------------
# 端到端跑一次
# ---------------------------------------------------------------------------
def test_run_eval_end_to_end(run_eval_module, tmp_path):
    """完整跑一次评测：核心指标必须全部达标，退出码为 0（可直接卡 CI）。"""
    out = tmp_path / "report.json"
    code = run_eval_module.main(["--json", str(out)])
    assert code == 0, "评测未达标，说明检索或生成链路出现回归"

    report = json.loads(out.read_text(encoding="utf-8"))
    metrics = report["metrics"]
    assert metrics["retrieval_hit_rate"] >= 0.9
    assert metrics["citation_traceability_rate"] >= 0.99
    assert metrics["number_faithfulness_rate"] >= 0.99
    assert metrics["answer_relevance"] >= 0.85
    assert metrics["refusal_accuracy"] >= 0.99
    assert metrics["faq_accuracy"] >= 0.9
    assert metrics["top1_accuracy"] > metrics["dense_only_top1_accuracy"]
    assert metrics["mrr"] > metrics["dense_only_mrr"]


def test_run_eval_report_contains_case_details(run_eval_module, tmp_path):
    out = tmp_path / "report2.json"
    run_eval_module.main(["--json", str(out), "--no-baseline"])
    report = json.loads(out.read_text(encoding="utf-8"))
    assert len(report["cases"]) >= 10
    assert report["faq_cases"]
    assert report["refusal_cases"]
    assert all("hit_rate" in row for row in report["cases"])
