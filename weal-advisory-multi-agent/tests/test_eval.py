"""评估脚本测试：7 项指标达标、退出码、报告渲染。"""

from __future__ import annotations

import pytest

from eval.run_eval import (
    THRESHOLDS,
    candidate_filter_accuracy,
    evaluate,
    main,
    oracle_admissible,
    render_text,
    suitability_probes,
)


@pytest.fixture(scope="module")
def report() -> dict:
    """模块级复用一次完整评估（评估本身会跑 5 条客户，避免重复开销）。"""
    return evaluate(engine="langgraph")


def test_evaluate_returns_all_seven_metrics(report):
    assert set(report["metrics"]) == {
        "constraint_satisfaction_rate",
        "suitability_pass_rate",
        "candidate_filter_accuracy",
        "counterfactual_coverage",
        "stress_coverage",
        "narrative_completeness",
        "avg_latency_ms",
    }


def test_all_metrics_meet_thresholds(report):
    assert report["failures"] == []
    for name, threshold in THRESHOLDS.items():
        if name == "avg_latency_ms":
            assert report["metrics"][name] <= threshold
        else:
            assert report["metrics"][name] >= threshold, name


def test_accepted_portfolios_have_zero_violations(report):
    counters = report["counters"]
    assert counters["accepted_portfolios"] == counters["accepted_with_zero_violation"]
    for item in report["details"]:
        assert item["violations"] == 0


def test_suitability_probes_are_all_blocked(report):
    assert report["counters"]["probe_total"] >= 10
    assert report["counters"]["probe_blocked"] == report["counters"]["probe_total"]
    for item in report["details"]:
        assert all(probe["blocked"] for probe in item["probes"])


def test_block_cases_are_correctly_handled(report):
    counters = report["counters"]
    assert counters["block_cases"] >= 1
    assert counters["block_handled"] == counters["block_cases"]


def test_counterfactual_and_stress_coverage_are_full(report):
    for item in report["details"]:
        assert item["counterfactual_coverage"] > 0
        assert item["stress_scenarios"] >= 3


def test_narrative_elements_complete(report):
    for item in report["details"]:
        assert item["elements_ok"] == item["elements_total"]


def test_candidate_filter_accuracy_matches_oracle(data):
    accuracy, mismatches = candidate_filter_accuracy(data)
    assert mismatches == []
    assert accuracy == 1.0


def test_oracle_matches_direct_admissibility(data):
    from src.constraints import check_product_admissibility

    for client in data.sample_clients():
        for product in data.products.values():
            direct = not check_product_admissibility(product, client)
            assert direct == oracle_admissible(product, client), f"{client.client_id}/{product.product_id}"


def test_probes_detect_violations_for_every_client(data):
    for client in data.sample_clients():
        probes = suitability_probes(client, data)
        assert len(probes) == 2
        assert all(blocked for _, blocked in probes), client.client_id


def test_render_text_contains_metric_labels(report):
    text = render_text(report)
    for label in ("约束满足率", "适当性通过率", "候选池过滤正确率", "反事实解释覆盖率", "压力测试覆盖率", "建议书要素完整率", "平均耗时"):
        assert label in text
    assert "退出码 0" in text


def test_main_returns_zero_and_writes_reports(report, tmp_path):
    exit_code = main(["--engine", "langgraph", "--out-dir", str(tmp_path)])
    assert exit_code == 0
    assert (tmp_path / "report.json").exists()
    assert (tmp_path / "report.md").exists()


def test_evaluation_is_reproducible(report):
    second = evaluate(engine="native")
    for name in (
        "constraint_satisfaction_rate",
        "candidate_filter_accuracy",
        "counterfactual_coverage",
        "stress_coverage",
        "narrative_completeness",
    ):
        assert second["metrics"][name] == report["metrics"][name]
    assert second["failures"] == []
