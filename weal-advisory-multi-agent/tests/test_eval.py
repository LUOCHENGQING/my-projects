"""评估脚本测试（`eval/run_eval.py`）。

覆盖对象
--------
7 项指标的定义与阈值（`THRESHOLDS`）、「预言机」口径（`oracle_admissible` /
`candidate_filter_accuracy`）、对客越界探针（`suitability_probes`）、
文本报告渲染（`render_text`）与 CLI 入口（`main`）。

覆盖策略
--------
- **阈值达成**：7 项指标全部达标，`report["failures"]` 为空列表。
- **交叉校验**：候选池筛选结果必须与「直接调用准入判定」的预言机逐条一致
  （`test_oracle_matches_direct_admissibility`）—— 这一条专门用来防止
  预言机自身写歪，否则它的 1.0 准确率毫无意义。
- **硬不变式**：被接受的组合违反数恒为 0；对客探针必须被全部拦下；
  block 场景必须被正确处理。
- **可复现**：换引擎（native）重跑，与实现无关的指标必须完全一致。
- **接口契约**：CLI 退出码、报告文件落盘、报告文案包含全部指标标签。

成本控制：`report` 夹具是 `module` 作用域，整个模块只跑一次完整评估
（一次评估会跑 5 位客户，含压力测试与反事实，开销明显大于普通单测）。
"""

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
    """契约：评估报告恰好给出约定的 7 项指标（不多不少，改指标名即视为破坏契约）。"""
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
    """验收：7 项指标全部达标，且 `failures` 为空列表。"""
    assert report["failures"] == []
    for name, threshold in THRESHOLDS.items():
        # 耗时是"上限型"指标（越小越好），其余 6 项是"下限型"指标（越大越好）
        if name == "avg_latency_ms":
            assert report["metrics"][name] <= threshold
        else:
            assert report["metrics"][name] >= threshold, name


def test_accepted_portfolios_have_zero_violations(report):
    """硬不变式：被接受的组合违反硬约束数恒为 0（计数器与逐客户明细双向核对）。"""
    counters = report["counters"]
    assert counters["accepted_portfolios"] == counters["accepted_with_zero_violation"]
    for item in report["details"]:
        assert item["violations"] == 0


def test_suitability_probes_are_all_blocked(report):
    """对抗路径：为每位客户构造的越界探针必须被闸门全部拦下，一处漏网即失败。"""
    assert report["counters"]["probe_total"] >= 10
    assert report["counters"]["probe_blocked"] == report["counters"]["probe_total"]
    for item in report["details"]:
        assert all(probe["blocked"] for probe in item["probes"])


def test_block_cases_are_correctly_handled(report):
    """职责：block 场景（本应直接拒绝的客户）必须被正确处理，且至少存在 1 例。"""
    counters = report["counters"]
    assert counters["block_cases"] >= 1
    assert counters["block_handled"] == counters["block_cases"]


def test_counterfactual_and_stress_coverage_are_full(report):
    """覆盖度：每位客户的方案都要有非空反事实，且至少覆盖 3 个压力情景。"""
    for item in report["details"]:
        assert item["counterfactual_coverage"] > 0
        assert item["stress_scenarios"] >= 3


def test_narrative_elements_complete(report):
    """完整度：建议书必备要素全部齐备（elements_ok == elements_total）。"""
    for item in report["details"]:
        assert item["elements_ok"] == item["elements_total"]


def test_candidate_filter_accuracy_matches_oracle(data):
    """一致性：真实筛选结果与预言机逐条一致（准确率 1.0、零失配）。"""
    accuracy, mismatches = candidate_filter_accuracy(data)
    assert mismatches == []
    assert accuracy == 1.0


def test_oracle_matches_direct_admissibility(data):
    """交叉校验：预言机的判定必须与直接调用 `check_product_admissibility` 的结果逐条吻合。"""
    from src.constraints import check_product_admissibility

    for client in data.sample_clients():
        for product in data.products.values():
            direct = not check_product_admissibility(product, client)
            assert direct == oracle_admissible(product, client), f"{client.client_id}/{product.product_id}"


def test_probes_detect_violations_for_every_client(data):
    """职责：探针生成器为每位客户产出 2 条越界组合，且两条都能被拦下。"""
    for client in data.sample_clients():
        probes = suitability_probes(client, data)
        assert len(probes) == 2
        assert all(blocked for _, blocked in probes), client.client_id


def test_render_text_contains_metric_labels(report):
    """契约：文本报告包含全部 7 项指标的中文标签，以及「退出码 0」结论。"""
    text = render_text(report)
    for label in ("约束满足率", "适当性通过率", "候选池过滤正确率", "反事实解释覆盖率", "压力测试覆盖率", "建议书要素完整率", "平均耗时"):
        assert label in text
    assert "退出码 0" in text


def test_main_returns_zero_and_writes_reports(report, tmp_path):
    """CLI 契约：`main` 返回退出码 0，并在指定目录落盘 `report.json` 与 `report.md`。"""
    exit_code = main(["--engine", "langgraph", "--out-dir", str(tmp_path)])
    assert exit_code == 0
    assert (tmp_path / "report.json").exists()
    assert (tmp_path / "report.md").exists()


def test_evaluation_is_reproducible(report):
    """可复现：换引擎重跑，与实现无关的 5 项指标数值完全一致且无失败项。"""
    second = evaluate(engine="native")
    # 只比与实现无关的指标：适当性通过率、平均耗时对执行路径敏感，不做跨引擎相等断言
    for name in (
        "constraint_satisfaction_rate",
        "candidate_filter_accuracy",
        "counterfactual_coverage",
        "stress_coverage",
        "narrative_completeness",
    ):
        assert second["metrics"][name] == report["metrics"][name]
    assert second["failures"] == []
