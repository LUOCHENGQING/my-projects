"""评测脚本测试：指标计算、分组召回率、MRR，以及一次完整的端到端评测跑通。

覆盖的被测模块
--------------
    eval/run_eval.py（不是包的一部分，因此用 importlib 按文件路径加载成模块）：
        load_cases                        评测用例集的结构（检索生成 / 拒答 / FAQ 三类）
        hit_rate / recall_at_1 / reciprocal_rank   纯函数指标的语义与边界
        summarize                         把逐条结果汇总成核心指标（含混合三路相对单一向量的增益）
        main                              端到端跑一次评测、落盘 JSON 报告并返回 CI 可用的退出码

覆盖策略
--------
    正常路径：在构造好的行数据上断言指标的精确值；完整跑一次评测并检查报告内容。
    边界：期望分组为空、召回列表为空、只看 Top-1、只命中部分分组。
    异常 / 回归：端到端断言各项核心指标的门槛（检索命中率 / 引用可追溯率 / 数字忠实度 /
                答案相关度 / 拒答正确率 / FAQ 准确率），任一回归即返回非 0 退出码，可直接卡 CI。
    对抗：混合三路 + 重排必须优于单一向量基线（Top-1 与 MRR 都要更高），
          防止「加了重排却没变好」被忽略。
报告与中间产物统一写入 tmp_path，不污染 eval/ 目录。
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def run_eval_module():
    """把 eval/run_eval.py 当成模块加载（它不是包的一部分）。

    作用域：session——脚本只加载一次，被本文件全部用例（含指标函数与端到端 main）使用。
    提供：加载后的模块对象；同时注册进 sys.modules，保证脚本内部相对引用可用。
    """
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
    """所有期望分组都没被召回时命中率必须为 0：召回了无关资料不算分。"""
    assert run_eval_module.hit_rate([["A"], ["B"]], ["C"]) == 0.0


def test_hit_rate_single_group_any_member(run_eval_module):
    """同一答案分布在多份资料里时，命中任一即算该组命中。"""
    assert run_eval_module.hit_rate([["A", "B", "C"]], ["B"]) == 1.0


def test_hit_rate_partial_groups(run_eval_module):
    """命中率按组取平均：两组中只命中一组即为 0.5（跨文档用例因此能表达「缺一份」）。"""
    assert run_eval_module.hit_rate([["A"], ["B"]], ["A", "Z"]) == 0.5


def test_hit_rate_empty_expectation(run_eval_module):
    """没有期望分组的用例视为满分：不存在可判定的召回要求，不应反向惩罚该用例。"""
    assert run_eval_module.hit_rate([], ["A"]) == 1.0


def test_recall_at_1(run_eval_module):
    """Top-1 必须落在任一标准答案分组内才算命中；召回列表为空一律计 0。"""
    assert run_eval_module.recall_at_1([["A"]], ["A", "B"]) == 1.0
    assert run_eval_module.recall_at_1([["A"]], ["B", "A"]) == 0.0
    assert run_eval_module.recall_at_1([["A"]], []) == 0.0


def test_reciprocal_rank(run_eval_module):
    """MRR 单条贡献是首个命中的排名的倒数，完全未命中计 0（衡量排序质量而非是否召回）。"""
    assert run_eval_module.reciprocal_rank([["A"]], ["A"]) == 1.0
    assert run_eval_module.reciprocal_rank([["A"]], ["X", "A"]) == 0.5
    assert run_eval_module.reciprocal_rank([["A"]], ["X", "Y"]) == 0.0


def test_load_cases_shape(run_eval_module):
    """用例集必须同时含检索生成、拒答、FAQ 三类样本，且每条检索用例带 id、问题与期望来源分组。"""
    payload = run_eval_module.load_cases()
    assert payload["cases"]
    assert payload["refusal_cases"]
    assert payload["faq_cases"]
    for case in payload["cases"]:
        assert case["id"] and case["question"]
        assert case["expect_source_groups"]


def test_summarize_computes_lift(run_eval_module):
    """summarize 必须把逐条结果汇总为整体指标，并算出混合三路相对单一向量的提升百分比。"""
    # 两行只差耗时（10 / 20 ms），混合命中率 1.0 对单一向量 0.5，增益正好是 100%
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
    """完整跑一次评测：核心指标必须全部达标，退出码为 0（可直接卡 CI）。

    不变式：这是「检索 + 生成」整条链路的回归门禁，任一核心指标跌破门槛即失败。
    注：实际实现为——报告里的 `answer_relevance` 是逐条 `coverage`（人工标注要点覆盖率）
    的均值，并非 faithfulness 里的 token 级 F1；`refusal_accuracy` / `faq_accuracy`
    由 main() 另行追加，不在 summarize() 的输出里。
    """
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
    """报告必须保留逐条明细与拒答 / FAQ 分组结果，指标不达标时才能下钻定位到具体用例。"""
    out = tmp_path / "report2.json"
    # --no-baseline 跳过单一向量对照，让本条用例只关心报告结构（也顺带覆盖该开关）
    run_eval_module.main(["--json", str(out), "--no-baseline"])
    report = json.loads(out.read_text(encoding="utf-8"))
    assert len(report["cases"]) >= 10
    assert report["faq_cases"]
    assert report["refusal_cases"]
    assert all("hit_rate" in row for row in report["cases"])
