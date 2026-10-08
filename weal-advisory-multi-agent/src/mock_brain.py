"""确定性 mock 大脑（无 API Key 时的降级推理）。

层级
----
措辞生成层，被 `src/llm.py` 的 `MockLLM` 单向调用（`llm.py` 只 import 本模块，
本模块不反向依赖 `llm.py`，因此不存在循环依赖）。它位于确定性业务层
（`src/constraints.py` / `src/optimizer.py` / `src/suitability/` / `src/stress.py` /
`src/counterfactual.py`）之上，只消费这些模块已经算好的结构化事实。

定位
----
**它不是"假装调用模型"**，而是一组确定性的中文文案生成器：所有业务判断
（约束求解、产品筛选、权重分配、适当性判定、压力测试）都由确定性代码完成，
mock 只负责把已经算好的结构化事实组织成人话。因此：

- demo 在没有网络、没有 Key 的环境下也一定能跑通且输出有意义的内容；
- mock 输出完全可复现，适合写进单测断言；
- 接上真实模型后，语义不变，只是措辞更自然。

输出是怎么算出来的（可复现性保证）
----------------------------------
每个任务对应一个 `_xxx(ctx)` 纯函数：只用 `ctx.get(...)` 取字段、只用 f-string
与 `pct()`（`src/utils.py`，比率按 `value * 100` 格式化为百分比）拼字符串，
**不含随机数、不读时钟、不发网络请求、不读全局可变状态**。因此同一份 `ctx`
必然得到逐字节相同的文案；缺字段时走 `ctx.get(key, 默认值)` 的兜底分支而不是抛异常。

约定：`ctx` 中的比率类字段一律是 0~1 的小数（如 `0.35` 表示 35%），
金额类字段是元；`mock_compose` 返回的字典键与下方 `TASK_SCHEMAS` 完全一致。

任务清单（key 即 `compose(task=...)` 的 task 名）
-------------------------------------------------
`client_profile_summary` / `screening_note` / `optimizer_rationale` /
`suitability_comment` / `advisor_summary` / `risk_disclosure` / `counterfactual_note`

对应字段（真实模型也必须按 `TASK_SCHEMAS` 返回同样的 JSON 结构）：
`summary` + `consistency_note` / `note` / `rationale` / `comment` / `summary` /
`disclosure` / `note`。
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

from .utils import pct

#: 各任务要求的输出字段（真实模型也必须按此 JSON 结构返回）
#: key 为 `compose(task=...)` 的 task 名，value 为必须存在的字段名元组；
#: 这一份表同时是 `llm._prompt_for()` 的字段白名单与 `FallbackLLM` 的字段校验依据。
TASK_SCHEMAS: dict[str, tuple[str, ...]] = {
    "client_profile_summary": ("summary", "consistency_note"),
    "screening_note": ("note",),
    "optimizer_rationale": ("rationale",),
    "suitability_comment": ("comment",),
    "advisor_summary": ("summary",),
    "risk_disclosure": ("disclosure",),
    "counterfactual_note": ("note",),
}


def _fmt_money(value: float) -> str:
    """金额千分位格式化。

    参数：
        value：金额（元）。
    返回：
        `f"{value:,.0f} 元"`，即四舍五入到整数、带千分位并附「元」后缀的字符串
        （注意：返回的是字符串，不是数值）。
    """
    return f"{value:,.0f} 元"


def _client_profile_summary(ctx: Mapping[str, Any]) -> dict[str, str]:
    """客户画像摘要。

    参数：
        ctx：客户画像上下文字典，实际由 `ClientProfilingAgent` 组装，字段为
            `display_name` / `age` / `risk_level`（**有效等级**，已按从严原则取
            档案与问卷的孰低者）/ `archived_level`（档案登记等级）/
            `questionnaire_level`（问卷折算等级，可能为 None）/ `horizon_years` /
            `investable_amount` / `caps`（含 `product` / `asset_class` / `issuer`
            三个集中度上限）/ `liquidity_floor` / `prohibited`（禁止品类与禁止产品号
            的合并列表）/ `experience`（已具备经验的品类列表）。

    返回：
        `{"summary": str, "consistency_note": str}`。
        `consistency_note` 分三种确定性分支：问卷等级低于档案等级 → 取孰低并书面提示；
        高于档案等级 → 仍取档案等级、不做上调；相等或任一侧缺失 → 说明无需调整。

    侧记：注：`summary` 里的 `风险承受等级 R{level}` 引用的是 `risk_level`，
    调用方传入的是**收紧后的有效等级**，因此该字段描述的是最终生效等级而非档案等级。
    """
    name = ctx.get("display_name", "客户")
    age = ctx.get("age", 0)
    level = ctx.get("risk_level", 3)
    horizon = ctx.get("horizon_years", 0)
    amount = ctx.get("investable_amount", 0.0)
    caps = ctx.get("caps", {})
    summary = (
        f"{name}，{age} 周岁，风险承受等级 R{level}，投资期限约 {horizon:g} 年，"
        f"可投金额 {_fmt_money(amount)}；"
        f"集中度约束为单一产品 {pct(caps.get('product', 0.0))}、"
        f"单一类别 {pct(caps.get('asset_class', 0.0))}、"
        f"同一发行人 {pct(caps.get('issuer', 0.0))}，"
        f"流动性资产占比不低于 {pct(ctx.get('liquidity_floor', 0.0))}。"
    )
    prohibited = ctx.get("prohibited") or []
    if prohibited:
        summary += f"客户明确排除：{'、'.join(prohibited)}。"
    experience = ctx.get("experience") or []
    summary += f"已具备投资经验的品类：{'、'.join(experience) if experience else '无'}。"

    archived = ctx.get("archived_level")
    scored = ctx.get("questionnaire_level")
    if scored is not None and archived is not None and int(scored) < int(archived):
        consistency = (
            f"风险测评问卷折算等级为 R{scored}，低于档案登记等级 R{archived}，"
            "按从严原则取孰低者（R" + str(scored) + "）作为硬约束上限，并以书面方式提示客户。"
        )
    elif scored is not None and archived is not None and int(scored) > int(archived):
        consistency = (
            f"风险测评问卷折算等级为 R{scored}，高于档案登记等级 R{archived}，"
            "按从严原则仍以较低的档案等级 R" + str(archived) + " 作为硬约束上限，不做上调。"
        )
    else:
        consistency = "问卷折算等级与档案登记等级一致，无需按从严原则调整。"
    return {"summary": summary, "consistency_note": consistency}


def _screening_note(ctx: Mapping[str, Any]) -> dict[str, str]:
    """候选池筛选说明。

    参数：
        ctx：筛选结果上下文，字段为 `universe_size`（全市场候选只数）/
            `included_count`（落入硬约束可行域的只数）/ `top_exclusions`
            （剔除原因字符串列表，取前 3 条）。
    返回：
        `{"note": str}`；剔除只数由 `universe_size - included_count` 现算得出，
        不读取额外字段。
    """
    universe = ctx.get("universe_size", 0)
    included = ctx.get("included_count", 0)
    excluded = universe - included
    top = ctx.get("top_exclusions") or []
    note = f"在 {universe} 只候选产品中，{included} 只落入硬约束可行域，剔除 {excluded} 只。"
    if top:
        note += "主要剔除原因：" + "；".join(str(item) for item in top[:3]) + "。"
    note += "被剔除产品不进入后续组合构建，剔除原因逐条留痕，可向客户逐项解释。"
    return {"note": note}


def _optimizer_rationale(ctx: Mapping[str, Any]) -> dict[str, str]:
    """组合权衡说明。

    参数：
        ctx：组合上下文，字段为 `metrics`（含 `expected_return` /
            `expected_volatility` / `liquidity_ratio` / `max_single_weight`）/
            `binding`（当前起约束作用的上限描述字符串列表，可为空）/
            `class_targets`（`{类别名: 目标权重}`，只列出大于 0 的项并按类别名排序）。
    返回：
        `{"rationale": str}`；`binding` 为空时输出"未触及任何硬约束上限"的分支。
    """
    metrics = ctx.get("metrics", {})
    binding = ctx.get("binding") or []
    targets = ctx.get("class_targets") or {}
    parts = [
        "组合按「期望收益 / 波动 / 流动性 / 集中度」四目标折中求解，"
        f"最终预期年化收益 {pct(metrics.get('expected_return', 0.0))}、"
        f"预期波动 {pct(metrics.get('expected_volatility', 0.0))}、"
        f"组合流动性资产占比 {pct(metrics.get('liquidity_ratio', 0.0))}、"
        f"最大单一持仓 {pct(metrics.get('max_single_weight', 0.0))}。"
    ]
    if targets:
        parts.append(
            "类别基准配置为 "
            + "、".join(f"{k} {pct(v)}" for k, v in sorted(targets.items()) if v > 0)
            + "。"
        )
    if binding:
        parts.append("当前起约束作用的上限：" + "；".join(binding) + "。")
    else:
        parts.append("当前组合未触及任何硬约束上限，仍有一定调整空间。")
    return {"rationale": "".join(parts)}


def _suitability_comment(ctx: Mapping[str, Any]) -> dict[str, str]:
    """适当性复核意见。

    参数：
        ctx：闸门上下文，字段为 `directive`（`pass` / `reoptimize` / `reject`，
            缺省按 `pass` 处理）/ `block_rules`（block 级命中规则号列表）/
            `warn_rules`（warn 级命中规则号列表）/ `round_index`（0 基轮次，
            文案中展示为 `round_index + 1`）。
    返回：
        `{"comment": str}`；`directive` 为其他未登记取值时返回通用文案
        "适当性复核完成。"，不抛异常。带 `warn_rules` 时追加揭示提示。
    """
    directive = ctx.get("directive", "pass")
    blocks = ctx.get("block_rules") or []
    warns = ctx.get("warn_rules") or []
    round_index = int(ctx.get("round_index", 0)) + 1
    if directive == "pass":
        comment = f"第 {round_index} 轮适当性复核通过，未命中 block 级规则。"
    elif directive == "reoptimize":
        comment = (
            f"第 {round_index} 轮适当性复核不通过，命中 {'、'.join(blocks)}，"
            "已下发约束收紧指令并打回重新配置。"
        )
    elif directive == "reject":
        comment = f"第 {round_index} 轮适当性复核判定不予通过，规则命中：{'、'.join(blocks)}。"
    else:
        comment = "适当性复核完成。"
    if warns:
        comment += f"另有告警级提示需向客户揭示：{'、'.join(warns)}。"
    return {"comment": comment}


def _advisor_summary(ctx: Mapping[str, Any]) -> dict[str, str]:
    """建议书摘要。

    参数：
        ctx：建议书上下文，字段为 `display_name` / `holding_count`（持仓只数）/
            `cash_weight`（现金及活期留存权重）/ `metrics`（含 `expected_return` /
            `expected_volatility`）/ `status`（`rejected` 时输出"不出具配置建议"
            分支，其余取值一律输出正常结论分支）。
    返回：
        `{"summary": str}`。
    """
    name = ctx.get("display_name", "客户")
    holdings = ctx.get("holding_count", 0)
    cash = ctx.get("cash_weight", 0.0)
    metrics = ctx.get("metrics", {})
    status = ctx.get("status", "draft")
    summary = (
        f"本建议书为 {name} 生成，共配置 {holdings} 只产品，"
        f"现金及活期留存 {pct(cash)}，"
        f"预期年化收益 {pct(metrics.get('expected_return', 0.0))}，"
        f"预期波动 {pct(metrics.get('expected_volatility', 0.0))}。"
    )
    if status == "rejected":
        summary += "因硬约束下不存在可行组合，本次不出具配置建议，仅出具拒绝对照说明。"
    else:
        summary += "全部持仓均落在客户硬约束可行域内，并通过适当性规则复核。"
    return {"summary": summary}


def _risk_disclosure(ctx: Mapping[str, Any]) -> dict[str, str]:
    """风险揭示。

    参数：
        ctx：风险上下文，字段为 `risk_level`（客户有效风险承受等级）/
            `drawdown_tolerance`（客户约定的最大回撤容忍度，0~1）/
            `worst_scenario_name`（最不利情景名，调用方已按组合冲击最小值选出）/
            `worst_drawdown`（该情景下的估计最大回撤，0~1）。
    返回：
        `{"disclosure": str}`；文案固定声明测算为确定性推算、不构成收益承诺。

    侧记：情景名的默认值是"压力情景"，而在 `narrative.build_narrative` 中传入的
    默认值是"情景压力测试"，两处默认文案不同（仅为兜底措辞，不影响数值）。
    """
    level = ctx.get("risk_level", 3)
    tolerance = ctx.get("drawdown_tolerance", 0.0)
    worst = ctx.get("worst_scenario_name", "压力情景")
    worst_drawdown = ctx.get("worst_drawdown", 0.0)
    disclosure = (
        f"本组合对应客户风险承受等级 R{level}，客户约定的最大回撤容忍度为 {pct(tolerance)}。"
        f"在「{worst}」情景下，组合估计最大回撤约 {pct(worst_drawdown)}。"
        "压力测试结果基于示例敏感性系数表的确定性推算，不构成对未来收益或损失的承诺；"
        "投资组合可能发生本金损失，历史业绩与情景测算均不代表未来表现。"
    )
    return {"disclosure": disclosure}


def _counterfactual_note(ctx: Mapping[str, Any]) -> dict[str, str]:
    """反事实解释导读。

    参数：
        ctx：反事实上下文，字段为 `variants`（假设变体列表，只取 `len()` 计数）/
            `non_empty`（产生实质差异的变体数）。
    返回：
        `{"note": str}`；强调结论由「修改约束 → 重新求解 → 差异比对」得出且可复现。
    """
    variants = ctx.get("variants") or []
    covered = ctx.get("non_empty", 0)
    note = (
        f"共对 {len(variants)} 条假设约束变化重新求解并做结构化差异比较，"
        f"其中 {covered} 条产生了实质差异。"
        "反事实结论由「修改约束 → 重新求解 → 差异比对」得出，可复现、可复核。"
    )
    return {"note": note}


#: 任务 -> 生成器
#: key 必须与 `TASK_SCHEMAS` 的 key 一一对应；未登记的任务由 `mock_compose` 兜底。
BUILDERS: dict[str, Callable[[Mapping[str, Any]], dict[str, str]]] = {
    "client_profile_summary": _client_profile_summary,
    "screening_note": _screening_note,
    "optimizer_rationale": _optimizer_rationale,
    "suitability_comment": _suitability_comment,
    "advisor_summary": _advisor_summary,
    "risk_disclosure": _risk_disclosure,
    "counterfactual_note": _counterfactual_note,
}


def mock_compose(task: str, context: Mapping[str, Any]) -> dict[str, str]:
    """按任务生成确定性文案（未知任务返回通用占位说明，不抛异常）。

    参数：
        task：任务名，取 `TASK_SCHEMAS` / `BUILDERS` 中的 key。
        context：结构化事实字典；`mock` 只读取、不修改该字典。

    返回：
        `dict[str, str]`，键与 `TASK_SCHEMAS[task]` 一致（缺失的字段补空串）；
        未登记的任务返回 `{"note": "（mock 大脑未登记任务 …，已按确定性规则跳过）"}`。

    副作用/异常：无副作用；任何输入都不抛异常（这是"无 Key 也能跑通"的最后一层兜底）。
    """
    builder = BUILDERS.get(task)
    if builder is None:
        return {"note": f"（mock 大脑未登记任务 {task}，已按确定性规则跳过）"}
    result = builder(context)
    # 保证输出字段与 schema 一致
    for key in TASK_SCHEMAS.get(task, ()):
        result.setdefault(key, "")
    return result
