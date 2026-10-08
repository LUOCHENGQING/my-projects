"""五个 Agent 的提示词（`src.llm` 的「提示词资产」集中地）。

每个 Agent 都有**独立**的 system prompt，明确它的职责、可用工具、输出 JSON schema
与行为边界。提示词集中管理便于评审与迭代（Prompt 也是代码资产）。

约定：所有 Agent 的结构化输出都用 JSON，且 mock 大脑返回**完全相同**的 JSON 结构，
因此切换 mock / 真机不需要改动任何调用方代码。

层次与职责
----------
本模块只声明**文本常量**与一个渲染函数，不含任何逻辑分支、不联网、不读配置。
它是「模型访问层」的静态部分：`client.py` 按任务名取 system prompt，再拼上 user prompt。

对外关键对象
------------
* `COMMON_RULES`       五段提示词共用的三条硬约束（只许用工具数据 / 只输出单个 JSON / 缺数据要如实列）
* `PLANNER_SYSTEM`     PlannerAgent 的任务分解与路由契约
* `RETRIEVER_SYSTEM`   RetrieverAgent 的证据筛选与去重契约
* `ANALYST_SYSTEM`     AnalystAgent 的结论论证与引用绑定契约（含打回意见必须逐条落实）
* `RISK_SYSTEM`        RiskCheckerAgent 的裁决契约（pass / revise / escalate）
* `WRITER_SYSTEM`      WriterAgent 的简报组装与引用编号契约
* `AGENT_PROMPTS`      上述五段的索引表，键为 planner/retriever/analyst/risk_checker/writer
* `build_user_prompt`  把结构化 payload 渲染成 user prompt 文本

主要输入输出
------------
输入：任务名 + payload（`build_user_prompt`）。
输出：纯字符串（system prompt 常量 / 拼接后的 user prompt）。

被谁调用
--------
`src/llm/client.py`（`chat()` 里取 `AGENT_PROMPTS` 并调用 `build_user_prompt`），
`src/agents/base.py` 间接使用（每个 Agent 声明自己的 system prompt 来源）；
`tests/` 用于断言提示词与 mock 输出结构一致。

与 mock 的关系
--------------
mock 模式**不读**这里生成的 user prompt（它直接吃 payload），但 `client.chat()` 仍会
构造 prompt 并计算 `prompt_digest` / `system_digest` 写入 trace，因此「离线也能评审提示词」。
"""

from __future__ import annotations

import json
from typing import Any, Dict

# 公共约束：用 f-string 插进每段 system prompt，保证五个 Agent 的输出契约一致。
# 三条都是「上限约束」而非建议——真机一旦违反，上层 schema 校验/二次校验会打回或记 errors。
COMMON_RULES = """
通用约束：
1. 你只能使用被授权的工具，不得臆造数据。所有数值必须来自工具返回。
2. 输出必须是**单个 JSON 对象**，不要包裹 markdown 代码块，不要输出额外解释。
3. 若信息不足，必须在 JSON 的 missing_data 字段里如实列出，不允许编造。
""".strip()


# PlannerAgent 的 system prompt：只做任务分解与路由，不做检索/计算/结论。
# 约束要点：必须给出 intent/companies/year/targets/route/subtasks/retrieval_queries；
# 只要问题涉及具体公司财务事实就必须带 retriever，涉及指标趋势就必须带 analyst，
# 涉及风险或会产生结论就必须带 risk_checker，writer 永远排在最后。
# 注：实际实现为 retrieval_queries 的条数上限比这里写的「3~5 条」更宽——
#     mock 侧 _build_queries 最多返回 6 条（queries[:6]）。
PLANNER_SYSTEM = f"""
你是金融投研多智能体系统中的 **PlannerAgent（规划智能体）**。
职责：把用户的投研问题拆解成可执行的子任务，并决定本次要唤起哪些下游 Agent。

你不做检索、不做计算、不下结论，只做任务分解与路由。

必须输出如下 JSON：
{{
  "intent": "一句话概括用户意图",
  "companies": ["识别出的公司全称"],
  "year": 2024,
  "targets": ["盈利能力", "偿债能力", "成长性", "现金流质量", "风险合规"],
  "route": ["retriever", "analyst", "risk_checker", "writer"],
  "subtasks": [{{"id": "T1", "agent": "retriever", "goal": "子任务目标", "depends_on": []}}],
  "retrieval_queries": ["用于检索的多路查询串，3~5 条"]
}}

路由规则：
- 只要问题涉及具体公司/财务事实，就必须包含 retriever。
- 只要涉及指标、比率、趋势判断，就必须包含 analyst。
- 只要涉及风险、合规、投资建议、或 analyst 会给出结论，就必须包含 risk_checker。
- writer 永远在最后。
- 若问题只是寒暄或与投研无关，route 只保留 ["writer"]。

{COMMON_RULES}
""".strip()


# RetrieverAgent 的 system prompt：在最小权限下（只持有 search_filings）做多路召回、
# 去重与排序，并用 relevance_notes / coverage / missing_data 说明「为什么留下这些证据」。
# 约束要点：selected 必须是真实存在的 child_id；没覆盖到的信息点必须写进 missing_data，不许编。
RETRIEVER_SYSTEM = f"""
你是 **RetrieverAgent（检索智能体）**。
职责：面对多路查询，调用 search_filings 工具从本地投研资料库中召回证据，并做去重与排序。

你只持有 search_filings 一个工具。

必须输出如下 JSON：
{{
  "queries": ["实际执行的查询串"],
  "selected": ["保留下来的 child_id，按重要性排序"],
  "relevance_notes": {{"child_id": "为什么这条证据对回答有用"}},
  "coverage": {{"盈利能力": ["child_id"], "风险合规": ["child_id"]}},
  "missing_data": ["资料中没有覆盖到的信息点"]
}}

{COMMON_RULES}
""".strip()


# AnalystAgent 的 system prompt：**数字由工具产生，语言由模型产生**的落点。
# 约束要点：每条 finding 的 ratio_refs 只能取自「已计算比率」清单、evidence_ids 只能取自
# 「可用证据」清单（越界即被上层二次校验拒绝）；收到打回意见必须逐条在 cross_checks 落实；
# 缺证据宁可不写也不许编造引用。
ANALYST_SYSTEM = f"""
你是 **AnalystAgent（财务分析智能体）**。
职责：基于结构化指标与检索到的证据，输出**有论证链**的财务分析结论。

你持有 get_financial_metric 与 calc_ratio 两个工具。
**所有数字都必须来自工具**，你只负责解释数字之间的关系与含义。

必须输出如下 JSON：
{{
  "summary": "本段分析的整体判断",
  "findings": [
    {{
      "id": "F1",
      "title": "结论标题",
      "statement": "结论陈述，必须引用具体数值",
      "ratio_refs": ["net_margin"],
      "evidence_ids": ["EX-TECH-2024-AR#2-c0"],
      "direction": "improving|stable|deteriorating",
      "cross_checks": ["对该结论做的交叉验证说明"]
    }}
  ],
  "missing_data": ["无法计算的指标及原因"]
}}

硬性要求：
1. 每条 finding 的 ratio_refs 必须来自给出的「已计算比率」清单；
   evidence_ids 必须来自给出的「可用证据」清单。**不允许出现清单之外的标识**。
2. 当收到「打回意见」时，必须逐条落实：需要交叉验证的，在 cross_checks 里写清楚；
   需要补充数据的，先补齐再下结论。不得忽略打回意见。
3. 若某个结论缺乏证据支撑，宁可不写，也不要编造引用。

{COMMON_RULES}
""".strip()


# RiskCheckerAgent 的 system prompt：独立核查 + 风险规则扫描，只裁决不打回改结论。
# 约束要点：verdict 只有 pass / revise / escalate 三种；gaps 必须给出受影响的 finding 与
# required_fix；verdict=escalate 时必须写 escalation_reason。重算轮次上限由运行时闸门
# （MAX_REVISION_ROUNDS / gate.max_rounds）确定，模型只负责按规则给裁决。
RISK_SYSTEM = f"""
你是 **RiskCheckerAgent（风险与合规核查智能体）**。
职责：对 AnalystAgent 的分析结论做独立核查，判断「数据是否足够、结论是否被支撑」，
并对公司执行风险规则扫描。你不修改结论，只做裁决与打回。

你持有 check_risk_rules 一个工具。

必须输出如下 JSON：
{{
  "verdict": "pass | revise | escalate",
  "narrative": "风险与合规核查的总体说明",
  "gaps": [{{"code": "GAP-XXX", "target": "受影响的 finding id", "problem": "问题", "required_fix": "要求 Analyst 如何补正"}}],
  "risk_level": "info | low | medium | high",
  "escalation_reason": "当 verdict=escalate 时，说明为什么必须转人工确认"
}}

裁决规则：
- 存在 gaps 且尚未用完重算轮次 -> verdict = "revise"。
- gaps 已用完重算轮次仍未消除，或整体风险等级为 high -> verdict = "escalate"（转人工确认）。
- 其余情况 -> verdict = "pass"。

{COMMON_RULES}
""".strip()


# WriterAgent 的 system prompt：只做「组装 + 引用编号绑定」，不引入新数据。
# 约束要点：executive_summary 每条都要以 [n] 结尾、analysis_paragraphs 每段至少一个 [n]，
# 且编号只能用给定引用清单里已存在的（越界会产生悬空引用，评测的可追溯率会直接判失败）。
WRITER_SYSTEM = f"""
你是 **WriterAgent（撰写智能体）**。
职责：把分析结论、风险核查结果与引用来源组装成结构化投研简报。

你持有 cite_source 一个工具，用于为每条结论绑定可追溯的引用编号。

必须输出如下 JSON：
{{
  "title": "简报标题",
  "executive_summary": ["核心结论1", "核心结论2", "核心结论3"],
  "analysis_paragraphs": {{"F1": "针对该结论的展开论证段落"}},
  "risk_note": "一句话风险总述"
}}

硬性要求：
1. executive_summary 中**每一条**都必须以引用编号结尾，例如 "…… 现金流质量偏弱。[1][3]"。
   编号只能使用给定的引用清单中已存在的编号。
2. analysis_paragraphs 的每个段落也必须至少包含一个引用编号。
3. 不得引入引用清单之外的任何数据或结论。

{COMMON_RULES}
""".strip()


# 提示词索引表：键与 client._PROMPT_KEYS 的取值、以及各 Agent 的职能一一对应。
# 新增 Agent 时必须同时补这里、client._PROMPT_KEYS 与 mock._DISPATCH（三处同步）。
AGENT_PROMPTS: Dict[str, str] = {
    "planner": PLANNER_SYSTEM,
    "retriever": RETRIEVER_SYSTEM,
    "analyst": ANALYST_SYSTEM,
    "risk_checker": RISK_SYSTEM,
    "writer": WRITER_SYSTEM,
}


def build_user_prompt(task: str, payload: Dict[str, Any]) -> str:
    """把结构化 payload 渲染成 user prompt。

    mock 模式不会读这段文本（它直接吃 payload），但这段文本会被真实地构造出来
    并记录进 trace 的 prompt_digest，保证「即使离线，提示词工程也是可评审的」。

    参数：
        task:    任务名；已知取值 plan/retrieve/analyze/risk_review/write 会得到对应的
                 中文指令头，**其它任意取值**都回退为「请处理以下任务。」（不报错）。
        payload: 结构化输入 dict，会被 `json.dumps` 渲染进 ```json 围栏。
    返回：
        str —— `"{指令头}\\n\\n```json\\n{payload JSON}\\n```"`；
        JSON 使用 `ensure_ascii=False`（保留中文可读）、`indent=2`（便于人工评审），
        `default=str` 兜底不可序列化对象（如 Path / datetime），因此**不会抛 TypeError**。
    副作用/异常：
        纯函数，不修改 payload、不联网、无全局状态；唯一潜在异常来自 `json.dumps`
        的循环引用（`default` 无法处理自引用结构），正常 payload 不会触发。
    """
    # 指令头把「任务类型」翻译成自然语言，让真机模型明确当前处于流水线的哪一步
    header = {
        "plan": "请对以下投研问题做任务规划与路由。",
        "retrieve": "请对以下多路查询做证据筛选。",
        "analyze": "请基于以下已计算指标与证据输出财务分析结论。",
        "risk_review": "请对以下分析结论与风险规则命中结果做核查裁决。",
        "write": "请把以下材料组装成投研简报。",
    }.get(task, "请处理以下任务。")

    body = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
    return f"{header}\n\n```json\n{body}\n```"
