"""命令行演示入口：`python -m src.demo`

无 OPENAI_API_KEY 时自动进入确定性 mock 模式，全流程离线可跑。

用法：
    python -m src.demo
    python -m src.demo --question "示例智造银行 2024 年资产质量如何？"
    python -m src.demo --auto                # 跳过人工确认交互
    python -m src.demo --engine native       # 强制使用自研降级引擎

层次与职责
----------
本模块属于「表现层 / CLI」，不含任何业务逻辑：它只做三件事——
1. 解析命令行参数；
2. 构造 `ResearchPipeline`（内部装配 RAG、工具、五个 Agent 与图引擎）并跑一次；
3. 把 `pipeline.run()` 返回的结果字典渲染成人类可读的分步报告。

对外关键函数
------------
* `main(argv)`            进程入口（`python -m src.demo` 与 `if __name__ == "__main__"` 都走它）
* `build_parser()`        argparse 解析器（`--question/--auto/--engine/--run-id`）
* `print_banner(...)`     运行前打印本次配置（引擎、模式、语料规模、工具清单）
* `render(result, question)`  把一次运行结果打印成五步过程报告 + 引用可追溯性自检 + 运行汇总
* `_citation_traceability(report, citations)`  报告内 `[n]` 与引用清单的比对（自检用）

主要输入输出
------------
输入：命令行参数（问题文本、引擎偏好、是否自动放行、自定义 run_id）。
输出：stdout 的人类可读报告（**不写业务文件**；轨迹文件由 orchestrator 写入 RUNS_DIR），
      另有进程退出码（见 `main`）。
**不依赖网络**：无 key 时 LLM 走 mock 大脑，因此整条演示链路离线即可复现。

被谁调用
--------
终端用户手动执行；`README`/CI 的冒烟步骤也可直接调用 `main`。
它依赖 `src.orchestrator.ResearchPipeline`（业务编排）与 `src.config`（配置），
自身不被 `src/` 内任何模块 import——CLI 是依赖图的叶子。
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from typing import Any, Dict, List

from .config import GRAPH_RECURSION_LIMIT, RUNS_DIR, runtime_config, use_mock_llm
from .orchestrator import ResearchPipeline, make_run_id
from .utils.console import ensure_utf8_console

# 默认问题刻意覆盖「盈利/偿债/现金流 + 风险」四类线索，便于一跑就能看到多维度结论、
# 交叉验证与风险规则命中，同时演示 mock 模式下的完整链路
DEFAULT_QUESTION = (
    "请分析示例科技股份有限公司 2024 年度的盈利能力、偿债能力和现金流质量，"
    "并结合应收账款、对外担保与股权质押情况提示主要风险。"
)

# 输出分隔线：主分隔用等号，步骤内用短横线，纯展示用，不参与任何逻辑判断
LINE = "=" * 78
SUB = "-" * 78


def _h(title: str) -> None:
    """打印一级标题（上下各一条等号分隔线）。

    参数：
        title: 标题文本，原样打印。
    返回：
        None。
    副作用/异常：
        只向 stdout 写若干行（前后各补一个空行以增强可读性）；不抛异常。
    """
    print("")
    print(LINE)
    print(title)
    print(LINE)


def _step(no: int, total: int, title: str) -> None:
    """打印步骤小标题（形如 `[2/5] RetrieverAgent ...`）。

    参数：
        no:    当前步骤序号（从 1 开始）。
        total: 总步骤数，仅用于展示，不参与校验。
        title: 步骤名称。
    返回：
        None。
    副作用/异常：
        只写 stdout（短横线 + 一行标题 + 短横线）；不抛异常。
    """
    print("")
    print(SUB)
    print(f"[{no}/{total}] {title}")
    print(SUB)


def _fmt(value: Any) -> str:
    """把数值格式化成便于阅读的字符串（仅在没有现成 display 字段时兜底）。

    参数：
        value: 任意值，通常为 float / int。
    返回：
        str —— float 保留 4 位小数并去掉末尾多余的 0 与小数点（如 1.5000 -> "1.5"）；
        int 加千分位；其它类型直接 `str(value)`。
    副作用/异常：
        纯函数；对 bool 会走 int 分支（Python 中 bool 是 int 子类），无异常。
    """
    if isinstance(value, float):
        return f"{value:,.4f}".rstrip("0").rstrip(".")
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def _citation_traceability(report: str, citations: List[Dict[str, Any]]) -> Dict[str, Any]:
    """检查报告里的每个 [n] 是否都能在引用清单中找到（引用可追溯率）。

    参数：
        report:    生成的简报全文，可能为 None/空串。
        citations: 引用清单，每项含 citation_no 字段。
    返回：
        Dict[str, Any] —— 键为 used（报告里出现的编号，升序去重）、known（清单编号，升序）、
        dangling（悬空的编号）、ok（是否无悬空）、rate（可追溯比例，保留 4 位小数）。
    副作用/异常：
        纯函数，不写文件；report 为空时 used 为空，rate 记为 1.0（无引用即无可追溯问题）。
    与评测的差异：
        `eval/run_eval.py` 的同名逻辑把「报告里一个引用都没有」判为不可追溯，
        此处只做自检展示，因此空引用按 1.0 处理。
    """
    used = sorted({int(m) for m in re.findall(r"\[(\d+)\]", report or "")})
    known = {int(c.get("citation_no", 0)) for c in citations}
    dangling = [n for n in used if n not in known]
    return {
        "used": used,
        "known": sorted(known),
        "dangling": dangling,
        "ok": not dangling,
        "rate": round(len([n for n in used if n in known]) / len(used), 4) if used else 1.0,
    }


def render(result: Dict[str, Any], question: str) -> None:
    """把一次运行的结果打印成人类可读的过程报告（不含顶部配置区，那部分在运行前打印）。

    参数：
        result:   `ResearchPipeline.run()` 的返回字典，读取 `state`（Blackboard 终态）、
                  `summary`（步数/耗时/计数/错误）、`llm_stats`、`run_id`、`trace_path`。
                  `state` 内会用到 plan / retrieval_meta / evidence / metrics / findings /
                  risk_report / citations / steps / report / needs_human / human_decision 等键。
        question: 研究问题；注：实际实现为函数体内**未使用**该参数（内容全部取自
                  `result["state"]`），保留它只是为了不改变既有调用签名。
    返回：
        None。
    副作用/异常：
        只向 stdout 打印（五步过程 + 简报正文 + 引用自检 + 运行汇总）；
        当 `state["report"]` 为空（例如人工确认环节被驳回）时提前 return，不打印简报区；
        各字段用 `.get(...)` 容错，缺键不会抛异常。
    """
    state = result["state"]
    plan = state.get("plan") or {}
    metrics = state.get("metrics") or {}
    findings = state.get("findings") or []
    evidence = state.get("evidence") or []
    risk = state.get("risk_report") or {}
    summary = result["summary"]

    # ---------------- 1. Planner ----------------
    _step(1, 5, "PlannerAgent —— 任务分解与路由")
    print(f"意图       : {plan.get('intent', '')}")
    print(f"识别公司   : {'、'.join(plan.get('companies') or [])}")
    print(f"分析年度   : {plan.get('year')}")
    print(f"分析维度   : {'、'.join(plan.get('targets') or [])}")
    print(f"路由决策   : {' -> '.join(plan.get('route') or [])}")
    print("子任务拆解 :")
    for task in plan.get("subtasks") or []:
        dep = f"  (依赖 {','.join(task.get('depends_on') or [])})" if task.get("depends_on") else ""
        print(f"  · {task.get('id')} [{task.get('agent')}] {task.get('goal')}{dep}")
    print("多路检索查询 :")
    for q in plan.get("retrieval_queries") or []:
        print(f"  · {q}")

    # ---------------- 2. Retriever ----------------
    _step(2, 5, "RetrieverAgent —— 多路检索（BM25 + 哈希向量 + 元数据加权重排）")
    meta = state.get("retrieval_meta") or {}
    print(f"执行查询数 : {len(meta.get('queries') or [])}    候选片段 : {meta.get('candidate_count')}    "
          f"保留证据 : {meta.get('selected_count')}")
    coverage = meta.get("coverage") or {}
    if coverage:
        print("维度覆盖   : " + "；".join(f"{k}={len(v)}条" for k, v in coverage.items()))
    print(f"{'#':<3}{'资料编号':<20}{'章节':<26}{'融合分':>8}  命中词")
    for idx, item in enumerate(evidence, 1):
        terms = "、".join((item.get("matched_terms") or [])[:5])
        print(f"{idx:<3}{str(item.get('source_id')):<20}{str(item.get('section_title'))[:24]:<26}"
              f"{float(item.get('score') or 0):>8.4f}  {terms}")
    missing = meta.get("missing_data") or []
    if missing:
        print(f"未覆盖维度 : {'、'.join(missing)}")

    # ---------------- 3. Analyst ----------------
    _step(3, 5, "AnalystAgent —— 指标计算与财务分析（数字全部来自工具）")
    print(f"{'比率':<18}{'数值':>12}   计算口径")
    for key, item in metrics.items():
        print(f"{str(item.get('label') or key):<18}{str(item.get('display') or _fmt(item.get('value'))):>12}   "
              f"{item.get('formula', '')}")
    print("")
    print(f"分析结论   : 共 {len(findings)} 条")
    for finding in findings:
        print(f"  [{finding.get('id')}] {finding.get('title')}")
        print(f"      陈述   : {finding.get('statement')}")
        print(f"      指标支撑: {'、'.join(finding.get('ratio_refs') or []) or '（无）'}")
        print(f"      证据支撑: {'、'.join(finding.get('evidence_ids') or []) or '（无）'}")
        for extra in finding.get("cross_checks") or []:
            print(f"      交叉验证: {extra}")

    # ---------------- 4. RiskChecker ----------------
    _step(4, 5, "RiskCheckerAgent —— 风险核查与反思循环（可打回 Analyst 重算）")
    steps = [s.get("agent") for s in (state.get("steps") or [])]
    # analyst 首次执行算第 0 轮重算，之后每多出现一次才算一轮反思重算，故减 1
    rounds = steps.count("analyst") - 1
    print(f"反思循环   : 实际重算 {max(rounds, 0)} 轮（上限 {state.get('max_revision_rounds')} 轮）")
    print(f"执行路径   : {' -> '.join(steps)}")
    history = risk.get("gate_history") or []
    if history:
        print("核查轮次留痕:")
        for item in history:
            print(f"  第{item.get('round')}轮  裁决={str(item.get('verdict')).upper():<9}"
                  f"风险等级={str(item.get('risk_level')).upper():<7}缺口={len(item.get('gaps') or [])} 项")
    gaps = risk.get("gaps") or []
    print(f"当前缺口   : {len(gaps)} 项" + ("（历史缺口见上方留痕）" if history and not gaps else ""))
    for gap in gaps:
        print(f"  · [{gap.get('code')}] {gap.get('problem')}")
        print(f"      要求补正: {gap.get('required_fix')}")
    print(f"裁决       : {str(state.get('risk_verdict', '')).upper()}"
          f"（整体风险等级 {str(state.get('risk_level', '')).upper()}）")
    if risk.get("verdict_override"):
        print("  注：模型裁决与确定性闸门不一致，已按确定性闸门执行（合规闸门不委托给模型）。")
    print(f"核查叙述   : {risk.get('narrative', '')}")
    risk_findings = risk.get("findings") or []
    if risk_findings:
        print(f"风险规则命中 {len(risk_findings)} 项：")
        for item in risk_findings:
            print(f"  [{str(item.get('level', '')).upper():<6}] {item.get('rule_id'):<12}"
                  f"{item.get('metric_display', ''):>10}  {item.get('title', '')}")

    # ---------------- 5. HITL + Writer ----------------
    _step(5, 5, "Human-in-the-loop 与 WriterAgent —— 人工确认 + 带引用的投研简报")
    if state.get("needs_human"):
        decision = state.get("human_decision") or {}
        print(f"人工确认   : 已触发（高风险 / 循环超限）")
        print(f"  升级原因 : {risk.get('escalation_reason', '')}")
        print(f"  人工决策 : {decision.get('decision')}（来源：{decision.get('source')}）")
        print(f"  决策说明 : {decision.get('reason')}")
    else:
        print("人工确认   : 未触发（结论证据链完整且风险等级未达 high）")

    report = state.get("report") or ""
    if not report:
        # 被人工驳回时流程仍然成功返回，只是没有简报可展示，这里如实提示而不是报错
        print("")
        print("!! 流程在人工确认环节被驳回，未产出简报。")
        return

    print("")
    print("---------------------------- 投研简报 ----------------------------")
    print("")
    print(report)

    trace = _citation_traceability(report, state.get("citations") or [])
    print("")
    print(SUB)
    print("引用可追溯性自检")
    print(SUB)
    print(f"报告中出现的引用编号 : {trace['used']}")
    print(f"引用清单中的编号     : {trace['known']}")
    print(f"悬空引用             : {trace['dangling'] if trace['dangling'] else '无'}")
    print(f"引用可追溯率         : {trace['rate'] * 100:.2f}%")

    _h("运行汇总")
    print(f"总耗时     : {summary['total_latency_ms']:.1f} ms")
    print(f"执行步数   : {summary['steps']}")
    print(f"证据 / 结论 / 引用 : {summary['evidence']} / {summary['findings']} / {summary['citations']}")
    print(f"LLM 调用   : {result['llm_stats']}")
    print(f"错误       : {summary['errors'] if summary['errors'] else '无'}")
    print(f"轨迹文件   : {result['trace_path']}")
    print(f"回放命令   : python -m src.replay {result['run_id']}")
    print("")


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。

    参数：无。
    返回：
        argparse.ArgumentParser —— 已注册四个参数：
        `--question/-q`（研究问题，默认 DEFAULT_QUESTION）、
        `--auto`（跳过人工确认交互，store_true）、
        `--engine`（auto|langgraph|native，默认 None 交给 orchestrator 走配置）、
        `--run-id`（自定义运行编号，默认由 make_run_id 生成）。
    副作用/异常：
        只构造对象，不读环境变量、不解析 sys.argv（由调用方传入 argv）；
        `--engine` 的 choices 会限制非法取值并在启动时报错退出。
    """
    parser = argparse.ArgumentParser(
        prog="python -m src.demo",
        description="金融投研多智能体系统演示（无 API Key 时自动进入 mock 模式）",
    )
    parser.add_argument("--question", "-q", default=DEFAULT_QUESTION, help="研究问题")
    parser.add_argument("--auto", action="store_true", help="跳过人工确认交互，自动放行")
    parser.add_argument("--engine", choices=["auto", "langgraph", "native"], default=None,
                        help="强制指定编排引擎（默认 auto：优先 LangGraph）")
    parser.add_argument("--run-id", default=None, help="自定义运行编号（默认自动生成）")
    return parser


def print_banner(pipeline: "ResearchPipeline", question: str, run_id: str, engine_pref: str) -> None:
    """运行前打印配置区，让人在交互式人工确认之前就知道这次跑的是什么。

    参数：
        pipeline:    已装配好的流水线；从这里读 config.max_revision_rounds、
                     documents / parents / children（语料规模）、fact_store（结构化事实）、
                     registry.names()（注册工具清单）。
        question:    研究问题文本，原样回显。
        run_id:      本次运行编号，用于后续 replay。
        engine_pref: 命令行传入的引擎偏好（None 或 "auto" 时展示为 auto 说明文案）。
    返回：
        None。
    副作用/异常：
        只写 stdout；不启动任何执行、不写文件，因此可以在人工确认**之前**安全调用。
    """
    _h("金融投研多智能体系统  ·  FinResearch-MAS")
    print(f"研究问题   : {question}")
    print(f"运行编号   : {run_id}")
    print(f"编排引擎   : {'auto（优先 LangGraph，失败自动降级自研引擎）' if engine_pref in (None, 'auto') else engine_pref}")
    print(f"生成方式   : {'确定性 mock 大脑（离线模式，无需 API Key）' if use_mock_llm() else 'OpenAI 兼容接口'}")
    print(f"反思循环上限: {pipeline.config.max_revision_rounds} 轮")
    print(f"文档/父块/子块: {len(pipeline.documents)} / {len(pipeline.parents)} / {len(pipeline.children)}")
    print(f"结构化事实 : {len(pipeline.fact_store)} 条（公司：{'、'.join(pipeline.fact_store.companies)}）")
    print(f"注册工具   : {'、'.join(pipeline.registry.names())}")
    print(f"图递归上限 : {GRAPH_RECURSION_LIMIT}")
    print(f"轨迹目录   : {RUNS_DIR}")


def main(argv: List[str] | None = None) -> int:
    """CLI 主流程：解析参数 -> 装配流水线 -> 运行 -> 渲染报告。

    参数：
        argv: 参数列表；None 时由 argparse 读取真实 `sys.argv`（便于测试注入）。
    返回：
        int —— 注：实际实现为**恒返回 0**；装配或运行阶段的问题由 orchestrator 记入
        state["errors"] 并以文本形式展示，不用退出码表达成败（真正的门禁在
        eval/run_eval.py 的退出码里）。`if __name__ == "__main__"` 会把它交给 sys.exit。
    副作用/异常：
        * 调用 `ensure_utf8_console()` 调整控制台编码，避免中文输出乱码；
        * 构造 ResearchPipeline（会加载语料并建索引）并写轨迹到 RUNS_DIR；
        * `--auto` 未给出且流程要求人工确认时会**阻塞等待 stdin 输入**；
        * 编码设置失败等底层异常不在此捕获，会直接向上抛出。
    """
    ensure_utf8_console()
    args = build_parser().parse_args(argv)

    config = runtime_config()
    print("")
    print("正在装载资料库并装配多智能体图……")
    started = time.perf_counter()
    pipeline = ResearchPipeline(auto=args.auto, engine=args.engine, config=config)
    run_id = args.run_id or make_run_id(args.question)

    print_banner(pipeline, args.question, run_id, args.engine)
    result = pipeline.run(args.question, run_id=run_id)

    print("")
    print(f"编排引擎（实际使用）: {result['engine']}"
          + ("（LangGraph StateGraph + MemorySaver checkpointer）" if result["engine"] == "langgraph"
             else "（自研兼容图引擎：条件边 + checkpointer）"))
    print(f"轨迹文件            : {result['trace_path']}")

    render(result, args.question)
    print(f"流程搭建 + 执行总耗时 {(time.perf_counter() - started) * 1000:.1f} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
