"""可观测：每一步一行 JSONL 的执行轨迹。

落盘位置：`runs/<run_id>.jsonl`
每行固定字段（题目要求的最小集合）：
    step            步序号（从 1 开始，全局单调递增）
    agent           执行者（planner / retriever / analyst / risk_checker / writer / human_review）
    input_digest    输入摘要（sha256 前 16 位）
    output_digest   输出摘要（sha256 前 16 位）
    latency_ms      本步耗时（毫秒）
    status          ok / error / skipped

额外附带（便于 replay 与排障，不影响固定字段）：
    ts, input_chars, output_chars, input_preview, output_preview, extra

设计要点：每写一行立即 flush，保证进程被强杀时轨迹依然可用；
preview 只截前 240 字符，既有可读性又不会让日志爆炸。

架构层次与职责：
    本模块是「可观测层」的地基：把每一步执行落成一行自包含的 JSONL。它不参与任何业务
    判断，只保证「谁、什么时候、拿什么当输入、产出什么、耗时多少」被如实记录——
    这是解题要求里的硬性可观测性，也是 replay 与 eval 的唯一数据来源。

对外关键对象：
    TraceRecorder   写轨迹（record() 追加一行，close() 追加汇总行）
    load_trace()    读轨迹（供 replay 与测试使用）

主要输入输出：
    输入：record(agent, input_obj, output_obj, latency_ms, status, extra) 的调用参数；
          input_obj / output_obj 可为任意对象，内部会被 digest + preview 压扁。
    输出：runs/<run_id>.jsonl，每行一个 JSON 对象；load_trace() 返回结构化结果
          {run_id, path, entries, summary, steps}。

被谁调用：
    * agents/base.py：BaseAgent.run() 是最大的写入方，每个 Agent 一步写一行（extra 带
      tool_calls 与业务留痕）；
    * orchestrator：human_review_node 手写一行 human_review；引擎降级时写一行
      engine_fallback；run() 末尾 recorder.close(summary) 写 run_summary；
    * replay.render() 与 tests/test_mock_llm.py 通过 load_trace() 读取轨迹；
    * state.snapshot() 产出的规模摘要最终也是经这里的汇总行落盘。

字段契约（题目要求的最小集合，测试会断言）：
    step / agent / input_digest / output_digest / latency_ms / status 六个字段恒存在；
    digest 固定 16 位（utils.digest.digest 默认长度），status ∈ {ok, error, skipped}，
    step 含 run_summary 行在内从 1 连续递增——tests/test_mock_llm.py 断言字段与步号，
    tests/test_state_flow.py 断言「JSONL 行数 == 状态里的 steps 数 + 1（汇总行）」。

注：实际实现为「step 连续编号」只保证在单个 TraceRecorder 实例内成立；每个 run 各建一个
实例，且 __init__ 会清空同名文件，所以一次 run 的轨迹一定是自包含的一份完整记录。
"""

from __future__ import annotations

import json
import re
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from .config import RUNS_DIR
from .utils.digest import digest

__all__ = ["TraceRecorder", "load_trace"]

_PREVIEW_CHARS = 240
# 为什么是 240：足够看清「这一步输入/输出大概是什么」，又能保证单行 JSONL 不因长简报
# 而膨胀到几百 KB；正文全文另有 digest 与 *_chars 字段可核对规模，不需要落全文。


def _preview(value: Any) -> str:
    """生成可读的一行预览（只压空白，不动标点，保证中文可读）。

    参数：
        value  任意对象；str 直接使用，其余走 json.dumps（不可序列化时退回 repr）。

    返回：
        压掉所有连续空白符后的单行字符串，最长 _PREVIEW_CHARS 字符（超长直接截断，
        不加省略号，长度由 TraceRecorder 的 *_chars 字段另行给出）。

    副作用 / 异常：
        纯函数，无副作用；json.dumps 抛异常时走 repr 兜底，因此不会向外抛异常。
    """
    if value is None:
        return ""
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
        except Exception:
            text = repr(value)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:_PREVIEW_CHARS]


class TraceRecorder:
    """把每一步执行写成一行 JSONL。

    职责：为单次 run 提供「追加一行」与「收尾一行」两个动作，并维护全局步号。
    写盘即真相：这里不做缓冲，每行写完立刻 flush，进程被 kill 也不丢已完成的步骤。

    关键属性：
        run_id     运行编号，决定文件名 runs/<run_id>.jsonl
        runs_dir   轨迹目录（默认 config.RUNS_DIR），构造时会 mkdir
        path       本次轨迹文件完整路径
        echo       是否把每行摘要同时打印到 stdout（CLI 排障用）
        _step      已写行数（同时也是下一次的步号），受 _lock 保护
        _lock      线程锁：保证并发记录时步号不重号、JSON 行不被交错写坏
        _started   实例创建时刻，用于 close() 计算本轮总耗时

    状态流转：__init__ 清空同名文件 -> 每个执行步骤 record() 追加一行 ->
    末尾 close(summary) 追加 run_summary 行；此后实例不再被使用。
    """

    def __init__(self, run_id: str, runs_dir: Optional[Path] = None, echo: bool = False) -> None:
        """创建记录器并准备轨迹文件。

        参数：
            run_id    运行编号（文件名前缀，通常来自 orchestrator.make_run_id()）。
            runs_dir  轨迹目录；None 表示用 config.RUNS_DIR。
            echo      为 True 时每写一行顺带打印一行摘要到 stdout。

        返回：无。
        副作用：
            创建 runs_dir（已存在则忽略），并**清空**同名 <run_id>.jsonl——
            这是有意的：同名 run_id 重跑时，replay 应该看到一次完整运行，而不是两次的拼接。
        """
        self.run_id = run_id
        self.runs_dir = Path(runs_dir) if runs_dir is not None else RUNS_DIR
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.runs_dir / f"{run_id}.jsonl"
        self.echo = echo
        self._step = 0
        self._lock = threading.Lock()
        self._started = time.perf_counter()
        # 新 run 覆盖同名文件，保证 replay 看到的是一次完整运行
        self.path.write_text("", encoding="utf-8")

    # ------------------------------------------------------------------
    @property
    def step_count(self) -> int:
        """已写入的轨迹行数（等于最后一步的 step 值）；只读，无副作用。"""
        return self._step

    @property
    def elapsed_ms(self) -> float:
        """从构造记录器到此刻的毫秒数；close() 用它作为本轮总耗时。只读，无副作用。"""
        return (time.perf_counter() - self._started) * 1000.0

    def record(
        self,
        *,
        agent: str,
        input_obj: Any,
        output_obj: Any,
        latency_ms: float,
        status: str = "ok",
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """写入一行轨迹，返回该行内容。

        参数（全部为关键字参数，避免调用方写错顺序）：
            agent       执行者名，写入 agent 字段（replay 靠它选展示样式）。
            input_obj   本步输入对象；被 digest 成 16 位指纹并生成预览。
            output_obj  本步输出对象；同上。
            latency_ms  本步耗时（毫秒），落盘前 round 到 3 位小数。
            status      ok / error / skipped，缺省 ok。
            extra       附加结构化信息（tool_calls、业务留痕、汇总字段等），缺省 {}。

        返回：
            刚写入的那一行 dict（含 step / ts / agent / input_digest / output_digest /
            latency_ms / status / input_chars / output_chars / input_preview /
            output_preview / extra），调用方据此把步号与耗时回填到共享状态。
            注：实际实现为 *_chars 对 str 输入取原始长度、对非 str 取预览长度。

        副作用：
            自增步号、向轨迹文件追加一行并 flush、echo=True 时打印一行摘要。
        异常：
            文件系统错误会向上抛（轨迹写不进去属于严重问题，不应被吞掉）。
        """
        # 为什么要加锁：图引擎目前是单线程执行，但 record 可能被信号/回调并发触发，
        # 加锁可保证 _step 不重号、单行 JSON 不被两条记录交叉写坏。
        with self._lock:
            self._step += 1
            entry = {
                "step": self._step,
                "ts": datetime.now().isoformat(timespec="milliseconds"),
                "agent": agent,
                "input_digest": digest(input_obj),
                "output_digest": digest(output_obj),
                "latency_ms": round(float(latency_ms), 3),
                "status": status,
                "input_chars": len(_preview(input_obj)) if not isinstance(input_obj, str) else len(input_obj),
                "output_chars": len(output_obj) if isinstance(output_obj, str) else len(_preview(output_obj)),
                "input_preview": _preview(input_obj),
                "output_preview": _preview(output_obj),
                "extra": extra or {},
            }
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
                # 为什么每行都 flush：演示/评测里进程可能被直接杀掉，只有立刻落盘才能保证
                # 「已完成的步骤」一定能在 replay 里看到，代价只是每步一次 syscall。
                fh.flush()
            if self.echo:
                print(f"[trace] step={entry['step']} agent={agent} status={status} {entry['latency_ms']}ms")
            return entry

    def close(self, summary: Dict[str, Any]) -> Dict[str, Any]:
        """写入结束汇总行。

        参数：
            summary  运行汇总（orchestrator._summary() 的产物），既当作 output_obj
                     算 digest/预览，也整份塞进 extra 供 load_trace() 拍平读取。

        返回：
            汇总行 dict（agent="run_summary"，step 为最后一行）。

        副作用：
            与 record() 相同：追加一行轨迹并 flush；close 之后本实例不再写盘。
            注：实际实现为 status 取 summary["status"]（"ok" / "no_report"），
            而不是固定的 "ok"；replay 会照原样展示。
        """
        return self.record(
            agent="run_summary",
            input_obj={"run_id": self.run_id},
            output_obj=summary,
            latency_ms=self.elapsed_ms,
            status=summary.get("status", "ok"),
            extra=summary,
        )


def load_trace(run_id: str, runs_dir: Optional[Path] = None) -> Dict[str, Any]:
    """读取一次运行的轨迹。返回 {run_id, path, entries, summary}。

    参数：
        run_id    运行编号；也接受带 .jsonl 后缀或只给片段（走下面的模糊匹配分支）。
        runs_dir  轨迹目录；None 表示用 config.RUNS_DIR。

    返回：
        {
          "run_id":  实际命中文件的 stem（模糊匹配时可能与你传入的不同）,
          "path":    实际读取的文件路径字符串,
          "entries": 全部行（含 run_summary 行）的 dict 列表,
          "summary": run_summary 行的字段拍平结果；没有汇总行时为 None,
          "steps":   entries 里剔除 run_summary 之后的部分（即真正的执行步骤）,
        }

    副作用 / 异常：
        只读文件，不写盘；目标文件不存在且模糊匹配也找不到时抛 FileNotFoundError
        （replay.render() 会捕获它并打印候选列表）。注：实际实现为坏行（JSON 解析失败）
        直接跳过而不是抛异常，也不再校验字段完整性——轨迹是「尽量可读」而不是强 schema。

    为什么要模糊匹配：手敲 run_id 时很容易多写/漏写后缀，或者只记住末尾的指纹段，
    这里用 *run_id*.jsonl 兜底并取排序后最后一个（即时间最新的那个）。
    """
    directory = Path(runs_dir) if runs_dir is not None else RUNS_DIR
    path = directory / f"{run_id}.jsonl"
    if not path.is_file():
        # 允许传入带 .jsonl 后缀或部分匹配
        candidates = sorted(directory.glob(f"*{run_id}*.jsonl")) if directory.is_dir() else []
        if not candidates:
            raise FileNotFoundError(f"未找到运行轨迹：{path}")
        path = candidates[-1]

    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            # 为什么跳过而不是报错：进程被强杀时最后一行可能是写了一半的 JSON，
            # 此时前面的步骤依然是可信的，能看多少看多少比整份回放失败更有用。
            continue
    summary_entry = next((e for e in reversed(entries) if e.get("agent") == "run_summary"), None)
    # 汇总行的业务字段挂在 extra 里，这里拍平，方便调用方直接 summary["question"] 取值
    summary = None
    if summary_entry is not None:
        summary = {**summary_entry, **(summary_entry.get("extra") or {})}
    return {
        "run_id": path.stem,
        "path": str(path),
        "entries": entries,
        "summary": summary,
        # steps 单独给一份：调用方 99% 的场景是「列出执行步骤」，而 run_summary 是元数据行，
        # 混在里面会让步数统计与 replay.get 的循环多一层判断。
        "steps": [e for e in entries if e.get("agent") != "run_summary"],
    }
