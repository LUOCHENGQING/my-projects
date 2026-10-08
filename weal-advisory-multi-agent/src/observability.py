"""可观测性：每步 trace 落盘 + 回放。

所处层次
--------
本模块属于 **基础设施层**（只依赖 `utils.digest` 与 `utils.now_iso`，import 不到
任何业务模块），为流水线提供"可回放的执行痕迹"：`pipeline.run_pipeline` 用
`Tracer(run_id=new_run_id("advisory"), runs_dir=...)` 创建并注入各 Agent，Agent 经
`agents.base.BaseAgent.trace` 间接调用 `Tracer.step`；随后 `replay.py` 读取落盘文件
回放，`tests/test_observability.py` 与 `tests/test_engine_parity.py` 也直接使用本模块。

数据格式
--------
`runs/<run_id>.jsonl` 每行一个步骤（JSON），字段固定为：
`step / agent / node / input_digest / output_digest / latency_ms / status / tool_calls`
（另附 timestamp 与可选的 extra，便于排查）。`TRACE_FIELDS` 是这几个字段的对外
契约，测试据此断言"每条记录都是它的超集"。

对外暴露的关键对象
------------------
- 常量：`RUNS_DIR`（默认 runs 目录）、`TRACE_FIELDS`（固定字段名及打印顺序）
- 采集：`Tracer`（逐步落盘）、`new_run_id`（生成可排序的 run 标识）
- 读取与展示：`read_trace`、`list_runs`、`format_trace_table`、`trace_summary`
- 私有：`_ellipsis`（表格列截断）

主要输入输出
------------
输入为各步骤的 Agent 名、节点名与输入/输出载荷（任意对象，落盘时只保留
`digest` 摘要）；输出为追加写入的 JSONL 行、内存中的 events 列表，以及统计字典。
`read_trace` 在 trace 文件缺失时抛 `FileNotFoundError`，其余函数不抛业务异常。

`python -m src.replay <run_id>` 读取该文件并打印步骤表。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .utils import digest, now_iso

#: runs 目录（相对仓库根定位）
#: 由 <repo>/src/observability.py 上推两级得到 <repo>/runs，不受运行时 cwd 影响
RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"

#: trace 行的固定字段（顺序即打印顺序）
#: 对外契约：每条落盘记录必须至少包含这些键；实际记录还含 `timestamp`，
#: 并在传入 extra 时额外含 `extra`（见 Tracer.step）
TRACE_FIELDS: tuple[str, ...] = (
    "step",
    "agent",
    "node",
    "input_digest",
    "output_digest",
    "latency_ms",
    "status",
    "tool_calls",
)


def new_run_id(prefix: str = "run") -> str:
    """生成一个按时间可排序的 run_id。

    参数:
        prefix: 前缀，默认 `"run"`；流水线调用时传 `"advisory"`。

    返回:
        形如 `f"{prefix}-%Y%m%d-%H%M%S-{毫秒:03d}"` 的字符串，例如
        `advisory-20261003-010203-456`。

    说明:
        时间部分前置，因此按字符串排序即时间先后；毫秒部分取
        `int(time.time() * 1000) % 1000`，同一毫秒内重复调用可能得到相同值
        （测试只要求 5 次调用"足够唯一"，不保证绝对不重复）。
    """
    return f"{prefix}-{time.strftime('%Y%m%d-%H%M%S')}-{int(time.time() * 1000) % 1000:03d}"


@dataclass
class Tracer:
    """逐步记录流水线执行痕迹。

    职责:
        按调用顺序为每个执行步骤生成一条 trace（Agent 名、节点、输入/输出摘要、
        耗时、状态、工具调用），追加写入 `runs/<run_id>.jsonl`，供
        `python -m src.replay <run_id>` 回放与双引擎路径比对。

    关键属性:
        run_id: 运行标识，同时决定 trace 文件名。
        runs_dir: trace 输出目录；构造时为 None 则在 `__post_init__` 回落到 `RUNS_DIR`。
        enabled: 是否落盘；为 False 时记录只留在内存 `events` 中（单测用）。
        events: 内存中的全部 trace 记录（dict 列表，与落盘内容一致）。
        _step: 步数计数器，`__post_init__` 初始化为 0，每次 `step()` 自增 1，
            是 trace 行 `step` 字段的唯一来源。

    状态流转:
        构造（`__post_init__`：runs_dir 兜底、_step=0）→ 反复 `step()`
        （_step+1、events 追加、enabled 时落盘）→ 只读查询（`step_count` /
        `total_latency_ms` / `agent_sequence` / `summary`）。本类无重置或删除接口，
        步数单调递增。

    被谁使用:
        由 `pipeline.run_pipeline` 创建并注入 Agent；`agents.base.BaseAgent.trace`
        调用 `step`；`replay.py` 读取落盘结果；测试直接构造。
    """

    run_id: str
    runs_dir: Path | None = None
    enabled: bool = True
    events: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        """初始化派生状态：目录兜底为 `RUNS_DIR`，步数计数器归零。

        副作用:
            把 `runs_dir` 就地转为 `Path`（未提供时用 `RUNS_DIR`），并新增实例
            属性 `_step = 0`（未在类字段中声明）。
        """
        self.runs_dir = Path(self.runs_dir) if self.runs_dir is not None else RUNS_DIR
        self._step = 0

    # ------------------------------------------------------------------
    @property
    def path(self) -> Path:
        """trace 文件路径。

        返回:
            `runs_dir / f"{run_id}.jsonl"`；此时并不校验该文件/目录是否存在。
        """
        return self.runs_dir / f"{self.run_id}.jsonl"

    def step(
        self,
        agent: str,
        *,
        node: str = "",
        input_payload: Any = None,
        output_payload: Any = None,
        status: str = "ok",
        tool_calls: Sequence[str] = (),
        latency_ms: float | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """记录一步执行，返回该条 trace（同时写入 jsonl）。

        参数:
            agent: 执行该步的 Agent 名（`agent_sequence` 用它比对双引擎执行路径）。
            node: 节点名；留空时回落为 `agent`。
            input_payload / output_payload: 该步输入/输出对象，落盘只保存
                `digest(...)` 摘要；为 None 时摘要字段记为空字符串。
            status: 该步状态，默认 `"ok"`。
            tool_calls: 该步调用的工具名序列，落盘为 list。
            latency_ms: 耗时（毫秒）；None 记 0.0，否则 round 到 3 位小数。
            extra: 附加诊断字段；为真值（非空）时以 `extra` 键并入记录。

        返回:
            本次写入的 trace 记录 dict，含 `step`、`timestamp` 与上述字段。

        副作用:
            1) `_step` 自增 1；2) 追加进 `self.events`；3) `enabled` 为真时追加写
            `self.path`（必要时创建父目录）。写盘 IO 错误会原样向上抛出。
        """
        self._step += 1
        record: dict[str, Any] = {
            "step": self._step,
            "agent": agent,
            # 节点名缺省回落为 Agent 名，保证 node 列永不为空
            "node": node or agent,
            # 注意：None 表示"本步无载荷"，记为 ""；若传入空字典则会得到
            # digest({}) 的真实摘要，两者语义不同
            "input_digest": digest(input_payload) if input_payload is not None else "",
            "output_digest": digest(output_payload) if output_payload is not None else "",
            "latency_ms": round(float(latency_ms), 3) if latency_ms is not None else 0.0,
            "status": status,
            "tool_calls": list(tool_calls),
            "timestamp": now_iso(),
        }
        if extra:
            record["extra"] = dict(extra)
        self.events.append(record)
        if self.enabled:
            self._write(record)
        return record

    def _write(self, record: Mapping[str, Any]) -> None:
        """把一条 trace 追加写入 jsonl（自动创建父目录）。

        参数:
            record: 待写入的 trace 记录。

        副作用:
            以 UTF-8 追加模式写 `self.path`，每条一行；`ensure_ascii=False` 保证
            中文原样落盘、`ensure_ascii` 关闭后仍是一行一个 JSON。父目录不存在时
            先 `mkdir(parents=True, exist_ok=True)`。仅由 `Tracer.step` 在
            `enabled` 为真时调用。
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------
    def step_count(self) -> int:
        """已记录的步骤数。

        返回:
            内部计数器 `_step`（与 `events` 长度一致，除非外部直接改过 events）。
        """
        return self._step

    def total_latency_ms(self) -> float:
        """累计耗时（毫秒）。

        返回:
            `events` 中 `latency_ms` 之和，round 到 3 位小数；无记录时为 0.0。
        """
        return round(sum(float(item.get("latency_ms", 0.0)) for item in self.events), 3)

    def agent_sequence(self) -> list[str]:
        """按顺序返回参与执行的 Agent 名（用于双引擎路径一致性比对）。

        返回:
            `events` 中 `agent` 字段组成的列表，保留重复与先后顺序。
        """
        return [str(item["agent"]) for item in self.events]

    def summary(self) -> dict[str, Any]:
        """trace 摘要。

        返回:
            `{"run_id", "steps", "total_latency_ms", "agents", "statuses"}`：
            `agents` 用 `dict.fromkeys` 保序去重，`statuses` 为排序后的去重集合。
        """
        return {
            "run_id": self.run_id,
            "steps": self._step,
            "total_latency_ms": self.total_latency_ms(),
            "agents": list(dict.fromkeys(self.agent_sequence())),
            "statuses": sorted({str(item["status"]) for item in self.events}),
        }


def read_trace(run_id: str, runs_dir: Path | str | None = None) -> list[dict[str, Any]]:
    """读取一个 run 的全部 trace 行。

    参数:
        run_id: 运行标识（不含 `.jsonl` 后缀）。
        runs_dir: trace 目录；None 表示用 `RUNS_DIR`。

    返回:
        按文件顺序排列的记录列表；空行被跳过，字段不做校验。

    异常:
        trace 文件不存在时抛 `FileNotFoundError`，消息里带完整路径。
    """
    base = Path(runs_dir) if runs_dir is not None else RUNS_DIR
    path = base / f"{run_id}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"未找到 trace 文件：{path}")
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def list_runs(runs_dir: Path | str | None = None) -> list[str]:
    """列出全部 run_id（按文件名排序）。

    参数:
        runs_dir: trace 目录；None 表示用 `RUNS_DIR`。

    返回:
        目录下 `*.jsonl` 的文件名（去后缀）列表；目录不存在时返回空列表，
        因此本函数不会抛 `FileNotFoundError`。
    """
    base = Path(runs_dir) if runs_dir is not None else RUNS_DIR
    if not base.exists():
        return []
    return sorted(path.stem for path in base.glob("*.jsonl"))


def _ellipsis(text: str, width: int) -> str:
    """按显示宽度截断（超长时以省略号结尾）。

    参数:
        text: 待截断文本。
        width: 目标显示宽度（字符数）。

    返回:
        `len(text) <= width` 时原样返回；否则返回 `text[: width - 1] + "…"`，
        结果长度恰为 `width`（省略号按单字符计）。
    """
    return text if len(text) <= width else text[: width - 1] + "…"


def format_trace_table(records: Iterable[Mapping[str, Any]]) -> str:
    """把 trace 渲染成等宽表格（CLI 回放用）。

    参数:
        records: trace 记录序列（通常来自 `read_trace`）。

    返回:
        多行字符串：表头 + 分隔线 + 每条记录一行。列宽与截断宽度由 `_ellipsis`
        控制（截断宽度比列宽小 1，留出列间空隙）；末列只显示
        `input_digest[:8]` → `output_digest[:8]` 的短前缀。

    说明:
        缺失字段有默认值（step→0、latency_ms→0.0、其余→空串），因此对不完整的
        记录也能渲染，不会因缺键报错。
    """
    header = (
        f"{'step':>4}  {'agent':<26}{'node':<14}{'latency(ms)':>12}  "
        f"{'status':<7}{'tool_calls':<44}{'input→output'}"
    )
    lines = [header, "-" * len(header)]
    for record in records:
        tools = _ellipsis(",".join(record.get("tool_calls") or []) or "-", 43)
        lines.append(
            f"{record.get('step', 0):>4}  {_ellipsis(str(record.get('agent', '')), 25):<26}"
            f"{_ellipsis(str(record.get('node', '')), 13):<14}"
            f"{float(record.get('latency_ms', 0.0)):>12.2f}  "
            f"{str(record.get('status', '')):<7}{tools:<44}"
            f"{record.get('input_digest', '')[:8]}→{record.get('output_digest', '')[:8]}"
        )
    return "\n".join(lines)


def trace_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """对一批 trace 行做统计汇总。

    参数:
        records: trace 记录序列。

    返回:
        `{"steps", "total_latency_ms", "agents", "statuses"}`；`records` 非空时
        额外带 `"run_id"` 键。空输入返回
        `{"steps": 0, "total_latency_ms": 0.0, "agents": [], "statuses": []}`
        （该分支不含 `run_id` 键，调用方需自行用 `.get`）。

    注：实际实现中，`Tracer.step` 写出的记录**不含** `run_id` 字段（见
    `TRACE_FIELDS` 与 `Tracer.step` 的字段表），因此正常流程下返回的 `run_id`
    恒为 None；只有外部构造的记录恰好带该字段时才有值。
    """
    if not records:
        return {"steps": 0, "total_latency_ms": 0.0, "agents": [], "statuses": []}
    return {
        "steps": len(records),
        "total_latency_ms": round(sum(float(item.get("latency_ms", 0.0)) for item in records), 3),
        "agents": list(dict.fromkeys(str(item.get("agent", "")) for item in records)),
        "statuses": sorted({str(item.get("status", "")) for item in records}),
        "run_id": str(records[0].get("run_id", "")) or None,
    }
