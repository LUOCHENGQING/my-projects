"""每一步一行 JSONL 的可观测记录。

RAG 系统的线上问题几乎都长这样：「同一个问题，昨天答得对，今天答错了」。
如果没有逐步轨迹，能查的只有最终答案，而答案错了可能是路由变了、召回少了、
重排顺序变了、缓存返回了旧结果——**每一种原因的修法完全不同**。

因此这里把每一步落成一行 JSONL：

    step / agent / input_digest / output_digest / latency_ms / status / extra

用 digest（短哈希 + 截断预览）而不是全文，是因为轨迹文件会被长期保留，
全文落盘既占空间又会把敏感数据（即使已脱敏）反复复制。

`read_trace()` 让一次运行可以被完整回放：召回了几条、哪条被去重、重排前后名次怎么变、
引用校验剔除了哪几个编号——面试时被追问"你怎么定位一个答错的 case"，
这就是答案。

字段口径（一行 JSON = 一条 `TraceRecord`，键名与 `TraceRecord.to_dict()` 完全一致）
--------------------------------------------------------------------------------
    ts              Unix 秒，写入时刻（保留 3 位小数）
    run_id          本次运行编号；与轨迹文件名 `<runs_dir>/<run_id>.jsonl` 同名
    step            步名。`RAGEngine.ask()` 依次写：cache_lookup → faq_lookup →
                    entity_gate → recall → rerank → generate → validate → done
    agent           产出该步的组件名：cache / faq / guard / retriever / reranker /
                    generator / validator / engine
    status          默认 "ok"；主体闸门拒答写 "refused"，忠实度校验未通过写 "warn"
    latency_ms      该步耗时（毫秒，3 位小数）。**调用方不传就是 0.0**：
                    注：实际实现为 `RAGEngine.ask()` 里 rerank 这一行就传 latency_ms=0.0，
                    因为重排耗时已并入同一次 retrieval_ms，不重复计时
    input_digest    输入摘要（`utils.text.digest`：短哈希 + 截断预览），不是全文
    output_digest   输出摘要，同上
    extra           可选字段，只在非空时写盘。逐步补充：cache_hit / mode / refused /
                    路由 route / queries 条数 / filter_expr / removed_duplicates /
                    top 命中 / dangling / faithful / support_rate / unsupported_numbers 等

被谁调用
--------
写入：`src/engine.py`（`RAGEngine.ask()` 每步一次 `recorder.step(...)`，
      `enable_trace=False` 时只累积在内存不落盘）；
读取：`read_trace()` / `summarize_trace()`（tests/test_engine_api.py、评测与人工排障）。
输入：步名 + 摘要对象；输出：`runs/<run_id>.jsonl` 追加一行，同时内存保留全量 records。
异常：`read_trace()` 对不存在的 run 抛 FileNotFoundError；落盘失败被 `_append()` 静默吞掉
      （只读文件系统 / 磁盘满不应拖垮问答主流程）。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .config import RUNS_DIR
from .utils.jsonable import to_plain
from .utils.text import digest

__all__ = ["TraceRecord", "TraceRecorder", "read_trace", "summarize_trace", "new_run_id"]


def new_run_id(prefix: str = "run") -> str:
    """生成可排序、可读的运行编号：run-20261003-012345-ab12cd。

    参数：prefix 编号前缀，`RAGEngine.ask()` 传 "ask"，默认 "run"。
    返回：str——`<prefix>-<YYYYmmdd-HHMMSS>-<6 位十六进制>`；
          时间戳保证可排序，后 6 位取自毫秒时间戳的低 24 位，用于降低同一秒内的碰撞。
    副作用/异常：无（只读系统时钟；不保证全局唯一，只是极低概率重复）。
    """
    stamp = time.strftime("%Y%m%d-%H%M%S")
    suffix = format(int(time.time() * 1000) % 0xFFFFFF, "06x")
    return f"{prefix}-{stamp}-{suffix}"


@dataclass
class TraceRecord:
    """一行轨迹（字段口径见模块 docstring）。

    关键属性：
        step / agent / run_id        步名、产出组件、所属运行（前三个必填，其余有默认值）
        status                       默认 "ok"，异常语义由调用方给（refused / warn）
        latency_ms                   该步耗时（毫秒）
        input_digest / output_digest digest 摘要，**不落全文**
        extra                        该步的补充字段（route、命中数等）
        ts                           记录时刻（Unix 秒，默认构造时取当前时间）

    状态流转：构造后字段不再变化（不是冻结 dataclass，但调用方只读）；
              `to_dict()` 是唯一的出站形态，负责把 ts / latency_ms 四舍五入到 3 位小数。
    """

    step: str
    agent: str
    run_id: str
    status: str = "ok"
    latency_ms: float = 0.0
    input_digest: str = ""
    output_digest: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """导出单行 JSON 的 dict 形态（落盘与 `read_trace()` 读回的结构都以此为准）。

        返回：dict，键 ts / run_id / step / agent / status / latency_ms /
              input_digest / output_digest；`extra` 为空时**整键省略**。
        副作用/异常：无；`extra` 里的 numpy 标量经 `to_plain()` 净化，避免序列化炸掉。
        """
        payload: Dict[str, Any] = {
            "ts": round(self.ts, 3),
            "run_id": self.run_id,
            "step": self.step,
            "agent": self.agent,
            "status": self.status,
            "latency_ms": round(self.latency_ms, 3),
            "input_digest": self.input_digest,
            "output_digest": self.output_digest,
        }
        if self.extra:
            payload["extra"] = to_plain(self.extra)
        return payload


class TraceRecorder:
    """把每一步追加写到 runs/<run_id>.jsonl。

    关键属性：
        run_id     本次运行编号（外部传入或用 `new_run_id()` 现生成）
        runs_dir   轨迹目录（默认取 config.RUNS_DIR）
        enabled    True 时每步落盘；False 时只留内存（测试与"关掉可观测"场景）
        records    内存里的全部 TraceRecord，供 `to_dict()` / `summary()` 直接汇总
        _started   perf_counter 起点，用于 `elapsed_ms`

    状态流转：`step()` 追加一条记录（内存 + 可选落盘）→ `to_dict()` / `summary()`
              汇总；**只增不改**，没有删除或重写轨迹的接口（轨迹即审计，不允许事后修饰）。
    副作用：在 runs_dir 下追加写 `<run_id>.jsonl`；写入失败静默（见 `_append()`）。
    """

    def __init__(self, run_id: Optional[str] = None, runs_dir: Optional[Path] = None, enabled: bool = True) -> None:
        """初始化记录器。

        参数：run_id 运行编号，None 时用 `new_run_id()` 生成；
              runs_dir 轨迹目录，None 时用 `config.RUNS_DIR`；
              enabled 是否落盘，False 时 `step()` 只写内存。
        返回：None。副作用：无（目录在第一次 `_append()` 时才创建）。
        异常：无（runs_dir 非法路径要到落盘时才暴露，且被静默处理）。
        """
        self.run_id = run_id or new_run_id()
        self.runs_dir = Path(runs_dir) if runs_dir is not None else RUNS_DIR
        self.enabled = enabled
        self.records: List[TraceRecord] = []
        self._started = time.perf_counter()

    # ------------------------------------------------------------------
    @property
    def path(self) -> Path:
        """本次运行的轨迹文件路径：`<runs_dir>/<run_id>.jsonl`。"""
        return self.runs_dir / f"{self.run_id}.jsonl"

    @property
    def elapsed_ms(self) -> float:
        """从构造记录器至今的毫秒数（perf_counter 计时，不受系统时钟调整影响）。"""
        return (time.perf_counter() - self._started) * 1000.0

    def step(
        self,
        step: str,
        agent: str,
        input_obj: Any = "",
        output_obj: Any = "",
        status: str = "ok",
        latency_ms: float = 0.0,
        **extra: Any,
    ) -> TraceRecord:
        """记录一步。input/output 会被压成 digest，避免轨迹里出现全文。

        参数：step 步名（cache_lookup / recall / generate …，见模块 docstring）；
              agent 产出该步的组件名；input_obj / output_obj 任意可序列化对象，
              统一经 `_as_text()` + `digest()` 变成摘要；
              status 状态语义，默认 "ok"（拒答用 "refused"、校验告警用 "warn"）；
              latency_ms 该步耗时，调用方不传即 0.0；
              **extra 透传为该行的 extra 字段（键名即字段名，不做转换）。
        返回：刚生成的 TraceRecord（调用方可继续读取，但不应修改）。
        副作用：追加进 `self.records`；`enabled` 为 True 时再向 JSONL 追一行。
        异常：无——`_as_text()` 兜住不可序列化对象，`_append()` 兜住落盘失败。
        """
        record = TraceRecord(
            step=step,
            agent=agent,
            run_id=self.run_id,
            status=status,
            latency_ms=latency_ms,
            input_digest=digest(_as_text(input_obj)),
            output_digest=digest(_as_text(output_obj)),
            extra=dict(extra),
        )
        self.records.append(record)
        if self.enabled:
            self._append(record)
        return record

    def _append(self, record: TraceRecord) -> None:
        """把一条记录以「一行 JSON」追加到轨迹文件（先建目录再开追加模式）。

        参数：record 待写入的记录。返回：None。副作用：创建 runs_dir、追加写文件。
        异常：**全部吞掉**——只读文件系统 / 磁盘满 / 并发重命名都不应让一次问答失败，
              此时内存里的 `records` 仍然完整，`to_dict()` 依旧可用。
        """
        try:
            self.runs_dir.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
        except Exception:
            # 落盘失败不应影响主流程（只读文件系统 / 磁盘满等），内存记录仍在
            pass

    def to_dict(self) -> Dict[str, Any]:
        """把本次运行的全部记录导成 dict（含 run_id、文件路径、逐步记录、总耗时）。

        返回：可序列化 dict：run_id / path（字符串路径）/ steps（`TraceRecord.to_dict()` 列表）/
              elapsed_ms（记录器存活时长，不是各步 latency 之和）。
        副作用/异常：无。
        """
        return {
            "run_id": self.run_id,
            "path": str(self.path),
            "steps": [r.to_dict() for r in self.records],
            "elapsed_ms": round(self.elapsed_ms, 3),
        }

    def summary(self) -> Dict[str, Any]:
        """按步汇总本次运行（等价于对自己的 records 调 `summarize_trace()`）。

        返回：`summarize_trace()` 的结构：steps 总数 / total_latency_ms / by_step / errors。
        副作用/异常：无。用于不读文件、直接看当前进程内这一次运行画像。
        """
        return summarize_trace([r.to_dict() for r in self.records])


def _as_text(obj: Any) -> str:
    """把任意对象转成用于算 digest 的文本。

    参数：obj 任意对象（None / str / 可 JSON 化的对象 / 其它）。
    返回：str——None → ""；str 原样返回；其余先 `to_plain()` 再 `json.dumps`，
          失败则退回 `str(obj)`。
    副作用/异常：无（内部异常被捕获）；返回空串时 `digest("")` 也是稳定的。
    """
    if obj is None:
        return ""
    if isinstance(obj, str):
        return obj
    try:
        return json.dumps(to_plain(obj), ensure_ascii=False)
    except Exception:  # noqa: BLE001
        return str(obj)


def read_trace(run_id: str, runs_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    """读取一次运行的完整轨迹。

    参数：run_id 运行编号，可带可不带 ".jsonl" 后缀（内部会补）；
          runs_dir 轨迹目录，None 时用 `config.RUNS_DIR`。
    返回：List[dict]，按写入顺序，每个元素是一行 JSON 反序列化后的原始 dict
          （**不做字段补全或校验**，老轨迹缺字段时调用方需自己容错）。
    副作用/异常：读文件；文件不存在抛 FileNotFoundError（消息里带完整路径，便于排障）；
                JSON 行损坏抛 json.JSONDecodeError——轨迹是审计数据，刻意不静默跳过坏行。
    """
    directory = Path(runs_dir) if runs_dir is not None else RUNS_DIR
    path = directory / (run_id if run_id.endswith(".jsonl") else f"{run_id}.jsonl")
    if not path.exists():
        raise FileNotFoundError(f"轨迹文件不存在：{path}")
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def summarize_trace(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """把轨迹汇总成「一次运行的画像」，用于快速定位性能与失败点。

    参数：rows 轨迹行（`read_trace()` 的结果或 `TraceRecorder.records` 的 dict 形式），
          任意可迭代，内部只读一遍。
    返回：dict——
        steps           总行数
        total_latency_ms 各行 latency_ms 之和（**不是**墙钟总耗时，也不是仅关键路径耗时）
        by_step         {步名: {count, latency_ms, errors}}，errors 记 status 非 "ok" 的行数
        errors          所有步 errors 之和
    副作用/异常：无；latency_ms 缺失或被写成 None 时按 0.0 计（老轨迹容错口径）。
    """
    items = list(rows)
    by_step: Dict[str, Dict[str, Any]] = {}
    total = 0.0
    for row in items:
        step = str(row.get("step", ""))
        bucket = by_step.setdefault(step, {"count": 0, "latency_ms": 0.0, "errors": 0})
        bucket["count"] += 1
        latency = float(row.get("latency_ms", 0.0) or 0.0)
        bucket["latency_ms"] += latency
        total += latency
        if row.get("status") not in (None, "ok"):
            bucket["errors"] += 1
    for bucket in by_step.values():
        bucket["latency_ms"] = round(bucket["latency_ms"], 3)
    return {
        "steps": len(items),
        "total_latency_ms": round(total, 3),
        "by_step": by_step,
        "errors": sum(b["errors"] for b in by_step.values()),
    }
