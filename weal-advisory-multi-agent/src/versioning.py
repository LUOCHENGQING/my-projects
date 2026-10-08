"""建议版本链（可回溯的不可变快照）。

所处层次
--------
本模块位于 **L2 领域持久化层**：向下依赖 `schemas`（载荷中的客户/组合/闸门/复核/
压力/反事实模型）与 `utils`（`canonical_json` / `digest` / `now_iso`），不依赖
`pipeline`。向上被这些调用方使用：
- `pipeline.AdvisoryPipeline._save_snapshot`（组装载荷并落盘）与 `node_narrative`
  （取 `next_version`、`chain_rows` 作为建议书上下文）
- `demo.py`（`VersionStore`、`chain_rows`）与 `history.py`（读链 + `diff_versions`）
- `eval/run_eval.py` 与 `tests/test_versioning.py`

设计要点
--------
1. **快照即文本**：`AdviceSnapshot` 内部只存规范化 JSON 文本（`payload_json`）
   与它的 SHA-256，因此快照天然不可变——调用方拿到 `payload` 属性时得到的是
   **每次重新解析出来的新对象**，改它不会污染历史版本；构造后外部再改原始
   dict/list 也不会影响已落盘的版本。
2. **版本链而非覆写**：每次生成建议追加一条记录，带 `parent_version` 与
   `change_reason`，形成 v1 → v2 → … 的可回溯链条（例如"被适当性闸门打回后重配"）。
3. **可 diff**：`diff_versions` 输出产品增删、权重变化、指标变化、规则命中变化
   与客户约束变化，`python -m src.history <client_id>` 直接打印。

对外暴露的关键对象
------------------
- 常量：`DEFAULT_CHAIN_PATH`（默认落盘路径）、`DIFF_METRIC_KEYS` 与
  `DIFF_CONSTRAINT_KEYS`（参与 diff 的键）
- 载荷组装：`build_payload`
- 快照：`AdviceSnapshot`（frozen dataclass；`create` / `from_record` 构造，
  `verify` / `to_record` 与若干只读属性）
- 存储：`VersionStore`（append-only JSONL，按客户切分版本链）
- 差异与展示：`diff_versions`、`chain_rows`

主要输入输出
------------
输入为 `ClientProfile`、`Portfolio` 以及各阶段结果对象（`ScreeningResult`、
`GateDecision`、`HumanReview`、`StressReport`、`CounterfactualReport`，后四者可为
None）；输出为可 JSON 序列化的载荷 dict、`AdviceSnapshot` 与差异 dict。
副作用仅集中在 `VersionStore.append`（建目录 + 追加写文件）与
`VersionStore.clear`（删除链文件），其余函数均为纯函数。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .schemas import (
    ClientProfile,
    CounterfactualReport,
    GateDecision,
    HumanReview,
    Portfolio,
    ScreeningResult,
    StressReport,
)
from .utils import canonical_json, digest, now_iso

#: 版本链默认落盘路径（相对仓库根）
#: 由 <repo>/src/versioning.py 上推两级得到 <repo>/runs/version_chain.jsonl，
#: 与运行时 cwd 无关；`history.py --chain-file` 默认也用它
DEFAULT_CHAIN_PATH = Path(__file__).resolve().parent.parent / "runs" / "version_chain.jsonl"

#: 参与 diff 的指标（共 8 项）
#: diff_versions 对这批键**全量输出**：即使某指标没变，也会以 0.0 出现在
#: metric_changes 中（history.py 打印时会再过滤 1e-12 以内的噪声）
DIFF_METRIC_KEYS: tuple[str, ...] = (
    "expected_return",
    "expected_volatility",
    "expected_fee_rate",
    "liquidity_ratio",
    "max_single_weight",
    "weighted_risk_level",
    "holding_count",
    "cash_weight",
)

#: 参与 diff 的客户约束字段
#: 与 DIFF_METRIC_KEYS 不同，这里只输出**取值实际发生变化**的键
DIFF_CONSTRAINT_KEYS: tuple[str, ...] = (
    "risk_capacity",
    "investment_horizon_years",
    "liquidity_floor_ratio",
    "max_single_product_ratio",
    "max_single_class_ratio",
    "max_single_issuer_ratio",
    "investable_amount",
    "prohibited_categories",
    "prohibited_product_ids",
    "experienced_categories",
    "qualified_investor",
    "currency",
    "tax_advantaged_quota",
    "dual_record_completed",
)


def build_payload(
    client: ClientProfile,
    portfolio: Portfolio,
    *,
    run_id: str,
    engine: str,
    status: str,
    directive: str,
    screening: ScreeningResult | None = None,
    gate: GateDecision | None = None,
    human_review: HumanReview | None = None,
    stress: StressReport | None = None,
    counterfactual: CounterfactualReport | None = None,
    narrative: str = "",
    notes: Sequence[str] = (),
) -> dict[str, Any]:
    """组装一份完整的建议快照载荷（客户约束快照 + 产品要素快照 + 规则命中 + 权重）。

    参数:
        client: 客户档案，用 `constraint_snapshot()` 冻结"当时按什么约束配的"。
        portfolio: 组合结果，提供权重、现金比、指标与理由。
        run_id: 本次流水线运行标识（与 trace 的 run_id 对应）。
        engine: 引擎名（langgraph / native），用于事后区分双引擎结果。
        status: 建议状态（如 final / escalated）。
        directive: 适当性闸门的处置方向（pass / reoptimize / reject 等）。
        screening: 产品筛选结果；为 None 时载荷中 `screening` 记为 None。
        gate: 闸门决策；为 None 时不写规则命中（rule_hits 为空列表）。
        human_review: 人工复核记录；为 None 时记为 None。
        stress: 压力测试报告；为 None 时记为 None。
        counterfactual: 反事实报告；为 None 时记为 None。
        narrative: 建议书正文；只落 `digest` 摘要，不落原文。
        notes: 附加说明文本列表。

    返回:
        可直接交给 `canonical_json` / `json.dumps` 的 dict（全部为原生类型）。

    说明:
        所有列表字段都做了 `list(...)` 复制，避免载荷与调用方持有的可变对象共享引用；
        产品权重与产品快照都以 `portfolio.held_ids()` 为准（只含权重 > 0 的产品，
        且顺序由 held_ids 排序决定），因此同一组合必得同一份载荷。
    """
    return {
        "run_id": run_id,
        "engine": engine,
        "status": status,
        "directive": directive,
        # 客户硬约束快照：事后回溯该版本是在什么约束下生成的
        "client_constraints": client.constraint_snapshot(),
        "portfolio": {
            # 键顺序来自 held_ids()（按 product_id 排序），保证载荷确定性
            "weights": {pid: portfolio.weights[pid] for pid in portfolio.held_ids()},
            "cash_weight": portfolio.cash_weight,
            "metrics": dict(portfolio.metrics),
            "rationale": list(portfolio.rationale),
        },
        # 与 weights 同序：只快照实际持有的产品要素（费率/风险等级等）
        "product_snapshots": [
            portfolio.products[pid].snapshot() for pid in portfolio.held_ids()
        ],
        "screening": {
            "included": list(screening.included),
            "excluded": [item.model_dump() for item in screening.excluded],
        }
        if screening
        else None,
        # blocks 在前、warns 在后，保留原始顺序（不排序），便于回溯"被谁拦下"
        "rule_hits": (
            [v.model_dump() for v in (gate.blocks + gate.warns)] if gate else []
        ),
        # dict.fromkeys 用于**保序去重**：同一规则可能命中多个产品
        "block_rules": list(dict.fromkeys(v.rule_id for v in gate.blocks)) if gate else [],
        "warn_rules": list(dict.fromkeys(v.rule_id for v in gate.warns)) if gate else [],
        "stress": stress.model_dump() if stress else None,
        "counterfactual": counterfactual.model_dump() if counterfactual else None,
        "human_review": human_review.model_dump() if human_review else None,
        # 建议书正文可能很长，只存摘要；空字符串表示"没有正文"，而不是 digest("")
        "narrative_digest": digest(narrative) if narrative else "",
        "notes": list(notes),
    }


@dataclass(frozen=True)
class AdviceSnapshot:
    """一条不可变的建议版本快照。

    职责:
        把一次生成的建议固化成一个可校验哈希、可 diff、可回放的历史版本；
        内容以规范化 JSON 文本形式保存，读取时再解析成新对象。

    关键属性:
        version: 版本号，同一客户内自 1 起递增。
        parent_version: 父版本号；首版为 None（版本链起点）。
        client_id / run_id: 所属客户与产生它的运行标识（run_id 与 trace 对应）。
        created_at: 创建时间，`utils.now_iso()` 的秒级本地时区 ISO 串。
        change_reason: 相对上一版的变更原因，由调用方给出。
        status: 建议状态；`create` 未显式传入时取载荷里的 `status`。
        payload_json: 规范化 JSON 文本，是快照内容的**唯一来源**。
        snapshot_hash: `digest(payload_json, length=32)`（SHA-256 前 32 个十六进制
            字符），与 payload_json 绑定，可用 `verify()` 自检。

    不可变性:
        `frozen=True` 使字段不可重绑定；所有读取接口（`payload` 等）都返回**新构造
        的对象**，因此外部修改不会回写快照。新增版本必须走 `create` / `from_record`。

    被谁使用:
        `pipeline.AdvisoryPipeline._save_snapshot` 创建，`VersionStore` 落盘与加载，
        `history.py` 展示，`diff_versions` / `chain_rows` 消费。
    """

    version: int
    parent_version: int | None
    client_id: str
    run_id: str
    created_at: str
    change_reason: str
    status: str
    payload_json: str
    snapshot_hash: str

    # ------------------------------------------------------------------
    @classmethod
    def create(
        cls,
        *,
        version: int,
        parent_version: int | None,
        client_id: str,
        run_id: str,
        change_reason: str,
        payload: Mapping[str, Any],
        status: str = "",
        created_at: str | None = None,
    ) -> "AdviceSnapshot":
        """由普通字典创建快照（深拷贝语义：立即序列化，与原始对象解耦）。

        参数:
            version: 版本号。
            parent_version: 父版本号，首版传 None。
            client_id: 客户号。
            run_id: 运行标识。
            change_reason: 变更原因。
            payload: 载荷 dict，通常由 `build_payload` 生成。
            status: 建议状态；传空字符串（默认）时回落到 `payload["status"]`。
            created_at: 创建时间；None 时取 `utils.now_iso()`。

        返回:
            新的 `AdviceSnapshot` 实例，`snapshot_hash` 为 `payload_json` 的
            SHA-256 前 32 位。

        副作用/异常:
            无副作用。`canonical_json` 会把未知类型兜底为字符串，故常规输入不会抛
            异常；自引用结构等病态输入可能触发 `RecursionError`。
        """
        # 立刻序列化：此后外部再修改原始 payload 也不会影响本快照
        text = canonical_json(payload)
        return cls(
            version=version,
            parent_version=parent_version,
            client_id=client_id,
            run_id=run_id,
            created_at=created_at or now_iso(),
            change_reason=change_reason,
            status=status or str(payload.get("status", "")),
            payload_json=text,
            snapshot_hash=digest(text, length=32),
        )

    # ------------------------------------------------------------------
    @property
    def payload(self) -> dict[str, Any]:
        """快照载荷（每次返回全新对象，修改它不影响快照本身）。

        返回:
            `json.loads(payload_json)` 得到的 dict；每次访问都重新解析，
            因此调用方拿到的是独立的新对象而非共享引用。
        """
        return json.loads(self.payload_json)

    @property
    def client_constraints(self) -> dict[str, Any]:
        """客户硬约束快照。

        返回:
            `payload["client_constraints"]` 的浅拷贝；键缺失或值为 None 时返回空 dict。
        """
        return dict(self.payload.get("client_constraints") or {})

    @property
    def portfolio_weights(self) -> dict[str, float]:
        """产品权重快照（只含载荷中记录的产品）。

        返回:
            product_id -> float 权重 的新 dict；值统一经 `float()` 转换，
            载荷缺失时为 {}。
        """
        payload = self.payload.get("portfolio") or {}
        return {str(k): float(v) for k, v in (payload.get("weights") or {}).items()}

    @property
    def cash_weight(self) -> float:
        """现金权重。

        返回:
            `payload["portfolio"]["cash_weight"]` 的 float 值；缺失时为 0.0。
        """
        return float((self.payload.get("portfolio") or {}).get("cash_weight", 0.0))

    @property
    def metrics(self) -> dict[str, float]:
        """组合指标快照（`DIFF_METRIC_KEYS` 中的各项，值经 float 转换）。

        返回:
            指标名 -> float 的新 dict；载荷缺失时为 {}。
        """
        payload = self.payload.get("portfolio") or {}
        return {str(k): float(v) for k, v in (payload.get("metrics") or {}).items()}

    @property
    def rule_ids(self) -> list[str]:
        """该版本命中的全部规则号（block + warn，保序去重）。

        返回:
            `payload["rule_hits"]` 中的 rule_id 列表，保持原始顺序并用
            `dict.fromkeys` 去重；未命中任何规则时为空列表。
        """
        return list(dict.fromkeys(v["rule_id"] for v in self.payload.get("rule_hits") or []))

    @property
    def product_ids(self) -> list[str]:
        """持有的产品号（升序）。

        返回:
            对 `portfolio_weights` 的键排序得到的新列表；无持仓时为空列表。
        """
        return sorted(self.portfolio_weights)

    def verify(self) -> bool:
        """校验快照哈希与内容一致（防篡改自检）。

        返回:
            就地重算 `digest(payload_json, length=32)` 并与 `snapshot_hash` 比对：
            True 表示内容未被改写，False 表示哈希与内容不一致。
        """
        return digest(self.payload_json, length=32) == self.snapshot_hash

    def to_record(self) -> dict[str, Any]:
        """转成 JSONL 记录。

        返回:
            含全部标量字段（version / parent_version / client_id / run_id /
            created_at / change_reason / status / snapshot_hash）与**解析后的
            payload** 对象的 dict。

        说明:
            落盘的是 payload 而非 `payload_json`；读取时由 `from_record` 重新
            规范化，只要内容一致就能得到同一个 `snapshot_hash`。
        """
        return {
            "version": self.version,
            "parent_version": self.parent_version,
            "client_id": self.client_id,
            "run_id": self.run_id,
            "created_at": self.created_at,
            "change_reason": self.change_reason,
            "status": self.status,
            "snapshot_hash": self.snapshot_hash,
            "payload": json.loads(self.payload_json),
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "AdviceSnapshot":
        """由 JSONL 记录还原快照。

        参数:
            record: `to_record()` 产出的 dict（或同结构的历史记录）。
                必需键为 `payload`、`version`、`client_id`；其余键缺失时取默认值。

        返回:
            还原出的 `AdviceSnapshot`。

        说明:
            payload 重新经 `canonical_json` 规范化，因此内容不变时还原后的
            `snapshot_hash` 与原快照一致；记录缺 `snapshot_hash` 时（`or` 短路）
            按重算值填充，保证 `verify()` 仍可通过。

        异常:
            缺少 `payload` / `version` / `client_id` 时抛 `KeyError`。
        """
        text = canonical_json(record["payload"])
        return cls(
            version=int(record["version"]),
            parent_version=None if record.get("parent_version") is None else int(record["parent_version"]),
            client_id=str(record["client_id"]),
            run_id=str(record.get("run_id", "")),
            created_at=str(record.get("created_at", "")),
            change_reason=str(record.get("change_reason", "")),
            status=str(record.get("status", "")),
            payload_json=text,
            snapshot_hash=str(record.get("snapshot_hash") or digest(text, length=32)),
        )


class VersionStore:
    """append-only 的建议版本链存储（JSONL）。

    职责:
        以 JSONL 文件保存所有客户的建议版本（一行一条记录），只追加、不覆写；
        读取时按 `client_id` 过滤并按 `version` 升序得到某客户的版本链。

    关键属性:
        path: 链文件路径；构造时把入参转为 `Path`，None 时用 `DEFAULT_CHAIN_PATH`
            （即 `<repo>/runs/version_chain.jsonl`）。

    被谁使用:
        `pipeline.AdvisoryPipeline`（默认以 `VersionStore()` 作为快照存储）、
        `demo.py`、`history.py`（只读）、`eval/run_eval.py`（临时路径）、
        `tests/test_versioning.py` 与 `tests/conftest.py` 的 `temp_store` fixture。

    注意:
        本类没有文件锁与并发控制；同一路径的并发 `append` 需调用方保证串行。
    """

    def __init__(self, path: Path | str | None = None) -> None:
        """初始化存储路径。

        参数:
            path: 链文件路径；None 表示使用 `DEFAULT_CHAIN_PATH`。

        副作用:
            无——既不建目录也不建文件，落盘推迟到 `append`。
        """
        self.path = Path(path) if path is not None else DEFAULT_CHAIN_PATH

    # ------------------------------------------------------------------
    def append(self, snapshot: AdviceSnapshot) -> None:
        """追加一条版本记录（父目录不存在时自动创建）。

        参数:
            snapshot: 待落盘的快照。

        副作用:
            创建父目录后以 UTF-8 追加写入一行 JSON（`ensure_ascii=False`，中文原样
            保存）。不校验版本号是否重复，也不做锁。

        异常:
            目录不可创建或文件不可写时由 `Path.mkdir` / `open` 抛出 `OSError`。
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        record = json.dumps(snapshot.to_record(), ensure_ascii=False)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(record + "\n")

    def load_all(self) -> list[AdviceSnapshot]:
        """读取全部版本快照（文件不存在时返回空列表）。

        返回:
            文件中全部记录还原出的快照列表，**按文件行序**（所有客户混在一起，
            未排序）；空行被跳过。

        异常:
            某行不是合法 JSON 时抛 `json.JSONDecodeError`（不做逐行容错跳过）。
        """
        if not self.path.exists():
            return []
        snapshots: list[AdviceSnapshot] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                snapshots.append(AdviceSnapshot.from_record(json.loads(line)))
        return snapshots

    def chain(self, client_id: str) -> list[AdviceSnapshot]:
        """某客户的版本链（按版本号升序）。

        参数:
            client_id: 客户号。

        返回:
            该客户的快照列表，按 `version` 升序；无记录时为空列表。
        """
        return sorted(
            (item for item in self.load_all() if item.client_id == client_id),
            key=lambda item: item.version,
        )

    def next_version(self, client_id: str) -> tuple[int, int | None]:
        """下一个版本号及其父版本号。

        参数:
            client_id: 客户号。

        返回:
            `(下一个版本号, 父版本号)`：该客户无历史时返回 `(1, None)`，否则返回
            `(链尾版本号 + 1, 链尾版本号)`。

        说明:
            只依赖 `chain()` 的最后一个元素（版本号最大者），因此要求版本号与
            追加顺序一致（`append` 不校验，由调用方保证）。
        """
        chain = self.chain(client_id)
        if not chain:
            return 1, None
        last = chain[-1]
        return last.version + 1, last.version

    def clear(self) -> None:
        """清空版本链（仅用于测试与演示重置）。

        副作用:
            链文件存在时删除整个文件——**所有客户**的记录一并消失，目录保留；
            文件不存在时什么也不做。
        """
        if self.path.exists():
            self.path.unlink()


def diff_versions(before: AdviceSnapshot, after: AdviceSnapshot) -> dict[str, Any]:
    """两个版本的完整结构化差异。

    参数:
        before: 变更前的快照。
        after: 变更后的快照；返回的 `client_id` 与 `change_reason` 都取自它，
            且**不校验**版本号先后（传反了不会报错，只是差异方向反过来）。

    返回:
        dict，键为：`client_id`、`from_version`、`to_version`、`change_reason`、
        `status{from,to}`、`products_added`、`products_removed`、`weight_changes`、
        `metric_changes`、`rule_hits_added`、`rule_hits_removed`、
        `constraint_changes`、`cash_weight{from,to}`、`hash{from,to}`。

    说明:
        - 权重差按 `round(after - before, 12)` 计算（与 `utils.QUANTIZE` 的 12 位
          量化对齐），并丢弃绝对值 `<= 1e-9` 的项（该字面量与 `utils.WEIGHT_TOL`
          同量级，此处未引用该常量）。因此"产品增删"按 0 权重参与相减，可能与
          `weight_changes` 的记录不同步。
        - `metric_changes` 对 `DIFF_METRIC_KEYS` **全键输出**，未变化的项为 0.0。
        - `constraint_changes` 只收录取值不等的 `DIFF_CONSTRAINT_KEYS`，每项形如
          `{"from": ..., "to": ...}`（不做 12 位量化，直接比较原值）。
        - 规则命中变化由 `AdviceSnapshot.rule_ids`（block + warn 保序去重）做集合差。
    """
    weights_before = before.portfolio_weights
    weights_after = after.portfolio_weights
    keys = sorted(set(weights_before) | set(weights_after))
    weight_changes = {
        key: round(weights_after.get(key, 0.0) - weights_before.get(key, 0.0), 12) for key in keys
    }
    # 1e-9 阈值：过滤量化残差，只保留"真实发生"的权重变化
    weight_changes = {key: value for key, value in weight_changes.items() if abs(value) > 1e-9}

    metrics_before = before.metrics
    metrics_after = after.metrics
    # 注意：这里不做阈值过滤，未变化的指标会以 0.0 保留在结果中
    metric_changes = {
        key: round(metrics_after.get(key, 0.0) - metrics_before.get(key, 0.0), 12)
        for key in DIFF_METRIC_KEYS
    }

    constraints_before = before.client_constraints
    constraints_after = after.client_constraints
    # 只输出真正变化的约束键，避免 diff 报告里塞满未变项
    constraint_changes = {
        key: {"from": constraints_before.get(key), "to": constraints_after.get(key)}
        for key in DIFF_CONSTRAINT_KEYS
        if constraints_before.get(key) != constraints_after.get(key)
    }

    rules_before = set(before.rule_ids)
    rules_after = set(after.rule_ids)

    return {
        "client_id": after.client_id,
        "from_version": before.version,
        "to_version": after.version,
        "change_reason": after.change_reason,
        "status": {"from": before.status, "to": after.status},
        "products_added": sorted(set(weights_after) - set(weights_before)),
        "products_removed": sorted(set(weights_before) - set(weights_after)),
        "weight_changes": weight_changes,
        "metric_changes": metric_changes,
        "rule_hits_added": sorted(rules_after - rules_before),
        "rule_hits_removed": sorted(rules_before - rules_after),
        "constraint_changes": constraint_changes,
        "cash_weight": {"from": before.cash_weight, "to": after.cash_weight},
        "hash": {"from": before.snapshot_hash, "to": after.snapshot_hash},
    }


def chain_rows(chain: Iterable[AdviceSnapshot]) -> list[dict[str, Any]]:
    """版本链摘要表（history CLI 与 demo 使用）。

    参数:
        chain: 快照的可迭代对象，通常是某客户的升序版本链。

    返回:
        list[dict]，每项含 `version`、`parent_version`、`created_at`、`status`、
        `change_reason`、`holdings`（持仓只数 = 权重表长度）、`cash_weight`、
        `hash`（快照哈希）；顺序与入参一致，不做排序。
    """
    rows: list[dict[str, Any]] = []
    for snapshot in chain:
        rows.append(
            {
                "version": snapshot.version,
                "parent_version": snapshot.parent_version,
                "created_at": snapshot.created_at,
                "status": snapshot.status,
                "change_reason": snapshot.change_reason,
                "holdings": len(snapshot.portfolio_weights),
                "cash_weight": snapshot.cash_weight,
                "hash": snapshot.snapshot_hash,
            }
        )
    return rows
