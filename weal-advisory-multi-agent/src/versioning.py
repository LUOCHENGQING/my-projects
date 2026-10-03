"""建议版本链（可回溯的不可变快照）。

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
DEFAULT_CHAIN_PATH = Path(__file__).resolve().parent.parent / "runs" / "version_chain.jsonl"

#: 参与 diff 的指标
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
    """组装一份完整的建议快照载荷（客户约束快照 + 产品要素快照 + 规则命中 + 权重）。"""
    return {
        "run_id": run_id,
        "engine": engine,
        "status": status,
        "directive": directive,
        "client_constraints": client.constraint_snapshot(),
        "portfolio": {
            "weights": {pid: portfolio.weights[pid] for pid in portfolio.held_ids()},
            "cash_weight": portfolio.cash_weight,
            "metrics": dict(portfolio.metrics),
            "rationale": list(portfolio.rationale),
        },
        "product_snapshots": [
            portfolio.products[pid].snapshot() for pid in portfolio.held_ids()
        ],
        "screening": {
            "included": list(screening.included),
            "excluded": [item.model_dump() for item in screening.excluded],
        }
        if screening
        else None,
        "rule_hits": (
            [v.model_dump() for v in (gate.blocks + gate.warns)] if gate else []
        ),
        "block_rules": list(dict.fromkeys(v.rule_id for v in gate.blocks)) if gate else [],
        "warn_rules": list(dict.fromkeys(v.rule_id for v in gate.warns)) if gate else [],
        "stress": stress.model_dump() if stress else None,
        "counterfactual": counterfactual.model_dump() if counterfactual else None,
        "human_review": human_review.model_dump() if human_review else None,
        "narrative_digest": digest(narrative) if narrative else "",
        "notes": list(notes),
    }


@dataclass(frozen=True)
class AdviceSnapshot:
    """一条不可变的建议版本快照。"""

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
        """由普通字典创建快照（深拷贝语义：立即序列化，与原始对象解耦）。"""
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
        """快照载荷（每次返回全新对象，修改它不影响快照本身）。"""
        return json.loads(self.payload_json)

    @property
    def client_constraints(self) -> dict[str, Any]:
        return dict(self.payload.get("client_constraints") or {})

    @property
    def portfolio_weights(self) -> dict[str, float]:
        payload = self.payload.get("portfolio") or {}
        return {str(k): float(v) for k, v in (payload.get("weights") or {}).items()}

    @property
    def cash_weight(self) -> float:
        return float((self.payload.get("portfolio") or {}).get("cash_weight", 0.0))

    @property
    def metrics(self) -> dict[str, float]:
        payload = self.payload.get("portfolio") or {}
        return {str(k): float(v) for k, v in (payload.get("metrics") or {}).items()}

    @property
    def rule_ids(self) -> list[str]:
        """该版本命中的全部规则号（block + warn，保序去重）。"""
        return list(dict.fromkeys(v["rule_id"] for v in self.payload.get("rule_hits") or []))

    @property
    def product_ids(self) -> list[str]:
        return sorted(self.portfolio_weights)

    def verify(self) -> bool:
        """校验快照哈希与内容一致（防篡改自检）。"""
        return digest(self.payload_json, length=32) == self.snapshot_hash

    def to_record(self) -> dict[str, Any]:
        """转成 JSONL 记录。"""
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
        """由 JSONL 记录还原快照。"""
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
    """append-only 的建议版本链存储（JSONL）。"""

    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path is not None else DEFAULT_CHAIN_PATH

    # ------------------------------------------------------------------
    def append(self, snapshot: AdviceSnapshot) -> None:
        """追加一条版本记录（父目录不存在时自动创建）。"""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        record = json.dumps(snapshot.to_record(), ensure_ascii=False)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(record + "\n")

    def load_all(self) -> list[AdviceSnapshot]:
        """读取全部版本快照（文件不存在时返回空列表）。"""
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
        """某客户的版本链（按版本号升序）。"""
        return sorted(
            (item for item in self.load_all() if item.client_id == client_id),
            key=lambda item: item.version,
        )

    def next_version(self, client_id: str) -> tuple[int, int | None]:
        """下一个版本号及其父版本号。"""
        chain = self.chain(client_id)
        if not chain:
            return 1, None
        last = chain[-1]
        return last.version + 1, last.version

    def clear(self) -> None:
        """清空版本链（仅用于测试与演示重置）。"""
        if self.path.exists():
            self.path.unlink()


def diff_versions(before: AdviceSnapshot, after: AdviceSnapshot) -> dict[str, Any]:
    """两个版本的完整结构化差异。"""
    weights_before = before.portfolio_weights
    weights_after = after.portfolio_weights
    keys = sorted(set(weights_before) | set(weights_after))
    weight_changes = {
        key: round(weights_after.get(key, 0.0) - weights_before.get(key, 0.0), 12) for key in keys
    }
    weight_changes = {key: value for key, value in weight_changes.items() if abs(value) > 1e-9}

    metrics_before = before.metrics
    metrics_after = after.metrics
    metric_changes = {
        key: round(metrics_after.get(key, 0.0) - metrics_before.get(key, 0.0), 12)
        for key in DIFF_METRIC_KEYS
    }

    constraints_before = before.client_constraints
    constraints_after = after.client_constraints
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
    """版本链摘要表（history CLI 与 demo 使用）。"""
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
