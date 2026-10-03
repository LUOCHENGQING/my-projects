"""建议版本链测试：不可变快照、版本推进、可 diff。"""

from __future__ import annotations

import dataclasses
import json

import pytest

from src.constraints import screen_products
from src.optimizer import build_portfolio
from src.versioning import AdviceSnapshot, VersionStore, build_payload, chain_rows, diff_versions


def _payload(client, portfolio, **overrides) -> dict:
    payload = build_payload(
        client,
        portfolio,
        run_id=overrides.pop("run_id", "run-1"),
        engine="native",
        status=overrides.pop("status", "final"),
        directive="pass",
    )
    payload.update(overrides)
    return payload


def test_snapshot_is_immutable_against_input_mutation(data):
    """硬性要求：传入后内容不被后续变更污染。"""
    client = data.client("C001")
    screening = screen_products(client, data.products, 0)
    portfolio = build_portfolio(client, screening.included, data.products)

    payload = _payload(client, portfolio)
    snapshot = AdviceSnapshot.create(
        version=1,
        parent_version=None,
        client_id=client.client_id,
        run_id="run-1",
        change_reason="首次生成建议",
        payload=payload,
    )

    # 篡改原始对象
    payload["portfolio"]["weights"]["HACK"] = 0.99
    payload["client_constraints"]["risk_capacity"] = 5
    assert "HACK" not in snapshot.portfolio_weights
    assert snapshot.client_constraints["risk_capacity"] == client.risk_capacity

    # 篡改返回的载荷副本
    fresh = snapshot.payload
    fresh["portfolio"]["weights"]["HACK2"] = 0.5
    assert "HACK2" not in snapshot.portfolio_weights


def test_snapshot_payload_property_returns_new_object_each_time():
    snapshot = AdviceSnapshot.create(
        version=1,
        parent_version=None,
        client_id="C001",
        run_id="r",
        change_reason="x",
        payload={"portfolio": {"weights": {"A": 0.5}}},
    )
    first = snapshot.payload
    second = snapshot.payload
    assert first == second
    assert first is not second


def test_snapshot_dataclass_is_frozen():
    snapshot = AdviceSnapshot.create(
        version=1,
        parent_version=None,
        client_id="C001",
        run_id="r",
        change_reason="x",
        payload={"a": 1},
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        snapshot.version = 2  # type: ignore[misc]


def test_snapshot_hash_verifies_and_detects_tampering():
    snapshot = AdviceSnapshot.create(
        version=1,
        parent_version=None,
        client_id="C001",
        run_id="r",
        change_reason="x",
        payload={"a": 1},
    )
    assert snapshot.verify() is True
    tampered = dataclasses.replace(snapshot, payload_json='{"a":2}')
    assert tampered.verify() is False


def test_store_append_and_load(tmp_path, data):
    store = VersionStore(tmp_path / "chain.jsonl")
    client = data.client("C001")
    screening = screen_products(client, data.products, 0)
    portfolio = build_portfolio(client, screening.included, data.products)
    snapshot = AdviceSnapshot.create(
        version=1,
        parent_version=None,
        client_id=client.client_id,
        run_id="r",
        change_reason="首次生成建议",
        payload=_payload(client, portfolio),
    )
    store.append(snapshot)
    loaded = store.load_all()
    assert len(loaded) == 1
    assert loaded[0].version == 1
    assert loaded[0].snapshot_hash == snapshot.snapshot_hash
    assert loaded[0].verify() is True


def test_next_version_progression(tmp_path):
    store = VersionStore(tmp_path / "chain.jsonl")
    assert store.next_version("C001") == (1, None)
    store.append(
        AdviceSnapshot.create(
            version=1, parent_version=None, client_id="C001", run_id="r", change_reason="x", payload={"a": 1}
        )
    )
    assert store.next_version("C001") == (2, 1)


def test_chain_is_per_client(tmp_path):
    store = VersionStore(tmp_path / "chain.jsonl")
    for client_id, version in (("C001", 1), ("C002", 1), ("C001", 2)):
        store.append(
            AdviceSnapshot.create(
                version=version,
                parent_version=None,
                client_id=client_id,
                run_id="r",
                change_reason="x",
                payload={"a": version},
            )
        )
    assert [item.version for item in store.chain("C001")] == [1, 2]
    assert [item.version for item in store.chain("C002")] == [1]


def test_diff_versions_reports_changes(data):
    client = data.client("C004")
    screening = screen_products(client, data.products, 0)
    portfolio = build_portfolio(client, screening.included, data.products)

    before = AdviceSnapshot.create(
        version=1,
        parent_version=None,
        client_id=client.client_id,
        run_id="r",
        change_reason="适当性闸门打回",
        payload=_payload(client, portfolio, status="blocked"),
        status="blocked",
    )
    reduced = type(portfolio)(
        weights={},
        products={},
        cash_weight=1.0,
        metrics={},
    )
    after = AdviceSnapshot.create(
        version=2,
        parent_version=1,
        client_id=client.client_id,
        run_id="r",
        change_reason="收紧约束后重配通过",
        payload=_payload(client, reduced, status="final"),
    )

    diff = diff_versions(before, after)
    assert diff["from_version"] == 1 and diff["to_version"] == 2
    assert diff["status"] == {"from": "blocked", "to": "final"}
    assert set(diff["products_removed"]) == set(portfolio.weights)
    assert diff["cash_weight"]["to"] == pytest.approx(1.0)
    assert diff["change_reason"] == "收紧约束后重配通过"


def test_diff_versions_detects_constraint_changes(data):
    client = data.client("C004")
    screening = screen_products(client, data.products, 0)
    portfolio = build_portfolio(client, screening.included, data.products)
    tightened = client.model_copy(update={"risk_capacity": 3}, deep=True)

    before = AdviceSnapshot.create(
        version=1, parent_version=None, client_id="C004", run_id="r", change_reason="x",
        payload=_payload(client, portfolio),
    )
    after = AdviceSnapshot.create(
        version=2, parent_version=1, client_id="C004", run_id="r", change_reason="y",
        payload=_payload(tightened, portfolio),
    )
    diff = diff_versions(before, after)
    assert diff["constraint_changes"]["risk_capacity"] == {"from": 4, "to": 3}


def test_chain_rows_shape(tmp_path):
    store = VersionStore(tmp_path / "chain.jsonl")
    snapshot = AdviceSnapshot.create(
        version=1, parent_version=None, client_id="C001", run_id="r", change_reason="首次", payload={"a": 1}
    )
    rows = chain_rows([snapshot])
    assert rows[0]["version"] == 1
    assert rows[0]["change_reason"] == "首次"
    assert rows[0]["hash"] == snapshot.snapshot_hash


def test_snapshot_roundtrip_record(tmp_path):
    snapshot = AdviceSnapshot.create(
        version=3, parent_version=2, client_id="C002", run_id="r", change_reason="重跑", payload={"nested": {"x": [1, 2]}}
    )
    record = json.loads(json.dumps(snapshot.to_record(), ensure_ascii=False))
    restored = AdviceSnapshot.from_record(record)
    assert restored.payload == snapshot.payload
    assert restored.snapshot_hash == snapshot.snapshot_hash
    assert restored.parent_version == 2


def test_store_clear(tmp_path):
    store = VersionStore(tmp_path / "chain.jsonl")
    store.append(
        AdviceSnapshot.create(
            version=1, parent_version=None, client_id="C001", run_id="r", change_reason="x", payload={"a": 1}
        )
    )
    assert store.load_all()
    store.clear()
    assert store.load_all() == []


def test_load_all_on_missing_file_is_empty(tmp_path):
    assert VersionStore(tmp_path / "nope.jsonl").load_all() == []
