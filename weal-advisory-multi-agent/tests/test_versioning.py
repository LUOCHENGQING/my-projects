"""建议版本链测试：不可变快照、版本推进、可 diff。

被测模块：`src.versioning`
- `AdviceSnapshot`：不可变（frozen）建议快照，内容寻址哈希 + 深拷贝隔离 + `verify()` 防篡改；
- `VersionStore`：按客户追加版本链（jsonl），`next_version` 给出"下一版号 + 父指针"；
- `build_payload` / `diff_versions` / `chain_rows`：快照载荷、版本间差异、展示行。

覆盖策略
- 正常路径：追加 → 读取 → 校验哈希一致，版本号与父指针逐版推进；
- 边界路径：空链首次写入返回 (1, None)、缺文件时 load_all 返回空、clear 后链为空；
- 异常/对抗探针：外部篡改原始载荷对象、篡改 `payload` 返回副本、或直接替换 `payload_json`
  都必须被隔离或被 `verify()` 识破；
- 口径：diff 必须覆盖状态迁移、产品增减、现金权重变化、客户约束变化与变更原因。

注：本模块全部为模块级测试函数，未定义测试类，故无类级 docstring。
"""

from __future__ import annotations

import dataclasses
import json

import pytest

from src.constraints import screen_products
from src.optimizer import build_portfolio
from src.versioning import AdviceSnapshot, VersionStore, build_payload, chain_rows, diff_versions


def _payload(client, portfolio, **overrides) -> dict:
    """用 build_payload 组装版本快照载荷。

    `run_id` / `status` 走具名参数（属于函数签名），其余键在构建完成后直接覆盖载荷，
    便于各用例只改自己关心的那一两个字段来构造版本差异。
    """
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
    """不变式：payload 属性每次返回新的深拷贝——值相等但身份不同，调用方无法通过别名改快照。"""
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
    """不变式：AdviceSnapshot 是 frozen dataclass，赋值必须抛 FrozenInstanceError（版本记录不可回改）。"""
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
    """不变式：verify() 对原快照为真；仅替换 payload_json 后必须为假——哈希绑定内容，防事后改账。"""
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
    """不变式：落盘后 load_all 能还原版本号与 snapshot_hash，且校验通过（写入即内容寻址）。"""
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
    """不变式：空链返回 (1, None)；已有 v1 时返回 (2, 1)——版本链父指针必须指向上一版。"""
    store = VersionStore(tmp_path / "chain.jsonl")
    assert store.next_version("C001") == (1, None)
    store.append(
        AdviceSnapshot.create(
            version=1, parent_version=None, client_id="C001", run_id="r", change_reason="x", payload={"a": 1}
        )
    )
    assert store.next_version("C001") == (2, 1)


def test_chain_is_per_client(tmp_path):
    """不变式：版本链按 client_id 隔离——交替写入 C001/C002 不得串链，且各自版本号升序。"""
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
    """口径：diff 必须给出起止版本、状态迁移、被移除产品、现金权重变化与变更原因。

    用 C004（首轮被适当性闸门打回）造出 blocked → final 的真实场景；
    after 组合换成 100% 现金，使 products_removed 恰好等于原持仓集合。
    """
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
    """口径：客户约束变化必须出现在 constraint_changes——此处 C004 风险等级 4 → 3，体现"打回后收紧重配"。"""
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
    """口径：chain_rows 每行必须含 version / change_reason / hash，供前端或评估报告直接渲染。

    这里的 store 仅为占位，本用例只校验行渲染口径，不经过存储层读取。
    """
    store = VersionStore(tmp_path / "chain.jsonl")
    snapshot = AdviceSnapshot.create(
        version=1, parent_version=None, client_id="C001", run_id="r", change_reason="首次", payload={"a": 1}
    )
    rows = chain_rows([snapshot])
    assert rows[0]["version"] == 1
    assert rows[0]["change_reason"] == "首次"
    assert rows[0]["hash"] == snapshot.snapshot_hash


def test_snapshot_roundtrip_record(tmp_path):
    """不变式：to_record → JSON → from_record 往返后 payload、哈希与父指针都不变（嵌套结构不丢失）。"""
    snapshot = AdviceSnapshot.create(
        version=3, parent_version=2, client_id="C002", run_id="r", change_reason="重跑", payload={"nested": {"x": [1, 2]}}
    )
    record = json.loads(json.dumps(snapshot.to_record(), ensure_ascii=False))
    restored = AdviceSnapshot.from_record(record)
    assert restored.payload == snapshot.payload
    assert restored.snapshot_hash == snapshot.snapshot_hash
    assert restored.parent_version == 2


def test_store_clear(tmp_path):
    """不变式：clear() 后版本链必须为空（重置留痕，供演示/测试复用同一运行目录）。"""
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
    """边界：链文件不存在时 load_all 返回空列表而不是抛异常（首次运行即可用）。"""
    assert VersionStore(tmp_path / "nope.jsonl").load_all() == []
