"""样例数据装载（全部虚构）。

数据分层：
- `data/clients.json`      客户档案（硬约束 + 软偏好 + 适当性属性）
- `data/products.json`     产品要素表（全部虚构产品与虚构发行主体）
- `data/questionnaire.json` 风险测评问卷题目、分档阈值与作答
- `data/stress_scenarios.json` 情景定义与资产类别敏感性系数表

合规红线：本文件及全部数据文件中不得出现任何真实机构、真实产品名称。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .schemas import ClientProfile, Product

#: 仓库 data 目录（相对于本文件定位，不受运行时 cwd 影响）
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _read_json(path: Path) -> Any:
    """读取 UTF-8 JSON 文件。"""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


@dataclass(frozen=True)
class DataBundle:
    """一次性装载的全部样例数据（冻结，防止流水线运行期被改写）。"""

    clients: dict[str, ClientProfile]
    products: dict[str, Product]
    questionnaire: dict[str, Any]
    stress_config: dict[str, Any]
    data_dir: Path

    # ------------------------------------------------------------------
    def client(self, client_id: str) -> ClientProfile:
        """按客户号取客户档案（返回深拷贝，调用方可安全收紧而不污染缓存）。"""
        if client_id not in self.clients:
            raise KeyError(f"未找到客户 {client_id}，可选：{', '.join(sorted(self.clients))}")
        return self.clients[client_id].model_copy(deep=True)

    def sample_clients(self) -> list[ClientProfile]:
        """评估样本客户（`eval_sample == true`），按客户号排序。"""
        return [self.client(cid) for cid in sorted(self.clients) if self.clients[cid].eval_sample]

    def all_clients(self) -> list[ClientProfile]:
        """全部样例客户（含反例客户），按客户号排序。"""
        return [self.client(cid) for cid in sorted(self.clients)]

    def products_subset(self, product_ids: Iterable[str]) -> dict[str, Product]:
        """按产品号取子集（保持传入顺序的排序结果）。"""
        return {pid: self.products[pid].model_copy(deep=True) for pid in sorted(set(product_ids))}

    def questionnaire_level(self, client_id: str) -> int | None:
        """由问卷作答折算风险等级（缺省返回 None）。"""
        return score_to_level(self.questionnaire_score(client_id), self.questionnaire)

    def questionnaire_score(self, client_id: str) -> int | None:
        """问卷得分合计（缺省返回 None）。"""
        responses = self.questionnaire.get("responses", {})
        answers = responses.get(client_id)
        if not answers:
            return None
        return int(sum(answers.values()))


def score_to_level(score: int | None, questionnaire: dict[str, Any]) -> int | None:
    """把问卷总分折算成风险等级（按 `level_bands` 分档）。"""
    if score is None:
        return None
    for band in questionnaire.get("level_bands", []):
        if band["min_score"] <= score <= band["max_score"]:
            return int(band["level"])
    return None


def level_label(level: int | None, questionnaire: dict[str, Any]) -> str:
    """风险等级对应的中文标签。"""
    if level is None:
        return "未测评"
    for band in questionnaire.get("level_bands", []):
        if int(band["level"]) == level:
            return str(band["label"])
    return f"R{level}"


def load_data(data_dir: Path | str | None = None) -> DataBundle:
    """装载样例数据并完成基础自洽校验。"""
    base = Path(data_dir) if data_dir is not None else DATA_DIR
    clients_raw = _read_json(base / "clients.json")
    products_raw = _read_json(base / "products.json")
    questionnaire = _read_json(base / "questionnaire.json")
    stress_config = _read_json(base / "stress_scenarios.json")

    clients = {item["client_id"]: ClientProfile.model_validate(item) for item in clients_raw}
    products = {item["product_id"]: Product.model_validate(item) for item in products_raw}

    duplicate_clients = len(clients_raw) - len(clients)
    duplicate_products = len(products_raw) - len(products)
    if duplicate_clients or duplicate_products:
        raise ValueError("样例数据存在重复主键，请检查 clients.json / products.json")

    return DataBundle(
        clients=clients,
        products=products,
        questionnaire=questionnaire,
        stress_config=stress_config,
        data_dir=base,
    )
