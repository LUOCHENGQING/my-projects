"""样例数据装载（全部虚构）。

所处层次
--------
本模块位于 **L1 数据装载层**：把 `data/` 下的四个 JSON 文件读成强类型对象，向
上层提供唯一的数据入口。调用方包括 `pipeline.AdvisoryPipeline`（构造参数
`data: DataBundle`）、`pipeline.run_pipeline`、`demo.py`、`eval/run_eval.py`，
以及 `tests/conftest.py` 的 `data` fixture。

数据分层：
- `data/clients.json`      客户档案（硬约束 + 软偏好 + 适当性属性）
- `data/products.json`     产品要素表（全部虚构产品与虚构发行主体）
- `data/questionnaire.json` 风险测评问卷题目、分档阈值与作答
- `data/stress_scenarios.json` 情景定义与资产类别敏感性系数表

对外暴露
--------
- `DATA_DIR`：仓库 data 目录（绝对路径，与运行时 cwd 无关）
- `DataBundle`：一次装载后的只读容器（客户 / 产品 / 问卷 / 情景 / 目录）
- `load_data`：装载入口
- `score_to_level` / `level_label`：问卷总分与风险等级的互相换算

主要输入输出
------------
输入为 data 目录路径（缺省用 `DATA_DIR`），输出为 `DataBundle`。异常情形：
文件缺失或 JSON 解析失败时由底层 `open`/`json.load` 抛出，主键重复时由
`load_data` 抛 `ValueError`。

合规红线：本文件及全部数据文件中不得出现任何真实机构、真实产品名称。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .schemas import ClientProfile, Product

#: 仓库 data 目录（相对于本文件定位，不受运行时 cwd 影响）
#: 例：由 <repo>/src/dataset.py 上推两级得到 <repo>/data
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _read_json(path: Path) -> Any:
    """读取 UTF-8 JSON 文件。

    参数:
        path: JSON 文件路径。

    返回:
        解析后的 Python 对象（顶层为 list 或 dict，取决于文件内容）。

    副作用/异常:
        只读打开文件；路径不存在时抛 `FileNotFoundError`，内容非法时抛
        `json.JSONDecodeError`。
    """
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


@dataclass(frozen=True)
class DataBundle:
    """一次性装载的全部样例数据（冻结，防止流水线运行期被改写）。

    职责:
        作为整条流水线的"数据只读视图"，同时提供若干取数辅助方法；所有取数方法
        都返回深拷贝，保证调用方的本地修改（如适当性收紧）不会污染这份共享数据。

    关键属性:
        clients: 客户号 -> `ClientProfile`（来自 clients.json）。
        products: 产品号 -> `Product`（来自 products.json）。
        questionnaire: 问卷原始字典，含题目、`level_bands` 分档与 `responses` 作答。
        stress_config: 压力测试情景与敏感性系数原始字典。
        data_dir: 实际装载使用的 data 目录，便于报错时定位数据源。

    被谁使用:
        `pipeline.AdvisoryPipeline`（`self.data`）、`demo.py`、`eval/run_eval.py`、
        `tests/conftest.py`；由 `load_data` 构造。
    """

    clients: dict[str, ClientProfile]
    products: dict[str, Product]
    questionnaire: dict[str, Any]
    stress_config: dict[str, Any]
    data_dir: Path

    # ------------------------------------------------------------------
    def client(self, client_id: str) -> ClientProfile:
        """按客户号取客户档案（返回深拷贝，调用方可安全收紧而不污染缓存）。

        参数:
            client_id: 客户号，例如 `C004`。

        返回:
            该客户的 `ClientProfile` 深拷贝（`model_copy(deep=True)`）。

        异常:
            客户号不存在时抛 `KeyError`，消息中附带全部可选客户号。
        """
        if client_id not in self.clients:
            raise KeyError(f"未找到客户 {client_id}，可选：{', '.join(sorted(self.clients))}")
        return self.clients[client_id].model_copy(deep=True)

    def sample_clients(self) -> list[ClientProfile]:
        """评估样本客户（`eval_sample == true`），按客户号排序。

        返回:
            深拷贝后的客户列表；过滤条件为 `ClientProfile.eval_sample` 为真，
            用于评测集（不含专用于反例的客户）。

        副作用/异常:
            无；不存在的字段不会出现，因为 `eval_sample` 在 schemas 中有默认值。
        """
        return [self.client(cid) for cid in sorted(self.clients) if self.clients[cid].eval_sample]

    def all_clients(self) -> list[ClientProfile]:
        """全部样例客户（含反例客户），按客户号排序。

        返回:
            深拷贝后的全部客户列表；与 `sample_clients` 的差别是不按 `eval_sample` 过滤。
        """
        return [self.client(cid) for cid in sorted(self.clients)]

    def products_subset(self, product_ids: Iterable[str]) -> dict[str, Product]:
        """按产品号取子集（保持传入顺序的排序结果）。

        参数:
            product_ids: 产品号可迭代对象。

        返回:
            产品号 -> `Product` 深拷贝 的字典。

        注：实际实现为 `sorted(set(product_ids))`，即**先去重再按产品号升序**，
        并非保持传入顺序。

        异常:
            传入未知产品号时抛 `KeyError`。
        """
        return {pid: self.products[pid].model_copy(deep=True) for pid in sorted(set(product_ids))}

    def questionnaire_level(self, client_id: str) -> int | None:
        """由问卷作答折算风险等级（缺省返回 None）。

        参数:
            client_id: 客户号。

        返回:
            `score_to_level` 折算出的风险等级 int；该客户没有作答记录时为 None。
        """
        return score_to_level(self.questionnaire_score(client_id), self.questionnaire)

    def questionnaire_score(self, client_id: str) -> int | None:
        """问卷得分合计（缺省返回 None）。

        参数:
            client_id: 客户号。

        返回:
            该客户 `responses` 中每题得分之和（int）；无作答记录或作答为空时为 None。
        """
        responses = self.questionnaire.get("responses", {})
        answers = responses.get(client_id)
        if not answers:
            return None
        return int(sum(answers.values()))


def score_to_level(score: int | None, questionnaire: dict[str, Any]) -> int | None:
    """把问卷总分折算成风险等级（按 `level_bands` 分档）。

    参数:
        score: 问卷总分；为 None 时直接返回 None。
        questionnaire: 问卷字典，需含 `level_bands` 列表，每档有
            `min_score` / `max_score` / `level`。

    返回:
        命中的档位等级 int；总分未落入任何档（档位不连续或有空洞）时返回 None。

    说明:
        判定使用闭区间 `min_score <= score <= max_score`，按列表顺序取第一个命中的档位。
    """
    if score is None:
        return None
    for band in questionnaire.get("level_bands", []):
        if band["min_score"] <= score <= band["max_score"]:
            return int(band["level"])
    return None


def level_label(level: int | None, questionnaire: dict[str, Any]) -> str:
    """风险等级对应的中文标签。

    参数:
        level: 风险等级；为 None 表示未测评。
        questionnaire: 问卷字典，需含带 `level` / `label` 的 `level_bands`。

    返回:
        该等级的中文标签；`level` 为 None 时返回 `"未测评"`；在 `level_bands`
        中找不到对应等级时回退为 `f"R{level}"`。
    """
    if level is None:
        return "未测评"
    for band in questionnaire.get("level_bands", []):
        if int(band["level"]) == level:
            return str(band["label"])
    return f"R{level}"


def load_data(data_dir: Path | str | None = None) -> DataBundle:
    """装载样例数据并完成基础自洽校验。

    参数:
        data_dir: data 目录路径；为 None 时使用模块常量 `DATA_DIR`。

    返回:
        填充完毕的 `DataBundle`（clients/products 已转成 Pydantic 模型并按主键建索引）。

    异常:
        文件缺失或 JSON 非法时由 `_read_json` 抛出；clients.json 或 products.json
        存在重复主键时抛 `ValueError`。
    """
    base = Path(data_dir) if data_dir is not None else DATA_DIR
    clients_raw = _read_json(base / "clients.json")
    products_raw = _read_json(base / "products.json")
    questionnaire = _read_json(base / "questionnaire.json")
    stress_config = _read_json(base / "stress_scenarios.json")

    clients = {item["client_id"]: ClientProfile.model_validate(item) for item in clients_raw}
    products = {item["product_id"]: Product.model_validate(item) for item in products_raw}

    # 字典以主键去重，故原始条数减索引条数即为被覆盖的重复主键数量；
    # 重复会让"取到哪一个"变得不确定，直接拒绝装载而不是静默覆盖。
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
