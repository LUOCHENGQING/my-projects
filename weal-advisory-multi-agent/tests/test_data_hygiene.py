"""数据合规红线与样例数据自洽性测试。

覆盖对象
--------
`data/*.json` 全部数据文件、`src/**/*.py` 全部源码、`README.md`，
以及 `src.dataset` 的装载与问卷折算（`load_data` / `score_to_level` / `DATA_DIR`）。

两类断言
--------
1. **合规红线（对抗路径）**：项目内不得出现任何真实机构、真实产品、真实指数
   或监管机构的具名 —— 用 `REAL_NAME_BLACKLIST` 对数据文件与源码做子串扫描。
   其中 `test_blacklist_itself_is_not_empty` 是元测试：若黑名单为空，
   所有扫描都会因为「无词可匹配」而假通过，所以必须先锁住黑名单的规模。
   README 缺失时跳过（而非失败），以免文档未生成时阻塞测试。
2. **数据自洽（正常路径 + 边界）**：主键唯一、评估样本恰好 5 位、
   必须保留一个「可行域为空」的反例客户、问卷得分与档案登记值一致、
   分档区间完整覆盖 0~100 分、产品与客户字段落在合理取值范围、
   样例客户被刻意设计成能命中各类剔除原因码（让 demo 有东西可拦）、
   压力情景至少覆盖利率/权益/信用利差三个维度、JSON 均为合法 UTF-8、
   源码中不存在 TODO 之类的占位标记。

参数化说明：`test_data_files_*` / `test_source_files_*` 的文件清单在
**模块导入期**由路径扫描生成，因此新增数据文件或源码文件会被自动纳入扫描范围。
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from src.constraints import screen_products
from src.dataset import DATA_DIR, load_data, score_to_level

PROJECT_ROOT = Path(__file__).resolve().parent.parent

#: 真实机构 / 产品 / 指数名称黑名单（仅用于本测试自动扫描，不出现在数据与源码中）
#: 匹配方式为朴素子串包含（不做分词）：宁可漏报也不误伤，命中即在断言消息里列出具体词
REAL_NAME_BLACKLIST: tuple[str, ...] = (
    # 银行
    "工商银行", "建设银行", "农业银行", "中国银行", "交通银行", "招商银行", "邮储银行",
    "中信银行", "浦发银行", "兴业银行", "民生银行", "光大银行", "平安银行", "华夏银行",
    "广发银行", "北京银行", "宁波银行", "江苏银行", "上海银行",
    # 券商
    "中信证券", "华泰证券", "国泰君安", "海通证券", "招商证券", "广发证券", "中金公司",
    "中信建投", "银河证券", "申万宏源", "东方证券", "国信证券", "兴业证券", "光大证券",
    # 公募基金
    "华夏基金", "易方达", "嘉实基金", "南方基金", "广发基金", "博时基金", "天弘基金",
    "富国基金", "汇添富", "银华基金", "中欧基金", "兴证全球", "景顺长城", "工银瑞信",
    "建信基金", "招商基金", "鹏华基金", "华安基金", "大成基金", "交银施罗德",
    # 资管 / 海外机构
    "贝莱德", "先锋领航", "富达", "道富", "瑞银", "高盛", "摩根士丹利", "摩根大通",
    "花旗", "汇丰", "桥水",
    # 评级机构
    "穆迪", "标普", "惠誉", "中诚信", "联合资信", "大公国际", "新世纪评级",
    # 互联网平台与产品
    "余额宝", "蚂蚁", "支付宝", "微信", "腾讯", "京东金融", "度小满",
    # 监管机构（本项目统一使用「监管要求」等泛称）
    "证监会", "银保监会", "中国人民银行", "人民银行", "国家金融监督管理总局", "基金业协会",
    # 真实指数
    "沪深300", "中证500", "中证800", "上证50", "创业板指", "纳斯达克", "标普500",
)


def _iter_source_files() -> list[Path]:
    """返回 `src` 下全部 Python 源码文件（排序，保证参数化用例顺序稳定）。"""
    return sorted((PROJECT_ROOT / "src").rglob("*.py"))


def _iter_data_files() -> list[Path]:
    """返回 `data` 目录下全部文件（排序，保证参数化用例顺序稳定）。"""
    return sorted(path for path in DATA_DIR.rglob("*") if path.is_file())


def test_blacklist_itself_is_not_empty():
    """元测试：黑名单必须足够大，否则后续扫描会因为「无词可匹配」而假通过。"""
    assert len(REAL_NAME_BLACKLIST) > 50


@pytest.mark.parametrize("path", [str(p) for p in _iter_data_files()], ids=lambda p: Path(p).name)
def test_data_files_contain_no_real_institution_names(path):
    """合规红线：每个数据文件都不得出现真实机构 / 产品 / 指数名称（逐文件参数化）。"""
    text = Path(path).read_text(encoding="utf-8")
    hits = [name for name in REAL_NAME_BLACKLIST if name in text]
    assert not hits, f"{path} 出现疑似真实机构/产品名：{hits}"


@pytest.mark.parametrize("path", [str(p) for p in _iter_source_files()], ids=lambda p: Path(p).name)
def test_source_files_contain_no_real_institution_names(path):
    """合规红线：每个源码文件都不得出现真实机构 / 产品 / 指数名称（逐文件参数化）。"""
    text = Path(path).read_text(encoding="utf-8")
    hits = [name for name in REAL_NAME_BLACKLIST if name in text]
    assert not hits, f"{path} 出现疑似真实机构/产品名：{hits}"


def test_readme_contains_no_real_institution_names():
    """合规红线：README 同样不得具名真实机构；文件不存在时跳过（不算失败）。"""
    readme = PROJECT_ROOT / "README.md"
    if not readme.exists():  # pragma: no cover - README 尚未生成时跳过
        pytest.skip("README 尚未生成")
    text = readme.read_text(encoding="utf-8")
    hits = [name for name in REAL_NAME_BLACKLIST if name in text]
    assert not hits, f"README 出现疑似真实机构/产品名：{hits}"


# ---------------------------------------------------------------------------
# 数据自洽
# ---------------------------------------------------------------------------
def test_data_files_exist():
    """契约：四份样例数据文件必须齐备（客户 / 产品 / 问卷 / 压力情景）。"""
    names = {path.name for path in _iter_data_files()}
    assert {"clients.json", "products.json", "questionnaire.json", "stress_scenarios.json"} <= names


def test_client_ids_and_product_ids_unique(data):
    """自洽：客户与产品主键唯一（`load_data` 在遇到重复主键时本就会直接报错）。"""
    assert len(data.clients) == len({client.client_id for client in data.all_clients()})
    assert len(data.products) == len({product.product_id for product in data.products.values()})


def test_exactly_five_evaluation_samples(data):
    """契约：评估样本客户恰好 5 位（评估脚本的指标口径依赖这一数量）。"""
    assert len(data.sample_clients()) == 5


def test_infeasible_example_client_exists(data):
    """必须保留一个"可行域为空"的反例客户，用于演示直接拒绝。"""
    # C006 的禁止项与流动性下限极端，导致候选池为空；它必须且只能是非评估样本
    client = data.client("C006")
    screening = screen_products(client, data.products, 0)
    assert screening.included == []
    assert client.eval_sample is False


def test_questionnaire_scores_match_client_profiles(data):
    """一致性：每位客户的问卷得分与档案登记值一致，且折算出的等级落在 R1–R5。"""
    for client in data.all_clients():
        score = data.questionnaire_score(client.client_id)
        assert score is not None, client.client_id
        assert score == client.risk_questionnaire_score, client.client_id
        level = score_to_level(score, data.questionnaire)
        assert level is not None
        assert 1 <= level <= 5


def test_questionnaire_bands_cover_full_range(data):
    """边界：分档区间必须完整覆盖 0~100 分，任何合法得分都能折算成等级、不留空档。"""
    bands = data.questionnaire["level_bands"]
    assert min(band["min_score"] for band in bands) == 0
    assert max(band["max_score"] for band in bands) == 100
    # 取两个端点 + R1–R5 各档内的代表分数，验证不存在「落在档位之间」的得分
    for score in (0, 25, 40, 55, 70, 85, 100):
        assert score_to_level(score, data.questionnaire) is not None


def test_products_have_sane_elements(data):
    """取值范围：产品要素（比例类 0~1、金额/收益率为正、等级 R1–R5）必须自洽，且名称以「示例」开头。"""
    for product in data.products.values():
        assert 0 <= product.liquidity_ratio <= 1
        assert product.min_investment > 0
        assert product.expected_return > 0
        assert product.volatility >= 0
        assert product.fee_rate >= 0
        assert 1 <= product.risk_level <= 5
        assert product.horizon_years >= 0
        # 名称前缀是「数据为虚构」的显式标记，便于人工复核时一眼识别
        assert product.name.startswith("示例")


def test_clients_have_sane_constraints(data):
    """取值范围：客户集中度上限与流动性下限落在合法区间，可投金额为正，且名称以「示例客户」开头。"""
    for client in data.all_clients():
        assert 0 < client.max_single_product_ratio <= 1
        assert 0 < client.max_single_class_ratio <= 1
        assert 0 < client.max_single_issuer_ratio <= 1
        assert 0 <= client.liquidity_floor_ratio <= 1
        assert client.investable_amount > 0
        assert client.display_name.startswith("示例客户")


def test_sample_clients_cover_intended_rule_hits(data):
    """样例客户必须故意设计成能被规则命中（让 demo 有东西可拦）。"""
    codes_by_client: dict[str, set[str]] = {}
    for client in data.all_clients():
        screening = screen_products(client, data.products, 0)
        codes_by_client[client.client_id] = {code for item in screening.excluded for code in item.reasons}
    # 先按客户聚合原因码再取并集：只要求全体样例数据合起来覆盖这些码，不要求单个客户全覆盖
    all_codes = set().union(*codes_by_client.values())
    for expected in ("C-RISK", "C-HORIZON", "C-EXPERIENCE", "C-PROHIBITED-CLASS", "C-CURRENCY", "C-QUALIFIED"):
        assert expected in all_codes, f"样例数据未覆盖剔除原因 {expected}"
    # C004 是演示"高龄 + 双录缺失 → 必须人工复核"这条链路的样板客户
    assert data.client("C004").is_elderly
    assert data.client("C004").dual_record_completed is False


def test_demo_client_ids_are_referenced_in_readme():
    """一致性：README 必须引用 demo 客户号，保证文档与数据不脱节。"""
    readme = PROJECT_ROOT / "README.md"
    if not readme.exists():  # pragma: no cover
        pytest.skip("README 尚未生成")
    text = readme.read_text(encoding="utf-8")
    assert "C001" in text


def test_stress_scenarios_cover_three_dimensions(data):
    """覆盖度：压力情景至少 3 个，且合计覆盖面包含利率、权益、信用利差三类冲击。"""
    scenarios = data.stress_config["scenarios"]
    assert len(scenarios) >= 3
    keys = set()
    for scenario in scenarios:
        keys.update(scenario["shocks"])
    assert {"rate", "equity", "credit_spread"} <= keys


def test_json_data_is_valid_utf8_json():
    """格式：所有数据文件都能以 UTF-8 解码并解析为合法 JSON（无 BOM、无注释、无尾逗号）。"""
    for path in _iter_data_files():
        if path.suffix == ".json":
            json.loads(path.read_text(encoding="utf-8"))


def test_no_todo_placeholders_in_source():
    """硬性要求：无 TODO 占位。"""
    # 匹配常见的"待实现"标记；命中任一即视为交付未完成
    pattern = re.compile(r"TODO|FIXME|XXX占位|pass\s*#\s*待实现")
    offenders: list[str] = []
    for path in _iter_source_files():
        text = path.read_text(encoding="utf-8")
        if pattern.search(text):
            offenders.append(path.name)
    assert not offenders, f"源码中存在占位标记：{offenders}"
