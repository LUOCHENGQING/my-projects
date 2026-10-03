"""数据合规红线测试：不含任何真实机构/产品名称，样例数据自洽。"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from src.constraints import screen_products
from src.dataset import DATA_DIR, load_data, score_to_level

PROJECT_ROOT = Path(__file__).resolve().parent.parent

#: 真实机构 / 产品 / 指数名称黑名单（仅用于本测试自动扫描，不出现在数据与源码中）
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
    """src 下全部 Python 源码。"""
    return sorted((PROJECT_ROOT / "src").rglob("*.py"))


def _iter_data_files() -> list[Path]:
    """data 下全部数据文件。"""
    return sorted(path for path in DATA_DIR.rglob("*") if path.is_file())


def test_blacklist_itself_is_not_empty():
    assert len(REAL_NAME_BLACKLIST) > 50


@pytest.mark.parametrize("path", [str(p) for p in _iter_data_files()], ids=lambda p: Path(p).name)
def test_data_files_contain_no_real_institution_names(path):
    text = Path(path).read_text(encoding="utf-8")
    hits = [name for name in REAL_NAME_BLACKLIST if name in text]
    assert not hits, f"{path} 出现疑似真实机构/产品名：{hits}"


@pytest.mark.parametrize("path", [str(p) for p in _iter_source_files()], ids=lambda p: Path(p).name)
def test_source_files_contain_no_real_institution_names(path):
    text = Path(path).read_text(encoding="utf-8")
    hits = [name for name in REAL_NAME_BLACKLIST if name in text]
    assert not hits, f"{path} 出现疑似真实机构/产品名：{hits}"


def test_readme_contains_no_real_institution_names():
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
    names = {path.name for path in _iter_data_files()}
    assert {"clients.json", "products.json", "questionnaire.json", "stress_scenarios.json"} <= names


def test_client_ids_and_product_ids_unique(data):
    assert len(data.clients) == len({client.client_id for client in data.all_clients()})
    assert len(data.products) == len({product.product_id for product in data.products.values()})


def test_exactly_five_evaluation_samples(data):
    assert len(data.sample_clients()) == 5


def test_infeasible_example_client_exists(data):
    """必须保留一个"可行域为空"的反例客户，用于演示直接拒绝。"""
    client = data.client("C006")
    screening = screen_products(client, data.products, 0)
    assert screening.included == []
    assert client.eval_sample is False


def test_questionnaire_scores_match_client_profiles(data):
    for client in data.all_clients():
        score = data.questionnaire_score(client.client_id)
        assert score is not None, client.client_id
        assert score == client.risk_questionnaire_score, client.client_id
        level = score_to_level(score, data.questionnaire)
        assert level is not None
        assert 1 <= level <= 5


def test_questionnaire_bands_cover_full_range(data):
    bands = data.questionnaire["level_bands"]
    assert min(band["min_score"] for band in bands) == 0
    assert max(band["max_score"] for band in bands) == 100
    for score in (0, 25, 40, 55, 70, 85, 100):
        assert score_to_level(score, data.questionnaire) is not None


def test_products_have_sane_elements(data):
    for product in data.products.values():
        assert 0 <= product.liquidity_ratio <= 1
        assert product.min_investment > 0
        assert product.expected_return > 0
        assert product.volatility >= 0
        assert product.fee_rate >= 0
        assert 1 <= product.risk_level <= 5
        assert product.horizon_years >= 0
        assert product.name.startswith("示例")


def test_clients_have_sane_constraints(data):
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
    all_codes = set().union(*codes_by_client.values())
    for expected in ("C-RISK", "C-HORIZON", "C-EXPERIENCE", "C-PROHIBITED-CLASS", "C-CURRENCY", "C-QUALIFIED"):
        assert expected in all_codes, f"样例数据未覆盖剔除原因 {expected}"
    assert data.client("C004").is_elderly
    assert data.client("C004").dual_record_completed is False


def test_demo_client_ids_are_referenced_in_readme():
    readme = PROJECT_ROOT / "README.md"
    if not readme.exists():  # pragma: no cover
        pytest.skip("README 尚未生成")
    text = readme.read_text(encoding="utf-8")
    assert "C001" in text


def test_stress_scenarios_cover_three_dimensions(data):
    scenarios = data.stress_config["scenarios"]
    assert len(scenarios) >= 3
    keys = set()
    for scenario in scenarios:
        keys.update(scenario["shocks"])
    assert {"rate", "equity", "credit_spread"} <= keys


def test_json_data_is_valid_utf8_json():
    for path in _iter_data_files():
        if path.suffix == ".json":
            json.loads(path.read_text(encoding="utf-8"))


def test_no_todo_placeholders_in_source():
    """硬性要求：无 TODO 占位。"""
    pattern = re.compile(r"TODO|FIXME|XXX占位|pass\s*#\s*待实现")
    offenders: list[str] = []
    for path in _iter_source_files():
        text = path.read_text(encoding="utf-8")
        if pattern.search(text):
            offenders.append(path.name)
    assert not offenders, f"源码中存在占位标记：{offenders}"
