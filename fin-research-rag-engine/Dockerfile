# ---------------------------------------------------------------------------
# 金融智研引擎（FinRAG-Engine）—— 容器镜像
#
# 两种用法：
#   1. 只跑核心链路（解析 / 切分 / 三路召回 / 重排 / 问答 / 评测 / 测试）
#      用 slim 基底即可，不装 OCR 与模型，镜像小、构建快。
#   2. 起服务（FastAPI 或 stdlib 兜底服务）
# ---------------------------------------------------------------------------
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONIOENCODING=utf-8 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# 先装依赖，利用镜像层缓存：只改代码时不必重装依赖
COPY requirements.txt ./
# 只装核心链路依赖；FastAPI / Redis / BGE-M3 / PaddleOCR 按需在部署时叠加。
# 引擎本身对它们全部做了降级处理，因此只装这几个也能完整跑通与评测。
RUN pip install --no-cache-dir "numpy>=1.24" "pydantic>=2.0" "pytest>=8.0"

COPY . .

# 构建期做一次自检：跑测试 + 跑评测（评测不达标会让构建失败，等于一道质量门）
RUN python -m pytest -q && python eval/run_eval.py --no-baseline

EXPOSE 8000

# 默认走 stdlib 兜底服务，镜像里没装 FastAPI 也能起
CMD ["python", "-m", "src.serve", "--host", "0.0.0.0", "--port", "8000"]
