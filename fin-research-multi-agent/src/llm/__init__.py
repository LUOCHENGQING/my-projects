"""LLM 接入层。

    client   OpenAI 兼容客户端；无 key 自动降级为确定性 mock 大脑
    prompts  五个 Agent 各自的 system prompt 与 user prompt 构造
    mock     确定性规则大脑（离线可跑，输出结构与真实 LLM 完全一致）

设计原则：**数字由工具产生，语言由模型产生**。
LLM 只负责选路、组织语言、写判断；所有数值一律来自 tools，且调用方会对 LLM 的
结构化输出做二次校验（引用/指标必须真实存在），这在 mock 与真机模式下走的是同一条路径。
"""

from __future__ import annotations

from .client import LLMClient, LLMResponse
from .prompts import AGENT_PROMPTS, build_user_prompt

__all__ = ["LLMClient", "LLMResponse", "AGENT_PROMPTS", "build_user_prompt"]
