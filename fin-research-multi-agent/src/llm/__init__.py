"""LLM 接入层（架构中的「模型访问层／LLM 网关」）。

    client   OpenAI 兼容客户端；无 key 自动降级为确定性 mock 大脑
    prompts  五个 Agent 各自的 system prompt 与 user prompt 构造
    mock     确定性规则大脑（离线可跑，输出结构与真实 LLM 完全一致）

设计原则：**数字由工具产生，语言由模型产生**。
LLM 只负责选路、组织语言、写判断；所有数值一律来自 tools，且调用方会对 LLM 的
结构化输出做二次校验（引用/指标必须真实存在），这在 mock 与真机模式下走的是同一条路径。

层次与职责
----------
本层被夹在「agents」（上层调用方）与「外部推理服务」（下层）之间：
* 向上只暴露一个统一入口 `LLMClient.chat(task, payload)`，任务名固定为
  `plan` / `retrieve` / `analyze` / `risk_review` / `write`；
* 向下要么发 HTTP（真实 OpenAI 兼容服务），要么走本地确定性规则（mock）；
* 本层**不做任何数值计算**，也不决定业务结论，只负责「把 payload 变成结构化 JSON」。

对外关键对象
------------
* `LLMClient`       统一入口，按需在真机／mock 之间切换，并记录调用计数
* `LLMResponse`     一次调用的完整结果（`data` / `mocked` / `degraded` / digest / usage）
* `AGENT_PROMPTS`   `{"planner"|"retriever"|"analyst"|"risk_checker"|"writer"}` -> system prompt
* `build_user_prompt(task, payload)` -> `str`，把结构化 payload 渲染成 user prompt

主要输入输出
------------
输入：任务名 + 结构化 payload（`Dict[str, Any]`）。
输出：`LLMResponse`，其 `data` 是与 mock **完全同构**的 dict，调用方无需区分模式。

被谁调用
--------
* `src/orchestrator.py` 装配流水线时创建实例（`self.llm = LLMClient(self.config)`）；
* `src/agents/base.py` 通过 `AgentContext.llm` 下发给五个 Agent，并用
  `response.to_trace()` 把调用摘要写进 trace；
* `src/demo.py`、`eval/run_eval.py`、`tests/` 通过 orchestrator 间接走同一条路径。

离线保证
--------
本包在导入时**不联网、不读 API Key**：`client.py` 里的 `openai` 客户端与凭据都是
延迟到真正发起调用时才创建，因此无网络无密钥的环境也能 `import src.llm` 成功。
"""

from __future__ import annotations

# 只在这里做聚合导出，让上层写 `from src.llm import LLMClient` 即可；
# 子模块路径（src.llm.client / src.llm.mock）仍可被直接导入，供测试单独使用。
from .client import LLMClient, LLMResponse
from .prompts import AGENT_PROMPTS, build_user_prompt

# 对外承诺的公共符号；新增公共 API 时同步这里，便于静态检查发现未导出项
__all__ = ["LLMClient", "LLMResponse", "AGENT_PROMPTS", "build_user_prompt"]
