"""财富管理投顾多智能体（Wealth Advisory Multi-Agent）。

第三条技术路线：**约束驱动**。
先把客户硬约束求解成可行域，再让 Agent 在可行域内做多目标权衡与解释，
核心机制是「硬约束求解 + 适当性闸门 + 反事实解释 + 情景压力测试 + 建议版本链」。
"""

from __future__ import annotations

__version__ = "1.0.0"
__all__ = ["__version__"]
