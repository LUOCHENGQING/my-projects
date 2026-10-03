# 我的项目 · 金融 AI Agent 作品集

三个可离线复现的金融场景 AI Agent 项目，分别落在 **检索层 / 编排层 / 决策层**。
三者共用同一条工程原则：**确定性的部分绝不交给模型，模型只负责语言、权衡与解释**——
因为金融场景里，「可追溯」与「可拦截」比「答得聪明」更重要。

| 目录 | 项目 | 核心命题 | 关键机制 | 测试 |
| --- | --- | --- | --- | --- |
| [`fin-research-rag-engine/`](fin-research-rag-engine/) | **金融智研引擎（RAG）** | 知识从哪来、答案能不能溯源 | 父子块切分 + BM25 / 稠密 / 稀疏三路混合召回 + RRF 融合 + bge-reranker 重排 + 引用账本 | 267 passed |
| [`fin-research-multi-agent/`](fin-research-multi-agent/) | **金融投研多智能体** | 多步推理怎么分工、怎么防幻觉 | 5 Agent 职责隔离（认知步骤 + 职责冲突）+ 数字归工具 / 语言归模型 + 反思循环三重保险 + 四级权限工具层 | 80 passed |
| [`weal-advisory-multi-agent/`](weal-advisory-multi-agent/) | **财富管理投顾多智能体** | 带硬约束的决策怎么做、怎么可解释 | 硬约束求解（可行域）+ 22 条 block / warn 适当性闸门 + 反事实解释 + 情景压力测试 + 建议版本链 | 291 passed |

## 目录结构

```
my-projects/
├── fin-research-rag-engine/      # 金融智研引擎（RAG）：检索层
│   ├── src/                      # ingest / chunking / index / retrieve / answer / cache
│   ├── data/                     # 虚构金融资料（制度、产品说明书、案例、研报、尽调档案…）
│   ├── eval/                     # 评估脚本与报告
│   ├── tests/                    # 267 个用例
│   └── Dockerfile / docker-compose.yml
├── fin-research-multi-agent/     # 金融投研多智能体：编排层
│   ├── src/agents/               # Planner / Retriever / Analyst / RiskChecker / Writer
│   ├── src/rag/  src/tools/      # 混合检索、最小权限工具层
│   ├── data/                     # 虚构年报 / 季报摘要
│   ├── eval/                     # 四项指标 + 退出码门禁
│   └── tests/                    # 80 个用例
└── weal-advisory-multi-agent/    # 财富管理投顾多智能体：决策层
    ├── src/agents/               # 画像 / 筛选 / 组合构建 / 适当性复核 / 建议书
    ├── src/suitability/          # 22 条适当性规则 + 硬闸门
    ├── src/constraints.py        # 硬约束求解（纯函数 / 确定性 / 幂等）
    ├── data/                     # 虚构客户档案 / 产品要素 / 问卷 / 压力情景
    ├── eval/                     # 7 项指标 + 对抗探针
    └── tests/                    # 291 个用例
```

## 快速开始

每个子项目彼此独立，均可**在无 API Key 的 mock 模式下跑通全链路**：

```bash
cd <子项目目录>
pip install -r requirements.txt

python -m src.demo          # 跑一次完整链路（mock 模式，离线）
python -m pytest -q         # 全量测试
python eval/run_eval.py     # 评估报告（不达标以退出码 1 卡 CI）
```

真实模型调用只需在 `.env` 中填入兼容 OpenAI 协议的 Key（各项目 `.env.example` 有说明），不填也能完整演示。

## 共用工程原则

1. **数字与合规归代码**：比率计算、约束校验、适当性判定全部由确定性函数完成，模型不参与；
2. **结论必须可追溯**：引用编号、证据清单、快照版本链，任一环节断了都有测试拦住；
3. **越权在代码层拒绝**：工具权限分级校验，而不是写在提示词里「请你不要」；
4. **评估即门禁**：每次改动都要跑评测脚本，指标不达标直接退出码 1。

## 许可

本项目采用 [MIT License](LICENSE) 开源。

> 仓库内全部客户、产品、公司与文档均为**虚构示例数据**，仅用于工程演示，不构成任何投资建议。
