"""测试包：让 `tests` 成为可导入的包。

存在意义
--------
把 `tests/` 标记为包，使 `from tests.helpers import make_client` 这类
**绝对导入**在任意工作目录下都能解析（与 pytest.ini 的 `pythonpath = .`
以及 conftest.py 里的 `sys.path` 注入形成双保险）。

边界约定
--------
本包只放测试代码与测试辅助代码，不含任何业务逻辑；被测对象全部来自 `src/`
（以及评估脚本所在的 `eval/`）。
"""
