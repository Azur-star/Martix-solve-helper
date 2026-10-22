# 示例输入输出

本目录包含矩阵计算器的示例输入和预期输出，便于评阅者复现和验证。

## 文件说明

- `sample_input.txt` — 示例输入与预期输出
- `demo_session.txt` — （待补充）完整的交互式会话记录

## 如何复现

```bash
cd MatrixCalculator
python main.py
```

然后按照 `sample_input.txt` 中的示例输入矩阵和命令。

## 自动化测试

```bash
python -m unittest discover tests
```

所有测试应全部通过（32 个测试用例）。
