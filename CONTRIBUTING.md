# CONTRIBUTING.md

## 分支命名

| 前缀 | 用途 | 示例 |
|------|------|------|
| `feat/` | 新功能 | `feat(tiling): add overlap-aware slicing` |
| `fix/` | 修复 | `fix(metric): count duplicate as FP` |
| `exp/` | 实验 | `exp(vehicle): test high-res feature` |
| `docs/` | 文档 | `docs(readme): update setup guide` |
| `refactor/` | 重构 | `refactor(eval): extract iou_util` |
| `chore/` | 杂务 | `chore: update .gitignore` |

## 规则

1. 一个分支只做一件事。
2. 不直接 push main。
3. PR 至少一人审核。
4. 正式实验必须从已提交 commit 运行。

## PR 模板

- **目标**: 做了什么
- **改动**: 具体修改
- **运行**: 如何运行
- **测试**: 测试结果
- **影响**: 对精度/速度/显存的影响
- **风险**: 已知问题

## 合并检查清单

- [ ] 测试通过 (`pytest`)
- [ ] 无个人路径硬编码
- [ ] 无数据/权重文件
- [ ] `git diff` 已检查
