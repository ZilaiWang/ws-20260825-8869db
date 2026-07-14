# PR 模板

```markdown
**关联 Issue**: #

**目标**: 做了什么

**主要修改**:
- 文件1: 改动
- 文件2: 改动

**运行命令**:
```
python scripts/xxx.py --config configs/xxx.yaml
```

**测试结果**:
```
pytest output
```

**数据版本**: (如涉及)

**配置**: (关键参数)

**对指标的影响**:
| 指标 | Before | After |
|------|--------|-------|
| Recall | | |
| FDR | | |
| Runtime | | |
| VRAM | | |

**已知风险**:

**回滚方式**: revert commit

**检查清单**:
- [ ] pytest 通过
- [ ] 无个人路径 / 无数据文件
- [ ] git diff 已检查
```
