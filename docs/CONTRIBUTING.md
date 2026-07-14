# 协作规范

## 日常流程

```bash
git checkout master
git pull origin master
git checkout -b feat/short-task-name
# 修改并测试
git add <files>
git commit -m "feat(tiling): add overlap-aware slicing"
git push -u origin feat/short-task-name
```

在 Gitee 创建 PR，至少一人审核后合并。不要直接 push `master`。

## 分支和提交

分支前缀：`feat/`、`fix/`、`exp/`、`docs/`、`refactor/`、`chore/`。一个分支只处理一个主要任务。

提交示例：

- `feat(tiling): add overlap-aware slicing`
- `fix(metric): count duplicate detections as false positives`
- `exp(vehicle): test high-resolution feature layer`

## PR 必填

- 目标和关联 Issue
- 主要改动与运行方法
- 测试结果
- 数据版本、配置和 commit
- 对 Recall、FDR、速度和显存的影响
- 已知风险和回滚方法

## 合并前

- [ ] `python -m compileall -q src`
- [ ] `python -m pytest -q`
- [ ] `python -m ruff check .`
- [ ] 无数据、权重、密钥和个人绝对路径
- [ ] 文档与公共接口同步更新
- [ ] `git diff` 已人工检查
