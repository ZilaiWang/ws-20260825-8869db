# 开发工作流

## 流程

1. **领取 Issue** — 在 Gitee Issues 中确认任务。
2. **拉取最新代码** — `git pull origin master`
3. **创建短期分支** — `git checkout -b feat/my-task`
4. **开发和本地测试** — `pytest` 通过后再提交。
5. **提交 PR** — push 分支，在 Gitee 创建 Pull Request。
6. **至少一人审核** — reviewer 确认后合并。
7. **删除分支** — 合并后删除远程和本地分支。

## Git 命令示例

```bash
# 开始工作
git pull origin master
git checkout -b feat/eval-add-iou-metric

# 开发
# ... 写代码 ...
git add -A
git commit -m "feat(eval): add per-class IoU metric"
git push origin feat/eval-add-iou-metric

# 在 Gitee 上创建 PR，等待审核

# 合并后清理
git checkout master
git pull origin master
git branch -d feat/eval-add-iou-metric
```

## 分支命名

| 前缀 | 用途 |
|------|------|
| `feat/` | 新功能 |
| `fix/` | 修复 |
| `exp/` | 实验 |
| `docs/` | 文档 |
| `refactor/` | 重构 |
| `chore/` | 杂务 |

## 注意事项

- **不要直接 push master**
- 一个分支只做一个任务
- 正式实验必须从已提交的 commit 运行
