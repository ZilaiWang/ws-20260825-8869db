# 数据目录

## 规则

1. **原始数据不得提交 Git**，通过 `.gitignore` 已排除。
2. 使用 `configs/local.yaml` 指定本地数据路径。
3. 只允许脱敏统计结果放入 `reports/data/`。
4. 测试集图像禁止放入仓库。

## 推荐结构

```
data/
├── raw/          # 原始数据（不上传）
├── processed/    # 处理后的中间产物（不上传）
└── splits/       # 训练/验证划分 manifest
```

## 数据版本

使用 manifest 和 checksum 管理数据版本，详见 `src/rsdet/data/manifests.py`。
