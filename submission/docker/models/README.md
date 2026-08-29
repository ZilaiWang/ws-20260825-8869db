# 模型目录

Git 中不保存权重。不要手工向本目录提交 `.pt/.pth/.onnx/.engine`。

运行 `scripts/build_submission_delivery.py` 时传入正式权重，脚本会将其复制为
`models/model.pt`，核验 SHA256，并把完整锁写入 `BUILD_MANIFEST.json`。

当前预测评默认资产为 M1 formal-CV3 fold0 fixed-epoch-160 `last.pt`：

- 本地来源：`artifacts/M1-CV3-OOF/last.pt/fold0_last.pt`
- SHA256：`d403ca0d5f760f8d7271af33c467426a3bd8ce8095b6d13e5dc63a6d8e19501d`
- 角色：仅用于 8 月 27–29 日预测评的 Docker 流程验证，不自动成为正式最终模型。
