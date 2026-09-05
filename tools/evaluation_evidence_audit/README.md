# Experiment Evidence Audit / 实验证据审计

这是一套**与任务类别无关、只读、离线**的实验元数据工具。它不执行模型、不读取图像、不反序列化权重，不搜索阈值，不选择候选，不修改训练、推理或部署系统。

用途：回答“这份结果是否与所声明的文件、训练血缘、数据独立性和测试覆盖一致”。它不能回答“哪个方法精度更高”“能否提升正式分数”“是否可以正式部署”。

## 已实际验证的范围

Python 标准库实现。当前环境已运行 46 项 unittest，全部通过、无跳过；四份合成命令行示例按预期返回。没有执行任何真实模型、GPU、用户数据或服务器集成测试。不要把这些测试数量与任何项目的测试数量相加。

建议使用 Python 3.10 或以上。无需 pip 安装依赖。

```bash
cd evaluation_evidence_audit_20260905
python -m unittest discover -s tests -v

# 使用新的输出路径；包内已经有示例报告，工具不会覆盖已有文件。
python -m evidence_audit \
  --manifest examples/declared_clean.json \
  --root examples \
  --out local_clean_check.json

# 这个例子有间接数据接触，预期退出码为 2，且仍输出审计报告。
python -m evidence_audit \
  --manifest examples/indirect_overlap.json \
  --root examples \
  --out local_overlap_check.json
```

返回码 0：没有发现所声明检查中的错误，不代表精度认证。返回码 2：输入无效，或存在与声明冲突的证据。加 `--fail-on-warning` 时，警告也返回 2。

## 输入结构

参考 `examples/declared_clean.json`。顶层字段如下：

|字段|内容|
|---|---|
|`schema_version`|固定整数 1|
|`datasets`|数据集 ID、来源组列表、用途、是否已经查看|
|`nodes`|训练、预测、选择、查看或打包节点及依赖关系|
|`claims`|候选节点、评估数据集与声称的证据等级|
|`artifacts`|相对文件路径及已知 SHA256|
|`comparisons`|两个文件的字节相等或严格 JSON 结构相等|
|`signature_checks`|两份训练设置中明确要求比较的字段|
|`test_coverage`|应覆盖和已覆盖的测试分支 ID|
|`notes`|可选说明|

### datasets

```json
{
  "id": "holdout",
  "groups": ["source-group-001", "source-group-002"],
  "role": "confirmation",
  "disclosure": "untouched"
}
```

`role` 取 `training / development / confirmation / diagnostic`。
`disclosure` 取 `untouched / inspected / unknown`。

来源组标识必须全局一致：同一原始来源生成的不同裁块、变体、派生数据不能另取独立组 ID。工具不会从图像自动发现近重复，也无法核实分组是否正确。

### nodes

```json
{
  "id": "procedure",
  "kind": "select",
  "parents": ["model", "calibration_predictions"],
  "exposure_datasets": ["development_set"],
  "lineage_complete": true
}
```

`parents` 必须列出所有影响该节点的上游资产，包括初始化、教师、伪标签来源、校准预测生成模型、人工选择所查看的数据。上游预测模型不是只列一个文件名就够，必须追到其训练依赖。

`exposure_datasets` 记录该节点直接接触的数据：不仅包括梯度训练，也包括参数选择、类别子集选择、读结果后拒绝或接受方案。模型预测未标注样本也是数据接触；在声称严格未见来源的确认中应如实记录。无标签传导设置并非一律无效，但不是这个工具要认证的严格未接触设置。

`lineage_complete=true` 只能在所有相关上游依赖确实登记完整时填写。拿不准填 false。工具无法证明用户填写的 true 是真实的。

### claims

```json
{
  "id": "claim-1",
  "candidate_node": "procedure",
  "evaluation_dataset": "holdout",
  "claimed_role": "independent_confirmation"
}
```

等级为 `independent_confirmation / development / diagnostic`。若用于形成方案的数据已经参与训练或选择，应如实降级，不要删掉相关 parent 来消除报错。

输出状态：

- `CONTRADICTED`：声明的候选及其祖先接触过评估来源组。
- `UNKNOWN`：血缘不完整或数据查看历史不明。
- `NOT_AN_UNTOUCHED_CONFIRMATION_SET`：没有找到记录中的来源重叠，但该数据集已被查看或不是确认用途。
- `CONSISTENT_WITH_DECLARED_METADATA_ONLY`：仅与声明的元数据相容，**不是独立性的证明**。

### artifacts / comparisons

所有文件必须在 `--root` 内，以相对路径给出。禁止路径穿越、绝对路径和指向根目录外的符号链接。不加载 pickle/checkpoint；模型文件只计算字节 SHA。

没有此前冻结的 SHA 时会报告 `NO_PREVIOUS_HASH_ATTESTATION`。现在算出的 SHA 只能冻结现在的字节，不能证明此前使用的就是这些字节。

`bytes` 比较包括全部字节。`json_exact` 不受对象键顺序或排版影响，但保留数组顺序、重复项、数值类型和所有字段；不自动忽略时间戳，不将 1 与 true 混为一谈，不将缺失字段当成 null，不做浮点近似。

若需要专门比较语义输出，应由你们已有的序列化合同导出固定结构的报告，再交给本工具。不要随意删除可能影响含义的字段后声称“全量一致”。

### signature_checks

字段缺失会标记 UNKNOWN，不把“两边都没有记录”当成等价。训练阶段顺序、每阶段轮数、初始化链、优化器/EMA、增强日程、有效 batch、BN 设置等，由具体研究的已冻结合同决定需要哪些字段；本工具不推荐训练配方。

### test_coverage

只检查声明的分支 ID 是否覆盖。它不是自动代码覆盖率，不验证测试执行真伪。测试数量与覆盖分支数量是不同指标；错误路径没有被运行，不能由一千个普通路径测试替代。

## 四个合成示例

`declared_clean.json`：声明上相容，无独立认证。

`indirect_overlap.json`：候选的直接训练数据与确认集不同，但用于选择程序的预测来自接触过确认来源的另一模型。工具输出完整依赖路径。

`post_selection_reuse.json`：看完确认结果后重新选择子集，再把原确认结果称作独立确认。工具报告冲突。

`unknown_and_missing_coverage.json`：训练来源未记全、训练阶段字段缺失、测试分支遗漏，输出 UNKNOWN 与覆盖警告。

## 限制

1. 不知道未登记的人工决策和数据接触。
2. 不证明分组正确或不存在近重复。
3. 不计算模型性能、显著性或正式成绩。
4. 文件一致性只对给定文件成立，不是对所有未来输入的证明。
5. 不自动决定发布、部署、提交或追加实验。
6. 报告不能覆盖已有文件；并行进程应使用各自独立输出路径。

## 理论参考

Cawley & Talbot, *On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation*, JMLR 11, 2079–2107, 2010。

Varma & Simon, *Bias in error estimation when using cross-validation for model selection*, BMC Bioinformatics 7:91, 2006, DOI: 10.1186/1471-2105-7-91。

Bengio & Grandvalet, *No Unbiased Estimator of the Variance of K-Fold Cross-Validation*, JMLR 5, 1089–1105, 2004。
