# R1-2 飞机类中心实验结果（2026-08-14）

## 1. 结论

ECC 启发的训练期类中心角度约束未通过主门禁，停止该路线，不再扫描 loss weight、margin
或 momentum。正式工作点继续保留 R1-1 的 proposal-domain CE + full D4。

核心原因不是实现失败：三折训练、D4 推理、outer cross-fit、冻结 C2 和 ship/vehicle
旁路均通过。失败是科学结果——训练域动态中心使类内特征更紧，但在来源隔离 OOF 上
破坏了宏平均和错误发现率。

## 2. 完整性

- 回传包 SHA256：
  `2ed829f801f048bc3020e1f8f64ccc82e64663a1de9b1d5bf3e9cfdd8c5a3f76`；
- 本地路径：`outputs/R1-2-AIRCRAFT-CLASS-CENTER-return/`；
- 18 个小型产物，回传包不含 checkpoint；
- 训练每折固定 5 epoch，约 98--105 秒；
- train accuracy 99.96%--99.98%，最终 positive cosine 约 0.951，center loss 约
  0.076，证明约束确实生效；
- ship/vehicle 四项指标最大差为 0；
- `decision.json: primary_gate_passed=false`。

## 3. 主条件：class-center identity vs CE identity

| 指标 | CE identity | class-center identity | 差值 |
|---|---:|---:|---:|
| Overall Recall | 0.926480 | 0.922945 | **-0.003535** |
| Overall FDR | 0.150541 | 0.151627 | **+0.001086** |
| Overall macro Recall | 0.886262 | 0.880333 | **-0.005929** |
| Overall macro FDR | 0.203530 | 0.204149 | **+0.000619** |
| Aircraft macro Recall | 0.934442 | 0.927031 | **-0.007411** |
| Aircraft macro FDR | 0.132540 | 0.133314 | **+0.000774** |
| Aircraft pooled Recall | 0.947112 | 0.942966 | **-0.004146** |
| Aircraft pooled FDR | 0.115709 | 0.116900 | **+0.001192** |

同对象转移为 `new_tp=26, broken_tp=100, net_tp=-74`，同时增加 16 个 FP。主条件在
Recall、FDR、宏平均和 paired TP 上全线更差，不存在保留理由。

## 4. D4 条件与当前最强工作点比较

class-center + D4 的 Overall Recall 为 0.931400，看似高于 CE + D4 的 0.930110；但这是
头部样本量加权结果。直接比较当前最强 CE + D4：

| 指标 | CE + D4 | class-center + D4 | 差值 |
|---|---:|---:|---:|
| Overall Recall | 0.930110 | 0.931400 | +0.001290（+27 TP） |
| Overall FDR | 0.146015 | 0.148565 | **+0.002550（+73 FP）** |
| Overall macro Recall | 0.891217 | 0.888690 | **-0.002527** |
| Aircraft macro Recall | 0.940637 | 0.937478 | **-0.003159** |
| Aircraft pooled Recall | 0.951370 | 0.952883 | +0.001513 |
| Aircraft pooled FDR | 0.110244 | 0.113474 | **+0.003230** |

类中心仅换取少量 pooled Recall，却损害宏平均并引入更多 FP。这与小样本、类别等权的
官方排名目标相反，因此不能用“总体 Recall 更高”将其准入。

## 5. 机理解释与后续影响

1. P03/CE 模型在训练 proposal 上已经接近 100% accuracy；继续将特征拉向训练域均值，
   主要压缩的是已容易样本，而不是补充跨来源判别信息。
2. fold0 是机场/来源偏移最强的压力折，早期正提议诊断也显示该折退化最大；动态中心随
   高频来源主导，会放大域内紧致、削弱域外边界。
3. pooled Recall 上升而 macro Recall 下降，说明额外 TP 更多来自头部类，尾类/困难类承担
   了代价。
4. 这不是“对比学习都无效”的证据。它只否定基于训练样本视觉均值的类别中心。下一步
   改为 class-level 物理属性辅助监督，用发动机数、翼型、任务角色等跨来源不变结构共享
   梯度，而不再建立视觉中心。

## 6. 决策

- class-center：停止；
- R1-1 CE + D4：继续作为飞机对象头最强工作点；
- R1-4 structured attribute：允许执行，必须以 CE identity 为主参考，并额外直接比较
  CE + D4；
- 若 R1-4 失败，不再在现有 OOF 上扫描辅助损失权重，转向 D4 ensemble-to-single-view
  distillation 或真实系统成本测量。
