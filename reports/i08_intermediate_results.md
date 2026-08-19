# RTX 5060 中尺度实验结果

本报告是 I 阶段中尺度结果，只用于估计效应量、失败率、资源与 H100 方案，不作为论文最终结论。

## 数据与执行完整性

- 原始 I-04：8,400 条冻结配方，8,389 成功、11 条显式失败。
- I-FIX-05 经显式协议修订恢复精确冻结 Calibration parent 后：8,391 成功、9 条显式失败。
- 修复只改变目标 Calibration clean/attacked-clean 两条记录；Train/Dev 与 Test 行逐条保持不变。
- Test 文档：2096；独立 Test 父样本：200。
- 当前工作区实验产物占用：34.75 GiB。

## 主结果（A+B 授权）

- 五类宏 F1：0.0655。
- OSCR：0.0000。
- 字符 IoU：0.1932。
- 事件 F1：0.1947。
- 父样本级 clean FWER：48/200 = 0.2400；95% Clopper–Pearson 上界 0.2949。

## 冻结基线

- Logistic Regression 宏 F1：0.2728。
- direct-feature MLP 宏 F1：0.2329。
- generic-only AUROC：0.6359。
- registered-only AUROC：0.7203。
- 线性证据融合 AUROC：0.7314。

## LOFO

- kgw: 宏 F1 0.1263，父级 FWER 0.2500。
- unigram: 宏 F1 0.1272，父级 FWER 0.2450。
- unbiased: 宏 F1 0.1293，父级 FWER 0.2450。
- synthid: 宏 F1 0.1355，父级 FWER 0.2500。

## 失败样本

最终保留 9 条显式失败，逐条见 `reports/i08_failed_samples.csv`；没有静默丢失。

## 结论边界

无论性能是否达到预设目标，本报告均保留完整结果。H100 正式实验不得把这里的 Test 当作新的 Dev。
