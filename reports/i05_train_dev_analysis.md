# I-05 Train/Dev 中尺度分析

本文件只报告冻结的 Train/Dev 结果；Calibration 与 Test 未读取。

## 主集成与基线

- Train 单父样本：2997；Dev 单父样本：1000。
- Logistic Regression 宏 F1：0.2696。
- 直接特征 MLP 宏 F1：0.2561。
- generic-only 水印 AUROC：0.6909。
- Mahalanobis 宏 F1：0.3204。

## 三完整模型稳定性

- generic-only AUROC：0.6749 ± 0.0068。
- 五类宏 F1：0.1347 ± 0.0528。

## 阶段边界

`registered-only`、线性注册融合和 MarkLLM 登记路线需要冻结的 Calibration，按预注册顺序留到 I-06/I-07；此处不以 Dev 代替 Calibration。
