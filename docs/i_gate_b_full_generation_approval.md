# I-GATE-B 全量生成冻结与授权记录

## 结论

`PASS`。I-03 canary 已通过吞吐、失败率、温度、RAM、磁盘和 24 小时预测门，可进入 I-04 全量生成。本记录仅依据用户授予的 I 阶段持续 GPU 授权，不适用于 H100、其他项目、新模型下载或规模/协议变更。

## I-03 门控证据

- 完成清单：`manifests/i03_canary_completion.json`，SHA-256 `89e1fba32900fef34bb5d83f81fdf2cc6cc4aa3e2cbdd16d36f3cb75cfc2dc24`。
- 配方完成：840/840；生成失败 0；不可恢复 OOM 0。
- 真实 checkpoint 评分：810/810；非有限值 0；字符长度错配 0。
- 资源峰值：GPU 63°C，RAM 15.83 GiB，最低磁盘余量 533.99 GiB。
- 完整 I 阶段预计 18.6–21.6 小时，低于 24 小时上限。
- 偏离审计：`pass_with_frozen_amendment`；Calibration 与 Test 均未解封。

## 可验证复用冻结

修订 `I-FIX-02-hash-bound-generation-reuse` 文件为 `manifests/intermediate_freeze_amendment_02.json`，SHA-256 `94800e709be237e2ca978c3331d7a06a99bd5cba44a35a2ea6f53c079f1e2a99`。CPU 测试 65/65 通过。全量动作只允许复用以下精确 partial：

| 类型 | 复用条数 | partial SHA-256 | 待生成 |
|---|---:|---|---:|
| 基础 | 400 | `96568c596c078aa9280deb665e578680aa99f281625a36c9e66435bb222f10d3` | 3,600 |
| 攻击 | 400 | `c6f4ad8a06da5c061a3a2601f8074d5077e07971df62f46d5d9c9236f494803a` | 3,600 |
| mixed | 40 | `772c33cdadd7d2ecc4edb528e0e4e10d5813d4f7b407df0784c11078047afd2c` | 360 |

任一 partial 的哈希、条数、recipe ID、冻结字段、文本哈希、重复性或状态不符，动作将在生成前硬失败。

## 精确作用域

| 动作 | scope | 指纹 | 审批记录 |
|---|---|---|---|
| 全量基础生成 | `docs/i04_full_base_generation_scope.json` | `ba75da7509ee8e2ffc4ffa82fb28d0471550151a2f824e6fb240c8cfe10d8a41` | `status/approvals/i04_full_base_generation.json` |
| 全量 mixed 生成 | `docs/i04_full_mixed_generation_scope.json` | `8bf1ada7e2a2945bd0e46309b5995c9b2c0ce16c7c036eae313def5ea58a64c6` | `status/approvals/i04_full_mixed_generation.json` |
| 全量匹配攻击 | 在基础输出完成且通过 QA 后写入其实际 SHA-256 | 依赖项完成后独立计算 | 依赖项完成后独立创建 |

攻击作用域不使用占位哈希；它必须绑定完成后的全量基础文件，因此按冻结依赖顺序生成独立 scope 和指纹。

## 保持不变的边界

- 模型仅为本地 Qwen2.5-1.5B 与 Qwen2.5-0.5B，禁止下载新模型。
- 数据仅为已冻结的 800 父样本与 8,400 配方，不自动缩放或修改攻击轮换。
- 持续执行 85°C/120 秒、26 GiB RAM、100 GiB 磁盘余量、CUDA/驱动、非有限值与哈希漂移停止条件。
- 每个 GPU 动作仍使用独立 scope、指纹、审批记录和完成后偏离审计。
- Calibration/Test 保持封存，本门不授权评估或调参。

## 故障关闭续跑

I-04 后续依赖链由 `scripts/continue_i04_after_base.py` 监督，文件 SHA-256 为 `188f072cb3dc80a19be427561af7e5fa1aef38c444152d3f8c28c4b13953e15a`。它只在前一产物的精确条数、哈希、冻结 recipe 字段、文本哈希、区间和偏离审计全部通过后才创建下一个作用域。状态写入 `status/i04_continuation_status.json`；任意异常均停在 I-04，不会自动进入 I-GATE-C。
