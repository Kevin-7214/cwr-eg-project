# I-GATE-A：80 父样本 canary 分阶段审批

状态：已由用户于 2026-08-14 授予仅限 `I` 阶段的持续 GPU 授权。旧 `G` 阶段批准和指纹全部无效，不得复用。

## 为什么分阶段审批

canary 的攻击输入哈希依赖基础生成结果，mixed、特征、tensor bundle 和训练 scope 又分别依赖前一步的真实输出哈希。因此无法在基础生成前为全部步骤计算真实的精确指纹。`I-GATE-A` 按依赖链逐动作冻结，每一步完成并核验后才冻结下一步 scope。后续动作由 `docs/i_stage_standing_authorization.md` 覆盖，但仍生成各自独立的 scope、指纹和审批记录。

## 当前申请：I-03 canary 基础生成

- 动作：`generate`
- 资源类：`local-rtx5060-8gb`
- 配置：`configs/intermediate.yaml`
- scope：`docs/i03_canary_base_generation_scope.json`
- 指纹：`49d5fc5508d4951fffc7d867c2d907757ebe9739635dad93e39843c1440304ed`
- 输入：80 个 Train/Dev 父样本、400 条基础生成配方；Calibration 与 Test 不在 scope 中。
- 模型：本地、哈希已验证的 Qwen2.5-1.5B-Instruct；`local_files_only=true`，不下载模型，不允许 remote code。
- 水印：KGW、Unigram、Unbiased、SynthID；每族 A/B 两把现有私钥，仅从 Git 忽略的本地密钥文件加载。
- 生成：每条最多 256 个新 token，固定配方种子，`do_sample=true`、temperature 0.8、top-p 0.95。
- 输出：`artifacts/i03_canary/base_generated.jsonl` 及结果 JSON；支持 partial 续跑，失败显式记录。
- 监控：每 1% 或 5 分钟写状态；GPU 连续 120 秒达到 85°C、RAM 超过 26 GiB、E 盘低于 100 GiB、CUDA/驱动异常、OOM 或哈希漂移立即停止并保留 partial。
- 当前只读资源核验：E 盘可用约 535.8 GiB；7 个模型关键文件、4 个代码文件、数据/密钥/冻结清单哈希和 MarkLLM commit 均通过；没有加载模型或调用 CUDA。
- 预计本动作耗时：约 30–60 分钟。canary 全链预计约 1.5–3 小时，实际值用于推算完整 18–24 小时阶段。

## 已授权命令

根据持续授权，先创建与上述指纹完全一致、限时有效的独立审批记录，然后执行：

```powershell
python -m cwr_eg.cli generate `
  --config configs\intermediate.yaml `
  --resource-class local-rtx5060-8gb `
  --scope-file docs\i03_canary_base_generation_scope.json `
  --approval status\approvals\i03_canary_base_generation.json
```

本文件不是运行时审批记录；运行时记录引用 `docs/i_stage_standing_authorization.md` 中保存的聊天授权证据。

## 第二个已冻结动作：I-03 canary 匹配攻击

- 动作：`attack-generate`
- 资源类：`local-rtx5060-8gb`
- scope：`docs/i03_canary_attack_generation_scope.json`
- 指纹：`b7a14232714fe10795b01b8dffdaa22e877e2896b0773da15f98d40048014e18`
- 输入：已验收的 400 条 canary 基础文本，SHA-256 `96568c596c078aa9280deb665e578680aa99f281625a36c9e66435bb222f10d3`。
- 模型：本地 Qwen2.5-0.5B-Instruct，只用于 paraphrase 与 translation roundtrip；copy edit 与 truncation 按冻结确定性规则执行。
- 配方：400 条唯一 Train/Dev 匹配攻击配方；Calibration 与 Test 为 0。
- 输出：`artifacts/i03_canary/attacked_generated.jsonl`；最大失败率 1%，其余资源停止条件不变。

## 第三个已冻结动作：I-03 canary mixed 生成

- 动作：`generate`
- 资源类：`local-rtx5060-8gb`
- scope：`docs/i03_canary_mixed_generation_scope.json`
- 指纹：`acda9a32d92ec3fbf181edd40580d6f32a5cc7abf64f601ca80896d99596157b`
- 输入：40 条唯一 mixed 配方、80 个组件，Train/Dev = 30/10，en/zh = 20/20；每个配方的两个父样本均同语言、同分区。
- 模型：本地 Qwen2.5-1.5B-Instruct；每组件最多 128 个新 token，采样参数与冻结计划一致。
- 输出：`artifacts/i03_canary/mixed_generated.jsonl`，保存 80 个精确 Unicode 字符半开区间；最大失败率 1%，Calibration/Test 为 0。

## 第四个已冻结动作：I-03 canary 数据组装

- 动作：`assemble-data`
- 资源类：`local-cpu-31gb`
- scope：`docs/i03_canary_assemble_scope.json`
- 指纹：`1df0e66d79967942099541df83b5b2f4a8e56042ba6db771d92d50fd9a4e8211`
- 输入：400 基础、400 攻击、40 mixed，三份正式产物哈希均已绑定。
- 输出：840 条按冻结配方排序的完整数据；特征入口 810 条，明确排除 30 条 Train mixed。

## 第五个已冻结动作：I-03 canary 特征提取

- 动作：`extract-features`
- 资源类：`local-rtx5060-8gb`
- scope：`docs/i03_canary_feature_extraction_scope.json`
- 指纹：`bd85a13f817a4979e9854f2c2ab9dee270d1006e84e9e52aa9aebbafe05bf314`
- 输入：810 条唯一文档，Train/Dev = 600/210；Train mixed、Calibration、Test 均为 0。
- 模型：本地 Qwen2.5-1.5B-Instruct，最多 1024 token。
- OOM 策略：只允许按 `4→2→1` 降低 microbatch；原子写入、逐文件哈希和断点续跑开启。

## 第六个已冻结动作：I-03 canary 特征断点恢复

- 原作用域在 740/810 后因唯一单 token 文档停止；无 OOM、无资源超限、无静默丢弃。
- 冻结增补：`manifests/intermediate_freeze_amendment_01.json`，SHA-256 `a1b7c76eaf6c9c78189f86202d4c0ae366b3bcf13c1b1df9e9cbfd844ba7aad3`。
- 动作：`extract-features`
- scope：`docs/i03_canary_feature_resume_scope.json`
- 指纹：`fd20ed49742867386a30057f4778b1b03a5f8a4d873f074ac27611bb6268d9a4`
- 断点绑定：740 条 manifest SHA-256 `878d0f42fcdb856d3bc186cb64d0c234b9b99777db6c6a9d657a4811b486364b`，740 个 NPZ 均通过逐文件哈希、有限值、维度和元数据校验。
- 修复：只对恰好一个 token 的文本添加无字符区间 EOS 上下文；普通文档计算保持 `v3`，该短文本单独标记 fallback 版本。
- 回归：隐藏 CUDA 后 61/61 CPU 测试通过；恢复时仍只允许 `4→2→1`。

## 第七个已冻结动作：I-03 canary 分片 tensor bundle

- 动作：`tensorize`
- 资源类：`local-cpu-31gb`
- scope：`docs/i03_canary_tensorize_scope.json`
- 指纹：`68ce05bbfff98da1e76562d73b04b1ad548ec1af571f391e8145d75313020e7f`
- 输入：810 条最终特征 manifest，SHA-256 `3ea23fe09a7b4df9ff54ed4adb461372b317f1c8bb2efa9290190cc5469ab297`。
- 输出：`sharded-v1`，Train 600 例/30 批/2 shard，Dev 200 例/10 批/1 shard；每 shard 最多 16 批。
- Dev mixed 的 10 个多父样本特征只用于后续定位评估，不进入训练 bundle。

## 第八个已冻结动作：I-03 canary 代表模型训练

- 动作：`train`
- 资源类：`local-rtx5060-8gb`
- scope：`docs/i03_canary_training_scope.json`
- 指纹：`b5ea88164628e047e316ab8281872318d978f40ab3800b1299bf12403782d040`
- 运行：训练矩阵中的 `full_seed_20260815`，仅作为 canary 代表模型；不替代 `I-05` 的 10 个完整训练。
- 模型：positions 256、batch 20、hidden 256、invariant/private 128/128、学习率 `3e-4`。
- 训练：最多 20 epochs、至少 5 epochs、Dev 连续 4 epochs 不改善早停；确定性算法开启。
- 输入：通过完整哈希验收的 `sharded-v1` bundle，索引 SHA-256 `d719ede3674ade618ce7446a48bb0b013445ce28b4b7e76fab13c418b90c542b`。

### 确定性环境重试

- 首次训练在任何 epoch 或 checkpoint 产生前被 PyTorch 拒绝，因为确定性 CUDA 还要求进程启动前设置 `CUBLAS_WORKSPACE_CONFIG`。
- 新 scope：`docs/i03_canary_training_retry_scope.json`
- 新指纹：`e6a74da1b5fbb9485c15c1796bea65620713e60255501ac08bddaa9c06e2abc4`
- 唯一运行差异：绑定 `CUBLAS_WORKSPACE_CONFIG=:4096:8`；模型、数据、损失、种子、epochs、早停和输出位置全部不变。
