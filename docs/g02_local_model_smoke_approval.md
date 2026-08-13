# G-02 本机模型冒烟审批申请

状态：**等待用户明确批准，尚未执行。** 截至 2026-08-13 17:26（Asia/Shanghai），本步骤没有加载模型、调用 CUDA 或生成文本。

## 1. 已完成的非实验准备

- 环境固定为项目隔离环境 Python 3.11.15、PyTorch 2.9.1+cu128、CUDA runtime 12.8。
- 当前配置哈希：`3aa92e52d266ee66670c030dbd49c96f4be0097db570c2131dcfbd29415d0e2d`。
- 两个模型均使用旧 `project1` 中的本地只读资产，无需下载；配置、生成配置、词表、tokenizer 和权重共 7 个加载关键文件已全部纳入加载前字节数与 SHA-256 检查。
- 0.5B 权重：988,097,824 字节（0.920 GiB），当前 SHA-256 复核通过。
- 1.5B 权重：3,087,467,144 字节（2.875 GiB），当前 SHA-256 复核通过。
- 模型运行器已拆分为 `forward_only` 和 `generate_only`，不再对每个模型同时执行两项操作；运行器源码 SHA-256 `9028fdc529bf51702a8f64138d31fd91c47a08ffc1277ce6208a89bb0c903925` 也已绑定到两个作用域。
- 33/33 个纯合成 CPU 单元测试通过；旧 G-01 批准记录对 G-02 的拒绝路径验证通过。

## 2. 请求批准的精确作用域

### G-02-A：Qwen2.5-0.5B 单次前向

- 动作：`model-smoke / forward_only`。
- 模型：`Qwen/Qwen2.5-0.5B-Instruct`。
- 固定 revision：`7ae557604adf67be50417f59c2c2f167def9a775`。
- 权重 SHA-256：`fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe`。
- 输入：一个固定短提示，批量大小 1。
- 设备与精度：`cuda:0`、`bfloat16`。
- 约束：仅调用一次独立前向；禁止调用 `generate`；禁止联网下载；禁止远程代码。
- 作用域文件：`docs/g02_qwen05_forward_scope.json`。
- 批准指纹：`de2ad5e5e22eee3a39037f6f847e37949083f494bb30593aa83e902262fb82e6`。

### G-02-B：Qwen2.5-1.5B 单次生成

- 动作：`model-smoke / generate_only`。
- 模型：`Qwen/Qwen2.5-1.5B-Instruct`。
- 固定 revision：`989aa7980e4cf806f80c7fef2b1adb7bc71aa306`。
- 权重 SHA-256：`dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee`。
- 输入：同一个固定短提示，批量大小 1，最多生成 16 个新 token，`do_sample=false`。
- 设备与精度：`cuda:0`、`bfloat16`。
- 约束：仅调用一次 `generate`，不额外执行独立前向测试；生成过程内部必然包含模型推理；禁止联网下载；禁止远程代码。
- 作用域文件：`docs/g02_qwen15_generate_scope.json`。
- 批准指纹：`801845f208c175492c7b85802312a6cf7310d727bc4cf58cb7dd8e490962e9d7`。

## 3. 资源影响与执行纪律

- 两个动作按两个独立进程顺序执行，不同时驻留显存。
- 不新增模型磁盘占用；只写两个小型 JSON 结果文件。当前 E 盘可用空间约 535.89 GiB。
- 根据 BF16 权重大小和短序列批量 1 估算，0.5B 峰值显存预计不超过约 2 GiB，1.5B 预计不超过约 4.5 GiB；这是执行前保守估算，不是实测值。
- 每个动作只尝试一次。发生权重哈希不匹配、CUDA 异常、显存不足、非有限 logits 或生成异常时立即停止，不切换 CPU、不联网补下载、不扩大范围。
- 本批准不涵盖水印生成、攻击生成、训练、反向传播、校准、推理评估、基准测试或完整数据生成。

## 4. 批准后才会执行的命令

批准后将从用户聊天证据分别创建有时效的两个批准记录，再执行：

```powershell
& 'E:\.cwr-eg-project-local\.conda\cwr-eg-win-py311\python.exe' -m cwr_eg.cli model-smoke --config configs\pilot.yaml --resource-class local-rtx5060 --scope-file docs\g02_qwen05_forward_scope.json --approval status\approvals\g02_qwen05_forward_20260813.json

& 'E:\.cwr-eg-project-local\.conda\cwr-eg-win-py311\python.exe' -m cwr_eg.cli model-smoke --config configs\pilot.yaml --resource-class local-rtx5060 --scope-file docs\g02_qwen15_generate_scope.json --approval status\approvals\g02_qwen15_generate_20260813.json
```

## 5. 可直接回复的批准文本

> 批准 G-02-A 指纹 de2ad5e5e22eee3a39037f6f847e37949083f494bb30593aa83e902262fb82e6 与 G-02-B 指纹 801845f208c175492c7b85802312a6cf7310d727bc4cf58cb7dd8e490962e9d7；允许依次从已核验本地路径加载两个模型并调用 CUDA。禁止联网下载，禁止超出审批申请所列作用域。

若只批准其中一个动作，请只写对应编号和指纹；另一个动作将继续保持禁止状态。
