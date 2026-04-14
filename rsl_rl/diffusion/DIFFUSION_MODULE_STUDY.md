# rsl_rl.diffusion 模块深度学习笔记（逐文件详解）

> 目标：把 `rsl_rl/diffusion` 目录下每个文件的核心代码“拆开讲清楚”，包括职责、输入输出、关键公式、隐含约束、以及在训练/采样链路里的位置

---

## 0. 模块总览

`__init__.py` 对外暴露的核心对象：

- 扩散基础：`DiffusionScheduler`
- 模型：`MotionEpsilonTransformer`
- 采样：`SMPDiffusionSampler`
- 训练：`SMPDiffusionTrainer`
- 风格条件：`StyleConditioner`、`maybe_drop_style`、`apply_classifier_free_guidance`
- 风格组合：`build_g1_body_part_feature_masks`、`compose_style_predictions_with_body_masks`
- GSI reset 解码：`SMPFeatureLayout`、`SMPGSIDecoder`、`SMPGSISampler` 等
- 奖励：`SMPReward`
- 稳定训练：`ExponentialMovingAverage`
- 日志：`log_smp_noise_metrics`、`log_smp_pretrain_metrics`

这说明这个子模块覆盖了完整闭环：

1. 训练阶段：`x0 -> q_sample 得到 xt -> 模型预测 eps_hat -> MSE`
2. 在线阶段：反向扩散采样得到 window，必要时转成 reset state
3. 奖励阶段：用若干时间步噪声误差构建 `exp(-scale * mse)` 的奖励

---

## 1. `scheduler.py` —— 前向扩散日程与加噪

### 1.1 `DiffusionScheduler.__init__`

构造三个关键序列（长度 = `num_steps`）：

- `beta_t`: 线性插值，范围 `[beta_start, beta_end]`
- `alpha_t = 1 - beta_t`
- `alpha_bar_t = prod_{i<=t} alpha_i`

关键点：

- 参数校验非常严格：`num_steps > 0`，`beta_start/beta_end > 0`，且 `beta_end > beta_start`
- 这里的线性 beta schedule 是 DDPM 的经典配置

### 1.2 `sample_timesteps(batch_size, device=None, timesteps_k=None)`

作用：抽训练用的时间步 `t`（shape `(B,)`）

- 若 `timesteps_k is None`：在 `[0, num_steps)` 均匀采样
- 若给了 `timesteps_k`：只在候选集合里采样

细节：

- `timesteps_k` 会先转成 1D long tensor，再抽“索引”，最后映射回时间步值
- 对候选范围有边界检查（必须在合法扩散步内）

### 1.3 `q_sample(x0, t, eps)`

前向扩散闭式：

$$
 x_t = \sqrt{\bar\alpha_t} x_0 + \sqrt{1-\bar\alpha_t} \epsilon
$$

要求：

- `x0.shape == eps.shape`
- `t` 必须是 `(B,)` 且 `B == x0.shape[0]`

实现细节：

- 从 `alpha_bar[t]` 按样本取值，reshape 成 `(B,1,...,1)` 以广播到任意维度数据
- 可直接支持 `(B, W, F)` 等多维输入

---

## 2. `conditioning.py` —— 风格条件与 CFG

### 2.1 常量 `NULL_STYLE_ID = -1`

表示无条件（null-style）分支的“逻辑标签”，并不直接拿这个索引查 embedding

### 2.2 `maybe_drop_style(style_id, drop_prob, null_style_id=-1)`

作用：训练时随机把一部分样本改成 null-style（Classifier-Free Guidance 训练范式）

- `drop_prob=0`：不丢
- `drop_prob=1`：全丢（全部置为 `null_style_id`）
- 中间值：Bernoulli 掩码逐样本丢弃

返回：与 `style_id` 同 shape 的新 tensor，或者 `None`

### 2.3 `apply_classifier_free_guidance(eps_uncond, eps_cond, guidance_scale)`

标准 CFG 融合：

$$
\hat\epsilon = \epsilon_{uncond} + s(\epsilon_{cond} - \epsilon_{uncond})
$$

其中 `s = guidance_scale`

### 2.4 `StyleConditioner`

把离散风格 id 映射到隐空间向量：

- `nn.Embedding(num_styles + 1, hidden_dim)`
- 额外一行（index = `num_styles`）专门给 null-style

关键函数 `_style_id_to_indices`：

- `style_id is None` -> 全部映射到 null embedding index
- 若存在 `-1`，替换成 null embedding index
- 禁止除 `-1` 以外的负值
- 非 null 的 id 必须 `< num_styles`

`forward` 输出 shape `(B, hidden_dim)`

---

## 3. `model.py` —— `MotionEpsilonTransformer`

### 3.1 架构组成

输入默认是 `xt`（`B, W, F`）：

1. `token_proj`: `F -> hidden_dim`
2. `pos_embedding`: 可学习位置向量 `(1, window_size, hidden_dim)`
3. `timestep_embedding + MLP`：扩散步 `t` 注入每个 token
4. （可选）`StyleConditioner` 注入风格向量
5. `TransformerEncoder`
6. `output_proj`: `hidden_dim -> F`，输出 `eps_hat`

### 3.2 关键约束

- `hidden_dim % num_heads == 0`
- 输入窗口长度 `xt.shape[1] <= window_size`
- `style_conditioner is None` 时禁止传 `style_id`

### 3.3 `forward(xt, t, style_id=None)`

流程：

- 先做维度校验
- `t` 转 `long`
- hidden 累加：token + pos + timestep (+ style)
- 过 encoder 后投影回特征维

输出：与 `xt` 同 shape 的噪声预测 `eps_hat`

---

## 4. `ema.py` —— 指数滑动平均

`ExponentialMovingAverage` 维护 `shadow_state`：

- 初始化：复制 `model.state_dict()`
- 更新：

$$
\theta_{ema} \leftarrow d\,\theta_{ema} + (1-d)\,\theta
$$

- `copy_to(model)`：把 EMA 权重覆盖回模型（常用于 eval/导出）
- 自带 `state_dict/load_state_dict`

注意：该实现包含 `state_dict` 的所有条目（参数 + buffer）

---

## 5. `composition.py` —— 身体部位掩码与风格组合

### 5.1 常量集合

文件顶部定义了 G1 机器人的关节/末端执行器/关键 body 分组：

- 下肢、上肢、共享（腰/躯干等）

### 5.2 `build_g1_body_part_feature_masks(...)`

输入：

- `joint_name_order`、`ee_name_order`、`key_body_name_order`
- `feature_block_offsets`：每个特征块在总特征中的 `[start, stop)`

输出：三个 1D mask（长度=feature_dim）：

- `shared_body`
- `lower_body`
- `upper_body`

核心逻辑：

1. 基座速度块默认归 `shared_body`
2. `joint_pos_rel` 逐关节按名字分配到三类
3. `ee_pos_b` 按末端执行器名称分配
4. `key_body_rot6d` 按 key body 名称分配
5. `_validate_masks` 保证：
   - 非空
   - 三个 mask 互斥且完整覆盖（逐维和为 1）

这是一个很强的正确性保障，避免混合风格时出现“某维没被覆盖/被多次覆盖”

### 5.3 `compose_style_predictions_with_body_masks(part_to_eps, feature_masks)`

目标：把“按部位各自跑出来的 `eps_part`”合成为完整 `eps`

要求：

- `part_to_eps` 和 `feature_masks` 键集合必须一致
- 各 `eps_part` shape 必须一致
- mask 的 feature 维必须与 `eps` 最后一维匹配

广播：

- 若 mask 是 1D，会 reshape 为 `(1,1,F)`，可乘 `(B,W,F)`

组合公式（逐 part 求和）：

$$
\epsilon_{comp} = \sum_{p} \epsilon_p \odot m_p
$$

---

## 6. `sampler.py` —— 反向扩散采样与风格程序

`SMPDiffusionSampler` 是在线采样核心

### 6.1 初始化

内部持有：

- 一个模型 `model`
- 一个 `DiffusionScheduler`（拥有 `beta/alpha/alpha_bar`）
- 采样元信息（步数、窗口长、特征维、device）

### 6.2 风格程序 `_resolve_style_program`

输入可能有两种：

1. 显式 `style_program`（高级模式）
2. 简单 `style_id` + `guidance_scale`（便捷模式）

统一解析成 dict：

- `mode = unconditional`
- `mode = single_style`
- `mode = single_style_batch`
- `mode = body_mask`

### 6.3 `predict_eps(...)`

按模式分支：

1. `unconditional`：直接前向
2. `single_style(_batch)`：
   - 先算 `eps_uncond`（style=-1）
   - 再算 `eps_cond`
   - 用 CFG 合成
3. `body_mask`：
   - 对每个身体 part 的 style_id 分别前向（带 cache，避免重复 style 重算）
   - 用 `compose_style_predictions_with_body_masks` 合成 `eps_cond_comp`
   - 若 `guidance_scale != 1`，再与 `eps_uncond` 做 CFG

这个函数是“风格控制语义中心”：单风格与分身体部位风格共用同一采样器

### 6.4 `p_sample(xt, timestep, eps_hat)`

单步 DDPM 反向：

- 先算后验均值 `mean`
- `t=0` 直接返回 `mean`
- `t>0` 额外加 `sqrt(posterior_var)*noise`

公式与 DDPM 标准实现一致，`posterior_var` 用 `alpha_bar_{t-1}` 构造

### 6.5 `sample(batch_size, ...)`

完整反向过程：

1. 从 `N(0,I)` 生成初始 `xt`
2. `t = T-1 -> 0` 迭代：
   - `eps_hat = predict_eps(...)`
   - `xt = p_sample(xt, t, eps_hat)`
3. 返回最终 window（`B,W,F`）

---

## 7. `gsi.py` —— 从采样窗口到 reset state

这个文件把“特征空间采样结果”转回“环境可用状态”

### 7.1 四元数工具函数

- `_normalize_quat`
- `_quat_conjugate`
- `_quat_multiply`
- `_quat_apply`
- `_quat_apply_inverse`

用途：在世界坐标与机体系之间转换速度向量

### 7.2 `_expand_batch_tensor`

把 `(D,)` 或 `(B,D)` 的 reference 张量统一扩到 `(B,D)`，用于批量 decode

### 7.3 `SMPFeatureLayout`

保存特征切片定义（start/stop）：

- 必需：`base_lin_vel_b`, `base_ang_vel_b`, `joint_pos_rel`
- 可选：`ee_pos_b`, `key_body_rot6d`

`from_feature_block_offsets` 会检查必需 key 是否齐全

### 7.4 Reset 数据结构

- `SMPResetReference`：参考状态（root pose、默认关节位等）
- `SMPResetState`：实际解码出的 reset 状态
- `SMPGSIDecodeResult`：decode 结果 + 可恢复性标志 + 误差

### 7.5 `SMPGSIDecoder`

核心思想：

- 当前实现只“可恢复”三块：
  - `base_lin_vel_b`
  - `base_ang_vel_b`
  - `joint_pos_rel`
- 若布局含 `ee_pos_b/key_body_rot6d`，它们被标记为 `unrecoverable_feature_blocks`

`decode(window, reference_state)` 流程：

1. 取最后一帧 `last_frame`
2. 从 `reference_state` 扩展出 batch 形态的 root pose / joint defaults
3. 切片取出三块 recoverable 特征
4. `body -> world`：用 root quaternion 旋转速度
5. `joint_pos = joint_default + joint_pos_rel`
6. 重编码 recoverable 块并计算 `reconstruction_mse`
7. `supports_reset_state = mse <= error_threshold`

这相当于“自检”：decode 后再 encode 看是否还原，误差小才认为可直接用于 reset

### 7.6 `SMPGSISampler`

简单封装：

- 先 `sampler.sample(...)`
- 再 `decoder.decode(...)`

对 runner 来说直接得到 `SMPGSIDecodeResult`

---

## 8. `smp_reward.py` —— 噪声误差奖励

`SMPReward` 设计目标：把多时间步噪声 MSE 变成稳定奖励

### 8.1 初始化

输入：

- `timesteps_k`：固定监督时间步集合
- `reward_scale`
- `adaptive_norm_decay`

内部维护 `running_mse[t]`（每个时间步一个 EMA 标量）

### 8.2 `_normalize(timestep, mse)`

- `mse` 是该时间步逐样本误差 `(B,)`
- 先更新该时间步 `running_mse`
- 返回 `mse / running_mse`

作用：不同时间步误差尺度不同，归一化后更可比较

### 8.3 `compute(eps, eps_hat)`

对每个 `t in timesteps_k`：

1. 校验 key/shape
2. 计算逐样本 MSE（高维展平后按样本均值）
3. 做时间步归一化

最后：

- `noise_mse = mean_t(normalized_mse_t)`
- `reward = exp(-reward_scale * noise_mse)`

输出 dict：

- `reward` `(B,)`
- `noise_mse` `(B,)`
- `per_timestep_mse`（字典）

---

## 9. `logging.py` —— TensorBoard 指标组织

### 9.1 工具函数

- `_to_scalar`：把 tensor / 数值统一成 float
- `_to_histogram_tensor`：detach + cpu + flatten

### 9.2 `log_smp_noise_metrics`

记录：

- 总噪声误差标量 `prefix/noise_mse`
- 每个时间步标量 `prefix/t{t}/noise_mse`
- 直方图：
  - `eps_true`
  - `eps_pred`
  - `eps_gap`

支持两种输入：

- `eps`/`eps_hat` 是普通 tensor
- 或按 timestep 分字典（此时逐 t 记直方图）

### 9.3 `log_smp_pretrain_metrics`

在 pretrain 命名空间下写：

- `loss`, `loss_total`
- 可选 `loss_cond`, `loss_uncond`
- 可选每种风格 `style/{name}/noise_mse`
- 再调用 `log_smp_noise_metrics(prefix="SMPPretrain")`

---

## 10. `trainer.py` —— 离线预训练主循环

### 10.1 `_collate_smp_samples`

把 dataset sample 列表聚合成 batch：

- `motion` 堆叠为 `(B,W,F)`
- `style_id`：只要有任一 `None`，整批设为 `None`（否则转 long tensor）
- 同时保留 `style_name`、`clip_id`、`source_name`

### 10.2 `SMPDiffusionTrainer.__init__`

构建训练所需全部对象：

1. `SMPMotionWindowDataset`
2. `DataLoader`（含自定义 collate）
3. 从首个 sample 推断 `feature_dim/window_size`
4. `DiffusionScheduler`
5. `MotionEpsilonTransformer`
6. `ExponentialMovingAverage`
7. `AdamW`
8. `SummaryWriter`

风格数量逻辑：

- `inferred_num_styles = len(dataset.style_to_id)`
- 若外部指定 `num_styles`，不能小于推断值

### 10.3 `_next_batch`

手动维护 dataloader 迭代器：

- 到头就重置 iterator
- 把张量字段搬到 device

### 10.4 `_compute_loss`

单步训练核心：

1. `x0 = batch["motion"]`
2. 采样时间步 `t`（默认来自 `timesteps_k`）
3. 采样高斯噪声 `eps`
4. `xt = q_sample(x0, t, eps)`
5. `dropped_style_id = maybe_drop_style(...)`
6. `eps_hat = model(xt, t, style_id=dropped_style_id)`
7. 样本级 MSE：`(eps_hat-eps)^2` 展平后按样本均值
8. `loss = sample_mse.mean()`

附加统计：

- `per_timestep_mse`
- `loss_cond / loss_uncond`（基于 null-style mask）
- `per_style_mse`（按 `style_name` 聚合）

### 10.5 `save_checkpoint`

保存：

- `model_state_dict`
- `ema_state_dict`
- `optimizer_state_dict`
- `feature_dim/window_size/timesteps_k`
- `model_cfg`（结构参数）
- `style_cfg`（style 名称映射、drop prob、null id）

默认路径：`log_dir/model_latest.pt`

### 10.6 `train`

循环 `global_step=1..max_iters`：

1. 拿 batch + 算 loss
2. `zero_grad -> backward -> step`
3. `ema.update(model)`
4. 记录日志

结束后：

- 存 checkpoint
- `writer.flush/close`
- 返回训练摘要（最终 loss、窗口维度、checkpoint 路径）

---

## 11. 端到端数据流（训练 / 采样 / reset / 奖励）

### 11.1 训练流

`dataset motion window (x0)`
-> sample `t`
-> sample `eps`
-> `q_sample` 得 `xt`
-> transformer 预测 `eps_hat`
-> `MSE(eps_hat, eps)`
-> optimizer + EMA + TensorBoard

### 11.2 采样流

`xt ~ N(0,I)`
-> for `t=T-1..0`:
   - `predict_eps`（支持单风格 / body mask 组合 / CFG）
   - `p_sample`
-> 得到最终 `window`

### 11.3 reset 流（GSI）

`window`
-> `SMPGSIDecoder.decode`
-> `SMPResetState`
-> 同时给出 `reconstruction_mse` 和 `supports_reset_state`

### 11.4 奖励流

多时间步 `eps, eps_hat`
-> 每个时间步样本 MSE
-> 时间步级 EMA 归一化
-> 求平均
-> `reward = exp(-scale * noise_mse)`

---

## 12. 读代码时最容易忽略的细节（实战重点）

1. `NULL_STYLE_ID=-1` 不是 embedding 索引，真正索引在 `StyleConditioner` 内映射到 `num_styles`
2. `trainer` 的 `timesteps_k` 默认是稀疏集合，不是全时域训练
3. `composition` 的 mask 校验要求“互斥+全覆盖”，这是风格组合正确性的前提
4. `sampler` 的 `body_mask` 模式支持“各身体部分不同 style id”
5. `gsi` 目前只对 recoverable blocks 做重构一致性检查；`ee/key_body` 被明确标成不可恢复块
6. `SMPReward` 做了“按时间步自适应归一化”，否则不同 t 的误差尺度会干扰奖励
7. EMA 维护的是完整 `state_dict` 项，不只是 parameters

---

## 13. 建议的学习顺序（最快建立全局认知）

1. 先看 `scheduler.py` + `model.py`（扩散最小闭环）
2. 再看 `trainer.py`（训练流水线如何拼装）
3. 再看 `conditioning.py` + `sampler.py`（风格控制与反向采样）
4. 最后看 `composition.py` + `gsi.py` + `smp_reward.py`（工程化落地：风格拼接、reset、奖励）

---

## 14. 你可以立即做的验证实验（建议）

1. 把 `style_drop_prob` 从 `0.0` 调到 `0.2`，观察 `loss_cond/loss_uncond` 变化
2. 在 `sampler.predict_eps` 里比较 `guidance_scale` 为 `1.0/2.0/3.0` 的生成差异
3. 构造 body-mask 风格程序，检查不同部位 style 对输出窗口不同维度的影响
4. 打印 `SMPGSIDecodeResult.reconstruction_mse` 分布，选一个合理的 `error_threshold`
5. 画 `SMPReward.running_mse[t]` 的轨迹，确认归一化是否稳定

---

## 15. 一句话总结

这个 `diffusion` 子模块不是“单一模型文件”，而是一个完整的扩散运动先验系统：离线训练（含风格条件+CFG）-> 在线采样（含身体部位风格组合）-> reset 解码 -> 噪声误差奖励，且每一步都有明确的形状约束与日志/稳定性机制
