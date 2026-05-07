# 三条路线分析与综合方案

## 路线1：自适应掩蔽 (Adaptive Bit-Rate Masking)

### 代码状态：✅ 已修复可用

**核心思路**：KV Cache 的离群值不是噪声，而是注意力锚点。给它们更多比特预算，而不是被 FWHT 抹平。

**已完成的修复**：
- `encode_matrix()` 原生支持 `adaptive_masking=True` + `mask_*` 参数
- `_encode_single_path()` 正确传递所有 masking 参数
- per-tile 检测逻辑：方差 + max-abs 阈值判断敏感 tile
- 敏感 tile 获得: `proj_beta * (1 + mask_proj_beta_boost)` + 额外 `mask_n_steps_boost` 次迭代

**触发方式**：
```python
cfg = DSKVCacheConfig(
    adaptive_masking=True,
    mask_outlier_threshold=3.0,   # 默认3σ外判定为离群
    mask_proj_beta_boost=0.5,     # proj_beta 提升50%
    mask_n_steps_boost=1,         # 多1次1-bit迭代
)
```

**预期收益**：直接解决 `first_divergence=1` 崩溃，因为锚点 tile 得到更多比特预算。

### 优先级：⭐⭐⭐（第一梯队）

---

## 路线2：差分注意力残差 (Differential Attention Residuals)

### 代码状态：❌ 需要新实现

**核心思路**：从"数值逼近"升维到"功能对齐"。不在 encoder 层追求更小的 MSE，而是在 attention 计算层面让误差相互抵消。

**当前 `cross_head_error_share` 做了什么**（数值层面）：
- 正向传播 Σ-Δ 动量 `(initial_momentum, initial_integrator2)` 在 GQA head 间链式传递
- 作用：让相邻 head 的量化误差状态相关 → 减少突发性离群

**路线2真正需要的是什么**（功能对齐层面）：
1. GQA 组内计算平均量化残差 **ē** = `mean_per_group(Q@K^T - Q@K_hat^T)`
2. 将 **ē** 作为预测性偏置注入下一 head 的 attention score：`score_n += bias * ē_{n-1}`
3. 这**无法**只在 encoder 层完成，需要在 attention hook 里实现

**需要的改动**：
- `model_wrapper.py` 中注册 attention 前向 hook
- 捕获每层每 head 的 `Q@K^T`（原始 vs 重建）
- 计算组内平均残差 ē
- 反向注入到同组下一 head 的 score 计算
- 实验复杂度较高，推荐作为 Phase 2

### 优先级：⭐（第二梯队）

---

## 路线3：DCT 能量聚集

### 代码状态：✅ 已部署可用

**核心思路**：DCT 比 FWHT 更适合 KV Cache 的局部相关性特征。DCT 将能量推向低频，Σ-Δ 调制器追踪"主要分量"的效率指数级提升。

**已完整实现的模块**：
- `rina/utils/transforms.py`：DCT/DWT/Hybrid/Auto 四种模式
- `_encode_single_path()`：encode 前调用 `apply_transform()`
- `reconstruct_all()`：decode 后调用 `apply_inverse_transform()`
- per-tile 决策存储：`transform_decisions` 列表
- raw buffer tail 的逆变换兼容

**用法**：
```python
cfg = DSKVCacheConfig(
    transform_mode="dct",        # 或 "dwt", "hybrid", "auto"
    # 自动模式需要以下阈值：
    transform_smooth_threshold=0.05,
    transform_outlier_threshold=3.0,
    use_differential=False,      # 必须关！DCT域残差 ≠ 空间域残差
    cross_token_group=1,         # 必须=1！DCT reshape 冲突
    v_orthogonal_transform=False,# 可选关，避免双旋转
)
```

**限制**：当前 DCT 与 differential residual + cross_token_group > 1 + v_orthogonal 不兼容（`_ablation_transform_roadmap3.py` 强制关闭了这些）。

### 优先级：⭐⭐⭐（第一梯队）

---

## 综合推荐执行计划

### Phase 1（立即执行）：路线3 + 路线1 对照实验

```
Day 1:  运行路线3 DCT vs FWHT vs none 对比
        预估收益：CosSim +0.01~0.03, KL↓30%, match_rate↑5~10%

Day 2:  运行路线1 Adaptive Masking 对照实验
        预估收益：first_divergence 后移或消除
```

两者不冲突——`adaptive_masking` 可以与任何 `transform_mode` 叠加（代码已支持）。

### Phase 2（路线2评估）：设计 attention hook 方案

如果 Phase 1 的 match_rate 仍不够（<95%），再启动路线2。

**路线2的工程代价**：
- 需要深入 HuggingFace attention 实现
- 需注册 `torch.Tensor.register_hook` 或替换 attention forward
- 实验成本高，调试周期长

### 关键结论

> **路线1（自适应掩蔽）和路线3（DCT）不冲突，可以组合使用，解决不同层面的问题**：
> - 路线3：全局能量集中 → 提高 SNR
> - 路线1：局部 outlier 保护 → 防止崩溃
> - 路线2：系统级误差抵消 → 当路线1+3还不够时启用

建议从路线3开始跑实验，因为代码改动量为 0（改个 config 就行），收益可量化。