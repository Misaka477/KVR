# DS-KVCache Quality Improvement: 三大思路可行性分析

Date: 2026-05-09 | Script: `scripts/evaluation/eval_generation_fidelity.py`

---

## 1. 基准测试结果 (Balanced Config)

**配置**: n_steps=5, CTG=2, use_recon_weights=False, cross_head_error_share=False, transform_mode="none"

| Route | char_match | prefix_match | rep_score | time |
|-------|-----------|-------------|-----------|------|
| native | 1.000 | — | 0.033 | — |
| baseline | 0.176 | 25.9 | 0.072 | 60s |
| baseline_mask | 0.176 | 25.9 | 0.072 | 42s |
| r1 | 0.176 | 25.9 | 0.072 | 67s |
| r1_mask | 0.176 | 25.9 | 0.072 | 65s |

**关键发现**:
- **全部 4 条 DS 路线输出完全一致** (char_match=0.176, prefix_match=25.9)，验证了 Plan 的猜测：`adaptive_masking` 和 `use_mask_gating` 只影响 tile 级编码，无法解决 decode-loop 累积误差。
- repetition_score 0.03-0.12，偏低（因为 3-gram 采样下重复不明显），但生成文本确实出现了循环模式（如 "The roots are 1.5 and -1.5. The roots are 1.5 and -"）。
- prefix_match=25.9 意味着前 ~26 个字符与 native 完全一致，之后开始分叉。

---

## 2. 想法一：步长自适应纠偏 (Dynamic n_steps Boosting)

### 原理
在 Σ-Δ 编码后检查 remaining 残差范数，若超过阈值则重新以更高 n_steps 编码当前 tile。

### 代码路径可行性

**关键文件**: `rina/ds_kv_cache.py`

**已有的基础设施**:
- `_encode_and_append_tile()` (line 391) 已经动态获取 n_steps：`n_steps = cfg.get_n_steps_v() if is_v else cfg.get_n_steps_k()` — 不同 tile 可以有不同步数。
- n_steps 维度对齐逻辑 (lines 467-512) 完整支持：当新 tile 的 n_steps 与已有 store 不同时，自动用中性值（bases=1.0, alphas=0.0）补齐。
- `_encode_single_path()` (line 1022) 接受任意 `n_steps` 参数，支持 per-tile 变化。
- 差分残差阶段 (lines 520-558) 在 primary encode 之后计算 `residual = tile - primary`，恰好可以得到残差范数。

**实现策略（推荐）**:

```python
# 在 _encode_and_append_tile 中，primary encode 之后：
primary = decode_from_bases(bases, alphas, shape, tile_size=tile_size)
residual = tile - primary
residual_norm = residual.norm() / tile.norm()  # 归一化残差

if residual_norm > BOOST_THRESHOLD and n_steps < N_STEPS_MAX:
    # 以更高 n_steps 重新编码
    boosted_n_steps = min(n_steps * 2, N_STEPS_MAX)
    bases, alphas, ... = _encode_single_path(tile, n_steps=boosted_n_steps, ...)
```

**复杂度评估**: 

| 维度 | 评估 |
|------|------|
| 实现难度 | ⭐⭐ 中等：改动集中在 `_encode_and_append_tile` 一个函数 |
| 计算开销 | 仅对高残差 tile 触发，预期 <20% tile 需要 boost，总开销 ~5-10% |
| 存储开销 | 高残差 tile 存储更多 bases → 压缩比下降。需要控制 N_STEPS_MAX |
| 调参风险 | 需要实验 `BOOST_THRESHOLD` 和 `N_STEPS_MAX` |

**关键问题**:
- 残差是在 primary encode 之后计算的，意味着我们已经花了一次 encode 的算力。如果 boost，等于编码了两次。
- 更好的策略是 **预测**：用 tile 的能量分布（max_abs/std ratio）预测编码难度，在编码前决定 n_steps。这其实就是现有的 `adaptive_masking` 做的事，只是它的触发条件不同（基于 outlier 检测而非残差）。
- 这个方案与 `adaptive_n` 功能重叠。建议：如果做，直接扩展 `adaptive_n` 的判断逻辑，加入残差范数作为一种触发条件。

**结论**: 技术上完全可行，最适合作为 `--quality high` 之外的**进一步优化**。推荐阈值调优后作为 `quality=extreme` 模式。

---

## 3. 想法二：关键 Token "锚点"刷新 (Periodic FP16 Bypass)

### 原理
每隔 N 个 token（如 16），将该位置的 KV 直接以 FP16 精度存储，不经过 Σ-Δ 编码，从而将累积误差完全归零。

### 代码路径可行性

**关键文件**: `rina/ds_kv_cache.py` + `rina/model_wrapper.py`

**已有的基础设施**:
- `protected` 模式 (ds_kv_cache.py:262-277)：当 `self.protected=True` 时，`append_incremental` 直接将 FP16 存入 `raw_buffer`，永远不触发 tile 编码。`reconstruct_all()` 在两种路径（multi-segment 和 single-segment）都能正确返回 raw_buffer 的内容。
- `raw_buffer` 机制本身就是为了暂存未编码的 token，`reconstruct_all` 总是将其拼接在解码输出的末尾。
- `_append_incremental` (model_wrapper.py:165-224) 已经有 decode_step 计数器可用。

**核心难点**: `protected` 是 store 级别的静态属性，而我们需要的是 **per-token 级别的选择性 bypass**。

**当前 flow 的问题**:

```
Step 7 (bypass): new_token → append_incremental → raw_buffer?
Step 8 (normal): new_token → append_incremental → raw_buffer += [new_token_8]
    → 此时 raw_buffer 中有 [bypass_7, normal_8]
    → buffer_full >= tile_size → encode BOTH as a tile
    → bypass_7 被错误地编码了！
```

Bypass token 和 normal token 混在同一个 raw_buffer 中，会被一起编码。

**解决方案（推荐）**:

新增一个独立的 `raw_bypass` 存储，避开 raw_buffer 的 tile 编码流程：

```python
# DSKVCacheStore 新增字段
raw_bypass: List[Tuple[int, torch.Tensor]]  # (logical_position, fp16_tensor)

# append_incremental 新增 bypass 模式
def append_incremental(self, new_vec, *, bypass=False, ...):
    if bypass:
        pos = self.n_tokens  # 当前逻辑位置
        self.raw_bypass.append((pos, new_vec.to(torch.float16)))
        self.bypass_positions.append(pos)
        return momentum, integrator2
    # ... 正常编码流程

# reconstruct_all 新增插值逻辑
def reconstruct_all(self, ...):
    result = ... # 解码所有 encoded tiles + raw_buffer
    for pos, vec in self.raw_bypass:  # 按位置插入
        result = torch.cat([result[:pos], vec.unsqueeze(0), result[pos:]])
    return result
```

但上面这个 naive 插值在每次 decode 时都是 O(n) 操作，且 bypass position 是动态变化的（因为 store 的 `n_tokens` 在增长）。

**更简洁的方案**——在 model_wrapper 层面处理:

不修改 `DSKVCacheStore`，在 `_append_incremental` (model_wrapper.py) 中处理：

```python
def _append_incremental(self, past_key_values, new_token_idx=-1, decode_step=0):
    is_bypass = (decode_step + 1) % self.cfg.refresh_interval == 0
    for layer_idx in range(self._num_layers):
        k_full, v_full = _past_get_kv(past_key_values, layer_idx)
        for h in range(n_kv):
            k_new = k_full[0, h, new_token_idx:]
            v_new = v_full[0, h, new_token_idx:]
            if is_bypass:
                # 直接追加 FP16 到 raw_buffer，但不触发编码
                # 给 store 加一个 bypass 标志位
                k_stores[h]._bypass_positions.append(k_stores[h].n_tokens)
                k_stores[h]._bypass_data.append(k_new.clone().half())
                # 也追加到 raw_buffer，但标记为"不编码"
            else:
                k_stores[h].append_incremental(k_new, ...)
```

**复杂度评估**:

| 维度 | 评估 |
|------|------|
| 实现难度 | ⭐⭐⭐ 中等偏高：需要处理 store 的 bypass 存储和 reconstruct_all 的插值逻辑 |
| 计算开销 | 极低：bypass token 不做任何编码，反而节省计算 |
| 存储开销 | 每 refresh_interval 步多存一个 FP16 tile (~2KB/head)，总增长 ~2-5% |
| 压缩比影响 | 可控：refresh_interval=16 时，~6% token 不压缩，压缩比从 4.3× 降至 ~4.0× |
| 效果 | **直接消除累积误差** — 最根本的解决方案 |

**结论**: 这是三个方案中**效果最确定、物理意义最清晰**的一个。实现复杂度中等，但收益直接。推荐作为 **P1 优先级**。

### 实现步骤（推荐方案）

1. `DSKVCacheStore` 新增 `raw_bypass_positions: List[int]` 和 `raw_bypass_cache: Dict[int, torch.Tensor]`
2. `append_incremental` 新增 `bypass: bool` 参数
3. `reconstruct_all` 在两个 decode 路径中加入 bypass 数据的插值
4. `model_wrapper._append_incremental` 根据 `decode_step % refresh_interval` 决定是否 bypass
5. `DSKVCacheConfig` 新增 `refresh_interval: int = 0` (0=disabled)

---

## 4. 想法三：Logits 引导的"幻觉抑制" (Logit-Guided Refinement)

### 原理
在 decode 循环中监控 logit 分布。当 Top-1 与 Top-2 概率差小于阈值时，对当前 token 的 KV 实施更高精度的重编码。

### 代码路径可行性

**关键文件**: `rina/model_wrapper.py` (generate + _append_incremental)

**时序问题（致命缺陷）**:

```
Step N 的 decode loop:
  ┌─────────────────────────────────────────────────────────┐
  │ 1. model.forward(last_token, past=DS_decoded_past)      │
  │    → output.logits (vocab distribution)                 │
  │    → output.past_key_values (原始 KV，仅含最新 token)    │
  │                                                         │
  │ 2. _append_incremental(past_key_values, ...)            │
  │    → 将新 token 编码到 DS stores                       │
  │    → stores 已修改，无法回滚                              │
  │                                                         │
  │ 3. past = _build_past_from_ds()                         │
  │    → 解码完整 KV 为下一步准备                             │
  │                                                         │
  │ 4. 此时才能检查 output.logits ← 但 stores 已在 step 2 修改│
  └─────────────────────────────────────────────────────────┘
```

logits 在 step 1 产生，但我们需要在 step 2 之前做决策。而 step 2 必须先执行（因为 N+1 步的 forward 需要新 past）。

**可能的 workaround —— 延迟一拍的 refine**:

在 step N 检测到 logits "纠结"，在 step N+1 对该 token 的 KV 进行更高精度重编码：

```python
# 伪代码
pending_refine = False
for step in range(max_new_tokens):
    output = model(input_ids=last_token, past_key_values=past)
    
    if pending_refine:
        # 上一拍的 token 需要高精度编码
        _append_with_higher_quality(...)
    else:
        _append_incremental(...)
    
    past = _build_past_from_ds()
    
    # 检查当前 step 的 logits
    probs = softmax(output.logits)
    if probs.topk(2)[0][0] - probs.topk(2)[0][1] < threshold:
        pending_refine = True
```

**问题**:
- 只 refine 当前 token，但分叉可能发生在历史 token 上。如果 step 7 的分叉在 step 15 才被检测到，refine step 15 的 KV 无法修复 step 7 的分叉。
- 需要在 store 中实现"替换最后一个 tile"的能力——删除 `_encode_segments` 的最后一项、回退 `self.bases` 和 `self.alphas` 的最后一列、重新编码。这是 fragile 的操作。

**复杂度评估**:

| 维度 | 评估 |
|------|------|
| 实现难度 | ⭐⭐⭐⭐ 高：时序问题 + store 回滚操作复杂 |
| 计算开销 | 仅对"纠结" token 触发（预计 5-15% token） |
| 效果 | 不确定：分叉点的 KV 噪声是累积的，不是单 token 造成的 |
| 风险 | 高：store 状态机回滚容易引入 bug |

**结论**: 想法很精妙但工程实现复杂、时序矛盾难以解决。不如将算力省下来直接全量提升 n_steps（`--quality high`）。不建议在当前阶段实现。

### 替代思路：Tile-Based 预判敏感度

不依赖 logits，而是在 `_encode_and_append_tile` 中计算 tile 的统计特征（max_abs/std、能量集中度），预测该 tile 是否"敏感"。敏感 tile 自动使用更高 n_steps。这是想法一的变体，但避免了残差重算的浪费。

---

## 5. 综合推荐

```
优先级   方案                          改动量        效果确定性    副作用
─────   ───────────────────────────  ──────────   ──────────   ──────
 P0     先用 --quality high 跑基准   0 (已实现)    高           压缩比 2.5×
 P1     想法二：锚点刷新             中等 (~200行)  极高         压缩比 ~4.0×
 P2     想法一：自适应步长            低 (~50行)    中高         压缩比 ~3.5×
 P3     想法三：Logits引导精炼        高 (~400行)   低           压缩比 ~3.5×
```

**推荐路径**:
1. 立即运行 `--quality high --measure-kv` 获取 high 配置的基准结果
2. 如果 high 的 char_match > 0.5，证明配置参数足以解决大部分问题，P1 的锚点刷新可以作为可选优化
3. 如果 high 的 char_match 仍然 < 0.3，优先实现 P1（锚点刷新），因为它直接解决了累积误差的根因

**P1 实现预估**:
- 核心改动：`ds_kv_cache.py` 中 `DSKVCacheStore` 新增 ~30 行（bypass 存储 + reconstruct_all 插值）
- 桥接改动：`model_wrapper.py` 中 `_append_incremental` 新增 ~15 行（bypass 判断 + 参数传递）
- 配置改动：`config.py` 新增 `refresh_interval` 字段
- 预计总改动量 < 200 行，3-4 小时实现
