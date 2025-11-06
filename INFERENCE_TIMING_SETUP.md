# 推理时间记录设置完成总结

## 已完成的修改

### 1. Benchmarker 改进（GGN & SparseSplat）

为两个项目的 `src/misc/benchmarker.py` 添加了 CUDA 同步支持：

```python
@contextmanager
def time(self, tag: str, num_calls: int = 1, sync_cuda: bool = True):
    """
    Time a code block with optional CUDA synchronization.

    Args:
        tag: Name for this timing measurement
        num_calls: Number of logical calls (for averaging batch operations)
        sync_cuda: If True, synchronize CUDA before and after timing to ensure
                  accurate GPU operation timing
    """
    try:
        if sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time()
        yield
    finally:
        if sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time()
        for _ in range(num_calls):
            self.execution_times[tag].append((end_time - start_time) / num_calls)
```

**为什么需要 CUDA 同步？**
- GPU 操作是异步的，CPU 发出 kernel 调用后立即返回
- 没有同步的话只能测量 CPU 调度时间，不是实际 GPU 计算时间
- 同步确保在计时前后所有 GPU 操作都已完成

### 2. GGN Model Wrapper 修改

**文件**: `GGN/src/model/model_wrapper.py`

启用了 encoder 时间记录并添加 CUDA 同步：

```python
# Line 201-206
# Time only the encoder inference (from input images to Gaussians prediction)
with self.benchmarker.time("encoder", sync_cuda=True):
    gaussians = self.encoder(
        batch["context"],
        self.global_step,
        deterministic=False,
    )

# Line 213
# Time decoder rendering (separate from encoder inference)
with self.benchmarker.time("decoder", num_calls=v, sync_cuda=True):
    ...
```

### 3. SparseSplat Model Wrapper 修改

**文件**: `SparseSplat/src/model/model_wrapper.py`

添加了与 GGN 一致的时间记录：

```python
# Line 400-406
# Time only the encoder inference (from input images to Gaussians prediction)
with self.benchmarker.time("encoder", sync_cuda=True):
    gaussians = self.encoder(
        batch["context"],
        self.global_step,
        deterministic=False,
        visualization_dump=visualization_dump,
    )

# Line 430
# Time decoder rendering (separate from encoder inference)
with self.benchmarker.time("decoder", num_calls=v, sync_cuda=True):
    ...
```

## 时间记录的范围

### ✅ Encoder 时间包含

- 图像特征提取
- 深度预测（如果有）
- 高斯参数生成
- Encoder forward 中的所有操作

### ❌ Encoder 时间不包含

- 数据加载（DataLoader）
- 数据预处理和转换（data_shim）
- Decoder 渲染
- 评测指标计算
- 结果保存（图像、视频、PLY 等）

### ✅ Decoder 时间包含

- 高斯光栅化
- 颜色混合
- 渲染图像生成

### ❌ Decoder 时间不包含

- 后处理
- 保存图像
- 评测指标计算

## 如何使用

### 1. 运行 GGN 评测

```bash
cd /data/zhangzicheng/workspace/SparseSplat-/GGN

python -m src.main +experiment=dl3dv \
  mode=test \
  dataset.roots=[/path/to/dl3dv] \
  checkpointing.load=/path/to/ggn_checkpoint.ckpt \
  dataset/view_sampler=evaluation \
  dataset.view_sampler.index_path=assets/dl3dv_start_0_distance_50_ctx_6v_video_0_50.json \
  test.compute_scores=true \
  test.eval_time_skip_steps=5 \
  output_dir=outputs/ggn_dl3dv_6view
```

### 2. 运行 SparseSplat 评测

```bash
cd /data/zhangzicheng/workspace/SparseSplat-/SparseSplat

python -m src.main +experiment=dl3dv \
  mode=test \
  dataset.roots=[/path/to/dl3dv] \
  checkpointing.load=/path/to/sparsesplat_checkpoint.ckpt \
  dataset/view_sampler=evaluation \
  dataset.view_sampler.index_path=assets/dl3dv_start_0_distance_50_ctx_6v_video_0_50.json \
  test.compute_scores=true \
  test.eval_time_skip_steps=5 \
  output_dir=outputs/sparsesplat_dl3dv_6view
```

### 3. 对比结果

```bash
cd /data/zhangzicheng/workspace/SparseSplat-

python compare_inference_time.py \
  --ggn GGN/outputs/ggn_dl3dv_6view/dl3dv/scores_all_avg.json \
  --sparsesplat SparseSplat/outputs/sparsesplat_dl3dv_6view/dl3dv/scores_all_avg.json
```

输出示例：
```
================================================================================
GGN vs SparseSplat - dl3dv 数据集评测对比
================================================================================

📊 推理时间对比 (Encoder: 输入图像 → 高斯点云)
--------------------------------------------------------------------------------
指标                  GGN                       SparseSplat               差异
--------------------------------------------------------------------------------
Encoder 平均时间      234.5 ms                  189.2 ms                  +23.9%
Encoder 调用次数      100                       100
Decoder 平均时间      12.3 ms                   11.8 ms                   +4.2%
Decoder 调用次数      5000                      5000

🎨 渲染质量对比
--------------------------------------------------------------------------------
指标                  GGN                       SparseSplat               差异
--------------------------------------------------------------------------------
PSNR ↑               28.5234                   29.1245                   -0.6011 (SparseSplat ✓)
SSIM ↑               0.8912                    0.9023                    -0.0111 (SparseSplat ✓)
LPIPS ↓              0.1234                    0.1156                    +0.0078 (SparseSplat ✓)

================================================================================

📌 总结:
  • 推理速度: SparseSplat 更快 (19.3% 差异)
  • 渲染质量: SparseSplat 更好 (基于 PSNR)
```

## 输出文件说明

每次评测会生成以下文件：

```
outputs/
└── {output_dir}/
    └── dl3dv/
        ├── benchmark.json           # 完整的时间记录（每个场景）
        ├── peak_memory.json         # GPU 峰值内存
        ├── scores_all_avg.json      # 平均指标和时间统计
        ├── scores_psnr_all.json     # 每个场景的 PSNR
        ├── scores_ssim_all.json     # 每个场景的 SSIM
        └── scores_lpips_all.json    # 每个场景的 LPIPS
```

### scores_all_avg.json 格式

```json
{
  "encoder": [100, 0.2345],  // [调用次数, 平均每次时间(秒)]
  "decoder": [5000, 0.0123],
  "psnr": 28.5234,
  "ssim": 0.8912,
  "lpips": 0.1234
}
```

## 配置选项

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `test.compute_scores` | `true` | 是否计算评测指标和时间统计 |
| `test.eval_time_skip_steps` | `0` | 跳过前 N 步用于 GPU warmup |
| `output_dir` | - | 输出目录（必需） |

## Warmup 步骤

- 前几个推理步骤通常较慢（GPU 初始化、CUDA kernel 编译等）
- 通过 `test.eval_time_skip_steps` 跳过这些步骤
- 建议设置为 5-10
- 跳过的步骤不会计入平均时间统计

## 技术细节

### CUDA 同步的重要性

```python
# 错误方式（无同步）- 只测量 CPU 调度时间
start = time()
output = model(input)  # CPU 立即返回，GPU 还在后台计算
end = time()  # ❌ 错误的时间

# 正确方式（有同步）- 测量实际 GPU 计算时间
torch.cuda.synchronize()  # 等待之前的操作完成
start = time()
output = model(input)
torch.cuda.synchronize()  # 等待当前操作完成
end = time()  # ✅ 正确的时间
```

### 批处理说明

- 两个项目在测试时都使用 `batch_size=1`
- Decoder 时间会除以 target views 数量得到单帧时间
- 这确保了公平对比

## 进一步参考

- `GGN/INFERENCE_TIMING.md` - 详细的技术文档
- `SparseSplat/INFERENCE_TIMING.md` - 同上
- `compare_inference_time.py` - 对比脚本源码

## 故障排除

### 问题：时间记录为空

**原因**：`test.compute_scores=false`

**解决**：设置 `test.compute_scores=true`

### 问题：Encoder 时间为 0

**原因**：之前被注释掉了，现在已修复

**解决**：确保使用最新的代码

### 问题：时间不稳定

**原因**：GPU 还未 warmup

**解决**：增加 `test.eval_time_skip_steps` 到 5-10
