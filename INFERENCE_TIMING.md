# Inference Timing Documentation

## Overview

GGN 现在支持精确的推理时间记录，专门测量从输入图像到预测出高斯点云的时间，不包括数据加载、渲染、后处理等额外时间。

## 实现细节

### CUDA 同步
为了准确测量 GPU 操作时间，Benchmarker 在计时前后自动执行 `torch.cuda.synchronize()`，确保：
- 开始计时前，所有之前的 GPU 操作已完成
- 结束计时前，当前 GPU 操作已完成

### 时间记录分类

1. **encoder**: 从输入图像到高斯点云预测的完整推理时间
   - 包括：特征提取、深度预测、高斯参数生成
   - 不包括：数据加载、预处理

2. **decoder**: 从高斯点云到渲染图像的时间
   - 包括：高斯光栅化、颜色混合
   - 不包括：后处理、保存图像

### Warmup 跳过

- 通过 `test.eval_time_skip_steps` 配置跳过前 N 步的时间记录
- 这些步骤用于 GPU warmup，避免影响平均时间统计
- 默认配置：
  - GGN dl3dv: `eval_time_skip_steps: 0`
  - 可根据需要调整

## 使用方法

### 运行评测并记录时间

```bash
cd /data/zhangzicheng/workspace/SparseSplat-/GGN

python -m src.main +experiment=dl3dv \
  mode=test \
  dataset.roots=[/path/to/dl3dv] \
  checkpointing.load=/path/to/checkpoint.ckpt \
  dataset/view_sampler=evaluation \
  dataset.view_sampler.index_path=assets/dl3dv_start_0_distance_50_ctx_6v_video_0_50.json \
  test.compute_scores=true \
  test.eval_time_skip_steps=5
```

### 查看时间统计

评测完成后，时间统计会保存在输出目录：

```bash
# 完整时间记录（每个场景）
cat outputs/dl3dv/benchmark.json

# 平均时间统计
cat outputs/dl3dv/scores_all_avg.json
```

输出格式：
```json
{
  "encoder": [100, 0.234],  // [调用次数, 平均每次时间(秒)]
  "decoder": [5000, 0.012],
  "psnr": 28.5,
  "ssim": 0.89,
  "lpips": 0.12
}
```

### 终端输出示例

```
encoder: 100 calls, avg. 0.234 seconds per call
decoder: 5000 calls, avg. 0.012 seconds per call
psnr 28.5
ssim 0.89
lpips 0.12
```

## 对比 GGN 和 SparseSplat

两个项目现在都使用相同的时间记录机制：
- ✓ CUDA 同步确保准确计时
- ✓ 分离 encoder 和 decoder 时间
- ✓ Warmup 跳过避免冷启动影响
- ✓ 自动保存 JSON 格式结果

### 运行对比实验

```bash
# 1. 运行 GGN
cd /data/zhangzicheng/workspace/SparseSplat-/GGN
python -m src.main +experiment=dl3dv mode=test ...
# 查看结果: outputs/dl3dv/scores_all_avg.json

# 2. 运行 SparseSplat
cd /data/zhangzicheng/workspace/SparseSplat-/SparseSplat
python -m src.main +experiment=dl3dv mode=test ...
# 查看结果: outputs/dl3dv/scores_all_avg.json

# 3. 对比结果
python compare_results.py \
  --ggn outputs/dl3dv/scores_all_avg.json \
  --sparsesplat outputs/dl3dv/scores_all_avg.json
```

## 技术说明

### 为什么需要 CUDA 同步？

GPU 操作是异步的：
- CPU 发出 CUDA kernel 调用后立即返回
- GPU 在后台执行实际计算
- 没有同步的话，计时只测量 CPU 调度时间，不是实际计算时间

示例：
```python
# 错误方式（无同步）
start = time()
output = model(input)  # CPU 立即返回，GPU 还在计算
end = time()  # 只测量了 CPU 调度时间

# 正确方式（有同步）
torch.cuda.synchronize()  # 等待之前的 GPU 操作完成
start = time()
output = model(input)
torch.cuda.synchronize()  # 等待当前 GPU 操作完成
end = time()  # 测量了实际 GPU 计算时间
```

### Benchmarker 实现

```python
@contextmanager
def time(self, tag: str, num_calls: int = 1, sync_cuda: bool = True):
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

## 配置选项

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `test.compute_scores` | `true` | 是否计算评测指标和时间统计 |
| `test.eval_time_skip_steps` | `0` | 跳过前 N 步用于 warmup |
| `sync_cuda` | `true` | 是否使用 CUDA 同步（Benchmarker 内部参数） |

## 注意事项

1. **Encoder 时间包含**:
   - 图像特征提取
   - 深度预测
   - 高斯参数生成
   - 所有在 encoder forward 中的操作

2. **Encoder 时间不包含**:
   - 数据加载（在 test_step 之前）
   - 数据 shim 转换（batch = self.data_shim(batch)）
   - Decoder 渲染
   - 指标计算
   - 结果保存

3. **峰值内存**：
   - 同时记录在 `peak_memory.json`
   - 使用 `torch.cuda.memory_stats()` 获取

4. **批处理**：
   - GGN 和 SparseSplat 都使用 batch_size=1 进行测试
   - Decoder 时间会除以 target views 数量得到单帧时间
