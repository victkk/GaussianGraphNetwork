#!/usr/bin/env python3
"""
对比 GGN 和 SparseSplat 的推理时间和评测指标。

用法:
    python compare_inference_time.py \\
        --ggn GGN/outputs/dl3dv/scores_all_avg.json \\
        --sparsesplat SparseSplat/outputs/dl3dv/scores_all_avg.json
"""

import argparse
import json
from pathlib import Path
from typing import Dict


def load_results(path: Path) -> Dict:
    """加载评测结果 JSON 文件"""
    with open(path, 'r') as f:
        return json.load(f)


def format_time(seconds: float) -> str:
    """格式化时间显示"""
    if seconds < 0.001:
        return f"{seconds * 1000000:.1f} μs"
    elif seconds < 1.0:
        return f"{seconds * 1000:.1f} ms"
    else:
        return f"{seconds:.3f} s"


def compare_results(ggn_results: Dict, sparsesplat_results: Dict):
    """对比并打印结果"""
    print("=" * 80)
    print("GGN vs SparseSplat - dl3dv 数据集评测对比")
    print("=" * 80)
    print()

    # 推理时间对比
    print("📊 推理时间对比 (Encoder: 输入图像 → 高斯点云)")
    print("-" * 80)
    print(f"{'指标':<20} {'GGN':<25} {'SparseSplat':<25} {'差异':<10}")
    print("-" * 80)

    # Encoder 时间
    if "encoder" in ggn_results and "encoder" in sparsesplat_results:
        ggn_encoder_calls, ggn_encoder_time = ggn_results["encoder"]
        ss_encoder_calls, ss_encoder_time = sparsesplat_results["encoder"]

        diff_pct = ((ggn_encoder_time - ss_encoder_time) / ss_encoder_time) * 100
        diff_sign = "+" if diff_pct > 0 else ""

        print(f"{'Encoder 平均时间':<20} "
              f"{format_time(ggn_encoder_time):<25} "
              f"{format_time(ss_encoder_time):<25} "
              f"{diff_sign}{diff_pct:.1f}%")
        print(f"{'Encoder 调用次数':<20} "
              f"{ggn_encoder_calls:<25} "
              f"{ss_encoder_calls:<25}")

    # Decoder 时间
    if "decoder" in ggn_results and "decoder" in sparsesplat_results:
        ggn_decoder_calls, ggn_decoder_time = ggn_results["decoder"]
        ss_decoder_calls, ss_decoder_time = sparsesplat_results["decoder"]

        diff_pct = ((ggn_decoder_time - ss_decoder_time) / ss_decoder_time) * 100
        diff_sign = "+" if diff_pct > 0 else ""

        print(f"{'Decoder 平均时间':<20} "
              f"{format_time(ggn_decoder_time):<25} "
              f"{format_time(ss_decoder_time):<25} "
              f"{diff_sign}{diff_pct:.1f}%")
        print(f"{'Decoder 调用次数':<20} "
              f"{ggn_decoder_calls:<25} "
              f"{ss_decoder_calls:<25}")

    print()

    # 渲染质量对比
    print("🎨 渲染质量对比")
    print("-" * 80)
    print(f"{'指标':<20} {'GGN':<25} {'SparseSplat':<25} {'差异':<10}")
    print("-" * 80)

    metrics = {
        "psnr": ("PSNR ↑", False),  # False = higher is better
        "ssim": ("SSIM ↑", False),
        "lpips": ("LPIPS ↓", True),  # True = lower is better
    }

    for key, (name, lower_is_better) in metrics.items():
        if key in ggn_results and key in sparsesplat_results:
            ggn_val = ggn_results[key]
            ss_val = sparsesplat_results[key]

            diff = ggn_val - ss_val
            diff_sign = "+" if diff > 0 else ""

            # 判断哪个更好
            if lower_is_better:
                better = "GGN ✓" if ggn_val < ss_val else "SparseSplat ✓"
            else:
                better = "GGN ✓" if ggn_val > ss_val else "SparseSplat ✓"

            print(f"{name:<20} "
                  f"{ggn_val:<25.4f} "
                  f"{ss_val:<25.4f} "
                  f"{diff_sign}{diff:.4f} ({better})")

    print()
    print("=" * 80)

    # 总结
    print("\n📌 总结:")
    if "encoder" in ggn_results and "encoder" in sparsesplat_results:
        _, ggn_time = ggn_results["encoder"]
        _, ss_time = sparsesplat_results["encoder"]
        faster = "GGN" if ggn_time < ss_time else "SparseSplat"
        speedup = abs(ggn_time - ss_time) / max(ggn_time, ss_time) * 100
        print(f"  • 推理速度: {faster} 更快 ({speedup:.1f}% 差异)")

    if "psnr" in ggn_results and "psnr" in sparsesplat_results:
        better_quality = "GGN" if ggn_results["psnr"] > sparsesplat_results["psnr"] else "SparseSplat"
        print(f"  • 渲染质量: {better_quality} 更好 (基于 PSNR)")

    print()


def main():
    parser = argparse.ArgumentParser(description="对比 GGN 和 SparseSplat 的评测结果")
    parser.add_argument("--ggn", type=str, required=True,
                        help="GGN 的 scores_all_avg.json 路径")
    parser.add_argument("--sparsesplat", type=str, required=True,
                        help="SparseSplat 的 scores_all_avg.json 路径")
    args = parser.parse_args()

    ggn_path = Path(args.ggn)
    sparsesplat_path = Path(args.sparsesplat)

    # 检查文件是否存在
    if not ggn_path.exists():
        print(f"错误: GGN 结果文件不存在: {ggn_path}")
        return

    if not sparsesplat_path.exists():
        print(f"错误: SparseSplat 结果文件不存在: {sparsesplat_path}")
        return

    # 加载结果
    ggn_results = load_results(ggn_path)
    sparsesplat_results = load_results(sparsesplat_path)

    # 对比结果
    compare_results(ggn_results, sparsesplat_results)


if __name__ == "__main__":
    main()
