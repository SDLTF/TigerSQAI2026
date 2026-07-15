# TIGER Fold-0 三算法 8 小时初测

将补丁中的文件复制到现有 `tiger_segmentation_starter`：

```text
tiger_segmentation_starter/
├── train_pranet_multiclass.py
├── compare_fold0.py
├── tigerseg/
│   └── pranet_multiclass.py
└── scripts/
    └── run_8h_fold0_benchmark.ps1
```

运行前确保：

- `.venv` 中的 CUDA PyTorch 可用
- `nnunetv2`、`timm`、`segmentation-models-pytorch` 已安装
- `Dataset501_TigerCoarse` 已预处理
- `splits_final.json` 已写入病例级五折划分
- 没有另一个训练进程占用 GPU

运行：

```powershell
cd D:\Code\TigerSQ-AI-2026\tiger_segmentation_starter

powershell -ExecutionPolicy Bypass `
  -File ".\scripts\run_8h_fold0_benchmark.ps1" `
  -Hours 8 `
  -Workers 6
```

脚本依次执行：

1. 检查 `runs\unet_coarse\fold_0\metrics.json`
2. 缺失时训练 U-Net Fold 0；存在时跳过
3. 训练 nnU-Net Fold 0，使用 `nnUNetTrainer_100epochs`
4. 用剩余墙钟时间训练 PraNet-MC Fold 0
5. 统一计算 Dice、IoU、Jaccard、95HD、ASSD
6. 生成 `runs\fold0_algorithm_comparison.csv`

PraNet-MC 是原始二值 PraNet 的多类别适配版：使用 Res2Net-50、RFB、部分解码器和基于多类不确定性的反向注意力。报告中应写为 `PraNet-MC` 或 `PraNet (multiclass adaptation)`，不要把它描述为未修改的原始 PraNet。
