# 7.15
OK 啊也是终于有时间写 README.md 了

原始数据集包含 140 张胸腔镜手术图像、140 张 16 类 coarse 分割掩膜和 140 张 31 类 fine 分割掩膜，来自 10 个匿名病例。

由于同一病例包含多个淋巴结站点图像，所以我没有随机划分图像因为这样子一个患者的信息可能同时进入训练集和验证集。

我构建了病例级五折交叉验证，每折使用 8 个病例训练、2 个病例验证，从根本上避免同一患者信息同时进入训练集和验证集。

实验统一使用 16 类 coarse 标签，输入尺寸为 640×384，区域重叠指标使用排除背景后的宏平均 Dice、IoU （Jaccard 被我吃了嗯对 ），边界指标使用 95HD 和 ASSD。

由于数据集未提供物理像素间距，所以 95HD 与 ASSD 的单位统一为 pixel。

对于预测与真实掩膜只有一方为空的情况，采用评价画布对角线作为有限惩罚，避免模型通过完全漏检小结构获得虚假的良好边界分数。

需要特别说明的是，本实验中的 PraNet 为针对 16 类语义分割设计的多分类适配版本，nnU-Net 使用 `nnUNetTrainer_100epochs`。为什么不跑 `250epochs` 的版本呢？因为我的电脑跑不动了（哭）

项目根目录规划如下：

```text
D:\Code\TigerSQ-AI-2026\
├── .venv\
├── TigerSQ-AI-2026.zip
├── TigerSQ-AI-2026-prepared\
├── tiger_segmentation_starter\
└── nnunet\
    ├── nnUNet_raw\
    ├── nnUNet_preprocessed\
    └── nnUNet_results\
```

其中：

- `.venv`：项目独立虚拟环境
- `TigerSQ-AI-2026.zip`：原始数据包
- `TigerSQ-AI-2026-prepared`：整理后的标准化数据集
- `tiger_segmentation_starter`：自编训练、评价和自动化脚本
- `nnunet`：nnU-Net v2 原始数据、预处理数据和训练结果


数据说明文件给出的结构为：

```text
tiger_challenge_part1/
├── images/
├── masks_fine/
├── masks_coarse/
└── labelmap.csv
```

其中：

- `images`：RGB PNG 手术图像
- `masks_fine`：31 类细粒度 RGB mask
- `masks_coarse`：16 类合并后 RGB mask
- `labelmap.csv`：颜色、类别名称、fine-to-coarse 映射和权重


但是实际师兄给的 ZIP 没有保留上述文件夹层级，图像和两类 mask 位于同一层，并依靠文件名后缀区分：

```text
name.png     -> coarse mask
name(1).png  -> fine mask
name(2).png  -> RGB image
```

真是神经病啊！

为解决压平结构、彩色 mask 和病例划分问题，编写 `prepare_dataset.py`。该脚本完成以下步骤：

1. 解压并扫描原始 ZIP
2. 根据无后缀、`(1)`、`(2)` 识别 coarse mask、fine mask 和原图
3. 校验三者是否一一对应
4. 校验图像与 mask 尺寸
5. 按 `labelmap.csv` 将 RGB mask 映射为单通道整数 ID mask
6. 检查是否存在未知 RGB 颜色
7. 生成 `metadata.csv`
8. 生成病例级五折 `case_folds.csv`
9. 统计 coarse 与 fine 标签的像素数量和图像出现频率

运行命令：

```powershell
python .\prepare_dataset.py `
  --zip "..\TigerSQ-AI-2026.zip" `
  --output "..\TigerSQ-AI-2026-prepared"
```

整理后的目录：

```text
TigerSQ-AI-2026-prepared/
├── images/
├── masks_coarse/
├── masks_fine/
├── metadata.csv
├── case_folds.csv
├── class_distribution_coarse.csv
├── class_distribution_fine.csv
├── labelmap.csv
├── lymph_node_station_visibility.csv
└── README.md
```

整理后的 mask 不再以 RGB 表示类别，而是：

- coarse mask：单通道 PNG，像素值为 0～15
- fine mask：单通道 PNG，像素值为 0～30

这使得 PyTorch 的 `CrossEntropyLoss` 可以直接使用 mask 中的整数类别 ID。

完整性检查结果：

```text
Images: 140
Coarse masks: 140
Fine masks: 140
```

还没完，fine 标签表声明 31 类，但 Part 1 中有 3 类没有任何标注像素：

- Right bronchial artery
- Gastric conduit
- Omentum

若直接开展 31 类训练，会出现声明类别与实际训练样本不一致的问题。相比之下，16 个 coarse 类别全部实际出现。因此本阶段选择：

```text
任务：16 类 coarse 语义分割
输入：RGB 手术图像
输出：0～15 的类别 ID mask
```

## 7.2 coarse 类别分布

| ID | 类别 | 像素数量 | 像素占比 | 出现图像数 | 图像出现率 |
|---:|---|---:|---:|---:|---:|
| 0 | Background | 103,921,365 | 37.9032% | 140 | 100.0000% |
| 1 | Respiratory Tract | 15,601,645 | 5.6904% | 103 | 73.5714% |
| 2 | Gastroesophageal | 33,429,404 | 12.1927% | 136 | 97.1429% |
| 3 | Pleura | 30,562,755 | 11.1471% | 126 | 90.0000% |
| 4 | Heart | 12,986,461 | 4.7365% | 83 | 59.2857% |
| 5 | Vessels | 673,251 | 0.2456% | 14 | 10.0000% |
| 6 | Nerves | 2,420,537 | 0.8828% | 63 | 45.0000% |
| 7 | Aorta | 13,127,032 | 4.7878% | 95 | 67.8571% |
| 8 | Azygos Vein | 2,584,543 | 0.9427% | 82 | 58.5714% |
| 9 | Superior Caval Vein | 2,018,739 | 0.7363% | 26 | 18.5714% |
| 10 | Lung | 15,894,465 | 5.7972% | 88 | 62.8571% |
| 11 | Lymphatic Tissue | 4,934,928 | 1.7999% | 101 | 72.1429% |
| 12 | Non-anatomical Other | 15,372,503 | 5.6068% | 138 | 98.5714% |
| 13 | Fatty Tissue | 9,017,707 | 3.2890% | 118 | 84.2857% |
| 14 | Pulmonary artery | 612,788 | 0.2235% | 25 | 17.8571% |
| 15 | Anatomical Other | 11,017,877 | 4.0185% | 123 | 87.8571% |

类别分布具有明显长尾特征。例如：

- Background 占 37.90%
- Gastroesophageal 占 12.19%
- Pleura 占 11.15%
- Vessels 仅占 0.2456%
- Pulmonary artery 仅占 0.2235%
- Superior Caval Vein 仅占 0.7363%
- Nerves 仅占 0.8828%

因此单纯使用未加权 Cross Entropy 容易使模型偏向背景和大面积组织。U-Net 与 PraNet-MC 中均使用基于训练折像素频率估计的类别权重：

```text
weight_c = 1 / sqrt(frequency_c)
```

随后将权重归一化并裁剪到 `[0.25, 10.0]`，在防止极端权重爆炸的同时提高稀有类别的损失贡献。


同一病例包含 14 张不同淋巴结站点的图像。如果随机拆分 140 张图像，同一患者的不同帧可能同时出现在训练集和验证集，造成患者级数据泄漏，使验证结果虚高。

因此采用病例级五折：

| 验证折 | 验证病例 | 训练规模 | 验证规模 |
|---|---|---:|---:|
| Fold 0 | center_1_case_10, center_1_case_6 | 8 个病例（112 张） | 2 个病例（28 张） |
| Fold 1 | center_1_case_12, center_1_case_13 | 8 个病例（112 张） | 2 个病例（28 张） |
| Fold 2 | center_1_case_14, center_1_case_8 | 8 个病例（112 张） | 2 个病例（28 张） |
| Fold 3 | center_1_case_7, center_1_case_9 | 8 个病例（112 张） | 2 个病例（28 张） |
| Fold 4 | center_1_case_11, center_1_case_15 | 8 个病例（112 张） | 2 个病例（28 张） |

每个病例恰好一次作为验证病例，五折结束后所有 10 个病例都得到独立验证。每折：

```text
训练：8 个病例，112 张图像
验证：2 个病例，28 张图像
```

该划分同时写入：

- 自编 PyTorch 数据集使用的 `metadata.csv`
- nnU-Net 使用的 `splits_final.json`

从而保证三种模型在完全相同的病例划分上比较。

原始图像主要为 16:9，但存在两种分辨率。若直接缩放到正方形，会扭曲解剖结构。因此采用 letterbox：

1. 按比例缩放到不超过 640×384
2. RGB 图像使用双线性插值
3. mask 使用最近邻插值
4. 剩余区域以黑色/背景类别 0 填充

网络实际输入张量：

```text
Image: [3, 384, 640], float32
Mask : [384, 640], int64
```

预测后再去除 padding，并使用最近邻插值恢复到原图尺寸，保存为与原图同名的单通道 PNG。

使用 ImageNet 均值和标准差：

```text
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

这与 U-Net 的 ImageNet 预训练 ResNet34 编码器和 PraNet-MC 的 ImageNet 预训练 Res2Net50 backbone 一致。

采用保守增强：

- 65% 概率随机旋转 -5°～5°
- 75% 概率调整亮度、对比度与颜色
- 15% 概率加入轻微高斯模糊

未使用水平翻转。原因是数据中存在左右侧解剖语义，fine 标签还明确区分 left/right 结构；如果只翻转图像和 mask 而不交换类别 ID，会制造错误标注。

为避免三种算法分别使用不同实现造成不可比，所有模型最终都导出原始分辨率整数 mask，并统一调用 `evaluate_predictions.py`。


```text
Dice_c = 2 TP_c / (2 TP_c + FP_c + FN_c)
```


```text
IoU_c = Jaccard_c = TP_c / (TP_c + FP_c + FN_c)
```

IoU 与 Jaccard 在语义分割中是同一指标。

重叠指标先在整个验证集上累积像素级混淆矩阵，再计算每个类别的指标，最后对 15 个前景类别取宏平均，排除类别 0 Background：

```text
mDice = mean(Dice_1, ..., Dice_15)
mIoU  = mean(IoU_1, ..., IoU_15)
```

这种做法避免背景类因像素数量巨大而掩盖小结构表现。

提取预测区域和真实区域的一像素内边界，计算双向表面距离集合 D：

```text
D = prediction surface -> target surface distances
  union target surface -> prediction surface distances
```

95HD 定义为：

```text
95HD = percentile_95(D)
```

它比最大 Hausdorff Distance 对单个离群像素更稳健。

```text
ASSD = mean(D)
```

ASSD 衡量预测边界与真实边界的平均对称距离。


对每张图像和每个类别：

- 预测与真实都为空：该组合未定义，不进入平均
- 只有一方为空：说明完整漏检或无中生有，使用评价画布对角线惩罚

评价画布为 640×384，对角线惩罚为：

```text
sqrt((640-1)^2 + (384-1)^2) ≈ 744.99 px
```

该策略防止模型在小类别上完全不预测，却因为跳过空掩膜而获得虚假的低边界距离。



主要文件功能：

| 文件 | 功能 |
|---|---|
| `prepare_dataset.py` | 解析压平 ZIP、解码 mask、统计分布、生成 folds |
| `dataset.py` | letterbox、增强、标准化、按 fold 读取数据 |
| `losses.py` | 多类别 Soft Dice 与 CE+Dice |
| `metrics.py` | 混淆矩阵、Dice/IoU、95HD/ASSD |
| `evaluation.py` | 统一读取预测目录并生成完整 metrics |
| `train_unet.py` | U-Net 单折训练、早停、预测和评价 |
| `prepare_nnunetv2.py` | 转换为 nnU-Net raw 格式 |
| `write_nnunet_splits.py` | 写入病例级 `splits_final.json` |
| `pranet_multiclass.py` | PraNet-MC 网络结构 |
| `train_pranet_multiclass.py` | PraNet-MC 训练与深监督 |
| `run_8h_fold0_benchmark.ps1` | 三算法 Fold 0 自动初测 |
| `run_remaining_folds.ps1` | Fold 1～4 分段运行与恢复 |
| `summarize_results.py` | 计算五折均值和样本标准差 |
| `compare_5fold.py` | 合并三种算法最终表格 |

每个 U-Net/PraNet-MC 折目录包含：

```text
best.pt
config.json
history.csv
metrics.json
predictions/
previews/
```

nnU-Net 模型文件保存在 `nnUNet_results`，统一评价结果另存到 `runs/nnunet_100e_coarse/fold_x/metrics.json`。



# 局限性与后续工作


1. 数据仅 10 个病例，病例规模较小
2. 没有独立隐藏测试集结果，当前为内部五折验证
3. nnU-Net 只训练 100 epochs，不是标准 1000 epochs
4. PraNet-MC 是自定义多分类适配版，不能与原始二值 PraNet 等同
5. 95HD 和 ASSD 的单位为 pixel，不是 mm
6. 小血管、神经和肺动脉类别严重长尾
7. 当前未进行病例级统计显著性检验
8. U-Net 和 PraNet-MC 训练脚本尚未实现完整 optimizer/scheduler checkpoint 续训
9. 当前结果以 coarse 16 类为主，尚未开展 fine 31 类任务


优先顺序建议如下：

1. 汇总三种模型的五折逐类别 Dice、IoU、95HD、ASSD
2. 重点比较 Nerves、Vessels、Pulmonary artery、Lymphatic Tissue
3. 生成每个病例的定性预测对比图
4. 对完整 1000 epochs nnU-Net 或 250 epochs nnU-Net进行补充实验
5. 对 PraNet-MC 与 U-Net进行病例级配对统计检验
6. 在保持五折一致的前提下测试更高输入分辨率
7. 研究稀有类别 patch/crop 采样与类别感知增强
8. 增加断点续训功能，提高长时间实验可靠性
9. 在 coarse 流程稳定后再扩展到 fine 31 类

---

#  附录：关键命令与脚本说明

## 1 激活环境

```powershell
cd D:\Code\TigerSQ-AI-2026
.\.venv\Scripts\Activate.ps1
cd .\tiger_segmentation_starter
```

## 2 验证 CUDA

```powershell
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0)); print(torch.cuda.get_arch_list())"
```

## 3 整理数据

```powershell
python .\prepare_dataset.py `
  --zip "..\TigerSQ-AI-2026.zip" `
  --output "..\TigerSQ-AI-2026-prepared"
```

## 4 U-Net 单折

```powershell
python .\train_unet.py `
  --data-root "..\TigerSQ-AI-2026-prepared" `
  --label-mode coarse `
  --fold 0 `
  --encoder resnet34 `
  --epochs 80 `
  --patience 15 `
  --batch-size 2 `
  --workers 12 `
  --lr 1e-4 `
  --weight-decay 1e-4 `
  --output ".\runs\unet_coarse\fold_0"
```

## 5 nnU-Net 环境变量

```powershell
$env:nnUNet_raw = "D:\Code\TigerSQ-AI-2026\nnunet\nnUNet_raw"
$env:nnUNet_preprocessed = "D:\Code\TigerSQ-AI-2026\nnunet\nnUNet_preprocessed"
$env:nnUNet_results = "D:\Code\TigerSQ-AI-2026\nnunet\nnUNet_results"
$env:nnUNet_n_proc_DA = "12"
$env:nnUNet_compile = "false"
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
```

## 6 nnU-Net 100 epochs 单折

```powershell
nnUNetv2_train 501 2d 0 `
  -tr nnUNetTrainer_100epochs `
  -device cuda
```

## 7 PraNet-MC 单折

```powershell
python .\train_pranet_multiclass.py `
  --data-root "..\TigerSQ-AI-2026-prepared" `
  --label-mode coarse `
  --fold 0 `
  --backbone res2net50_26w_4s `
  --epochs 80 `
  --patience 12 `
  --batch-size 2 `
  --workers 12 `
  --lr 1e-4 `
  --weight-decay 1e-4 `
  --output ".\runs\pranet_mc_coarse\fold_0"
```

## 8 分阶段运行一个 Fold

```powershell
powershell -ExecutionPolicy Bypass `
  -File ".\scripts\run_remaining_folds.ps1" `
  -StartFold 1 `
  -EndFold 1 `
  -Workers 12
```

## 9 最终结果位置

```text
runs\fivefold_algorithm_comparison.csv
```

最终可填写结果：

```text
U-Net
Dice     0.4807 ± 0.0683
IoU      0.3546 ± 0.0641
Jaccard  0.3546 ± 0.0641
95HD     338.5266 ± 29.6371 px
ASSD     286.9640 ± 26.5272 px

nnU-Net (100 epochs)
Dice     0.4279 ± 0.0414
IoU      0.3044 ± 0.0383
Jaccard  0.3044 ± 0.0383
95HD     336.7707 ± 31.8634 px
ASSD     283.4178 ± 31.4826 px

PraNet-MC
Dice     0.5052 ± 0.0770
IoU      0.3733 ± 0.0720
Jaccard  0.3733 ± 0.0720
95HD     309.2480 ± 37.4891 px
ASSD     260.1332 ± 35.7451 px
```
