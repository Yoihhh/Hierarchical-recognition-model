混叠信号识别框架（FD-MCNN-BiLSTM）
==================================

概览
----
本项目搭建一个两阶段的混叠信号识别流程：数据预处理 → FD-MCNN 强信号检测 → 注意力增强弱信号 → BiLSTM 弱信号分类。

环境要求
--------
- Python 3.9
- PyTorch 1.12.0+cu116
- TorchVision 0.13.0+cu116
- numpy、scipy

数据集
------
- 路径：`E:\1My_Research_Content\SCI_code\dataset\30dB`
- 划分：`train`、`test`、`val`
- 文件：`.mat`，字段名 `data`，为复数 IQ 序列；尺寸约 3,000,000×1（train），1,000,000×1（test/val）
- 混叠类别：共 20 种，例如 `2×2ASK+4FSK`、`2×16QAM+FM` 等

预处理流程
----------
1) 读取 .mat 中的复数 IQ 信号  
2) 按 2048 点不重叠切片（末尾不足丢弃）  
3) 频域支路：对每片做 STFT（nperseg=128，noverlap=120，nfft=1024），取幅值谱，缩放到 384×256  
4) 时域支路：拆分 I/Q 得到 2×2048，逐片 min-max 归一化到 [0,1]（常数片置零）  
5) 结果缓存到磁盘，便于训练直接加载

输出目录结构
------------
```
<save_root>/
├─ train/
│  ├─ stft/  # 384x256 频谱图，npy 或 pt
│  └─ iq/    # 2x2048 归一化 I/Q，npy 或 pt
├─ test/
└─ val/
```
文件命名：`{原始文件名}_{切片序号}.npy`（或 .pt）

快速上手
--------
```bash
# 1) 可选：创建虚拟环境
python -m venv .venv && .venv\Scripts\activate

# 2) 安装依赖
pip install numpy scipy
# 如需保存为 pt 或后续训练：
pip install torch==1.12.0+cu116 torchvision==0.13.0+cu116 -f https://download.pytorch.org/whl/cu116/torch_stable.html

# 3) 运行预处理
python -m data_preprocess.preprocess ^
  --data-root "E:\1My_Research_Content\SCI_code\dataset\30dB" ^
  --save-root "E:\1My_Research_Content\SCI_code\dataset\30dB_preprocessed" ^
  --format npy
```
完成后可在 `<save_root>` 中看到 stft/iq 子目录。

仓库结构
--------
```
FD-MCNN-BiLSTM/
├─ readme.md
└─ src/
   ├─ data_preprocess/
   │  ├─ __init__.py
   │  └─ preprocess.py
   ├─ first_network/
   │  ├─ __init__.py
   │  └─ fd_mcnn.py
   ├─ attention_mechanism/
   │  ├─ __init__.py
   │  └─ attention_layer.py
   └─ second_network/
      ├─ __init__.py
      └─ bilstm_classifier.py
```

后续工作
--------
- 补充具体训练脚本
- 增加数据集配置、日志与可视化
- 编写单元测试与端到端实验脚本

----------------------------------------


FD-MCNN First Layer (四分支概述)
--------------------------------
- 输入 / Inputs: IQ `2×2048`, STFT `384×256` (C=1 or 3; averaged to 1 inside the model).
- 分支 / Branches:
  - Time-Domain: Conv1d(2→16,k7)→BN→ReLU→Conv1d(16→32,k5)→BN→ReLU→Conv1d(32→64,k3)→BN→ReLU
  - Modulation: SeparableConv1d(2→16,k5)→BN→ReLU→SeparableConv1d(16→32,k3)→BN→ReLU→ChannelAttention(r=8)
  - Frequency: Conv2d(1→16,k3×1)→BN→ReLU→Conv2d(16→32,k1×3)→BN→ReLU→MaxPool(2×2)
  - Energy: Conv2d(1→16,k5×1)→BN→ReLU→Conv2d(16→32,k1×5)→BN→ReLU→MaxPool(2×2)
- 融合 / Fusion:
  - 时域+调制通道拼接 → 1×1 Conv 降维 → GAP → 向量 v_tm
  - 频域×能量 逐元素乘 → 1×1 Conv 降维 → GAP → 向量 v_fe
- v_tm 与 v_fe 拼接，经 MLP 注意力加权得到 `feat_for_attention`；线性层输出 logits (默认 20 类)。
- 代码 / Code: `src/first_network/fd_mcnn.py`, 类 `FDMcnnDetector`。
- 示例 / Usage:
  ```python
  from first_network.fd_mcnn import FDMcnnDetector
  import torch
  model = FDMcnnDetector(num_classes=20, reduce_channels=64)
  iq = torch.randn(2, 2, 2048)
  stft = torch.randn(2, 3, 384, 256)
  logits, feat = model(iq, stft)
  print(logits.shape, feat.shape)  # torch.Size([2,20]) torch.Size([2,128])
  ```

Attention Layer (SEBlock + TimesFormer)
---------------------------------------
- 输入 / Input: 来自 FD-MCNN 的特征 `feat_for_attention`，形状 (B, 2, 128)；若为 (B, 128) 将自动 reshape 为 (B, 2, 64)。
- 结构 / Structure:
  - SEBlock: 全局池化 → 1×1 Conv 压缩 → ReLU → 1×1 Conv 恢复 → Sigmoid，加权原特征。
  - TimesFormer: Patch Embedding → Time Attention → Space Attention → LayerNorm → MLP → 残差输出。
- 输出 / Output: (B, 2, embed_dim) 增强特征，供第二层 BiLSTM 使用。
- 代码 / Code: `src/attention_mechanism/attention_layer.py`, 类 `WeakSignalAttention`。

BiLSTM Second Layer
-------------------
- 输入 / Input: CSTA 输出特征，形状 (B, 2, 2048)（或 (B, 2048, 2)）。
- 结构 / Structure:
  - 三层双向 LSTM：hidden 128 → 64 → 16（输出通道 256 → 128 → 32）。
  - 时间维全局平均池化得到 (B, 32)。
  - MLP: 32→64→16→num_classes（默认 20）。
  - Softmax 在损失或推理阶段计算。
- 输出 / Output: logits 形状 (B, num_classes)。
- 代码 / Code: `src/second_network/bilstm_classifier.py`, 类 `BiLstmClassifier`。

End-to-End Training (main.py)
-----------------------------
- 数据：默认读取 `E:\1My_Research_Content\SCI_code\dataset_preprocessed\30dB` 下的 `train/val/test` 及 `metadata.csv`，可通过 `--data-root` 指定。
- 流程：预处理缓存检查 → FD-MCNN（强信号 logits + feat_for_attention）→ 线性投影到 (B,2,2048) → 注意力层 → BiLSTM → 弱信号 logits。
- 标签：从文件名 `2×A+B` 解析强/弱基类 A/B，基类集合 {2ASK,4FSK,16QAM,AM,FM}，生成 5 维 one-hot，强弱各一份。
- 训练：BCEWithLogitsLoss（强/弱各一），Adam lr=0.001，batch=16，epoch=50，设备自动选择 CUDA/CPU。
- 输出：在控制台打印 loss/准确率；结果与混淆矩阵保存在 `experimental_results/data_results`；最佳模型权重保存在 `experimental_results/model_results/model_best.pt`。
- 运行示例：
  ```bash
  python main.py ^
    --data-root "E:\1My_Research_Content\SCI_code\dataset_preprocessed\30dB" ^
    --results-root "E:\1My_Research_Content\SCI_code\experimental_results\data_results" ^
    --models-root "E:\1My_Research_Content\SCI_code\experimental_results\model_results" ^
    --batch-size 16 --epochs 50 --lr 0.001
  ```
