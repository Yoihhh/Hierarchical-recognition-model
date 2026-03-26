混叠信号识别框架（FD-MCNN-BiLSTM）
==================================

概览
----
本项目搭建一个两阶段的混叠信号识别流程：数据预处理 → FD-MCNN 强信号检测 → 注意力增强弱信号 → BiLSTM 弱信号分类

环境要求
--------
- Python 3.9
- PyTorch 1.12.0+cu116
- TorchVision 0.13.0+cu116
- numpy、scipy

数据集
------
- 路径：`..\dataset_preprocessed\{0-30}dB`
- 划分：`train`、`test`、`val`
- 文件：`.mat`，字段名 `data`，为复数 IQ 序列；尺寸约 3,000,000×1（train），1,000,000×1（test/val）
- 混叠类别：共 20 种，例如 `2×2ASK+4FSK`、`2×16QAM+FM` 等

预处理流程
----------
1) 读取 .mat 中的复数 IQ 信号  
2) 按 2048 点不重叠切片（末尾不足丢弃）  
3) 频域支路：对每片做 STFT
4) 时域支路：拆分 I/Q ，逐片 min-max 归一化到 [0,1]  
5) 结果缓存到磁盘，便于训练直接加载

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

Training (main.py)
-----------------------------
- 数据：默认读取 `..\dataset_preprocessed\{0-30}dB` 下的 `train/val/test` 及 `metadata.csv`，可通过 `--data-root` 指定
- 流程：预处理缓存检查 → FD-MCNN → 线性投影到 (B,2,2048) → 注意力层 → BiLSTM → 弱信号 logits
- 标签：从文件名 `2×A+B` 解析强/弱基类 A/B，基类集合 {2ASK,4FSK,16QAM,AM,FM}
- 训练：BCEWithLogitsLoss，Adam, lr=0.001，batch=16，epoch=40，设备自动选择 CUDA/CPU
- 输出：实验数据保存在 `..\experimental_results\data_results`；最佳模型权重保存在 `..\experimental_results\model_results\model_best.pt`
