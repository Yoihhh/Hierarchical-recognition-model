Mixed Signal Recognition Framework（FD-MCNN-BiLSTM）
==================================

Overview
----
This project builds a two-stage mixed signal recognition framework: data preprocessing → FD-MCNN (first network) → attention-enhanced weak signal → BiLSTM (second network).

Environmental requirements
--------
- Python 3.9
- PyTorch 1.12.0+cu116
- TorchVision 0.13.0+cu116
- numpy、scipy

Dataset Description
------
- Path：`..\{your_root}\dataset\{0-30}dB\{train or test or val}`
- Dataset Partitioning：`train`、`test`、`val`
- File Partitioning：`.mat`，Field Name `data`
- Mixed categories: a total of 20 types, such as `2×2ASK+4FSK`, `2×16QAM+FM`, etc.
- This is the link to the Baidu Netdisk I shared, you can find the dataset inside: https://pan.baidu.com/s/1W6HW5kvV7BZxi27PghWXUA?pwd=nt5n Extraction code: nt5n

Preprocessing process
----------
1) Read the complex IQ signal from a .mat file
2) Slice into 2048-point non-overlapping segments
3) Perform STFT on each segment (nperseg=128, noverlap=120, nfft=1024), take the magnitude spectrum, resize to 384×256
4) Split I/Q to get 2×2048, perform min-max normalization on each segment to [0,1]
5) Cache the results to disk for direct loading during training

Output directory structure
------------
```
<save_root>/
├─ train/
│  ├─ stft/  # 384x256 Spectrum diagram，npy or pt
│  └─ iq/    # 2x2048 Normalization I/Q，npy or pt
├─ test/
└─ val/
```
----------------------------------------


FD-MCNN First Layer 
--------------------------------
- Input: IQ `2×2048`, STFT `384×256` (C=1 or 3; averaged to 1 inside the model)
- Branches:
  - Time-domain
  - Modulation-Characteristics
  - Frequency-domain
  - Energy-sensing
- Fusion:
  - Time domain modulation channel stitching
  - Frequency domain and energy splicing
- Code: `src/first_network/fd_mcnn.py`

Attention Layer 
---------------------------------------
- Input: Feature `feat_for_attention` from FD-MCNN, with shape (B, 2, 128); if it is (B, 128) it will automatically be reshaped to (B, 2, 64)
- Structure:
  - SEBlock
  - TimesFormer
- Output: (B, 2, embed_dim) Enhance features for use by the second layer BiLSTM
- Code: `src/attention_mechanism/attention_layer.py`

BiLSTM Second Layer
-------------------
- Input: CSTA outputs features with shape (B, 2, 2048)
- Structure:
  - Three-layer bidirectional LSTM
  - Obtained by temporal global average pooling (B, 32)
  - MLP
  - Softmax
- Output: logits shape (B, num_classes)
- Code: `src/second_network/bilstm_classifier.py`
