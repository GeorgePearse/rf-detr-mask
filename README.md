# 🎭 RF-DETR-MASK: Real-time Instance Segmentation with Transformer Magic

<div align="center">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Python-3.13-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/COCO%20AP-60%2B-green" alt="COCO AP">
  <img src="https://img.shields.io/badge/License-MIT-yellow" alt="License">
</div>

## ✨ What is RF-DETR-MASK?

**RF-DETR-MASK** takes the lightning-fast object detection of RF-DETR and adds pixel-perfect instance segmentation—think of it as giving your AI superhuman vision that not only spots objects but traces their exact silhouettes! 🎯

### 🚀 Key Features

- **⚡ Real-time Performance**: Maintains RF-DETR's blazing speed while adding segmentation
- **🎯 High Accuracy**: Exceeds 60 AP on [Microsoft COCO](https://cocodataset.org/#home)
- **🔧 Production Ready**: Optimized for edge deployment with ONNX export support
- **🌈 Multi-scale Features**: FPN-style architecture for capturing fine details
- **🧩 Modular Design**: Easy to extend and customize for your needs

### 🏗️ Architecture Highlights

Built on the shoulders of giants, RF-DETR-MASK combines:
- **DINOv2 Backbone**: State-of-the-art visual representations from Meta AI
- **Deformable Attention**: Smart focus on relevant image regions
- **Lightweight Mask Head**: Inspired by [Facebook DETR](https://github.com/facebookresearch/detr/blob/main/models/segmentation.py)
- **End-to-End Training**: No complex post-processing needed!

**🎭 RF-DETR-MASK = Edge-friendly Performance + Pixel-precise Boundaries**
