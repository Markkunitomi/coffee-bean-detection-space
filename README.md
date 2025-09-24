---
title: Coffee Bean Detection
emoji: ☕
colorFrom: red
colorTo: yellow
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: apache-2.0
---

# ☕ Coffee Bean Detection with Mask R-CNN

An interactive demo for detecting and segmenting coffee beans using a fine-tuned Mask R-CNN model.

## Features

🎯 **High Accuracy Detection**
- Precision: 99.92%
- Recall: 96.71%
- Average IoU: 90.93%

🔧 **Adjustable Parameters**
- Confidence threshold for detection sensitivity
- NMS threshold for overlap handling
- Maximum detection limits

📊 **Detailed Results**
- Individual bean segmentation masks
- Confidence scores for each detection
- Summary statistics

## How to Use

1. **Upload an Image**: Drop or select an image of coffee beans
2. **Adjust Settings** (optional): Fine-tune detection parameters
3. **View Results**: See detected beans with masks and confidence scores

## Model Details

- **Architecture**: Mask R-CNN with ResNet-50 FPN backbone
- **Framework**: PyTorch/TorchVision
- **Training**: Fine-tuned on 128 coffee bean images
- **Hardware**: Trained on Mac Mini M2 (CPU only)
- **Model Size**: 176MB in SafeTensors format

## Applications

- Coffee bean quality control
- Automated inventory counting
- Bean size and shape analysis
- Agricultural research
- Educational demonstrations

## Links

- 🤗 [Model Repository](https://huggingface.co/Kunitomi/coffee-bean-maskrcnn)
- 💻 [Source Code](https://github.com/Markkunitomi/bean-vision)
- 📖 [Documentation](https://github.com/Markkunitomi/bean-vision/blob/main/README.md)

---

Built by [Mark Kunitomi](https://huggingface.co/Kunitomi)