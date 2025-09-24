import gradio as gr
import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.transforms import functional as F
import torchvision.ops as ops
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import cm
import io
import requests
from huggingface_hub import hf_hub_download

# Download model from Hugging Face Hub
@torch.no_grad()
def load_model():
    model_path = hf_hub_download(
        repo_id="Kunitomi/coffee-bean-maskrcnn",
        filename="maskrcnn_coffeebeans_v1.safetensors"
    )

    model = maskrcnn_resnet50_fpn(num_classes=2)  # background + bean

    from safetensors.torch import load_file
    state_dict = load_file(model_path)
    model.load_state_dict(state_dict)
    model.eval()

    return model

# Load model once at startup
model = load_model()

def predict_beans(image, confidence_threshold, nms_threshold, max_detections):
    """Run inference on uploaded image."""
    if image is None:
        return None, "Please upload an image first."

    # Convert to PIL if needed
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    # Convert to RGB
    image = image.convert('RGB')

    # Preprocess image
    image_tensor = F.to_tensor(image).unsqueeze(0)

    # Run inference
    with torch.no_grad():
        predictions = model(image_tensor)[0]

    # Apply NMS
    keep = ops.nms(predictions['boxes'], predictions['scores'], nms_threshold)
    predictions = {k: v[keep] for k, v in predictions.items()}

    # Filter by confidence threshold
    mask = predictions['scores'] > confidence_threshold
    filtered_predictions = {
        'boxes': predictions['boxes'][mask],
        'labels': predictions['labels'][mask],
        'scores': predictions['scores'][mask],
        'masks': predictions['masks'][mask]
    }

    # Limit number of detections
    if len(filtered_predictions['boxes']) > max_detections:
        # Keep top detections by confidence
        top_indices = torch.topk(filtered_predictions['scores'], max_detections)[1]
        filtered_predictions = {k: v[top_indices] for k, v in filtered_predictions.items()}

    bean_count = len(filtered_predictions['boxes'])

    # Create visualization
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(image)
    ax.axis('off')

    # Colors for visualization
    colors = plt.colormaps.get_cmap('tab20')

    # Draw detections
    for i in range(bean_count):
        color = colors(i % 20)

        # Draw mask
        mask = filtered_predictions['masks'][i][0].cpu().numpy()
        masked = np.ma.masked_where(mask < 0.5, mask)
        ax.imshow(masked, alpha=0.4, cmap=plt.cm.colors.ListedColormap([color]))

        # Draw bounding box
        box = filtered_predictions['boxes'][i].cpu().numpy()
        x1, y1, x2, y2 = box
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor=color, facecolor='none'
        )
        ax.add_patch(rect)

        # Add confidence score
        score = filtered_predictions['scores'][i].cpu().item()
        ax.text(
            x1, y1 - 5, f'{score:.2f}',
            color='white', fontsize=10, weight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.8)
        )

    ax.set_title(f'Detected {bean_count} Coffee Beans', fontsize=16, fontweight='bold')

    plt.tight_layout()

    # Convert plot to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    result_image = Image.open(buf)
    plt.close()

    # Create summary text
    if bean_count > 0:
        avg_confidence = filtered_predictions['scores'].mean().item()
        summary = f"**Detected {bean_count} coffee beans** with {avg_confidence:.1%} average confidence"
    else:
        summary = "**No beans detected.** Try lowering the confidence threshold or check image quality."

    return result_image, summary

# Example images
examples = [
    ["examples/green_beans.png", 0.5, 0.5, 300],
    ["examples/roasted_beans.png", 0.5, 0.3, 300],
]

# Create Gradio interface
with gr.Blocks(title="Coffee Bean Detection", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # ☕ Coffee Bean Detection with Mask R-CNN

    Upload an image of coffee beans to detect and segment individual beans using a fine-tuned Mask R-CNN model.
    """)

    with gr.Row():
        with gr.Column(scale=1):
            # Input controls
            input_image = gr.Image(
                type="pil",
                label="Upload Coffee Bean Image",
                height=400
            )

            with gr.Accordion("Advanced Settings", open=False):
                confidence_threshold = gr.Slider(
                    minimum=0.1,
                    maximum=0.9,
                    value=0.5,
                    step=0.05,
                    label="Confidence Threshold",
                    info="Higher values = fewer but more confident detections"
                )

                nms_threshold = gr.Slider(
                    minimum=0.1,
                    maximum=0.8,
                    value=0.5,
                    step=0.05,
                    label="NMS Threshold",
                    info="Lower values = less overlap between detections"
                )

                max_detections = gr.Slider(
                    minimum=10,
                    maximum=300,
                    value=100,
                    step=10,
                    label="Maximum Detections",
                    info="Limit total number of detections shown"
                )

            detect_btn = gr.Button("🔍 Detect Beans", variant="primary", size="lg")

        with gr.Column(scale=1):
            # Output
            output_image = gr.Image(label="Detection Results", height=400)
            results_text = gr.Markdown()

    # Event handlers
    detect_btn.click(
        fn=predict_beans,
        inputs=[input_image, confidence_threshold, nms_threshold, max_detections],
        outputs=[output_image, results_text]
    )

    # Auto-detect when image is uploaded
    input_image.change(
        fn=predict_beans,
        inputs=[input_image, confidence_threshold, nms_threshold, max_detections],
        outputs=[output_image, results_text]
    )

    # Examples section
    gr.Markdown("## 📸 Try These Examples")
    gr.Examples(
        examples=examples,
        inputs=[input_image, confidence_threshold, nms_threshold, max_detections],
        outputs=[output_image, results_text],
        fn=predict_beans,
        cache_examples=True
    )

    # Footer
    gr.Markdown("""
    ---
    **Model Details:**
    - Architecture: Mask R-CNN with ResNet-50 FPN backbone
    - Framework: PyTorch/TorchVision
    - Fine-tuned on 128 coffee bean images
    - Model size: 176MB (SafeTensors format)

    **Links:**
    - 🤗 [Model on Hugging Face](https://huggingface.co/Kunitomi/coffee-bean-maskrcnn)
    - 💻 [Code Repository](https://github.com/Markkunitomi/bean-vision)

    Built by [Mark Kunitomi](https://huggingface.co/Kunitomi)
    """)

if __name__ == "__main__":
    demo.launch()