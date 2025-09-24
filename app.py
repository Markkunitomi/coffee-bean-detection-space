import gradio as gr
import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.transforms import functional as F
import torchvision.ops as ops
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import colorsys
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

# Pre-generate colors for visualization
def generate_colors(n=20):
    """Generate n distinct colors using HSV color space."""
    colors = []
    for i in range(n):
        hue = i / n
        saturation = 0.8 + 0.2 * (i % 2)  # Alternate between 0.8 and 1.0
        value = 0.8 + 0.2 * ((i + 1) % 2)  # Alternate between 0.8 and 1.0
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        colors.append(tuple(int(255 * c) for c in rgb))
    return colors

COLORS = generate_colors(20)

def draw_detection_pil(image, predictions, bean_count, show_confidence=True):
    """Fast PIL-based visualization instead of matplotlib."""
    # Create a copy of the image to draw on
    result_img = image.copy()
    draw = ImageDraw.Draw(result_img)

    # Try to load a font, fall back to default if not available
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        try:
            font = ImageFont.load_default()
        except:
            font = None

    # Draw each detection
    for i in range(bean_count):
        color = COLORS[i % len(COLORS)]

        # Get detection data
        box = predictions['boxes'][i].cpu().numpy()
        score = predictions['scores'][i].cpu().item()
        mask = predictions['masks'][i][0].cpu().numpy()

        x1, y1, x2, y2 = box.astype(int)

        # Create mask overlay - resize mask to match image size
        mask_resized = Image.fromarray((mask * 255).astype(np.uint8), mode='L').resize(result_img.size, Image.NEAREST)

        # Create colored overlay for this mask
        colored_mask = Image.new('RGBA', result_img.size, (*color, 120))  # Semi-transparent colored overlay

        # Apply mask transparency
        colored_mask.putalpha(mask_resized)

        # Composite the mask overlay onto the result image
        result_img = result_img.convert('RGBA')
        result_img = Image.alpha_composite(result_img, colored_mask)
        result_img = result_img.convert('RGB')
        draw = ImageDraw.Draw(result_img)

        # Draw bounding box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        # Draw confidence score or bean number
        if show_confidence:
            label = f"{score:.2f}"
        else:
            label = f"#{i+1}"

        if font:
            # Get text size for background
            bbox = draw.textbbox((0, 0), label, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        else:
            text_width, text_height = 30, 15  # Fallback size

        # Draw text background
        text_bg_coords = [x1, y1 - text_height - 4, x1 + text_width + 8, y1]
        draw.rectangle(text_bg_coords, fill=color)

        # Draw text
        draw.text((x1 + 4, y1 - text_height - 2), label, fill='white', font=font)

    return result_img

def predict_beans(image, confidence_threshold, nms_threshold, max_detections, show_confidence):
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

    # Create fast PIL-based visualization
    if bean_count > 0:
        result_image = draw_detection_pil(image, filtered_predictions, bean_count, show_confidence)
    else:
        result_image = image.copy()

    # Create summary text
    if bean_count > 0:
        avg_confidence = filtered_predictions['scores'].mean().item()
        summary = f"**Detected {bean_count} coffee beans** with {avg_confidence:.1%} average confidence"
    else:
        summary = "**No beans detected.** Try lowering the confidence threshold or check image quality."

    return result_image, summary

# Example images
examples = [
    ["examples/green_beans.png", 0.5, 0.5, 300, True],
    ["examples/roasted_beans.png", 0.5, 0.3, 300, True],
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
                    value=300,
                    step=10,
                    label="Maximum Detections",
                    info="Limit total number of detections shown"
                )

                show_confidence = gr.Checkbox(
                    value=True,
                    label="Show Confidence Scores",
                    info="Show confidence scores instead of bean numbers"
                )
            detect_btn = gr.Button("🔍 Detect Beans", variant="primary", size="lg")

        with gr.Column(scale=1):
            # Output
            output_image = gr.Image(label="Detection Results", height=400)
            results_text = gr.Markdown()

    # Event handlers
    detect_btn.click(
        fn=predict_beans,
        inputs=[input_image, confidence_threshold, nms_threshold, max_detections, show_confidence],
        outputs=[output_image, results_text]
    )

    # Auto-detect when image is uploaded
    input_image.change(
        fn=predict_beans,
        inputs=[input_image, confidence_threshold, nms_threshold, max_detections, show_confidence],
        outputs=[output_image, results_text]
    )

    # Examples section
    gr.Markdown("## 📸 Try These Examples")
    gr.Examples(
        examples=examples,
        inputs=[input_image, confidence_threshold, nms_threshold, max_detections, show_confidence],
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

    Built by [Mark Kunitomi](https://huggingface.co/Kunitomi)
    """)

if __name__ == "__main__":
    demo.launch()