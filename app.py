import gradio as gr
from transformers import pipeline
import numpy as np
from PIL import Image

# Load the depth estimation pipeline using the Depth Anything V2 Large model
depth_pipeline = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Large-hf")

# Global variables to store current state
current_original_image = None
current_depth_image = None

def process_depth_to_pil(depth_pil):
    """
    Normalize the depth PIL image to a grayscale 0-255 PIL image.
    """
    depth_array = np.array(depth_pil, dtype=np.float32)
    # If the depth image has multiple channels, take the first channel
    if depth_array.ndim == 3:
        depth_array = depth_array[..., 0]
    depth_min = depth_array.min()
    depth_max = depth_array.max()
    depth_norm = (depth_array - depth_min) / (depth_max - depth_min + 1e-6)
    depth_255 = depth_norm * 255.0
    depth_img = Image.fromarray(depth_255.astype(np.uint8), mode="L")
    return depth_img

def apply_chromo_stereopsis(
    original_img: Image.Image,
    depth_img: Image.Image,
    threshold: float,
    feather: float,
    black_level: float,
    white_level: float
) -> Image.Image:
    """
    Apply chromostereopsis red-blue effect to the original image using a depth map.
    """
    # Convert original image to grayscale
    gray = np.array(original_img.convert("L"), dtype=np.float32)
    # Levels adjustment
    denom = (white_level - black_level) if (white_level > black_level) else 1e-6
    adjusted_gray = (gray - black_level) / denom * 255.0
    adjusted_gray = np.clip(adjusted_gray, 0, 255)
    adjusted_gray_01 = adjusted_gray / 255.0
    # Depth-based blend factor
    depth_arr = np.array(depth_img, dtype=np.float32)
    half_feather = feather / 2.0
    blend = np.clip((depth_arr - (threshold - half_feather)) / (feather + 1e-6), 0, 1)
    # Red / Blue channels
    red = blend * adjusted_gray_01 * 255.0
    blue = (1.0 - blend) * adjusted_gray_01 * 255.0
    # Create the output image
    h, w = gray.shape
    output = np.zeros((h, w, 3), dtype=np.uint8)
    output[..., 0] = np.clip(red, 0, 255).astype(np.uint8)   # Red channel
    output[..., 1] = 0                                       # Green channel (always 0)
    output[..., 2] = np.clip(blue, 0, 255).astype(np.uint8)  # Blue channel
    return Image.fromarray(output, mode="RGB")

def generate_depth_map(input_image):
    """
    Generate a depth map from the input image and create an initial chromostereopsis output.
    """
    global current_original_image, current_depth_image
    if input_image is None:
        current_original_image = None
        current_depth_image = None
        return None, None
    try:
        # Store original image
        current_original_image = input_image
        # Run depth estimation
        result = depth_pipeline(input_image)
        depth_pil = result["depth"]
        # Normalize depth map
        current_depth_image = process_depth_to_pil(depth_pil)
        # Use default parameters for initial chromostereopsis effect
        threshold = 50.0 * 255.0 / 100.0
        feather = 10.0 * 255.0 / 100.0
        black_level = 0.0
        white_level = 255.0
        chromostereo_result = apply_chromo_stereopsis(
            current_original_image,
            current_depth_image,
            threshold,
            feather,
            black_level,
            white_level
        )
        return current_depth_image.convert("RGB"), chromostereo_result
    except Exception as e:
        print(f"Error during depth generation: {e}")
        current_original_image = None
        current_depth_image = None
        return None, None

def update_chromostereopsis(threshold_percent, feather_percent, black_level, white_level):
    """
    Update the chromostereopsis effect with new parameters for live preview.
    """
    global current_original_image, current_depth_image
    if current_original_image is None or current_depth_image is None:
        return None
    try:
        threshold = (threshold_percent / 100.0) * 255.0
        feather = (feather_percent / 100.0) * 255.0
        chromostereo_result = apply_chromo_stereopsis(
            current_original_image,
            current_depth_image,
            threshold,
            feather,
            black_level,
            white_level
        )
        return chromostereo_result
    except Exception as e:
        print(f"Error during chromostereopsis update: {e}")
        return None

def clear_results():
    """
    Clear stored images and reset outputs.
    """
    global current_original_image, current_depth_image
    current_original_image = None
    current_depth_image = None
    return None, None

with gr.Blocks(title="ChromoStereoizer", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# ChromoStereoizer")
    with gr.Row():
        # Left column: Input image and generate button
        with gr.Column(scale=1):
            input_image = gr.Image(
                label="Upload Image",
                type="pil",
                height=400
            )
            generate_btn = gr.Button(
                "Generate Depth Map",
                variant="primary",
                size="lg"
            )
        # Right column: Outputs and controls
        with gr.Column(scale=1):
            gr.Markdown("**Depth Map**")
            depth_output = gr.Image(
                type="pil",
                height=400,
                interactive=False,
                show_download_button=True,
                show_label=False
            )
            gr.Markdown("**ChromoStereoizer Result**")
            chromo_output = gr.Image(
                type="pil",
                height=400,
                interactive=False,
                show_download_button=True,
                show_label=False
            )
            with gr.Row():
                threshold_slider = gr.Slider(
                    minimum=1,
                    maximum=100,
                    value=50,
                    step=1,
                    label="Threshold (%)"
                )
                feather_slider = gr.Slider(
                    minimum=0,
                    maximum=100,
                    value=10,
                    step=1,
                    label="Feather (%)"
                )
            with gr.Row():
                black_level_slider = gr.Slider(
                    minimum=0,
                    maximum=255,
                    value=0,
                    step=1,
                    label="Black Level"
                )
                white_level_slider = gr.Slider(
                    minimum=0,
                    maximum=255,
                    value=255,
                    step=1,
                    label="White Level"
                )
            with gr.Row():
                clear_btn = gr.Button("Clear", variant="secondary")
    # Event bindings
    generate_btn.click(
        fn=generate_depth_map,
        inputs=[input_image],
        outputs=[depth_output, chromo_output],
        show_progress=True
    )
    threshold_slider.change(
        fn=update_chromostereopsis,
        inputs=[threshold_slider, feather_slider, black_level_slider, white_level_slider],
        outputs=chromo_output,
        show_progress=False
    )
    feather_slider.change(
        fn=update_chromostereopsis,
        inputs=[threshold_slider, feather_slider, black_level_slider, white_level_slider],
        outputs=chromo_output,
        show_progress=False
    )
    black_level_slider.change(
        fn=update_chromostereopsis,
        inputs=[threshold_slider, feather_slider, black_level_slider, white_level_slider],
        outputs=chromo_output,
        show_progress=False
    )
    white_level_slider.change(
        fn=update_chromostereopsis,
        inputs=[threshold_slider, feather_slider, black_level_slider, white_level_slider],
        outputs=chromo_output,
        show_progress=False
    )
    clear_btn.click(
        fn=clear_results,
        inputs=[],
        outputs=[depth_output, chromo_output]
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
