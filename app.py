import gradio as gr
import numpy as np
from PIL import Image
from transformers import pipeline

try:
    import cv2
except ImportError:
    cv2 = None

# Load depth estimation model once
depth_pipe = pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Large-hf")

# Global state
current_original_image = None
current_depth_norm = None
current_depth_map_pil = None


def preprocess_depth(depth_norm, smoothing_radius):
    """Smooth the depth map using bilateral filtering if radius > 0 and cv2 is available."""
    if smoothing_radius > 0 and cv2 is not None:
        depth_uint8 = (depth_norm * 255.0).astype(np.uint8)
        sigma = max(smoothing_radius * 10.0, 1.0)
        smoothed = cv2.bilateralFilter(depth_uint8, d=5, sigmaColor=sigma, sigmaSpace=sigma)
        return smoothed.astype(np.float32) / 255.0
    return depth_norm


def apply_effect(threshold, depth_scale, feather, red_brightness, blue_brightness, gamma,
                 black_level_percent, white_level_percent, smoothing_percent):
    """
    Apply chromostereopsis effect using adjustable parameters.
    threshold: percentage [0,100] controlling blend midpoint.
    depth_scale: percentage [0,100] controlling steepness of logistic curve.
    feather: percentage [0,100] affecting the smoothness of the transition.
    red_brightness, blue_brightness: percentages [0,100] controlling channel intensities.
    gamma: percentage [0,100] mapped to gamma range [0.1, 3.0].
    black_level_percent, white_level_percent: percentages mapped to 0..255 levels.
    smoothing_percent: percentage [0,100] mapped to bilateral filter radius.
    """
    global current_original_image, current_depth_norm
    if current_original_image is None or current_depth_norm is None:
        return None

    # Levels adjustment
    black_level = black_level_percent * 2.55
    white_level = white_level_percent * 2.55
    gray = np.array(current_original_image.convert("L"), dtype=np.float32)
    denom = max(white_level - black_level, 1e-6)
    adjusted_gray = (gray - black_level) / denom
    adjusted_gray = np.clip(adjusted_gray, 0.0, 1.0)

    # Gamma correction
    gamma_val = 0.1 + (gamma / 100.0) * 2.9
    adjusted_gray = np.clip(adjusted_gray ** gamma_val, 0.0, 1.0)

    # Smooth depth map
    smoothing_radius = smoothing_percent / 10.0
    depth_smoothed = preprocess_depth(current_depth_norm, smoothing_radius)

    # Compute blend factor using logistic function
    threshold_norm = threshold / 100.0
    steepness = max(depth_scale, 1e-3)
    feather_norm = feather / 100.0
    steepness_adj = steepness / (feather_norm * 10.0 + 1.0)
    blend = 1.0 / (1.0 + np.exp(-steepness_adj * (depth_smoothed - threshold_norm)))

    # Map brightness to factors (0-2)
    red_factor = red_brightness / 50.0
    blue_factor = blue_brightness / 50.0

    red_channel = red_factor * adjusted_gray * blend
    blue_channel = blue_factor * adjusted_gray * (1.0 - blend)

    red_img = np.clip(red_channel * 255.0, 0, 255).astype(np.uint8)
    blue_img = np.clip(blue_channel * 255.0, 0, 255).astype(np.uint8)

    h, w = red_img.shape
    output = np.zeros((h, w, 3), dtype=np.uint8)
    output[..., 0] = red_img
    output[..., 1] = 0
    output[..., 2] = blue_img

    return Image.fromarray(output, mode="RGB")


def generate_depth_map(input_image):
    """Generate normalized depth map and initial effect image."""
    global current_original_image, current_depth_norm, current_depth_map_pil
    if input_image is None:
        current_original_image = None
        current_depth_norm = None
        current_depth_map_pil = None
        return None, None
    current_original_image = input_image
    # Run depth estimation
    result = depth_pipe(input_image)
    depth = np.array(result["depth"], dtype=np.float32)
    depth -= depth.min()
    max_val = depth.max()
    if max_val > 0:
        depth /= max_val
    current_depth_norm = depth
    current_depth_map_pil = Image.fromarray((depth * 255.0).astype(np.uint8), mode="L")
    # Default effect parameters
    effect = apply_effect(
        threshold=50,
        depth_scale=50,
        feather=10,
        red_brightness=50,
        blue_brightness=50,
        gamma=50,
        black_level_percent=0,
        white_level_percent=100,
        smoothing_percent=0,
    )
    return current_depth_map_pil.convert("RGB"), effect


def update_effect(threshold, depth_scale, feather, red_brightness, blue_brightness,
                  gamma, black_level, white_level, smoothing):
    """Update the effect when any slider changes."""
    return apply_effect(
        threshold=threshold,
        depth_scale=depth_scale,
        feather=feather,
        red_brightness=red_brightness,
        blue_brightness=blue_brightness,
        gamma=gamma,
        black_level_percent=black_level,
        white_level_percent=white_level,
        smoothing_percent=smoothing,
    )


def clear_results():
    """Reset global state and clear outputs."""
    global current_original_image, current_depth_norm, current_depth_map_pil
    current_original_image = None
    current_depth_norm = None
    current_depth_map_pil = None
    return None, None


with gr.Blocks(title="ChromoStereoizer Enhanced", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# ChromoStereoizer Enhanced")
    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(label="Upload Image", type="pil", height=400)
            generate_btn = gr.Button("Generate Depth Map", variant="primary", size="lg")
        with gr.Column(scale=1):
            gr.Markdown("**Depth Map**")
            depth_output = gr.Image(type="pil", height=400, interactive=False, show_download_button=True, show_label=False)
            gr.Markdown("**ChromoStereoizer Result**")
            chromo_output = gr.Image(type="pil", height=400, interactive=False, show_download_button=True, show_label=False)
            gr.Markdown("## Controls")
            threshold_slider = gr.Slider(minimum=0, maximum=100, value=50, step=1, label="Threshold (%)")
            depth_scale_slider = gr.Slider(minimum=0, maximum=100, value=50, step=1, label="Depth Scale (Steepness)")
            feather_slider = gr.Slider(minimum=0, maximum=100, value=10, step=1, label="Feather (%)")
            red_slider = gr.Slider(minimum=0, maximum=100, value=50, step=1, label="Red Brightness")
            blue_slider = gr.Slider(minimum=0, maximum=100, value=50, step=1, label="Blue Brightness")
            gamma_slider = gr.Slider(minimum=0, maximum=100, value=50, step=1, label="Gamma")
            black_slider = gr.Slider(minimum=0, maximum=100, value=0, step=1, label="Black Level (%)")
            white_slider = gr.Slider(minimum=0, maximum=100, value=100, step=1, label="White Level (%)")
            smoothing_slider = gr.Slider(minimum=0, maximum=100, value=0, step=1, label="Smoothing (%)")
            clear_btn = gr.Button("Clear", variant="secondary")
    # Event bindings
    generate_btn.click(
        fn=generate_depth_map,
        inputs=[input_image],
        outputs=[depth_output, chromo_output],
        show_progress=True,
    )
    for slider in [threshold_slider, depth_scale_slider, feather_slider, red_slider, blue_slider, gamma_slider, black_slider, white_slider, smoothing_slider]:
        slider.change(
            fn=update_effect,
            inputs=[threshold_slider, depth_scale_slider, feather_slider, red_slider, blue_slider, gamma_slider, black_slider, white_slider, smoothing_slider],
            outputs=chromo_output,
            show_progress=False,
        )
    clear_btn.click(
        fn=clear_results,
        inputs=[],
        outputs=[depth_output, chromo_output],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
