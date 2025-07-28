import gradio as gr
import numpy as np
from PIL import Image
try:
    import cv2
except ImportError:
    cv2 = None
import torch
from transformers import pipeline

# Load depth estimation model (large)
depth_pipe = pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Large-hf")

# Global variables to store state
current_original_image = None
current_depth_norm = None
current_gray = None
current_depth_map_pil = None


def preprocess_gray(image: Image.Image, gamma: float) -> np.ndarray:
    """
    Convert image to grayscale and apply gamma correction.
    Returns a float32 array in [0,1].
    """
    gray = np.array(image.convert("L"), dtype=np.float32) / 255.0
    gray = np.clip(gray ** gamma, 0.0, 1.0)
    return gray


def smooth_depth(depth_norm: np.ndarray, radius: float) -> np.ndarray:
    """
    Apply edge‑preserving smoothing to the normalized depth map using bilateral filtering.
    radius determines the strength; 0 means no smoothing.
    """
    if radius <= 0 or depth_norm is None or cv2 is None:
        return depth_norm
    depth_uint8 = (depth_norm * 255.0).astype(np.uint8)
    sigma = max(radius * 10.0, 1.0)
    smoothed = cv2.bilateralFilter(depth_uint8, d=5, sigmaColor=sigma, sigmaSpace=sigma)
    return smoothed.astype(np.float32) / 255.0


def logistic_blend(depth_norm: np.ndarray, threshold: float, steepness: float) -> np.ndarray:
    """
    Compute a blend factor from the depth map using a logistic (sigmoid) function.
    threshold controls the midpoint (0..1), steepness controls the slope (>0).
    Returns array in [0,1].
    """
    s = max(steepness, 1e-3)
    return 1.0 / (1.0 + np.exp(-s * (depth_norm - threshold)))


def build_chromostereopsis_image(gray: np.ndarray, blend: np.ndarray, red_brightness: float, blue_brightness: float, parallax_shift: float) -> Image.Image:
    """
    Build an RGB image using red/blue channels weighted by the blend factor.
    Optionally shift red and blue channels horizontally for parallax effect.
    gray: grayscale luminance in [0,1]
    blend: blend factor in [0,1], same shape as gray
    red_brightness, blue_brightness: multipliers for channel intensities
    parallax_shift: maximum pixel shift for red (left) and blue (right) channels
    """
    h, w = gray.shape
    red_intensity = red_brightness * gray * blend
    blue_intensity = blue_brightness * gray * (1.0 - blend)
    red_img = np.clip(red_intensity * 255.0, 0, 255).astype(np.uint8)
    blue_img = np.clip(blue_intensity * 255.0, 0, 255).astype(np.uint8)
    shift = int(parallax_shift)
    if shift > 0:
        red_img = np.roll(red_img, -shift, axis=1)
        blue_img = np.roll(blue_img, shift, axis=1)
    output = np.zeros((h, w, 3), dtype=np.uint8)
    output[..., 0] = red_img
    output[..., 1] = 0
    output[..., 2] = blue_img
    return Image.fromarray(output, mode="RGB")


def generate_depth_map(input_image: Image.Image):
    global current_original_image, current_depth_norm, current_gray, current_depth_map_pil
    if input_image is None:
        current_original_image = None
        current_depth_norm = None
        current_gray = None
        current_depth_map_pil = None
        return None, None
    current_original_image = input_image
    result = depth_pipe(input_image)
    depth_pil = result["depth"]
    depth_np = np.array(depth_pil, dtype=np.float32)
    depth_np -= depth_np.min()
    max_val = depth_np.max()
    if max_val > 0:
        depth_np /= max_val
    current_depth_norm = depth_np
    depth_uint8 = (depth_np * 255.0).astype(np.uint8)
    current_depth_map_pil = Image.fromarray(depth_uint8, mode="L")
    current_gray = preprocess_gray(current_original_image, gamma=1.0)
    blend = logistic_blend(current_depth_norm, threshold=0.5, steepness=10.0)
    chromo_img = build_chromostereopsis_image(current_gray, blend, red_brightness=1.0, blue_brightness=1.0, parallax_shift=10.0)
    return current_depth_map_pil.convert("RGB"), chromo_img


def update_chromostereopsis(threshold_percent, depth_scale, red_brightness, blue_brightness, gamma_value, parallax_shift, smoothing_radius):
    global current_original_image, current_depth_norm, current_gray
    if current_original_image is None or current_depth_norm is None:
        return None
    current_gray = preprocess_gray(current_original_image, gamma=gamma_value)
    depth_smoothed = smooth_depth(current_depth_norm, radius=smoothing_radius)
    threshold = threshold_percent / 100.0
    blend = logistic_blend(depth_smoothed, threshold=threshold, steepness=depth_scale)
    result_img = build_chromostereopsis_image(current_gray, blend, red_brightness=red_brightness, blue_brightness=blue_brightness, parallax_shift=parallax_shift)
    return result_img


def clear_results():
    global current_original_image, current_depth_norm, current_gray, current_depth_map_pil
    current_original_image = None
    current_depth_norm = None
    current_gray = None
    current_depth_map_pil = None
    return None, None


with gr.Blocks(title="Enhanced ChromoStereoizer", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Enhanced ChromoStereoizer")
    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(label="Upload Image", type="pil", height=400)
            generate_btn = gr.Button("Generate Depth Map", variant="primary", size="lg")
        with gr.Column(scale=1):
            gr.Markdown("**Depth Map**")
            depth_output = gr.Image(type="pil", height=400, interactive=False, show_download_button=True, show_label=False)
            gr.Markdown("**ChromoStereoizer Result**")
            chromo_output = gr.Image(type="pil", height=400, interactive=False, show_download_button=True, show_label=False)
            gr.Markdown("## Effect Controls")
            threshold_slider = gr.Slider(minimum=0, maximum=100, value=50, step=1, label="Threshold (%)")
            depth_scale_slider = gr.Slider(minimum=1, maximum=40, value=10, step=1, label="Depth Scale (Steepness)")
            red_brightness_slider = gr.Slider(minimum=0.0, maximum=2.0, value=1.0, step=0.1, label="Red Brightness")
            blue_brightness_slider = gr.Slider(minimum=0.0, maximum=2.0, value=1.0, step=0.1, label="Blue Brightness")
            gamma_slider = gr.Slider(minimum=0.1, maximum=3.0, value=1.0, step=0.1, label="Gamma")
            parallax_slider = gr.Slider(minimum=0.0, maximum=50.0, value=10.0, step=1.0, label="Parallax Shift (px)")
            smoothing_slider = gr.Slider(minimum=0.0, maximum=10.0, value=0.0, step=1.0, label="Smoothing Radius")
            clear_btn = gr.Button("Clear", variant="secondary")
    generate_btn.click(fn=generate_depth_map, inputs=[input_image], outputs=[depth_output, chromo_output], show_progress=True)
    for slider in [threshold_slider, depth_scale_slider, red_brightness_slider, blue_brightness_slider, gamma_slider, parallax_slider, smoothing_slider]:
        slider.change(fn=update_chromostereopsis, inputs=[threshold_slider, depth_scale_slider, red_brightness_slider, blue_brightness_slider, gamma_slider, parallax_slider, smoothing_slider], outputs=chromo_output, show_progress=False)
    clear_btn.click(fn=clear_results, inputs=[], outputs=[depth_output, chromo_output])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
