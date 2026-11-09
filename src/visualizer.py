import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

def save_segmentation_plot(rgb_16, mask_16, pts_16, lbl_16, rgb_26, mask_26, pts_26, lbl_26, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for row, (year, rgb, mask, pts, lbl) in enumerate([
        ("2016", rgb_16, mask_16, pts_16, lbl_16),
        ("2026", rgb_26, mask_26, pts_26, lbl_26)
    ]):
        axes[row, 0].imshow(rgb)
        for (x, y), l in zip(pts, lbl):
            color = 'cyan' if l == 1 else 'red'
            axes[row, 0].scatter(x, y, c=color, s=50, edgecolors='black', linewidth=1.2)
        axes[row, 0].set_title(f"{year}: RGB & Auto-Prompts (Cyan=Water, Red=Land)")
        axes[row, 0].axis("off")

        overlay = rgb.copy()
        overlay[mask == 1] = (overlay[mask == 1] * 0.5 + np.array([0, 120, 255]) * 0.5).astype(np.uint8)
        axes[row, 1].imshow(overlay)
        axes[row, 1].set_title(f"{year}: MobileSAM Water Segmentation")
        axes[row, 1].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def save_analytics_dashboard(rgb_26: np.ndarray, metrics: dict, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))

    shore_16_thick = cv2.dilate(metrics["shore_16"], np.ones((2, 2), np.uint8))
    shore_26_thick = cv2.dilate(metrics["shore_26"], np.ones((2, 2), np.uint8))

    vis_lines = rgb_26.copy()
    vis_lines[shore_16_thick == 1] = [0, 255, 255]
    vis_lines[shore_26_thick == 1] = [255, 50, 50]

    axes[0].imshow(vis_lines)
    axes[0].set_title("Coastline Shift\n(Cyan = 2016, Red = 2026)", fontsize=12, fontweight='bold')
    axes[0].axis("off")

    vis_zone = rgb_26.copy()
    receded = metrics["receded_mask"]
    vis_zone[receded == 1] = (vis_zone[receded == 1] * 0.3 + np.array([255, 60, 0]) * 0.7).astype(np.uint8)
    axes[1].imshow(vis_zone)
    axes[1].set_title(f"Exposed Seabed Zone\n(Total Loss: {metrics['receded_area_ha']:.1f} Hectares)", fontsize=12, fontweight='bold')
    axes[1].axis("off")

    row_recession_m = np.sum(receded, axis=1) * metrics["px_w"]
    y_coords_km = np.arange(len(row_recession_m)) * metrics["px_h"] / 1000.0

    axes[2].plot(row_recession_m, y_coords_km, color='#d9534f', linewidth=2)
    axes[2].fill_betweenx(y_coords_km, 0, row_recession_m, color='#d9534f', alpha=0.3)
    axes[2].invert_yaxis()
    axes[2].set_xlabel("Horizontal Recession Distance (m)", fontsize=11)
    axes[2].set_ylabel("North-to-South Coastline Length (km)", fontsize=11)
    axes[2].set_title("Coastline Recession Profile\n(North -> South)", fontsize=12, fontweight='bold')
    axes[2].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(out_path, dpi=250)
    plt.close()

def create_comparison_gif(rgb_16, rgb_26, metrics: dict, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    frames = []
    kernel = np.ones((2, 2), np.uint8)
    s16 = cv2.dilate(metrics["shore_16"], kernel)
    s26 = cv2.dilate(metrics["shore_26"], kernel)

    scale = 2
    h, w = rgb_16.shape[:2]
    new_size = (w * scale, h * scale)

    for label, rgb in [
        (f"AUGUST 2016 (Water: {metrics['area_16_ha']:.1f} ha)", rgb_16),
        (f"JULY 2026 (Water: {metrics['area_26_ha']:.1f} ha | -{metrics['receded_area_ha']:.1f} ha)", rgb_26)
    ]:
        canvas = rgb.copy()
        canvas[s16 == 1] = [0, 255, 255]
        if "2026" in label:
            canvas[s26 == 1] = [255, 50, 50]

        canvas_resized = cv2.resize(canvas, new_size, interpolation=cv2.INTER_CUBIC)
        pil_img = Image.fromarray(canvas_resized)
        draw = ImageDraw.Draw(pil_img)
        draw.rectangle([10, 10, 345, 38], fill=(0, 0, 0))
        draw.text((16, 16), label, fill=(255, 255, 255))
        frames.append(pil_img)

    frames[0].save(out_path, save_all=True, append_images=[frames[1]], duration=1200, loop=0)