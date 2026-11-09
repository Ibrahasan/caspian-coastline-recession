import numpy as np
import cv2

def extract_shoreline_contour(mask: np.ndarray) -> np.ndarray:
    """Extracts the water-land boundary contour and removes image frame border artifacts."""
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(mask, kernel)
    boundary = mask - eroded
    boundary[0, :] = 0
    boundary[-1, :] = 0
    boundary[:, 0] = 0
    boundary[:, -1] = 0
    return boundary

def compute_recession_metrics(mask_16: np.ndarray, mask_26: np.ndarray, transform) -> dict:
    """Computes area and distance recession statistics based on the 2016 and 2026 water masks."""
    px_w = transform[0]
    px_h = -transform[4]
    px_area_m2 = px_w * px_h
    avg_px_size_m = np.sqrt(px_area_m2)

    area_16_ha = (np.sum(mask_16) * px_area_m2) / 10000.0
    area_26_ha = (np.sum(mask_26) * px_area_m2) / 10000.0

    receded_mask = ((mask_16 == 1) & (mask_26 == 0)).astype(np.uint8)
    receded_area_ha = (np.sum(receded_mask) * px_area_m2) / 10000.0

    shore_16 = extract_shoreline_contour(mask_16)
    shore_26 = extract_shoreline_contour(mask_26)

    dist_map_px = cv2.distanceTransform(1 - shore_26, cv2.DIST_L2, 5)
    shifts_m = dist_map_px[shore_16 == 1] * avg_px_size_m

    return {
        "px_w": px_w,
        "px_h": px_h,
        "px_area_m2": px_area_m2,
        "area_16_ha": area_16_ha,
        "area_26_ha": area_26_ha,
        "receded_area_ha": receded_area_ha,
        "receded_area_km2": receded_area_ha / 100.0,
        "mean_shift_m": float(np.mean(shifts_m)),
        "median_shift_m": float(np.median(shifts_m)),
        "p95_shift_m": float(np.percentile(shifts_m, 95)),
        "max_shift_m": float(np.max(shifts_m)),
        "receded_mask": receded_mask,
        "shore_16": shore_16,
        "shore_26": shore_26
    }