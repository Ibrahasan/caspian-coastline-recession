import numpy as np
import cv2
from ultralytics import SAM

class CoastlineSegmenter:
    def __init__(self, model_path: str = "mobile_sam.pt", device: str = "cpu"):
        self.model = SAM(model_path)
        self.device = device

    @staticmethod
    def get_auto_prompts(bands: dict, num_points: int = 8):
        """Automatically selects water (1) and land (0) prompt points based on the NDWI index."""
        green = bands['B03']
        nir = bands['B08']
        ndwi = (green - nir) / (green + nir + 1e-6)

        water_thresh = np.percentile(ndwi, 45)
        land_thresh = np.percentile(ndwi, 20)

        sure_water = (ndwi >= water_thresh).astype(np.uint8)
        sure_land = (ndwi <= land_thresh).astype(np.uint8)

        kernel = np.ones((5, 5), np.uint8)
        eroded_water = cv2.erode(sure_water, kernel)
        eroded_land = cv2.erode(sure_land, kernel)

        if np.sum(eroded_water) >= num_points:
            sure_water = eroded_water
        if np.sum(eroded_land) >= num_points:
            sure_land = eroded_land

        water_yx = np.column_stack(np.where(sure_water > 0))
        land_yx = np.column_stack(np.where(sure_land > 0))

        np.random.seed(42)
        w_idx = np.random.choice(len(water_yx), size=num_points, replace=False)
        l_idx = np.random.choice(len(land_yx), size=num_points, replace=False)

        water_pts = water_yx[w_idx][:, ::-1]
        land_pts = land_yx[l_idx][:, ::-1]

        points = np.vstack([water_pts, land_pts]).tolist()
        labels = [1] * num_points + [0] * num_points
        return points, labels

    @staticmethod
    def clean_water_mask(mask: np.ndarray) -> np.ndarray:
        """Retains only the largest water body (the Caspian Sea) using Connected Components."""
        mask_uint8 = (mask > 0).astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)
        
        if num_labels > 1:
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            clean_mask = (labels == largest_label).astype(np.uint8)
        else:
            clean_mask = mask_uint8

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel)
        return clean_mask

    def segment(self, bands: dict, rgb: np.ndarray, num_points: int = 8):
        """Executes the full segmentation pipeline."""
        points, labels = self.get_auto_prompts(bands, num_points=num_points)
        results = self.model.predict(rgb, points=[points], labels=[labels], device=self.device, verbose=False)
        raw_mask = results[0].masks.data[0].cpu().numpy()
        final_mask = self.clean_water_mask(raw_mask)
        return final_mask, points, labels