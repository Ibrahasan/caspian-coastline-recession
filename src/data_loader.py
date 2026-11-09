import os
import glob
import rasterio
import numpy as np

def load_sentinel_data(folder_path: str):
    """Loads Sentinel-2 B02, B03, B04, B08 bands and GeoTIFF metadata."""
    bands = {}
    meta = None
    transform = None
    crs = None

    for b in ['B02', 'B03', 'B04', 'B08']:
        pattern = os.path.join(folder_path, f"*{b}*.tif*")
        files = glob.glob(pattern)
        if not files:
            raise FileNotFoundError(f"Band {b} not found in directory '{folder_path}'!")
            
        with rasterio.open(files[0]) as src:
            bands[b] = src.read(1).astype(np.float32)
            if meta is None:
                meta = src.meta.copy()
                transform = src.transform
                crs = src.crs

    rgb_uint8 = create_normalized_rgb(bands)
    return bands, rgb_uint8, meta, transform, crs

def create_normalized_rgb(bands: dict) -> np.ndarray:
    """Converts 32-bit float bands to 8-bit RGB format using 2%-98% percentile clipping."""
    rgb = np.stack([bands['B04'], bands['B03'], bands['B02']], axis=-1)
    p2, p98 = np.percentile(rgb, (2, 98))
    rgb_norm = np.clip((rgb - p2) / (p98 - p2 + 1e-6), 0, 1)
    return (rgb_norm * 255).astype(np.uint8)

def save_geotiff_mask(mask: np.ndarray, meta: dict, out_path: str):
    """Saves the extracted water mask as a GeoTIFF while preserving the coordinate reference system."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    mask_meta = meta.copy()
    mask_meta.update(dtype=rasterio.uint8, count=1, nodata=0)
    with rasterio.open(out_path, "w", **mask_meta) as dst:
        dst.write(mask.astype(rasterio.uint8), 1)