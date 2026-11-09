from src.data_loader import load_sentinel_data, save_geotiff_mask
from src.segmenter import CoastlineSegmenter
from src.analytics import compute_recession_metrics
from src.visualizer import save_segmentation_plot, save_analytics_dashboard, create_comparison_gif

def main():
    print("1. Loading Sentinel-2 GeoTIFF data...")
    bands_16, rgb_16, meta_16, trans_16, crs_16 = load_sentinel_data("data/raw/2016")
    bands_26, rgb_26, meta_26, trans_26, crs_26 = load_sentinel_data("data/raw/2026")

    print("2. Running water segmentation with MobileSAM model...")
    segmenter = CoastlineSegmenter(model_path="mobile_sam.pt", device="cpu")
    mask_16, pts_16, lbl_16 = segmenter.segment(bands_16, rgb_16)
    mask_26, pts_26, lbl_26 = segmenter.segment(bands_26, rgb_26)

    save_geotiff_mask(mask_16, meta_16, "data/masks/water_mask_2016.tif")
    save_geotiff_mask(mask_26, meta_26, "data/masks/water_mask_2026.tif")

    print("3. Computing coastline recession analytics...")
    metrics = compute_recession_metrics(mask_16, mask_26, trans_16)

    print("\n==================================================")
    print("   CASPIAN SEA (SUMGAYIT) 2016-2026 REPORT")
    print("==================================================")
    print(f"• 2016 Water Area         : {metrics['area_16_ha']:.2f} ha")
    print(f"• 2026 Water Area         : {metrics['area_26_ha']:.2f} ha")
    print(f"• Exposed Seabed Area     : {metrics['receded_area_ha']:.2f} ha ({metrics['receded_area_km2']:.3f} km²)")
    print("--------------------------------------------------")
    print(f"• Mean Recession Distance : {metrics['mean_shift_m']:.1f} meters")
    print(f"• Median Recession Dist.  : {metrics['median_shift_m']:.1f} meters")
    print(f"• 95th Percentile Recess. : {metrics['p95_shift_m']:.1f} meters")
    print(f"• Maximum Recession Point : {metrics['max_shift_m']:.1f} meters")
    print("==================================================\n")

    print("4. Generating visual reports and comparison GIF...")
    save_segmentation_plot(
        rgb_16, mask_16, pts_16, lbl_16,
        rgb_26, mask_26, pts_26, lbl_26,
        out_path="outputs/02_segmentation_results.png"
    )
    save_analytics_dashboard(rgb_26, metrics, out_path="outputs/03_final_coastline_analytics.png")
    create_comparison_gif(rgb_16, rgb_26, metrics, out_path="outputs/caspian_recession.gif")
    print("Completed! All results have been saved to the 'outputs/' directory.")

if __name__ == "__main__":
    main()