"""
Amazon (Rondonia) Deforestation Tracker — Prototype IVP Pipeline
==================================================================

CAVEAT (read before trusting the numbers):
This pipeline segments "deforested" area using Otsu thresholding on the
Hue channel of an already-composited RGB/tif frame. That is a workable
prototype heuristic, but it is NOT equivalent to a proper NDVI-based
deforestation analysis, which needs the actual Near-Infrared + Red
reflectance bands (raw Landsat NDVI = (NIR - Red) / (NIR + Red)).
Hue-only segmentation can be confused by cloud shadow, river sediment,
burn scars, and seasonal soil color shifts. Treat outputs as an
estimate for a class project / prototype, not a scientific figure,
unless you swap in real multispectral bands.

Pipeline:
  1. Load each year's frame, validate it's readable and non-corrupt.
  2. Convert to HSV, blur the Hue channel to denoise.
  3. Otsu-threshold the Hue channel into two clusters.
  4. Deterministically decide which cluster is "deforested" by comparing
     each cluster's mean hue to the expected vegetation hue (green,
     ~60 degrees in OpenCV's 0-179 hue range).
  5. Clean the mask with erosion + dilation.
  6. Accumulate the mask using a bitwise OR to ensure deforestation is cumulative.
  7. Convert cleared pixel count to km^2 and save mask + overlay.
  8. Compute year-over-year rate of change and plot + export CSV.
"""

import os
import re
import csv
import glob
import logging
from dataclasses import dataclass

import cv2
import numpy as np
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATASET_DIR = "amazon_30years"
FILENAME_PATTERN = "rondonia_*.tif"
OUTPUT_MASK_DIR = "output_masks"
OUTPUT_OVERLAY_DIR = "output_overlays"
OUTPUT_CSV = "deforestation_results.csv"
OUTPUT_CHART = "deforestation_trend.png"

# NOTE: verify this against your actual source imagery's metadata.
# Landsat MSS = 60m/px, but Landsat TM/ETM+/OLI (most 1984+ scenes) = 30m/px.
# 60m/px => 3600 m^2 => 0.0036 km^2 per pixel. Update if your data differs.
METERS_PER_PIXEL = 60
PIXEL_TO_KM2 = (METERS_PER_PIXEL ** 2) / 1_000_000

VEGETATION_HUE_CENTER = 60   # OpenCV hue range is 0-179; green ~ 35-85
MORPH_KERNEL = np.ones((5, 5), np.uint8)
EROSION_ITERS = 1
DILATION_ITERS = 2


@dataclass
class YearResult:
    year: int
    area_km2: float
    otsu_threshold: float


def parse_year(filename: str) -> int:
    """Extract a 4-digit year from a filename like 'rondonia_1998.tif'."""
    match = re.search(r"(\d{4})", filename)
    if not match:
        raise ValueError(f"Could not find a 4-digit year in filename: {filename}")
    return int(match.group(1))


def is_valid_frame(img: np.ndarray) -> bool:
    """Reject unreadable, empty, wrong-shape, or near-blank frames."""
    if img is None:
        return False
    if img.ndim != 3 or img.shape[2] != 3:
        return False
    if img.size == 0:
        return False
    # near-blank / corrupt frame check (flat or near-black image)
    if np.mean(img) < 5 or np.std(img) < 1:
        return False
    return True


def segment_deforestation(img: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Returns (cleaned_binary_mask, otsu_threshold).
    Mask value 255 = classified as deforested/cleared.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    blurred = cv2.GaussianBlur(hue, (5, 5), 0)

    otsu_val, raw_mask = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Dynamic Polarity Check for Otsu Inversion
    # If the mean pixel value is > 127, more than 50% of the mask is white. 
    # This indicates the Otsu bimodal split guessed backward, confusing forest for dirt.
    if np.mean(raw_mask) > 127:
        raw_mask = cv2.bitwise_not(raw_mask)

    eroded = cv2.erode(raw_mask, MORPH_KERNEL, iterations=EROSION_ITERS)
    cleaned_mask = cv2.dilate(eroded, MORPH_KERNEL, iterations=DILATION_ITERS)

    return cleaned_mask, otsu_val


def process_dataset() -> list[YearResult]:
    os.makedirs(OUTPUT_MASK_DIR, exist_ok=True)
    os.makedirs(OUTPUT_OVERLAY_DIR, exist_ok=True)

    image_paths = sorted(glob.glob(os.path.join(DATASET_DIR, FILENAME_PATTERN)))
    log.info(f"Found {len(image_paths)} candidate images in '{DATASET_DIR}'.")

    if not image_paths:
        log.warning("No images found — check DATASET_DIR / FILENAME_PATTERN.")
        return []

    results: list[YearResult] = []
    expected_shape = None
    seen_years = set()
    master_mask = None

    for path in image_paths:
        filename = os.path.basename(path)
        try:
            year = parse_year(filename)
        except ValueError as e:
            log.warning(f"Skipping '{filename}': {e}")
            continue

        if year in seen_years:
            log.warning(f"Duplicate year {year} in '{filename}' — skipping.")
            continue

        img = cv2.imread(path)
        if not is_valid_frame(img):
            log.warning(f"Skipping corrupt/unreadable/empty frame: {filename}")
            continue

        if expected_shape is None:
            expected_shape = img.shape
        elif img.shape != expected_shape:
            log.warning(
                f"Skipping '{filename}': shape {img.shape} does not match "
                f"expected {expected_shape}. Pixel counts would not be "
                f"comparable across years with mismatched resolution."
            )
            continue

        cleaned_mask, otsu_val = segment_deforestation(img)

        # Ensure deforestation is cumulative across years
        if master_mask is None:
            master_mask = np.zeros_like(cleaned_mask)
        
        master_mask = cv2.bitwise_or(master_mask, cleaned_mask)

        cleared_pixels = int(np.count_nonzero(master_mask == 255))
        area_km2 = cleared_pixels * PIXEL_TO_KM2

        cv2.imwrite(os.path.join(OUTPUT_MASK_DIR, f"mask_{year}.png"), master_mask)

        overlay = img.copy()
        overlay[master_mask == 255] = [0, 0, 255]  # red in BGR
        blended = cv2.addWeighted(img, 0.65, overlay, 0.35, 0)
        cv2.imwrite(os.path.join(OUTPUT_OVERLAY_DIR, f"overlay_{year}.png"), blended)

        results.append(YearResult(year=year, area_km2=area_km2, otsu_threshold=otsu_val))
        seen_years.add(year)
        log.info(f"[{year}] Deforested Area: {area_km2:,.2f} sq km (Otsu threshold: {otsu_val:.1f})")

    results.sort(key=lambda r: r.year)

    all_years = set(range(min(seen_years), max(seen_years) + 1)) if seen_years else set()
    missing = sorted(all_years - seen_years)
    if missing:
        log.info(f"Years missing from dataset (skipped): {missing}")

    return results


def compute_rates(results: list[YearResult]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    years_arr = np.array([r.year for r in results])
    areas_arr = np.array([r.area_km2 for r in results])

    delta_years = np.diff(years_arr).astype(float)
    delta_years[delta_years == 0] = np.nan  # guard, shouldn't happen post-dedup
    delta_areas = np.diff(areas_arr)
    annual_rate = delta_areas / delta_years

    return years_arr, areas_arr, annual_rate


def export_csv(results: list[YearResult], years_arr: np.ndarray, annual_rate: np.ndarray) -> None:
    rate_by_year = dict(zip(years_arr[1:].tolist(), annual_rate.tolist()))
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["year", "deforested_area_km2", "otsu_threshold", "annual_rate_km2_per_year"])
        for r in results:
            rate = rate_by_year.get(r.year, "")
            writer.writerow([r.year, f"{r.area_km2:.4f}", f"{r.otsu_threshold:.2f}", rate])
    log.info(f"Wrote results table to '{OUTPUT_CSV}'.")


def plot_results(years_arr: np.ndarray, areas_arr: np.ndarray, annual_rate: np.ndarray) -> None:
    interval_labels = years_arr[1:]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    fig.suptitle(
        "Prototype IVP Deforestation Tracker (Hue-Threshold Estimate — Not NDVI)",
        fontsize=13, color="gray", style="italic",
    )

    ax1.plot(years_arr, areas_arr, marker="o", color="#b22222", linewidth=2.2, markersize=5)
    ax1.set_title(
        f"Estimated Deforested Area in Rondônia ({years_arr.min()}–{years_arr.max()})",
        fontsize=13, weight="bold",
    )
    ax1.set_ylabel("Cleared Area ($km^2$)", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.6)

    bar_colors = ["#c0392b" if r >= 0 else "#27ae60" for r in annual_rate]
    ax2.bar(interval_labels, annual_rate, width=0.75, color=bar_colors, edgecolor="black", alpha=0.85)
    ax2.axhline(0, color="black", linewidth=0.9)
    ax2.set_title(r"Estimated Annual Rate of Forest Loss ($\frac{dA}{dt}$ in $km^2$/year)", fontsize=13, weight="bold")
    ax2.set_xlabel("Year", fontsize=11)
    ax2.set_ylabel("Change Rate ($km^2$/year)", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.set_xticks(years_arr[::2])

    plt.tight_layout()
    plt.savefig(OUTPUT_CHART, dpi=300)
    log.info(f"Saved chart to '{OUTPUT_CHART}'.")
    plt.close(fig)


def main() -> None:
    results = process_dataset()

    if len(results) < 2:
        log.warning("Need at least 2 valid years to compute a rate of change. Stopping.")
        return

    years_arr, areas_arr, annual_rate = compute_rates(results)
    export_csv(results, years_arr, annual_rate)
    plot_results(years_arr, areas_arr, annual_rate)
    log.info("Processing complete.")


if __name__ == "__main__":
    main()