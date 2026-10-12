"""
Amazon (Rondônia) Deforestation Tracker + Fractal Dimension Analysis
=====================================================================

RESEARCH QUESTION:
  Does the fractal dimension of deforestation-frontier boundaries in
  Rondônia show a measurable temporal trend between 1994–2024, and do
  detected shifts correlate with known transitions between smallholder
  and industrial-scale clearing patterns?

PIPELINE (extends the original Hue-Otsu prototype):
  1–7. Same as original: load, HSV, Otsu, clean, cumulate, export.
  NEW STAGES:
  8.  Extract the FRONTIER mask: newly deforested pixels per year
      (difference between consecutive cumulative masks).
  9.  Extract contours from the frontier mask using Canny edge detection.
  10. Compute fractal dimension of frontier contours via box-counting.
  11. Report FD per year, FD trend, and correlation with known policy
      events (PPCDAm 2004, Soy Moratorium 2006, Forest Code 2012, etc.)

CAVEATS:
  - Hue-based segmentation is a prototype proxy, not NDVI.
  - Fractal dimension at 30m resolution captures meso-scale boundary
    geometry; sub-pixel features (individual roads) are still lost.
  - Box-counting FD is sensitive to the range of box sizes; we report
    the regression R² alongside FD to flag unreliable estimates.
  - The FD-to-cause mapping is correlational, not causal.
"""

import os
import re
import csv
import glob
import logging
from dataclasses import dataclass, field

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as sp_stats

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATASET_DIR = "amazon_30years_clean"
FILENAME_PATTERN = "rondonia_*.tif"

OUTPUT_MASK_DIR      = "output_masks"
OUTPUT_OVERLAY_DIR   = "output_overlays"
OUTPUT_FRONTIER_DIR  = "output_frontiers"
OUTPUT_CSV           = "deforestation_results.csv"
OUTPUT_CHART         = "deforestation_trend.png"
OUTPUT_FD_CHART      = "fractal_dimension_trend.png"
OUTPUT_COMBINED      = "combined_analysis.png"

# 30m/px (Landsat native) => 900 m^2 => 0.0009 km^2 per pixel
METERS_PER_PIXEL = 30
PIXEL_TO_KM2 = (METERS_PER_PIXEL ** 2) / 1_000_000

VEGETATION_HUE_CENTER = 60
MORPH_KERNEL = np.ones((5, 5), np.uint8)
EROSION_ITERS = 1
DILATION_ITERS = 2

# Box-counting parameters
# Box sizes as powers of 2, from small (2px) to large (512px).
# The upper bound should be < ~1/4 of image dimension.
BOX_SIZES = [2, 4, 8, 16, 32, 64, 128, 256]
MIN_CONTOUR_PIXELS = 200  # skip trivially small frontiers

# Known policy events for annotation
POLICY_EVENTS = {
    2004: "PPCDAm\nlaunched",
    2006: "Soy\nMoratorium",
    2012: "Forest Code\nrevision",
    2019: "Enforcement\nweakened",
    2023: "Enforcement\nrestored",
}

# ---------------------------------------------------------------------------
# NEW: Canopy openness / logging-road detection (structural fragmentation)
# ---------------------------------------------------------------------------
OUTPUT_CANOPY_DIR    = "output_canopy"
OUTPUT_CANOPY_CSV    = "canopy_fragmentation_results.csv"
OUTPUT_CANOPY_CHART  = "canopy_fragmentation_trend.png"

CANOPY_BLUR_KERNEL        = (5, 5)
CANOPY_CANNY_LOW          = 30     # lower = catches subtler edges
CANOPY_CANNY_HIGH         = 100
CANOPY_CLOSE_KERNEL       = (5, 5) # joins nearby edge pixels into solid shapes
CANOPY_MIN_COMPONENT_PX   = 100    # drops tiny natural gaps / speckle
CANOPY_FILL_HOLES_MAX_PX  = 1500   # fill closed edge loops smaller than this (round gaps)
CANOPY_NORMALIZE_CONTRAST = True   # standardise forest texture contrast across years
CANOPY_TARGET_SPREAD      = 10.0   # robust spread (gray levels) the forest is scaled to
CANOPY_USE_FOREST_ROI     = True   # only analyse forest not yet in the cleared mask
CANOPY_ROI_BUFFER_PX      = 3      # stay this far away from the cleared-area boundary
CANOPY_LINEAR_ELONGATION  = 4.0    # length / width >= this => road- or trail-like
CANOPY_MIN_ROAD_LENGTH_PX = 15     # 15 px = 450 m at 30 m/px
CANOPY_EDGE_SATURATION    = 0.25   # warn if > 25% of forest pixels are edge pixels
CANOPY_MIN_ROI_FRACTION   = 0.05   # skip a year if < 5% of the tile is analysable forest
CANOPY_EXCLUDE_MASK_PATH  = None   # optional PNG, white = ignore (rivers, water, nodata)

CANOPY_RESULTS: list[dict] = []


@dataclass
class YearResult:
    year: int
    area_km2: float
    otsu_threshold: float
    frontier_km2: float = 0.0
    fractal_dim: float = float("nan")
    fd_r_squared: float = float("nan")
    frontier_contour_count: int = 0


# ---------------------------------------------------------------------------
# Core segmentation (unchanged from original)
# ---------------------------------------------------------------------------

def parse_year(filename: str) -> int:
    match = re.search(r"(\d{4})", filename)
    if not match:
        raise ValueError(f"No 4-digit year in: {filename}")
    return int(match.group(1))


def is_valid_frame(img: np.ndarray) -> bool:
    if img is None:
        return False
    if img.ndim != 3 or img.shape[2] != 3:
        return False
    if img.size == 0:
        return False
    if np.mean(img) < 5 or np.std(img) < 1:
        return False
    return True


def segment_deforestation(img: np.ndarray) -> tuple[np.ndarray, float]:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    blurred = cv2.GaussianBlur(hue, (5, 5), 0)

    otsu_val, raw_mask = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    if np.mean(raw_mask) > 127:
        raw_mask = cv2.bitwise_not(raw_mask)

    eroded = cv2.erode(raw_mask, MORPH_KERNEL, iterations=EROSION_ITERS)
    cleaned_mask = cv2.dilate(eroded, MORPH_KERNEL, iterations=DILATION_ITERS)

    return cleaned_mask, otsu_val


# ---------------------------------------------------------------------------
# NEW: Fractal Dimension via Box-Counting
# ---------------------------------------------------------------------------

def box_count(binary_edge_image: np.ndarray, box_sizes: list[int]) -> list[int]:
    """
    Count the number of boxes of each size needed to cover all edge pixels.

    For each box size s, tile the image into s×s cells and count how many
    cells contain at least one foreground (255) pixel.
    """
    counts = []
    h, w = binary_edge_image.shape

    for s in box_sizes:
        # Skip box sizes larger than the image
        if s > h or s > w:
            continue

        count = 0
        for y in range(0, h, s):
            for x in range(0, w, s):
                # Extract the box region
                box = binary_edge_image[y:y+s, x:x+s]
                if np.any(box > 0):
                    count += 1
        counts.append(count)

    return counts


def compute_fractal_dimension(binary_edge_image: np.ndarray) -> tuple[float, float]:
    """
    Estimate fractal dimension from a binary edge image using box-counting.

    Returns:
        (fractal_dimension, r_squared)

    The fractal dimension D is the negative slope of log(N) vs log(1/s),
    where N(s) is the number of boxes of side length s needed to cover
    the edge pixels.

    r_squared indicates how well the data fits a power law — values
    below ~0.90 mean the "fractal dimension" is not a reliable descriptor
    for this particular shape.
    """
    # Filter box sizes to those smaller than the image
    h, w = binary_edge_image.shape
    valid_sizes = [s for s in BOX_SIZES if s < min(h, w)]

    if len(valid_sizes) < 3:
        return float("nan"), 0.0

    counts = box_count(binary_edge_image, valid_sizes)

    if len(counts) < 3 or min(counts) == 0:
        return float("nan"), 0.0

    # log-log regression: log(N) = -D * log(s) + c
    log_sizes = np.log(valid_sizes[:len(counts)])
    log_counts = np.log(counts)

    slope, intercept, r_value, p_value, std_err = sp_stats.linregress(
        log_sizes, log_counts
    )

    fractal_dim = -slope  # D = -slope because N ~ s^(-D)
    r_squared = r_value ** 2

    return fractal_dim, r_squared


def extract_frontier_and_fd(
    current_cumulative: np.ndarray,
    previous_cumulative: np.ndarray | None,
    year: int,
) -> tuple[np.ndarray, float, float, int]:
    """
    Extract the deforestation frontier (newly cleared pixels this year)
    and compute its fractal dimension.

    Returns:
        (frontier_mask, fractal_dim, r_squared, contour_count)
    """
    if previous_cumulative is None:
        # First year: the entire mask IS the frontier
        frontier = current_cumulative.copy()
    else:
        # Frontier = pixels in current cumulative but NOT in previous
        frontier = cv2.bitwise_and(
            current_cumulative,
            cv2.bitwise_not(previous_cumulative)
        )

    frontier_pixels = int(np.count_nonzero(frontier == 255))

    if frontier_pixels < MIN_CONTOUR_PIXELS:
        log.info(f"  [{year}] Frontier too small ({frontier_pixels} px) — skipping FD.")
        return frontier, float("nan"), 0.0, 0

    # Extract edges of the frontier using Canny
    edges = cv2.Canny(frontier, 50, 150)

    # Count distinct contours for reporting
    contours, _ = cv2.findContours(frontier, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour_count = len(contours)

    # Compute fractal dimension on the edge image
    fd, r2 = compute_fractal_dimension(edges)

    return frontier, fd, r2, contour_count


# ---------------------------------------------------------------------------
# NEW: Canopy openness / logging-road detection (structural fragmentation)
# ---------------------------------------------------------------------------

def _canopy_roi(master_mask: np.ndarray) -> np.ndarray:
    """255 = standing forest to analyse (outside the cumulative cleared mask + buffer)."""
    if CANOPY_USE_FOREST_ROI:
        k = 2 * CANOPY_ROI_BUFFER_PX + 1
        cleared = cv2.dilate(master_mask, np.ones((k, k), np.uint8))
        roi = cv2.bitwise_not(cleared)
    else:
        roi = np.full(master_mask.shape, 255, np.uint8)

    if CANOPY_EXCLUDE_MASK_PATH:
        ex = cv2.imread(CANOPY_EXCLUDE_MASK_PATH, cv2.IMREAD_GRAYSCALE)
        if ex is not None and ex.shape == roi.shape:
            roi[ex > 127] = 0
        else:
            log.warning("Canopy exclude mask unreadable or wrong shape — ignored.")
    return roi


def _normalize_gray(gray: np.ndarray, roi: np.ndarray) -> np.ndarray:
    """Rescale so the forest's robust texture spread is the same in every year."""
    g = gray.astype(np.float32)
    sample = g[roi > 0] if np.count_nonzero(roi) > 1000 else g.ravel()
    med = float(np.median(sample))
    spread = max(float(np.median(np.abs(sample - med))) * 1.4826, 1.0)
    out = (g - med) * (CANOPY_TARGET_SPREAD / spread) + 128.0
    return np.clip(out, 0, 255).astype(np.uint8)


def _fill_small_holes(mask: np.ndarray) -> np.ndarray:
    """Fill enclosed holes up to CANOPY_FILL_HOLES_MAX_PX so edge outlines of round gaps become solid."""
    n, lab, st, _ = cv2.connectedComponentsWithStats(cv2.bitwise_not(mask), connectivity=4)
    h, w = mask.shape
    fill = np.zeros(n, bool)
    for i in range(1, n):
        x, y, bw, bh, area = (int(v) for v in st[i])
        touches_border = x == 0 or y == 0 or x + bw >= w or y + bh >= h
        fill[i] = (not touches_border) and area <= CANOPY_FILL_HOLES_MAX_PX
    return np.where(fill[lab], 255, mask).astype(np.uint8)


def detect_canopy_breaks(img: np.ndarray, roi: np.ndarray) -> dict:
    """
    Canny edges -> morphological closing -> fill small loops -> size filter, then split the surviving
    components into road/trail-like (elongated) and opening-like (blobby) using the
    distance transform: width ~ 2*max(DT)-1, length ~ area/width, elongation = length/width.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if CANOPY_NORMALIZE_CONTRAST:
        gray = _normalize_gray(gray, roi)
    blurred = cv2.GaussianBlur(gray, CANOPY_BLUR_KERNEL, 0)

    edges = cv2.bitwise_and(cv2.Canny(blurred, CANOPY_CANNY_LOW, CANOPY_CANNY_HIGH), roi)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, CANOPY_CLOSE_KERNEL)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    closed = cv2.bitwise_and(_fill_small_holes(closed), roi)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    keep = stats[:, cv2.CC_STAT_AREA] >= CANOPY_MIN_COMPONENT_PX
    keep[0] = False  # label 0 is the background

    roads = np.zeros_like(closed)
    openings = np.zeros_like(closed)
    road_length_px = 0.0
    n_roads = n_openings = 0

    for i in np.flatnonzero(keep):
        x, y, w, h, area = (int(v) for v in stats[i])
        crop = (labels[y:y + h, x:x + w] == i).astype(np.uint8)
        padded = cv2.copyMakeBorder(crop, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        width_px = max(2.0 * float(cv2.distanceTransform(padded, cv2.DIST_L2, 3).max()) - 1.0, 1.0)
        length_px = area / width_px
        elongation = length_px / width_px

        if elongation >= CANOPY_LINEAR_ELONGATION and length_px >= CANOPY_MIN_ROAD_LENGTH_PX:
            roads[y:y + h, x:x + w][crop == 1] = 255
            road_length_px += length_px
            n_roads += 1
        else:
            openings[y:y + h, x:x + w][crop == 1] = 255
            n_openings += 1

    return {
        "edges": edges, "roads": roads, "openings": openings,
        "road_length_px": road_length_px, "n_roads": n_roads, "n_openings": n_openings,
    }


def analyze_canopy_fragmentation(img: np.ndarray, master_mask: np.ndarray, year: int) -> dict:
    """Run the canopy-break detector for one year, save images, return a metrics row."""
    roi = _canopy_roi(master_mask)
    roi_px = int(np.count_nonzero(roi))
    roi_fraction = roi_px / roi.size
    haze = float(np.mean(cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 1] < 30))  # low-saturation share

    row = {
        "year": year, "roi_fraction": roi_fraction, "edge_density": float("nan"),
        "break_fraction_pct": float("nan"), "n_openings": 0, "opening_area_km2": float("nan"),
        "n_linear": 0, "linear_length_km": float("nan"), "linear_density_km_per_km2": float("nan"),
        "haze_low_sat_fraction": haze, "flag": "",
    }

    if roi_fraction < CANOPY_MIN_ROI_FRACTION:
        row["flag"] = "roi_too_small"
        log.info(f"  [{year}] Canopy analysis skipped: only {roi_fraction:.1%} of tile is analysable forest.")
        return row

    res = detect_canopy_breaks(img, roi)
    roads, openings = res["roads"], res["openings"]
    roi_km2 = roi_px * PIXEL_TO_KM2
    break_px = int(np.count_nonzero(roads)) + int(np.count_nonzero(openings))
    edge_density = int(np.count_nonzero(res["edges"])) / roi_px
    road_km = res["road_length_px"] * METERS_PER_PIXEL / 1000.0

    row.update({
        "edge_density": edge_density,
        "break_fraction_pct": 100.0 * break_px / roi_px,
        "n_openings": res["n_openings"],
        "opening_area_km2": int(np.count_nonzero(openings)) * PIXEL_TO_KM2,
        "n_linear": res["n_roads"],
        "linear_length_km": road_km,
        "linear_density_km_per_km2": road_km / roi_km2,
    })
    if edge_density > CANOPY_EDGE_SATURATION:
        row["flag"] = "edge_saturated"
        log.warning(f"  [{year}] {edge_density:.0%} of forest pixels are edges — raise CANOPY_CANNY_LOW/HIGH.")

    vis = img.copy()
    vis[openings == 255] = [255, 255, 0]   # cyan   = canopy opening
    vis[roads == 255] = [255, 0, 255]      # magenta = road / trail-like
    cv2.imwrite(os.path.join(OUTPUT_CANOPY_DIR, f"canopy_{year}.png"),
                cv2.addWeighted(img, 0.55, vis, 0.45, 0))
    cv2.imwrite(os.path.join(OUTPUT_CANOPY_DIR, f"canopy_mask_{year}.png"),
                np.maximum(roads, openings // 2))  # 255 = road-like, 127 = opening

    log.info(
        f"  [{year}] Canopy breaks: {row['break_fraction_pct']:.2f}% of forest | "
        f"roads/trails: {res['n_roads']} ({road_km:,.1f} km) | openings: {res['n_openings']}"
    )
    return row


def export_canopy_csv() -> None:
    rows = sorted(CANOPY_RESULTS, key=lambda r: r["year"])
    if not rows:
        log.warning("No canopy results to export.")
        return
    cols = list(rows[0].keys())
    with open(OUTPUT_CANOPY_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(cols)
        for r in rows:
            writer.writerow([
                "" if isinstance(r[c], float) and np.isnan(r[c]) else
                (f"{r[c]:.6f}" if isinstance(r[c], float) else r[c])
                for c in cols
            ])
    log.info(f"Wrote canopy results to '{OUTPUT_CANOPY_CSV}'.")

    ok = [r for r in rows if not np.isnan(r["break_fraction_pct"])]
    brk_v = [r["break_fraction_pct"] for r in ok]
    haze_v = [r["haze_low_sat_fraction"] for r in ok]
    if len(ok) >= 5 and np.ptp(brk_v) > 0 and np.ptp(haze_v) > 0:
        rho, p = sp_stats.spearmanr(brk_v, haze_v)
        log.info(f"Canopy break fraction vs haze (low-saturation) fraction: Spearman rho={rho:.2f}, p={p:.3g}")
        if abs(rho) > 0.5:
            log.warning("  Canopy metrics track haze strongly — treat year-to-year changes with caution.")


def plot_canopy_fragmentation() -> None:
    rows = sorted(CANOPY_RESULTS, key=lambda r: r["year"])
    ok = [r for r in rows if not np.isnan(r["break_fraction_pct"])]
    if len(ok) < 2:
        log.warning("Not enough canopy results to plot.")
        return

    yrs = np.array([r["year"] for r in ok])
    brk = np.array([r["break_fraction_pct"] for r in ok])
    road_density = np.array([r["linear_density_km_per_km2"] for r in ok])
    opening_km2 = np.array([r["opening_area_km2"] for r in ok])
    haze = np.array([r["haze_low_sat_fraction"] for r in ok]) * 100.0

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    fig.suptitle("Rondônia Canopy Structural Fragmentation (Prototype)",
                 fontsize=14, weight="bold", y=0.98)

    ax1.plot(yrs, brk, marker="o", color="#8e44ad", linewidth=2.0, markersize=5)
    ax1.set_title("Canopy Breaks in Remaining Forest (roads, trails, openings)", fontsize=12, weight="bold")
    ax1.set_ylabel("% of remaining forest", fontsize=11)

    ax2.plot(yrs, road_density, marker="s", color="#d35400", linewidth=2.0, markersize=5,
             label="Road/trail-like length density (km per km²)")
    ax2.set_ylabel("km / km²", fontsize=11)
    ax2b = ax2.twinx()
    ax2b.plot(yrs, opening_km2, marker="^", color="#16a085", linewidth=1.6, markersize=5,
              label="Canopy opening area (km²)")
    ax2b.set_ylabel("km²", fontsize=11)
    ax2.set_title("Road/Trail-like Features and Canopy Openings", fontsize=12, weight="bold")
    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax2b.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, fontsize=9, loc="upper left")

    ax3.plot(yrs, haze, marker="o", color="#7f8c8d", linewidth=1.8, markersize=5)
    ax3.set_title("Haze Proxy (share of low-saturation pixels) — check against panels above",
                  fontsize=12, weight="bold")
    ax3.set_ylabel("% of tile", fontsize=11)
    ax3.set_xlabel("Year", fontsize=11)

    for ax in (ax1, ax2, ax3):
        ax.grid(True, linestyle="--", alpha=0.6)
        for evt_year in POLICY_EVENTS:
            if yrs.min() <= evt_year <= yrs.max():
                ax.axvline(evt_year, color="#7f8c8d", linewidth=0.9, linestyle=":", alpha=0.7)
    ax3.set_xticks(yrs[::2])

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(OUTPUT_CANOPY_CHART, dpi=300)
    log.info(f"Saved canopy fragmentation chart to '{OUTPUT_CANOPY_CHART}'.")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main processing loop
# ---------------------------------------------------------------------------

def process_dataset() -> list[YearResult]:
    os.makedirs(OUTPUT_MASK_DIR, exist_ok=True)
    os.makedirs(OUTPUT_OVERLAY_DIR, exist_ok=True)
    os.makedirs(OUTPUT_FRONTIER_DIR, exist_ok=True)
    os.makedirs(OUTPUT_CANOPY_DIR, exist_ok=True)
    CANOPY_RESULTS.clear()

    image_paths = sorted(glob.glob(os.path.join(DATASET_DIR, FILENAME_PATTERN)))
    log.info(f"Found {len(image_paths)} candidate images in '{DATASET_DIR}'.")

    if not image_paths:
        log.warning("No images found — check DATASET_DIR / FILENAME_PATTERN.")
        return []

    results: list[YearResult] = []
    expected_shape = None
    seen_years = set()
    master_mask = None
    prev_master = None

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
                f"Skipping '{filename}': shape {img.shape} != expected {expected_shape}."
            )
            continue

        cleaned_mask, otsu_val = segment_deforestation(img)

        # Save previous cumulative before updating
        prev_master = master_mask.copy() if master_mask is not None else None

        # Cumulative OR
        if master_mask is None:
            master_mask = np.zeros_like(cleaned_mask)
        master_mask = cv2.bitwise_or(master_mask, cleaned_mask)

        # Area calculation
        cleared_pixels = int(np.count_nonzero(master_mask == 255))
        area_km2 = cleared_pixels * PIXEL_TO_KM2

        # --- NEW: frontier extraction + fractal dimension ---
        frontier, fd, r2, n_contours = extract_frontier_and_fd(
            master_mask, prev_master, year
        )
        frontier_pixels = int(np.count_nonzero(frontier == 255))
        frontier_km2 = frontier_pixels * PIXEL_TO_KM2

        # Save outputs
        cv2.imwrite(os.path.join(OUTPUT_MASK_DIR, f"mask_{year}.png"), master_mask)
        cv2.imwrite(os.path.join(OUTPUT_FRONTIER_DIR, f"frontier_{year}.png"), frontier)

        # Overlay: cumulative=red, new frontier=bright yellow
        overlay = img.copy()
        overlay[master_mask == 255] = [0, 0, 255]
        overlay[frontier == 255] = [0, 255, 255]  # yellow in BGR
        blended = cv2.addWeighted(img, 0.60, overlay, 0.40, 0)
        cv2.imwrite(os.path.join(OUTPUT_OVERLAY_DIR, f"overlay_{year}.png"), blended)

        # Save frontier edge visualization
        frontier_edges = cv2.Canny(frontier, 50, 150)
        edge_vis = cv2.cvtColor(frontier_edges, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(os.path.join(OUTPUT_FRONTIER_DIR, f"edges_{year}.png"), edge_vis)

        # --- NEW: canopy openness / logging-road detection (structural fragmentation) ---
        try:
            CANOPY_RESULTS.append(analyze_canopy_fragmentation(img, master_mask, year))
        except Exception as e:
            log.warning(f"  [{year}] Canopy fragmentation analysis failed: {e}")

        results.append(YearResult(
            year=year,
            area_km2=area_km2,
            otsu_threshold=otsu_val,
            frontier_km2=frontier_km2,
            fractal_dim=fd,
            fd_r_squared=r2,
            frontier_contour_count=n_contours,
        ))
        seen_years.add(year)

        fd_str = f"{fd:.4f} (R²={r2:.3f})" if not np.isnan(fd) else "N/A"
        log.info(
            f"[{year}] Area: {area_km2:,.2f} km² | "
            f"Frontier: {frontier_km2:,.2f} km² | "
            f"FD: {fd_str} | Contours: {n_contours}"
        )

    results.sort(key=lambda r: r.year)

    all_years = set(range(min(seen_years), max(seen_years) + 1)) if seen_years else set()
    missing = sorted(all_years - seen_years)
    if missing:
        log.info(f"Years missing from dataset (skipped): {missing}")

    return results


# ---------------------------------------------------------------------------
# Output: CSV
# ---------------------------------------------------------------------------

def compute_rates(results: list[YearResult]):
    years_arr  = np.array([r.year for r in results])
    areas_arr  = np.array([r.area_km2 for r in results])
    fd_arr     = np.array([r.fractal_dim for r in results])
    r2_arr     = np.array([r.fd_r_squared for r in results])

    delta_years = np.diff(years_arr).astype(float)
    delta_years[delta_years == 0] = np.nan
    delta_areas = np.diff(areas_arr)
    annual_rate = delta_areas / delta_years

    return years_arr, areas_arr, annual_rate, fd_arr, r2_arr


def export_csv(results: list[YearResult], years_arr, annual_rate) -> None:
    rate_by_year = dict(zip(years_arr[1:].tolist(), annual_rate.tolist()))
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "year",
            "deforested_area_km2",
            "otsu_threshold",
            "annual_rate_km2_per_year",
            "frontier_km2",
            "fractal_dimension",
            "fd_r_squared",
            "frontier_contour_count",
        ])
        for r in results:
            rate = rate_by_year.get(r.year, "")
            fd = f"{r.fractal_dim:.4f}" if not np.isnan(r.fractal_dim) else ""
            r2 = f"{r.fd_r_squared:.4f}" if not np.isnan(r.fd_r_squared) else ""
            writer.writerow([
                r.year,
                f"{r.area_km2:.4f}",
                f"{r.otsu_threshold:.2f}",
                rate,
                f"{r.frontier_km2:.4f}",
                fd,
                r2,
                r.frontier_contour_count,
            ])
    log.info(f"Wrote results to '{OUTPUT_CSV}'.")


# ---------------------------------------------------------------------------
# Output: Combined 3-panel chart
# ---------------------------------------------------------------------------

def plot_combined(years_arr, areas_arr, annual_rate, fd_arr, r2_arr) -> None:
    """Three-panel figure: cumulative area, annual rate, fractal dimension trend."""

    # Filter to years with valid FD
    fd_valid = ~np.isnan(fd_arr)
    fd_years = years_arr[fd_valid]
    fd_vals  = fd_arr[fd_valid]
    r2_vals  = r2_arr[fd_valid]

    # Mark high-confidence FD points (R² > 0.90)
    high_conf = r2_vals >= 0.90

    interval_labels = years_arr[1:]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 14), sharex=False)
    fig.suptitle(
        "Rondônia Deforestation Analysis: Area, Rate, and Boundary Fractal Dimension (1994–2024)",
        fontsize=14, weight="bold", y=0.98,
    )

    # --- Panel 1: Cumulative area ---
    ax1.plot(years_arr, areas_arr, marker="o", color="#b22222",
             linewidth=2.2, markersize=5)
    ax1.set_title("Estimated Cumulative Deforested Area", fontsize=12, weight="bold")
    ax1.set_ylabel("Cleared Area (km²)", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.6)

    # --- Panel 2: Annual rate ---
    bar_colors = ["#c0392b" if r >= 0 else "#27ae60" for r in annual_rate]
    ax2.bar(interval_labels, annual_rate, width=0.75, color=bar_colors,
            edgecolor="black", alpha=0.85)
    ax2.axhline(0, color="black", linewidth=0.9)
    ax2.set_title(
        r"Annual Rate of Forest Loss ($\frac{dA}{dt}$ in km²/year)",
        fontsize=12, weight="bold",
    )
    ax2.set_ylabel("Change Rate (km²/year)", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.6)

    # --- Panel 3: Fractal dimension trend ---
    # Plot all valid points
    ax3.scatter(
        fd_years[~high_conf], fd_vals[~high_conf],
        color="#aaaaaa", edgecolors="black", s=50, zorder=3,
        label="Low confidence (R² < 0.90)", alpha=0.6,
    )
    ax3.scatter(
        fd_years[high_conf], fd_vals[high_conf],
        color="#2c3e50", edgecolors="black", s=70, zorder=4,
        label="High confidence (R² ≥ 0.90)",
    )

    # Trend line through high-confidence points only
    if np.sum(high_conf) >= 3:
        hc_years = fd_years[high_conf]
        hc_vals  = fd_vals[high_conf]
        slope, intercept, r, p, se = sp_stats.linregress(hc_years, hc_vals)
        trend_x = np.linspace(hc_years.min(), hc_years.max(), 100)
        trend_y = slope * trend_x + intercept
        ax3.plot(trend_x, trend_y, "--", color="#e74c3c", linewidth=1.8, alpha=0.8,
                 label=f"Trend (slope={slope:.5f}/yr, R²={r**2:.3f}, p={p:.4f})")

    # Policy event annotations
    for evt_year, label in POLICY_EVENTS.items():
        for ax in [ax2, ax3]:
            ax.axvline(evt_year, color="#7f8c8d", linewidth=0.9, linestyle=":", alpha=0.7)
        # Annotate only on FD panel
        if years_arr.min() <= evt_year <= years_arr.max():
            ax3.annotate(
                label, xy=(evt_year, ax3.get_ylim()[1] if ax3.get_ylim()[1] else 2.0),
                xytext=(evt_year, 0), fontsize=7, ha="center", color="#7f8c8d",
                rotation=0,
            )

    ax3.set_title(
        "Fractal Dimension of Deforestation Frontier Boundaries",
        fontsize=12, weight="bold",
    )
    ax3.set_xlabel("Year", fontsize=11)
    ax3.set_ylabel("Fractal Dimension (D)", fontsize=11)
    ax3.legend(fontsize=9, loc="lower right")
    ax3.grid(True, linestyle="--", alpha=0.6)

    # Reference lines for interpretation
    ax3.axhline(1.0, color="#27ae60", linewidth=0.7, linestyle="-.", alpha=0.5)
    ax3.text(years_arr.min() + 0.5, 1.01, "D=1.0 (smooth line)", fontsize=7,
             color="#27ae60", alpha=0.7)
    ax3.axhline(2.0, color="#8e44ad", linewidth=0.7, linestyle="-.", alpha=0.5)
    ax3.text(years_arr.min() + 0.5, 2.01, "D=2.0 (space-filling)", fontsize=7,
             color="#8e44ad", alpha=0.7)

    for ax in [ax1, ax2, ax3]:
        ax.set_xticks(years_arr[::2])

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(OUTPUT_COMBINED, dpi=300)
    log.info(f"Saved combined chart to '{OUTPUT_COMBINED}'.")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    results = process_dataset()

    if len(results) < 2:
        log.warning("Need >= 2 valid years. Stopping.")
        return

    years_arr, areas_arr, annual_rate, fd_arr, r2_arr = compute_rates(results)
    export_csv(results, years_arr, annual_rate)
    plot_combined(years_arr, areas_arr, annual_rate, fd_arr, r2_arr)

    # --- NEW: canopy fragmentation outputs ---
    export_canopy_csv()
    plot_canopy_fragmentation()

    # Summary statistics
    valid_fd = fd_arr[~np.isnan(fd_arr)]
    if len(valid_fd) > 0:
        log.info(f"\n{'='*60}")
        log.info(f"FRACTAL DIMENSION SUMMARY")
        log.info(f"  Valid years:  {len(valid_fd)} / {len(fd_arr)}")
        log.info(f"  Mean FD:      {np.mean(valid_fd):.4f}")
        log.info(f"  Std FD:       {np.std(valid_fd):.4f}")
        log.info(f"  Range:        [{np.min(valid_fd):.4f}, {np.max(valid_fd):.4f}]")
        log.info(f"{'='*60}")

    log.info("Processing complete.")


if __name__ == "__main__":
    main()