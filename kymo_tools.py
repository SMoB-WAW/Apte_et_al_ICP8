import lumicks.pylake as lk
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import peakutils

import os
import pandas as pd



def load_h5(filepath):
    """Load a LUMICKS .h5 file."""
    return lk.File(filepath)


def list_kymos(file):
    """Show available kymographs in the file."""
    return list(file.kymos.keys())


def get_kymo(file, kymo_name):
    """Get one kymograph by name."""
    return file.kymos[kymo_name]


def get_kymo_image(kymo, channel="rgb"):
    """Extract image data from a kymograph."""
    return kymo.get_image(channel)


### Plotting kymograph images with contrast adjustment
def plot_kymo(kymo, channel="rgb", save_path=None, contrast="percentile", flip_y=False, blue_gain=2,):
    image = kymo.get_image(channel=channel).astype(float)
    if flip_y:
        image = image[::-1, :, :]

    plt.figure(figsize=(10, 4))

    if contrast == "percentile":
        vmin = np.percentile(image, 1)
        vmax = np.percentile(image, 99)
    elif contrast == "minmax":
        vmin = np.min(image)

        vmax = np.max(image)
    else:
        vmin = None
        vmax = None


    # Image normalization zakresu 0–1
    image = (image - vmin) / (vmax - vmin)
    image = np.clip(image, 0, 1)

    # Blue channel gain adjustment 
    if channel == "rgb" and image.ndim == 3:
        image[..., 2] *= blue_gain
        image[..., 2] = np.clip(image[..., 2], 0, 1)

    plt.imshow(image, aspect="auto", vmin=vmin, vmax=vmax)
    plt.xlabel("Time")
    plt.ylabel("Position")
    plt.title("Kymograph")

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()



### Extracting RGB intensity data from kymograph images
def extract_rgb_cumulative_intensity(data, axis=1):
    if hasattr(data, "get_image"):
        image = data.get_image(channel="rgb")
    else:
        image = data

    R = np.sum(image[:, :, 0], axis=axis)
    G = np.sum(image[:, :, 1], axis=axis)
    B = np.sum(image[:, :, 2], axis=axis)

    return R, G, B


### Exporting RGB intensity data to CSV
def export_rgb_intensity_csv(R, G, B, kymo_name, coords=None, save_dir="../results"):
    os.makedirs(save_dir, exist_ok=True)

    def format_range(coords, key):
        if coords is None:
            return f"{key}full"
        if key not in coords:
            return f"{key}full"
        if coords[key] is None:
            return f"{key}full"
        return f"{key}{coords[key][0]}-{coords[key][1]}"

    y_label = format_range(coords, "y")
    x_label = format_range(coords, "x")
    z_label = format_range(coords, "z")

    coord_label = f"{y_label}_{x_label}_{z_label}"

    safe_kymo_name = (
        kymo_name
        .replace(" ", "_")
        .replace("/", "-")
        .replace("\\", "-")
        .replace(":", "-")
    )

    csv_path = os.path.join(
        save_dir,
        f"{safe_kymo_name}_{coord_label}_rgb_intensity.csv"
    )

    df = pd.DataFrame({
        "index": np.arange(len(R)),
        "red": R,
        "green": G,
        "blue": B,
    })

    df.to_csv(csv_path, index=False)

    return csv_path


### Export RGB intensity plot to PNG
def export_rgb_intensity_plot(R, G, B, kymo_name, coords=None, save_dir="../results"):
    os.makedirs(save_dir, exist_ok=True)

    def format_range(coords, key):
        if coords is None:
            return f"{key}full"
        if key not in coords:
            return f"{key}full"
        if coords[key] is None:
            return f"{key}full"
        return f"{key}{coords[key][0]}-{coords[key][1]}"

    y_label = format_range(coords, "y")
    x_label = format_range(coords, "x")
    z_label = format_range(coords, "z")

    coord_label = f"{y_label}_{x_label}_{z_label}"

    safe_kymo_name = (
        kymo_name
        .replace(" ", "_")
        .replace("/", "-")
        .replace("\\", "-")
        .replace(":", "-")
    )

    png_path = os.path.join(
        save_dir,
        f"{safe_kymo_name}_{coord_label}_rgb_intensity.png"
    )

    plt.figure(figsize=(10, 4))
    plt.plot(R, color="red", label="Red")
    plt.plot(G, color="green", label="Green")
    plt.plot(B, color="blue", label="Blue")
    plt.xlabel("Position")
    plt.ylabel("Cumulative intensity")
    plt.legend()

    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close()

    return png_path



# Peak detection and relative positioning

def get_pixel_size_um(kymo):
    """Return position-axis pixel size in µm from a pylake Kymograph."""
    try:
        ps = kymo.pixelsize_um
        return float(ps[0]) if hasattr(ps, "__len__") else float(ps)
    except (AttributeError, TypeError):
        return None


def _detect_peaks(profile, threshold=0.3, min_dist=5):
    """Normalise profile and return peak indices via peakutils."""
    p = profile.astype(float)
    p_range = p.max() - p.min()
    if p_range == 0:
        return np.array([], dtype=int)
    p_norm = (p - p.min()) / p_range
    return peakutils.indexes(p_norm, thres=threshold, min_dist=min_dist)


def find_red_reference_peak(R, threshold=0.3, min_dist=5):
    """Return pixel index of the single red reference peak (tallest if multiple found)."""
    peaks = _detect_peaks(R, threshold=threshold, min_dist=min_dist)
    if len(peaks) == 0:
        raise ValueError("No red peak found — try lowering red_threshold.")
    return int(peaks[np.argmax(R[peaks])])


def analyze_window(kymo, x_range=None, y_range=None,
                   green_threshold=0.3, red_threshold=0.3, min_dist=5,
                   force_channel=None, flip_y=False, red_peak_override=None):
    """
    One time window of a kymograph.

    Returns a dict with intensity profiles, detected peak indices (pixels),
    and positions relative to the red reference peak in both px and µm.
    """
    image = kymo.get_image(channel="rgb")
    if flip_y:
        image = image[::-1, :, :]

    y_sl = slice(*y_range) if y_range else slice(None)
    x_sl = slice(*x_range) if x_range else slice(None)
    sub = image[y_sl, x_sl, :]

    R = np.sum(sub[:, :, 0], axis=1).astype(float)
    G = np.sum(sub[:, :, 1], axis=1).astype(float)
    B = np.sum(sub[:, :, 2], axis=1).astype(float)

    pixel_size_um = get_pixel_size_um(kymo)

    if red_peak_override is not None:
        red_px = int(red_peak_override)
    else:
        red_px = find_red_reference_peak(R, threshold=red_threshold, min_dist=min_dist)
    green_peaks = _detect_peaks(G, threshold=green_threshold, min_dist=min_dist)

    # Absolute distance from red peak (flip-invariant)
    rel_px = np.abs(green_peaks - red_px)
    rel_um = rel_px * pixel_size_um if pixel_size_um is not None else None

    # x-axis array in µm with red peak at 0 (used for plotting)
    n_pixels = len(R)
    if pixel_size_um is not None:
        positions_um = (np.arange(n_pixels) - red_px) * pixel_size_um
    else:
        positions_um = None

    # Median force over the selected time window
    median_force_pN = None
    if force_channel is not None:
        all_line_ts = kymo.line_timestamp_ranges()
        x0 = x_range[0] if x_range else 0
        x1 = x_range[1] if x_range else len(all_line_ts)
        selected_ts = all_line_ts[x0:x1]
        force_per_line = force_channel.downsampled_over(selected_ts)
        if len(force_per_line.data) > 0:
            median_force_pN = float(np.median(force_per_line.data))

    return {
        "R": R, "G": G, "B": B,
        "red_peak_px": red_px,
        "red_peak_um": red_px * pixel_size_um if pixel_size_um is not None else None,
        "green_peaks_px": green_peaks,
        "green_rel_px": rel_px,
        "green_rel_um": rel_um,       # signed: + means further from red along position axis
        "positions_um": positions_um, # x-axis for plotting, 0 = red peak
        "pixel_size_um": pixel_size_um,
        "median_force_pN": median_force_pN,
        "x_range": x_range,
        "y_range": y_range,
    }


def plot_window_analysis(result, kymo_name="", title_suffix="", save_path=None):
    """
    Plot intensity profiles with the x-axis in µm, red peak at 0.
    Green peak labels show their distance from the red reference in µm.
    If save_path is given, saves to file and does not render in Jupyter.
    """
    R, G = result["R"], result["G"]
    green_peaks = result["green_peaks_px"]
    red_px = result["red_peak_px"]
    px_size = result["pixel_size_um"]
    positions_um = result["positions_um"]

    # Fall back to pixel axis if no calibration available
    use_um = positions_um is not None
    x = positions_um if use_um else np.arange(len(R))
    x_unit = "µm" if use_um else "px"

    # x-coordinates of each peak on the chosen axis
    green_x = (green_peaks - red_px) * px_size if use_um else green_peaks.astype(float)
    red_x = 0.0 if use_um else float(red_px)

    #fig, axes = plt.subplots(2, 1, figsize=(6, 6), sharex=True)

    #axes[0].plot(x, R, color="red")
    #axes[0].axvline(red_x, color="darkred", linestyle="--", label="Red ref (0)")
    #axes[0].set_ylabel("Cumulative intensity [a.u.]")
    #axes[0].set_title(f"Red channel — reference  {title_suffix}")
    #axes[0].legend(fontsize=8)

    #axes[1].plot(x, G, color="green")
    #for gx, rel_um in zip(green_x, result["green_rel_um"] if use_um else result["green_rel_px"]):
     #   lbl = f"{rel_um:.3f} {x_unit}" if use_um else f"{int(rel_um):+d} px"
      #  axes[0].axvline(gx, color="darkgreen", linestyle="--", alpha=0.8, label=lbl)
    #axes[1].axvline(red_x, color="darkred", linestyle="--", alpha=0.4, label="Red ref (0)")
    #axes[1].set_ylabel("Cumulative intensity [a.u.]")
    #axes[1].set_xlabel(f"Distance from red reference ({x_unit})")
    #axes[1].set_title(f"Green channel — peaks  {title_suffix}")
    #axes[1].legend(fontsize=8)


    fig, ax = plt.subplots(figsize=(4, 3))
    
    ax.plot(x, R, color="red", linewidth=3, label="Red channel")
    ax.plot(x, G, color="green", linewidth=3, label="Green channel")

    ax.axvline(
        red_x,
        color="darkred",
        linestyle="--",
        alpha=0.6,
        label="Reference marker (0)",
    )

    green_rel = result["green_rel_um"] if use_um else result["green_rel_px"]

    for gx, rel_value in zip(green_x, green_rel):   #ADD minus sign to green_x and green_rel to reverse x axis
        lbl = (
            f"{rel_value:.3f} {x_unit}"
            if use_um
            else f"{int(rel_value):+d} px"
        )
        ax.axvline(
            gx,
            color="darkgreen",
            linestyle="--",
            alpha=0.8,
            label=lbl,
        )

    ax.set_xlabel(f"Distance from red reference marker ({x_unit})")
    ax.set_ylabel("Cumulative intensity [a.u.]")
    #ax.set_xlim(-0.6,3)
    #ax.set_ylim(-20,1300)
    #ax.set_title(f"Red and green channels {title_suffix}")
    ax.legend(fontsize=8)

    fig.tight_layout()





    force_str = ""
    if result.get("median_force_pN") is not None:
        force_str = f"   |   Median Trap2 force: {result['median_force_pN']:.2f} pN"
    plt.suptitle(f"{kymo_name}{force_str}", fontsize=10)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
    return fig


def export_peak_analysis_csv(results, kymo_name, save_dir="../results"):
    """
    Export peak positions from a list of analyze_window() results to CSV.

    Each row = one green peak.  Columns: kymo_name, x_range, green_peak_px,
    red_peak_px, green_rel_px, green_rel_um.
    """
    os.makedirs(save_dir, exist_ok=True)
    rows = []
    for res in results:
        n_peaks = len(res["green_peaks_px"])
        rel_um_vals = res["green_rel_um"] if res["green_rel_um"] is not None else [None] * n_peaks
        for pk, rel_px, rel_um in zip(res["green_peaks_px"], res["green_rel_px"], rel_um_vals):
            y_range = res["y_range"]
            y_offset = y_range[0] if y_range is not None else 0
            rows.append({
                "kymo_name": kymo_name,
                "x_range": str(res["x_range"]),
                "median_force_pN": res.get("median_force_pN"),
                "red_peak_px": res["red_peak_px"],
                "red_peak_orig_y_px": y_offset + res["red_peak_px"],
                "green_peak_px": int(pk),
                "green_peak_orig_y_px": y_offset + int(pk),
                "green_pos_from_red_px": int(rel_px),
                "green_pos_from_red_um": float(rel_um) if rel_um is not None else None,
            })

    df = pd.DataFrame(rows)
    safe_name = kymo_name.replace(" ", "_").replace("/", "-").replace("\\", "-").replace(":", "-")
    csv_path = os.path.join(save_dir, f"{safe_name}_peak_analysis.csv")
    df.to_csv(csv_path, index=False)
    return csv_path, df


def plot_peak_comparison(df, title="Green peak positions relative to red reference"):
    """
    Dot/strip plot of green peak positions across kymograph windows.
    One column per (kymo, x_range) combination; dots = individual peaks.
    """
    use_um = "green_pos_from_red_um" in df.columns and df["green_pos_from_red_um"].notna().any()
    val_col = "green_pos_from_red_um" if use_um else "green_pos_from_red_px"
    unit = "µm" if use_um else "px"

    labels = df["kymo_name"] + "\nx=" + df["x_range"].astype(str)
    unique_labels = labels.unique()

    fig, ax = plt.subplots(figsize=(max(6, len(unique_labels) * 1.4), 5))
    for i, lbl in enumerate(unique_labels):
        vals = df.loc[labels == lbl, val_col].dropna().values
        ax.scatter([i] * len(vals), vals, s=60, alpha=0.75, zorder=3)

    ax.axhline(0, color="red", linestyle="--", linewidth=1, label="Red ref (0)")
    ax.set_xticks(range(len(unique_labels)))
    ax.set_xticklabels(unique_labels, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel(f"Position relative to red peak ({unit})")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()
    return fig