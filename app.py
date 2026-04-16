# Updated Pass Map -> Actions + xT (events replaced with user-provided coordinates)
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mplsoccer import Pitch
import pandas as pd
import numpy as np
from PIL import Image
from io import BytesIO
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle
from streamlit_image_coordinates import streamlit_image_coordinates
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, ListedColormap, LinearSegmentedColormap

# ==========================
# Page Configuration
# ==========================
st.set_page_config(layout="wide", page_title="Pass Map Dashboard (Actions + xT)")

# ==========================
# CSS
# ==========================
st.markdown(
    """
    <style>
    /* Container spacing */
    .small-metric { padding: 6px 8px; }

    /* Labels (small) */
    .small-metric .label {
      font-size: 12px;
      color: #ffffff;
      margin-bottom: 3px;
      opacity: 0.95;
    }

    /* Main value */
    .small-metric .value {
      font-size: 18px;
      font-weight: 600;
      color: #ffffff;
    }

    /* Delta / secondary text */
    .small-metric .delta {
      font-size: 11px;
      color: #e6e6e6;
      margin-top: 4px;
    }

    /* Section titles inside expanders */
    .stats-section-title {
      font-size: 14px;
      font-weight: 600;
      margin-bottom: 6px;
      color: #ffffff;
    }

    /* Expander headers */
    .streamlit-expanderHeader {
      color: #ffffff !important;
    }

    .streamlit-expander {
      background: rgba(255,255,255,0.02);
    }

    /* ===== Filter sidebar background ===== */
    .filter-panel {
      background: linear-gradient(168deg, rgba(30, 39, 56, 0.92) 0%, rgba(22, 28, 40, 0.97) 100%);
      border: 1px solid rgba(255, 255, 255, 0.08);
      border-radius: 14px;
      padding: 24px 18px 20px 18px;
      box-shadow: 0 4px 24px rgba(0,0,0,0.25), 0 1px 4px rgba(0,0,0,0.12);
      backdrop-filter: blur(6px);
    }

    .filter-panel h3 {
      font-size: 15px;
      color: #c8d6e5;
      letter-spacing: 0.5px;
      margin-bottom: 8px;
    }

    .filter-panel .filter-divider {
      border: none;
      border-top: 1px solid rgba(255,255,255,0.07);
      margin: 14px 0;
    }

    /* Make Streamlit subheaders slightly lighter on dark bg when we still use them */
    .stSubheader { color: #ffffff !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

def small_metric(label: str, value: str, delta: str | None = None):
    html = f"""
    <div class="small-metric">
      <div class="label">{label}</div>
      <div class="value">{value}</div>
    """
    if delta is not None:
        html += f'<div class="delta">{delta}</div>'
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


# ==========================
# Configuration / constants
# ==========================
st.title("Pass Map Dashboard")

FIELD_X, FIELD_Y = 120.0, 80.0
HALF_LINE_X = FIELD_X / 2
FINAL_THIRD_LINE_X = 80

LANE_LEFT_MIN = 53.33
LANE_RIGHT_MAX = 26.67

NX, NY = 16, 12

LATERAL_MIN_DIST = 12.0

# Colours
COLOR_SUCCESS = "#ffffff"
COLOR_FAIL = "#D45B5B"
COLOR_PROGRESSIVE = "#2F80ED"
COLOR_SWITCH = "#DAA520"

# ==========================
# xT GRID
# ==========================
x_progress = np.linspace(0.01, 1.00, NX)
y_central = 1 - np.abs(np.linspace(0, 1, NY) - 0.5) * 2

XT_GRID = np.zeros((NY, NX))
for iy in range(NY):
    for ix in range(NX):
        XT_GRID[iy, ix] = 0.80 * x_progress[ix] + 0.20 * x_progress[ix] * y_central[iy]
XT_GRID = (XT_GRID - XT_GRID.min()) / (XT_GRID.max() - XT_GRID.min() + 1e-9)

def zone_index(x, y):
    x = np.clip(x, 0, FIELD_X - 1e-9)
    y = np.clip(y, 0, FIELD_Y - 1e-9)
    ix = int((x / FIELD_X) * NX)
    iy = int((y / FIELD_Y) * NY)
    return ix, iy

def xt_value(x, y):
    ix, iy = zone_index(x, y)
    return XT_GRID[iy, ix]

# ==========================
# DATA (REPLACED WITH USER EVENTS: actions start -> end)
# ==========================
matches_data = {
    "Ali vs Vancouver": [
        # Positivos
        ("PASS WON", 50.03, 5.76, 48.86, 14.07, None),
        ("PASS WON", 42.05, 4.26, 65.82, 19.39, None),
        ("PASS WON", 53.68, 12.57, 39.72, 28.20, None),
        ("PASS WON", 43.88, 37.17, 44.54, 44.65, None),
        ("PASS WON", 76.29, 23.21, 65.65, 22.38, None),
        ("PASS WON", 78.62, 25.54, 87.26, 26.37, None),
        ("PASS WON", 67.48, 5.76, 76.96, 6.42, None),
        ("PASS WON", 61.83, 3.43, 111.20, 9.75, None),
        ("PASS WON", 83.27, 2.93, 118.51, 19.89, None),
        ("PASS WON", 97.90, 6.75, 111.53, 9.91, None),
        ("PASS WON", 114.03, 1.93, 107.71, 12.57, None),
        ("PASS WON", 98.23, 5.59, 90.09, 7.58, None),
        ("PASS WON", 96.57, 5.92, 91.92, 14.73, None),
        ("PASS WON", 87.43, 12.24, 78.78, 9.41, None),
        ("PASS WON", 77.62, 1.93, 72.30, 3.93, None),
        ("PASS WON", 79.28, 5.59, 70.81, 2.26, None),
        ("PASS WON", 62.83, 3.43, 79.62, 7.25, None),
        ("PASS WON", 53.18, 9.41, 68.98, 13.74, None),
        ("PASS WON", 51.69, 4.76, 40.38, 8.58, None),
        # Negativos
        ("PASS LOST", 116.35, 2.93, 118.68, 11.74, None),
        ("PASS LOST", 107.88, 10.58, 109.54, 39.83, None),
        ("PASS LOST", 86.10, 3.43, 87.59, 4.09, None),
        ("PASS LOST", 73.46, 2.43, 75.13, 3.43, None),
        ("PASS LOST", 53.18, 2.60, 70.47, 8.58, None),
        ("PASS LOST", 50.19, 6.09, 67.15, 10.91, None),
        ("PASS LOST", 47.70, 6.09, 55.01, 14.90, None),
        ("PASS LOST", 45.87, 35.84, 79.28, 50.14, None),
        ("PASS LOST", 54.51, 4.43, 54.35, 15.56, None),
        ("PASS LOST", 64.99, 0.94, 70.97, 1.93, None),
        ("PASS LOST", 87.43, 7.25, 87.43, 20.88, None),
        ("PASS LOST", 93.25, 7.92, 119.18, 39.67, None),
        ("PASS LOST", 99.90, 13.57, 98.90, 23.21, None),
    ],
    "Vs Dallas": [
        # Positivos
        ("PASS WON", 56.01, 3.43, 45.21, 8.08, None),
        ("PASS WON", 44.04, 2.10, 38.22, 7.25, None),
        ("PASS WON", 46.54, 11.24, 35.56, 9.91, None),
        ("PASS WON", 41.22, 10.91, 50.03, 15.23, None),
        ("PASS WON", 96.57, 2.26, 104.05, 28.86, None),
        ("PASS WON", 82.28, 22.55, 106.55, 1.43, None),
        ("PASS WON", 78.78, 21.05, 84.94, 20.72, None),
        ("PASS WON", 75.79, 18.89, 86.60, 55.63, None),
        ("PASS WON", 96.07, 39.00, 101.39, 39.00, None),
        # Negativos
        ("PASS LOST", 88.09, 12.24, 87.43, 4.26, None),
        ("PASS LOST", 78.62, 4.76, 87.59, 1.60, None),
        ("PASS LOST", 53.85, 1.60, 52.69, 1.10, None),
        ("PASS LOST", 52.85, 2.93, 62.49, 13.07, None),
        ("PASS LOST", 40.22, 22.55, 91.09, 25.54, None),
    ],
    "vs Sagoya": [
        # Sucesso
        ("PASS WON", 116.19, 14.40, 109.54, 29.36, None),
        ("PASS WON", 91.92, 3.43, 85.27, 7.75, None),
        ("PASS WON", 57.51, 6.09, 56.01, 26.70, None),
        ("PASS WON", 118.35, 1.43, 108.87, 46.82, None),
        ("PASS WON", 103.72, 40.83, 105.05, 42.49, None),
        ("PASS WON", 86.93, 4.76, 107.88, 31.36, None),
        ("PASS WON", 65.82, 40.50, 79.95, 30.86, None),
        ("PASS WON", 75.79, 8.08, 74.79, 27.53, None),
        ("PASS WON", 74.46, 5.09, 71.64, 14.07, None),
        ("PASS WON", 67.31, 2.10, 61.83, 10.91, None),
        ("PASS WON", 67.65, 5.92, 51.52, 8.08, None),
        ("PASS WON", 62.49, 2.60, 66.65, 9.41, None),
        ("PASS WON", 47.03, 2.43, 50.03, 15.73, None),
        ("PASS WON", 37.23, 10.24, 53.35, 12.57, None),
        ("PASS WON", 23.59, 2.76, 32.07, 4.92, None),
        ("PASS WON", 20.94, 14.23, 33.24, 7.25, None),
        ("PASS WON", 14.62, 18.22, 6.64, 37.01, None),
        # Falha
        ("PASS LOST", 51.19, 3.59, 117.68, 14.07, None),
        ("PASS LOST", 65.15, 6.59, 113.86, 20.05, None),
        ("PASS LOST", 90.92, 2.76, 94.24, 4.76, None),
        ("PASS LOST", 97.74, 7.09, 101.56, 20.05, None),
        ("PASS LOST", 84.44, 6.59, 91.09, 13.90, None),
    ],
    "Vs Busan Park": [
        # Sucesso
        ("PASS WON", 114.52, 19.05, 103.72, 21.22, None),
        ("PASS WON", 92.25, 21.88, 112.20, 24.21, None),
        ("PASS WON", 99.90, 23.21, 90.59, 24.87, None),
        ("PASS WON", 86.93, 2.10, 82.61, 10.74, None),
        ("PASS WON", 85.93, 4.92, 94.41, 32.69, None),
        ("PASS WON", 89.59, 3.26, 80.95, 26.87, None),
        ("PASS WON", 84.27, 10.74, 76.12, 3.59, None),
        ("PASS WON", 54.51, 2.76, 52.85, 17.56, None),
        ("PASS WON", 56.01, 9.08, 46.04, 8.75, None),
        ("PASS WON", 20.94, 2.43, 2.15, 7.58, None),
        ("PASS WON", 96.90, 10.41, 111.03, 35.68, None),
        ("PASS WON", 88.26, 33.35, 97.74, 8.08, None),
        ("PASS WON", 51.02, 18.39, 66.48, 15.23, None),
        ("PASS WON", 34.57, 56.12, 69.31, 5.92, None),
        ("PASS WON", 53.52, 33.35, 65.15, 45.98, None),
        ("PASS WON", 46.37, 51.47, 85.60, 46.48, None),
        ("PASS WON", 88.26, 47.98, 107.21, 56.29, None),
        ("PASS WON", 89.42, 50.31, 100.89, 65.10, None),
        # Falha
        ("PASS LOST", 113.53, 9.25, 119.01, 38.67, None),
        ("PASS LOST", 63.16, 37.34, 80.95, 37.67, None),
        ("PASS LOST", 58.34, 16.56, 67.65, 26.04, None),
        ("PASS LOST", 67.81, 6.59, 75.96, 31.52, None),
        ("PASS LOST", 34.57, 57.95, 49.03, 55.63, None),
    ],
    "Vs Atlanta": [
        # Sucesso
        ("PASS WON", 95.08, 11.57, 94.41, 0.44, None),
        ("PASS WON", 54.68, 14.23, 49.36, 19.22, None),
        ("PASS WON", 33.74, 0.60, 28.42, 7.42, None),
        ("PASS WON", 38.06, 10.24, 20.27, 24.04, None),
        ("PASS WON", 15.28, 11.41, 3.48, 30.52, None),
        ("PASS WON", 26.25, 35.68, 32.07, 41.83, None),
        ("PASS WON", 53.85, 44.65, 80.95, 51.30, None),
        ("PASS WON", 72.97, 36.18, 98.23, 65.60, None),
        ("PASS WON", 102.56, 66.76, 93.08, 47.31, None),
        # Falha
        ("PASS LOST", 67.81, 69.59, 70.97, 78.73, None),
        ("PASS LOST", 31.91, 2.43, 41.05, 13.40, None),
    ],
}

# ==========================
# Helpers, DataFrames, Stats, Draw functions
# (unchanged from original code)
# ==========================
def has_video_value(v) -> bool:
    return pd.notna(v) and str(v).strip() != ""

def get_lane(y):
    if y >= LANE_LEFT_MIN:
        return "left"
    elif y < LANE_RIGHT_MAX:
        return "right"
    else:
        return "center"

def is_switch_pass(x_start, y_start, y_end) -> bool:
    if x_start >= FINAL_THIRD_LINE_X:
        return False
    lane_start = get_lane(y_start)
    lane_end = get_lane(y_end)
    if lane_start == "left" and lane_end == "right":
        return True
    if lane_start == "right" and lane_end == "left":
        return True
    return False

def progressive_wyscout(x_start, x_end) -> bool:
    dist_start = FIELD_X - x_start
    dist_end = FIELD_X - x_end
    closer_by = dist_start - dist_end
    start_own = x_start < HALF_LINE_X
    end_own = x_end < HALF_LINE_X
    start_opp = x_start >= HALF_LINE_X
    end_opp = x_end >= HALF_LINE_X
    if start_own and end_own:
        return closer_by >= 30.0
    if (start_own and end_opp) or (start_opp and end_own):
        return closer_by >= 15.0
    if start_opp and end_opp:
        return closer_by >= 10.0
    return False

def classify_pass_direction(x_start, y_start, x_end, y_end) -> str:
    dx = x_end - x_start
    dy = y_end - y_start
    dist = np.sqrt(dx ** 2 + dy ** 2)
    angle_deg = np.degrees(np.arctan2(abs(dy), dx))
    if angle_deg <= 45.0:
        return "forward"
    elif angle_deg >= 135.0:
        return "backward"
    else:
        if dist > LATERAL_MIN_DIST:
            return "lateral"
        else:
            if dx >= 0:
                return "forward"
            else:
                return "backward"

# Build DataFrames
dfs_by_match = {}
for match_name, events in matches_data.items():
    dfm = pd.DataFrame(events, columns=["type", "x_start", "y_start", "x_end", "y_end", "video"])
    dfm["match"] = match_name
    dfm["number"] = np.arange(1, len(dfm) + 1)
    dfm["is_won"] = dfm["type"].str.contains("WON", case=False)
    dfm["outcome"] = np.where(dfm["is_won"], "successful", "failed")
    dfm["switch"] = dfm.apply(lambda row: is_switch_pass(row["x_start"], row["y_start"], row["y_end"]), axis=1)
    dfm["direction"] = dfm.apply(lambda row: classify_pass_direction(row["x_start"], row["y_start"], row["x_end"], row["y_end"]), axis=1)
    dfm["is_forward"] = dfm["direction"] == "forward"
    dfm["is_backward"] = dfm["direction"] == "backward"
    dfm["is_lateral"] = dfm["direction"] == "lateral"
    dfm["is_progressive_wyscout"] = dfm.apply(lambda row: progressive_wyscout(row["x_start"], row["x_end"]), axis=1)
    dfm["xt_start"] = dfm.apply(lambda r: xt_value(r["x_start"], r["y_start"]), axis=1)
    dfm["xt_end"] = dfm.apply(lambda r: xt_value(r["x_end"], r["y_end"]), axis=1)
    dfm["delta_xt"] = np.where(dfm["outcome"].eq("successful"), dfm["xt_end"] - dfm["xt_start"], 0.0)
    dfm["pass_distance"] = np.sqrt((dfm["x_end"] - dfm["x_start"]) ** 2 + (dfm["y_end"] - dfm["y_start"]) ** 2)
    dfs_by_match[match_name] = dfm

df_all = pd.concat(dfs_by_match.values(), ignore_index=True)
full_data = {"All Matches": df_all}
full_data.update(dfs_by_match)

def compute_stats(df: pd.DataFrame) -> dict:
    total_passes = len(df)
    successful = int(df["is_won"].sum())
    unsuccessful = total_passes - successful
    accuracy = (successful / total_passes * 100.0) if total_passes else 0.0
    key_passes = int(df["video"].apply(has_video_value).sum())
    forward_total = int(df["is_forward"].sum())
    forward_success = int((df["is_forward"] & df["is_won"]).sum())
    pct_forward = (forward_total / total_passes * 100.0) if total_passes else 0.0
    backward_total = int(df["is_backward"].sum())
    backward_success = int((df["is_backward"] & df["is_won"]).sum())
    pct_backward = (backward_total / total_passes * 100.0) if total_passes else 0.0
    lateral_total = int(df["is_lateral"].sum())
    lateral_success = int((df["is_lateral"] & df["is_won"]).sum())
    pct_lateral = (lateral_total / total_passes * 100.0) if total_passes else 0.0
    switch_total = int(df["switch"].sum())
    switch_success = int((df["switch"] & df["is_won"]).sum())
    switch_unsuccess = switch_total - switch_success
    switch_accuracy = (switch_success / switch_total * 100.0) if switch_total else 0.0
    switch_pct_of_total = (switch_total / total_passes * 100.0) if total_passes else 0.0
    prog_wyscout_total = int(df["is_progressive_wyscout"].sum())
    prog_wyscout_success = int((df["is_progressive_wyscout"] & df["is_won"]).sum())
    pct_progressive_wyscout = prog_wyscout_total / total_passes * 100.0 if total_passes else 0.0
    prog_wyscout_accuracy = prog_wyscout_success / prog_wyscout_total * 100.0 if prog_wyscout_total else 0.0
    prog_success_mask = df["is_progressive_wyscout"] & (df["outcome"] == "successful")
    xt_prog_sum = float(df.loc[prog_success_mask, "delta_xt"].sum())
    xt_prog_mean = float(df.loc[prog_success_mask, "delta_xt"].mean()) if prog_success_mask.any() else 0.0
    xt_total_sum = float(df.loc[df["outcome"] == "successful", "delta_xt"].sum())
    xt_total_mean = float(df.loc[df["outcome"] == "successful", "delta_xt"].mean()) if (df["outcome"] == "successful").any() else 0.0
    positive_xt_mask = (df["outcome"] == "successful") & (df["delta_xt"] > 0)
    positive_xt_total = int(positive_xt_mask.sum())
    positive_xt_sum = float(df.loc[positive_xt_mask, "delta_xt"].sum())
    positive_xt_mean = float(df.loc[positive_xt_mask, "delta_xt"].mean()) if positive_xt_mask.any() else 0.0
    return {
        "total_passes": total_passes,
        "successful_passes": successful,
        "unsuccessful_passes": unsuccessful,
        "accuracy_pct": round(accuracy, 2),
        "key_passes": key_passes,
        "forward_total": forward_total,
        "forward_success": forward_success,
        "pct_forward": pct_forward,
        "backward_total": backward_total,
        "backward_success": backward_success,
        "pct_backward": pct_backward,
        "lateral_total": lateral_total,
        "lateral_success": lateral_success,
        "pct_lateral": pct_lateral,
        "switch_total": switch_total,
        "switch_success": switch_success,
        "switch_unsuccess": switch_unsuccess,
        "switch_accuracy_pct": round(switch_accuracy, 2),
        "switch_pct_of_total": round(switch_pct_of_total, 2),
        "prog_wyscout_total": prog_wyscout_total,
        "prog_wyscout_success": prog_wyscout_success,
        "pct_progressive_wyscout": round(pct_progressive_wyscout, 2),
        "prog_wyscout_accuracy_pct": round(prog_wyscout_accuracy, 2),
        "xt_prog_sum": round(xt_prog_sum, 4),
        "xt_prog_mean": round(xt_prog_mean, 4),
        "xt_total_sum": round(xt_total_sum, 4),
        "xt_total_mean": round(xt_total_mean, 4),
        "positive_xt_total": positive_xt_total,
        "positive_xt_sum": round(positive_xt_sum, 4),
        "positive_xt_mean": round(positive_xt_mean, 4),
    }

# Draw functions
FIG_W, FIG_H = 7.9, 5.3
FIG_DPI = 110

def draw_pass_map(df: pd.DataFrame, title: str):
    pitch = Pitch(pitch_type="statsbomb", pitch_color="#1a1a2e", line_color="#ffffff", line_alpha=0.95)
    fig, ax = pitch.draw(figsize=(FIG_W, FIG_H))
    fig.set_facecolor("#1a1a2e")
    fig.set_dpi(FIG_DPI)
    ax.axvline(x=FINAL_THIRD_LINE_X, color="#FFD54F", linewidth=1.0, alpha=0.20)
    ax.axvline(x=HALF_LINE_X, color="#ffffff", linewidth=0.6, alpha=0.12, linestyle="--")
    START_DOT_SIZE = 45
    for _, row in df.iterrows():
        is_lost = not row["is_won"]
        is_sw = bool(row["switch"])
        is_prog_w = bool(row["is_progressive_wyscout"])
        has_vid = has_video_value(row["video"])
        if is_lost:
            color = COLOR_FAIL
            alpha = 0.92
        elif is_sw:
            color = COLOR_SWITCH
            alpha = 0.92
        elif is_prog_w:
            color = COLOR_PROGRESSIVE
            alpha = 0.88
        else:
            color = COLOR_SUCCESS
            alpha = 0.03
        pitch.arrows(row["x_start"], row["y_start"], row["x_end"], row["y_end"],
                     color=color, width=1.55, headwidth=2.25, headlength=2.25,
                     ax=ax, zorder=3, alpha=alpha)
        if has_vid:
            pitch.scatter(row["x_start"], row["y_start"], s=95, marker="o", facecolors="none",
                          edgecolors="#FFD54F", linewidths=2.0, ax=ax, zorder=5)
        pitch.scatter(row["x_start"], row["y_start"], s=START_DOT_SIZE, marker="o", color=color,
                      edgecolors="white", linewidths=0.8, ax=ax, zorder=6, alpha=alpha)
        # Mark the final of the action (end point) with a small x (to show where the action finished)
        pitch.scatter(row["x_end"], row["y_end"], s=30, marker="x", color=color, ax=ax, zorder=5, alpha=0.85)
    ax.set_title(title, fontsize=12, color="#ffffff", pad=8)
    legend_elements = [
        Line2D([0], [0], color="#ffffff", lw=2.5, label="Successful", alpha=0.5),
        Line2D([0], [0], color=COLOR_FAIL, lw=2.5, label="Unsuccessful", alpha=0.9),
        Line2D([0], [0], color=COLOR_PROGRESSIVE, lw=2.5, label="Progressive", alpha=0.9),
        Line2D([0], [0], marker='x', color='w', label='End of action', markerfacecolor='w', markeredgecolor='w', markersize=6),
    ]
    legend = ax.legend(handles=legend_elements, loc="upper left", bbox_to_anchor=(0.01, 0.99),
                       frameon=True, facecolor="#1a1a2e", edgecolor="#444466", shadow=False,
                       fontsize="x-small", labelspacing=0.5, borderpad=0.5)
    for txt in legend.get_texts():
        txt.set_color("white")
    legend.get_frame().set_alpha(0.92)
    arrow = FancyArrowPatch((0.45, 0.05), (0.55, 0.05), transform=fig.transFigure,
                           arrowstyle="-|>", mutation_scale=15, linewidth=2, color="#cccccc")
    fig.patches.append(arrow)
    fig.text(0.5, 0.02, "Attack Direction", ha="center", va="center", fontsize=9, color="#cccccc")
    fig.tight_layout()
    fig.canvas.draw()
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=FIG_DPI, facecolor=fig.get_facecolor())
    buf.seek(0)
    img_obj = Image.open(buf)
    return img_obj, ax, fig

def draw_corridor_heatmap(df: pd.DataFrame, title: str = "Zone Heatmap — Completed Passes"):
    df_success = df[df["is_won"]].copy()
    x_bins = np.linspace(0.0, FIELD_X, 7)
    left_y0, left_y1 = LANE_LEFT_MIN, FIELD_Y
    right_y0, right_y1 = 0.0, LANE_RIGHT_MAX
    center_y0, center_y1 = LANE_RIGHT_MAX, LANE_LEFT_MIN
    corridors = {"left": (left_y0, left_y1), "center": (center_y0, center_y1), "right": (right_y0, right_y1)}
    counts = {}
    for cname, (y0, y1) in corridors.items():
        arr = np.zeros(6, dtype=int)
        for i in range(6):
            x0, x1 = x_bins[i], x_bins[i + 1]
            mask = ((df_success["x_end"] >= x0) & (df_success["x_end"] < x1)
                    & (df_success["y_end"] >= y0) & (df_success["y_end"] < y1))
            arr[i] = int(mask.sum())
        counts[cname] = arr
    all_vals = np.concatenate([counts[c] for c in counts])
    vmax = max(1, int(all_vals.max()))
    pitch = Pitch(pitch_type="statsbomb", pitch_color="#1a1a2e", line_color="#ffffff", line_alpha=0.95)
    fig, ax = pitch.draw(figsize=(FIG_W, FIG_H))
    fig.set_facecolor("#1a1a2e")
    fig.set_dpi(FIG_DPI)
    cmap = LinearSegmentedColormap.from_list("white_red", ["#ffffff", "#ffecec", "#ffbfbf", "#ff8080", "#ff3b3b", "#ff0000"])
    norm = Normalize(vmin=0, vmax=vmax)
    text_light_threshold = max(1, vmax * 0.35)
    for cname, (y0, y1) in corridors.items():
        arr = counts[cname]
        for i in range(6):
            x0, x1 = x_bins[i], x_bins[i + 1]
            value = arr[i]
            color = cmap(norm(value))
            rect = Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor=color, edgecolor=(1.0, 1.0, 1.0, 0.12),
                             linewidth=0.6, alpha=0.95, zorder=2)
            ax.add_patch(rect)
            cx = (x0 + x1) / 2.0
            cy = (y0 + y1) / 2.0
            text_color = "#000000" if value <= text_light_threshold else "#ffffff"
            fontw = "700" if value >= vmax * 0.5 else "600"
            ax.text(cx, cy, str(value), ha="center", va="center", color=text_color, fontsize=11, fontweight=fontw, zorder=4)
    ax.set_title(title, fontsize=12, color="#ffffff", pad=8)
    ax.axhline(y=LANE_LEFT_MIN, color="#ffffff", linewidth=0.5, alpha=0.15, linestyle="--", zorder=3)
    ax.axhline(y=LANE_RIGHT_MAX, color="#ffffff", linewidth=0.5, alpha=0.15, linestyle="--", zorder=3)
    arrow = FancyArrowPatch((0.45, 0.05), (0.55, 0.05), transform=fig.transFigure, arrowstyle="-|>", mutation_scale=15, linewidth=2, color="#cccccc")
    fig.patches.append(arrow)
    fig.text(0.5, 0.02, "Attack Direction", ha="center", va="center", fontsize=9, color="#cccccc")
    fig.tight_layout()
    fig.canvas.draw()
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=FIG_DPI, facecolor=fig.get_facecolor())
    buf.seek(0)
    img_obj = Image.open(buf)
    return img_obj, ax, fig

# ==========================
# Layout: Filters / Field / Stats
# ==========================
st.caption("Click the start dot to select the pass event.")

col_filters, col_field, col_stats = st.columns([0.9, 2, 1], gap="large")

with col_filters:
    st.markdown('<div class="filter-panel">', unsafe_allow_html=True)
    st.markdown("### 🏟️ Match Selection")
    selected_match = st.selectbox("Choose the match", list(full_data.keys()), index=0)
    st.markdown('<hr class="filter-divider">', unsafe_allow_html=True)
    st.markdown("### 🎯 Pass Filter")
    pass_filter = st.radio(
        "Filter passes",
        [
            "All Passes",
            "Successful Only",
            "Unsuccessful Only",
            "Progressive Only (All)",
            "Positive xT Only (Successful)",
        ],
        index=0,
    )
    st.markdown('</div>', unsafe_allow_html=True)

# session state
if "heat_selection" not in st.session_state:
    st.session_state["heat_selection"] = None
if "last_match" not in st.session_state:
    st.session_state["last_match"] = selected_match
if "last_filter" not in st.session_state:
    st.session_state["last_filter"] = pass_filter

# Clear heat selection automatically if match or filter changed
if st.session_state["last_match"] != selected_match:
    st.session_state["heat_selection"] = None
    st.session_state["last_match"] = selected_match
if st.session_state["last_filter"] != pass_filter:
    st.session_state["heat_selection"] = None
    st.session_state["last_filter"] = pass_filter

with col_field:
    df_base = full_data[selected_match].copy()
    if pass_filter == "All Passes":
        df_base = df_base.reset_index(drop=True)
    elif pass_filter == "Successful Only":
        df_base = df_base[df_base["is_won"]].reset_index(drop=True)
    elif pass_filter == "Unsuccessful Only":
        df_base = df_base[~df_base["is_won"]].reset_index(drop=True)
    elif pass_filter == "Progressive Only (All)":
        df_base = df_base[df_base["is_progressive_wyscout"]].reset_index(drop=True)
    elif pass_filter == "Positive xT Only (Successful)":
        df_base = df_base[(df_base["outcome"] == "successful") & (df_base["delta_xt"] > 0)].reset_index(drop=True)

    DISPLAY_WIDTH = 780

    # Reserve placeholder so Pass Map appears visually above heatmap
    pass_map_placeholder = st.empty()

    # ---- Heatmap (render & handle click first) ----
    st.markdown('<h4 style="color:#ffffff; margin:6px 0 6px 0;">Zone Heatmap</h4>', unsafe_allow_html=True)
    heat_img, hax, hfig = draw_corridor_heatmap(df_base)
    heat_click = streamlit_image_coordinates(heat_img, width=DISPLAY_WIDTH)

    if heat_click is not None:
        real_w, real_h = heat_img.size
        disp_w = heat_click["width"]
        disp_h = heat_click["height"]
        pixel_x = heat_click["x"] * (real_w / disp_w)
        pixel_y = heat_click["y"] * (real_h / disp_h)
        mpl_pixel_y = real_h - pixel_y
        field_x, field_y = hax.transData.inverted().transform((pixel_x, mpl_pixel_y))
        x_bins = np.linspace(0.0, FIELD_X, 7)
        ix = np.searchsorted(x_bins, field_x, side="right") - 1
        ix = max(0, min(5, ix))
        x0, x1 = x_bins[ix], x_bins[ix + 1]
        if field_y >= LANE_LEFT_MIN:
            cname = "left"; y0, y1 = LANE_LEFT_MIN, FIELD_Y
        elif field_y < LANE_RIGHT_MAX:
            cname = "right"; y0, y1 = 0.0, LANE_RIGHT_MAX
        else:
            cname = "center"; y0, y1 = LANE_RIGHT_MAX, LANE_LEFT_MIN
        st.session_state["heat_selection"] = {"ix": int(ix), "corridor": cname, "x0": float(x0), "x1": float(x1), "y0": float(y0), "y1": float(y1)}
    plt.close(hfig)  # free memory

    # ---- Render Pass Map placeholder content: title + clear button (so clear can affect selection before drawing map) ----
    with pass_map_placeholder.container():
        st.markdown('<h4 style="color:#ffffff; margin:0 0 6px 0;">Pass Map</h4>', unsafe_allow_html=True)
        # Clear button under title (clears selection in the same run)
        if st.button("Limpar filtro do quadrante", key="clear_heat_filter"):
            st.session_state["heat_selection"] = None

        # Now compute df_to_draw AFTER handling clear button (so button takes immediate effect)
        df_to_draw = df_base
        if st.session_state["heat_selection"] is not None:
            sel = st.session_state["heat_selection"]
            df_to_draw = df_base[
                (df_base["x_end"] >= sel["x0"])
                & (df_base["x_end"] < sel["x1"])
                & (df_base["y_end"] >= sel["y0"])
                & (df_base["y_end"] < sel["y1"])
            ].reset_index(drop=True)

        # Draw pass map with current selection and render it interactively
        img_obj, ax, fig = draw_pass_map(df_to_draw, title=f"Pass Map — {selected_match}")
        click = streamlit_image_coordinates(img_obj, width=DISPLAY_WIDTH)

    selected_pass = None
    # Process click on pass map (select pass)
    if click is not None:
        real_w, real_h = img_obj.size
        disp_w = click["width"]
        disp_h = click["height"]
        pixel_x = click["x"] * (real_w / disp_w)
        pixel_y = click["y"] * (real_h / disp_h)
        mpl_pixel_y = real_h - pixel_y
        field_x, field_y = ax.transData.inverted().transform((pixel_x, mpl_pixel_y))
        df_sel = df_to_draw.copy()
        df_sel["dist"] = np.sqrt((df_sel["x_start"] - field_x) ** 2 + (df_sel["y_start"] - field_y) ** 2)
        RADIUS = 5.0
        candidates = df_sel[df_sel["dist"] < RADIUS].copy()
        if not candidates.empty:
            candidates = candidates.sort_values(by="dist", ascending=True)
            selected_pass = candidates.iloc[0]
    plt.close(fig)

    # Show info about current heatmap selection and counts
    if st.session_state["heat_selection"] is not None:
        sel = st.session_state["heat_selection"]
        sel_mask = (
            (df_base["x_end"] >= sel["x0"])
            & (df_base["x_end"] < sel["x1"])
            & (df_base["y_end"] >= sel["y0"])
            & (df_base["y_end"] < sel["y1"])
        )
        sel_count = int(sel_mask.sum())
        st.markdown(
            f"<div style='color:#ffffff; margin-top:6px;'>"
            f"<strong>Filtro aplicado:</strong> corredor <code>{sel['corridor']}</code>, coluna X #{sel['ix']+1} — {sel_count} passes</div>",
            unsafe_allow_html=True,
        )

    # ---- Selected Event and data table ----
    st.divider()
    st.subheader("Selected Event")
    if selected_pass is None:
        st.info("Click the start dot to inspect the pass details.")
    else:
        st.success(f"Selected pass: #{int(selected_pass['number'])} ({selected_pass['type']})")
        det1, det2 = st.columns(2)
        with det1:
            st.write(f"**Start:** ({selected_pass['x_start']:.2f}, {selected_pass['y_start']:.2f})")
        with det2:
            st.write(f"**End:** ({selected_pass['x_end']:.2f}, {selected_pass['y_end']:.2f})")
        dir_emoji = {"forward": "⬆️", "backward": "⬇️", "lateral": "↔️"}
        direction_label = selected_pass["direction"].capitalize()
        emoji = dir_emoji.get(selected_pass["direction"], "")
        tag1, tag2, tag3 = st.columns(3)
        tag1.write(f"**Direction:** {emoji} {direction_label}")
        tag2.write(f"**Progressive (Wyscout):** {'✅' if selected_pass['is_progressive_wyscout'] else '❌'}")
        tag3.write(f"**Switch:** {'✅' if selected_pass['switch'] else '❌'}")
        xt_col1, xt_col2, xt_col3, xt_col4 = st.columns(4)
        xt_col1.metric("Distance", f"{selected_pass['pass_distance']:.1f}m")
        xt_col2.metric("xT Start", f"{selected_pass['xt_start']:.4f}")
        xt_col3.metric("xT End", f"{selected_pass['xt_end']:.4f}")
        xt_col4.metric("ΔxT", f"{selected_pass['delta_xt']:.4f}", delta=f"{selected_pass['delta_xt']:.4f}" if selected_pass["delta_xt"] != 0 else None)
        if has_video_value(selected_pass["video"]):
            try:
                st.video(selected_pass["video"])
            except Exception:
                st.error(f"Video file not found: {selected_pass['video']}")
        else:
            st.warning("No video is attached to this event.")

    with st.expander("📊 Full Pass Data Table"):
        display_cols = [
            "number", "type", "outcome", "direction",
            "x_start", "y_start", "x_end", "y_end",
            "pass_distance",
            "is_forward", "is_backward", "is_lateral",
            "is_progressive_wyscout",
            "switch", "xt_start", "xt_end", "delta_xt",
        ]
        st.dataframe(
            df_to_draw[display_cols].style.format({
                "x_start": "{:.2f}", "y_start": "{:.2f}",
                "x_end": "{:.2f}", "y_end": "{:.2f}",
                "pass_distance": "{:.1f}",
                "xt_start": "{:.4f}", "xt_end": "{:.4f}",
                "delta_xt": "{:.4f}",
            }),
            use_container_width=True,
            height=400,
        )

# RIGHT: Statistics (based on df_to_draw)
with col_stats:
    stats_safe = compute_stats(df_to_draw)
    with st.expander("General Statistics", expanded=False):
        st.markdown('<div class="stats-section-title">Overview</div>', unsafe_allow_html=True)
        row1, row2, row3 = st.columns(3)
        with row1:
            small_metric("Total Passes", f"{stats_safe['total_passes']}")
        with row2:
            small_metric("Successful", f"{stats_safe['successful_passes']}")
        with row3:
            small_metric("Accuracy", f"{stats_safe['accuracy_pct']:.1f}%")
        st.markdown("<hr style='margin:6px 0 8px 0;'>", unsafe_allow_html=True)
        st.markdown('<div class="stats-section-title">Pass Direction</div>', unsafe_allow_html=True)
        dir1, dir2, dir3 = st.columns(3)
        with dir1:
            small_metric("⬆️ Forward", f"{stats_safe['forward_total']} ({stats_safe['pct_forward']:.0f}%)")
        with dir2:
            small_metric("⬇️ Backward", f"{stats_safe['backward_total']} ({stats_safe['pct_backward']:.0f}%)")
        with dir3:
            small_metric("↔️ Lateral", f"{stats_safe['lateral_total']} ({stats_safe['pct_lateral']:.0f}%)")
    with st.expander("Advanced Statistics", expanded=False):
        st.markdown('<div class="stats-section-title">Progressive Passes (Wyscout)</div>', unsafe_allow_html=True)
        pw1, pw2, pw3, pw4 = st.columns(4)
        with pw1:
            small_metric("Total", f"{stats_safe['prog_wyscout_total']}")
        with pw2:
            small_metric("Successful", f"{stats_safe['prog_wyscout_success']}")
        with pw3:
            small_metric("Accuracy", f"{stats_safe['prog_wyscout_accuracy_pct']:.1f}%")
        with pw4:
            small_metric("% of Total", f"{stats_safe['pct_progressive_wyscout']:.1f}%")
        st.markdown("<hr style='margin:6px 0 8px 0;'>", unsafe_allow_html=True)
        st.markdown('<div class="stats-section-title">Expected Threat (xT)</div>', unsafe_allow_html=True)
        xt1, xt2 = st.columns(2)
        with xt1:
            small_metric("xT Σ (Progressive)", f"{stats_safe['xt_prog_sum']:.2f}")
        with xt2:
            small_metric("xT Mean (Progressive)", f"{stats_safe['xt_prog_mean']:.2f}")
        xt3, xt4 = st.columns(2)
        with xt3:
            small_metric("xT Σ (Positive ΔxT)", f"{stats_safe['positive_xt_sum']:.2f}")
        with xt4:
            small_metric("xT Mean (Positive ΔxT)", f"{stats_safe['positive_xt_mean']:.2f}")
    st.divider()
    st.caption("Notas: 'Progressive' segue a definição Wyscout; ΔxT só é contabilizado para passes bem-sucedidos.")
