"""
app.py - Phone-friendly web version of the bar tracker, built with Streamlit.

It uses the analysis code in track_bar.py, so keep both files in the same folder.
Run it on your computer with:   streamlit run app.py
"""

import io
import tempfile
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import track_bar as tb

st.set_page_config(page_title="Bar Tracker", page_icon="🏋️", layout="centered")


# ---------------- helpers ----------------

def read_first_frame(path):
    cap = cv2.VideoCapture(path)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    frame, _ = tb.resize_to(frame, tb.PROCESS_MAX_SIDE)
    return frame


def guess_plate(frame):
    """Starting guess for the plate: the largest clear circle in the first frame."""
    h, w = frame.shape[:2]
    short = min(h, w)
    gray = cv2.medianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 5)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.5, minDist=short / 4,
                               param1=120, param2=60,
                               minRadius=int(short * 0.05), maxRadius=int(short * 0.35))
    if circles is not None:
        x, y, r = max(circles[0], key=lambda c: c[2])
        return int(x), int(y), int(r)
    return w // 2, h // 2, int(short * 0.12)


def draw_circle_preview(frame, cx, cy, r):
    """Full frame with the circle drawn, plus a zoomed-in view of the plate."""
    img = frame.copy()
    thick = max(2, frame.shape[1] // 300)
    cv2.circle(img, (cx, cy), r, (0, 255, 0), thick)
    cv2.drawMarker(img, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, max(10, r // 4), thick)
    h, w = frame.shape[:2]
    pad = int(r * 1.6)
    x0, x1 = max(0, cx - pad), min(w, cx + pad)
    y0, y1 = max(0, cy - pad), min(h, cy + pad)
    zoom = img[y0:y1, x0:x1]
    return img[..., ::-1], (zoom[..., ::-1] if zoom.size else None)


def bar_path_image(frame, centers, reps):
    """First frame with the tracked bar path drawn on it (reps in color)."""
    img = frame.copy()
    xs = tb.fill_gaps(centers[:, 0])
    ys = tb.fill_gaps(centers[:, 1])
    pts = np.stack([xs, ys], axis=1).astype(np.int32)
    thick = max(2, frame.shape[1] // 350)
    cv2.polylines(img, [pts], False, (200, 200, 200), thick)
    cmap = plt.cm.viridis(np.linspace(0, 0.9, max(1, len(reps))))
    for k, (s, e) in enumerate(reps):
        color = tuple(int(255 * c) for c in cmap[k][2::-1])  # RGB -> BGR
        cv2.polylines(img, [pts[s:e + 1]], False, color, thick + 2)
    return img[..., ::-1]


def fig_to_png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    return buf.getvalue()


# ---------------- page ----------------

st.title("🏋️ Bar Tracker")
st.caption("Film from the side with the phone steady and square to the plate. "
           "1080p at 60 fps works best.")

uploaded = st.file_uploader("Choose or record a lifting video",
                            type=["mp4", "mov", "m4v", "avi"])
lift = st.radio("Lift", list(tb.MVT_BY_EXERCISE), horizontal=True)
mvt = st.number_input(
    "Failure speed (m/s)", min_value=0.05, max_value=1.0, step=0.01,
    value=tb.MVT_BY_EXERCISE[lift], key=f"mvt_{lift}",
    help="Mean speed of a true last rep before failure. Lower it if this lifter "
         "can finish reps slower than this.")

if uploaded is None:
    st.info("Upload a video to get started.")
    st.stop()

# Save each new upload to a temporary file (OpenCV needs a file path)
if st.session_state.get("file_id") != uploaded.file_id:
    old = st.session_state.get("path")
    if old:
        Path(old).unlink(missing_ok=True)
    suffix = Path(uploaded.name).suffix or ".mp4"
    tmp = Path(tempfile.gettempdir()) / f"bartracker_{uploaded.file_id}{suffix}"
    tmp.write_bytes(uploaded.getvalue())
    frame = read_first_frame(str(tmp))
    st.session_state.update(file_id=uploaded.file_id, path=str(tmp), frame=frame,
                            guess=guess_plate(frame) if frame is not None else None,
                            results=None)

frame = st.session_state["frame"]
if frame is None:
    st.error("Couldn't read this video. Try exporting it as a standard .mp4 or .mov.")
    st.stop()

# ---- Step 1: line up the circle with the plate ----
st.subheader("1. Line up the circle with the plate")
st.caption("Adjust the sliders until the green circle traces the outer edge of the biggest "
           "plate. The plate's size is used as a ruler, so get the edge as close as you can.")
h, w = frame.shape[:2]
gx, gy, gr = st.session_state["guess"]
fid = st.session_state["file_id"]
cx = st.slider("Left ↔ right", 0, w, gx, key=f"cx_{fid}")
cy = st.slider("Up ↕ down", 0, h, gy, key=f"cy_{fid}")
r = st.slider("Plate size", 10, min(h, w) // 2, min(gr, min(h, w) // 2), key=f"r_{fid}")

full_img, zoom_img = draw_circle_preview(frame, cx, cy, r)
if zoom_img is not None:
    st.image(zoom_img, caption="Zoomed in on the plate", width="stretch")
with st.expander("Show full frame"):
    st.image(full_img, width="stretch")

if cx - r < 0 or cy - r < 0 or cx + r > w or cy + r > h:
    st.warning("The circle runs off the edge of the frame. Results will be less accurate; "
               "next time, leave more space around the plate when filming.")
x0, y0 = max(0, cx - r), max(0, cy - r)
roi = (x0, y0, min(w, cx + r) - x0, min(h, cy + r) - y0)

# ---- Step 2: analyze ----
st.subheader("2. Analyze")
if st.button("Analyze lift", type="primary", width="stretch"):
    bar = st.progress(0.0, text="Tracking the plate…")
    try:
        t, centers, fps, roi_used = tb.track_video(
            Path(st.session_state["path"]), roi=roi, show=False,
            progress=lambda f: bar.progress(f, text=f"Tracking the plate… {int(100 * f)}%"))
    except SystemExit as err:
        bar.empty()
        st.error(str(err))
        st.stop()
    bar.progress(1.0, text="Crunching the numbers…")
    x, y, v, reps, m_per_px = tb.analyze(t, centers, fps, roi_used)
    rows = tb.summarize(t, x, y, v, reps)
    st.session_state["results"] = dict(t=t, centers=centers, fps=fps, x=x, y=y, v=v,
                                       reps=reps, rows=rows, m_per_px=m_per_px, roi=roi)
    bar.empty()

res = st.session_state.get("results")
if not res:
    st.stop()
if res["roi"] != roi:
    st.info("You moved the circle since the last analysis. Tap **Analyze lift** to update.")

# ---- Step 3: results ----
st.subheader("3. Results")
rows = res["rows"]
lost = int(np.isnan(res["centers"][:, 0]).sum())
if lost:
    st.warning(f"The tracker lost the plate in {lost} of {len(res['centers'])} frames. "
               "Check the bar path picture below to make sure it stayed on the plate.")
if not rows:
    st.warning("No reps found. Check that the circle is on the plate, or that the video "
               "shows full reps.")
    st.stop()

est = tb.estimate_max_reps(rows, mvt)
c1, c2, c3 = st.columns(3)
c1.metric("Reps done", len(rows))
c2.metric("Est. max reps", est.get("max_reps", "n/a"))
c3.metric("In reserve", est.get("in_reserve", "n/a"))
if "reason" in est:
    st.caption(est["reason"])
elif est["low_confidence"]:
    st.caption("Low confidence: " + "; ".join(est["notes"]) + ".")

trend = tb.velocity_trend(rows)
if trend:
    slope, intercept, r2 = trend
    st.caption(f"Velocity trend: {slope:+.3f} m/s per rep (R² = {r2:.2f}). "
               f"{lift} failure speed: {mvt:.2f} m/s.")

table = pd.DataFrame(rows).rename(columns={
    "rep": "Rep", "rom_cm": "ROM (cm)", "mean_vel": "Mean (m/s)", "peak_vel": "Peak (m/s)",
    "duration_s": "Time (s)", "horiz_range_cm": "Drift (cm)", "vel_loss_pct": "Loss (%)"})
st.dataframe(table.round(2), hide_index=True, width="stretch")

with plt.rc_context({"font.size": 11}):
    fig = tb.plot(res["t"], res["x"], res["y"], res["v"], res["reps"], rows,
                  f"{uploaded.name}  ({lift})", None, est, lift, mvt, tall=True)
st.pyplot(fig, width="stretch")

st.image(bar_path_image(frame, res["centers"], res["reps"]),
         caption="Tracked bar path (reps in color)", width="stretch")

# ---- downloads ----
stem = Path(uploaded.name).stem
data = pd.DataFrame({"time_s": res["t"], "x_m": res["x"], "y_m": res["y"], "vy_m_s": res["v"]})
st.download_button("Download chart (PNG)", fig_to_png(fig), f"{stem}_chart.png",
                   "image/png", width="stretch")
st.download_button("Download rep results (CSV)", table.to_csv(index=False), f"{stem}_reps.csv",
                   "text/csv", width="stretch")
st.download_button("Download frame-by-frame data (CSV)", data.round(4).to_csv(index=False),
                   f"{stem}_data.csv", "text/csv", width="stretch")
plt.close(fig)
