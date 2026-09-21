"""
track_bar.py - Barbell path and velocity tracker (no marker needed)

How to use:
  1. Run this script (play button in VS Code). A file picker opens; choose your video.
     Then click which lift it is (this sets the failure speed for max-rep estimates).
  2. The first frame appears. Drag a box tightly around the WHOLE plate
     (edge to edge), then press ENTER or SPACE. Press C to cancel.
  3. Watch it track. Press Q or ESC to stop early.
  4. You get a per-rep table in the terminal, a chart, and output files
     saved next to your video.

Filming tips: side view, phone level and square to the plate, 3-4 m away,
60 fps (not slo-mo), nothing passing in front of the plate.
"""

import csv
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from scipy.signal import savgol_filter

# ---------------- Settings you can change ----------------
PLATE_DIAMETER_M = 0.450   # standard 20 kg / 45 lb plates are ~450 mm
PROCESS_MAX_SIDE = 1280    # downscale big videos to this size for speed
DISPLAY_MAX_SIDE = 900     # size of the preview window on screen
FPS_OVERRIDE = None        # set e.g. 60 if the detected frame rate looks wrong
SMOOTH_WINDOW_S = 0.15     # smoothing window for velocity (seconds)
V_THRESHOLD = 0.05         # m/s, bar counts as "moving up" above this
EDGE_V = 0.02              # m/s, where a rep is considered to start and end
MIN_ROM_M = 0.15           # ignore upward moves shorter than this (unracking, etc.)
MERGE_GAP_S = 0.20         # join an ascent split by a brief sticking point
SAVE_OVERLAY_VIDEO = True  # save a copy of the video with the bar path drawn on
EXERCISE = None            # None = ask each run, or "Squat", "Bench press", "Deadlift"
MVT_OVERRIDE = None        # your own minimum velocity threshold (m/s), if you know it
# Minimum velocity threshold (MVT): typical mean speed of a true last rep before
# failure. These are averages from VBT research; individual lifters vary.
MVT_BY_EXERCISE = {"Squat": 0.21, "Bench press": 0.17, "Deadlift": 0.15}
# -----------------------------------------------------------


def pick_video_file():
    """Use a command-line argument if given, otherwise open a file picker."""
    if len(sys.argv) > 1:
        return Path(sys.argv[1])
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.askopenfilename(
        title="Choose a lifting video",
        filetypes=[("Video files", "*.mp4 *.mov *.avi *.mkv *.m4v"), ("All files", "*.*")],
    )
    root.destroy()
    if not path:
        sys.exit("No video selected.")
    return Path(path)


def choose_exercise():
    """Ask which lift is in the video (sets the failure speed used for max-rep estimates)."""
    if EXERCISE:
        return EXERCISE
    import tkinter as tk
    choice = {"name": "Squat"}
    root = tk.Tk()
    root.title("Which lift?")
    root.attributes("-topmost", True)
    tk.Label(root, text="Which lift is in this video?", font=("Segoe UI", 11)).pack(padx=30, pady=(15, 10))

    def pick(name):
        choice["name"] = name
        root.destroy()

    for name in MVT_BY_EXERCISE:
        tk.Button(root, text=name, width=18, command=lambda n=name: pick(n)).pack(pady=3)
    tk.Label(root, text="(Closing this window picks Squat)", fg="gray").pack(pady=(8, 12))
    root.mainloop()
    return choice["name"]


def make_tracker():
    """Create a CSRT tracker, handling the different names across OpenCV versions."""
    if hasattr(cv2, "TrackerCSRT") and hasattr(cv2.TrackerCSRT, "create"):
        return cv2.TrackerCSRT.create()
    if hasattr(cv2, "TrackerCSRT_create"):
        return cv2.TrackerCSRT_create()
    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
        return cv2.legacy.TrackerCSRT_create()
    sys.exit("CSRT tracker not found. Run: pip install opencv-contrib-python")


def resize_to(frame, max_side):
    """Return (resized_frame, scale) so the longer side is at most max_side."""
    h, w = frame.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return frame, scale


def select_plate(frame):
    """Let the user drag a box around the plate. Returns (x, y, w, h) in frame coordinates."""
    disp, s = resize_to(frame, DISPLAY_MAX_SIDE)
    disp = disp.copy()
    msg = "Drag a box around the WHOLE plate, then ENTER"
    cv2.putText(disp, msg, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
    cv2.putText(disp, msg, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
    x, y, w, h = cv2.selectROI("Select plate", disp, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow("Select plate")
    if w == 0 or h == 0:
        sys.exit("No box drawn. Run the script again and drag a box around the plate.")
    return tuple(int(round(v / s)) for v in (x, y, w, h))


def track_video(video_path, roi=None, show=True, overlay_path=None):
    """Track the plate through the video. Returns frame times, plate centers (px), fps, box."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        sys.exit(f"Could not open video: {video_path}")
    fps = FPS_OVERRIDE or cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps < 1:
        sys.exit("Could not read the frame rate. Set FPS_OVERRIDE at the top of the script.")

    ok, frame = cap.read()
    if not ok:
        sys.exit("Could not read the first frame.")
    frame, _ = resize_to(frame, PROCESS_MAX_SIDE)

    if roi is None:
        roi = select_plate(frame)
    tracker = make_tracker()
    tracker.init(frame, roi)

    writer = None
    if overlay_path is not None:
        h, w = frame.shape[:2]
        writer = cv2.VideoWriter(str(overlay_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    centers = [(roi[0] + roi[2] / 2, roi[1] + roi[3] / 2)]
    trail = [tuple(int(v) for v in centers[0])]
    lost = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame, _ = resize_to(frame, PROCESS_MAX_SIDE)
        found, box = tracker.update(frame)
        if found:
            bx, by, bw, bh = box
            c = (bx + bw / 2, by + bh / 2)
            centers.append(c)
            trail.append((int(c[0]), int(c[1])))
            cv2.rectangle(frame, (int(bx), int(by)), (int(bx + bw), int(by + bh)), (0, 255, 0), 2)
        else:
            centers.append((np.nan, np.nan))
            lost += 1
            cv2.putText(frame, "LOST", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        if len(trail) > 1:
            cv2.polylines(frame, [np.array(trail, np.int32)], False, (0, 255, 255), 2)
        if writer is not None:
            writer.write(frame)
        if show:
            disp, _ = resize_to(frame, DISPLAY_MAX_SIDE)
            cv2.imshow("Tracking (Q to stop)", disp)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break

    cap.release()
    if writer is not None:
        writer.release()
    if show:
        cv2.destroyAllWindows()

    centers = np.array(centers, dtype=float)
    t = np.arange(len(centers)) / fps
    if lost:
        print(f"Note: tracker lost the plate in {lost} of {len(centers)} frames "
              f"({100 * lost / len(centers):.0f}%). Gaps were filled by interpolation.")
    return t, centers, fps, roi


def fill_gaps(a):
    """Linearly interpolate over NaN values."""
    idx = np.arange(len(a))
    good = ~np.isnan(a)
    return np.interp(idx, idx[good], a[good])


def find_reps(y, v, fps):
    """Find concentric (upward) phases. Returns list of (start_idx, end_idx)."""
    moving = v > V_THRESHOLD
    segs, i, n = [], 0, len(v)
    while i < n:
        if moving[i]:
            j = i
            while j + 1 < n and moving[j + 1]:
                j += 1
            segs.append([i, j])
            i = j + 1
        else:
            i += 1

    # Merge ascents split by a short sticking point where the bar barely dips
    merged = []
    for s in segs:
        if merged:
            ps, pe = merged[-1]
            gap_ok = (s[0] - pe) <= MERGE_GAP_S * fps
            dip = y[pe] - np.min(y[pe:s[0] + 1])
            if gap_ok and dip < 0.03:
                merged[-1][1] = s[1]
                continue
        merged.append(s)

    # Extend each ascent out to where the bar is essentially stopped
    reps = []
    for s, e in merged:
        while s > 0 and v[s - 1] > EDGE_V:
            s -= 1
        while e < n - 1 and v[e + 1] > EDGE_V:
            e += 1
        if y[e] - y[s] >= MIN_ROM_M:
            reps.append((s, e))
    return reps


def analyze(t, centers, fps, roi):
    """Convert to meters, smooth, compute velocity, and find reps."""
    m_per_px = PLATE_DIAMETER_M / ((roi[2] + roi[3]) / 2)
    aspect = roi[2] / roi[3]
    if abs(aspect - 1) > 0.15:
        print(f"Warning: your box is {aspect:.2f}x wider than tall. Either the box isn't tight "
              "around the plate or the camera isn't square to it, so distances may be off.")

    x_px = fill_gaps(centers[:, 0])
    y_px = fill_gaps(centers[:, 1])
    x = (x_px - x_px[0]) * m_per_px
    y = -(y_px - y_px[0]) * m_per_px  # image y points down; flip so up is positive

    win = max(5, int(round(SMOOTH_WINDOW_S * fps)))
    if win % 2 == 0:
        win += 1  # filter window must be odd
    if win > len(y):
        win = len(y) if len(y) % 2 else len(y) - 1
    y_s = savgol_filter(y, win, 2)
    x_s = savgol_filter(x, win, 2)
    vy = savgol_filter(y, win, 2, deriv=1, delta=1 / fps)

    reps = find_reps(y_s, vy, fps)
    return x_s, y_s, vy, reps, m_per_px


def summarize(t, x, y, v, reps):
    rows = []
    for k, (s, e) in enumerate(reps, 1):
        rows.append({
            "rep": k,
            "rom_cm": 100 * (y[e] - y[s]),
            "mean_vel": float(np.mean(v[s:e + 1])),
            "peak_vel": float(np.max(v[s:e + 1])),
            "duration_s": t[e] - t[s],
            "horiz_range_cm": 100 * (np.max(x[s:e + 1]) - np.min(x[s:e + 1])),
        })
    if rows:
        best = max(r["mean_vel"] for r in rows)
        for r in rows:
            r["vel_loss_pct"] = 100 * (best - r["mean_vel"]) / best
    return rows


def velocity_trend(rows):
    """Fit a straight line of mean velocity vs rep number.
    Returns (slope, intercept, r_squared), or None if there are fewer than 2 reps."""
    if len(rows) < 2:
        return None
    reps = np.array([r["rep"] for r in rows], dtype=float)
    mcv = np.array([r["mean_vel"] for r in rows])
    slope, intercept = np.polyfit(reps, mcv, 1)
    pred = slope * reps + intercept
    ss_res = np.sum((mcv - pred) ** 2)
    ss_tot = np.sum((mcv - mcv.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return slope, intercept, r2


def estimate_max_reps(rows, mvt):
    """Extend the velocity trend line down to the minimum velocity threshold (MVT).
    The rep number where the line hits the MVT is the estimated last possible rep."""
    n_done = len(rows)
    if n_done < 3:
        return {"reason": "Need at least 3 reps to estimate max reps."}
    slope, intercept, r2 = velocity_trend(rows)
    if slope >= 0:
        return {"reason": "Bar speed didn't drop across the set, so failure can't be projected."}
    x_fail = (mvt - intercept) / slope          # rep number where predicted speed = MVT
    if x_fail > 30:
        return {"reason": "Bar speed barely dropped, so the set was far from failure "
                          "and an estimate would be unreliable."}
    max_reps = max(n_done, int(np.floor(x_fail)))
    notes = []
    if x_fail > 2 * n_done:
        notes.append("the set stopped far from failure, so this is a long extrapolation")
    if r2 < 0.5:
        notes.append("rep speeds were inconsistent (low R\u00b2)")
    return {"x_fail": x_fail, "max_reps": max_reps, "in_reserve": max_reps - n_done,
            "low_confidence": bool(notes), "notes": notes}


def print_estimate(est, exercise, mvt):
    print(f"\nLift: {exercise}   Failure speed (MVT): {mvt:.2f} m/s")
    if "reason" in est:
        print(f"Max reps estimate: not available. {est['reason']}")
        return
    print(f"Estimated max reps at this weight: {est['max_reps']} "
          f"(about {est['in_reserve']} left in the tank)")
    if est["low_confidence"]:
        print("Low confidence: " + "; ".join(est["notes"]) + ".")


def print_table(rows, fps, m_per_px):
    print(f"\nFrame rate: {fps:.1f} fps   Scale: {1000 * m_per_px:.2f} mm per pixel")
    if not rows:
        print("No reps found. Check that the plate was tracked, or lower MIN_ROM_M.")
        return
    print(f"\n{'Rep':>3} {'ROM cm':>7} {'Mean m/s':>9} {'Peak m/s':>9} {'Time s':>7} "
          f"{'Drift cm':>9} {'Loss %':>7}")
    for r in rows:
        print(f"{r['rep']:>3} {r['rom_cm']:>7.1f} {r['mean_vel']:>9.2f} {r['peak_vel']:>9.2f} "
              f"{r['duration_s']:>7.2f} {r['horiz_range_cm']:>9.1f} {r['vel_loss_pct']:>7.1f}")
    print("\nMean = mean concentric velocity. Drift = horizontal bar travel during the rep.")
    print("Loss = how much slower than your fastest rep.")
    trend = velocity_trend(rows)
    if trend:
        slope, intercept, r2 = trend
        print(f"\nVelocity trend: {slope:+.3f} m/s per rep (R\u00b2 = {r2:.2f})")
        if len(rows) == 2:
            print("(With only 2 reps the line fits perfectly, so R\u00b2 isn't meaningful.)")


def save_outputs(video_path, t, x, y, v, rows):
    base = video_path.with_suffix("")
    with open(f"{base}_data.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_s", "x_m", "y_m", "vy_m_s"])
        for row in zip(t, x, y, v):
            w.writerow([f"{val:.4f}" for val in row])
    if rows:
        with open(f"{base}_reps.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow({k: (round(val, 3) if isinstance(val, float) else val) for k, val in r.items()})
    return base


def plot(t, x, y, v, reps, rows, title, save_path, est, exercise, mvt):
    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 2])
    ax1 = fig.add_subplot(gs[:, 0])   # bar path, full height on the left
    ax2 = fig.add_subplot(gs[0, 1])   # velocity vs time, top right
    ax3 = fig.add_subplot(gs[1, 1])   # mean velocity vs rep, bottom right
    colors = plt.cm.viridis(np.linspace(0, 0.9, max(1, len(reps))))

    ax1.plot(100 * x, 100 * y, color="lightgray", lw=1, label="full path")
    for k, (s, e) in enumerate(reps):
        ax1.plot(100 * x[s:e + 1], 100 * y[s:e + 1], color=colors[k], lw=2, label=f"rep {k + 1}")
    ax1.set_aspect("equal", adjustable="datalim")
    ax1.set_xlabel("Horizontal (cm)")
    ax1.set_ylabel("Vertical (cm)")
    ax1.set_title("Bar path (concentric highlighted)")
    ax1.grid(alpha=0.3)
    if reps:
        ax1.legend(fontsize=8)

    ax2.plot(t, v, color="black", lw=1)
    ax2.axhline(0, color="gray", lw=0.8)
    for k, ((s, e), r) in enumerate(zip(reps, rows)):
        ax2.axvspan(t[s], t[e], color=colors[k], alpha=0.25)
        ax2.text((t[s] + t[e]) / 2, r["peak_vel"] + 0.05, f"{r['mean_vel']:.2f}",
                 ha="center", fontsize=9)
    if rows:
        ax2.set_ylim(top=max(r["peak_vel"] for r in rows) + 0.15)  # room for labels
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Vertical velocity (m/s)")
    ax2.set_title("Velocity (numbers = mean concentric velocity)")
    ax2.grid(alpha=0.3)

    # Mean concentric velocity vs rep number, regression line, and max-rep estimate
    ax3_title = "Mean velocity by rep"
    if rows:
        rep_nums = np.array([r["rep"] for r in rows])
        mcv = np.array([r["mean_vel"] for r in rows])
        n_done = len(rows)
        x_fail = est.get("x_fail")
        # How far right to draw: out to the failure point, but not absurdly far
        x_end = n_done + 0.3
        if x_fail is not None:
            x_end = max(x_end, min(x_fail + 0.5, max(3 * n_done, n_done + 5)))

        ax3.scatter(rep_nums, mcv, c=colors[:n_done], s=70, zorder=3, edgecolors="black")
        for n, val in zip(rep_nums, mcv):
            ax3.annotate(f"{val:.2f}", (n, val), textcoords="offset points", xytext=(0, 9),
                         ha="center", fontsize=9)
        trend = velocity_trend(rows)
        if trend:
            slope, intercept, r2 = trend
            xs = np.array([rep_nums.min() - 0.3, rep_nums.max() + 0.3])
            ax3.plot(xs, slope * xs + intercept, "--", color="crimson", lw=1.5,
                     label=f"v = {slope:+.3f}\u00b7rep + {intercept:.3f}   (R\u00b2 = {r2:.2f})")
            if x_fail is not None:
                xs_ext = np.array([xs[1], x_end])
                ax3.plot(xs_ext, slope * xs_ext + intercept, ":", color="crimson", lw=1.5,
                         label="trend extended")
                if x_fail <= x_end:
                    ax3.plot(x_fail, mvt, "X", color="crimson", ms=11, zorder=4)
        ax3.axhline(mvt, color="dimgray", ls="-.", lw=1,
                    label=f"{exercise} failure speed \u2248 {mvt:.2f} m/s")
        ax3.legend(fontsize=8, loc="upper right")

        if "reason" in est:
            ax3_title += "  |  max reps: n/a"
        else:
            ax3_title += (f"  |  estimated max: {est['max_reps']} reps "
                          f"({est['in_reserve']} in reserve)")
            if est["low_confidence"]:
                ax3_title += "  [low confidence]"

        ax3.set_xlim(0.5, x_end + 0.3)
        ax3.xaxis.set_major_locator(MaxNLocator(integer=True))
        lo = min(mvt, mcv.min())
        pad = max(0.05, 0.15 * (mcv.max() - lo))
        ax3.set_ylim(lo - pad, mcv.max() + 3 * pad)
    else:
        ax3.text(0.5, 0.5, "No reps detected", ha="center", va="center", transform=ax3.transAxes)
    ax3.set_xlabel("Rep")
    ax3.set_ylabel("Mean concentric velocity (m/s)")
    ax3.set_title(ax3_title)
    ax3.grid(alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    return fig


def main():
    video_path = pick_video_file()
    exercise = choose_exercise()
    mvt = MVT_OVERRIDE or MVT_BY_EXERCISE.get(exercise, 0.30)
    print(f"Video: {video_path.name}   Lift: {exercise}")
    overlay = video_path.with_name(video_path.stem + "_overlay.mp4") if SAVE_OVERLAY_VIDEO else None
    t, centers, fps, roi = track_video(video_path, overlay_path=overlay)
    x, y, v, reps, m_per_px = analyze(t, centers, fps, roi)
    rows = summarize(t, x, y, v, reps)
    print_table(rows, fps, m_per_px)
    est = estimate_max_reps(rows, mvt)
    if rows:
        print_estimate(est, exercise, mvt)
    base = save_outputs(video_path, t, x, y, v, rows)
    plot(t, x, y, v, reps, rows, f"{video_path.name}  ({exercise})", f"{base}_chart.png",
         est, exercise, mvt)
    print(f"\nSaved results next to your video ({video_path.parent}).")
    plt.show()


if __name__ == "__main__":
    main()