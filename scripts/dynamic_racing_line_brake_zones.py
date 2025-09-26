"""
Dynamic Racing Line (Brake-Zones Method)
=======================================

Segments the lap into zones where a zone starts when the first of the two
drivers begins braking and ends when the next braking event (from either
driver) occurs. For each zone, color the racing line by the driver who gained
time delta within that zone (i.e., whose time delta change from the start to
the end of the zone is more favorable), rather than the raw delta from lap start.

Keeps the original dynamic_racing_line.py intact.
"""

import json
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import argparse
import os
from typing import Dict, List, Optional, Tuple
import logging

# Colors
ORANGE = '#FF8C00'  # Driver1 better
GREY = '#404040'    # Driver2 better

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BrakeZonesDynamicLine:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.session_data = None
        self.driver_data: Dict[str, Dict] = {}

    def _find_session_file(self) -> Optional[str]:
        files = [f for f in os.listdir(self.data_dir) if f.endswith('.json') and 'telemetry' not in f]
        return os.path.join(self.data_dir, files[0]) if files else None

    def _find_telemetry_dir(self) -> Optional[str]:
        for item in os.listdir(self.data_dir):
            if item.endswith('_telemetry'):
                return os.path.join(self.data_dir, item)
        return None

    def load(self, driver1: str, driver2: str) -> None:
        session_file = self._find_session_file()
        if not session_file:
            raise FileNotFoundError('Session JSON not found')
        with open(session_file, 'r') as f:
            self.session_data = json.load(f)

        telemetry_dir = self._find_telemetry_dir()
        if not telemetry_dir:
            raise FileNotFoundError('Telemetry directory not found')
        for drv in [driver1, driver2]:
            fp = os.path.join(telemetry_dir, f"{drv}_telemetry.json")
            with open(fp, 'r') as f:
                self.driver_data[drv] = json.load(f)

    @staticmethod
    def _parse_time_seconds(time_series: List[Optional[str]]) -> List[Optional[float]]:
        out: List[Optional[float]] = []
        start = None
        for t in time_series:
            if t is None:
                out.append(None)
                continue
            try:
                parts = t.split()
                clock = parts[2]
                h, m, s = clock.split(':')
                total = float(h) * 3600 + float(m) * 60 + float(s)
                if start is None:
                    start = total
                out.append(total - start)
            except Exception:
                out.append(None)
        return out

    @staticmethod
    def _cum_distance(xs: List[float], ys: List[float]) -> List[float]:
        if not xs or not ys:
            return []
        d = [0.0]
        for i in range(1, len(xs)):
            if xs[i] is None or ys[i] is None or xs[i-1] is None or ys[i-1] is None:
                d.append(d[-1])
            else:
                dx = xs[i] - xs[i-1]
                dy = ys[i] - ys[i-1]
                d.append(d[-1] + (dx*dx + dy*dy) ** 0.5)
        return d

    def _get_xy_brake_time(self, driver: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        d = self.driver_data[driver]
        xs = d['position_data']['X']
        ys = d['position_data']['Y']
        time_series = d['telemetry']['Time']
        brake = d['telemetry']['Brake']

        # Clean lists
        cx, cy, cb, ct = [], [], [], []
        for x, y, b, t in zip(xs, ys, brake, time_series):
            if x is not None and y is not None:
                cx.append(x); cy.append(y)
                cb.append(0 if b is None else b)
                ct.append(t)

        tsec = self._parse_time_seconds(ct)
        # Ensure no None in tsec for interpolation
        tsec = [0.0 if v is None else v for v in tsec]

        return np.array(cx), np.array(cy), np.array(cb, dtype=float), np.array(tsec, dtype=float)

    def _build_common_dist_grid(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        d = self._cum_distance(x.tolist(), y.tolist())
        if not d:
            return np.array([]), np.array([])
        total = d[-1] if d[-1] > 0 else 1.0
        dist_pct = np.array([val / total * 100.0 for val in d], dtype=float)
        common = np.linspace(0, 100, 1200)
        return np.array(d, dtype=float), common

    def _interp_to_common(self, dist_pct_src: np.ndarray, common: np.ndarray, arr: np.ndarray) -> np.ndarray:
        # dist_pct_src might not be strictly monotonic; enforce monotonic by unique increasing indices
        order = np.argsort(dist_pct_src)
        x = dist_pct_src[order]
        y = arr[order]
        # Remove duplicates
        uniq_x, uniq_idx = np.unique(x, return_index=True)
        uniq_y = y[uniq_idx]
        return np.interp(common, uniq_x, uniq_y)

    def _segment_zones(self, brake1: np.ndarray, brake2: np.ndarray, common: np.ndarray) -> List[Tuple[int, int]]:
        """Return list of (start_idx, end_idx) for zones.
        A zone starts when either driver's brake goes from 0 to >0, and ends at the next such event.
        """
        starts = []
        prev_b1 = 0.0
        prev_b2 = 0.0
        for i in range(1, len(common)):
            b1 = brake1[i] > 0.0 and brake1[i-1] == 0.0
            b2 = brake2[i] > 0.0 and brake2[i-1] == 0.0
            if b1 or b2:
                starts.append(i)
        # Build zones from consecutive starts
        zones: List[Tuple[int,int]] = []
        if not starts:
            zones.append((0, len(common) - 1))
            return zones
        # Ensure a start at 0 if the first braking occurs mid-lap
        if starts[0] != 0:
            starts = [0] + starts
        # Ensure the last zone closes at the end
        if starts[-1] < len(common) - 1:
            starts.append(len(common) - 1)
        for i in range(len(starts) - 1):
            zones.append((starts[i], max(starts[i+1]-1, starts[i])))
        return zones

    def _zone_winner_color(self, td: np.ndarray, start: int, end: int) -> str:
        # Delta change within zone: smaller (more negative) change in td means driver1 gained less time relative to driver2.
        # We want the driver whose delta change is better for them. Since td = t1 - t2:
        # If td decreases (end < start), driver1 gained → ORANGE
        # If td increases (end > start), driver2 gained → GREY
        d_change = td[end] - td[start]
        return ORANGE if d_change < 0 else GREY

    def plot(self, driver1: str, driver2: str, save_path: Optional[str] = None) -> None:
        # Load series
        x1, y1, b1, t1s = self._get_xy_brake_time(driver1)
        x2, y2, b2, t2s = self._get_xy_brake_time(driver2)

        # Build distance percentages for driver1 (reference for XY path)
        d1, common = self._build_common_dist_grid(x1, y1)
        if common.size == 0:
            raise RuntimeError('Could not build distance grid')
        dist_pct1 = d1 / (d1[-1] if d1[-1] > 0 else 1.0) * 100.0

        # Interpolate series onto common distance grid
        t1c = self._interp_to_common(dist_pct1, common, t1s)
        t2c = self._interp_to_common(dist_pct1, common, t2s)
        x1c = self._interp_to_common(dist_pct1, common, x1)
        y1c = self._interp_to_common(dist_pct1, common, y1)
        b1c = self._interp_to_common(dist_pct1, common, b1)
        # For driver2 brake aligned to driver1 distance
        # Build dist grid for driver2 and then map brake to common
        d2 = self._cum_distance(x2.tolist(), y2.tolist())
        dist_pct2 = np.array([val / (d2[-1] if d2[-1] > 0 else 1.0) * 100.0 for val in d2], dtype=float)
        b2c = self._interp_to_common(dist_pct2, common, b2)

        # Compute time delta on common grid
        td = t1c - t2c

        # Zones from brake-on events
        zones = self._segment_zones(b1c, b2c, common)

        # Plot
        plt.figure(figsize=(15, 12))
        # Light track outline
        plt.plot(x1c, y1c, color='#d3d3d3', linewidth=1.5, alpha=0.9, zorder=1)

        # Draw zone-colored segments
        for (s, e) in zones:
            color = self._zone_winner_color(td, s, e)
            plt.plot(x1c[s:e+1], y1c[s:e+1], color=color, linewidth=8, alpha=0.9, zorder=2)

        # Ensure closure at finish
        plt.plot([x1c[-1], x1c[0]], [y1c[-1], y1c[0]], color=self._zone_winner_color(td, zones[-1][0], zones[-1][1]), linewidth=8, alpha=0.9, zorder=2)

        # Cosmetics
        plt.axis('equal')
        plt.grid(False)
        x_min, x_max = np.nanmin(x1c), np.nanmax(x1c)
        y_min, y_max = np.nanmin(y1c), np.nanmax(y1c)
        xpad = (x_max - x_min) * 0.05
        ypad = (y_max - y_min) * 0.05
        plt.xlim(x_min - xpad, x_max + xpad)
        plt.ylim(y_min - ypad, y_max + ypad)

        if save_path:
            os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Brake-zones dynamic racing line saved to: {save_path}")
        else:
            plt.show()

    def animate(self, driver1: str, driver2: str, video_path: str, duration_sec: float = 30.0) -> None:
        # Load series and prepare common grid
        x1, y1, b1, t1s = self._get_xy_brake_time(driver1)
        x2, y2, b2, t2s = self._get_xy_brake_time(driver2)
        d1, common = self._build_common_dist_grid(x1, y1)
        if common.size == 0:
            raise RuntimeError('Could not build distance grid')
        dist_pct1 = d1 / (d1[-1] if d1[-1] > 0 else 1.0) * 100.0

        t1c = self._interp_to_common(dist_pct1, common, t1s)
        t2c = self._interp_to_common(dist_pct1, common, t2s)
        x1c = self._interp_to_common(dist_pct1, common, x1)
        y1c = self._interp_to_common(dist_pct1, common, y1)
        b1c = self._interp_to_common(dist_pct1, common, b1)
        d2 = self._cum_distance(x2.tolist(), y2.tolist())
        dist_pct2 = np.array([val / (d2[-1] if d2[-1] > 0 else 1.0) * 100.0 for val in d2], dtype=float)
        b2c = self._interp_to_common(dist_pct2, common, b2)

        td = t1c - t2c
        zones = self._segment_zones(b1c, b2c, common)

        # Precompute per-point color by zone winner
        point_colors = np.array([GREY] * len(common), dtype=object)
        for (s, e) in zones:
            color = self._zone_winner_color(td, s, e)
            point_colors[s:e+1] = color

        fig, ax = plt.subplots(figsize=(15, 12))
        ax.set_aspect('equal', adjustable='box')
        # Outline
        ax.plot(np.r_[x1c, x1c[0]], np.r_[y1c, y1c[0]], color='#d3d3d3', linewidth=1.5, alpha=0.9, zorder=1)

        # Prepare animated segments as invisible initially
        seg_lines = []
        for i in range(len(common) - 1):
            ln, = ax.plot([x1c[i], x1c[i+1]], [y1c[i], y1c[i+1]], color=point_colors[i], linewidth=8, alpha=0.9, zorder=2)
            ln.set_visible(False)
            seg_lines.append(ln)
        # Closing segment
        ln_close, = ax.plot([x1c[-1], x1c[0]], [y1c[-1], y1c[0]], color=point_colors[-1], linewidth=8, alpha=0.9, zorder=2)
        ln_close.set_visible(False)

        x_min, x_max = np.nanmin(x1c), np.nanmax(x1c)
        y_min, y_max = np.nanmin(y1c), np.nanmax(y1c)
        xpad = (x_max - x_min) * 0.05
        ypad = (y_max - y_min) * 0.05
        ax.set_xlim(x_min - xpad, x_max + xpad)
        ax.set_ylim(y_min - ypad, y_max + ypad)
        ax.grid(False)

        total_frames = len(seg_lines) + 1
        duration_sec = max(20.0, min(40.0, duration_sec))
        interval_ms = (duration_sec * 1000.0) / total_frames
        fps = max(1, int(round(1000.0 / interval_ms)))

        def init():
            for ln in seg_lines:
                ln.set_visible(False)
            ln_close.set_visible(False)
            return seg_lines + [ln_close]

        def update(fi: int):
            if fi < len(seg_lines):
                seg_lines[fi].set_visible(True)
                return [seg_lines[fi]]
            else:
                ln_close.set_visible(True)
                return [ln_close]

        anim = animation.FuncAnimation(fig, update, init_func=init, frames=total_frames, interval=interval_ms, blit=True, repeat=False)

        os.makedirs(os.path.dirname(video_path) or '.', exist_ok=True)
        writer = animation.FFMpegWriter(fps=fps, metadata={'artist': 'F1 Telemetry Analysis'})
        anim.save(video_path, writer=writer, dpi=200)
        plt.close(fig)
        logger.info(f"Brake-zones animation saved to: {video_path} (~{duration_sec:.1f}s, fps {fps})")


def main():
    parser = argparse.ArgumentParser(description='Dynamic Racing Line using Brake-Zones segmentation')
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--driver1', type=str, required=True)
    parser.add_argument('--driver2', type=str, required=True)
    parser.add_argument('--save', type=str, help='Output image path')
    parser.add_argument('--animate', action='store_true', help='Export an animated MP4 of the brake-zones line')
    parser.add_argument('--video', type=str, help='Output MP4 path when using --animate')
    parser.add_argument('--duration', type=float, default=30.0, help='Video duration (20–40s recommended)')
    args = parser.parse_args()

    tool = BrakeZonesDynamicLine(args.data_dir)
    tool.load(args.driver1, args.driver2)
    if args.animate:
        if not args.video:
            raise ValueError('Please provide --video when using --animate')
        tool.animate(args.driver1, args.driver2, args.video, duration_sec=args.duration)
    else:
        tool.plot(args.driver1, args.driver2, args.save)


if __name__ == '__main__':
    main()


