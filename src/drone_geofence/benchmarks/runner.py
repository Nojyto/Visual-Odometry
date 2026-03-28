import argparse
import csv
import json
import math
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..engine import GeofenceStatus, TrackingEngine


LAT_RE = re.compile(r"\[latitude:\s*([-+]?\d+(?:\.\d+)?)\]")
LON_RE = re.compile(r"\[longitude:\s*([-+]?\d+(?:\.\d+)?)\]")


@dataclass
class Scenario:
    name: str
    map_path: str
    video_path: str
    crop_center: tuple[int, int] | None = None
    crop_pad: int = 3000
    srt_path: str | None = None


@dataclass
class RunMetrics:
    scenario: str
    params_name: str
    samples: int
    tracked_ratio: float
    lost_ratio: float
    violation_ratio: float
    mean_proc_ms: float
    throughput_fps: float
    mean_step_px: float
    p95_step_px: float
    path_len_m: float
    gps_rmse_m: float | None
    score: float


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371000.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.atan2(math.sqrt(a), math.sqrt(max(1e-12, 1 - a)))


def _extract_srt_latlon(srt_path: str) -> list[tuple[float, float] | None]:
    p = Path(srt_path)
    if not p.exists():
        return []

    lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()
    out: list[tuple[float, float] | None] = []
    current_lat = None
    current_lon = None

    for line in lines:
        mlat = LAT_RE.search(line)
        mlon = LON_RE.search(line)
        if mlat:
            current_lat = float(mlat.group(1))
        if mlon:
            current_lon = float(mlon.group(1))

        if "</font>" in line:
            if current_lat is not None and current_lon is not None:
                out.append((current_lat, current_lon))
            else:
                out.append(None)
            current_lat = None
            current_lon = None

    return out


def _discover_default_scenarios(workspace: Path) -> list[Scenario]:
    map_path = workspace / "data" / "Radviln-pl-3-7-2026-orthophoto.tif"
    feeds_dir = workspace / "data" / "DroneFeeds"
    scenarios: list[Scenario] = []

    if not feeds_dir.exists() or not map_path.exists():
        return scenarios

    for mp4 in sorted(feeds_dir.glob("*.MP4")):
        srt = mp4.with_suffix(".SRT")
        scenarios.append(
            Scenario(
                name=mp4.stem,
                map_path=str(map_path),
                video_path=str(mp4),
                crop_center=None,
                crop_pad=3000,
                srt_path=str(srt) if srt.exists() else None,
            )
        )

    return scenarios


def _load_config(config_path: Path, workspace: Path) -> tuple[list[Scenario], list[dict[str, Any]], int, int | None]:
    if not config_path.exists():
        scenarios = _discover_default_scenarios(workspace)
        param_sets = [{"name": "baseline", "params": {}}]
        return scenarios, param_sets, 2, 1200

    cfg = json.loads(config_path.read_text(encoding="utf-8"))

    scenarios = [
        Scenario(
            name=s["name"],
            map_path=s["map_path"],
            video_path=s["video_path"],
            crop_center=tuple(s["crop_center"]) if s.get("crop_center") else None,
            crop_pad=int(s.get("crop_pad", 3000)),
            srt_path=s.get("srt_path"),
        )
        for s in cfg.get("scenarios", [])
    ]

    param_sets = cfg.get("parameter_sets", [{"name": "baseline", "params": {}}])
    stride = int(cfg.get("stride", 2))
    max_frames = cfg.get("max_frames", 1200)
    if max_frames is not None:
        max_frames = int(max_frames)

    return scenarios, param_sets, stride, max_frames


def _safe_mean(vals: list[float]) -> float:
    return float(np.mean(vals)) if vals else 0.0


def _safe_percentile(vals: list[float], p: float) -> float:
    return float(np.percentile(vals, p)) if vals else 0.0


def _score(metrics: RunMetrics) -> float:
    score = 100.0
    score -= (1.0 - metrics.tracked_ratio) * 60.0
    score -= metrics.lost_ratio * 20.0
    score -= metrics.violation_ratio * 15.0
    score -= min(20.0, metrics.p95_step_px / 25.0)
    if metrics.gps_rmse_m is not None:
        score -= min(25.0, metrics.gps_rmse_m / 5.0)
    return max(0.0, score)


def _run_one(
    scenario: Scenario,
    params_name: str,
    params: dict[str, Any],
    stride: int,
    max_frames: int | None,
) -> RunMetrics:
    engine = TrackingEngine(scenario.map_path, **params)
    info = engine.map_info()

    if scenario.crop_center is None:
        cx, cy = info.full_w // 2, info.full_h // 2
    else:
        cx, cy = scenario.crop_center
    engine.crop_to_region(cx, cy, pad=scenario.crop_pad)

    srt_samples = _extract_srt_latlon(scenario.srt_path) if scenario.srt_path else []

    cap = cv2.VideoCapture(scenario.video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {scenario.video_path}")

    samples = 0
    tracked = 0
    lost = 0
    violations = 0
    proc_times_ms: list[float] = []
    step_px: list[float] = []
    path_m: list[float] = []
    gps_err_m: list[float] = []

    last_pos = None
    last_coords = None
    frame_idx = -1

    t0 = time.perf_counter()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        if stride > 1 and (frame_idx % stride) != 0:
            continue
        if max_frames is not None and samples >= max_frames:
            break

        st = time.perf_counter()
        result = engine.process_frame(frame)
        proc_times_ms.append((time.perf_counter() - st) * 1000.0)

        samples += 1
        if result.status == GeofenceStatus.LOST:
            lost += 1
        if result.status == GeofenceStatus.VIOLATION:
            violations += 1

        if result.pos is not None and result.status != GeofenceStatus.INITIALIZING:
            tracked += 1
            if last_pos is not None:
                dx = result.pos[0] - last_pos[0]
                dy = result.pos[1] - last_pos[1]
                step_px.append(math.hypot(dx, dy))
            last_pos = result.pos

        if result.coords is not None:
            if last_coords is not None:
                de = result.coords[0] - last_coords[0]
                dn = result.coords[1] - last_coords[1]
                path_m.append(math.hypot(de, dn))
            last_coords = result.coords

            if frame_idx < len(srt_samples) and srt_samples[frame_idx] is not None:
                pred_ll = engine.coords_to_latlon(result.coords[0], result.coords[1])
                if pred_ll is not None:
                    gt_lat, gt_lon = srt_samples[frame_idx]
                    gps_err_m.append(_haversine_m(pred_ll[0], pred_ll[1], gt_lat, gt_lon))

    cap.release()

    elapsed = max(1e-9, time.perf_counter() - t0)
    metrics = RunMetrics(
        scenario=scenario.name,
        params_name=params_name,
        samples=samples,
        tracked_ratio=(tracked / samples) if samples else 0.0,
        lost_ratio=(lost / samples) if samples else 0.0,
        violation_ratio=(violations / samples) if samples else 0.0,
        mean_proc_ms=_safe_mean(proc_times_ms),
        throughput_fps=(samples / elapsed),
        mean_step_px=_safe_mean(step_px),
        p95_step_px=_safe_percentile(step_px, 95),
        path_len_m=float(np.sum(path_m)) if path_m else 0.0,
        gps_rmse_m=float(np.sqrt(np.mean(np.square(gps_err_m)))) if gps_err_m else None,
        score=0.0,
    )
    metrics.score = _score(metrics)
    return metrics


def _write_csv(path: Path, rows: list[RunMetrics]):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(rows[0]).keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))


def main():
    parser = argparse.ArgumentParser(description="Benchmark visual odometry tracking across videos and parameters")
    parser.add_argument(
        "--config",
        default="benchmarks/benchmark_config.json",
        help="Path to benchmark config JSON. If missing, auto-discovers data/DroneFeeds/*.MP4",
    )
    parser.add_argument("--csv", default="benchmarks/results/benchmark_results.csv", help="Output CSV path")
    parser.add_argument("--stride", type=int, default=None, help="Override frame stride from config")
    parser.add_argument("--max-frames", type=int, default=None, help="Override max frames per run")
    args = parser.parse_args()

    workspace = Path.cwd()
    scenarios, param_sets, stride, max_frames = _load_config(Path(args.config), workspace)

    if args.stride is not None:
        stride = max(1, args.stride)
    if args.max_frames is not None:
        max_frames = args.max_frames

    if not scenarios:
        raise SystemExit("No scenarios found. Provide --config or place videos in data/DroneFeeds.")

    results: list[RunMetrics] = []

    print(f"Scenarios: {len(scenarios)} | Param sets: {len(param_sets)} | stride={stride} | max_frames={max_frames}")
    for s in scenarios:
        for pset in param_sets:
            name = pset.get("name", "params")
            params = pset.get("params", {})
            print(f"Running {s.name} :: {name} ...")
            m = _run_one(s, name, params, stride=stride, max_frames=max_frames)
            results.append(m)
            gps_txt = f" | gps_rmse={m.gps_rmse_m:.2f}m" if m.gps_rmse_m is not None else ""
            print(
                f"  score={m.score:.1f} tracked={m.tracked_ratio:.3f} lost={m.lost_ratio:.3f}"
                f" viol={m.violation_ratio:.3f} p95step={m.p95_step_px:.1f}px fps={m.throughput_fps:.1f}{gps_txt}"
            )

    results.sort(key=lambda r: r.score, reverse=True)

    out_csv = Path(args.csv)
    _write_csv(out_csv, results)

    print("\nTop 10 runs:")
    for r in results[:10]:
        print(
            f"  {r.scenario:18s} | {r.params_name:16s} | score={r.score:6.2f}"
            f" | tracked={r.tracked_ratio:.3f} | lost={r.lost_ratio:.3f} | fps={r.throughput_fps:.1f}"
        )

    print(f"\nSaved CSV: {out_csv}")


if __name__ == "__main__":
    main()
