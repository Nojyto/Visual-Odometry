# Visual Drone Geofencing

A Python prototype for GPS-denied drone localization and geofencing. This script uses OpenCV and ORB (Oriented FAST and Rotated BRIEF) feature matching to track a live drone camera feed against a high-resolution pre-mapped image.

## Getting Started (Windows)

### 1. Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install the project

```powershell
python -m pip install --upgrade pip
pip install -e .
```

### 3. Run the app

Use either command:

```powershell
python -m drone_geofence
```

or:

```powershell
drone-geofence
```

## Benchmark The Tracking Algorithm

Run objective cross-validation across videos and parameter sets.

### Quick start

```powershell
drone-geofence-benchmark --config benchmarks/benchmark_config.json --csv benchmarks/results/benchmark_results.csv
```

This writes a CSV with per-run metrics and prints top runs by score.

### Fast iteration mode

Use frame skipping and shorter runs when tuning:

```powershell
drone-geofence-benchmark --config benchmarks/benchmark_config.json --stride 3 --max-frames 600
```

### Metrics included

- `tracked_ratio`: fraction of sampled frames with valid tracked pose
- `lost_ratio`: fraction of frames reported as LOST
- `violation_ratio`: fraction of frames reported as VIOLATION
- `mean_proc_ms`, `throughput_fps`: runtime speed metrics
- `mean_step_px`, `p95_step_px`: tracking jitter / motion stability proxy
- `path_len_m`: accumulated path length in map metres
- `gps_rmse_m`: optional GPS error (if DJI `.SRT` exists and has lat/lon)

### Config file

Example config is provided at:

```text
benchmarks/benchmark_config.json
```

You can define multiple `scenarios` (video/map pairs) and `parameter_sets`.

## Build .exe (Windows)

### 1. Install PyInstaller

```powershell
python -m pip install pyinstaller
```

### 2. Build executable

```powershell
python -m PyInstaller --noconfirm --clean --name drone-geofence --windowed --icon src\drone_geofence\assets\icon.ico --paths src --collect-submodules rasterio --collect-submodules shapely --add-data "src\drone_geofence\assets;drone_geofence\assets" src\drone_geofence\__main__.py
```

### 3. Output location

```text
dist\drone-geofence\drone-geofence.exe
```

ui apsirasyt funckionaluma
webodm but nice to have integruota

pasirinkti testavimo scnerijus
a. skirtingos zonos skirtingi video
b. skirtinga zemelapio rezoliucija
c. pasvirimo kampas
d. isivesti metrikas kad butu galima apibendrinta informcija (ant kiek bludina)

- preprocessing ant feed nes jis iskreiptas
- kaip itakoja oro salygos pvz sniegas

- aprasyt ka naudoji algoritma kurios dalys gerai kurios blogint
- pasigilint kas veikia kad suprastum

kaip veikia jeigu nespeja iki galo apdoroti dabartinio frame ir jau yra sekantis
