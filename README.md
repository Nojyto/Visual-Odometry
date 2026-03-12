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
