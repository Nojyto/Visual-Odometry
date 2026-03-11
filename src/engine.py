import cv2
import numpy as np
import rasterio
from rasterio.transform import xy, rowcol
from shapely.geometry import Point, Polygon
from threading import Thread, Lock
from dataclasses import dataclass
from enum import Enum
from typing import Callable


class GeofenceStatus(Enum):
    INITIALIZING = "INITIALIZING"
    SAFE = "SAFE"
    APPROACHING = "APPROACHING"
    VIOLATION = "VIOLATION"
    LOST = "LOST"


@dataclass
class TrackingResult:
    pos: tuple[int, int] | None = None  # (x, y) in crop-space pixels
    footprint: np.ndarray | None = None  # 4x1x2 int32
    status: GeofenceStatus = GeofenceStatus.INITIALIZING
    coords: tuple[float, float] | None = None  # (easting, northing)
    lost_count: int = 0
    distance_to_fence: float | None = None  # metres, negative = inside


@dataclass
class MapInfo:
    full_w: int = 0
    full_h: int = 0
    crs: str = ""
    resolution: float = 0.0  # m/pixel


class TrackingEngine:
    def __init__(
        self,
        map_image_path: str,
        roi_radius: int = 1500,
        smoothing: float = 0.3,
        frame_scale: float = 0.35,
        max_jump: int = 400,
        detect_interval: int = 5,
        buffer_meters: float = 10.0,
    ):
        # Load GeoTIFF --------------------------------------------------
        with rasterio.open(map_image_path) as src:
            self.geo_transform = src.transform
            self.crs = src.crs
            rgb = src.read([1, 2, 3])
            self.full_map_color = np.transpose(rgb, (1, 2, 0)).astype(np.uint8)

        self.full_map_gray = cv2.cvtColor(self.full_map_color, cv2.COLOR_RGB2GRAY)
        self.full_h, self.full_w = self.full_map_gray.shape
        self.resolution = abs(self.geo_transform.a)  # m/pixel

        # Parameters -----------------------------------------------------
        self.roi_radius = roi_radius
        self.smoothing = smoothing
        self.frame_scale = frame_scale
        self.max_jump = max_jump
        self.detect_interval = detect_interval
        self.buffer_meters = buffer_meters

        # Crop state -----------------------------------------------------
        self.crop_x = 0
        self.crop_y = 0
        self.map_gray = self.full_map_gray
        self.map_h, self.map_w = self.map_gray.shape

        # ORB + matcher --------------------------------------------------
        self.orb_map = cv2.ORB_create(nfeatures=15000, scaleFactor=1.2, nlevels=8)
        self.orb_frame = cv2.ORB_create(nfeatures=2000, scaleFactor=1.2, nlevels=8)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

        # Tracking state -------------------------------------------------
        self.last_pos: tuple[int, int] | None = None
        self.smooth_pos: np.ndarray | None = None
        self.smooth_footprint: np.ndarray | None = None
        self.last_status = GeofenceStatus.INITIALIZING
        self.last_coords: tuple[float, float] | None = None
        self.last_distance: float | None = None
        self.kp_map = None
        self.des_map = None
        self.kp_map_pts: np.ndarray | None = None
        self.lost_count = 0
        self.max_lost_before_recovery = 15
        self.prev_gray = None
        self.frame_count = 0

        # Threading ------------------------------------------------------
        self._lock = Lock()
        self._detect_running = False
        self._detect_thread: Thread | None = None

        # Geofence polygon (Shapely, in UTM metres) ----------------------
        self._geofence_poly: Polygon | None = None
        self._geofence_buffer: Polygon | None = None  # outer warning zone

        # Callbacks ------------------------------------------------------
        self._on_result: Callable[[TrackingResult], None] | None = None

    def map_info(self) -> MapInfo:
        return MapInfo(
            full_w=self.full_w,
            full_h=self.full_h,
            crs=str(self.crs),
            resolution=self.resolution,
        )

    def crop_to_region(self, cx: int, cy: int, pad: int = 3000):
        x1 = max(0, cx - pad)
        y1 = max(0, cy - pad)
        x2 = min(self.full_w, cx + pad)
        y2 = min(self.full_h, cy + pad)
        self.crop_x = x1
        self.crop_y = y1
        self.map_gray = self.full_map_gray[y1:y2, x1:x2]
        self.map_h, self.map_w = self.map_gray.shape
        self._reset_tracking()
        self.extract_map_features()

    def get_crop_color(self) -> np.ndarray:
        return self.full_map_color[
            self.crop_y : self.crop_y + self.map_h,
            self.crop_x : self.crop_x + self.map_w,
        ]

    def set_geofence(self, coords_utm: list[tuple[float, float]]):
        if len(coords_utm) < 3:
            self._geofence_poly = None
            self._geofence_buffer = None
            return
        self._geofence_poly = Polygon(coords_utm)
        self._geofence_buffer = self._geofence_poly.buffer(self.buffer_meters)

    def set_geofence_pixels(self, pixel_coords: list[tuple[int, int]]):
        utm_coords = [self.pixel_to_meters(x, y) for x, y in pixel_coords]
        self.set_geofence(utm_coords)

    def get_geofence_pixel_coords(self) -> list[tuple[int, int]] | None:
        if self._geofence_poly is None:
            return None
        coords = list(self._geofence_poly.exterior.coords)[:-1]
        return [self.meters_to_pixel(e, n) for e, n in coords]

    def has_geofence(self) -> bool:
        return self._geofence_poly is not None

    def pixel_to_meters(self, pixel_x: int, pixel_y: int) -> tuple[float, float]:
        full_x = pixel_x + self.crop_x
        full_y = pixel_y + self.crop_y
        return xy(self.geo_transform, full_y, full_x)

    def meters_to_pixel(self, easting: float, northing: float) -> tuple[int, int]:
        row, col = rowcol(self.geo_transform, easting, northing)
        return int(col - self.crop_x), int(row - self.crop_y)

    def set_result_callback(self, cb: Callable[[TrackingResult], None]):
        self._on_result = cb

    def process_frame(self, frame: np.ndarray) -> TrackingResult:
        self.frame_count += 1

        # --- optical flow (every frame, ~2-5 ms) ---
        small = cv2.resize(frame, None, fx=self.frame_scale, fy=self.frame_scale)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        flow = self._estimate_flow(gray)

        with self._lock:
            if self.smooth_pos is not None:
                self.smooth_pos += flow
                if self.smooth_footprint is not None:
                    self.smooth_footprint[:, :, 0] += flow[0]
                    self.smooth_footprint[:, :, 1] += flow[1]
                self.last_pos = (int(self.smooth_pos[0]), int(self.smooth_pos[1]))
                self.last_coords = self.pixel_to_meters(*self.last_pos)

            result = self._build_result()

        # --- submit ORB detection (non-blocking) ---
        submit = self.frame_count % self.detect_interval == 0 or self.lost_count > 3
        if submit:
            self._submit_detection(frame, gray)

        if self._on_result:
            self._on_result(result)

        return result

    def extract_map_features(self):
        self.kp_map, self.des_map = self.orb_map.detectAndCompute(self.map_gray, None)
        self.kp_map_pts = np.array([kp.pt for kp in self.kp_map], dtype=np.float32)

    def _reset_tracking(self):
        self.last_pos = None
        self.smooth_pos = None
        self.smooth_footprint = None
        self.last_status = GeofenceStatus.INITIALIZING
        self.last_coords = None
        self.last_distance = None
        self.prev_gray = None
        self.lost_count = 0
        self.frame_count = 0

    def _estimate_flow(self, gray_small: np.ndarray) -> np.ndarray:
        if self.prev_gray is None or self.smooth_pos is None:
            self.prev_gray = gray_small
            return np.array([0.0, 0.0])

        prev_pts = cv2.goodFeaturesToTrack(self.prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=10)
        if prev_pts is None or len(prev_pts) < 10:
            self.prev_gray = gray_small
            return np.array([0.0, 0.0])

        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray_small, prev_pts, None)
        good = status.ravel() == 1
        if np.sum(good) < 5:
            self.prev_gray = gray_small
            return np.array([0.0, 0.0])

        dx = np.median(next_pts[good, 0, 0] - prev_pts[good, 0, 0])
        dy = np.median(next_pts[good, 0, 1] - prev_pts[good, 0, 1])
        self.prev_gray = gray_small
        return np.array([-dx / self.frame_scale, -dy / self.frame_scale])

    def _get_roi_features(self):
        if self.last_pos is None or self.lost_count >= self.max_lost_before_recovery:
            return self.kp_map, self.des_map
        cx, cy = self.last_pos
        r = self.roi_radius
        pts = self.kp_map_pts
        mask = (pts[:, 0] > cx - r) & (pts[:, 0] < cx + r) & (pts[:, 1] > cy - r) & (pts[:, 1] < cy + r)
        indices = np.where(mask)[0]
        if len(indices) < 50:
            return self.kp_map, self.des_map
        return [self.kp_map[i] for i in indices], self.des_map[indices]

    @staticmethod
    def _valid_footprint(footprint, max_area_ratio=50.0, min_area=1000) -> bool:
        pts = footprint.reshape(4, 2).astype(np.float32)
        area = cv2.contourArea(pts)
        if area < min_area or not cv2.isContourConvex(pts):
            return False
        w, h = cv2.minAreaRect(pts)[1]
        if w == 0 or h == 0:
            return False
        return max(w, h) / min(w, h) <= max_area_ratio

    def _check_geofence(self) -> tuple[GeofenceStatus, float | None]:
        if self.last_coords is None:
            return GeofenceStatus.LOST, None
        if self._geofence_poly is None:
            return GeofenceStatus.SAFE, None

        pt = Point(self.last_coords)
        dist_m = self._geofence_poly.exterior.distance(pt)

        if self._geofence_poly.contains(pt):
            # Inside the blocked area — violation
            return GeofenceStatus.VIOLATION, -dist_m
        else:
            # Outside — check if close to the boundary
            if dist_m < self.buffer_meters:
                return GeofenceStatus.APPROACHING, dist_m
            return GeofenceStatus.SAFE, dist_m

    def _submit_detection(self, frame: np.ndarray, gray_small: np.ndarray):
        with self._lock:
            if self._detect_running:
                return
            self._detect_running = True
        h, w = frame.shape[:2]
        self._detect_thread = Thread(target=self._detect_worker, args=(gray_small.copy(), h, w), daemon=True)
        self._detect_thread.start()

    def _detect_worker(self, small_gray: np.ndarray, orig_h: int, orig_w: int):
        kp_frame, des_frame = self.orb_frame.detectAndCompute(small_gray, None)
        if des_frame is None or len(des_frame) < 4:
            with self._lock:
                self.lost_count += 1
                self._detect_running = False
            return

        with self._lock:
            kp_roi, des_roi = self._get_roi_features()

        matches = self.matcher.knnMatch(des_frame, des_roi, k=2)
        good = [m for m, n in (p for p in matches if len(p) == 2) if m.distance < 0.80 * n.distance]

        if len(good) < 10:
            with self._lock:
                self.lost_count += 1
                self._detect_running = False
            return

        src = np.float32([kp_frame[m.queryIdx].pt for m in good]).reshape(-1, 1, 2) / self.frame_scale
        dst = np.float32([kp_roi[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)

        if M is None or (np.sum(mask) / len(mask) if mask is not None else 0) < 0.25:
            with self._lock:
                self.lost_count += 1
                self._detect_running = False
            return

        corners = np.float32([[0, 0], [0, orig_h - 1], [orig_w - 1, orig_h - 1], [orig_w - 1, 0]]).reshape(-1, 1, 2)
        fp = cv2.perspectiveTransform(corners, M)

        if not self._valid_footprint(fp):
            with self._lock:
                self.lost_count += 1
                self._detect_running = False
            return

        raw_pos = np.array([np.mean(fp[:, 0, 0]), np.mean(fp[:, 0, 1])], dtype=np.float64)
        raw_fp = fp.astype(np.float64)

        with self._lock:
            if self.smooth_pos is not None:
                jump = np.linalg.norm(raw_pos - self.smooth_pos)
                if jump > self.max_jump and self.lost_count < self.max_lost_before_recovery:
                    self.lost_count += 1
                    self._detect_running = False
                    return

            if self.smooth_pos is None or self.lost_count >= self.max_lost_before_recovery:
                self.smooth_pos = raw_pos
                self.smooth_footprint = raw_fp
            else:
                a = self.smoothing
                self.smooth_pos = a * raw_pos + (1 - a) * self.smooth_pos
                self.smooth_footprint = a * raw_fp + (1 - a) * self.smooth_footprint

            self.last_pos = (int(self.smooth_pos[0]), int(self.smooth_pos[1]))
            self.last_coords = self.pixel_to_meters(*self.last_pos)
            self.lost_count = 0

            status, dist = self._check_geofence()
            self.last_status = status
            self.last_distance = dist
            self._detect_running = False

    def _build_result(self) -> TrackingResult:
        if self.lost_count > 0 and self.last_pos is None:
            return TrackingResult(status=GeofenceStatus.LOST, lost_count=self.lost_count)

        status, dist = self._check_geofence()

        return TrackingResult(
            pos=self.last_pos,
            footprint=np.int32(self.smooth_footprint) if self.smooth_footprint is not None else None,
            status=status if self.last_pos else GeofenceStatus.INITIALIZING,
            coords=self.last_coords,
            lost_count=self.lost_count,
            distance_to_fence=dist,
        )
