import cv2
import numpy as np
import rasterio
from rasterio.transform import xy, rowcol
from rasterio.warp import transform as warp_transform
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
        frame_scale: float = 0.50,
        max_jump: int = 400,
        detect_interval: int = 2,
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
        self.flow_min_inliers = 20
        self.flow_redetect_confidence = 0.12
        self.flow_reproj_threshold = 2.5

        # Crop state -----------------------------------------------------
        self.crop_x = 0
        self.crop_y = 0
        self.map_gray = self.full_map_gray
        self.map_h, self.map_w = self.map_gray.shape

        # CLAHE for contrast normalisation (helps with illumination & angle) --
        self.clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))

        # AKAZE detectors  -----------------------------------------------
        self.det_map = cv2.AKAZE_create(
            descriptor_type=cv2.AKAZE_DESCRIPTOR_MLDB,
            threshold=0.003,
            nOctaves=4,
            nOctaveLayers=4,
        )
        self.det_frame = cv2.AKAZE_create(
            descriptor_type=cv2.AKAZE_DESCRIPTOR_MLDB,
            threshold=0.001,
            nOctaves=4,
            nOctaveLayers=4,
        )
        self.matcher = cv2.FlannBasedMatcher(
            dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1),  # LSH
            dict(checks=50),
        )

        # OpenCL / GPU acceleration ---------------
        self._use_opencl = cv2.ocl.haveOpenCL()
        if self._use_opencl:
            cv2.ocl.setUseOpenCL(True)

        # Tracking state -------------------------------------------------
        self.last_pos: tuple[int, int] | None = None
        self.smooth_pos: np.ndarray | None = None
        self.smooth_footprint: np.ndarray | None = None
        self.smooth_heading: float = 0.0
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
        self.last_flow_confidence = 0.0
        self.flow_bad_streak = 0

        # Anchor-based tracking ------------------------------------------
        self._anchor_pos: np.ndarray | None = None
        self._flow_accum = np.zeros(2, dtype=np.float64)
        self._heading_anchor: float = 0.0  # last absolute AKAZE heading
        self._heading_accum: float = 0.0  # accumulated flow rotation since anchor
        self._fp_local: np.ndarray | None = None  # 4x2 corners in heading-aligned local frame

        # Display-level EMA (anti-flicker) -------------------------------
        self._display_pos: np.ndarray | None = None
        self._display_heading: float = 0.0
        self._display_alpha: float = 0.35

        # Threading ------------------------------------------------------
        self._lock = Lock()
        self._detect_running = False
        self._detect_thread: Thread | None = None
        self._map_features_ready = False

        self._wgs84_crs = "EPSG:4326"

        # Geofence polygon (Shapely, in UTM metres) ----------------------
        self._geofence_poly: Polygon | None = None
        self._geofence_holes_utm: list[list[tuple[float, float]]] = []

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
        self._map_features_ready = False
        t = Thread(target=self._extract_map_features_bg, daemon=True)
        t.start()

    def _extract_map_features_bg(self):
        self.extract_map_features()
        self._map_features_ready = True

    def get_crop_color(self) -> np.ndarray:
        return self.full_map_color[
            self.crop_y : self.crop_y + self.map_h,
            self.crop_x : self.crop_x + self.map_w,
        ]

    def set_geofence(
        self,
        coords_utm: list[tuple[float, float]],
        holes_utm: list[list[tuple[float, float]]] | None = None,
    ):
        if len(coords_utm) < 3:
            self._geofence_poly = None
            self._geofence_holes_utm = []
            return

        valid_holes: list[list[tuple[float, float]]] = []
        if holes_utm:
            valid_holes = [h for h in holes_utm if len(h) >= 3]

        poly = Polygon(coords_utm, holes=valid_holes if valid_holes else None)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty or not poly.is_valid:
            self._geofence_poly = None
            self._geofence_holes_utm = []
            return

        self._geofence_poly = poly
        self._geofence_holes_utm = [list(h) for h in valid_holes]

    def set_geofence_pixels(
        self,
        pixel_coords: list[tuple[int, int]],
        hole_pixel_coords_list: list[list[tuple[int, int]]] | None = None,
    ):
        utm_coords = [self.pixel_to_meters(x, y) for x, y in pixel_coords]
        holes_utm = None
        if hole_pixel_coords_list:
            holes_utm = [
                [self.pixel_to_meters(x, y) for x, y in hole] for hole in hole_pixel_coords_list if len(hole) >= 3
            ]
        self.set_geofence(utm_coords, holes_utm)

    def get_geofence_pixel_coords(self) -> list[tuple[int, int]] | None:
        if self._geofence_poly is None:
            return None
        coords = list(self._geofence_poly.exterior.coords)[:-1]
        return [self.meters_to_pixel(e, n) for e, n in coords]

    def get_geofence_holes_pixel_coords(self) -> list[list[tuple[int, int]]]:
        if self._geofence_poly is None:
            return []
        holes: list[list[tuple[int, int]]] = []
        for interior in self._geofence_poly.interiors:
            coords = list(interior.coords)[:-1]
            holes.append([self.meters_to_pixel(e, n) for e, n in coords])
        return holes

    def get_geofence_definition_pixels(self) -> tuple[list[tuple[int, int]] | None, list[list[tuple[int, int]]]]:
        return self.get_geofence_pixel_coords(), self.get_geofence_holes_pixel_coords()

    def has_geofence(self) -> bool:
        return self._geofence_poly is not None

    def pixel_to_meters(self, pixel_x: int, pixel_y: int) -> tuple[float, float]:
        full_x = pixel_x + self.crop_x
        full_y = pixel_y + self.crop_y
        return xy(self.geo_transform, full_y, full_x)

    def meters_to_pixel(self, easting: float, northing: float) -> tuple[int, int]:
        row, col = rowcol(self.geo_transform, easting, northing)
        return int(col - self.crop_x), int(row - self.crop_y)

    def coords_to_latlon(self, easting: float, northing: float) -> tuple[float, float] | None:
        try:
            xs, ys = warp_transform(self.crs, self._wgs84_crs, [easting], [northing])
            return float(ys[0]), float(xs[0])  # lat, lon
        except Exception:
            return None

    def set_result_callback(self, cb: Callable[[TrackingResult], None]):
        self._on_result = cb

    def process_frame(self, frame: np.ndarray) -> TrackingResult:
        self.frame_count += 1

        # --- optical flow (every frame, ~2-5 ms) ---
        small = cv2.resize(frame, None, fx=self.frame_scale, fy=self.frame_scale)
        gray = self.clahe.apply(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
        flow_trans, flow_rot, flow_conf = self._estimate_flow(gray)

        with self._lock:
            self.last_flow_confidence = flow_conf
            if flow_conf < self.flow_redetect_confidence:
                self.flow_bad_streak += 1
            else:
                self.flow_bad_streak = 0

            if self._anchor_pos is not None and flow_trans is not None and flow_conf > 0.05:
                gain = 0.25 + 0.65 * float(flow_conf)
                self._flow_accum[0] += flow_trans[0] * gain
                self._flow_accum[1] += flow_trans[1] * gain
                self._flow_accum *= 0.96

                max_accum = self.max_jump * 0.35
                mag = np.linalg.norm(self._flow_accum)
                if mag > max_accum:
                    self._flow_accum *= max_accum / mag

                self.smooth_pos = self._anchor_pos + self._flow_accum

                if flow_rot is not None:
                    rot_gain = gain * 0.6
                    self._heading_accum += float(np.degrees(flow_rot)) * rot_gain
                    self._heading_accum *= 0.96
                    max_rot = 30.0
                    self._heading_accum = max(-max_rot, min(max_rot, self._heading_accum))
                    self.smooth_heading = self._heading_anchor + self._heading_accum

            if self.smooth_pos is not None and self._fp_local is not None:
                if self._display_pos is None:
                    self._display_pos = self.smooth_pos.copy()
                    self._display_heading = self.smooth_heading
                else:
                    a = self._display_alpha
                    self._display_pos = a * self.smooth_pos + (1 - a) * self._display_pos
                    hdiff = self._wrap_angle_deg(self.smooth_heading - self._display_heading)
                    self._display_heading += a * hdiff

                self.smooth_footprint = self._make_footprint_display()
                self.last_pos = (int(self._display_pos[0]), int(self._display_pos[1]))
                self.last_coords = self.pixel_to_meters(*self.last_pos)

            result = self._build_result()

        # --- submit ORB detection (non-blocking) ---
        submit = (
            self.frame_count % self.detect_interval == 0
            or self.lost_count > 3
            or self.last_flow_confidence < self.flow_redetect_confidence
            or self.flow_bad_streak >= 2
        )
        if submit:
            self._submit_detection(frame, gray)

        if self._on_result:
            self._on_result(result)

        return result

    def extract_map_features(self):
        enhanced = self.clahe.apply(self.map_gray)
        self.kp_map, self.des_map = self.det_map.detectAndCompute(enhanced, None)
        self.kp_map_pts = np.array([kp.pt for kp in self.kp_map], dtype=np.float32)

    def _reset_tracking(self):
        self.last_pos = None
        self.smooth_pos = None
        self.smooth_footprint = None
        self.smooth_heading = 0.0
        self.last_status = GeofenceStatus.INITIALIZING
        self.last_coords = None
        self.last_distance = None
        self.prev_gray = None
        self.lost_count = 0
        self.frame_count = 0
        self.last_flow_confidence = 0.0
        self.flow_bad_streak = 0
        self._anchor_pos = None
        self._flow_accum = np.zeros(2, dtype=np.float64)
        self._heading_anchor = 0.0
        self._heading_accum = 0.0
        self._fp_local = None
        self._display_pos = None
        self._display_heading = 0.0

    def _make_footprint_display(self) -> np.ndarray:
        return self._build_footprint(self._display_pos, self._display_heading)

    def _make_footprint(self) -> np.ndarray:
        return self._build_footprint(self.smooth_pos, self.smooth_heading)

    @staticmethod
    def _wrap_angle_deg(angle: float) -> float:
        return (angle + 180.0) % 360.0 - 180.0

    def _shape_to_local(self, corners: np.ndarray, center: np.ndarray, heading_deg: float) -> np.ndarray:
        a = np.radians(heading_deg)
        ca, sa = np.cos(a), np.sin(a)
        r_inv = np.array([[ca, sa], [-sa, ca]], dtype=np.float64)
        return (corners - center) @ r_inv.T

    def _build_footprint(self, center: np.ndarray | None, heading_deg: float) -> np.ndarray | None:
        if center is None or self._fp_local is None:
            return None
        a = np.radians(heading_deg)
        ca, sa = np.cos(a), np.sin(a)
        r = np.array([[ca, -sa], [sa, ca]], dtype=np.float64)
        world = self._fp_local @ r.T + center
        return world.reshape(4, 1, 2)

    def _estimate_flow(self, gray_small: np.ndarray) -> tuple[np.ndarray | None, float | None, float]:
        if self.prev_gray is None or self.smooth_pos is None:
            self.prev_gray = gray_small
            return None, None, 0.0

        prev_pts = cv2.goodFeaturesToTrack(self.prev_gray, maxCorners=600, qualityLevel=0.005, minDistance=6)
        if prev_pts is None or len(prev_pts) < 10:
            self.prev_gray = gray_small
            return None, None, 0.0

        lk_params = dict(
            winSize=(25, 25),
            maxLevel=4,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )

        if self._use_opencl:
            prev_u = cv2.UMat(self.prev_gray)
            curr_u = cv2.UMat(gray_small)
            pts_u = cv2.UMat(prev_pts)
            next_pts, status_fwd, _ = cv2.calcOpticalFlowPyrLK(prev_u, curr_u, pts_u, None, **lk_params)
            back_pts, status_bwd, _ = cv2.calcOpticalFlowPyrLK(curr_u, prev_u, next_pts, None, **lk_params)
            next_pts = next_pts.get() if isinstance(next_pts, cv2.UMat) else next_pts
            back_pts = back_pts.get() if isinstance(back_pts, cv2.UMat) else back_pts
            status_fwd = status_fwd.get() if isinstance(status_fwd, cv2.UMat) else status_fwd
            status_bwd = status_bwd.get() if isinstance(status_bwd, cv2.UMat) else status_bwd
        else:
            next_pts, status_fwd, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray_small, prev_pts, None, **lk_params)
            back_pts, status_bwd, _ = cv2.calcOpticalFlowPyrLK(gray_small, self.prev_gray, next_pts, None, **lk_params)

        fb_err = np.linalg.norm((prev_pts - back_pts).reshape(-1, 2), axis=1)
        good = (status_fwd.ravel() == 1) & (status_bwd.ravel() == 1) & (fb_err < 1.5)
        if np.sum(good) < 8:
            self.prev_gray = gray_small
            return None, None, 0.0

        prev_good = prev_pts[good, 0, :].astype(np.float32)
        next_good = next_pts[good, 0, :].astype(np.float32)

        A, inliers = cv2.estimateAffinePartial2D(
            prev_good,
            next_good,
            method=cv2.RANSAC,
            ransacReprojThreshold=self.flow_reproj_threshold,
            maxIters=500,
            confidence=0.995,
        )

        if A is None:
            self.prev_gray = gray_small
            return None, None, 0.0

        inlier_count = int(np.sum(inliers)) if inliers is not None else 0
        if inlier_count < self.flow_min_inliers:
            self.prev_gray = gray_small
            return None, None, 0.0

        a11 = float(A[0, 0])
        a21 = float(A[1, 0])
        scale = np.sqrt(a11 * a11 + a21 * a21)
        if scale < 0.7 or scale > 1.4:
            self.prev_gray = gray_small
            return None, None, 0.0

        cam_rot_rad = -float(np.arctan2(a21, a11))

        h, w = gray_small.shape[:2]
        cx, cy = w * 0.5, h * 0.5
        mapped_cx = A[0, 0] * cx + A[0, 1] * cy + A[0, 2]
        mapped_cy = A[1, 0] * cx + A[1, 1] * cy + A[1, 2]
        s = self.frame_scale
        cam_dx = -(float(mapped_cx) - cx) / s  # map-pixel translation
        cam_dy = -(float(mapped_cy) - cy) / s

        max_step = self.max_jump * 0.25  # at most 1/4 max_jump per frame
        mag = np.sqrt(cam_dx * cam_dx + cam_dy * cam_dy)
        if mag > max_step:
            cam_dx *= max_step / mag
            cam_dy *= max_step / mag

        inlier_ratio = inlier_count / max(1, len(prev_good))
        flow_conf = min(1.0, inlier_ratio * 1.5) * min(1.0, inlier_count / 40.0)
        self.prev_gray = gray_small
        return np.array([cam_dx, cam_dy], dtype=np.float64), cam_rot_rad, float(flow_conf)

    def _get_roi_features(self):
        if self.kp_map is None or self.des_map is None or self.kp_map_pts is None:
            return None, None

        if self.lost_count >= self.max_lost_before_recovery:
            return self.kp_map, self.des_map

        if self.smooth_pos is not None:
            cx, cy = float(self.smooth_pos[0]), float(self.smooth_pos[1])
        elif self.last_pos is not None:
            cx, cy = float(self.last_pos[0]), float(self.last_pos[1])
        else:
            return self.kp_map, self.des_map

        conf_scale = 1.2 - 0.5 * max(0.0, min(1.0, float(self.last_flow_confidence)))
        lost_scale = 1.0 + 0.30 * min(self.lost_count, 6)
        r = int(self.roi_radius * conf_scale * lost_scale)
        r = max(450, min(r, 3500))

        pts = self.kp_map_pts
        mask = (pts[:, 0] > cx - r) & (pts[:, 0] < cx + r) & (pts[:, 1] > cy - r) & (pts[:, 1] < cy + r)
        indices = np.where(mask)[0]

        if len(indices) < 100:
            r2 = min(int(r * 1.8), 5000)
            mask = (pts[:, 0] > cx - r2) & (pts[:, 0] < cx + r2) & (pts[:, 1] > cy - r2) & (pts[:, 1] < cy + r2)
            indices = np.where(mask)[0]

        if len(indices) < 60:
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

    def _is_outside_loaded_map(self) -> bool:
        if self.last_pos is None:
            return False

        x, y = self.last_pos
        if x < 0 or y < 0 or x >= self.map_w or y >= self.map_h:
            return True

        if self.smooth_footprint is not None:
            pts = self.smooth_footprint.reshape(-1, 2)
            if np.any(pts[:, 0] < 0) or np.any(pts[:, 1] < 0):
                return True
            if np.any(pts[:, 0] >= self.map_w) or np.any(pts[:, 1] >= self.map_h):
                return True

        return False

    def _check_geofence(self) -> tuple[GeofenceStatus, float | None]:
        if self.last_coords is None:
            return GeofenceStatus.LOST, None

        if self._is_outside_loaded_map():
            return GeofenceStatus.VIOLATION, None

        if self._geofence_poly is None:
            return GeofenceStatus.SAFE, None

        pt = Point(self.last_coords)
        dist_m = self._geofence_poly.boundary.distance(pt)

        if self._geofence_poly.contains(pt):
            # Inside the blocked area — violation
            return GeofenceStatus.VIOLATION, -dist_m
        else:
            # Outside — check if close to the boundary
            if dist_m < self.buffer_meters:
                return GeofenceStatus.APPROACHING, dist_m
            return GeofenceStatus.SAFE, dist_m

    def _submit_detection(self, frame: np.ndarray, gray_small: np.ndarray):
        if not self._map_features_ready:
            return
        with self._lock:
            if self._detect_running:
                return
            self._detect_running = True
        h, w = frame.shape[:2]
        self._detect_thread = Thread(target=self._detect_worker, args=(gray_small.copy(), h, w), daemon=True)
        self._detect_thread.start()

    def _detect_worker(self, small_gray: np.ndarray, orig_h: int, orig_w: int):
        h, w = small_gray.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[int(h * 0.10) :, int(w * 0.05) : int(w * 0.95)] = 255

        kp_frame, des_frame = self.det_frame.detectAndCompute(small_gray, mask)
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
        inlier_ratio = (np.sum(mask) / len(mask)) if (M is not None and mask is not None and len(mask) > 0) else 0.0

        if M is None or inlier_ratio < 0.25:
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

            pts = raw_fp.reshape(4, 2)
            raw_heading = float(np.degrees(np.arctan2(pts[3, 1] - pts[0, 1], pts[3, 0] - pts[0, 0])))
            raw_local = self._shape_to_local(pts, raw_pos, raw_heading)

            if self.smooth_pos is None or self.lost_count >= self.max_lost_before_recovery:
                # First detection — initialise from AKAZE
                self._anchor_pos = raw_pos.copy()
                self._flow_accum[:] = 0
                self._heading_anchor = raw_heading
                self._heading_accum = 0.0
                self.smooth_pos = raw_pos.copy()
                self.smooth_heading = raw_heading
                self._fp_local = raw_local
                self.smooth_footprint = self._make_footprint()
            else:
                # AKAZE is authoritative — snap position and heading
                conf = max(0.0, min(1.0, (inlier_ratio - 0.25) / 0.55))
                snap = 0.45 + 0.40 * conf

                # Position: snap anchor to detection, decay flow accumulator
                self._anchor_pos = snap * raw_pos + (1 - snap) * self.smooth_pos
                self._flow_accum *= 1 - snap
                self.smooth_pos = self._anchor_pos + self._flow_accum

                # Heading: snap anchor, decay accumulator (same pattern)
                diff = self._wrap_angle_deg(raw_heading - self.smooth_heading)
                self._heading_anchor += snap * diff
                self._heading_accum *= 1 - snap
                self.smooth_heading = self._heading_anchor + self._heading_accum

                # Footprint shape (slow-adapt) keeps trapezoid under tilt.
                if self._fp_local is None:
                    self._fp_local = raw_local
                else:
                    self._fp_local = 0.25 * raw_local + 0.75 * self._fp_local
                self.smooth_footprint = self._make_footprint()

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
