import math
import urllib.request
from dataclasses import dataclass

from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage

TILE_SIZE = 256
USER_AGENT = "DroneGeofenceMonitor/1.0"
TILE_URL = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
MAX_TILES = 64


@dataclass
class PositionedTile:
    """A fetched tile with its scene-space position."""

    image: QImage
    scene_x: float
    scene_y: float
    scene_w: float
    scene_h: float


def _latlon_to_tile(lat: float, lon: float, zoom: int) -> tuple[int, int]:
    n = 2**zoom
    x = int((lon + 180.0) / 360.0 * n)
    lat_rad = math.radians(lat)
    y = int((1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi) / 2.0 * n)
    x = max(0, min(n - 1, x))
    y = max(0, min(n - 1, y))
    return x, y


def _tile_to_latlon(tx: int, ty: int, zoom: int) -> tuple[float, float]:
    n = 2**zoom
    lon = tx / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ty / n)))
    lat = math.degrees(lat_rad)
    return lat, lon


def _choose_zoom(lat_span: float, lon_span: float) -> int:
    for z in range(17, 0, -1):
        n = 2**z
        tile_lon_span = 360.0 / n
        tiles_x = lon_span / tile_lon_span
        tiles_y = lat_span / tile_lon_span
        if tiles_x * tiles_y <= MAX_TILES:
            return z
    return 1


class TileLoaderThread(QThread):
    tiles_ready = Signal(list)

    def __init__(
        self,
        center_latlon: tuple[float, float],
        center_scene: tuple[float, float],
        px_per_deg_lon: float,
        px_per_deg_lat: float,
        bounds_latlon: tuple[float, float, float, float],
        parent=None,
    ):
        super().__init__(parent)
        self._center_lat, self._center_lon = center_latlon
        self._center_sx, self._center_sy = center_scene
        self._px_per_deg_lon = px_per_deg_lon
        self._px_per_deg_lat = px_per_deg_lat
        self._bounds = bounds_latlon

    def run(self):
        min_lat, max_lat, min_lon, max_lon = self._bounds

        lat_pad = (max_lat - min_lat) * 0.25
        lon_pad = (max_lon - min_lon) * 0.25
        min_lat -= lat_pad
        max_lat += lat_pad
        min_lon -= lon_pad
        max_lon += lon_pad

        zoom = _choose_zoom(max_lat - min_lat, max_lon - min_lon)

        tx_min, ty_max = _latlon_to_tile(min_lat, min_lon, zoom)
        tx_max, ty_min = _latlon_to_tile(max_lat, max_lon, zoom)
        if tx_min > tx_max:
            tx_min, tx_max = tx_max, tx_min
        if ty_min > ty_max:
            ty_min, ty_max = ty_max, ty_min

        results: list[PositionedTile] = []
        for ty in range(ty_min, ty_max + 1):
            for tx in range(tx_min, tx_max + 1):
                tile = self._fetch_tile(tx, ty, zoom)
                if tile:
                    results.append(tile)

        self.tiles_ready.emit(results)

    def _fetch_tile(self, tx: int, ty: int, zoom: int) -> PositionedTile | None:
        url = TILE_URL.format(z=zoom, x=tx, y=ty)
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = resp.read()
        except Exception:
            return None

        qimg = QImage()
        if not qimg.loadFromData(data):
            return None

        nw_lat, nw_lon = _tile_to_latlon(tx, ty, zoom)
        se_lat, se_lon = _tile_to_latlon(tx + 1, ty + 1, zoom)

        scene_x = self._center_sx + (nw_lon - self._center_lon) * self._px_per_deg_lon
        scene_y = self._center_sy - (nw_lat - self._center_lat) * self._px_per_deg_lat
        scene_w = (se_lon - nw_lon) * self._px_per_deg_lon
        scene_h = (nw_lat - se_lat) * self._px_per_deg_lat

        return PositionedTile(
            image=qimg,
            scene_x=scene_x,
            scene_y=scene_y,
            scene_w=scene_w,
            scene_h=scene_h,
        )


def load_tiles_async(engine, callback) -> TileLoaderThread | None:
    corners_px = [
        (0, 0),
        (engine.map_w, 0),
        (engine.map_w, engine.map_h),
        (0, engine.map_h),
    ]
    latitudes, longitudes = [], []
    for px, py in corners_px:
        easting, northing = engine.pixel_to_meters(px, py)
        ll = engine.coords_to_latlon(easting, northing)
        if ll is None:
            return None
        latitudes.append(ll[0])
        longitudes.append(ll[1])

    min_lat, max_lat = min(latitudes), max(latitudes)
    min_lon, max_lon = min(longitudes), max(longitudes)

    center_lat = (min_lat + max_lat) / 2
    center_lon = (min_lon + max_lon) / 2
    center_scene_x = engine.map_w / 2.0
    center_scene_y = engine.map_h / 2.0

    res_x = abs(engine.geo_transform.a)
    res_y = abs(engine.geo_transform.e)
    meters_per_deg_lon = 111_320 * math.cos(math.radians(center_lat))
    meters_per_deg_lat = 110_540
    px_per_deg_lon = meters_per_deg_lon / res_x
    px_per_deg_lat = meters_per_deg_lat / res_y

    thread = TileLoaderThread(
        center_latlon=(center_lat, center_lon),
        center_scene=(center_scene_x, center_scene_y),
        px_per_deg_lon=px_per_deg_lon,
        px_per_deg_lat=px_per_deg_lat,
        bounds_latlon=(min_lat, max_lat, min_lon, max_lon),
    )
    thread.tiles_ready.connect(callback)
    thread.start()
    return thread
