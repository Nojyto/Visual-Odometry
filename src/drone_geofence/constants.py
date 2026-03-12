import cv2
import numpy as np
from PySide6.QtGui import QImage, QPixmap

from .engine import GeofenceStatus

DRONE_COLORS = [
    {"name": "Cyan", "hex": "#00e5ff", "rgb": (0, 229, 255)},
    {"name": "Magenta", "hex": "#ff4081", "rgb": (255, 64, 129)},
    {"name": "Lime", "hex": "#76ff03", "rgb": (118, 255, 3)},
]

STATUS_COLORS = {
    GeofenceStatus.INITIALIZING: "#888888",
    GeofenceStatus.SAFE: "#00e676",
    GeofenceStatus.APPROACHING: "#ffab00",
    GeofenceStatus.VIOLATION: "#ff1744",
    GeofenceStatus.LOST: "#ff6e40",
}

STATUS_BG = {
    GeofenceStatus.INITIALIZING: "rgba(80,80,80,0.15)",
    GeofenceStatus.SAFE: "rgba(0,230,118,0.10)",
    GeofenceStatus.APPROACHING: "rgba(255,171,0,0.15)",
    GeofenceStatus.VIOLATION: "rgba(255,23,68,0.20)",
    GeofenceStatus.LOST: "rgba(255,110,64,0.15)",
}

MAX_FEEDS = 3


def cv_to_qimage(cv_img: np.ndarray) -> QImage:
    rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    return QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()


def cv_to_qpixmap(cv_img: np.ndarray) -> QPixmap:
    return QPixmap.fromImage(cv_to_qimage(cv_img))
