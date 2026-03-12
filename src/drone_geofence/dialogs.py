import cv2
import numpy as np

from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QLabel,
    QFormLayout,
    QSpinBox,
    QDoubleSpinBox,
    QDialogButtonBox,
)
from PySide6.QtCore import Qt, QPointF
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen, QBrush, QColor

from .engine import TrackingEngine


class CropCenterDialog(QDialog):
    def __init__(self, full_map_color: np.ndarray, full_w: int, full_h: int, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Map Crop Centre")
        self.setMinimumSize(900, 650)

        self._full_w = full_w
        self._full_h = full_h
        self._selected = (full_w // 2, full_h // 2)

        self._preview_w = 1280
        self._ratio = self._preview_w / full_w
        self._preview_h = int(full_h * self._ratio)
        preview_rgb = cv2.resize(full_map_color, (self._preview_w, self._preview_h))
        qimg = QImage(
            preview_rgb.data,
            self._preview_w,
            self._preview_h,
            3 * self._preview_w,
            QImage.Format.Format_RGB888,
        ).copy()
        self._base_pixmap = QPixmap.fromImage(qimg)

        layout = QVBoxLayout(self)

        hint = QLabel(
            "Click to set the crop centre.  Green = recommended (map middle).  "
            "Red = your selection.  Press OK to confirm.",
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #e0e0e0; padding: 4px;")
        layout.addWidget(hint)

        self._preview_label = QLabel()
        self._preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview_label.setMinimumSize(self._preview_w, self._preview_h)
        self._preview_label.mousePressEvent = self._on_click
        layout.addWidget(self._preview_label)

        self._coord_label = QLabel(self._coord_text())
        self._coord_label.setStyleSheet("color: #aaa; padding: 2px;")
        layout.addWidget(self._coord_label)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

        self._redraw()

    def selected_center(self) -> tuple[int, int]:
        return self._selected

    def _coord_text(self) -> str:
        return (
            f"Selected: ({self._selected[0]}, {self._selected[1]})  —  "
            f"recommended: ({self._full_w // 2}, {self._full_h // 2})"
        )

    def _on_click(self, event):
        pos = event.position()
        x, y = pos.x(), pos.y()
        full_x = max(0, min(int(x / self._ratio), self._full_w - 1))
        full_y = max(0, min(int(y / self._ratio), self._full_h - 1))
        self._selected = (full_x, full_y)
        self._coord_label.setText(self._coord_text())
        self._redraw()

    def _redraw(self):
        pixmap = self._base_pixmap.copy()
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        rc_x = int((self._full_w // 2) * self._ratio)
        rc_y = int((self._full_h // 2) * self._ratio)
        painter.setPen(QPen(QColor("#00e676"), 2))
        painter.drawLine(rc_x - 15, rc_y, rc_x + 15, rc_y)
        painter.drawLine(rc_x, rc_y - 15, rc_x, rc_y + 15)
        painter.drawEllipse(QPointF(rc_x, rc_y), 10, 10)

        sx = int(self._selected[0] * self._ratio)
        sy = int(self._selected[1] * self._ratio)
        painter.setPen(QPen(QColor("#ff1744"), 2))
        painter.setBrush(QBrush(QColor(255, 23, 68, 120)))
        painter.drawEllipse(QPointF(sx, sy), 12, 12)
        painter.drawLine(sx - 20, sy, sx + 20, sy)
        painter.drawLine(sx, sy - 20, sx, sy + 20)

        painter.end()
        self._preview_label.setPixmap(pixmap)


class SettingsDialog(QDialog):
    def __init__(self, engine: TrackingEngine, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Tracking Settings")
        self._engine = engine
        layout = QFormLayout(self)

        self.frame_scale = QDoubleSpinBox()
        self.frame_scale.setRange(0.1, 1.0)
        self.frame_scale.setSingleStep(0.05)
        self.frame_scale.setValue(engine.frame_scale)
        layout.addRow("Frame scale:", self.frame_scale)

        self.smoothing = QDoubleSpinBox()
        self.smoothing.setRange(0.05, 1.0)
        self.smoothing.setSingleStep(0.05)
        self.smoothing.setValue(engine.smoothing)
        layout.addRow("EMA smoothing:", self.smoothing)

        self.detect_interval = QSpinBox()
        self.detect_interval.setRange(1, 30)
        self.detect_interval.setValue(engine.detect_interval)
        layout.addRow("Detect every N frames:", self.detect_interval)

        self.roi_radius = QSpinBox()
        self.roi_radius.setRange(200, 5000)
        self.roi_radius.setSingleStep(100)
        self.roi_radius.setValue(engine.roi_radius)
        layout.addRow("ROI radius (px):", self.roi_radius)

        self.max_jump = QSpinBox()
        self.max_jump.setRange(50, 2000)
        self.max_jump.setSingleStep(50)
        self.max_jump.setValue(engine.max_jump)
        layout.addRow("Max jump (px):", self.max_jump)

        self.buffer_meters = QDoubleSpinBox()
        self.buffer_meters.setRange(1.0, 200.0)
        self.buffer_meters.setSingleStep(1.0)
        self.buffer_meters.setValue(engine.buffer_meters)
        layout.addRow("Warning buffer (m):", self.buffer_meters)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def accept(self):
        self._engine.frame_scale = self.frame_scale.value()
        self._engine.smoothing = self.smoothing.value()
        self._engine.detect_interval = self.detect_interval.value()
        self._engine.roi_radius = self.roi_radius.value()
        self._engine.max_jump = self.max_jump.value()
        self._engine.buffer_meters = self.buffer_meters.value()
        super().accept()
