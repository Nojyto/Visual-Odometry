import sys
import json
import cv2
from pathlib import Path

from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QLabel,
    QToolBar,
    QStatusBar,
    QFileDialog,
    QDialog,
    QMessageBox,
    QStyle,
)
from PySide6.QtCore import Qt, QPointF, QTimer, QThread, Signal
from PySide6.QtGui import QAction, QColor, QIcon

import time
from threading import Lock

from .engine import TrackingEngine, TrackingResult, GeofenceStatus
from .constants import DRONE_COLORS, MAX_FEEDS
from .widgets import MapWidget, FeedPanel
from .dialogs import CropCenterDialog, SettingsDialog


class FeedWorker(QThread):
    finished = Signal(int)

    TARGET_FPS = 30

    def __init__(self, index: int, cap: cv2.VideoCapture, engine: TrackingEngine):
        super().__init__()
        self._idx = index
        self._cap = cap
        self._engine = engine
        self._active = True

        self._lock = Lock()
        self._latest_frame = None
        self._latest_result: TrackingResult | None = None
        self._new_data = False

    def run(self):
        interval = 1.0 / self.TARGET_FPS
        while self._active:
            t0 = time.perf_counter()
            ret, frame = self._cap.read()
            if not ret:
                self.finished.emit(self._idx)
                return
            result = self._engine.process_frame(frame)
            with self._lock:
                self._latest_frame = frame
                self._latest_result = result
                self._new_data = True

            elapsed = time.perf_counter() - t0
            if elapsed < interval:
                time.sleep(interval - elapsed)

    def stop(self):
        self._active = False

    def take_latest(self):
        with self._lock:
            if not self._new_data:
                return None, None
            self._new_data = False
            return self._latest_frame, self._latest_result


class DroneFeedSlot:
    def __init__(self, index: int):
        self.index = index
        self.engine: TrackingEngine | None = None
        self.cap: cv2.VideoCapture | None = None
        self.active = False
        self.last_result = TrackingResult()
        self.worker: FeedWorker | None = None

    def release(self):
        if self.worker:
            self.worker.stop()
            self.worker.wait(2000)
            self.worker = None
        if self.cap:
            self.cap.release()
            self.cap = None
        self.active = False


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Drone Geofence Monitor")
        self.resize(1500, 900)
        self.setStyleSheet(
            """
            QMainWindow { background-color: #0f0f23; }
            QLabel { color: #e0e0e0; }
            QToolBar { background: #16213e; border: none; spacing: 6px; padding: 4px; }
            QToolBar QToolButton { color: #e0e0e0; }
            QStatusBar { background: #16213e; color: #e0e0e0; }
            """
        )

        self._map_path: str | None = None
        self._engine_template: TrackingEngine | None = None
        self._crop_center: tuple[int, int] | None = None
        self._crop_pad: int = 3000
        self._running = False

        self._feeds: list[DroneFeedSlot] = [DroneFeedSlot(i) for i in range(MAX_FEEDS)]
        self._prev_alert: dict[int, GeofenceStatus] = {}

        self._build_toolbar()
        self._build_central()
        self._build_statusbar()

        self._timer = QTimer(self)
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._tick)

        self._fps_counter = 0
        self._fps_timer_start = 0

    def _build_toolbar(self):
        tb = QToolBar("Main")
        tb.setMovable(False)
        tb.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.addToolBar(tb)
        SP = QStyle.StandardPixmap
        style = self.style()

        self._act_load_map = QAction(style.standardIcon(SP.SP_DirOpenIcon), "Map", self)
        self._act_load_map.setToolTip("Load orthophoto GeoTIFF")
        self._act_load_map.triggered.connect(self._on_load_map)
        tb.addAction(self._act_load_map)

        tb.addSeparator()

        self._act_start = QAction(style.standardIcon(SP.SP_MediaPlay), "Start", self)
        self._act_start.triggered.connect(self._on_start)
        self._act_start.setEnabled(False)
        tb.addAction(self._act_start)

        self._act_stop = QAction(style.standardIcon(SP.SP_MediaStop), "Stop", self)
        self._act_stop.triggered.connect(self._on_stop)
        self._act_stop.setEnabled(False)
        tb.addAction(self._act_stop)

        tb.addSeparator()

        self._act_draw_fence = QAction(
            style.standardIcon(SP.SP_FileDialogContentsView),
            "Draw Fence",
            self,
        )
        self._act_draw_fence.triggered.connect(self._on_draw_fence)
        self._act_draw_fence.setEnabled(False)
        tb.addAction(self._act_draw_fence)

        self._act_finish_fence = QAction(
            style.standardIcon(SP.SP_DialogApplyButton),
            "Finish",
            self,
        )
        self._act_finish_fence.triggered.connect(self._on_finish_fence)
        self._act_finish_fence.setEnabled(False)
        self._act_finish_fence.setVisible(False)
        tb.addAction(self._act_finish_fence)

        self._act_cancel_fence = QAction(
            style.standardIcon(SP.SP_DialogCancelButton),
            "Cancel",
            self,
        )
        self._act_cancel_fence.triggered.connect(self._on_cancel_fence)
        self._act_cancel_fence.setEnabled(False)
        self._act_cancel_fence.setVisible(False)
        tb.addAction(self._act_cancel_fence)

        self._act_clear_fence = QAction(
            style.standardIcon(SP.SP_DialogResetButton),
            "Clear Fence",
            self,
        )
        self._act_clear_fence.triggered.connect(self._on_clear_fence)
        self._act_clear_fence.setEnabled(False)
        tb.addAction(self._act_clear_fence)

        tb.addSeparator()

        self._act_save_fence = QAction(
            style.standardIcon(SP.SP_DialogSaveButton),
            "Save Fence",
            self,
        )
        self._act_save_fence.triggered.connect(self._on_save_fence)
        self._act_save_fence.setEnabled(False)
        tb.addAction(self._act_save_fence)

        self._act_load_fence = QAction(
            style.standardIcon(SP.SP_DirOpenIcon),
            "Load Fence",
            self,
        )
        self._act_load_fence.triggered.connect(self._on_load_fence)
        self._act_load_fence.setEnabled(False)
        tb.addAction(self._act_load_fence)

        tb.addSeparator()

        self._act_settings = QAction(
            style.standardIcon(SP.SP_FileDialogDetailedView),
            "Settings",
            self,
        )
        self._act_settings.triggered.connect(self._on_settings)
        self._act_settings.setEnabled(False)
        tb.addAction(self._act_settings)

    def _build_central(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        layout.setContentsMargins(4, 4, 4, 4)

        # Left: map
        self._map_widget = MapWidget()
        layout.addWidget(self._map_widget, stretch=3)

        # Right: feed panels stacked
        right = QVBoxLayout()
        right.setSpacing(4)

        self._feed_panels: list[FeedPanel] = []
        for i in range(MAX_FEEDS):
            panel = FeedPanel(i)
            panel.load_requested.connect(self._on_load_feed)
            panel.clear_requested.connect(self._on_clear_feed)
            right.addWidget(panel, stretch=1)
            self._feed_panels.append(panel)

        self._lbl_fps = QLabel("FPS: —")
        self._lbl_fps.setStyleSheet("color: #888; padding: 2px;")
        right.addWidget(self._lbl_fps)

        layout.addLayout(right, stretch=1)

        self._map_widget.geofence_changed.connect(self._on_geofence_drawn)

    def _build_statusbar(self):
        sb = QStatusBar()
        self.setStatusBar(sb)
        self._status_label = QLabel("Load a map to begin.")
        sb.addWidget(self._status_label)

    def _on_load_map(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Orthophoto",
            "",
            "GeoTIFF (*.tif *.tiff);;All (*)",
        )
        if not path:
            return

        self._status_label.setText("Loading map...")
        QApplication.processEvents()

        engine = TrackingEngine(path)
        info = engine.map_info()
        self._map_path = path

        dlg = CropCenterDialog(engine.full_map_color, info.full_w, info.full_h, self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            self._status_label.setText("Map load cancelled.")
            return

        cx, cy = dlg.selected_center()
        self._crop_center = (cx, cy)
        engine.crop_to_region(cx, cy, pad=self._crop_pad)
        self._engine_template = engine

        crop_color = engine.get_crop_color()
        self._map_widget.set_map_image(crop_color)
        self._feeds[0].engine = engine

        self._status_label.setText(
            f"Map: {info.full_w}x{info.full_h} | CRS: {info.crs} | {info.resolution:.3f} m/px | Crop: ({cx}, {cy})",
        )

        for p in self._feed_panels:
            p.btn_load.setEnabled(True)
        self._act_draw_fence.setEnabled(True)
        self._act_clear_fence.setEnabled(True)
        self._act_save_fence.setEnabled(True)
        self._act_load_fence.setEnabled(True)
        self._act_settings.setEnabled(True)

    def _on_load_feed(self, idx: int):
        path, _ = QFileDialog.getOpenFileName(
            self,
            f"Select Video for Drone {idx + 1}",
            "",
            "Video (*.mp4 *.avi *.mov *.mkv);;All (*)",
        )
        if not path:
            return

        feed = self._feeds[idx]
        feed.release()

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            QMessageBox.critical(self, "Error", f"Cannot open video: {Path(path).name}")
            return
        feed.cap = cap

        if feed.engine is None and self._engine_template:
            eng = TrackingEngine(self._map_path)
            cx, cy = self._crop_center
            eng.crop_to_region(cx, cy, pad=self._crop_pad)
            t = self._engine_template
            if t.has_geofence():
                coords = t.get_geofence_pixel_coords()
                eng.set_geofence_pixels(coords)
            feed.engine = eng

        self._act_start.setEnabled(True)
        self._feed_panels[idx].btn_clear.setEnabled(True)
        c = DRONE_COLORS[idx]
        self._status_label.setText(f"Drone {idx + 1} ({c['name']}): {Path(path).name}")

    def _on_clear_feed(self, idx: int):
        feed = self._feeds[idx]
        feed.release()
        self._feed_panels[idx].fpv.reset_display()
        self._feed_panels[idx].btn_clear.setEnabled(False)
        self._map_widget.hide_drone(idx)
        self._status_label.setText(f"Feed {idx + 1} cleared.")

    def _on_start(self):
        for feed in self._feeds:
            if feed.worker:
                feed.worker.stop()
        for feed in self._feeds:
            if feed.worker:
                feed.worker.wait(2000)
                feed.worker = None

        any_active = False
        for feed in self._feeds:
            if feed.cap and feed.engine:
                feed.active = True
                worker = FeedWorker(feed.index, feed.cap, feed.engine)
                worker.finished.connect(self._on_feed_ended)
                feed.worker = worker
                any_active = True

        if not any_active:
            QMessageBox.information(self, "No feeds", "Load at least one video feed first.")
            return

        self._running = True
        self._act_start.setEnabled(False)
        self._act_stop.setEnabled(True)
        self._fps_counter = 0
        self._fps_timer_start = cv2.getTickCount()
        for feed in self._feeds:
            if feed.worker:
                feed.worker.start()
        self._timer.start()

    def _on_stop(self):
        self._running = False
        self._timer.stop()
        for feed in self._feeds:
            if feed.worker:
                feed.worker.stop()
        for feed in self._feeds:
            if feed.worker:
                feed.worker.wait(2000)
                feed.worker = None
            feed.active = False
        self._act_start.setEnabled(True)
        self._act_stop.setEnabled(False)

    def _on_draw_fence(self):
        self._map_widget.start_drawing()
        self._act_draw_fence.setEnabled(False)
        self._act_draw_fence.setVisible(False)
        self._act_finish_fence.setEnabled(True)
        self._act_finish_fence.setVisible(True)
        self._act_cancel_fence.setEnabled(True)
        self._act_cancel_fence.setVisible(True)
        self._status_label.setText(
            "Click on the map to place geofence vertices.  Click 'Finish' when done.",
        )

    def _on_finish_fence(self):
        self._map_widget.finish_drawing()
        self._act_draw_fence.setEnabled(True)
        self._act_draw_fence.setVisible(True)
        self._act_finish_fence.setEnabled(False)
        self._act_finish_fence.setVisible(False)
        self._act_cancel_fence.setEnabled(False)
        self._act_cancel_fence.setVisible(False)
        self._status_label.setText("Geofence set.")

    def _on_cancel_fence(self):
        self._map_widget.cancel_drawing()
        self._act_draw_fence.setEnabled(True)
        self._act_draw_fence.setVisible(True)
        self._act_finish_fence.setEnabled(False)
        self._act_finish_fence.setVisible(False)
        self._act_cancel_fence.setEnabled(False)
        self._act_cancel_fence.setVisible(False)
        self._status_label.setText("Drawing cancelled.")

    def _on_geofence_drawn(self, points: list[QPointF]):
        pixel_coords = [(int(p.x()), int(p.y())) for p in points]
        for feed in self._feeds:
            if feed.engine:
                feed.engine.set_geofence_pixels(pixel_coords)

    def _on_clear_fence(self):
        for feed in self._feeds:
            if feed.engine:
                feed.engine.set_geofence([])
        self._map_widget.clear_fence()
        self._status_label.setText("Geofence cleared.")

    def _on_save_fence(self):
        engine = self._engine_template
        if not engine or not engine.has_geofence():
            QMessageBox.information(self, "No Geofence", "Draw a geofence first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save Geofence", "", "JSON (*.json)")
        if not path:
            return
        coords = engine.get_geofence_pixel_coords()
        data = {
            "version": 1,
            "crs": str(engine.crs),
            "crop_x": engine.crop_x,
            "crop_y": engine.crop_y,
            "buffer_meters": engine.buffer_meters,
            "vertices_pixel": [(x, y) for x, y in coords],
            "vertices_utm": [engine.pixel_to_meters(x, y) for x, y in coords],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        self._status_label.setText(f"Geofence saved to {Path(path).name}")

    def _on_load_fence(self):
        if not self._engine_template:
            return
        path, _ = QFileDialog.getOpenFileName(self, "Load Geofence", "", "JSON (*.json)")
        if not path:
            return
        with open(path, "r") as f:
            data = json.load(f)

        vertices = data.get("vertices_pixel", [])
        if len(vertices) < 3:
            QMessageBox.warning(self, "Invalid", "Geofence needs >= 3 vertices.")
            return

        pixel_coords = [(int(x), int(y)) for x, y in vertices]
        for feed in self._feeds:
            if feed.engine:
                feed.engine.set_geofence_pixels(pixel_coords)
        self._map_widget.set_fence_polygon(pixel_coords)
        self._status_label.setText(f"Geofence loaded: {len(pixel_coords)} vertices")

    def _on_settings(self):
        engine = self._engine_template
        if not engine:
            return
        dlg = SettingsDialog(engine, self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            for feed in self._feeds:
                if feed.engine and feed.engine is not engine:
                    feed.engine.frame_scale = engine.frame_scale
                    feed.engine.smoothing = engine.smoothing
                    feed.engine.detect_interval = engine.detect_interval
                    feed.engine.roi_radius = engine.roi_radius
                    feed.engine.max_jump = engine.max_jump
                    feed.engine.buffer_meters = engine.buffer_meters

    def _tick(self):
        if not self._running:
            return

        for feed in self._feeds:
            if not feed.active or not feed.worker:
                continue
            frame, result = feed.worker.take_latest()
            if frame is None:
                continue
            feed.last_result = result
            self._map_widget.update_drone(feed.index, result.pos, result.footprint, result.status)
            self._feed_panels[feed.index].fpv.update_frame(frame, result.status)
            self._feed_panels[feed.index].info.update_result(result)
            self._check_alerts(feed.index, result)

        self._fps_counter += 1
        elapsed = (cv2.getTickCount() - self._fps_timer_start) / cv2.getTickFrequency()
        if elapsed >= 1.0:
            fps = self._fps_counter / elapsed
            self._lbl_fps.setText(f"FPS: {fps:.1f}")
            self._fps_counter = 0
            self._fps_timer_start = cv2.getTickCount()

    def _on_feed_ended(self, idx: int):
        feed = self._feeds[idx]
        feed.active = False
        if feed.worker:
            feed.worker.wait(2000)
            feed.worker = None
        if not any(f.active for f in self._feeds):
            self._on_stop()
            self._status_label.setText("All feeds ended.")

    def _check_alerts(self, idx: int, result: TrackingResult):
        prev = self._prev_alert.get(idx)
        if result.status == prev:
            return
        self._prev_alert[idx] = result.status
        c = DRONE_COLORS[idx]

        if result.status == GeofenceStatus.VIOLATION:
            self.statusBar().showMessage(
                f"VIOLATION — Drone {idx + 1} ({c['name']}) inside blocked area!",
                5000,
            )
            QApplication.beep()
        elif result.status == GeofenceStatus.APPROACHING:
            self.statusBar().showMessage(
                f"WARNING — Drone {idx + 1} ({c['name']}) approaching blocked area",
                3000,
            )

    def closeEvent(self, event):
        self._running = False
        self._timer.stop()
        for feed in self._feeds:
            if feed.worker:
                feed.worker.stop()
        for feed in self._feeds:
            feed.release()
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    from PySide6.QtGui import QPalette

    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(15, 15, 35))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(224, 224, 224))
    palette.setColor(QPalette.ColorRole.Base, QColor(22, 33, 62))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(30, 40, 70))
    palette.setColor(QPalette.ColorRole.Text, QColor(224, 224, 224))
    palette.setColor(QPalette.ColorRole.Button, QColor(22, 33, 62))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(224, 224, 224))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(52, 152, 219))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
    app.setPalette(palette)

    icon_path = Path(__file__).parent / "assets" / "icon.ico"
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
