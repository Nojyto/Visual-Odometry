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
    QSizePolicy,
    QToolButton,
    QMenu,
    QInputDialog,
)
from PySide6.QtCore import Qt, QTimer, QThread, Signal, QSettings, QUrl
from PySide6.QtGui import QAction, QColor, QIcon, QDesktopServices

try:
    import qtawesome as qta
except ImportError:
    qta = None

import time
from threading import Lock, Thread

from .engine import TrackingEngine, TrackingResult, GeofenceStatus
from .constants import DRONE_COLORS, MAX_FEEDS
from .widgets import MapWidget, FeedPanel
from .dialogs import CropCenterDialog, SettingsDialog
from .tile_loader import load_tiles_async


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

        self._track_slot = None
        self._track_lock = Lock()

    def run(self):
        tracker = Thread(target=self._tracking_loop, daemon=True)
        tracker.start()

        interval = 1.0 / self.TARGET_FPS
        while self._active:
            t0 = time.perf_counter()
            ret, frame = self._cap.read()
            if not ret:
                self._active = False
                self.finished.emit(self._idx)
                tracker.join(timeout=2.0)
                return

            with self._lock:
                self._latest_frame = frame
                self._new_data = True

            with self._track_lock:
                self._track_slot = frame

            elapsed = time.perf_counter() - t0
            if elapsed < interval:
                time.sleep(interval - elapsed)

    def _tracking_loop(self):
        while self._active:
            with self._track_lock:
                frame = self._track_slot
                if frame is not None:
                    self._track_slot = None

            if frame is None:
                time.sleep(0.005)
                continue

            result = self._engine.process_frame(frame)
            with self._lock:
                self._latest_result = result

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
        self.setMaximumSize(1600, 900)
        self.setStyleSheet(
            """
            QMainWindow { background-color: #0f0f23; }
            QLabel { color: #e0e0e0; }
            QToolBar { background: #16213e; border: none; spacing: 6px; padding: 4px; }
            QToolBar QToolButton {
                color: #e0e0e0;
                background: #1f2a44;
                border: 1px solid #2f3a56;
                border-radius: 4px;
                padding: 4px 8px;
            }
            QToolBar QToolButton:hover:!disabled { background: #283655; }
            QToolBar QToolButton:disabled {
                color: #7f8a9d;
                background: #1a1f2e;
                border: 1px solid #242c3d;
            }
            QToolBar QToolButton#startBtn:!disabled {
                background: #1f6f3f;
                border: 1px solid #2b9a59;
                color: #e9fff1;
                font-weight: bold;
            }
            QToolBar QToolButton#startBtn:hover:!disabled { background: #25864c; }
            QToolBar QToolButton#stopBtn:!disabled {
                background: #7a2830;
                border: 1px solid #b6404c;
                color: #fff1f2;
                font-weight: bold;
            }
            QToolBar QToolButton#stopBtn:hover:!disabled { background: #92303a; }
            QStatusBar { background: #16213e; color: #e0e0e0; }
            """
        )

        self._map_path: str | None = None
        self._engine_template: TrackingEngine | None = None
        self._crop_center: tuple[int, int] | None = None
        self._crop_pad: int = 3000
        self._running = False
        self._settings = QSettings("drone-geofence", "DroneGeofenceMonitor")
        self._status_full_text = "Load a map to begin."

        self._feeds: list[DroneFeedSlot] = [DroneFeedSlot(i) for i in range(MAX_FEEDS)]
        self._prev_alert: dict[int, GeofenceStatus] = {}
        self._tile_worker = None

        self._build_toolbar()
        self._build_central()
        self._build_statusbar()

        self._timer = QTimer(self)
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._tick)

        self._fps_counter = 0
        self._fps_timer_start = 0

        self._try_restore_last_map()

    def _material_icon(self, name: str, fallback: QStyle.StandardPixmap, color: str = "#e0e0e0") -> QIcon:
        if qta is not None:
            try:
                return qta.icon(name, color=color)
            except Exception:
                pass
        return self.style().standardIcon(fallback)

    def _build_toolbar(self):
        tb = QToolBar("Main")
        tb.setMovable(False)
        tb.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.addToolBar(tb)
        SP = QStyle.StandardPixmap

        self._act_load_map = QAction(self._material_icon("mdi6.map-search", SP.SP_DirOpenIcon), "Map", self)
        self._act_load_map.setToolTip("Load orthophoto GeoTIFF")
        self._act_load_map.triggered.connect(self._on_load_map)
        tb.addAction(self._act_load_map)

        tb.addSeparator()

        self._act_start = QAction(self._material_icon("mdi6.play", SP.SP_MediaPlay, color="#e9fff1"), "Start", self)
        self._act_start.triggered.connect(self._on_start)
        self._act_start.setEnabled(False)
        self._btn_start = QToolButton()
        self._btn_start.setObjectName("startBtn")
        self._btn_start.setDefaultAction(self._act_start)
        tb.addWidget(self._btn_start)

        self._act_stop = QAction(self._material_icon("mdi6.stop", SP.SP_MediaStop, color="#fff1f2"), "Stop", self)
        self._act_stop.triggered.connect(self._on_stop)
        self._act_stop.setEnabled(False)
        self._btn_stop = QToolButton()
        self._btn_stop.setObjectName("stopBtn")
        self._btn_stop.setDefaultAction(self._act_stop)
        tb.addWidget(self._btn_stop)

        tb.addSeparator()

        self._act_draw_fence = QAction(
            self._material_icon("mdi6.draw", SP.SP_FileDialogContentsView),
            "Draw",
            self,
        )
        self._act_draw_fence.setToolTip("Start drawing a geofence polygon")
        self._act_draw_fence.triggered.connect(self._on_draw_fence)
        self._act_draw_fence.setEnabled(False)

        self._act_subtract_fence = QAction(
            self._material_icon("mdi6.vector-difference", SP.SP_CommandLink),
            "Subtract",
            self,
        )
        self._act_subtract_fence.setToolTip("Draw a cutout area to subtract from geofence")
        self._act_subtract_fence.triggered.connect(self._on_subtract_fence)
        self._act_subtract_fence.setEnabled(False)

        self._act_undo_fence = QAction(
            self._material_icon("mdi6.undo", SP.SP_ArrowBack),
            "Undo",
            self,
        )
        self._act_undo_fence.setToolTip("Remove the last geofence point while drawing")
        self._act_undo_fence.triggered.connect(self._on_undo_fence_point)
        self._act_undo_fence.setEnabled(False)

        self._act_finish_fence = QAction(
            self._material_icon("mdi6.check", SP.SP_DialogApplyButton),
            "Finish",
            self,
        )
        self._act_finish_fence.setToolTip("Finish geofence polygon (double-click also works)")
        self._act_finish_fence.triggered.connect(self._on_finish_fence)
        self._act_finish_fence.setEnabled(False)

        self._act_cancel_fence = QAction(
            self._material_icon("mdi6.close", SP.SP_DialogCancelButton),
            "Cancel",
            self,
        )
        self._act_cancel_fence.setToolTip("Cancel geofence drawing")
        self._act_cancel_fence.triggered.connect(self._on_cancel_fence)
        self._act_cancel_fence.setEnabled(False)

        self._act_rect_fence = QAction(
            self._material_icon("mdi6.rectangle-outline", SP.SP_FileDialogListView),
            "Rect",
            self,
        )
        self._act_rect_fence.setToolTip("Create a rectangular geofence fitted to the current map")
        self._act_rect_fence.triggered.connect(self._on_rect_fence)
        self._act_rect_fence.setEnabled(False)

        self._act_circle_fence = QAction(
            self._material_icon("mdi6.circle-outline", SP.SP_BrowserReload),
            "Circle",
            self,
        )
        self._act_circle_fence.setToolTip("Create a circular geofence")
        self._act_circle_fence.triggered.connect(self._on_circle_fence)
        self._act_circle_fence.setEnabled(False)

        self._act_cut_circle = QAction(
            self._material_icon("mdi6.chart-donut", SP.SP_BrowserStop),
            "Cut Circle",
            self,
        )
        self._act_cut_circle.setToolTip("Subtract a circular cutout from geofence")
        self._act_cut_circle.triggered.connect(self._on_cut_circle)
        self._act_cut_circle.setEnabled(False)

        self._act_move_shape = QAction(
            self._material_icon("mdi6.cursor-move", SP.SP_ArrowRight),
            "Move Shape",
            self,
        )
        self._act_move_shape.setToolTip("Move the last created shape (outer/cutout)")
        self._act_move_shape.triggered.connect(self._on_move_shape)
        self._act_move_shape.setEnabled(False)

        self._act_resize_shape = QAction(
            self._material_icon("mdi6.resize", SP.SP_ArrowUp),
            "Resize Shape",
            self,
        )
        self._act_resize_shape.setToolTip("Scale the last created shape (outer/cutout)")
        self._act_resize_shape.triggered.connect(self._on_resize_shape)
        self._act_resize_shape.setEnabled(False)

        self._act_clear_fence = QAction(
            self._material_icon("mdi6.delete-outline", SP.SP_DialogResetButton),
            "Clear",
            self,
        )
        self._act_clear_fence.setToolTip("Clear current geofence")
        self._act_clear_fence.triggered.connect(self._on_clear_fence)
        self._act_clear_fence.setEnabled(False)

        self._act_save_fence = QAction(
            self._material_icon("mdi6.content-save", SP.SP_DialogSaveButton),
            "Save",
            self,
        )
        self._act_save_fence.setToolTip("Save geofence to JSON")
        self._act_save_fence.triggered.connect(self._on_save_fence)
        self._act_save_fence.setEnabled(False)

        self._act_load_fence = QAction(
            self._material_icon("mdi6.folder-open", SP.SP_DirOpenIcon),
            "Load",
            self,
        )
        self._act_load_fence.setToolTip("Load geofence from JSON")
        self._act_load_fence.triggered.connect(self._on_load_fence)
        self._act_load_fence.setEnabled(False)

        self._build_fence_toolbox(tb)

        tb.addSeparator()

        self._act_settings = QAction(
            self._material_icon("mdi6.cog", SP.SP_FileDialogDetailedView),
            "Settings",
            self,
        )
        self._act_settings.triggered.connect(self._on_settings)
        self._act_settings.setEnabled(False)
        tb.addAction(self._act_settings)

    def _add_menu_button(
        self,
        toolbar: QToolBar,
        text: str,
        icon: QIcon,
        actions: list[QAction],
        tooltip: str,
    ) -> QToolButton:
        btn = QToolButton()
        btn.setText(text)
        btn.setIcon(icon)
        btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        btn.setToolTip(tooltip)

        menu = QMenu(btn)
        for i, act in enumerate(actions):
            if act is None:
                menu.addSeparator()
            else:
                menu.addAction(act)
        btn.setMenu(menu)
        toolbar.addWidget(btn)
        return btn

    def _build_fence_toolbox(self, tb: QToolBar):
        SP = QStyle.StandardPixmap
        self._btn_fence_mode = self._add_menu_button(
            tb,
            "Fence Mode",
            self._material_icon("mdi6.vector-polyline-edit", SP.SP_FileDialogContentsView),
            [
                self._act_draw_fence,
                self._act_subtract_fence,
                None,
                self._act_undo_fence,
                self._act_finish_fence,
                self._act_cancel_fence,
            ],
            "Start and edit geofence drawing modes",
        )

        self._btn_fence_presets = self._add_menu_button(
            tb,
            "Presets",
            self._material_icon("mdi6.shape-plus", SP.SP_FileDialogListView),
            [
                self._act_rect_fence,
                self._act_circle_fence,
                self._act_cut_circle,
                None,
                self._act_move_shape,
                self._act_resize_shape,
            ],
            "Quick geofence templates and cutouts",
        )

        self._btn_fence_file = self._add_menu_button(
            tb,
            "Fence File",
            self._material_icon("mdi6.file-cog-outline", SP.SP_DialogSaveButton),
            [self._act_save_fence, self._act_load_fence, None, self._act_clear_fence],
            "Save, load, or clear geofence",
        )

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
            panel.info.btn_open_maps.clicked.connect(lambda _=False, idx=i: self._on_open_google_maps(idx))
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
        self._status_label = QLabel(self._status_full_text)
        self._status_label.setWordWrap(False)
        self._status_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        sb.addWidget(self._status_label)
        self._refresh_status_label()

    def _set_status(self, text: str):
        self._status_full_text = text
        self._refresh_status_label()

    def _refresh_status_label(self):
        if not hasattr(self, "_status_label") or self._status_label is None:
            return
        width = max(80, self.statusBar().width() - 20)
        elided = self._status_label.fontMetrics().elidedText(self._status_full_text, Qt.TextElideMode.ElideRight, width)
        self._status_label.setText(elided)
        self._status_label.setToolTip(self._status_full_text)

    def _save_session_state(self):
        if self._map_path:
            self._settings.setValue("last_map_path", self._map_path)
        if self._crop_center:
            self._settings.setValue("last_crop_x", int(self._crop_center[0]))
            self._settings.setValue("last_crop_y", int(self._crop_center[1]))
        self._settings.setValue("last_crop_pad", int(self._crop_pad))

    def _try_restore_last_map(self):
        last_path = self._settings.value("last_map_path", "", str)
        if not last_path:
            return
        if not Path(last_path).exists():
            return
        self._load_map_from_path(last_path, prompt_crop=False)

    def _load_map_from_path(self, path: str, prompt_crop: bool):
        self._set_status("Loading map...")
        QApplication.processEvents()

        engine = TrackingEngine(path)
        info = engine.map_info()
        self._map_path = path

        cx = info.full_w // 2
        cy = info.full_h // 2
        if prompt_crop:
            dlg = CropCenterDialog(engine.full_map_color, info.full_w, info.full_h, self)
            if dlg.exec() != QDialog.DialogCode.Accepted:
                self._set_status("Map load cancelled.")
                return
            cx, cy = dlg.selected_center()
        else:
            sx = self._settings.value("last_crop_x", None)
            sy = self._settings.value("last_crop_y", None)
            if sx is not None and sy is not None:
                try:
                    cx = int(sx)
                    cy = int(sy)
                except (TypeError, ValueError):
                    pass
            saved_pad = self._settings.value("last_crop_pad", self._crop_pad)
            try:
                self._crop_pad = int(saved_pad)
            except (TypeError, ValueError):
                pass

        self._crop_center = (cx, cy)
        engine.crop_to_region(cx, cy, pad=self._crop_pad)
        self._engine_template = engine
        self._feeds[0].engine = engine

        crop_color = engine.get_crop_color()
        self._map_widget.set_map_image(crop_color, resolution=engine.resolution)
        self._map_widget.set_tiles_loading(True)

        # Load OSM tiles as background behind the orthophoto
        self._tile_worker = load_tiles_async(
            engine,
            self._on_tiles_loaded,
        )
        if self._tile_worker is None:
            self._map_widget.set_tiles_loading(False)

        self._set_status(
            f"Map: {Path(path).name} | {info.full_w}x{info.full_h} | CRS: {info.crs} | {info.resolution:.3f} m/px | Crop: ({cx}, {cy})",
        )

        for p in self._feed_panels:
            p.btn_load.setEnabled(True)
        self._set_fence_actions_ready(True)
        self._act_settings.setEnabled(True)
        self._save_session_state()

    def _on_tiles_loaded(self, tiles):
        self._map_widget.set_tile_background(tiles)
        self._map_widget.set_tiles_loading(False)

    def _set_fence_actions_ready(self, enabled: bool):
        self._btn_fence_mode.setEnabled(enabled)
        self._btn_fence_presets.setEnabled(enabled)
        self._btn_fence_file.setEnabled(enabled)
        self._act_draw_fence.setEnabled(enabled)
        self._act_subtract_fence.setEnabled(enabled)
        self._act_rect_fence.setEnabled(enabled)
        self._act_circle_fence.setEnabled(enabled)
        self._act_cut_circle.setEnabled(enabled)
        self._act_move_shape.setEnabled(enabled)
        self._act_resize_shape.setEnabled(enabled)
        self._act_clear_fence.setEnabled(enabled)
        self._act_save_fence.setEnabled(enabled)
        self._act_load_fence.setEnabled(enabled)

    def _set_fence_edit_mode(self, editing: bool):
        self._act_draw_fence.setEnabled(not editing)
        self._act_subtract_fence.setEnabled(not editing)

        self._act_undo_fence.setEnabled(editing)
        self._act_finish_fence.setEnabled(editing)
        self._act_cancel_fence.setEnabled(editing)

        presets_allowed = (not editing) or self._map_widget.is_subtract_mode()
        self._act_rect_fence.setEnabled(presets_allowed)
        self._act_circle_fence.setEnabled(presets_allowed)
        self._act_cut_circle.setEnabled(presets_allowed)
        self._act_move_shape.setEnabled(presets_allowed)
        self._act_resize_shape.setEnabled(presets_allowed)
        self._act_clear_fence.setEnabled(not editing)
        self._act_save_fence.setEnabled(not editing)
        self._act_load_fence.setEnabled(not editing)

        self._btn_fence_mode.setEnabled(True)
        self._btn_fence_presets.setEnabled(presets_allowed)
        self._btn_fence_file.setEnabled(not editing)

    def _on_load_map(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Orthophoto",
            "",
            "GeoTIFF (*.tif *.tiff);;All (*)",
        )
        if not path:
            return
        self._load_map_from_path(path, prompt_crop=True)

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

        ok, preview = cap.read()
        if ok:
            self._feed_panels[idx].fpv.update_frame(preview, GeofenceStatus.INITIALIZING)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        else:
            self._feed_panels[idx].fpv.show_loaded_ready(Path(path).name)

        if feed.engine is None and self._engine_template:
            eng = TrackingEngine(self._map_path)
            cx, cy = self._crop_center
            eng.crop_to_region(cx, cy, pad=self._crop_pad)
            t = self._engine_template
            if t.has_geofence():
                outer, holes = t.get_geofence_definition_pixels()
                eng.set_geofence_pixels(outer or [], holes)
            feed.engine = eng

        self._act_start.setEnabled(True)
        self._feed_panels[idx].btn_clear.setEnabled(True)
        c = DRONE_COLORS[idx]
        self._set_status(f"Drone {idx + 1} ({c['name']}) video loaded: {Path(path).name}. Ready to start.")

    def _on_clear_feed(self, idx: int):
        feed = self._feeds[idx]
        feed.release()
        self._feed_panels[idx].fpv.reset_display()
        self._feed_panels[idx].btn_clear.setEnabled(False)
        self._map_widget.hide_drone(idx)
        self._set_status(f"Feed {idx + 1} cleared.")

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
        self._map_widget.start_drawing("add")
        self._set_fence_edit_mode(True)
        self._set_status(
            "Drawing geofence: left-click add point, right-click undo, double-click finish.",
        )

    def _on_subtract_fence(self):
        if not self._engine_template or not self._engine_template.has_geofence():
            self._set_status("Create or load a geofence before subtracting from it.")
            return
        self._map_widget.start_drawing("subtract")
        self._set_fence_edit_mode(True)
        self._set_status("Subtract mode: draw a cutout polygon. Right-click undo, double-click finish.")

    def _on_undo_fence_point(self):
        if self._map_widget.undo_last_point():
            self._set_status(f"Drawing geofence: {self._map_widget.point_count()} point(s).")
        else:
            self._set_status("Drawing geofence: no points to undo.")

    def _on_finish_fence(self):
        if self._map_widget.finish_drawing():
            self._set_fence_edit_mode(False)
            self._set_status("Geofence set.")
        else:
            self._set_status("Need at least 3 points to finish geofence.")

    def _on_cancel_fence(self):
        self._map_widget.cancel_drawing()
        self._set_fence_edit_mode(False)
        self._set_status("Drawing cancelled.")

    def _on_rect_fence(self):
        if self._map_widget.is_subtract_mode():
            if self._map_widget.subtract_rectangle_cutout():
                self._set_status("Rectangle cutout subtracted from geofence.")
            else:
                self._set_status("Create or load a geofence before subtracting with Rect.")
            return
        self._map_widget.set_rectangle_fence()
        self._set_status("Rectangle geofence created.")

    def _on_circle_fence(self):
        if self._map_widget.is_subtract_mode():
            if self._map_widget.subtract_circle_cutout():
                self._set_status("Circle cutout subtracted from geofence.")
            else:
                self._set_status("Create or load a geofence before subtracting with Circle.")
            return
        self._map_widget.set_circle_fence()
        self._set_status("Circle geofence created.")

    def _on_cut_circle(self):
        if self._map_widget.subtract_circle_cutout():
            self._set_status("Circular cutout subtracted from geofence.")
        else:
            self._set_status("Create or load a geofence before using Cut Circle.")

    def _on_move_shape(self):
        dx, ok_dx = QInputDialog.getDouble(self, "Move Shape", "Offset X (px):", 0.0, -10000.0, 10000.0, 1)
        if not ok_dx:
            return
        dy, ok_dy = QInputDialog.getDouble(self, "Move Shape", "Offset Y (px):", 0.0, -10000.0, 10000.0, 1)
        if not ok_dy:
            return
        if self._map_widget.move_active_shape(dx, dy):
            self._set_status(f"Shape moved by ({dx:.1f}, {dy:.1f}) px.")
        else:
            self._set_status("No shape available to move.")

    def _on_resize_shape(self):
        factor, ok = QInputDialog.getDouble(self, "Resize Shape", "Scale factor:", 1.1, 0.1, 10.0, 2)
        if not ok:
            return
        if self._map_widget.scale_active_shape(factor):
            self._set_status(f"Shape scaled by {factor:.2f}x.")
        else:
            self._set_status("No shape available to resize.")

    def _on_geofence_drawn(self, geometry: dict):
        pixel_coords = geometry.get("outer", [])
        hole_pixels = geometry.get("holes", [])
        for feed in self._feeds:
            if feed.engine:
                feed.engine.set_geofence_pixels(pixel_coords, hole_pixels)

    def _on_clear_fence(self):
        self._set_fence_edit_mode(False)
        for feed in self._feeds:
            if feed.engine:
                feed.engine.set_geofence([])
        self._map_widget.clear_fence()
        self._set_status("Geofence cleared.")

    def _on_save_fence(self):
        engine = self._engine_template
        if not engine or not engine.has_geofence():
            QMessageBox.information(self, "No Geofence", "Draw a geofence first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save Geofence", "", "JSON (*.json)")
        if not path:
            return
        coords = engine.get_geofence_pixel_coords()
        holes = engine.get_geofence_holes_pixel_coords()
        data = {
            "version": 1,
            "crs": str(engine.crs),
            "crop_x": engine.crop_x,
            "crop_y": engine.crop_y,
            "buffer_meters": engine.buffer_meters,
            "vertices_pixel": [(x, y) for x, y in coords],
            "holes_pixel": [[(x, y) for x, y in hole] for hole in holes],
            "vertices_utm": [engine.pixel_to_meters(x, y) for x, y in coords],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        self._set_status(f"Geofence saved to {Path(path).name}")

    def _on_load_fence(self):
        if not self._engine_template:
            return
        self._set_fence_edit_mode(False)
        path, _ = QFileDialog.getOpenFileName(self, "Load Geofence", "", "JSON (*.json)")
        if not path:
            return
        with open(path, "r") as f:
            data = json.load(f)

        vertices = data.get("vertices_pixel", [])
        holes = data.get("holes_pixel", [])
        if len(vertices) < 3:
            QMessageBox.warning(self, "Invalid", "Geofence needs >= 3 vertices.")
            return

        pixel_coords = [(int(x), int(y)) for x, y in vertices]
        hole_pixels = [[(int(x), int(y)) for x, y in hole] for hole in holes if len(hole) >= 3]
        for feed in self._feeds:
            if feed.engine:
                feed.engine.set_geofence_pixels(pixel_coords, hole_pixels)
        self._map_widget.set_fence_polygon(pixel_coords, hole_pixels)
        self._set_status(f"Geofence loaded: {len(pixel_coords)} vertices, {len(hole_pixels)} cutout(s)")

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
            if frame is None or result is None:
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
            self._set_status("All feeds ended.")

    def _to_google_maps_url(self, idx: int) -> str | None:
        feed = self._feeds[idx]
        if feed.engine is None:
            return None
        coords = feed.last_result.coords if feed.last_result else None
        if coords is None:
            return None
        latlon = feed.engine.coords_to_latlon(coords[0], coords[1])
        if latlon is None:
            return None
        lat, lon = latlon
        return f"https://www.google.com/maps?q={lat:.8f},{lon:.8f}"

    def _on_open_google_maps(self, idx: int):
        url = self._to_google_maps_url(idx)
        if not url:
            self._set_status(f"Drone {idx + 1}: no coordinate available to open.")
            return
        QDesktopServices.openUrl(QUrl(url))
        self._set_status(f"Drone {idx + 1}: opening Google Maps.")

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

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh_status_label()

    def closeEvent(self, event):
        self._save_session_state()
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
