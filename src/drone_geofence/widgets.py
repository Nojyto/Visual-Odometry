import numpy as np

from PySide6.QtWidgets import (
    QApplication,
    QGraphicsView,
    QGraphicsScene,
    QGraphicsPixmapItem,
    QGraphicsPolygonItem,
    QGraphicsPathItem,
    QGraphicsEllipseItem,
    QGraphicsItem,
    QLabel,
    QGroupBox,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QToolButton,
    QSizePolicy,
    QStyle,
)
from PySide6.QtCore import Qt, QPointF, QRectF, Signal, QSize
from PySide6.QtGui import (
    QPolygonF,
    QPen,
    QBrush,
    QColor,
    QPainter,
    QWheelEvent,
    QMouseEvent,
    QKeyEvent,
    QFont,
    QIcon,
    QPainterPath,
)

try:
    import qtawesome as qta
except ImportError:
    qta = None

from .engine import GeofenceStatus, TrackingResult
from .constants import (
    DRONE_COLORS,
    STATUS_COLORS,
    STATUS_BG,
    cv_to_qpixmap,
)


def _material_icon(name: str, fallback: QStyle.StandardPixmap, color: str = "#e0e0e0") -> QIcon:
    if qta is not None:
        try:
            return qta.icon(name, color=color)
        except Exception:
            pass
    return QApplication.style().standardIcon(fallback)


class MapWidget(QGraphicsView):
    geofence_changed = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setMinimumSize(600, 400)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setBackgroundBrush(QBrush(QColor(15, 15, 35)))

        self._map_item: QGraphicsPixmapItem | None = None
        self._drone_overlays: list[dict] = []

        # Geofence drawing
        self._drawing = False
        self._draw_mode = "add"
        self._fence_points: list[QPointF] = []
        self._fence_outer_points: list[QPointF] = []
        self._fence_holes_points: list[list[QPointF]] = []
        self._fence_item: QGraphicsPathItem | None = None
        self._fence_hole_items: list[QGraphicsPolygonItem] = []
        self._fence_vertex_items: list[QGraphicsEllipseItem] = []
        self._buffer_item: QGraphicsPolygonItem | None = None
        self._preview_item: QGraphicsPolygonItem | None = None
        self._hover_scene_pos: QPointF | None = None
        self._active_shape = "outer"
        self._active_hole_index: int | None = None
        self._shape_dragging = False
        self._shape_drag_last_scene_pos: QPointF | None = None

    def set_map_image(self, cv_color: np.ndarray):
        pixmap = cv_to_qpixmap(cv_color)
        if self._map_item:
            self._scene.removeItem(self._map_item)
        self._map_item = self._scene.addPixmap(pixmap)
        self._map_item.setZValue(0)
        self.setSceneRect(QRectF(pixmap.rect()))
        self.fitInView(self._map_item, Qt.AspectRatioMode.KeepAspectRatio)

    def wheelEvent(self, event: QWheelEvent):
        if not self._drawing and (event.modifiers() & Qt.KeyboardModifier.ShiftModifier):
            factor = 1.05 if event.angleDelta().y() > 0 else 1 / 1.05
            if self.scale_active_shape(factor):
                event.accept()
                return
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.scale(factor, factor)

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key.Key_Home and self._map_item:
            self.fitInView(self._map_item, Qt.AspectRatioMode.KeepAspectRatio)
        super().keyPressEvent(event)

    def _ensure_drone_overlay(self, idx: int):
        while len(self._drone_overlays) <= idx:
            i = len(self._drone_overlays)
            c = DRONE_COLORS[i % len(DRONE_COLORS)]
            color = QColor(c["hex"])
            glow_color = QColor(c["hex"])
            glow_color.setAlpha(80)

            r_glow = 18
            glow = self._scene.addEllipse(
                -r_glow,
                -r_glow,
                r_glow * 2,
                r_glow * 2,
                QPen(glow_color, 3),
                QBrush(Qt.BrushStyle.NoBrush),
            )
            glow.setZValue(9)
            glow.setVisible(False)
            glow.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)

            r = 11
            dot = self._scene.addEllipse(
                -r,
                -r,
                r * 2,
                r * 2,
                QPen(QColor("white"), 2),
                QBrush(color),
            )
            dot.setZValue(11)
            dot.setVisible(False)
            dot.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)

            fp_item = self._scene.addPolygon(
                QPolygonF(),
                QPen(color, 3),
                QBrush(QColor(color.red(), color.green(), color.blue(), 30)),
            )
            fp_item.setZValue(8)
            fp_item.setVisible(False)

            label = self._scene.addText(
                f"Drone {i + 1}",
                QFont("Segoe UI", 11, QFont.Weight.Bold),
            )
            label.setDefaultTextColor(color)
            label.setZValue(12)
            label.setVisible(False)
            label.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)

            self._drone_overlays.append(
                {"dot": dot, "glow": glow, "footprint": fp_item, "label": label, "color": color},
            )

    def update_drone(
        self,
        idx: int,
        pos: tuple[int, int] | None,
        footprint: np.ndarray | None,
        status: GeofenceStatus,
    ):
        self._ensure_drone_overlay(idx)
        ov = self._drone_overlays[idx]
        base_color = ov["color"]

        if status == GeofenceStatus.VIOLATION:
            ring_color = QColor("#ff1744")
            fill_color = QColor("#ff1744")
        elif status == GeofenceStatus.APPROACHING:
            ring_color = QColor("#ffab00")
            fill_color = QColor("#ffab00")
        elif status == GeofenceStatus.LOST:
            ring_color = QColor("#ff6e40")
            fill_color = QColor(base_color)
            fill_color.setAlpha(120)
        else:
            ring_color = base_color
            fill_color = base_color

        if pos:
            ov["dot"].setPos(pos[0], pos[1])
            ov["dot"].setBrush(QBrush(fill_color))
            ov["dot"].setPen(QPen(QColor("white"), 2))
            ov["dot"].setVisible(True)

            glow_c = QColor(ring_color)
            glow_c.setAlpha(90)
            ov["glow"].setPos(pos[0], pos[1])
            ov["glow"].setPen(QPen(glow_c, 4))
            ov["glow"].setVisible(True)

            ov["label"].setPos(pos[0] + 2, pos[1] - 2)
            ov["label"].setDefaultTextColor(ring_color)
            ov["label"].setVisible(True)
        else:
            ov["dot"].setVisible(False)
            ov["glow"].setVisible(False)
            ov["label"].setVisible(False)

        if footprint is not None:
            poly = QPolygonF(
                [QPointF(float(footprint[i, 0, 0]), float(footprint[i, 0, 1])) for i in range(4)],
            )
            fp_brush = QBrush(QColor(ring_color.red(), ring_color.green(), ring_color.blue(), 35))
            ov["footprint"].setPolygon(poly)
            ov["footprint"].setPen(QPen(ring_color, 3))
            ov["footprint"].setBrush(fp_brush)
            ov["footprint"].setVisible(True)
        else:
            ov["footprint"].setVisible(False)

    def hide_drone(self, idx: int):
        if idx < len(self._drone_overlays):
            ov = self._drone_overlays[idx]
            for key in ("dot", "glow", "footprint", "label"):
                ov[key].setVisible(False)

    def start_drawing(self, mode: str = "add"):
        self._drawing = True
        self._draw_mode = mode
        self._fence_points.clear()
        self._hover_scene_pos = None
        self._clear_fence_vertices()
        if mode == "add":
            self._fence_outer_points = []
            self._fence_holes_points = []
            self._active_shape = "outer"
            self._active_hole_index = None
            self._redraw_fence_visuals()
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self._update_drawing_preview()

    def finish_drawing(self) -> bool:
        if len(self._fence_points) < 3:
            return False

        if self._draw_mode == "subtract":
            if len(self._fence_outer_points) < 3:
                return False
            self._fence_holes_points.append(self._fence_points.copy())
            self._active_shape = "hole"
            self._active_hole_index = len(self._fence_holes_points) - 1
        else:
            self._fence_outer_points = self._fence_points.copy()
            self._fence_holes_points = []
            self._active_shape = "outer"
            self._active_hole_index = None

        self._drawing = False
        self._hover_scene_pos = None
        self._fence_points.clear()
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self._clear_preview_polygon()
        self._clear_fence_vertices()
        self._redraw_fence_visuals()
        self.geofence_changed.emit(self.get_fence_geometry_pixels())
        return True

    def cancel_drawing(self):
        self._drawing = False
        self._hover_scene_pos = None
        self._fence_points.clear()
        self._clear_fence_vertices()
        self._clear_preview_polygon()
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setCursor(Qt.CursorShape.ArrowCursor)

    def point_count(self) -> int:
        return len(self._fence_points)

    def undo_last_point(self) -> bool:
        if not self._drawing or not self._fence_points:
            return False
        self._fence_points.pop()
        if self._fence_vertex_items:
            item = self._fence_vertex_items.pop()
            self._scene.removeItem(item)
        if self._fence_item and len(self._fence_points) < 3:
            self._scene.removeItem(self._fence_item)
            self._fence_item = None
        elif self._fence_item:
            self._draw_fence_polygon()
        self._update_drawing_preview()
        return True

    def set_rectangle_fence(self, inset_ratio: float = 0.12):
        if self._map_item is None:
            return
        rect = self.sceneRect()
        inset = max(20.0, min(rect.width(), rect.height()) * inset_ratio)
        left = rect.left() + inset
        top = rect.top() + inset
        right = rect.right() - inset
        bottom = rect.bottom() - inset
        if right - left < 20 or bottom - top < 20:
            left, top, right, bottom = rect.left(), rect.top(), rect.right(), rect.bottom()
        self._drawing = False
        self._hover_scene_pos = None
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self._clear_preview_polygon()
        self._clear_fence_vertices()
        self._fence_outer_points = [
            QPointF(left, top),
            QPointF(right, top),
            QPointF(right, bottom),
            QPointF(left, bottom),
        ]
        self._fence_holes_points = []
        self._active_shape = "outer"
        self._active_hole_index = None
        for p in self._fence_outer_points:
            r = 7
            dot = self._scene.addEllipse(
                p.x() - r,
                p.y() - r,
                r * 2,
                r * 2,
                QPen(QColor("white"), 2),
                QBrush(QColor("#ff1744")),
            )
            dot.setZValue(20)
            self._fence_vertex_items.append(dot)
        self._redraw_fence_visuals()
        self.geofence_changed.emit(self.get_fence_geometry_pixels())

    def subtract_rectangle_cutout(self, inset_ratio: float = 0.22) -> bool:
        if len(self._fence_outer_points) < 3:
            return False
        xs = [p.x() for p in self._fence_outer_points]
        ys = [p.y() for p in self._fence_outer_points]
        left, right = min(xs), max(xs)
        top, bottom = min(ys), max(ys)
        inset = max(12.0, min(right - left, bottom - top) * inset_ratio)
        if right - left < 24 or bottom - top < 24:
            return False
        hole = [
            QPointF(left + inset, top + inset),
            QPointF(right - inset, top + inset),
            QPointF(right - inset, bottom - inset),
            QPointF(left + inset, bottom - inset),
        ]
        self._fence_holes_points.append(hole)
        self._active_shape = "hole"
        self._active_hole_index = len(self._fence_holes_points) - 1
        self._redraw_fence_visuals()
        self.geofence_changed.emit(self.get_fence_geometry_pixels())
        return True

    def set_circle_fence(self, radius_ratio: float = 0.22, points: int = 48):
        if self._map_item is None:
            return
        rect = self.sceneRect()
        cx = rect.center().x()
        cy = rect.center().y()
        radius = max(20.0, min(rect.width(), rect.height()) * radius_ratio)
        circle = []
        for i in range(points):
            a = 2.0 * np.pi * i / points
            circle.append(QPointF(cx + radius * np.cos(a), cy + radius * np.sin(a)))

        self._drawing = False
        self._draw_mode = "add"
        self._hover_scene_pos = None
        self._fence_points.clear()
        self._fence_outer_points = circle
        self._fence_holes_points = []
        self._active_shape = "outer"
        self._active_hole_index = None
        self._clear_preview_polygon()
        self._clear_fence_vertices()
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self._redraw_fence_visuals()
        self.geofence_changed.emit(self.get_fence_geometry_pixels())

    def subtract_circle_cutout(self, radius_ratio: float = 0.12, points: int = 36) -> bool:
        if len(self._fence_outer_points) < 3:
            return False
        xs = [p.x() for p in self._fence_outer_points]
        ys = [p.y() for p in self._fence_outer_points]
        cx = (min(xs) + max(xs)) / 2.0
        cy = (min(ys) + max(ys)) / 2.0
        radius = max(12.0, min(max(xs) - min(xs), max(ys) - min(ys)) * radius_ratio)
        hole = []
        for i in range(points):
            a = 2.0 * np.pi * i / points
            hole.append(QPointF(cx + radius * np.cos(a), cy + radius * np.sin(a)))
        self._fence_holes_points.append(hole)
        self._active_shape = "hole"
        self._active_hole_index = len(self._fence_holes_points) - 1
        self._redraw_fence_visuals()
        self.geofence_changed.emit(self.get_fence_geometry_pixels())
        return True

    def is_subtract_mode(self) -> bool:
        return self._drawing and self._draw_mode == "subtract"

    def move_active_shape(self, dx: float, dy: float) -> bool:
        pts = self._get_active_shape_points()
        if pts is None:
            return False
        for i, p in enumerate(pts):
            pts[i] = QPointF(p.x() + dx, p.y() + dy)
        self._redraw_fence_visuals()
        self.geofence_changed.emit(self.get_fence_geometry_pixels())
        return True

    def scale_active_shape(self, factor: float) -> bool:
        if factor <= 0:
            return False
        pts = self._get_active_shape_points()
        if pts is None or len(pts) < 3:
            return False
        cx = sum(p.x() for p in pts) / len(pts)
        cy = sum(p.y() for p in pts) / len(pts)
        for i, p in enumerate(pts):
            nx = cx + (p.x() - cx) * factor
            ny = cy + (p.y() - cy) * factor
            pts[i] = QPointF(nx, ny)
        self._redraw_fence_visuals()
        self.geofence_changed.emit(self.get_fence_geometry_pixels())
        return True

    def _get_active_shape_points(self) -> list[QPointF] | None:
        if self._active_shape == "hole":
            if self._active_hole_index is None:
                return None
            if self._active_hole_index < 0 or self._active_hole_index >= len(self._fence_holes_points):
                return None
            return self._fence_holes_points[self._active_hole_index]
        if len(self._fence_outer_points) < 3:
            return None
        return self._fence_outer_points

    def _point_in_active_shape(self, scene_pos: QPointF) -> bool:
        pts = self._get_active_shape_points()
        if pts is None or len(pts) < 3:
            return False
        poly = QPolygonF(pts)
        return poly.containsPoint(scene_pos, Qt.FillRule.WindingFill)

    def get_fence_geometry_pixels(self) -> dict:
        outer = [(int(p.x()), int(p.y())) for p in self._fence_outer_points]
        holes = [[(int(p.x()), int(p.y())) for p in h] for h in self._fence_holes_points]
        return {"outer": outer, "holes": holes}

    def set_fence_polygon(
        self,
        pixel_coords: list[tuple[int, int]],
        hole_pixel_coords_list: list[list[tuple[int, int]]] | None = None,
        buffer_pixels: list[tuple[int, int]] | None = None,
    ):
        self._fence_outer_points = [QPointF(x, y) for x, y in pixel_coords]
        self._fence_holes_points = []
        if hole_pixel_coords_list:
            self._fence_holes_points = [[QPointF(x, y) for x, y in hole] for hole in hole_pixel_coords_list]
        self._redraw_fence_visuals()
        if buffer_pixels:
            self._draw_buffer_polygon(buffer_pixels)

    def clear_fence(self):
        self._drawing = False
        self._hover_scene_pos = None
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self._fence_points.clear()
        self._fence_outer_points.clear()
        self._fence_holes_points.clear()
        self._active_shape = "outer"
        self._active_hole_index = None
        self._clear_fence_vertices()
        self._clear_preview_polygon()
        if self._fence_item:
            self._scene.removeItem(self._fence_item)
            self._fence_item = None
        for item in self._fence_hole_items:
            self._scene.removeItem(item)
        self._fence_hole_items.clear()
        if self._buffer_item:
            self._scene.removeItem(self._buffer_item)
            self._buffer_item = None

    def mousePressEvent(self, event: QMouseEvent):
        if (
            not self._drawing
            and event.button() == Qt.MouseButton.LeftButton
            and (event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
        ):
            scene_pos = self.mapToScene(event.position().toPoint())
            if self._point_in_active_shape(scene_pos):
                self._shape_dragging = True
                self._shape_drag_last_scene_pos = scene_pos
                self.setDragMode(QGraphicsView.DragMode.NoDrag)
                self.setCursor(Qt.CursorShape.SizeAllCursor)
                event.accept()
                return
        if self._drawing and event.button() == Qt.MouseButton.RightButton:
            self.undo_last_point()
            return
        if self._drawing and event.button() == Qt.MouseButton.LeftButton:
            scene_pos = self.mapToScene(event.position().toPoint())
            if len(self._fence_points) >= 3:
                first = self._fence_points[0]
                if (scene_pos - first).manhattanLength() <= 20:
                    self.finish_drawing()
                    return
            self._fence_points.append(scene_pos)
            r = 7
            dot = self._scene.addEllipse(
                scene_pos.x() - r,
                scene_pos.y() - r,
                r * 2,
                r * 2,
                QPen(QColor("white"), 2),
                QBrush(QColor("#ff1744")),
            )
            dot.setZValue(20)
            self._fence_vertex_items.append(dot)
            self._update_drawing_preview()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._shape_dragging and self._shape_drag_last_scene_pos is not None:
            scene_pos = self.mapToScene(event.position().toPoint())
            dx = scene_pos.x() - self._shape_drag_last_scene_pos.x()
            dy = scene_pos.y() - self._shape_drag_last_scene_pos.y()
            if self.move_active_shape(dx, dy):
                self._shape_drag_last_scene_pos = scene_pos
            event.accept()
            return
        if self._drawing:
            self._hover_scene_pos = self.mapToScene(event.position().toPoint())
            self._update_drawing_preview()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if self._shape_dragging and event.button() == Qt.MouseButton.LeftButton:
            self._shape_dragging = False
            self._shape_drag_last_scene_pos = None
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            self.setCursor(Qt.CursorShape.ArrowCursor)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent):
        if self._drawing and event.button() == Qt.MouseButton.LeftButton:
            self.finish_drawing()
            return
        super().mouseDoubleClickEvent(event)

    def _update_drawing_preview(self):
        if not self._drawing:
            self._clear_preview_polygon()
            return
        preview_points = list(self._fence_points)
        if self._hover_scene_pos is not None:
            preview_points.append(self._hover_scene_pos)
        if len(preview_points) < 2:
            self._clear_preview_polygon()
            return
        poly = QPolygonF(preview_points)
        pen = QPen(QColor("#ff617f"), 2, Qt.PenStyle.DashLine)
        brush = QBrush(QColor(255, 23, 68, 25)) if len(preview_points) >= 3 else QBrush(Qt.BrushStyle.NoBrush)
        if self._preview_item is None:
            self._preview_item = self._scene.addPolygon(poly, pen, brush)
            self._preview_item.setZValue(6)
        else:
            self._preview_item.setPolygon(poly)
            self._preview_item.setPen(pen)
            self._preview_item.setBrush(brush)

    def _clear_preview_polygon(self):
        if self._preview_item is not None:
            self._scene.removeItem(self._preview_item)
            self._preview_item = None

    def _draw_fence_polygon(self):
        if len(self._fence_outer_points) < 3:
            if self._fence_item is not None:
                self._scene.removeItem(self._fence_item)
                self._fence_item = None
            return

        path = QPainterPath()
        path.setFillRule(Qt.FillRule.OddEvenFill)
        path.addPolygon(QPolygonF(self._fence_outer_points))
        for hole in self._fence_holes_points:
            if len(hole) >= 3:
                path.addPolygon(QPolygonF(hole))

        pen = QPen(QColor("#ff1744"), 3, Qt.PenStyle.DashLine)
        brush = QBrush(QColor(255, 23, 68, 50))
        if self._fence_item is None:
            self._fence_item = self._scene.addPath(path, pen, brush)
            self._fence_item.setZValue(5)
        else:
            self._fence_item.setPath(path)
            self._fence_item.setPen(pen)
            self._fence_item.setBrush(brush)

    def _redraw_fence_visuals(self):
        self._draw_fence_polygon()
        for item in self._fence_hole_items:
            self._scene.removeItem(item)
        self._fence_hole_items.clear()
        pen = QPen(QColor("#ffab00"), 2, Qt.PenStyle.DashLine)
        brush = QBrush(Qt.BrushStyle.NoBrush)
        for hole in self._fence_holes_points:
            if len(hole) < 3:
                continue
            item = self._scene.addPolygon(QPolygonF(hole), pen, brush)
            item.setZValue(7)
            self._fence_hole_items.append(item)

    def _draw_buffer_polygon(self, pixel_coords: list[tuple[int, int]]):
        poly = QPolygonF([QPointF(x, y) for x, y in pixel_coords])
        pen = QPen(QColor("#ffab00"), 2, Qt.PenStyle.DotLine)
        brush = QBrush(QColor(255, 171, 0, 25))
        if self._buffer_item is None:
            self._buffer_item = self._scene.addPolygon(poly, pen, brush)
            self._buffer_item.setZValue(4)
        else:
            self._buffer_item.setPolygon(poly)

    def _clear_fence_vertices(self):
        for item in self._fence_vertex_items:
            self._scene.removeItem(item)
        self._fence_vertex_items.clear()


class FPVWidget(QLabel):
    def __init__(self, drone_index: int, parent=None):
        super().__init__(parent)
        self._idx = drone_index
        c = DRONE_COLORS[drone_index % len(DRONE_COLORS)]
        self._base_color = c["hex"]
        self.setMinimumSize(320, 180)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self._set_border(self._base_color)
        self.setText(f"Drone {drone_index + 1}\n(no feed)")

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, width: int) -> int:
        # Keep a stable 16:9 viewport to avoid stretched feed containers.
        return max(1, int(width * 9 / 16))

    def sizeHint(self) -> QSize:
        return QSize(480, 270)

    def _set_border(self, color: str):
        self.setStyleSheet(
            f"background-color: #0d0d1a; border: 3px solid {color}; "
            f"border-top-left-radius: 4px; border-top-right-radius: 4px; "
            f"border-bottom-left-radius: 0px; border-bottom-right-radius: 0px; color: #888;",
        )

    def update_frame(
        self,
        frame: np.ndarray,
        status: GeofenceStatus = GeofenceStatus.INITIALIZING,
    ):
        w, h = self.width(), self.height()
        if w < 10 or h < 10:
            return

        if status == GeofenceStatus.VIOLATION:
            self._set_border("#ff1744")
        elif status == GeofenceStatus.APPROACHING:
            self._set_border("#ffab00")
        elif status == GeofenceStatus.LOST:
            self._set_border("#ff6e40")
        else:
            self._set_border(self._base_color)

        source = cv_to_qpixmap(frame)
        # Fill the viewport and crop from the center to avoid side letterboxing.
        expanded = source.scaled(
            w,
            h,
            Qt.AspectRatioMode.KeepAspectRatioByExpanding,
            Qt.TransformationMode.SmoothTransformation,
        )
        x = max(0, (expanded.width() - w) // 2)
        y = max(0, (expanded.height() - h) // 2)
        self.setPixmap(expanded.copy(x, y, w, h))

    def reset_display(self):
        self.clear()
        self.setText(f"Drone {self._idx + 1}\n(no feed)")
        self._set_border(self._base_color)

    def show_loaded_ready(self, label: str):
        self.clear()
        self.setText(f"Drone {self._idx + 1}\nloaded: {label}\nready to start")
        self._set_border(self._base_color)


class DroneInfoPanel(QGroupBox):
    def __init__(self, drone_index: int, parent=None):
        c = DRONE_COLORS[drone_index % len(DRONE_COLORS)]
        super().__init__(parent)
        self.setTitle("")
        self.setStyleSheet(
            f"""
            QGroupBox {{
                color: {c["hex"]};
                border: 1px solid {c["hex"]};
                border-top-left-radius: 0px;
                border-top-right-radius: 0px;
                border-bottom-left-radius: 6px;
                border-bottom-right-radius: 6px;
                padding: 4px;
                margin: 0;
                background: rgba(10, 16, 30, 0.65);
                font-weight: 600;
            }}
            QLabel {{
                color: #d8dbe5;
                font-size: 12px;
            }}
            QLabel#valueLabel {{
                color: #f1f4ff;
                font-weight: 600;
                font-size: 12px;
            }}
            QToolButton#mapsButton {{
                color: #e5efff;
                background: #254a74;
                border: 1px solid #2f679d;
                border-radius: 4px;
                padding: 0px;
            }}
            QToolButton#mapsButton:hover:!disabled {{
                background: #2d5b8d;
            }}
            QToolButton#mapsButton:disabled {{
                color: #7e8ca2;
                background: #1a2232;
                border: 1px solid #2a3347;
            }}
            """
        )
        layout = QFormLayout(self)
        layout.setContentsMargins(3, 3, 3, 3)
        layout.setSpacing(3)
        layout.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        layout.setFormAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)

        self.lbl_status = QLabel("—")
        self.lbl_status.setObjectName("valueLabel")
        self.lbl_status.setStyleSheet("font-size: 13px; font-weight: bold;")
        layout.addRow("Status", self.lbl_status)

        coords_row = QWidget()
        coords_layout = QHBoxLayout(coords_row)
        coords_layout.setContentsMargins(0, 0, 0, 0)
        coords_layout.setSpacing(4)

        self.lbl_coords = QLabel("—")
        self.lbl_coords.setObjectName("valueLabel")
        self.lbl_coords.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        coord_font = QFont("Consolas", 11)
        coord_font.setStyleHint(QFont.StyleHint.Monospace)
        self.lbl_coords.setFont(coord_font)
        self.lbl_coords.setMinimumWidth(self.lbl_coords.fontMetrics().horizontalAdvance("E 0000000.0  N 0000000.0"))
        self.lbl_coords.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
        coords_layout.addWidget(self.lbl_coords)

        coords_layout.addStretch()

        self.btn_open_maps = QToolButton()
        self.btn_open_maps.setObjectName("mapsButton")
        self.btn_open_maps.setText("")
        self.btn_open_maps.setIcon(_material_icon("mdi6.open-in-new", QStyle.StandardPixmap.SP_ArrowForward))
        self.btn_open_maps.setIconSize(QSize(12, 12))
        self.btn_open_maps.setAutoRaise(True)
        self.btn_open_maps.setFixedSize(18, 18)
        self.btn_open_maps.setToolTip("Open current position in Google Maps")
        self.btn_open_maps.setEnabled(False)
        coords_layout.addWidget(self.btn_open_maps)

        layout.addRow("Coords", coords_row)

        self.lbl_distance = QLabel("—")
        self.lbl_distance.setObjectName("valueLabel")
        layout.addRow("Fence", self.lbl_distance)

        self.lbl_lost = QLabel("0")
        self.lbl_lost.setObjectName("valueLabel")
        layout.addRow("Lost", self.lbl_lost)

    def update_result(self, result: TrackingResult):
        sc = STATUS_COLORS.get(result.status, "#888")
        bg = STATUS_BG.get(result.status, "transparent")
        self.lbl_status.setText(result.status.value)
        self.lbl_status.setStyleSheet(
            f"font-size: 13px; font-weight: bold; color: {sc}; background: {bg}; padding: 2px 6px; border-radius: 3px;",
        )

        if result.coords:
            self.lbl_coords.setText(f"E {result.coords[0]:.1f}  N {result.coords[1]:.1f}")
            self.btn_open_maps.setEnabled(True)
        else:
            self.lbl_coords.setText("—")
            self.btn_open_maps.setEnabled(False)

        if result.distance_to_fence is not None:
            d = result.distance_to_fence
            txt = f"{abs(d):.1f}m {'inside' if d < 0 else 'outside'}"
            dc = "#ff1744" if d < 0 else "#00e676"
            self.lbl_distance.setText(txt)
            self.lbl_distance.setStyleSheet(f"color: {dc}; font-weight: bold;")
        else:
            self.lbl_distance.setText("—")
            self.lbl_distance.setStyleSheet("")

        self.lbl_lost.setText(str(result.lost_count))


class FeedPanel(QWidget):
    load_requested = Signal(int)
    clear_requested = Signal(int)

    def __init__(self, drone_index: int, parent=None):
        super().__init__(parent)
        self._idx = drone_index
        c = DRONE_COLORS[drone_index % len(DRONE_COLORS)]

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header row: coloured label + load / clear buttons
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)

        title = QLabel(f"● Drone {drone_index + 1}")
        title.setStyleSheet(f"color: {c['hex']}; font-weight: bold; font-size: 13px;")
        header.addWidget(title)
        header.addStretch()

        SP = QStyle.StandardPixmap

        self.btn_load = QToolButton()
        self.btn_load.setIcon(_material_icon("mdi6.file-video-outline", SP.SP_DirOpenIcon))
        self.btn_load.setToolTip(f"Load video for Drone {drone_index + 1}")
        self.btn_load.setEnabled(False)
        self.btn_load.clicked.connect(lambda: self.load_requested.emit(self._idx))
        header.addWidget(self.btn_load)

        self.btn_clear = QToolButton()
        self.btn_clear.setIcon(_material_icon("mdi6.close-circle-outline", SP.SP_TitleBarCloseButton))
        self.btn_clear.setToolTip(f"Clear feed for Drone {drone_index + 1}")
        self.btn_clear.setEnabled(False)
        self.btn_clear.clicked.connect(lambda: self.clear_requested.emit(self._idx))
        header.addWidget(self.btn_clear)

        layout.addLayout(header)
        layout.addSpacing(2)

        self.fpv = FPVWidget(drone_index)
        layout.addWidget(self.fpv, stretch=2)

        self.info = DroneInfoPanel(drone_index)
        layout.addWidget(self.info, stretch=0)
