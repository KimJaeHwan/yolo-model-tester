import ctypes
import threading

from PyQt5.QtCore import Qt, QPoint, QRect, QTimer
from PyQt5.QtGui import QColor, QCursor, QFont, QKeySequence, QPainter, QPen
from PyQt5.QtWidgets import QApplication, QShortcut, QWidget

from config import Config
from inference import Detection


CLASS_COLORS = [
    QColor(255, 0, 0),
    QColor(0, 255, 0),
    QColor(0, 120, 255),
    QColor(255, 255, 0),
    QColor(255, 0, 255),
    QColor(0, 255, 255),
    QColor(255, 128, 0),
    QColor(128, 0, 255),
]

BORDER = 3          # 테두리 두께 (px)
HANDLE = 10         # 모서리 핸들 크기 (px)
HEADER = 20         # 상단 드래그 바 높이 (px)

# 리사이즈 방향 상수
_NONE  = 0
_N, _S, _W, _E     = 1, 2, 4, 8
_NW, _NE, _SW, _SE = _N|_W, _N|_E, _S|_W, _S|_E


class OverlayWindow(QWidget):
    """
    투명 스캔 영역 창.

    - 기본(편집 모드): 드래그/리사이즈 가능, 게임 클릭 통과 안 됨
    - F7 / 잠금 버튼: 오버레이 모드 전환 → 클릭 통과, 탐지 박스만 표시
    - 이 창의 위치·크기가 곧 YOLO 탐지 범위
    """

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self._detections: list[Detection] = []
        self._locked = False            # False=편집모드, True=오버레이(클릭통과)
        self._current_model: str = config.active_model
        self._model_timer: QTimer | None = None
        self._logging_active: bool = False

        # thread-safe 스캔 범위 캐시 (추론 스레드에서 읽음)
        self._scan_rect_cache: tuple[int, int, int, int] = (
            100, 100,
            self.config.preview_width,
            self.config.preview_height,
        )
        self._scan_rect_lock = threading.Lock()

        # 드래그/리사이즈 상태
        self._drag_offset: QPoint | None = None
        self._resize_dir: int = _NONE
        self._resize_start_geom: QRect | None = None
        self._resize_start_global: QPoint | None = None

        # 콜백
        self.on_toggle_logging = None
        self.on_manual_screenshot = None
        self.on_conf_up = None
        self.on_conf_down = None
        self.on_next_model = None
        self.on_exit = None

        self._init_window()
        self._setup_shortcuts()

    def _init_window(self) -> None:
        self.setWindowFlags(
            Qt.FramelessWindowHint |
            Qt.WindowStaysOnTopHint |
            Qt.Tool
        )
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setMouseTracking(True)

        self.resize(self.config.preview_width, self.config.preview_height)
        self.move(100, 100)

    def _setup_shortcuts(self) -> None:
        defs = {
            "F7":     self._toggle_lock,
            "F2":     self._trigger_logging,
            "F3":     self._trigger_screenshot,
            "F4":     self._trigger_conf_up,
            "F5":     self._trigger_conf_down,
            "F6":     self._trigger_next_model,
            "Escape": self._trigger_exit,
        }
        for key, slot in defs.items():
            sc = QShortcut(QKeySequence(key), self)
            sc.activated.connect(slot)
            sc.setContext(Qt.ApplicationShortcut)

    # ── 공개 API ──────────────────────────────────────────────────────────────

    def get_scan_rect(self) -> tuple[int, int, int, int]:
        """현재 스캔 범위 반환. 추론 스레드에서 안전하게 호출 가능."""
        with self._scan_rect_lock:
            return self._scan_rect_cache

    def _update_scan_rect_cache(self) -> None:
        geo = self.geometry()
        with self._scan_rect_lock:
            self._scan_rect_cache = (geo.x(), geo.y(), geo.width(), geo.height())

    def update_detections(self, detections: list[Detection]) -> None:
        self._detections = detections
        self.update()

    def show_model_label(self, model_name: str) -> None:
        self._current_model = model_name
        if self._model_timer:
            self._model_timer.stop()
        t = QTimer(self)
        t.setSingleShot(True)
        t.timeout.connect(self.update)
        t.start(3000)
        self._model_timer = t
        self.update()

    # ── 잠금 토글 ─────────────────────────────────────────────────────────────

    def _toggle_lock(self) -> None:
        if self._locked:
            self._unlock()
        else:
            self._lock()

    def _lock(self) -> None:
        """오버레이 모드: 클릭 통과."""
        self._locked = True
        self._set_click_through(True)
        self.update()
        print("[Overlay] 잠금 (오버레이 모드) — F7로 해제")

    def _unlock(self) -> None:
        """편집 모드: 드래그/리사이즈 가능."""
        self._locked = False
        self._set_click_through(False)
        self.update()
        print("[Overlay] 편집 모드 — 드래그/리사이즈 가능")

    def _set_click_through(self, enable: bool) -> None:
        hwnd = int(self.winId())
        GWL_EXSTYLE   = -20
        WS_EX_TRANSPARENT = 0x20
        WS_EX_LAYERED     = 0x80000
        style = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
        if enable:
            new_style = style | WS_EX_TRANSPARENT | WS_EX_LAYERED
        else:
            new_style = (style & ~WS_EX_TRANSPARENT) | WS_EX_LAYERED
        ctypes.windll.user32.SetWindowLongW(hwnd, GWL_EXSTYLE, new_style)

    # ── 그리기 ────────────────────────────────────────────────────────────────

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        w, h = self.width(), self.height()

        if self._locked:
            self._paint_overlay(painter, w, h)
        else:
            self._paint_edit(painter, w, h)

        painter.end()

    def _paint_edit(self, painter: QPainter, w: int, h: int) -> None:
        """편집 모드: 탐지 박스 + 테두리 + 모서리 핸들 + HUD."""
        painter.setRenderHint(QPainter.Antialiasing)

        # 탐지 박스 (편집 모드에서도 표시 — 반투명)
        self._draw_detection_boxes(painter, alpha=160)

        # 테두리
        pen = QPen(QColor(0, 200, 255, 220), BORDER)
        painter.setPen(pen)
        painter.drawRect(BORDER // 2, BORDER // 2, w - BORDER, h - BORDER)

        # 상단 드래그 바 (반투명)
        painter.fillRect(0, 0, w, HEADER, QColor(0, 180, 255, 60))

        # 모서리 핸들
        for rx, ry in self._handle_rects(w, h):
            painter.fillRect(rx, ry, HANDLE, HANDLE, QColor(0, 200, 255, 200))

        # HUD 텍스트
        painter.setFont(QFont("Arial", 9, QFont.Bold))
        painter.setPen(QPen(QColor(255, 255, 255, 230)))
        hint = f"  {w}×{h}  탐지:{len(self._detections)}  드래그:이동  모서리:리사이즈  F7:잠금(클릭통과)"
        painter.drawText(QPoint(4, HEADER - 4), hint)

        # conf / 모델
        painter.setPen(QPen(QColor(200, 255, 100, 220)))
        painter.drawText(QPoint(4, h - 6),
                         f"모델:{self._current_model}  conf:{self.config.conf_threshold:.2f}")

    def _draw_detection_boxes(self, painter: QPainter, alpha: int = 255) -> None:
        """탐지 박스 렌더링. alpha: 박스/라벨 불투명도 (0~255)."""
        if not self._detections:
            return

        painter.setFont(QFont("Arial", 9, QFont.Bold))
        fm = painter.fontMetrics()

        for det in self._detections:
            base = CLASS_COLORS[det.class_id % len(CLASS_COLORS)]
            color = QColor(base.red(), base.green(), base.blue(), alpha)
            painter.setPen(QPen(color, self.config.box_thickness))
            rect = QRect(det.x1, det.y1, det.x2 - det.x1, det.y2 - det.y1)
            painter.drawRect(rect)

            parts = []
            if self.config.show_class_id:
                parts.append(f"cls{det.class_id}")
            if self.config.show_confidence:
                parts.append(f"{det.confidence:.2f}")
            if parts:
                label = ": ".join(parts)
                text_w = fm.horizontalAdvance(label) + 6
                text_h = fm.height() + 2
                ly = max(det.y1, text_h)
                painter.fillRect(det.x1, ly - text_h, text_w, text_h, QColor(0, 0, 0, int(alpha * 0.7)))
                painter.setPen(QPen(color))
                painter.drawText(QPoint(det.x1 + 3, ly - 2), label)

    def _paint_overlay(self, painter: QPainter, w: int, h: int) -> None:
        """오버레이 모드: 탐지 박스 + 얇은 테두리."""
        painter.setRenderHint(QPainter.Antialiasing)

        # 얇은 초록 테두리 (범위 표시용)
        painter.setPen(QPen(QColor(0, 255, 0, 100), 1))
        painter.drawRect(0, 0, w - 1, h - 1)

        # 탐지 박스
        self._draw_detection_boxes(painter, alpha=255)

        # HUD (좌상단)
        painter.setFont(QFont("Arial", 9, QFont.Bold))
        lines = [
            f"모델: {self._current_model}",
            f"conf: {self.config.conf_threshold:.2f}  탐지: {len(self._detections)}",
        ]
        if self._logging_active:
            lines.append("[REC]")
        for i, txt in enumerate(lines):
            ty = 14 + i * 16
            painter.setPen(QPen(QColor(0, 0, 0, 180)))
            painter.drawText(QPoint(5, ty + 1), txt)
            c = QColor(255, 60, 60) if txt == "[REC]" else QColor(255, 255, 0)
            painter.setPen(QPen(c))
            painter.drawText(QPoint(4, ty), txt)

    # ── 마우스: 드래그/리사이즈 ───────────────────────────────────────────────

    def mousePressEvent(self, event) -> None:
        if self._locked or event.button() != Qt.LeftButton:
            return
        pos = event.pos()
        self._resize_dir = self._hit_test(pos)
        if self._resize_dir != _NONE:
            self._resize_start_geom = self.geometry()
            self._resize_start_global = event.globalPos()
        else:
            self._drag_offset = event.globalPos() - self.pos()

    def mouseMoveEvent(self, event) -> None:
        if self._locked:
            return

        if not (event.buttons() & Qt.LeftButton):
            # 커서 모양 업데이트
            self._update_cursor(event.pos())
            return

        if self._resize_dir != _NONE:
            self._do_resize(event.globalPos())
        elif self._drag_offset is not None:
            self.move(event.globalPos() - self._drag_offset)

    def mouseReleaseEvent(self, event) -> None:
        self._drag_offset = None
        self._resize_dir = _NONE
        self._resize_start_geom = None
        self._resize_start_global = None
        self._update_scan_rect_cache()
        self._update_cursor(event.pos())

    def moveEvent(self, event) -> None:
        super().moveEvent(event)
        self._update_scan_rect_cache()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._update_scan_rect_cache()

    def _update_cursor(self, pos: QPoint) -> None:
        d = self._hit_test(pos)
        cursors = {
            _N: Qt.SizeVerCursor,  _S: Qt.SizeVerCursor,
            _W: Qt.SizeHorCursor,  _E: Qt.SizeHorCursor,
            _NW: Qt.SizeFDiagCursor, _SE: Qt.SizeFDiagCursor,
            _NE: Qt.SizeBDiagCursor, _SW: Qt.SizeBDiagCursor,
        }
        self.setCursor(QCursor(cursors.get(d, Qt.SizeAllCursor if self._in_header(pos) else Qt.ArrowCursor)))

    def _in_header(self, pos: QPoint) -> bool:
        return 0 <= pos.y() <= HEADER and HANDLE < pos.x() < self.width() - HANDLE

    def _hit_test(self, pos: QPoint) -> int:
        x, y, w, h = pos.x(), pos.y(), self.width(), self.height()
        d = _NONE
        if x < HANDLE:       d |= _W
        if x > w - HANDLE:   d |= _E
        if y < HANDLE:       d |= _N
        if y > h - HANDLE:   d |= _S
        return d

    def _do_resize(self, global_pos: QPoint) -> None:
        dx = global_pos.x() - self._resize_start_global.x()
        dy = global_pos.y() - self._resize_start_global.y()
        g = self._resize_start_geom
        x, y, w, h = g.x(), g.y(), g.width(), g.height()
        min_w, min_h = 200, 150

        if self._resize_dir & _E:  w = max(min_w, w + dx)
        if self._resize_dir & _S:  h = max(min_h, h + dy)
        if self._resize_dir & _W:
            new_w = max(min_w, w - dx)
            x = x + (w - new_w)
            w = new_w
        if self._resize_dir & _N:
            new_h = max(min_h, h - dy)
            y = y + (h - new_h)
            h = new_h

        self.setGeometry(x, y, w, h)

    @staticmethod
    def _handle_rects(w: int, h: int) -> list[tuple[int, int]]:
        hs = HANDLE
        return [
            (0, 0), (w - hs, 0),
            (0, h - hs), (w - hs, h - hs),
            ((w - hs) // 2, 0), ((w - hs) // 2, h - hs),
            (0, (h - hs) // 2), (w - hs, (h - hs) // 2),
        ]

    # ── 단축키 핸들러 ──────────────────────────────────────────────────────────

    def _trigger_logging(self) -> None:
        self._logging_active = not self._logging_active
        self.update()
        if self.on_toggle_logging:
            self.on_toggle_logging()

    def _trigger_screenshot(self) -> None:
        if self.on_manual_screenshot:
            self.on_manual_screenshot()

    def _trigger_conf_up(self) -> None:
        self.config.conf_threshold = min(0.95, round(self.config.conf_threshold + 0.05, 2))
        self.update()
        if self.on_conf_up:
            self.on_conf_up()

    def _trigger_conf_down(self) -> None:
        self.config.conf_threshold = max(0.05, round(self.config.conf_threshold - 0.05, 2))
        self.update()
        if self.on_conf_down:
            self.on_conf_down()

    def _trigger_next_model(self) -> None:
        if self.on_next_model:
            self.on_next_model()

    def _trigger_exit(self) -> None:
        if self.on_exit:
            self.on_exit()
        else:
            QApplication.quit()
