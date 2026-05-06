import numpy as np

from PyQt5.QtCore import Qt, QPoint, QRect, QSize, QTimer
from PyQt5.QtGui import (
    QColor, QFont, QImage, QKeySequence, QPainter, QPen, QPixmap
)
from PyQt5.QtWidgets import QApplication, QLabel, QShortcut, QSizePolicy, QWidget

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


class ViewerWindow(QWidget):
    """
    리사이즈 가능한 뷰어 창.
    캡처 프레임을 배경으로 표시하고, 그 위에 탐지 박스를 렌더링.
    """

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self._frame: np.ndarray | None = None
        self._detections: list[Detection] = []
        self._current_model_name: str = config.active_model
        self._model_label_timer: QTimer | None = None
        self._logging_active: bool = False

        # 콜백 (main에서 연결)
        self.on_toggle_logging = None
        self.on_manual_screenshot = None
        self.on_conf_up = None
        self.on_conf_down = None
        self.on_next_model = None
        self.on_exit = None

        self._init_window()
        self._setup_shortcuts()

    def _init_window(self) -> None:
        self.setWindowTitle("YOLO Overlay Viewer")
        self.resize(self.config.preview_width, self.config.preview_height)
        self.setMinimumSize(320, 180)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # 항상 최상위 (선택 가능하게 플래그 없이도 됨)
        self.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)

    def _setup_shortcuts(self) -> None:
        shortcuts = {
            "F1": self._toggle_visible,
            "F2": self._trigger_logging,
            "F3": self._trigger_screenshot,
            "F4": self._trigger_conf_up,
            "F5": self._trigger_conf_down,
            "F6": self._trigger_next_model,
            "Escape": self._trigger_exit,
        }
        for key, slot in shortcuts.items():
            sc = QShortcut(QKeySequence(key), self)
            sc.activated.connect(slot)
            sc.setContext(Qt.ApplicationShortcut)

    def update_frame(
        self,
        frame: np.ndarray,
        detections: list[Detection],
    ) -> None:
        self._frame = frame
        self._detections = detections
        self.update()

    def show_model_label(self, model_name: str) -> None:
        self._current_model_name = model_name
        if self._model_label_timer:
            self._model_label_timer.stop()
        timer = QTimer(self)
        timer.setSingleShot(True)
        timer.timeout.connect(self.update)
        timer.start(3000)
        self._model_label_timer = timer
        self.update()

    def toggle_visible(self) -> None:
        if self.isVisible():
            self.hide()
        else:
            self.show()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        w, h = self.width(), self.height()

        # ── 배경 프레임 렌더링 ────────────────────────────────────────────────
        if self._frame is not None:
            frame = self._frame
            fh, fw = frame.shape[:2]

            # BGR → RGB 변환 후 QImage
            rgb = frame[:, :, ::-1].copy()
            qimg = QImage(rgb.data, fw, fh, fw * 3, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)

            # 비율 유지하며 창 크기에 맞춤
            scaled = pixmap.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            x_off = (w - scaled.width()) // 2
            y_off = (h - scaled.height()) // 2
            painter.drawPixmap(x_off, y_off, scaled)

            # 탐지 박스 좌표 스케일 계산
            scale_x = scaled.width() / fw
            scale_y = scaled.height() / fh
        else:
            painter.fillRect(0, 0, w, h, QColor(30, 30, 30))
            painter.setPen(QPen(QColor(180, 180, 180)))
            painter.setFont(QFont("Arial", 14))
            painter.drawText(self.rect(), Qt.AlignCenter, "캡처 대기 중...")
            painter.end()
            return

        # ── 탐지 박스 렌더링 ──────────────────────────────────────────────────
        painter.setRenderHint(QPainter.Antialiasing)
        label_font = QFont("Arial", 9, QFont.Bold)
        painter.setFont(label_font)
        fm = painter.fontMetrics()

        for det in self._detections:
            color = CLASS_COLORS[det.class_id % len(CLASS_COLORS)]
            pen = QPen(color, self.config.box_thickness)
            painter.setPen(pen)

            rx = int(det.x1 * scale_x) + x_off
            ry = int(det.y1 * scale_y) + y_off
            rw = int((det.x2 - det.x1) * scale_x)
            rh = int((det.y2 - det.y1) * scale_y)
            rect = QRect(rx, ry, rw, rh)
            painter.drawRect(rect)

            if self.config.show_confidence or self.config.show_class_id:
                parts = []
                if self.config.show_class_id:
                    parts.append(f"cls{det.class_id}")
                if self.config.show_confidence:
                    parts.append(f"{det.confidence:.2f}")
                label = ": ".join(parts)

                text_w = fm.horizontalAdvance(label) + 6
                text_h = fm.height() + 2
                label_y = max(ry, text_h)
                bg = QRect(rx, label_y - text_h, text_w, text_h)
                painter.fillRect(bg, QColor(0, 0, 0, 180))
                painter.setPen(QPen(color))
                painter.drawText(QPoint(rx + 3, label_y - 2), label)

        # ── HUD ───────────────────────────────────────────────────────────────
        painter.setFont(QFont("Arial", 10, QFont.Bold))

        hud_lines = [
            f"모델: {self._current_model_name}",
            f"conf: {self.config.conf_threshold:.2f}",
            f"탐지: {len(self._detections)}",
        ]
        if self._logging_active:
            hud_lines.append("[REC]")

        for i, text in enumerate(hud_lines):
            tx, ty = 8, 18 + i * 18
            # 그림자
            painter.setPen(QPen(QColor(0, 0, 0, 200)))
            painter.drawText(QPoint(tx + 1, ty + 1), text)
            # 텍스트
            color = QColor(255, 80, 80) if text == "[REC]" else QColor(255, 255, 0)
            painter.setPen(QPen(color))
            painter.drawText(QPoint(tx, ty), text)

        # 단축키 힌트 (하단)
        hint = "F2:로깅  F3:스크린샷  F4/F5:conf  F6:모델전환  ESC:종료"
        painter.setFont(QFont("Arial", 8))
        painter.setPen(QPen(QColor(180, 180, 180, 160)))
        painter.drawText(QPoint(8, h - 6), hint)

        painter.end()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.update()

    # ── 단축키 핸들러 ──────────────────────────────────────────────────────────

    def _toggle_visible(self) -> None:
        self.toggle_visible()

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
        if self.on_conf_up:
            self.on_conf_up()

    def _trigger_conf_down(self) -> None:
        self.config.conf_threshold = max(0.05, round(self.config.conf_threshold - 0.05, 2))
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
