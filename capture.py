import queue
import threading
import time
import numpy as np

from config import Config


class WindowNotFoundError(Exception):
    pass


class CaptureModule:
    def __init__(self, config: Config):
        self.config = config
        self._queue: queue.Queue = queue.Queue(maxsize=2)
        self._running = False
        self._thread: threading.Thread | None = None
        self._window_rect: tuple[int, int, int, int] = (0, 0, 1920, 1080)
        self._dxcam_camera = None
        self._lock = threading.Lock()

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._dxcam_camera is not None:
            try:
                self._dxcam_camera.stop()
            except Exception:
                pass
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def get_frame(self) -> np.ndarray | None:
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return None

    def get_window_rect(self) -> tuple[int, int, int, int]:
        with self._lock:
            return self._window_rect

    def set_capture_rect(self, x: int, y: int, w: int, h: int) -> None:
        with self._lock:
            self._window_rect = (x, y, w, h)

    def _capture_loop(self) -> None:
        # set_capture_rect로 외부에서 범위가 주입되므로
        # target_window_title이 있을 때만 창 위치를 초기화
        if self.config.target_window_title:
            try:
                rect = self._resolve_window_rect()
                with self._lock:
                    self._window_rect = rect
            except WindowNotFoundError as e:
                print(f"[Capture] {e} — 전체 화면으로 폴백")

        # mss 우선 (다중 모니터 절대 좌표 지원)
        # dxcam은 단일 모니터 캡처라 overlay가 보조 모니터에 있으면 오작동
        if self.config.capture_mode == "dxcam":
            try:
                self._start_dxcam()
                return
            except Exception as e:
                print(f"[Capture] dxcam 실패: {e} — mss로 폴백")

        self._start_mss()

    def _start_dxcam(self) -> None:
        """
        dxcam은 단일 모니터를 전체 캡처 후 크롭.
        overlay 창이 어느 모니터에 있는지 감지해서 해당 모니터를 캡처하고
        모니터 내 상대 좌표로 크롭.
        """
        import dxcam

        while self._running:
            with self._lock:
                rx, ry, rw, rh = self._window_rect

            if rw <= 0 or rh <= 0:
                time.sleep(0.05)
                continue

            # overlay가 속한 모니터 감지
            mon_idx, mon_left, mon_top = self._find_monitor_for_rect(rx, ry)

            if self._dxcam_camera is None or getattr(self, '_dxcam_mon_idx', -1) != mon_idx:
                if self._dxcam_camera is not None:
                    try:
                        self._dxcam_camera.stop()
                    except Exception:
                        pass
                camera = dxcam.create(output_idx=mon_idx, output_color="BGR")
                camera.start(target_fps=self.config.capture_fps, video_mode=True)
                self._dxcam_camera = camera
                self._dxcam_mon_idx = mon_idx

            frame = self._dxcam_camera.get_latest_frame()
            if frame is None:
                time.sleep(0.005)
                continue

            # 모니터 내 상대 좌표로 크롭
            rel_x = rx - mon_left
            rel_y = ry - mon_top
            fh, fw = frame.shape[:2]
            x1 = max(0, rel_x)
            y1 = max(0, rel_y)
            x2 = min(fw, rel_x + rw)
            y2 = min(fh, rel_y + rh)
            if x2 > x1 and y2 > y1:
                self._put_frame(frame[y1:y2, x1:x2])

        if self._dxcam_camera is not None:
            self._dxcam_camera.stop()

    def _start_mss(self) -> None:
        """mss는 가상 데스크톱 절대 좌표를 그대로 사용 — 다중 모니터 완벽 지원."""
        import mss
        interval = 1.0 / self.config.capture_fps

        with mss.mss() as sct:
            while self._running:
                with self._lock:
                    rx, ry, rw, rh = self._window_rect

                if rw <= 0 or rh <= 0:
                    time.sleep(0.05)
                    continue

                monitor = {"top": ry, "left": rx, "width": rw, "height": rh}
                try:
                    raw = sct.grab(monitor)
                    frame = np.array(raw)[:, :, :3]  # BGRA → BGR
                    self._put_frame(frame)
                except Exception as e:
                    print(f"[Capture] mss 오류: {e}")

                time.sleep(interval)

    def _find_monitor_for_rect(self, x: int, y: int) -> tuple[int, int, int]:
        """(x,y) 좌표가 속한 모니터 인덱스와 해당 모니터의 left, top 반환."""
        try:
            import mss
            with mss.mss() as sct:
                for i, m in enumerate(sct.monitors[1:], start=0):
                    ml, mt = m["left"], m["top"]
                    mr, mb = ml + m["width"], mt + m["height"]
                    if ml <= x < mr and mt <= y < mb:
                        return i, ml, mt
                # 못 찾으면 주 모니터
                m = sct.monitors[1]
                return 0, m["left"], m["top"]
        except Exception:
            return 0, 0, 0

    def _resolve_window_rect(self) -> tuple[int, int, int, int]:
        try:
            import win32gui
            hwnd = win32gui.FindWindow(None, self.config.target_window_title)
            if not hwnd:
                results = []
                def enum_cb(h, _):
                    if self.config.target_window_title.lower() in win32gui.GetWindowText(h).lower():
                        results.append(h)
                win32gui.EnumWindows(enum_cb, None)
                hwnd = results[0] if results else None
            if hwnd:
                x, y, x2, y2 = win32gui.GetWindowRect(hwnd)
                return (x, y, x2 - x, y2 - y)
        except Exception as e:
            print(f"[Capture] 창 탐색 실패: {e}")
        raise WindowNotFoundError(f"창을 찾을 수 없음: '{self.config.target_window_title}'")

    def _put_frame(self, frame: np.ndarray) -> None:
        if self._queue.full():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
        try:
            self._queue.put_nowait(frame)
        except queue.Full:
            pass
