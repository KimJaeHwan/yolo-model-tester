import re
import threading
from dataclasses import dataclass

import cv2
import numpy as np

from config import Config


@dataclass
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    class_id: int


class InferenceModule:
    def __init__(self, config: Config):
        self.config = config
        self._net = None
        self._model_type: str = "yolov5"
        self._input_size: int = 640
        self._num_classes: int = 1
        self._in_layer: str = "in0"
        self._out_layers: list[str] = []
        self._lock = threading.Lock()

    def load_model(self, param_path: str, bin_path: str) -> None:
        import ncnn
        net = ncnn.Net()
        net.opt.use_vulkan_compute = False
        net.opt.num_threads = self.config.num_threads
        net.load_param(param_path)
        net.load_model(bin_path)

        input_size = self.config.input_size if self.config.input_size != 0 else 640
        self._input_size = input_size

        in_layer, out_layers = _parse_layers(param_path)
        self._in_layer = in_layer
        self._out_layers = out_layers

        model_type = _detect_model_type(param_path)
        self._model_type = model_type

        # 실제 추론으로 num_classes 확정
        num_classes = self._probe_num_classes(net, input_size, out_layers, model_type)
        self._num_classes = num_classes if self.config.num_classes == 0 else self.config.num_classes

        with self._lock:
            self._net = net

        print(f"[Inference] 로드 완료 / type={model_type} input={input_size} classes={self._num_classes}")
        print(f"[Inference] layers: in={in_layer} out={out_layers}")

    def switch_model(self, model_prefix: str) -> None:
        import os
        param_path = os.path.join(self.config.model_dir, f"{model_prefix}.param")
        bin_path = os.path.join(self.config.model_dir, f"{model_prefix}.bin")
        if not os.path.exists(param_path) or not os.path.exists(bin_path):
            raise FileNotFoundError(f"모델 파일 없음: {param_path}")
        with self._lock:
            self._net = None
        self.load_model(param_path, bin_path)

    def infer(self, frame: np.ndarray) -> list[Detection]:
        with self._lock:
            if self._net is None:
                return []
            net = self._net

        if frame is None or frame.size == 0:
            return []
        h, w = frame.shape[:2]
        if w == 0 or h == 0:
            return []

        resized, pad_top, pad_left, scale = _letterbox(frame, self._input_size)

        try:
            import ncnn
            ex = net.create_extractor()
            mat_in = ncnn.Mat.from_pixels(
                resized, ncnn.Mat.PixelType.PIXEL_BGR,
                self._input_size, self._input_size,
            )
            ex.input(self._in_layer, mat_in)

            if self._model_type == "yolov8":
                return self._postprocess_yolov8(ex, w, h, scale, pad_left, pad_top)
            else:
                return self._postprocess_yolov5(ex, w, h, scale, pad_left, pad_top)
        except Exception as e:
            print(f"[Inference] 오류 (프레임 스킵): {e}")
            return []

    def _postprocess_yolov5(
        self, ex, orig_w, orig_h, scale, pad_left, pad_top
    ) -> list[Detection]:
        all_boxes, all_scores, all_class_ids = [], [], []

        for layer_name in self._out_layers:
            try:
                _, out = ex.extract(layer_name)
            except Exception:
                continue

            data = np.array(out)  # shape: (gh, gw, channels)

            if data.ndim != 3:
                continue

            gh, gw, c = data.shape
            num_anchors = 3
            attrs = c // num_anchors  # ex) 39//3=13 (8cls), 18//3=6 (1cls)
            num_classes = attrs - 5
            if num_classes <= 0:
                continue

            # (gh, gw, num_anchors, attrs) → (-1, attrs)
            data = data.reshape(gh * gw * num_anchors, attrs)

            # NCNN YOLOv5 출력: sigmoid 이미 적용됨, 좌표는 0~1 정규화
            cx = data[:, 0]
            cy = data[:, 1]
            bw = data[:, 2]
            bh = data[:, 3]
            obj_conf = data[:, 4]
            cls_conf = data[:, 5:5 + num_classes]

            scores = obj_conf[:, np.newaxis] * cls_conf
            max_scores = scores.max(axis=1)
            class_ids = scores.argmax(axis=1)
            mask = max_scores >= self.config.conf_threshold

            if not mask.any():
                continue

            s = self._input_size
            # 정규화 좌표 → letterbox 픽셀 → 원본 픽셀
            x1s = ((cx[mask] - bw[mask] / 2) * s - pad_left) / scale
            y1s = ((cy[mask] - bh[mask] / 2) * s - pad_top) / scale
            x2s = ((cx[mask] + bw[mask] / 2) * s - pad_left) / scale
            y2s = ((cy[mask] + bh[mask] / 2) * s - pad_top) / scale

            for i in range(mask.sum()):
                all_boxes.append([float(x1s[i]), float(y1s[i]),
                                   float(x2s[i] - x1s[i]), float(y2s[i] - y1s[i])])
                all_scores.append(float(max_scores[mask][i]))
                all_class_ids.append(int(class_ids[mask][i]))

        return _apply_nms(all_boxes, all_scores, all_class_ids,
                          self.config.conf_threshold, self.config.nms_threshold,
                          orig_w, orig_h, self.config.min_box_size)

    def _postprocess_yolov8(
        self, ex, orig_w, orig_h, scale, pad_left, pad_top
    ) -> list[Detection]:
        try:
            _, out = ex.extract(self._out_layers[-1])
        except Exception:
            return []

        data = np.array(out)

        # shape 정규화: (4+cls, 8400) 또는 (8400, 4+cls)
        if data.ndim == 2:
            if data.shape[0] < data.shape[1]:  # (4+cls, 8400)
                data = data.T
        elif data.ndim == 1:
            n_cls = self._num_classes
            cols = 4 + n_cls
            if len(data) % cols == 0:
                data = data.reshape(-1, cols)
            else:
                return []

        cls_scores = data[:, 4:]
        max_scores = cls_scores.max(axis=1)
        class_ids = cls_scores.argmax(axis=1)
        mask = max_scores >= self.config.conf_threshold

        if not mask.any():
            return []

        s = self._input_size
        cx = data[mask, 0]
        cy = data[mask, 1]
        bw = data[mask, 2]
        bh = data[mask, 3]

        x1s = ((cx - bw / 2) - pad_left) / scale
        y1s = ((cy - bh / 2) - pad_top) / scale
        x2s = ((cx + bw / 2) - pad_left) / scale
        y2s = ((cy + bh / 2) - pad_top) / scale

        all_boxes = [[float(x1s[i]), float(y1s[i]),
                      float(x2s[i] - x1s[i]), float(y2s[i] - y1s[i])]
                     for i in range(mask.sum())]

        return _apply_nms(all_boxes, max_scores[mask].tolist(),
                          class_ids[mask].tolist(),
                          self.config.conf_threshold, self.config.nms_threshold,
                          orig_w, orig_h, self.config.min_box_size)

    @staticmethod
    def _probe_num_classes(net, input_size: int, out_layers: list[str], model_type: str) -> int:
        import ncnn
        try:
            dummy = np.zeros((input_size, input_size, 3), dtype=np.uint8)
            ex = net.create_extractor()
            mat_in = ncnn.Mat.from_pixels(dummy, ncnn.Mat.PixelType.PIXEL_BGR, input_size, input_size)
            ex.input("in0", mat_in)
            _, out = ex.extract(out_layers[0] if out_layers else "out0")
            arr = np.array(out)
            if arr.ndim == 3:
                gh, gw, c = arr.shape
                if model_type == "yolov8":
                    return c - 4
                else:
                    return (c // 3) - 5
        except Exception:
            pass
        return 1


# ── 헬퍼 함수 ──────────────────────────────────────────────────────────────────

def _letterbox(img: np.ndarray, target: int) -> tuple[np.ndarray, int, int, float]:
    h, w = img.shape[:2]
    scale = min(target / w, target / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_top = (target - new_h) // 2
    pad_left = (target - new_w) // 2
    pad_bottom = target - new_h - pad_top
    pad_right = target - new_w - pad_left
    out = cv2.copyMakeBorder(resized, pad_top, pad_bottom, pad_left, pad_right,
                              cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return out, pad_top, pad_left, scale


def _apply_nms(
    boxes: list, scores: list, class_ids: list,
    conf_threshold: float, nms_threshold: float,
    orig_w: int, orig_h: int,
    min_box_size: int = 10,
) -> list[Detection]:
    if not boxes:
        return []
    indices = cv2.dnn.NMSBoxes(boxes, scores, conf_threshold, nms_threshold)
    detections = []
    for idx in (indices.flatten() if hasattr(indices, "flatten") else indices):
        x, y, w, h = boxes[idx]
        x1 = max(0, int(x))
        y1 = max(0, int(y))
        x2 = min(orig_w, int(x + w))
        y2 = min(orig_h, int(y + h))
        # 너무 작은 박스 제거
        if (x2 - x1) < min_box_size or (y2 - y1) < min_box_size:
            continue
        detections.append(Detection(
            x1=x1, y1=y1, x2=x2, y2=y2,
            confidence=float(scores[idx]),
            class_id=int(class_ids[idx]),
        ))
    return detections


def _parse_layers(param_path: str) -> tuple[str, list[str]]:
    """
    param 파일에서 입력/출력 blob 이름을 파싱.
    출력 blob: "out"으로 시작하는 blob이 실제로 존재하는 레이어의 출력인 것만 수집.
    """
    in_layer = "in0"
    # 모든 출력 blob을 수집 (레이어 타입 무관)
    all_output_blobs: set[str] = set()
    all_input_blobs: set[str] = set()

    try:
        with open(param_path, "r") as f:
            lines = f.readlines()

        for line in lines[2:]:
            parts = line.strip().split()
            if len(parts) < 4:
                continue

            layer_type = parts[0]
            # num_inputs, num_outputs는 parts[2], parts[3]
            try:
                num_inputs = int(parts[2])
                num_outputs = int(parts[3])
            except (ValueError, IndexError):
                continue

            blob_start = 4
            input_blobs = parts[blob_start: blob_start + num_inputs]
            output_blobs = parts[blob_start + num_inputs: blob_start + num_inputs + num_outputs]

            if layer_type == "Input":
                if output_blobs:
                    in_layer = output_blobs[0]

            for b in input_blobs:
                all_input_blobs.add(b)
            for b in output_blobs:
                all_output_blobs.add(b)

    except Exception:
        pass

    # "out"으로 시작하는 blob 중 다른 레이어의 입력으로 쓰이지 않는 것 = 최종 출력
    terminal_outs = sorted(
        [b for b in all_output_blobs if b.startswith("out") and b not in all_input_blobs],
        key=lambda x: int(x[3:]) if x[3:].isdigit() else 99
    )

    if not terminal_outs:
        # fallback: "out"으로 시작하는 모든 output blob
        terminal_outs = sorted(
            [b for b in all_output_blobs if b.startswith("out")],
            key=lambda x: int(x[3:]) if x[3:].isdigit() else 99
        )

    if not terminal_outs:
        terminal_outs = ["out0"]

    return in_layer, terminal_outs


def _detect_model_type(param_path: str) -> str:
    try:
        with open(param_path, "r") as f:
            content = f.read()
        for line in content.splitlines():
            if "Reshape" in line and "8400" in line:
                return "yolov8"
    except Exception:
        pass
    return "yolov5"
