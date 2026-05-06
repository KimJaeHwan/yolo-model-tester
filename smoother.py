from dataclasses import dataclass, field
from inference import Detection


@dataclass
class TrackedBox:
    det: Detection
    hit_count: int = 1      # 연속 감지 횟수
    miss_count: int = 0     # 연속 미감지 횟수


class DetectionSmoother:
    """
    IOU 기반 시간적 스무딩.
    - confirm_frames 프레임 연속 감지되면 표시 시작
    - miss_frames  프레임 연속 미감지되면 표시 중단
    """

    def __init__(self, confirm_frames: int = 2, miss_frames: int = 3, iou_threshold: float = 0.3):
        self.confirm_frames = confirm_frames
        self.miss_frames = miss_frames
        self.iou_threshold = iou_threshold
        self._tracked: list[TrackedBox] = []

    def update(self, detections: list[Detection]) -> list[Detection]:
        # 기존 트랙과 새 탐지 매칭
        matched_track = set()
        matched_det = set()

        for ti, track in enumerate(self._tracked):
            best_iou = 0.0
            best_di = -1
            for di, det in enumerate(detections):
                if di in matched_det:
                    continue
                if det.class_id != track.det.class_id:
                    continue
                iou = _iou(track.det, det)
                if iou > best_iou:
                    best_iou = iou
                    best_di = di

            if best_iou >= self.iou_threshold and best_di >= 0:
                # 매칭 성공 → 위치 갱신
                self._tracked[ti].det = detections[best_di]
                self._tracked[ti].hit_count += 1
                self._tracked[ti].miss_count = 0
                matched_track.add(ti)
                matched_det.add(best_di)
            else:
                # 매칭 실패 → miss 카운트 증가
                self._tracked[ti].miss_count += 1
                self._tracked[ti].hit_count = 0

        # 새로 등장한 탐지 추가
        for di, det in enumerate(detections):
            if di not in matched_det:
                self._tracked.append(TrackedBox(det=det))

        # miss_frames 초과 트랙 제거
        self._tracked = [t for t in self._tracked if t.miss_count < self.miss_frames]

        # confirm_frames 이상 감지된 것만 반환
        return [t.det for t in self._tracked if t.hit_count >= self.confirm_frames]

    def reset(self) -> None:
        self._tracked.clear()


def _iou(a: Detection, b: Detection) -> float:
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a.x2 - a.x1) * (a.y2 - a.y1)
    area_b = (b.x2 - b.x1) * (b.y2 - b.y1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0
