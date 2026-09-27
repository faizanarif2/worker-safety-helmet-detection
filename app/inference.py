"""Real, serialized YOLO inference; uploaded images and results stay in memory."""

from PIL import Image, ImageDraw, ImageFont

from app.config import DEFAULT_CONFIDENCE_THRESHOLD, HEAD_CONFLICT_IOU_THRESHOLD
from app.model import LoadedModel
from app.results import Detection, PredictionResult


class InferenceError(RuntimeError):
    pass


def box_iou(first: Detection, second: Detection) -> float:
    """Intersection over union; missing or zero-area boxes cannot conflict."""
    if first.xyxy is None or second.xyxy is None:
        return 0.0
    ax1, ay1, ax2, ay2 = first.xyxy
    bx1, by1, bx2, by2 = second.xyxy
    intersection = max(0, min(ax2, bx2) - max(ax1, bx1)) * max(0, min(ay2, by2) - max(ay1, by1))
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - intersection
    return intersection / union if union > 0 else 0.0


def resolve_head_conflicts(detections: tuple[Detection, ...]) -> tuple[Detection, ...]:
    """Keep the higher score for conflicting head labels with IoU >= 0.5.

    Greedy selection compares only retained head boxes. Person boxes and
    same-class detections are unchanged; YOLO already handles same-class NMS.
    Equal scores retain the first model result. Preserve original output order.
    """
    heads = [i for i, detection in enumerate(detections) if detection.class_name in {"helmet", "no_helmet"}]
    kept = []
    suppressed = set()
    for index in sorted(heads, key=lambda i: detections[i].confidence, reverse=True):
        candidate = detections[index]
        if any(candidate.class_name != detections[other].class_name
               and box_iou(candidate, detections[other]) >= HEAD_CONFLICT_IOU_THRESHOLD
               for other in kept):
            suppressed.add(index)
        else:
            kept.append(index)
    return tuple(detection for i, detection in enumerate(detections) if i not in suppressed)


def annotate(image: Image.Image, detections: tuple[Detection, ...]) -> Image.Image:
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    font = ImageFont.load_default(size=max(12, round(min(image.size) / 40)))
    colors = {"person": "#547e9a", "helmet": "#27835a", "no_helmet": "#c45132"}
    for detection in detections:
        color = colors[detection.class_name]
        x1, y1, x2, y2 = detection.xyxy
        draw.rectangle((x1, y1, x2, y2), outline=color, width=max(2, round(min(image.size) / 250)))
        label = f"{detection.class_name} {detection.confidence:.1%}"
        bounds = draw.textbbox((0, 0), label, font=font)
        width, height = bounds[2] - bounds[0] + 8, bounds[3] - bounds[1] + 8
        x = max(0, min(x1, image.width - width))
        y = max(0, y1 - height)
        draw.rectangle((x, y, x + width, y + height), fill=color)
        draw.text((x + 4, y + 4 - bounds[1]), label, fill="white", font=font)
    return annotated


def predict(image: Image.Image, resource: LoadedModel) -> PredictionResult:
    try:
        with resource.lock:
            try:
                results = resource.model.predict(
                    source=image, conf=DEFAULT_CONFIDENCE_THRESHOLD,
                    device=resource.device, imgsz=640, stream=False,
                    save=False, save_txt=False, save_crop=False, show=False,
                    verbose=False, augment=False,
                )
                if len(results) != 1 or results[0].boxes is None:
                    raise ValueError("No detection result was returned")
                boxes = results[0].boxes
                detections = tuple(
                    Detection(resource.names[int(class_id)], float(score), tuple(float(v) for v in coords))
                    for class_id, score, coords in zip(
                        boxes.cls.cpu().tolist(), boxes.conf.cpu().tolist(),
                        boxes.xyxy.cpu().tolist(), strict=True,
                    )
                )
            finally:
                # The cached resource must not retain the last visitor's image
                # via Ultralytics' predictor. Model weights remain cached.
                resource.model.predictor = None
        detections = resolve_head_conflicts(detections)
        return PredictionResult(detections, annotate(image, detections))
    except Exception as exc:
        raise InferenceError("Detection could not finish. Try a smaller image or retry. If it persists, check the model configuration.") from exc
