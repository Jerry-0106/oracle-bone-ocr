"""
Visualization for OCR Pipeline Output.
Draws bounding boxes with Chinese character labels on images.
"""

from pathlib import Path
import cv2
import numpy as np


# Try to find a CJK-capable font; fall back to OpenCV default
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SIZE = 0.6
FONT_THICK = 2
BOX_THICK = 2
COLORS = [
    (0, 255, 0),    # green
    (255, 0, 0),    # blue
    (0, 0, 255),    # red
    (255, 255, 0),  # cyan
    (255, 0, 255),  # magenta
    (0, 255, 255),  # yellow
]


def can_render(text):
    """Check if text can be rendered with default font (ASCII-safe fallback)."""
    try:
        text.encode('ascii')
        return True
    except UnicodeEncodeError:
        return False


def draw_detections(img, detections, show_conf=True, show_topn=1, color_by_conf=False):
    """
    Draw bounding boxes and labels on an image.

    Args:
        img: numpy array (H, W, 3) BGR
        detections: list of dicts with bbox, rec results
        show_conf: show confidence score
        show_topn: show top-N predictions
        color_by_conf: color by recognition confidence (green=high, red=low)

    Returns:
        annotated image (numpy array)
    """
    vis = img.copy()
    img_h, img_w = vis.shape[:2]

    for i, det in enumerate(detections):
        bbox = det["bbox"]
        x1, y1, x2, y2 = bbox
        rec = det.get("recognition", {})

        # Color
        if color_by_conf and rec:
            rec_conf = rec.get("top1", {}).get("confidence", 0.5)
            color = _conf_color(rec_conf)
        else:
            color = COLORS[i % len(COLORS)]

        # Draw bbox
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, BOX_THICK)

        # Build label
        label_parts = []
        if rec:
            top1 = rec.get("top1", {})
            char = top1.get("char", "?")
            if can_render(char):
                label = char
            else:
                label = top1.get("class_id", "?")

            rec_conf = top1.get("confidence", 0)
            det_conf = det.get("det_confidence", 0)

            if show_conf:
                label_parts.append(f"{label} d:{det_conf:.2f} r:{rec_conf:.2f}")
            else:
                label_parts.append(label)

            # Show top-2..top-N
            if show_topn > 1 and "top5" in rec:
                for alt in rec["top5"][1:show_topn]:
                    alt_char = alt.get("char", "?")
                    alt_conf = alt.get("confidence", 0)
                    label_parts.append(f"  {alt_char} ({alt_conf:.2f})")
        else:
            label_parts.append(f"d:{det.get('det_confidence', 0):.2f}")

        label = " ".join(label_parts)

        # Draw label background
        (lw, lh), lb = cv2.getTextSize(label, FONT, FONT_SIZE, FONT_THICK)
        label_y = max(y1 - 8, lh + 4)
        cv2.rectangle(vis, (x1, label_y - lh - 4), (x1 + lw + 4, label_y + 2), color, -1)
        cv2.putText(vis, label, (x1 + 2, label_y), FONT, FONT_SIZE, (255, 255, 255), FONT_THICK)

    return vis


def draw_summary(vis, stats, margin=10):
    """Draw summary statistics in the top-left corner."""
    lines = [
        f"Detections: {stats.get('total_detections', 0)}",
        f"Avg det conf: {stats.get('avg_det_conf', 0):.3f}",
        f"Avg rec conf: {stats.get('avg_rec_conf', 0):.3f}",
    ]
    y = margin + 15
    for line in lines:
        cv2.putText(vis, line, (margin, y), FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        y += 18
    return vis


def _conf_color(conf):
    """Map confidence to color: green (1.0) → yellow (0.5) → red (0.0)."""
    conf = max(0, min(1, conf))
    if conf > 0.66:
        return (0, int(255 * (1 - conf) / 0.34), 255)
    elif conf > 0.33:
        return (0, 255, int(255 * (conf - 0.33) / 0.33))
    else:
        return (int(255 * (1 - conf / 0.33)), 255, 0)
