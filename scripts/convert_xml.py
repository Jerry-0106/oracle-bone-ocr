#!/usr/bin/env python3
"""
Convert XML annotations to YOLO txt format.
All characters treated as class_id=0 (class_name=character).

Handles:
- UTF-16 XML encoding
- Polygon positions (semicolons in position attribute)
- Missing images/XMLs
- Empty annotations
- Abnormal coordinates (out of bounds, zero-area boxes)

Logs all issues to ./logs/
"""

import os
import re
import sys
from pathlib import Path
from datetime import datetime
from xml.etree import ElementTree as ET

PROJECT_ROOT = Path(__file__).resolve().parent
XML_DIR = PROJECT_ROOT / "data" / "raw"
LABELS_DIR = PROJECT_ROOT / "data" / "labels"
LOGS_DIR = PROJECT_ROOT / "logs"

CLASS_ID = 0
CLASS_NAME = "character"

# Statistics and logs
stats = {
    "total_xml": 0,
    "total_png": 0,
    "matched": 0,
    "missing_xml": 0,
    "missing_png": 0,
    "empty_annotations": 0,
    "abnormal_coords": 0,
    "polygon_converted": 0,
    "total_chars": 0,
    "parse_errors": 0,
}

issue_log = []


def log_issue(level, msg):
    """Log an issue with timestamp."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"[{ts}] [{level}] {msg}"
    issue_log.append(entry)
    if level in ("ERROR", "WARN"):
        print(f"  {level}: {msg}")


def parse_position(pos_str):
    """
    Parse position string. Handles:
    - Simple rect: "x1,y1,x2,y2"
    - Polygon: "x1,y1;x2,y2;x3,y3;..."

    Returns (x_center, y_center, width, height) normalized to [0,1].
    Returns None if invalid.
    """
    if not pos_str or not pos_str.strip():
        return None

    pos_str = pos_str.strip()

    if ";" in pos_str:
        # Polygon format: convert to bounding box
        stats["polygon_converted"] += 1
        coords = []
        for pt in pos_str.split(";"):
            pt = pt.strip()
            if not pt:
                continue
            parts = pt.split(",")
            if len(parts) >= 2:
                try:
                    coords.append((float(parts[0]), float(parts[1])))
                except ValueError:
                    return None
        if not coords:
            return None
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
    else:
        # Simple rect format: "x1,y1,x2,y2"
        parts = pos_str.split(",")
        if len(parts) < 4:
            return None
        try:
            xmin = float(parts[0])
            ymin = float(parts[1])
            xmax = float(parts[2])
            ymax = float(parts[3])
        except ValueError:
            return None

    return (xmin, ymin, xmax, ymax)


def convert_to_yolo(xmin, ymin, xmax, ymax, img_w, img_h):
    """
    Convert pixel coordinates to YOLO normalized format.
    Clips coordinates to image boundaries.
    Returns (x_center, y_center, width, height) all in [0,1].
    """
    # Clip to image boundaries
    xmin = max(0, min(xmin, img_w))
    ymin = max(0, min(ymin, img_h))
    xmax = max(0, min(xmax, img_w))
    ymax = max(0, min(ymax, img_h))

    # Ensure non-zero area after clipping
    if xmax <= xmin or ymax <= ymin:
        return None

    x_center = ((xmin + xmax) / 2.0) / img_w
    y_center = ((ymin + ymax) / 2.0) / img_h
    width = (xmax - xmin) / img_w
    height = (ymax - ymin) / img_h
    return (x_center, y_center, width, height)


def validate_bbox(x, y, w, h, img_w, img_h, xml_name):
    """Check for abnormal boxes. Returns True if valid."""
    issues = []

    if w <= 0 or h <= 0:
        issues.append(f"zero or negative size after clip: w={w:.4f}, h={h:.4f}")
    # Check for tiny boxes (less than 2 pixels in either dimension)
    if w > 0 and h > 0 and (w * img_w < 2 or h * img_h < 2):
        issues.append(f"very small box: {w*img_w:.1f}x{h*img_h:.1f} px")

    if issues:
        stats["abnormal_coords"] += 1
        for issue in issues:
            log_issue("WARN", f"{xml_name}: {issue}")
        if w <= 0 or h <= 0:
            return False
    return True


def parse_xml(xml_path):
    """Parse a single XML file. Returns list of (class_id, xc, yc, w, h) or None on error."""
    try:
        # Read with utf-16 encoding
        with open(xml_path, "r", encoding="utf-16") as f:
            content = f.read()

        # Handle potential BOM and encoding declaration
        root = ET.fromstring(content)
    except Exception as e:
        log_issue("ERROR", f"Failed to parse {xml_path.name}: {e}")
        stats["parse_errors"] += 1
        return None

    img_w = int(root.get("width", 0))
    img_h = int(root.get("height", 0))

    if img_w <= 0 or img_h <= 0:
        log_issue("ERROR", f"{xml_path.name}: invalid image dimensions {img_w}x{img_h}")
        stats["parse_errors"] += 1
        return None

    annotations = []

    # Find all <char> elements
    for char_elem in root.iter("char"):
        pos_str = char_elem.get("position", "")
        bbox = parse_position(pos_str)

        if bbox is None:
            log_issue("WARN", f"{xml_path.name}: char id={char_elem.get('id')}: invalid position '{pos_str[:50]}'")
            stats["abnormal_coords"] += 1
            continue

        xmin, ymin, xmax, ymax = bbox
        yolo_bbox = convert_to_yolo(xmin, ymin, xmax, ymax, img_w, img_h)

        if yolo_bbox is None:
            stats["abnormal_coords"] += 1
            continue

        if validate_bbox(*yolo_bbox, img_w, img_h, xml_path.name):
            annotations.append((CLASS_ID,) + yolo_bbox)
            stats["total_chars"] += 1

    return annotations


def process_all():
    """Main processing function."""
    print("=" * 60)
    print("XML to YOLO Conversion")
    print("=" * 60)

    # Setup directories
    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    # Get all files
    xml_files = sorted([f for f in os.listdir(XML_DIR) if f.lower().endswith(".xml")])
    png_files = set(f for f in os.listdir(XML_DIR) if f.lower().endswith(".png"))

    stats["total_xml"] = len(xml_files)
    stats["total_png"] = len(png_files)

    print(f"\nFound {stats['total_xml']} XML files")
    print(f"Found {stats['total_png']} PNG files")

    # Process each XML
    for i, xml_name in enumerate(xml_files):
        if (i + 1) % 500 == 0:
            print(f"  Processing... {i+1}/{len(xml_files)}")

        xml_path = XML_DIR / xml_name
        base_name = xml_name.rsplit(".", 1)[0]

        # Find matching PNG
        png_candidates = [base_name + ext for ext in [".png", ".PNG", ".jpg", ".JPG"]]
        matched_png = None
        for candidate in png_candidates:
            if candidate in png_files:
                matched_png = candidate
                break

        if matched_png is None:
            log_issue("WARN", f"No matching image for {xml_name}")
            stats["missing_png"] += 1
            # Still process XML - we just won't have the image
            continue

        # Parse XML
        annotations = parse_xml(xml_path)
        stats["matched"] += 1

        if annotations is None:
            continue

        if len(annotations) == 0:
            stats["empty_annotations"] += 1
            log_issue("INFO", f"{xml_name}: no valid character annotations found")

        # Write YOLO label file
        label_path = LABELS_DIR / (base_name + ".txt")
        with open(label_path, "w", encoding="utf-8") as f:
            for ann in annotations:
                f.write(f"{ann[0]} {ann[1]:.6f} {ann[2]:.6f} {ann[3]:.6f} {ann[4]:.6f}\n")

    # Check for PNGs without XMLs
    xml_basenames = set(f.rsplit(".", 1)[0] for f in xml_files)
    for png_name in sorted(png_files):
        base = png_name.rsplit(".", 1)[0]
        if base not in xml_basenames:
            log_issue("WARN", f"No matching XML for {png_name}")
            stats["missing_xml"] += 1

    # Print summary
    print("\n" + "=" * 60)
    print("Conversion Summary")
    print("=" * 60)
    print(f"  Total XML files:        {stats['total_xml']}")
    print(f"  Total PNG files:        {stats['total_png']}")
    print(f"  Successfully matched:   {stats['matched']}")
    print(f"  Missing XML:            {stats['missing_xml']}")
    print(f"  Missing PNG:            {stats['missing_png']}")
    print(f"  Empty annotations:      {stats['empty_annotations']}")
    print(f"  Abnormal coordinates:   {stats['abnormal_coords']}")
    print(f"  Polygon conversions:    {stats['polygon_converted']}")
    print(f"  Total characters:       {stats['total_chars']}")
    print(f"  Parse errors:           {stats['parse_errors']}")

    # Write logs
    log_path = LOGS_DIR / "convert_xml.log"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"Conversion run: {datetime.now().isoformat()}\n")
        f.write(f"\nSummary:\n")
        for key, val in stats.items():
            f.write(f"  {key}: {val}\n")
        f.write(f"\nIssues ({len(issue_log)}):\n")
        for entry in issue_log:
            f.write(entry + "\n")

    print(f"\nDetailed log saved to: {log_path}")
    print(f"Labels saved to: {LABELS_DIR}")

    return stats


if __name__ == "__main__":
    process_all()
