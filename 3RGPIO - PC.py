# -*- coding: utf-8 -*-
"""
Python 3.6 Compatible
Saves detections to JSON + Summary Statistics
No Excel used
"""

import cv2
import numpy as np
from datetime import datetime
import json
import os

# ----------------------------------------------------
# JSON Setup
# ----------------------------------------------------
JSON_FILE = "results.json"

# If no file exists yet → create empty structure
if not os.path.exists(JSON_FILE):
    data = {
        "detections": [],
        "summary": {
            "total_detected": 0,
            "recyclable": 0,
            "non_recyclable": 0
        }
    }
    with open(JSON_FILE, "w") as f:
        json.dump(data, f, indent=4)
else:
    with open(JSON_FILE, "r") as f:
        data = json.load(f)

# Shortcut to summary
summary = data["summary"]

# ----------------------------------------------------
# Dummy items database (color → item type)
# ----------------------------------------------------
ITEMS_DB = {
    "green": ("plastic bottle", "recyclable", 0.85),
    "white": ("styrofoam", "non recyclable", 0.7),
    "yellow": ("can", "recyclable", 0.8),
    "brown": ("paper", "recyclable", 0.9)
}

# ----------------------------------------------------
# Camera Init (Laptop webcam)
# ----------------------------------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise Exception("Laptop webcam not found. Try index 0, 1, or 2.")

print("System ready. Press 's' to stop.")

object_id = summary["total_detected"]

# ----------------------------------------------------
# Color ranges (HSV)
# ----------------------------------------------------
COLORS = {
    "green": ([30, 40, 40], [90, 255, 255]),
    "white": ([0, 0, 200], [180, 30, 255]),
    "yellow": ([20, 100, 100], [30, 255, 255]),
    "brown": ([10, 100, 20], [20, 255, 200])
}

# ----------------------------------------------------
# Main Loop
# ----------------------------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("Could not read frame.")
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    detected_objects = []

    for color_name, (lower, upper) in COLORS.items():
        lower = np.array(lower)
        upper = np.array(upper)
        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.GaussianBlur(mask, (7, 7), 0)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            if cv2.contourArea(c) > 500:
                x, y, w, h = cv2.boundingRect(c)
                item_name, label, confidence = ITEMS_DB[color_name]

                detected_objects.append({
                    "box": (x, y, w, h),
                    "item": item_name,
                    "label": label,
                    "confidence": confidence
                })

    # ------------------------------------------------
    # Process & Save Detections
    # ------------------------------------------------
    for obj in detected_objects:
        object_id += 1
        x, y, w, h = obj["box"]
        item = obj["item"]
        label = obj["label"]
        confidence = obj["confidence"]
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Draw box
        box_color = (0, 255, 0) if label == "recyclable" else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)

        cv2.putText(frame, "{}: {}".format(label.upper(), item.upper()), 
                    (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)

        cv2.putText(frame, "CONFIDENCE: {:.1f}%".format(confidence * 100),
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 255), 2)

        # ------------------------
        # Save to JSON
        # ------------------------
        entry = {
            "id": object_id,
            "item": item,
            "label": label,
            "confidence": confidence,
            "timestamp": timestamp
        }

        data["detections"].append(entry)

        # Update summary counts
        summary["total_detected"] += 1
        if label == "recyclable":
            summary["recyclable"] += 1
        else:
            summary["non_recyclable"] += 1

        # Save JSON file
        with open(JSON_FILE, "w") as f:
            json.dump(data, f, indent=4)

        print("Object #{} → {} → {} → {:.1f}%".format(
            object_id, item, label, confidence * 100
        ))

    cv2.imshow("Recyclability System", frame)

    key = cv2.waitKey(1)
    if key == ord('s') or key == 27:  # ESC or 's'
        print("Stopping stream...")
        break

cap.release()
cv2.destroyAllWindows()

# Final summary shown on exit
print("\n===== SUMMARY =====")
print("Total detected:", summary["total_detected"])
print("Recyclable:", summary["recyclable"])
print("Non-recyclable:", summary["non_recyclable"])
print("====================\n")

print("Program terminated. JSON saved as results.json")
