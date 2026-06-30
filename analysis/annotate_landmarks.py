"""
Produces 7 annotated copies of person1.jpg, one per formula.
Run from t:\\thesis:
    python.exe analysis/annotate_landmarks.py
"""
import sys
sys.path.insert(0, "feat-extraction")

import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ── Setup ─────────────────────────────────────────────────────────────────────
LANDMARKER_PATH = os.path.join("pretrained_models", "face_landmarker.task")
INPUT_IMAGE     = "person1.jpg"
OUTPUT_DIR      = "report"

base_options = python.BaseOptions(model_asset_path=LANDMARKER_PATH)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=False,
    num_faces=1,
)
landmarker = vision.FaceLandmarker.create_from_options(options)

img_bgr = cv2.imread(INPUT_IMAGE)
h, w    = img_bgr.shape[:2]

mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                    data=cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
result   = landmarker.detect(mp_image)
lm       = result.face_landmarks[0]

def pt(idx):
    return (int(lm[idx].x * w), int(lm[idx].y * h))

# ── Colours ───────────────────────────────────────────────────────────────────
RED    = (0,   0,   255)
BLUE   = (255, 80,  0  )
GREEN  = (0,   210, 0  )
WHITE  = (255, 255, 255)
ORANGE = (0,   180, 255)
GREY   = (160, 160, 160)
CB_BLUE   = (178, 114,   0)   # Okabe-Ito blue   #0072B2
CB_ORANGE = (  0, 159, 230)   # Okabe-Ito orange #E69F00

# ── Typography ────────────────────────────────────────────────────────────────
FONT = cv2.FONT_HERSHEY_SIMPLEX
FS   = 0.9    # font scale
FT   = 2      # font thickness
DOT_R  = 7
LINE_W = 2

def _tw(text):
    """Text pixel width."""
    return cv2.getTextSize(text, FONT, FS, FT)[0][0]

def _th():
    """Text pixel height (approximate)."""
    return cv2.getTextSize("A", FONT, FS, FT)[0][1]

def _text(img, text, pos, color):
    (tw, th), baseline = cv2.getTextSize(text, FONT, FS, FT)
    x, y = pos
    pad = 3
    cv2.rectangle(img, (x - pad, y - th - pad), (x + tw + pad, y + baseline + pad),
                  (255, 255, 255), -1)
    cv2.putText(img, text, pos, FONT, FS, color, FT, cv2.LINE_AA)

def dot(img, idx, color, r=DOT_R):
    p = pt(idx)
    cv2.circle(img, p, r, color, -1)
    cv2.circle(img, p, r, WHITE, 1)

def line(img, idx_a, idx_b, color, thickness=LINE_W):
    cv2.line(img, pt(idx_a), pt(idx_b), color, thickness, cv2.LINE_AA)

def dashed_line(img, idx_a, idx_b, color, dash=10, gap=6):
    p1, p2 = np.array(pt(idx_a)), np.array(pt(idx_b))
    dist    = np.linalg.norm(p2 - p1)
    steps   = int(dist / (dash + gap))
    for i in range(steps + 1):
        s  = min(i * (dash + gap),        dist)
        e  = min(i * (dash + gap) + dash, dist)
        ps = tuple((p1 + (p2 - p1) * s / dist).astype(int))
        pe = tuple((p1 + (p2 - p1) * e / dist).astype(int))
        cv2.line(img, ps, pe, color, LINE_W, cv2.LINE_AA)

def filled_polygon(img, indices, color, alpha=0.25):
    pts     = np.array([pt(i) for i in indices], dtype=np.int32)
    overlay = img.copy()
    cv2.fillPoly(overlay, [pts], color)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    cv2.polylines(img, [pts], isClosed=True, color=color,
                  thickness=LINE_W, lineType=cv2.LINE_AA)

def label_below(img, idx, color, text):
    """Text centred below the dot."""
    p  = pt(idx)
    lx = max(4, min(p[0] - _tw(text) // 2, w - _tw(text) - 4))
    ly = p[1] + _th() + 12
    _text(img, text, (lx, ly), color)

def label_side(img, idx, color, text):
    """Text to the right or left depending on which image half the dot is in."""
    p  = pt(idx)
    lx = p[0] - _tw(text) - 12 if p[0] > w // 2 else p[0] + 12
    _text(img, text, (lx, p[1] + 6), color)

def callout(img, color, text, anchor, tip, extra_gap=10):
    """
    Draw text above anchor, then an arrow whose tail starts just outside the
    text bounding box (+ extra_gap px) so it never overlaps the label.
    """
    tw = _tw(text)
    th = _th()
    pad = 3
    # text box top-left and bottom-right
    tx = max(4, min(anchor[0] - tw // 2, w - tw - 4))
    ty = anchor[1] - 6          # baseline
    box_x0, box_y0 = tx - pad,       ty - th - pad
    box_x1, box_y1 = tx + tw + pad,  ty + pad + 4
    _text(img, text, (tx, ty), color)

    # walk along the arrow direction until we step outside the text box
    direction = np.array(tip, dtype=float) - np.array(anchor, dtype=float)
    dist = np.linalg.norm(direction)
    if dist == 0:
        return
    unit = direction / dist
    # find t (in pixels) where the ray from anchor exits the box
    t_exit = 0.0
    for t in range(0, int(dist), 2):
        px = anchor[0] + unit[0] * t
        py = anchor[1] + unit[1] * t
        if not (box_x0 <= px <= box_x1 and box_y0 <= py <= box_y1):
            t_exit = t
            break
    tail_gap = t_exit + extra_gap
    if dist > tail_gap:
        start = tuple((np.array(anchor) + unit * tail_gap).astype(int))
    else:
        start = anchor
    cv2.arrowedLine(img, start, tip, color, 2, cv2.LINE_AA, tipLength=0.12)

def save(img, name):
    path = os.path.join(OUTPUT_DIR, name)
    cv2.imwrite(path, img)
    print(f"Saved {path}")

def fresh():
    return img_bgr.copy()


# ── 1. IOD ────────────────────────────────────────────────────────────────────
img = fresh()
dot(img, 33,  RED)
dot(img, 263, RED)
line(img, 33, 263, RED)
label_below(img, 33,  RED, "left eye outer corner")
label_below(img, 263, RED, "right eye outer corner")
save(img, "fig_iod.jpg")


# ── 2. Lip Distance ───────────────────────────────────────────────────────────
img = fresh()
dot(img, 33,  GREY)
dot(img, 263, GREY)
dashed_line(img, 33, 263, GREY)
dot(img, 13, RED)
dot(img, 14, RED)
line(img, 13, 14, RED)
label_side(img, 13, RED, "upper lip center")
label_side(img, 14, RED, "lower lip center")
save(img, "fig_lip_distance.jpg")


# ── 3. Mouth Width ────────────────────────────────────────────────────────────
img = fresh()
dot(img, 33,  GREY)
dot(img, 263, GREY)
dashed_line(img, 33, 263, GREY)
dot(img, 78,  BLUE)
dot(img, 308, BLUE)
line(img, 78, 308, BLUE)
save(img, "fig_mouth_width.jpg")


# ── 4. MAR ────────────────────────────────────────────────────────────────────
img = fresh()
dot(img, 78,  BLUE)
dot(img, 308, BLUE)
line(img, 78, 308, BLUE)
for top, bot, col in [
    (81,  178, RED),
    (13,  14,  GREEN),
    (311, 402, ORANGE),
]:
    dot(img, top, col)
    dot(img, bot, col)
    line(img, top, bot, col)
save(img, "fig_mar.jpg")


# ── 5. Lip Curvature ─────────────────────────────────────────────────────────
img = fresh()
dot(img, 78,  RED)
dot(img, 308, RED)
dot(img, 13,  BLUE)
dot(img, 14,  BLUE)

# dotted line between the two blue lip-centre dots, green dot at midpoint
p13 = pt(13); p14 = pt(14)
p1n, p2n = np.array(p13), np.array(p14)
dist_c = np.linalg.norm(p2n - p1n)
dash, gap = 8, 5
steps = int(dist_c / (dash + gap))
for i in range(steps + 1):
    s  = min(i * (dash + gap),        dist_c)
    e  = min(i * (dash + gap) + dash, dist_c)
    ps = tuple((p1n + (p2n - p1n) * s / dist_c).astype(int))
    pe = tuple((p1n + (p2n - p1n) * e / dist_c).astype(int))
    cv2.line(img, ps, pe, GREEN, LINE_W, cv2.LINE_AA)
mid = tuple(((p1n + p2n) / 2).astype(int))
cv2.circle(img, mid, DOT_R, GREEN, -1)
cv2.circle(img, mid, DOT_R, WHITE, 1)

# ── callouts — anchors spread to avoid overlap ────────────────────────────────
p78  = pt(78);  p308 = pt(308)

# left corner
callout(img, RED,  "left corner",
        anchor=(90, p78[1]),
        tip=p78)

# right corner
callout(img, RED,  "right corner",
        anchor=(w - 90, p308[1] + 80),
        tip=p308)

# upper lip center
callout(img, BLUE, "upper lip center",
        anchor=(w - 90, p13[1] - 80),
        tip=p13)

# lower lip center
callout(img, BLUE, "lower lip center",
        anchor=(90, p14[1] + 100),
        tip=p14)

save(img, "fig_lip_curvature.jpg")


# ── 6. Mouth Area ─────────────────────────────────────────────────────────────
INNER_LIP = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
             308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
img = fresh()
filled_polygon(img, INNER_LIP, BLUE, alpha=0.3)
for idx in INNER_LIP:
    dot(img, idx, BLUE, r=4)
save(img, "fig_mouth_area.jpg")


# ── 7. Mouth Perimeter ───────────────────────────────────────────────────────
img = fresh()
pts_list = [pt(i) for i in INNER_LIP]
for i in range(len(pts_list)):
    p1  = pts_list[i]
    p2  = pts_list[(i + 1) % len(pts_list)]
    col = CB_BLUE if i % 2 == 0 else CB_ORANGE
    cv2.line(img, p1, p2, col, LINE_W + 1, cv2.LINE_AA)
for idx in INNER_LIP:
    p = pt(idx)
    cv2.circle(img, p, 4, WHITE, -1)
    cv2.circle(img, p, 4, GREY,  1)
save(img, "fig_mouth_circularity.jpg")

print("All done.")
