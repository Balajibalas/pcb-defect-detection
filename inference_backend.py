import os
import json
import cv2
import numpy as np
import torch
import timm
from skimage.metrics import structural_similarity as ssim
import albumentations as A
from albumentations.pytorch import ToTensorV2

# --------------------- CONFIG ---------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "best_model.pth")
CLASS_JSON = os.path.join(BASE_DIR, "class_names.json")
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")

IMG_SIZE = 160
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONF_THRESH = 0.80
AREA_THRESH = 200
# --------------------------------------------------



# ---------- Load Classes ----------
def load_classes():
    if not os.path.exists(CLASS_JSON):
        raise FileNotFoundError(f"class_names.json not found: {CLASS_JSON}")

    with open(CLASS_JSON, "r") as f:
        return json.load(f)


# ---------- Load Model ----------
_model = None
_classes = None

def create_model(num_classes):
    return timm.create_model(
        "tf_efficientnet_b1_ns",
        pretrained=False,
        num_classes=num_classes
    )


def load_model():
    global _model, _classes

    if _model is not None:
        return _model, _classes

    _classes = load_classes()
    model = create_model(len(_classes))

    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    _model = model
    return _model, _classes


# ---------- Albumentations ----------
roi_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(),
    ToTensorV2()
])


def preprocess_roi(img_rgb):
    return roi_transform(image=img_rgb)["image"].unsqueeze(0)


def classify_roi(model, classes, roi_rgb):
    tensor = preprocess_roi(roi_rgb).to(DEVICE)

    with torch.no_grad():
        out = model(tensor)
        probs = torch.softmax(out, dim=1)
        conf, idx = torch.max(probs, dim=1)

    return classes[idx.item()], float(conf.item())


# ---------- Auto Template Utilities ----------
def load_all_templates():
    if not os.path.exists(TEMPLATE_DIR):
        return [], []

    templates, paths = [], []

    for f in sorted(os.listdir(TEMPLATE_DIR)):
        if f.lower().endswith(("jpg", "jpeg", "png")):
            path = os.path.join(TEMPLATE_DIR, f)
            img = cv2.imread(path)
            if img is not None:
                templates.append(img)
                paths.append(path)

    return templates, paths


def find_best_template(test_bgr):
    templates, paths = load_all_templates()

    if len(templates) == 0:
        raise ValueError("No templates found for auto selection.")

    test_gray = cv2.cvtColor(test_bgr, cv2.COLOR_BGR2GRAY)

    best_score = -1
    best_tpl = None
    best_path = None

    for tpl, path in zip(templates, paths):
        tpl_gray = cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY)
        resized_test = cv2.resize(test_gray, (tpl.shape[1], tpl.shape[0]))

        score = ssim(tpl_gray, resized_test)

        if score > best_score:
            best_score = score
            best_tpl = tpl
            best_path = path

    return best_tpl, best_path, best_score


# ---------- Core Defect Detection ----------
def detect_defects(model, classes, template_bgr, test_bgr):

    test_bgr = cv2.resize(test_bgr, (template_bgr.shape[1], template_bgr.shape[0]))

    gray_tpl = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)
    gray_test = cv2.cvtColor(test_bgr, cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(gray_tpl, gray_test)
    _, th = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((5, 5), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)
    th = cv2.dilate(th, kernel, iterations=1)

    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections = []

    for cnt in contours:
        if cv2.contourArea(cnt) < AREA_THRESH:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        roi = test_bgr[y:y+h, x:x+w]
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

        label, conf = classify_roi(model, classes, roi_rgb)

        if conf >= CONF_THRESH:
            detections.append({
                "x1": x, "y1": y, "x2": x+w, "y2": y+h,
                "label": label, "conf": conf
            })

    return detections, th


def annotate_image(img_bgr, detections):
    out = img_bgr.copy()

    for d in detections:
        cv2.rectangle(out, (d["x1"], d["y1"]), (d["x2"], d["y2"]), (0,0,255), 2)
        cv2.putText(out, f'{d["label"]} {d["conf"]:.2f}',
                    (d["x1"], d["y1"]-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    return out


# =================================================
# =============== INFERENCE MODES ==================
# =================================================

# ---------- AUTO TEMPLATE MODE ----------
def infer_pcb_from_array(test_bgr):
    model, classes = load_model()

    if test_bgr is None:
        raise ValueError("Invalid PCB image.")

    template, path, score = find_best_template(test_bgr)

    if score < 0.80:
        raise ValueError("Template similarity too low.")

    detections, mask = detect_defects(model, classes, template, test_bgr)
    annotated = annotate_image(test_bgr, detections)

    return annotated, detections, mask, path


# ---------- MANUAL TEMPLATE UPLOAD MODE ----------
def infer_with_uploaded_template(test_bgr, template_bgr):
    model, classes = load_model()

    if test_bgr is None or template_bgr is None:
        raise ValueError("Invalid PCB or template image.")

    detections, mask = detect_defects(model, classes, template_bgr, test_bgr)
    annotated = annotate_image(test_bgr, detections)

    return annotated, detections, mask





