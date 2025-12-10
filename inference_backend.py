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
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best_model.pth")
CLASS_JSON = os.path.join(BASE_DIR, "class_names.json")

TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")   # folder with 10 templates

IMG_SIZE = 160
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONF_THRESH = 0.80
AREA_THRESH = 200
# --------------------------------------------------


# ---------- Load Model ----------
def load_classes():
    with open(CLASS_JSON, "r") as f:
        return json.load(f)


def create_model(num_classes):
    return timm.create_model(
        "tf_efficientnet_b7_ns",
        pretrained=False,
        num_classes=num_classes
    )


def load_model():
    classes = load_classes()
    model = create_model(len(classes))

    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    return model, classes


# ---------- Albumentations transform ----------
roi_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(),
    ToTensorV2()
])


def preprocess_roi(img_rgb):
    tensor = roi_transform(image=img_rgb)["image"].unsqueeze(0)
    return tensor


def classify_roi(model, classes, roi_rgb):
    tensor = preprocess_roi(roi_rgb).to(DEVICE)

    with torch.no_grad():
        out = model(tensor)
        probs = torch.softmax(out, dim=1)
        conf, idx = torch.max(probs, dim=1)

    return classes[idx.item()], float(conf.item())


# ---------- Template Selection ----------
def load_all_templates():
    templates = []
    template_paths = []

    for file in sorted(os.listdir(TEMPLATE_DIR)):
        if file.lower().endswith(("jpg", "jpeg", "png")):
            path = os.path.join(TEMPLATE_DIR, file)
            img = cv2.imread(path)
            if img is not None:
                templates.append(img)
                template_paths.append(path)

    return templates, template_paths


def find_best_template(test_bgr):
    """
    Computes SSIM of test image with all templates and returns best match.
    """
    templates, template_paths = load_all_templates()

    test_gray = cv2.cvtColor(test_bgr, cv2.COLOR_BGR2GRAY)

    best_score = -1
    best_template = None
    best_path = None

    for tpl_bgr, tpl_path in zip(templates, template_paths):
        resized_test = cv2.resize(test_gray, (tpl_bgr.shape[1], tpl_bgr.shape[0]))
        tpl_gray = cv2.cvtColor(tpl_bgr, cv2.COLOR_BGR2GRAY)

        score, _ = ssim(tpl_gray, resized_test, full=True)

        if score > best_score:
            best_score = score
            best_template = tpl_bgr
            best_path = tpl_path

    return best_template, best_path


# ---------- Defect Detection Pipeline ----------
def detect_defects(model, classes, template_bgr, test_bgr):

    # Resize test to match template
    test_bgr = cv2.resize(test_bgr, (template_bgr.shape[1], template_bgr.shape[0]))

    # Convert to grayscale
    gray_tpl = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)
    gray_test = cv2.cvtColor(test_bgr, cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(gray_tpl, gray_test)

    # Otsu threshold
    _, th = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphology
    kernel = np.ones((5, 5), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)
    th = cv2.dilate(th, kernel, iterations=1)

    # Find contours
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
        x1, y1, x2, y2 = d["x1"], d["y1"], d["x2"], d["y2"]
        label, conf = d["label"], d["conf"]

        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(out, f"{label} {conf:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return out


# ---------- MASTER FUNCTION ----------
def infer_pcb_from_array(test_bgr):
    model, classes = load_model()

    if test_bgr is None:
        raise ValueError("Invalid PCB array!")

    # 1. Auto-select template
    best_template, best_path = find_best_template(test_bgr)

    # 2. Detect defects
    detections, mask = detect_defects(model, classes, best_template, test_bgr)

    # 3. Annotate
    annotated = annotate_image(test_bgr, detections)

    return annotated, detections, mask, best_path
