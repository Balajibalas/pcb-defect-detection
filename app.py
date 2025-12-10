import streamlit as st
import numpy as np
import cv2
from inference_backend import infer_pcb_from_array
from PIL import Image

st.set_page_config(page_title="PCB Defect Detection", layout="wide")

st.title("🔍 PCB Defect Detection (Auto Template Selection)")

uploaded = st.file_uploader("Upload Full PCB Image", type=["jpg", "jpeg", "png"])

if uploaded:
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    test_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    st.info("Selecting the best template... Running defect detection...")

    with st.spinner("Processing the PCB..."):
        annotated, detections, mask, template_used = infer_pcb_from_array(test_bgr)
        #annotated, detections, mask, template_used = infer_pcb(uploaded.name)

    st.success(f"Template Used: {template_used}")

    st.subheader("Annotated Output")
    st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))

    st.subheader("Difference Mask")
    st.image(mask, clamp=True)

    st.subheader("Detected Defects")
    if len(detections) == 0:
        st.success("No defects detected.")
    else:
        for d in detections:
            st.write(
                f"- **{d['label']}** (Conf: {d['conf']:.2f}) "
                f"Location: ({d['x1']}, {d['y1']}) → ({d['x2']}, {d['y2']})"
            )
