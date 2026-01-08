import streamlit as st
import numpy as np
import cv2
from inference_backend import (
    infer_pcb_from_array,
    infer_with_uploaded_template
)

# ================== PAGE CONFIG ==================
st.set_page_config(
    page_title="PCB Defect Detection",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ================== CUSTOM CSS ==================
st.markdown("""
<style>
    .block-container {
        padding-top: 3rem;
        padding-left: 4rem;
        padding-right: 4rem;
        max-width: 100%;
    }

    h1 {
        font-size: 3rem !important;
        text-align: center;
    }

    h2 {
        font-size: 2rem !important;
    }

    h3 {
        font-size: 1.5rem !important;
    }

    label, div, span {
        font-size: 1.05rem !important;
    }

    .stFileUploader {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# ================== TITLE ==================
st.markdown("<h1>🔍 PCB Defect Detection System</h1>", unsafe_allow_html=True)
st.markdown("---")

# ================== CONTROLS SECTION ==================
control_col, upload_col = st.columns([1, 2])

with control_col:
    st.subheader("Template Mode")
    mode = st.radio(
        "Select Template Mode",
        ["Auto Template Selection", "Manual Template Upload"]
    )

with upload_col:
    st.subheader("Upload PCB Image")
    uploaded_test = st.file_uploader(
        "Upload FULL PCB Image",
        type=["jpg", "jpeg", "png"]
    )

# ================== MANUAL TEMPLATE UPLOAD ==================
uploaded_template = None
if mode == "Manual Template Upload":
    st.subheader("Upload Template Image")
    uploaded_template = st.file_uploader(
        "Upload TEMPLATE PCB Image",
        type=["jpg", "jpeg", "png"],
        key="template_upload"
    )

st.markdown("---")

# ================== PROCESSING ==================
if uploaded_test:

    file_bytes = np.asarray(bytearray(uploaded_test.read()), dtype=np.uint8)
    test_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if test_bgr is None:
        st.error("❌ Could not read PCB image.")
        st.stop()

    # ---------- AUTO MODE ----------
    if mode == "Auto Template Selection":
        with st.spinner("🔍 Auto-detecting template and processing PCB..."):
            annotated, detections, mask, template_used = infer_pcb_from_array(test_bgr)

        st.success(f"✅ Auto-selected template: `{template_used}`")

    # ---------- MANUAL MODE ----------
    else:
        if uploaded_template is None:
            st.warning("⚠ Please upload a template image.")
            st.stop()

        temp_bytes = np.asarray(bytearray(uploaded_template.read()), dtype=np.uint8)
        template_bgr = cv2.imdecode(temp_bytes, cv2.IMREAD_COLOR)

        if template_bgr is None:
            st.error("❌ Could not read template image.")
            st.stop()

        with st.spinner("🧠 Comparing uploaded template with PCB..."):
            annotated, detections, mask = infer_with_uploaded_template(
                test_bgr, template_bgr
            )

        st.success("✅ Manual template applied successfully.")

    # ================== RESULTS ==================
    st.markdown("---")
    st.subheader("🖼 Annotated PCB Output")
    st.image(
        cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
        use_container_width=True
    )

    st.subheader("🧪 Difference Mask")
    st.image(mask, clamp=True, use_container_width=True)

    st.subheader("📌 Detected Defects")

    if len(detections) == 0:
        st.success("🎉 No visible defects detected.")
    else:
        for d in detections:
            st.markdown(
                f"- **{d['label']}** — Confidence: `{d['conf']:.2f}`"
            )

else:
    st.info("⬆ Upload a FULL PCB image to start detection.")
