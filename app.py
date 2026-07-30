import io
import csv
from datetime import datetime

import streamlit as st
from PIL import Image
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Skin Acne Detector", page_icon="🔬", layout="wide")

MODEL_PATH = "skinyolo11.pt"  # keep this file in the same folder as app.py when deploying

# Rough severity weighting — inflamed/deep lesions count for more than surface ones.
# This is a simple heuristic for a quick visual summary, NOT a clinical diagnosis.
SEVERITY_WEIGHTS = {
    "Cyst": 4,
    "Pustules": 3,
    "Papules": 2,
    "Blackheads": 1,
    "Whiteheads": 1,
}


# ---------------------------------------------------------------------------
# Backend helpers
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model(model_path: str):
    return YOLO(model_path)


def run_inference(model, image: Image.Image, conf: float, iou: float, imgsz: int):
    results = model.predict(image, conf=conf, iou=iou, imgsz=imgsz, verbose=False)
    return results[0]


def summarize_detections(result):
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return {}, []

    names = result.names
    class_counts, rows = {}, []
    for cls_id, conf, xyxy in zip(boxes.cls.tolist(), boxes.conf.tolist(), boxes.xyxy.tolist()):
        cls_name = names[int(cls_id)]
        class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
        rows.append({"class": cls_name, "confidence": round(conf, 3), "box": [round(v, 1) for v in xyxy]})
    return class_counts, rows


def crop_detections(image: Image.Image, rows, max_crops: int = 12):
    """Returns a list of (label, cropped PIL image) for the highest-confidence detections."""
    sorted_rows = sorted(rows, key=lambda r: r["confidence"], reverse=True)[:max_crops]
    crops = []
    for row in sorted_rows:
        x1, y1, x2, y2 = row["box"]
        pad = 6
        w, h = image.size
        box = (max(0, x1 - pad), max(0, y1 - pad), min(w, x2 + pad), min(h, y2 + pad))
        crops.append((f"{row['class']} ({row['confidence']:.2f})", image.crop(box)))
    return crops


def compute_severity(class_counts: dict):
    score = sum(SEVERITY_WEIGHTS.get(cls, 1) * count for cls, count in class_counts.items())
    if score == 0:
        return 0, "Clear", "🟢"
    elif score <= 8:
        return score, "Mild", "🟡"
    elif score <= 20:
        return score, "Moderate", "🟠"
    else:
        return score, "Severe", "🔴"


def image_to_png_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def rows_to_csv_bytes(rows) -> bytes:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=["class", "confidence", "box"])
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return buf.getvalue().encode("utf-8")


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []  # list of dicts: timestamp, thumb, total, class_counts, severity

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
st.title("🔬 Skin Acne Detector")
st.caption("Detects blackheads, whiteheads, papules, pustules, and cysts from a photo.")

try:
    model = load_model(MODEL_PATH)
except Exception as e:
    st.error(
        f"Couldn't load the model from **{MODEL_PATH}**. "
        f"Make sure `{MODEL_PATH}` is in the same folder as `app.py`.\n\nError detail: {e}"
    )
    st.stop()

with st.sidebar:
    st.header("⚙️ Settings")
    conf_threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.25, 0.05)
    iou_threshold = st.slider("IoU threshold (NMS)", 0.0, 1.0, 0.45, 0.05)
    img_size = st.selectbox("Inference image size", [640, 768, 1024], index=2)
    show_labels = st.checkbox("Show class labels on boxes", value=True)
    show_conf = st.checkbox("Show confidence on boxes", value=True)
    st.divider()
    st.caption(f"📊 {len(st.session_state.history)} scan(s) this session")
    if st.session_state.history and st.button("🗑️ Clear history"):
        st.session_state.history = []
        st.rerun()

tab_single, tab_batch, tab_history = st.tabs(["📷 Single Image", "📁 Batch Analysis", "🕒 History"])

# ---------------------------------------------------------------------------
# TAB 1 — single image, with upload OR live camera
# ---------------------------------------------------------------------------
with tab_single:
    source = st.radio("Image source", ["Upload a photo", "Use camera"], horizontal=True)

    image = None
    if source == "Upload a photo":
        uploaded_file = st.file_uploader("Upload a skin image", type=["jpg", "jpeg", "png"], key="single_upload")
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
    else:
        camera_file = st.camera_input("Take a photo")
        if camera_file is not None:
            image = Image.open(camera_file).convert("RGB")

    if image is not None:
        col_left, col_right = st.columns(2)
        with col_left:
            st.image(image, caption="Input image", use_container_width=True)

        predict_clicked = st.button("🔍 Predict", type="primary")

        if predict_clicked:
            with st.spinner("Running detection..."):
                result = run_inference(model, image, conf_threshold, iou_threshold, img_size)

            annotated_bgr = result.plot(labels=show_labels, conf=show_conf)
            annotated_rgb = annotated_bgr[:, :, ::-1]
            annotated_image = Image.fromarray(annotated_rgb)

            with col_right:
                st.image(annotated_rgb, caption="Detections", use_container_width=True)

            class_counts, rows = summarize_detections(result)
            total = len(rows)
            score, severity_label, severity_emoji = compute_severity(class_counts)

            st.subheader(f"Results: {total} lesion(s) detected")

            m1, m2 = st.columns(2)
            m1.metric("Total lesions", total)
            m2.metric(
                "Severity (heuristic)",
                f"{severity_emoji} {severity_label}",
                help="A simple weighted count, not a medical diagnosis.",
            )

            if total > 0:
                st.write("**Breakdown by type:**")
                cols = st.columns(len(class_counts))
                for col, (cls_name, count) in zip(cols, class_counts.items()):
                    col.metric(cls_name, count)

                st.write("**Closest-up lesions (highest confidence):**")
                crops = crop_detections(image, rows)
                crop_cols = st.columns(4)
                for i, (label, crop_img) in enumerate(crops):
                    with crop_cols[i % 4]:
                        st.image(crop_img, caption=label, use_container_width=True)

                with st.expander("Full detection details"):
                    for i, row in enumerate(rows, start=1):
                        st.write(f"**{i}. {row['class']}** — confidence: {row['confidence']}, box: {row['box']}")

                dl1, dl2 = st.columns(2)
                dl1.download_button(
                    "⬇️ Download annotated image",
                    data=image_to_png_bytes(annotated_image),
                    file_name="annotated_result.png",
                    mime="image/png",
                )
                dl2.download_button(
                    "⬇️ Download detections (CSV)",
                    data=rows_to_csv_bytes(rows),
                    file_name="detections.csv",
                    mime="text/csv",
                )
            else:
                st.info("No lesions detected. Try lowering the confidence threshold in the sidebar.")

            # log to session history
            thumb = image.copy()
            thumb.thumbnail((120, 120))
            st.session_state.history.insert(
                0,
                {
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "thumb": thumb,
                    "total": total,
                    "class_counts": class_counts,
                    "severity": f"{severity_emoji} {severity_label}",
                },
            )
    else:
        st.info("Upload an image or take a photo to get started.")

# ---------------------------------------------------------------------------
# TAB 2 — batch analysis across multiple images
# ---------------------------------------------------------------------------
with tab_batch:
    st.write("Upload several images at once to compare lesion counts across them.")
    batch_files = st.file_uploader(
        "Upload multiple images", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key="batch_upload"
    )

    if batch_files:
        run_batch = st.button("🔍 Run batch prediction", type="primary")
        if run_batch:
            summary_rows = []
            all_class_names = set()
            progress = st.progress(0.0, text="Starting...")

            for idx, f in enumerate(batch_files):
                progress.progress(idx / len(batch_files), text=f"Processing {f.name}...")
                img = Image.open(f).convert("RGB")
                result = run_inference(model, img, conf_threshold, iou_threshold, img_size)
                class_counts, rows = summarize_detections(result)
                all_class_names.update(class_counts.keys())
                summary_rows.append({"filename": f.name, "total": len(rows), **class_counts})

            progress.progress(1.0, text="Done!")
            st.success(f"Processed {len(batch_files)} image(s).")

            st.dataframe(summary_rows, use_container_width=True)

            csv_buf = io.StringIO()
            fieldnames = ["filename", "total"] + sorted(all_class_names)
            writer = csv.DictWriter(csv_buf, fieldnames=fieldnames)
            writer.writeheader()
            for row in summary_rows:
                writer.writerow({k: row.get(k, 0) for k in fieldnames})
            st.download_button(
                "⬇️ Download batch summary (CSV)",
                data=csv_buf.getvalue().encode("utf-8"),
                file_name="batch_summary.csv",
                mime="text/csv",
            )
    else:
        st.info("Upload 2 or more images to run a batch comparison.")

# ---------------------------------------------------------------------------
# TAB 3 — session history
# ---------------------------------------------------------------------------
with tab_history:
    if not st.session_state.history:
        st.info("No scans yet this session — run a prediction in the Single Image tab.")
    else:
        for entry in st.session_state.history:
            c1, c2, c3 = st.columns([1, 2, 2])
            with c1:
                st.image(entry["thumb"], use_container_width=True)
            with c2:
                st.write(f"**{entry['timestamp']}**")
                st.write(f"Total lesions: {entry['total']}")
                st.write(f"Severity: {entry['severity']}")
            with c3:
                if entry["class_counts"]:
                    st.write(", ".join(f"{k}: {v}" for k, v in entry["class_counts"].items()))
                else:
                    st.write("No detections")
            st.divider()
