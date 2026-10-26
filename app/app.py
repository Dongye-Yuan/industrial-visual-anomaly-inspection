from pathlib import Path
import sys

import joblib
from matplotlib import colormaps
import numpy as np
import streamlit as st
from PIL import Image
from skimage.feature import hog


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.patchcore import (
    choose_device,
    extract_patch_embeddings,
    image_to_tensor,
    load_artifact,
    score_embeddings,
    upsample_anomaly_maps,
)


HOG_MODEL_PATH = (
    PROJECT_ROOT / "outputs" / "hog_ocsvm_calibrated" / "model.joblib"
)
PATCH_MODEL_PATH = PROJECT_ROOT / "outputs" / "patchcore" / "model.pt"

PATCH_MODEL_NAME = "Deep patch memory + heatmap"
HOG_MODEL_NAME = "HOG + calibrated One-Class SVM"


st.set_page_config(
    page_title="Bottle-Mouth Anomaly Inspection",
    page_icon="🔍",
    layout="wide",
)


@st.cache_resource
def load_hog_model():
    if not HOG_MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {HOG_MODEL_PATH}")
    return joblib.load(HOG_MODEL_PATH)


@st.cache_resource
def load_patch_model():
    if not PATCH_MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {PATCH_MODEL_PATH}")
    device = choose_device()
    artifact, encoder = load_artifact(PATCH_MODEL_PATH, device)
    return artifact, encoder, device


def extract_hog_feature(image, image_size, hog_parameters):
    processed_image = image.convert("L").resize(
        image_size,
        Image.Resampling.BILINEAR,
    )
    feature = hog(
        np.asarray(processed_image),
        **hog_parameters,
        feature_vector=True,
    )
    return feature.reshape(1, -1)


def predict_hog(image, model_artifact):
    feature = extract_hog_feature(
        image,
        model_artifact["image_size"],
        model_artifact["hog_parameters"],
    )
    anomaly_score = float(
        -model_artifact["pipeline"].decision_function(feature)[0]
    )
    threshold = float(model_artifact["threshold"])
    label = "Anomaly" if anomaly_score > threshold else "Good"
    return anomaly_score, threshold, label, None, None


def predict_patch(image, artifact, encoder, device):
    image_size = tuple(artifact["image_size"])
    patch_grid = tuple(artifact["patch_grid"])
    tensor = image_to_tensor(image, image_size).unsqueeze(0).to(device)
    embeddings = extract_patch_embeddings(
        encoder,
        tensor,
        patch_grid,
    ).cpu().numpy()
    patch_scores = score_embeddings(
        embeddings,
        artifact["memory_bank"].cpu().numpy(),
    )
    anomaly_score = float(patch_scores.max())
    threshold = float(artifact["threshold"])
    label = "Anomaly" if anomaly_score > threshold else "Good"
    anomaly_map = upsample_anomaly_maps(
        patch_scores,
        patch_grid=patch_grid,
        image_size=image_size,
    )[0]

    lower, upper = np.percentile(anomaly_map, [5, 99])
    normalised = np.clip(
        (anomaly_map - lower) / max(upper - lower, 1e-8),
        0,
        1,
    )
    heatmap = Image.fromarray(
        (colormaps["turbo"](normalised)[..., :3] * 255).astype(np.uint8)
    )
    resized = image.convert("RGB").resize(image_size, Image.Resampling.BILINEAR)
    overlay = Image.blend(resized, heatmap, 0.45)
    return anomaly_score, threshold, label, heatmap, overlay


st.title("Bottle-Mouth Visual Anomaly Inspection")
st.caption("固定俯视视角瓶口区域异常检测与缺陷定位原型")

st.write(
    "Upload a compatible top-view image of the bottle opening to classify "
    "it as **Good** or **Anomaly**. The deep model also highlights suspicious "
    "regions."
)
st.write(
    "上传一张与数据集视角相近的瓶口区域图片，系统将判断其为**正常**或"
    "**异常**；深度模型还会用热力图标出可疑区域。"
)

with st.sidebar:
    st.header("Model / 模型")
    selected_model = st.selectbox(
        "Choose an inspection model / 选择检测模型",
        [PATCH_MODEL_NAME, HOG_MODEL_NAME],
    )

    if selected_model == PATCH_MODEL_NAME:
        st.write("**Feature:** pretrained ResNet-18 patches")
        st.write("**Detector:** nearest normal-patch memory")
        st.write("**Input size:** 224 × 224")
        st.write("**Output:** decision + localisation heatmap")
        st.caption(
            "Compact educational implementation inspired by PatchCore; "
            "it is not an exact reproduction of the paper."
        )
    else:
        st.write("**Feature:** HOG")
        st.write("**Detector:** One-Class SVM")
        st.write("**Input size:** 128 × 128")
        st.write("**Output:** image-level decision")

    st.write("**Training:** normal images only")
    st.write("**Threshold:** held-out normal calibration")
    st.divider()
    st.info(
        "Validated scope: the bottle opening and surrounding rim from a "
        "fixed top view. This is not full-bottle inspection.\n\n"
        "适用范围：固定俯视视角下的瓶口及周围边缘，不代表整瓶质检能力。"
    )

try:
    if selected_model == PATCH_MODEL_NAME:
        model_artifact, encoder, device = load_patch_model()
    else:
        model_artifact = load_hog_model()
except Exception as error:
    st.error(f"Unable to load the model: {error}")
    st.stop()

uploaded_file = st.file_uploader(
    "Upload a top-view bottle-opening image / 上传俯视瓶口图片",
    type=["png", "jpg", "jpeg"],
)

if uploaded_file is None:
    st.info("Please upload an image to begin inspection. 请上传图片开始检测。")
    st.stop()

try:
    uploaded_image = Image.open(uploaded_file).convert("RGB")
    if selected_model == PATCH_MODEL_NAME:
        result = predict_patch(uploaded_image, model_artifact, encoder, device)
    else:
        result = predict_hog(uploaded_image, model_artifact)
    anomaly_score, threshold, predicted_label, heatmap, overlay = result
except Exception as error:
    st.error(f"Prediction failed: {error}")
    st.stop()

image_column, result_column = st.columns([1.25, 1])

with image_column:
    st.subheader("Uploaded image / 上传图片")
    st.image(uploaded_image, width="stretch")

with result_column:
    st.subheader("Inspection result / 检测结果")
    if predicted_label == "Good":
        st.success("GOOD — Normal view / 当前瓶口视图正常")
    else:
        st.error("ANOMALY — Possible defect / 可能存在缺陷")

    metric_column_1, metric_column_2 = st.columns(2)
    metric_column_1.metric("Anomaly score", f"{anomaly_score:.4f}")
    metric_column_2.metric("Decision threshold", f"{threshold:.4f}")
    relation = "above" if anomaly_score > threshold else "not above"
    chinese_relation = "高于" if anomaly_score > threshold else "未高于"
    st.write(
        f"The score is {relation} the calibrated threshold. "
        f"异常分数{chinese_relation}校准阈值。"
    )

if overlay is not None:
    st.subheader("Defect localisation / 缺陷定位")
    heatmap_column, overlay_column = st.columns(2)
    with heatmap_column:
        st.image(heatmap, caption="Anomaly heatmap / 异常热力图", width="stretch")
    with overlay_column:
        st.image(overlay, caption="Heatmap overlay / 热力图叠加", width="stretch")
    st.caption(
        "Warm colours indicate regions least similar to normal training patches; "
        "the heatmap is model evidence, not a guaranteed defect boundary. / "
        "暖色区域与正常训练补丁最不相似；热力图是模型依据，并非精确缺陷边界。"
    )

st.divider()
st.caption(f"Active model / 当前模型: {selected_model}")
