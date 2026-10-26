from pathlib import Path

import joblib
import numpy as np
import streamlit as st
from PIL import Image
from skimage.feature import hog


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "hog_ocsvm_calibrated"
    / "model.joblib"
)


st.set_page_config(
    page_title="Bottle-Mouth Anomaly Inspection",
    page_icon="🔍",
    layout="wide",
)


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


def extract_feature(image, image_size, hog_parameters):
    processed_image = image.convert("L")
    processed_image = processed_image.resize(
        image_size,
        Image.Resampling.BILINEAR,
    )

    image_array = np.asarray(processed_image)

    feature = hog(
        image_array,
        **hog_parameters,
        feature_vector=True,
    )

    return feature.reshape(1, -1)


def predict(image, model_artifact):
    pipeline = model_artifact["pipeline"]
    threshold = float(model_artifact["threshold"])
    image_size = model_artifact["image_size"]
    hog_parameters = model_artifact["hog_parameters"]

    feature = extract_feature(
        image,
        image_size,
        hog_parameters,
    )

    anomaly_score = float(
        -pipeline.decision_function(feature)[0]
    )

    predicted_label = (
        "Anomaly"
        if anomaly_score > threshold
        else "Good"
    )

    return anomaly_score, threshold, predicted_label


st.title("Bottle-Mouth Visual Anomaly Inspection")
st.caption("固定俯视视角瓶口区域异常检测原型")

st.write(
    "Upload a compatible top-view image of the bottle "
    "opening to classify it as "
    "**Good** or **Anomaly**."
)
st.write(
    "上传一张与数据集视角相近的瓶口区域图片，系统将判断该图片是"
    "**正常**还是**异常**。"
)

with st.sidebar:
    st.header("Model information")
    st.write("**Feature:** HOG")
    st.write("**Detector:** One-Class SVM")
    st.write("**Input size:** 128 × 128")
    st.write("**Training:** Normal images only")
    st.write("**Threshold:** Normal-only calibration")

    st.divider()

    st.info(
        "Validated scope: image-level inspection of the "
        "bottle opening and surrounding rim from a fixed "
        "top view. This is not full-bottle inspection and "
        "does not locate defect pixels.\n\n"
        "适用范围：固定俯视视角下的瓶口及周围边缘图片级检测；"
        "不代表整瓶质检能力，也不定位缺陷像素。"
    )

try:
    model_artifact = load_model()
except Exception as error:
    st.error(f"Unable to load the model: {error}")
    st.stop()

uploaded_file = st.file_uploader(
    "Upload a top-view bottle-opening image / 上传俯视瓶口图片",
    type=["png", "jpg", "jpeg"],
)

if uploaded_file is None:
    st.info(
        "Please upload an image to begin inspection. "
        "请上传图片开始检测。"
    )
    st.stop()

try:
    uploaded_image = Image.open(uploaded_file)

    anomaly_score, threshold, predicted_label = predict(
        uploaded_image,
        model_artifact,
    )

except Exception as error:
    st.error(f"Prediction failed: {error}")
    st.stop()

image_column, result_column = st.columns([1.2, 1])

with image_column:
    st.subheader("Uploaded image / 上传图片")
    st.image(
        uploaded_image,
        width="stretch",
    )

with result_column:
    st.subheader("Inspection result / 检测结果")

    if predicted_label == "Good":
        st.success("GOOD — Normal view / 当前瓶口视图正常")
    else:
        st.error("ANOMALY — Possible defect / 可能存在缺陷")

    metric_column_1, metric_column_2 = st.columns(2)

    with metric_column_1:
        st.metric(
            "Anomaly score",
            f"{anomaly_score:.4f}",
        )

    with metric_column_2:
        st.metric(
            "Decision threshold",
            f"{threshold:.4f}",
        )

    if anomaly_score > threshold:
        st.write(
            "The anomaly score is above the calibrated "
            "threshold, so the image is classified as "
            "**Anomaly**."
        )
    else:
        st.write(
            "The anomaly score is not above the calibrated "
            "threshold, so the image is classified as "
            "**Good**."
        )

st.divider()

st.caption(
    "HOG + calibrated One-Class SVM | "
    "Image-level prediction only"
)
