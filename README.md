# Industrial Visual Anomaly Inspection｜工业视觉异常检测

[![Live Demo](https://img.shields.io/badge/Live_Demo-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://bottle-anomaly-inspection.streamlit.app/)
[![CI](https://github.com/Dongye-Yuan/industrial-visual-anomaly-inspection/actions/workflows/ci.yml/badge.svg)](https://github.com/Dongye-Yuan/industrial-visual-anomaly-inspection/actions/workflows/ci.yml)
[![Python 3.14](https://img.shields.io/badge/Python-3.14-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)

**Live demo / 在线演示:** [bottle-anomaly-inspection.streamlit.app](https://bottle-anomaly-inspection.streamlit.app/)

**English** | [中文](#中文)

## English

An end-to-end, normal-only anomaly detection and localisation prototype for fixed top-view inspection of the bottle-opening region in the MVTec AD bottle category. The repository covers dataset exploration, handcrafted and pretrained deep features, one-class model training, threshold calibration, image- and pixel-level evaluation, model comparison, artifact persistence, and single-image inference.

The deployed app classifies a compatible image as **Good** or **Anomaly** and can highlight suspicious pixels with a deep-feature heatmap. Its validated scope is the bottle opening and surrounding rim under a camera view similar to the dataset; it is not a general full-bottle, bottle-side, or bottle-bottom inspection system. Heatmaps are model evidence rather than guaranteed defect boundaries.

### Engineering ownership

This repository presents my independent end-to-end implementation. I designed and built:

- dataset validation and exploratory analysis;
- the HOG feature extraction pipeline;
- One-Class SVM and Isolation Forest training workflows;
- a compact PatchCore-style ResNet-18 patch-memory workflow;
- normal-only threshold calibration and model evaluation;
- model persistence, single-image inference, and heatmap localisation;
- the bilingual Streamlit interface; and
- reproducibility documentation and presentation-ready visualisations.

![Representative bottle samples](outputs/figures/sample_overview.png)

### Why this project matters

Industrial defects are rare and evolve over time, while normal products are comparatively easy to collect. This project therefore learns normal bottle appearance without requiring defective training samples. A calibrated decision threshold then balances defect detection against false rejection of acceptable products.

### Solution at a glance

```text
Bottle image
    ├─ HOG branch → grayscale 128 × 128 → HOG → one-class detector
    └─ Deep branch → RGB 224 × 224 → frozen ResNet-18 patches → normal memory bank
                         ↓
               anomaly score + calibrated threshold
                         ↓
             Good / Anomaly + localisation heatmap
```

Four configurations were evaluated:

- HOG + One-Class SVM baseline
- HOG + calibrated One-Class SVM
- HOG + calibrated Isolation Forest
- compact PatchCore-style deep patch memory

The deep patch-memory model is the strongest evaluated model. It is a compact educational implementation inspired by PatchCore, not an exact reproduction of the paper.

### Key results

| Model | Accuracy | Recall | Specificity | F1 score | Latency |
|---|---:|---:|---:|---:|---:|
| OCSVM baseline | 0.8434 | 0.9841 | 0.4000 | 0.9051 | 12.11 ms/image |
| OCSVM calibrated | 0.9639 | 0.9524 | **1.0000** | 0.9756 | **11.94 ms/image** |
| Isolation Forest | 0.8795 | 0.8730 | 0.9000 | 0.9167 | 11.99 ms/image |
| **Deep patch memory** | **0.9880** | **1.0000** | 0.9500 | **0.9921** | 23.47 ms/image |

![Model comparison](outputs/model_comparison/model_comparison_bar_chart_ppt.png)

On the available test set, deep patch memory detected all 63 anomalous images and accepted 19 of 20 good images. It achieved image AUROC 1.0000, pixel AUROC 0.9861, pixel average precision 0.7747, and AU-PRO@0.30 0.9516. The stronger localisation comes with roughly twice the CPU inference latency of the HOG models.

![Deep patch localisation examples](outputs/patchcore/heatmap_examples.png)

### Dataset

The project uses the **MVTec AD bottle** category:

In this dataset, the visible inspection region is the bottle opening and surrounding rim captured from a consistent top view. Results should therefore be interpreted within that imaging setup.

| Split and category | Images |
|---|---:|
| Train / good | 209 |
| Test / good | 20 |
| Test / broken large | 20 |
| Test / broken small | 22 |
| Test / contamination | 21 |

The dataset is intentionally excluded from Git. Place it at:

```text
data/mvtec_ad/bottle/
├── train/good/
├── test/good/
├── test/broken_large/
├── test/broken_small/
├── test/contamination/
└── ground_truth/
```

### Quick start

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Windows PowerShell activation:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Run the full workflow:

```bash
python src/explore_dataset.py
python src/train_hog_ocsvm.py
python src/train_hog_ocsvm_calibrated.py
python src/train_hog_isolation_forest.py
python src/train_patchcore.py
python src/compare_models.py
```

Inspect one image with the saved final model:

```bash
python src/predict_image.py data/mvtec_ad/bottle/test/good/000.png
python src/predict_image.py data/mvtec_ad/bottle/test/broken_large/000.png
python src/predict_patchcore.py data/mvtec_ad/bottle/test/broken_large/000.png
```

Example output:

```text
Prediction result
  Anomaly score: 0.7405
  Threshold: 0.5604
  Predicted label: Anomaly
```

### Automated tests

The repository includes data-independent unit tests for the HOG feature contract, calibrated-threshold decisions, deep patch scoring and coreset selection, missing-input validation, and the versioned model-artifact schema:

```bash
python -m unittest discover -s tests -v
```

GitHub Actions runs the tests and Python syntax checks automatically on every push to `main` and every pull request. The workflow does not download the MVTec AD dataset.

### Web demo

**Public deployment:** [Open the live bilingual demo](https://bottle-anomaly-inspection.streamlit.app/)

Launch the bilingual Streamlit interface locally:

```bash
streamlit run app/app.py
```

Upload a compatible top-view PNG or JPEG image of the bottle-opening region. The model selector provides both the lightweight HOG baseline and the deep patch-memory model; the latter also displays a localisation heatmap and overlay.

![Streamlit anomaly inspection demo](docs/images/streamlit_demo.png)

### Repository structure

```text
industrial-visual-anomaly-inspection/
├── app/
│   └── app.py                    # bilingual Streamlit demo
├── data/                         # local dataset, ignored by Git
├── docs/                         # developer guide and demo images
├── outputs/                      # metrics, plots, predictions, model artifacts
├── src/
│   ├── explore_dataset.py
│   ├── train_hog_ocsvm.py
│   ├── train_hog_ocsvm_calibrated.py
│   ├── train_hog_isolation_forest.py
│   ├── patchcore.py
│   ├── train_patchcore.py
│   ├── predict_patchcore.py
│   ├── compare_models.py
│   └── predict_image.py
├── requirements.txt
└── README.md
```

### Documentation

- [Developer Guide｜开发者文档](docs/DEVELOPER_GUIDE.md)

---

## 中文

这是一个面向固定俯视视角瓶口区域的端到端、仅正常样本训练、异常检测与定位原型。项目使用 MVTec AD bottle 类别完成数据探索、HOG 与预训练深度特征提取、单类模型训练、阈值校准、图片级与像素级评估、模型对比、模型保存以及单张图片推理。

部署应用会把符合输入条件的图片判断为 **Good** 或 **Anomaly**，深度模型还能用热力图标出可疑区域。目前验证范围是与数据集拍摄条件相近的瓶口及周围边缘区域，不代表整瓶、瓶身侧面或瓶底质检能力。热力图是模型判断依据，并不等同于保证准确的缺陷边界。

### 个人实现

本仓库展示了我独立完成的端到端实现，包括：

- 数据集验证与探索性分析；
- HOG 特征提取流水线；
- One-Class SVM 与 Isolation Forest 训练流程；
- 基于 ResNet-18 的紧凑型 PatchCore 风格补丁记忆流程；
- 仅使用正常样本的阈值校准与模型评估；
- 模型保存、单图推理与热力图定位；
- 中英双语 Streamlit 界面；
- 可复现技术文档与演示级可视化。

### 项目价值

工业缺陷样本通常数量少、类型变化快，而正常产品数据相对容易获取。因此，本项目只使用正常瓶子图片训练模型，不依赖缺陷训练样本。随后利用独立的正常校准集确定异常阈值，在缺陷检出率与正常产品误报率之间取得平衡。

### 技术流程

```text
瓶口图片
    ├─ HOG 分支 → 灰度 128 × 128 → HOG → 单类检测器
    └─ 深度分支 → RGB 224 × 224 → 冻结的 ResNet-18 补丁 → 正常记忆库
                         ↓
                   异常分数与校准阈值
                         ↓
              Good / Anomaly + 定位热力图
```

项目比较了四种配置：

- HOG + One-Class SVM 基线模型
- HOG + 校准阈值的 One-Class SVM
- HOG + 校准阈值的 Isolation Forest
- 紧凑型 PatchCore 风格深度补丁记忆模型

深度补丁记忆模型是当前表现最强的模型。它是受 PatchCore 启发的教学型紧凑实现，并非论文的完全复现。

### 核心结果

| 模型 | 准确率 | 召回率 | 特异度 | F1 分数 | 推理延迟 |
|---|---:|---:|---:|---:|---:|
| OCSVM 基线 | 0.8434 | 0.9841 | 0.4000 | 0.9051 | 12.11 ms/张 |
| 校准 OCSVM | 0.9639 | 0.9524 | **1.0000** | 0.9756 | **11.94 ms/张** |
| Isolation Forest | 0.8795 | 0.8730 | 0.9000 | 0.9167 | 11.99 ms/张 |
| **深度补丁记忆** | **0.9880** | **1.0000** | 0.9500 | **0.9921** | 23.47 ms/张 |

在当前测试集上，深度补丁记忆模型检出了全部 63 张异常图片，并正确接受了 20 张正常图片中的 19 张。其图片 AUROC 为 1.0000、像素 AUROC 为 0.9861、像素平均精确率为 0.7747、AU-PRO@0.30 为 0.9516；代价是 CPU 推理延迟约为 HOG 模型的两倍。

### 快速运行

安装依赖并运行完整流程：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

python src/explore_dataset.py
python src/train_hog_ocsvm.py
python src/train_hog_ocsvm_calibrated.py
python src/train_hog_isolation_forest.py
python src/train_patchcore.py
python src/compare_models.py
```

使用最终模型检测单张图片：

```bash
python src/predict_image.py path/to/image.png
python src/predict_patchcore.py path/to/image.png
```

程序会输出异常分数、校准阈值以及最终判断结果。分数高于阈值时输出 `Anomaly`，否则输出 `Good`。

### 自动测试

仓库提供不依赖原始数据集的单元测试，覆盖 HOG 特征维度、校准阈值两侧的分类结果、输入文件校验以及已保存模型的字段约定：

```bash
python -m unittest discover -s tests -v
```

GitHub Actions 会在每次向 `main` 推送代码以及创建 Pull Request 时自动执行测试和 Python 语法检查，测试过程不下载 MVTec AD 数据集。

### 可视化网页演示

**公开在线演示：**[点击打开中英双语 Streamlit 应用](https://bottle-anomaly-inspection.streamlit.app/)

在本地启动中英双语 Streamlit 界面：

```bash
streamlit run app/app.py
```

上传与数据集视角相近的瓶口区域 PNG 或 JPEG 图片后，页面会显示图片级判断、异常分数和校准阈值。按 `Control + C` 可以停止本地服务。

### 设计亮点

- 只使用正常样本训练，符合真实工业缺陷数据稀缺的场景。
- 校准阈值只使用留出的正常图片，不使用测试标签。
- 保存了预处理器、模型、阈值、图片尺寸和 HOG 参数，确保推理流程一致。
- 输出指标、预测结果、混淆矩阵、ROC/PR 曲线和模型对比图。
- 提供独立的单图预测入口，能够验证训练后模型的实际可用性。

### 已知限制与改进方向

- 当前只进行图片级异常检测，尚未定位缺陷像素区域。
- 灰度 HOG 对颜色变化和低对比度污染不够敏感。
- 当前实验只覆盖固定俯视视角下的 MVTec AD bottle 瓶口区域，不能直接代表整瓶、瓶身侧面或瓶底检测能力。
- 后续可增加颜色直方图、LBP、预训练深度特征和像素级定位方法。

详细的复现、维护和扩展说明请阅读：[开发者文档](docs/DEVELOPER_GUIDE.md)。

## Project status｜项目状态

The training, evaluation, model comparison, saved-model inference, and bilingual web demo workflows are complete. The next technical priorities are stronger texture or colour representations and pixel-level defect localisation.

训练、评估、模型对比、模型保存与推理以及中英双语网页演示流程均已完成。下一阶段的技术重点是增强颜色与纹理特征，并实现像素级缺陷定位。
