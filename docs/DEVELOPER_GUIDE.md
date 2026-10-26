# Developer Guide｜开发者文档

This document explains how to reproduce, maintain, debug, and extend the industrial visual anomaly inspection project.

本文档说明如何复现、维护、调试和扩展工业视觉异常检测项目。

## 1. System scope｜系统范围

The system performs **normal-only anomaly detection and localisation** for the fixed top-view bottle-opening region in the MVTec AD bottle category. Training uses normal images only. Evaluation uses good and labelled anomalous images. Ground-truth masks are used only for pixel-level evaluation and never as model inputs. The validated scope does not include full-bottle, bottle-side, or bottle-bottom inspection.

当前系统针对 MVTec AD bottle 类别中固定俯视视角的瓶口区域进行**仅正常样本训练的异常检测与定位**。训练阶段只使用正常图片；评估阶段使用正常和带标签的异常图片。像素级掩膜只用于像素级评估，不作为模型输入。目前验证范围不包括整瓶、瓶身侧面或瓶底质检。

### Supported workflow｜已支持流程

```text
Dataset validation and EDA
    → HOG or pretrained deep-patch extraction
    → one-class detector or normal patch memory
    → normal-only threshold calibration
    → image- and pixel-level evaluation
    → result visualisation and comparison
    → model serialization
    → single-image inference
```

## 2. Environment setup｜环境配置

The project was developed in a Python virtual environment. The pinned `requirements.txt` records the versions used during development.

项目使用 Python 虚拟环境开发，`requirements.txt` 记录了开发阶段使用的依赖版本。

### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run all commands from the repository root. Relative paths in the scripts assume that the current working directory is the project root.

所有命令都应在仓库根目录执行。脚本使用相对路径，因此若在其他目录运行，可能出现找不到数据或模型的错误。

## 3. Data contract｜数据约定

The dataset must be available locally at `data/mvtec_ad/bottle/`. The entire `data/` directory is excluded by `.gitignore` and must not be committed.

数据集必须存放在 `data/mvtec_ad/bottle/`。整个 `data/` 目录已被 `.gitignore` 排除，不应提交到 GitHub。

```text
data/mvtec_ad/bottle/
├── train/
│   └── good/*.png
├── test/
│   ├── good/*.png
│   ├── broken_large/*.png
│   ├── broken_small/*.png
│   └── contamination/*.png
└── ground_truth/
    ├── broken_large/*.png
    ├── broken_small/*.png
    └── contamination/*.png
```

The scripts currently discover PNG files with `Path.glob("*.png")`. Supporting another extension requires updating `list_images()` in the training and exploration scripts.

脚本目前通过 `Path.glob("*.png")` 查找图片。若需要支持其他格式，应同步修改数据探索和训练脚本中的 `list_images()`。

## 4. Repository architecture｜仓库结构

| Path | Responsibility / 作用 |
|---|---|
| `src/explore_dataset.py` | Validate folders, count images, and create EDA figures / 验证目录、统计图片并生成探索图 |
| `src/train_hog_ocsvm.py` | Train and evaluate the uncalibrated OCSVM baseline / 训练和评估未校准的 OCSVM 基线 |
| `src/train_hog_ocsvm_calibrated.py` | Train, calibrate, evaluate, and save the selected OCSVM / 训练、校准、评估并保存最终 OCSVM |
| `src/train_hog_isolation_forest.py` | Train and evaluate the calibrated Isolation Forest / 训练并评估校准后的 Isolation Forest |
| `src/patchcore.py` | Shared ResNet-18 patch extraction, coreset, scoring, and map utilities / 深度补丁特征、核心集、评分与热力图工具 |
| `src/train_patchcore.py` | Train and evaluate compact PatchCore-style deep patch memory / 训练并评估紧凑型 PatchCore 风格补丁记忆模型 |
| `src/predict_patchcore.py` | Inspect one image and save its localisation heatmap / 检测单图并保存定位热力图 |
| `src/compare_models.py` | Read metrics and generate comparison assets / 读取指标并生成模型对比材料 |
| `src/predict_image.py` | Load the saved artifact and inspect one image / 加载已保存模型并检测单张图片 |
| `app/app.py` | Provide a bilingual Streamlit upload and prediction interface / 提供中英双语图片上传与预测界面 |
| `outputs/` | Versioned experiment results and model artifacts / 实验结果与模型文件 |
| `docs/` | Developer documentation and demo assets / 开发者文档与演示素材 |

The HOG experiments remain independent for easy review. Deep-patch training, command-line inference, and the web app share `src/patchcore.py` so preprocessing and scoring stay consistent.

目前各训练脚本相互独立，便于单独执行与审查。如果后续模型增多，可将特征提取和评估逻辑重构为共享模块。

## 5. Feature pipeline｜特征流水线

The selected pipeline uses the following deterministic preprocessing:

1. Open the input with Pillow.
2. Convert it to grayscale with `convert("L")`.
3. Resize it to `128 × 128` using bilinear interpolation.
4. Convert it to a NumPy array.
5. Extract HOG using the stored parameters.

最终流水线使用以下固定预处理步骤：

1. 使用 Pillow 打开图片。
2. 通过 `convert("L")` 转为灰度图。
3. 使用双线性插值缩放到 `128 × 128`。
4. 转换为 NumPy 数组。
5. 使用保存的参数提取 HOG 特征。

### HOG configuration｜HOG 配置

| Parameter | Value |
|---|---:|
| Orientations | 9 |
| Pixels per cell | 16 × 16 |
| Cells per block | 2 × 2 |
| Block normalisation | L2-Hys |
| Output dimension | 1,764 |

Any change to image size or HOG parameters changes the feature dimension. A model trained with one feature definition must not be used with another.

修改图片尺寸或 HOG 参数会改变特征维度。训练和推理必须使用完全相同的特征定义。

### Deep patch-memory configuration｜深度补丁记忆配置

The deep branch resizes RGB images to `224 × 224`, applies ImageNet normalisation, and extracts a `14 × 14` grid of 128-dimensional features from a frozen pretrained ResNet-18 through `layer2`. A deterministic k-centre coreset keeps 1,024 representative normal patches. Each test-patch score is its nearest-memory Euclidean distance; the image score is the maximum patch score. Bilinear upsampling converts the patch grid into a localisation heatmap.

深度分支将 RGB 图片缩放至 `224 × 224`，使用 ImageNet 归一化，并从冻结的预训练 ResNet-18 `layer2` 提取 `14 × 14` 个 128 维补丁特征。确定性的 k-centre 核心集保留 1,024 个代表性正常补丁。每个测试补丁的异常分数是其到正常记忆库的最近欧氏距离；图片分数取所有补丁的最大值，再通过双线性上采样生成定位热力图。

The implementation is intentionally compact and educational. It is inspired by PatchCore but does not claim exact architectural or benchmark equivalence with the paper.

该实现以教学和可维护性为目标，受 PatchCore 启发，但不声称与论文架构或基准结果完全等价。

## 6. Training and calibration｜训练与阈值校准

### Selected OCSVM pipeline

```text
HOG vector → StandardScaler → RBF OneClassSVM(nu=0.05, gamma="scale")
```

The 209 normal training images are split using `random_state=42`:

- 167 images fit the scaler and OCSVM.
- 42 images calibrate the decision threshold.

Anomaly scores are defined as:

```python
anomaly_score = -pipeline.decision_function(features)
```

This sign convention means that a larger value indicates a stronger anomaly. The threshold is the 95th percentile of the held-out normal calibration scores:

```python
threshold = np.quantile(calibration_scores, 0.95)
```

Prediction rule:

```python
predicted_label = 1 if anomaly_score > threshold else 0
```

其中 `1` 表示异常，`0` 表示正常。测试集标签不会参与阈值选择，这一点可避免直接使用测试答案调整模型。

### Isolation Forest comparison

The alternative model uses the same image preprocessing, HOG representation, normal-only split, and calibration concept. Keeping the input representation fixed makes the detector comparison easier to interpret.

Isolation Forest 使用相同的图片预处理、HOG 特征、正常样本划分和阈值校准方法，因此性能差异主要来自异常检测器本身。

## 7. Reproducing experiments｜复现实验

Run the commands in this order:

```bash
python src/explore_dataset.py
python src/train_hog_ocsvm.py
python src/train_hog_ocsvm_calibrated.py
python src/train_hog_isolation_forest.py
python src/train_patchcore.py
python src/compare_models.py
```

Expected high-level checks:

| Check | Expected value |
|---|---:|
| Training good images | 209 |
| Calibrated model-training images | 167 |
| Calibration images | 42 |
| Test images | 83 |
| Selected threshold | approximately 0.5604 |
| Selected model F1 | approximately 0.9756 |
| Deep patch-memory image F1 | approximately 0.9921 |
| Deep patch-memory pixel AUROC | approximately 0.9861 |
| Deep patch-memory AU-PRO@0.30 | approximately 0.9516 |

Small latency differences are expected across machines. Classification metrics should remain stable when the dataset and dependency versions are unchanged.

不同计算机上的推理时间可能略有不同；在数据集和依赖版本不变时，分类指标应保持一致。

## 8. Output contract｜输出约定

Each evaluated model writes its results to a dedicated directory under `outputs/`.

每个模型将结果保存到 `outputs/` 下的独立目录。

| File | Meaning / 含义 |
|---|---|
| `metrics.json` | Configuration, aggregate metrics, confusion matrix, and per-category accuracy / 配置、总体指标、混淆矩阵和分类别准确率 |
| `predictions.csv` | Image path, category, true label, anomaly score, and predicted label / 图片路径、类别、真实标签、异常分数和预测标签 |
| `confusion_matrix.png` | Classification error summary / 分类错误汇总 |
| `roc_curve.png` | Ranking performance across thresholds / 不同阈值下的排序性能 |
| `precision_recall_curve.png` | Precision-recall trade-off / 精确率与召回率关系 |
| `score_distribution.png` | Score distribution by defect category / 各缺陷类别的分数分布 |
| `model.joblib` | Serialized inference artifact / 序列化推理模型 |
| `model.pt` | Tensor-only deep patch-memory artifact / 仅张量的深度补丁记忆模型 |
| `heatmap_examples.png` | Ground truth, anomaly maps, and overlays / 标注、异常图与叠加效果 |

## 9. Saved model contract｜模型文件约定

`outputs/hog_ocsvm_calibrated/model.joblib` contains a dictionary with four required keys:

```python
{
    "pipeline": fitted_sklearn_pipeline,
    "threshold": calibrated_float_threshold,
    "image_size": (128, 128),
    "hog_parameters": {
        "orientations": 9,
        "pixels_per_cell": (16, 16),
        "cells_per_block": (2, 2),
        "block_norm": "L2-Hys",
    },
}
```

`predict_image.py` reads the preprocessing metadata from this artifact instead of duplicating constants. This prevents training-serving skew.

`predict_image.py` 直接从模型文件读取图片尺寸和 HOG 参数，而不是重复写死配置，从而降低训练与推理不一致的风险。

Only load `joblib` files from trusted sources. Python pickle-based formats can execute code during deserialization.

只应加载可信来源的 `joblib` 文件，因为基于 pickle 的格式在反序列化时可能执行代码。

`outputs/patchcore/model.pt` stores the frozen encoder weights, 1,024 × 128 memory bank, image and pixel thresholds, input size, patch grid, and normalisation metadata. Loading uses `torch.load(..., weights_only=True)` and reconstructs the encoder locally; no network download is needed for inference.

`outputs/patchcore/model.pt` 保存冻结编码器权重、1,024 × 128 记忆库、图片与像素阈值、输入尺寸、补丁网格和归一化参数。推理时使用 `torch.load(..., weights_only=True)` 并在本地重建编码器，不需要联网下载权重。

## 10. Single-image inference｜单图推理

Default usage:

```bash
python src/predict_image.py path/to/image.png
```

Use another compatible model artifact:

```bash
python src/predict_image.py path/to/image.png --model path/to/model.joblib
```

Deep patch-memory inference with a saved heatmap:

```bash
python src/predict_patchcore.py path/to/image.png
```

Known validation examples:

```bash
python src/predict_image.py data/mvtec_ad/bottle/test/good/000.png
python src/predict_image.py data/mvtec_ad/bottle/test/broken_large/000.png
```

Expected results with the current artifact:

| Image | Score | Threshold | Prediction |
|---|---:|---:|---|
| `test/good/000.png` | 0.1084 | 0.5604 | Good |
| `test/broken_large/000.png` | 0.7405 | 0.5604 | Anomaly |

## 11. Web demo｜网页演示

The public application is available at:

- **Live URL:** [https://bottle-anomaly-inspection.streamlit.app/](https://bottle-anomaly-inspection.streamlit.app/)
- **Source branch:** `main`
- **Entrypoint:** `app/app.py`
- **Cloud runtime:** Python 3.14

The deployment reads the public GitHub repository directly. A push to `main` triggers Streamlit Community Cloud to rebuild the application. No secrets or private-repository access are required.

公开应用地址如下：

- **在线地址：**[https://bottle-anomaly-inspection.streamlit.app/](https://bottle-anomaly-inspection.streamlit.app/)
- **代码分支：**`main`
- **入口文件：**`app/app.py`
- **云端运行环境：**Python 3.14

部署直接读取公开 GitHub 仓库。向 `main` 分支推送更新后，Streamlit Community Cloud 会重新构建应用；本项目不需要配置密钥，也不需要授予私有仓库访问权限。

Start the local bilingual interface from the repository root:

```bash
streamlit run app/app.py
```

The app offers a model selector. The HOG option loads `model.joblib`; the deep option loads `model.pt` and displays a heatmap and overlay. Uploaded files are processed in memory and are not added to the dataset or repository. Reliable use requires a bottle-opening image with framing and acquisition conditions similar to the training data.

网页提供模型选择器：HOG 选项加载 `model.joblib`，深度选项加载 `model.pt` 并显示热力图和叠加图。上传图片仅在内存中处理，不会自动加入数据集或仓库。可靠使用需要上传构图与采集条件接近训练数据的俯视瓶口图片。

Use `Control + C` in the terminal running Streamlit to stop the local server.

在运行 Streamlit 的终端中按 `Control + C` 可以停止本地服务。

## 12. Automated testing and CI｜自动测试与持续集成

Run the data-independent test suite from the repository root:

```bash
python -m unittest discover -s tests -v
```

The tests validate:

- the 1,764-dimensional HOG feature contract;
- Good and Anomaly decisions on opposite sides of the calibrated threshold;
- deterministic coreset selection, patch-distance scoring, and heatmap dimensions;
- clear errors for missing image and model paths; and
- the required metadata stored in the versioned `model.joblib` artifact.

The workflow at `.github/workflows/ci.yml` runs on every push to `main` and every pull request. It installs the pinned dependencies on Python 3.14, checks Python syntax, and executes the unit tests. The workflow intentionally avoids downloading the MVTec AD dataset, so CI remains deterministic and suitable for a public repository.

在仓库根目录运行以下命令即可执行不依赖原始数据集的测试：

```bash
python -m unittest discover -s tests -v
```

测试覆盖 HOG 特征维度、校准阈值两侧的分类结果、图片和模型路径校验，以及已提交模型文件的字段约定。GitHub Actions 会在每次向 `main` 推送代码或创建 Pull Request 时，使用 Python 3.14 安装固定版本依赖、检查 Python 语法并运行测试；CI 不会下载 MVTec AD 数据集。

## 13. Extending the project｜扩展项目

### Add another detector｜增加新的异常检测器

1. Copy the calibrated training structure into a new clearly named script.
2. Keep the same train/calibration/test split for a fair comparison.
3. Fit the detector only on model-training normal images.
4. Convert the detector output so larger scores always mean more anomalous.
5. Calibrate the threshold using only normal calibration scores.
6. Save the same metrics and prediction columns.
7. Store outputs in a new model-specific directory.
8. Add the new metrics path to `compare_models.py`.

### Add colour or texture features｜增加颜色或纹理特征

Recommended options include colour histograms, Local Binary Patterns, or pretrained embeddings. When adding a feature extractor:

- Store all feature parameters in `model.joblib`.
- Reuse exactly the same transformation during inference.
- Record the resulting feature dimension in `metrics.json`.
- Compare models with the same test set and threshold protocol.

### Add localisation｜增加缺陷定位

The current output is one label per image. Pixel-level localisation requires a separate method and evaluation protocol using the ground-truth masks. Do not describe the current model as a localisation model.

当前模型每张图片只输出一个标签。像素级定位需要单独的方法，并使用 ground-truth mask 评估；不要将当前系统描述为缺陷定位模型。

## 14. Troubleshooting｜常见问题

### `Dataset not found`

- Confirm that the command is executed from the repository root.
- Confirm that `data/mvtec_ad/bottle/` exists.
- Confirm that category folder names match the expected structure.

### `Model not found`

Run the calibrated training script first:

```bash
python src/train_hog_ocsvm_calibrated.py
```

### Feature-count or shape error

The model and image preprocessing configuration do not match. Retrain the model after changing image size or HOG parameters, and ensure inference reads configuration from the new artifact.

### Results differ from the documented values

Check dataset completeness, dependency versions, source changes, the random state, and whether the correct output directory or model artifact is being used.

### PowerShell blocks environment activation

Use an appropriate PowerShell execution policy approved for the machine, or run the commands in Command Prompt with:

```bat
.venv\Scripts\activate.bat
```

### Streamlit page does not open

- Confirm the virtual environment is active and `streamlit --version` succeeds.
- Confirm port 8501 is not already occupied by another process.
- Copy the Local URL printed in the terminal into the browser.
- Stop an existing server with `Control + C` before starting another one.

## 15. Validation checklist｜验收清单

Before merging a change:

- [ ] Run `python -m py_compile` on modified Python files.
- [ ] Run `python -m unittest discover -s tests -v`.
- [ ] Confirm the dataset remains untracked.
- [ ] Run the affected training or inference command.
- [ ] Check that scores use the documented direction: larger means more anomalous.
- [ ] Confirm calibration uses normal images only.
- [ ] Inspect `metrics.json` and `predictions.csv`.
- [ ] Inspect newly generated figures for clipping or incorrect labels.
- [ ] Test at least one known good and one known anomalous image.
- [ ] Launch the Streamlit app after changing inference or UI code.
- [ ] Update README and this guide when commands, files, or model contracts change.

提交修改前：

- [ ] 对修改过的 Python 文件执行语法检查。
- [ ] 运行 `python -m unittest discover -s tests -v`。
- [ ] 确认数据集没有进入 Git。
- [ ] 运行受影响的训练或推理命令。
- [ ] 确认异常分数方向仍为“越大越异常”。
- [ ] 确认阈值校准只使用正常图片。
- [ ] 检查指标、预测 CSV 和生成图表。
- [ ] 至少测试一张正常图片和一张异常图片。
- [ ] 修改推理或界面代码后启动 Streamlit 页面验证。
- [ ] 当命令、目录或模型格式变化时同步更新文档。

## 16. Contribution workflow｜协作流程

1. Pull the latest `main` branch.
2. Make one focused change at a time.
3. Validate locally using the checklist above.
4. Review the Git diff and exclude datasets, virtual environments, temporary Office files, and macOS metadata.
5. Use a concise commit summary that describes the outcome.
6. Review reproducibility and documentation completeness before merging or publishing the change.

建议每次提交只包含一个明确目标。不要提交 `data/`、`.venv/`、`~$` 开头的 Office 临时文件或 `.DS_Store`。

## 17. Current limitations｜当前限制

- Image-level detection only; no pixel-level localisation.
- Grayscale HOG underrepresents colour and subtle texture changes.
- Evaluation covers one MVTec AD object category.
- The good test set is small, so the specificity estimate has uncertainty.
- The documented metrics come from one fixed calibration split.

Recommended next steps are colour and texture features, pretrained visual embeddings, repeated calibration splits, additional object categories, and mask-based localisation evaluation.

建议下一步增加颜色与纹理特征、预训练视觉特征、多次随机校准划分、更多物体类别以及基于掩膜的定位评估。
