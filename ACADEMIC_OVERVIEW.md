# Academic Overview — Multi‑Class Skin Cancer Classification with OSA Threshold Optimization

This document is written in a **student → professor** explanation style. It provides the **complete context** of the project: what problem we solved, how the dataset is handled, how the model is trained, why the chosen model makes sense, how the classifier makes decisions, and how the **frontend + backend** implement the end‑to‑end system.

---

## 1) What problem are we solving?

We solve **multi-class skin lesion classification** from dermoscopic images into **7 diagnostic categories** (a standard setting aligned with HAM10000-style labels). 

Given an image \(x\), the network predicts a probability vector:

\[
\mathbf{p}(x) = [p_1(x), p_2(x), \ldots, p_7(x)], \quad \sum_{i=1}^{7} p_i(x)=1
\]

The classic decision rule is **argmax**:

\[
\hat{y}=\arg\max_i p_i(x)
\]

However, medical datasets are **imbalanced** (some classes have many more samples than others). Under imbalance, plain argmax often biases decisions toward majority classes and hurts macro‑F1 / minority recall (clinically important for melanoma).

So we add a **post-hoc threshold optimization layer** (OSA optimizer) that learns a per‑class threshold vector \(\mathbf{t}\in[0,1]^7\). The deployed decision rule becomes:

\[
\hat{y}=\arg\max_i (p_i(x)-t_i)
\]

This is model‑agnostic: we do **not retrain** the CNN to apply thresholds; we adjust the decision boundary after training to maximize **macro‑F1**.

---

## 2) What are the dataset classes?

The system is built around these **7 classes** (short codes used in training, inference, backend artifacts, and frontend UI):

1. `akiec` — Actinic keratoses / intraepithelial carcinoma  
2. `bcc` — Basal cell carcinoma  
3. `bkl` — Benign keratosis-like lesions  
4. `df` — Dermatofibroma  
5. `mel` — Melanoma  
6. `nv` — Melanocytic nevi (moles)  
7. `vasc` — Vascular lesions  

### Dataset organization expected by training

Training uses `tf.keras.utils.image_dataset_from_directory`, which expects this folder structure:

```text
<data_root>/
  train/
    akiec/*.jpg
    bcc/*.jpg
    ...
  val/
    akiec/*.jpg
    ...
  test/
    akiec/*.jpg
    ...
```

Why this layout?
- It is a standard Keras pipeline.
- Labels are inferred from folder names; the class indices are stable and reproducible.

---

## 3) What is the complete project structure?

```text
backend/
  app/
    main.py
    models/schemas.py
    routes/health.py
    routes/predict.py
    services/predict_service.py
    utils/config.py
    utils/logging_config.py
  inference/
    model_loader.py
    predictor.py
    preprocessing.py
  artifacts/
    model.keras
    thresholds.json
  Dockerfile
  requirements.txt

frontend/
  package.json
  vite.config.js
  src/
    App.jsx
    services/api.js
    components/ImageUploader.jsx
    components/PredictionResult.jsx
    components/Loader.jsx

ml/
  config/settings.py
  models/efficientnet_model.py
  training/train.py
  evaluation/metrics.py
  optimization/
    base.py
    grid_search.py
    random_search.py
    ga.py
    pso.py
    osa.py
  experiments/run_experiments.py
```

How to read this as a professor:
- **`ml/`** is the research/training/experimentation code (offline).
- **`backend/`** is the deployment/inference API (online).
- **`frontend/`** is the UI that consumes the API.

---

## 4) Data preprocessing and what “classification” means here

### 4.1 Image preprocessing (training and inference)

At inference time, the backend does:

1. Decode uploaded bytes into an RGB image (PIL).
2. Resize to **224×224** (matches the model input size).
3. Convert to float32 and normalize to **[0,1]**.
4. Add batch dimension: shape becomes `(1, 224, 224, 3)`.

This matches what training does: the training script normalizes pixels with `x/255.0`.

### 4.2 What the CNN is learning

The CNN learns a mapping:

\[
f_\theta: \mathbb{R}^{224\times224\times3} \rightarrow \Delta^7
\]

where \(\Delta^7\) is the probability simplex (7 probabilities summing to 1).

It is trained with **sparse categorical cross entropy**:

\[
\mathcal{L} = -\log(p_{y}(x))
\]

where \(y\) is the true class index.

---

## 5) Model choice: why EfficientNetB0 (and ResNet50) and not training from scratch?

### 5.1 Transfer learning rationale (academic justification)

Medical image datasets are often **limited** compared to natural image datasets. Training a deep CNN from scratch usually leads to:
- Overfitting (memorizing training images)
- Poor generalization
- Large compute requirements

So we use **transfer learning**:
- Start from a backbone pretrained on ImageNet.
- Replace the final classifier head for our 7 classes.
- Fine‑tune carefully.

### 5.2 EfficientNetB0 vs ResNet50 in this project

This project supports two backbones:
- `efficientnetb0`
- `resnet50`

**Why EfficientNetB0 is a strong default:**
- EfficientNet family is designed for better accuracy/compute trade‑off using compound scaling.
- B0 is lightweight: good for deployment (faster inference, smaller memory footprint).
- Often works very well for medical images after fine‑tuning.

**Why include ResNet50 anyway:**
- ResNet is a classic strong baseline with stable optimization.
- Comparing backbones is academically relevant.

**Why not “other models” (e.g., Vision Transformers, larger EfficientNets, etc.)?**
- Transformers often need more data/augmentation to outperform CNNs reliably.
- Larger backbones (EfficientNetB3/B7) may improve accuracy but increase:
  - training time
  - inference latency
  - deployment cost
- The project goal is a **balanced research + deployable system**, not just a leaderboard model.

### 5.3 The head architecture and why it’s designed that way

The project’s model head:
- Backbone (feature extractor)
- GlobalAveragePooling2D (reduces parameters vs flattening)
- BatchNorm (stabilizes training)
- Dense + ReLU (nonlinear classifier head)
- Dropout(0.5) (regularization)
- Dense + softmax (outputs 7 probabilities)

Softmax is essential because:
- We need a probability vector for **threshold optimization**.
- The backend returns per‑class probabilities for interpretability.

---

## 6) Training: exactly how it’s done (two-stage fine‑tuning)

Training code: `ml/training/train.py`

### 6.1 Dataset loading

The script creates three datasets:
- Train: shuffled
- Validation: not shuffled
- Test: not shuffled

It uses:
- `image_dataset_from_directory(...)` with inferred labels
- A preprocessing map: `x = tf.cast(x, tf.float32)/255.0`
- Prefetching for performance

### 6.2 Handling class imbalance (critical in medical datasets)

The script computes **balanced class weights** from the training labels:

```python
class_weight="balanced"
```

Intuition:
- If a class has fewer examples, it gets a higher weight.
- Misclassifying rare classes is penalized more during training.

This helps improve macro‑F1 and minority recall.

### 6.3 Two training stages (why two stages?)

**Stage 1 (feature extractor mode):**
- Backbone frozen (`trainable=False`)
- Train only the new head
- Learning rate ≈ \(10^{-3}\)
- Purpose: learn a stable classifier head without destroying pretrained features.

**Stage 2 (fine-tuning mode):**
- Unfreeze last N backbone layers (default N=40)
- Smaller learning rate ≈ \(10^{-5}\)
- Purpose: adapt higher-level features to dermoscopic patterns.

### 6.4 Optimization details

Optimizer:
- Adam (good default for transfer learning)

Callbacks:
- EarlyStopping on validation loss (reduces overfitting)
- ReduceLROnPlateau (automatically lowers LR if validation plateaus)

### 6.5 Training outputs (artifacts)

At the end:
- The model is saved to: `backend/artifacts/model.keras`
- Test predictions are saved to: `ml/data/test_predictions.npz` containing:
  - `y_true`
  - `probs` (probability vectors)

These prediction dumps are later used by optimization algorithms (OSA, GA, PSO, etc.).

---

## 7) Threshold optimization: what it is, why we do it, and how OSA works

### 7.1 The motivation (professor-level)

Even with class weights, the trained model’s raw argmax decision may not maximize macro-F1.

We want to directly optimize the decision rule to improve **macro‑F1**, which treats each class equally:

\[
F1_{macro}=\frac{1}{7}\sum_{c=1}^{7}F1_c
\]

Threshold optimization:
- searches for a threshold vector \(\mathbf{t}\)
- uses a validation/test prediction dump
- chooses thresholds that give best macro‑F1

It’s like calibrating the classifier’s decision boundary per class.

### 7.2 What OSA (Owl Search Algorithm) is doing here

OSA is a population-based metaheuristic:
- Each “owl” represents a candidate threshold vector \(\mathbf{t}\in[0,1]^7\).
- Fitness of an owl = macro‑F1 when predictions are computed using `argmax(p - t)`.
- Iterations update owls by combining:
  - exploration noise (search new regions)
  - exploitation toward the best solution found so far (refine)
- Thresholds are clipped to `[0,1]`.

In code, the project’s OSA objective is:
- maximize `fitness_macro_f1(y_true, probs, thresholds)`

### 7.3 Why not only one optimizer?

Academically, comparing optimizers strengthens the work:
- Grid search: simple but expensive in higher dimensions
- Random search: baseline metaheuristic
- GA/PSO: well-known population methods
- OSA: the proposed/featured method for this project

The experiments script benchmarks them and exports:
- comparison table (mean/std F1, runtime)
- convergence curves
- best thresholds (deployed)

### 7.4 What gets deployed

The chosen thresholds are written to:

`backend/artifacts/thresholds.json`

The backend then uses them in production inference.

---

## 8) Backend API: how exactly prediction happens (step-by-step)

Backend is FastAPI.

### 8.1 API endpoints

- `GET /health`  
  Used to verify the server is alive (for deployment checks).

- `POST /predict`  
  Accepts an image file upload and returns a JSON response:
  - predicted label (string)
  - class index (0–6)
  - probabilities (length 7)
  - latency in ms

### 8.2 What happens inside `/predict`

1. Validate the file exists and content-type is supported.
2. Read bytes.
3. Convert bytes to PIL image and validate it’s a real image.
4. Fetch model singleton (cached in memory).
5. Run preprocessing and inference.
6. Apply TTA (optional).
7. Apply threshold-based decision rule.
8. Return response.

### 8.3 Model caching (why it matters)

The model is loaded once per process using an LRU cache.
This prevents:
- loading the model file on every request (slow)
- huge latency spikes

---

## 9) Test-Time Augmentation (TTA): what it is in this project

TTA is an inference technique: run multiple “views” of the same image and average predictions.

In this project the backend uses a **minimal deterministic TTA set**:
- original image
- horizontal flip
- slight brightness increase

Why this helps:
- Dermoscopy images may vary in lighting and orientation.
- Averaging reduces prediction variance.

Why it’s minimal:
- Keep inference fast enough for a web app.

You can disable it in backend settings (`enable_tta=false`) if needed.

---

## 10) Frontend: how exactly the web app works

### 10.1 What the UI does

1. User uploads an image (`jpeg/png/webp`).
2. UI shows a local preview.
3. User clicks “Predict”.
4. UI sends a multipart POST request to `/predict`.
5. UI displays:
   - top predicted class (human-friendly name)
   - all class probabilities (bars)
   - latency

### 10.2 Why show all class probabilities?

Academic + practical reasons:
- The model is not a definitive diagnosis; showing distribution is more transparent.
- Helps interpret ambiguous cases (close probabilities).
- Useful for demo/viva to discuss uncertainty.

---

## 11) Deployment: local + Docker

### 11.1 Local (recommended for development)

Backend:
- install backend deps
- run `uvicorn backend.app.main:app`

Frontend:
- `npm install`
- `npm run dev`
- optionally set `VITE_API_BASE_URL`

### 11.2 Docker (backend)

Docker builds a Python 3.11 slim image, installs requirements, copies backend code, and runs uvicorn.

Academic angle:
- Docker ensures reproducibility: same environment, fewer “it works on my machine” problems.

---

## 12) What are the key evaluation metrics and why?

### Accuracy (not enough alone)
Accuracy can look high if the majority class dominates (e.g., many `nv`).

### Macro‑F1 (primary)
Macro‑F1 gives equal weight to each class, making it better for imbalanced medical datasets.

### Per-class recall (clinically important)
For melanoma (`mel`), recall is crucial: missing melanoma is high risk.

This project’s threshold optimization directly targets macro‑F1 to reduce imbalance bias.

---

## 13) “Why does thresholding work mathematically?”

Argmax selects the class with the largest probability.

But if one class tends to have systematically higher scores (due to prevalence or bias), it can dominate.
Subtracting class-specific thresholds is equivalent to adding a per-class “penalty/offset”:

\[
\arg\max_i (p_i - t_i)
\]

This shifts the decision boundary to favor minority classes when justified by data.

Important nuance:
- Thresholds do not change the probability outputs.
- They change only the **decision** (the chosen label).

So the CNN remains the same; we improve decision policy.

---

## 14) Typical viva questions (with strong answers)

### Q1: Why EfficientNetB0 instead of a bigger model?
Because we need a model that is accurate but also deployable (fast, small). EfficientNetB0 offers a strong accuracy/compute trade‑off. Larger models may improve marginal accuracy but increase latency and deployment cost.

### Q2: Why not only accuracy as the metric?
Because the dataset is imbalanced. Accuracy can be misleading; macro‑F1 and per-class recall better reflect performance across minority classes, especially melanoma.

### Q3: Why do you need threshold optimization if you already used class weights?
Class weights help during training, but they don’t guarantee optimal decision boundaries for macro‑F1. Threshold optimization is a post-hoc step that directly optimizes macro‑F1 for the final decision rule.

### Q4: Is OSA guaranteed to find the global optimum?
No—OSA is a metaheuristic. But it is effective for searching a continuous 7D space and is benchmarked against other optimizers (Grid/Random/GA/PSO) in this project.

### Q5: Is the system a medical diagnostic tool?
No. It’s an ML-based classification system to assist research/demonstration. Clinical diagnosis requires professional evaluation and additional patient context.

### Q6: What happens if `thresholds.json` is missing?
The backend falls back to all-zero thresholds (equivalent to argmax). So the system remains functional, just without the optimized decision rule.

### Q7: Why 224×224 input size?
Both EfficientNetB0 and ResNet50 standard pretrained variants commonly use 224×224. It is a practical balance between feature detail and compute cost.

---

## 15) Exact “how to reproduce” checklist (professor-ready)

### Step A — Prepare dataset
- Put images into `train/`, `val/`, `test/` folders with 7 class subfolders exactly named:
  `akiec, bcc, bkl, df, mel, nv, vasc`.

### Step B — Train CNN
Run:
- `python ml/training/train.py --data_root <DATA_ROOT> --backbone efficientnetb0`

Outputs:
- `backend/artifacts/model.keras`
- `ml/data/test_predictions.npz`

### Step C — Run threshold optimization experiments
Run:
- `python ml/experiments/run_experiments.py --predictions_npz ml/data/test_predictions.npz --runs 3`

Outputs:
- `backend/artifacts/thresholds.json` (deployed)
- `ml/experiments/results/*` reports

### Step D — Start backend + frontend
- Backend: `uvicorn backend.app.main:app --port 8000`
- Frontend: `npm run dev` in `frontend/`

### Step E — Demo
- Upload dermoscopy image in UI.
- Observe predicted class + probability distribution.

---

## 16) What makes this project academically strong?

- **End-to-end pipeline**: data → training → evaluation → optimization → deployment → UI.
- **Imbalance-aware design**: class weights + macro‑F1 emphasis.
- **Decision policy research**: OSA threshold optimization is separable and measurable.
- **Reproducibility**: artifacts (`model.keras`, `thresholds.json`) + Docker backend.
- **Interpretability at demo level**: full probability vector displayed, not just label.

---

## 17) Where to look in code (mapping from concept → implementation)

- **Training pipeline**: `ml/training/train.py`
- **Model architecture**: `ml/models/efficientnet_model.py`
- **Metric computation**: `ml/evaluation/metrics.py`
- **Optimization experiments**: `ml/experiments/run_experiments.py`
- **OSA optimizer**: `ml/optimization/osa.py`
- **Backend API**: `backend/app/main.py`, `backend/app/routes/predict.py`
- **Inference logic**: `backend/inference/predictor.py`, `backend/inference/preprocessing.py`, `backend/inference/model_loader.py`
- **Frontend UI**: `frontend/src/App.jsx`, `frontend/src/services/api.js`, `frontend/src/components/*`

