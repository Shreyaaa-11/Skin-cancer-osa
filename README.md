# Multi-Class Skin Cancer Classification with OSA Threshold Optimization

Production-grade and research-grade system for 7-class skin lesion classification with post-hoc threshold optimization.

## 1) Research Objective

Given an image \(x\), the model predicts class probabilities:

\[
\mathbf{p}(x) = [p_1(x), p_2(x), \ldots, p_7(x)], \quad \sum_i p_i(x)=1
\]

Baseline decision:

\[
\hat{y}_{argmax} = \arg\max_i p_i(x)
\]

Proposed calibrated decision:

\[
\hat{y}_{thr} = \arg\max_i \left(p_i(x)-t_i\right), \quad \mathbf{t}\in[0,1]^7
\]

Optimization:

\[
\max_{\mathbf{t}} F1_{macro}(\mathbf{t}) \quad \text{s.t.}\quad 0\le t_i \le 1
\]

Why this helps:
- Argmax can be suboptimal under class imbalance because dominant classes can win with small margins.
- Class-specific thresholds shift the decision boundary per class, improving minority-class recall (e.g., melanoma).
- Post-hoc optimization is model-agnostic and can be applied after model training without re-fitting the network.

## 2) Model Design Rationale

- **Transfer Learning (EfficientNetB0/ResNet50)**: medical datasets are often small; pretraining brings robust low/mid-level visual priors.
- **GlobalAveragePooling2D**: lowers parameter count versus flattening, reducing overfitting.
- **BatchNorm + Dense(ReLU) + Dropout(0.5)**: stabilizes optimization and improves generalization.
- **Softmax output**: required to obtain calibrated class probability vector for threshold optimization.

Training strategy:
1. Stage-1: frozen backbone, train head.
2. Stage-2: unfreeze top backbone blocks, fine-tune with low LR.
3. Class imbalance: class weights or focal loss.
4. Metrics: accuracy, macro-F1 (primary), per-class recall (melanoma emphasized).

## 3) Project Structure

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
  index.html
  src/
    main.jsx
    App.jsx
    components/ImageUploader.jsx
    components/PredictionResult.jsx
    components/Loader.jsx
    services/api.js

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

## 4) Run Training + Experiments

1) Install ML dependencies:
- `pip install tensorflow scikit-learn numpy pandas matplotlib pyyaml`

2) Prepare dataset as:
- `data_root/train/<class_name>/*.jpg`
- `data_root/val/<class_name>/*.jpg`
- `data_root/test/<class_name>/*.jpg`

3) Train:
- `python ml/training/train.py --data_root <path_to_data_root> --backbone efficientnetb0`

Outputs:
- `backend/artifacts/model.keras`
- validation/test predictions for optimization and analysis

4) Run optimizer comparison + ablations:
- `python ml/experiments/run_experiments.py --predictions_npz ml/data/test_predictions.npz`

## 5) Backend (FastAPI)

Run locally:
- `pip install -r backend/requirements.txt`
- `uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload`

API:
- `GET /health`
- `POST /predict` with multipart image file

Sample curl:

```bash
curl -X POST "http://localhost:8000/predict" ^
  -H "accept: application/json" ^
  -H "Content-Type: multipart/form-data" ^
  -F "file=@sample.jpg"
```

## 6) Frontend (React + Vite)

Run:
- `cd frontend`
- `npm install`
- `npm run dev`

Set backend URL with:
- `VITE_API_BASE_URL=http://localhost:8000`

## 7) Docker (Backend)

Build:
- `docker build -t skin-cancer-api ./backend`

Run:
- `docker run -p 8000:8000 skin-cancer-api`

## 8) Expected Experimental Outputs

- Baseline argmax metrics
- +TTA metrics
- +TTA + threshold optimization metrics
- Confusion matrix
- Per-class recall
- Convergence curves for Grid, Random, GA, PSO, OSA
- Stability table (mean/std over multiple runs)

