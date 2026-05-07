# Local Setup Guide

This guide explains how to run the project locally on Windows.

## 1. Project Location

Open PowerShell and move into the project folder:

```powershell
cd "C:\Users\Shreya S\Downloads\final year project\Skin-cancer-osa"
```

## 2. Prerequisites

Install these first:

- Python 3.10 or 3.11
- Node.js 18 or later
- `pip`
- `npm`

Check versions:

```powershell
python --version
pip --version
node --version
npm --version
```

## 3. Backend Setup

Create a virtual environment:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.venv\Scripts\Activate.ps1
```

Install backend dependencies:

```powershell
pip install -r backend\requirements.txt
```

## 4. Required Model Files

This project expects these files to exist:

- `backend\artifacts\model.keras`
- `backend\artifacts\thresholds.json`

They are already present in this repository, so you do not need to generate them before running the API.

If `thresholds.json` is missing, the backend will still run with default zero thresholds.
If `model.keras` is missing, prediction will fail.

## 5. Run the Backend

Start the FastAPI server from the project root:

```powershell
python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

The backend will run at:

```text
http://localhost:8000
```

Health check:

```text
http://localhost:8000/health
```

Keep this terminal open.

## 6. Frontend Setup

Open a second PowerShell window and go to the frontend folder:

```powershell
cd "C:\Users\Shreya S\Downloads\final year project\Skin-cancer-osa\frontend"
```

Install frontend dependencies:

```powershell
npm install
```

Set the backend URL for the current PowerShell session:

```powershell
$env:VITE_API_BASE_URL="http://localhost:8000"
```

Start the frontend:

```powershell
npm run dev
```

Vite will print a local URL, usually:

```text
http://localhost:5173
```

Open that URL in your browser.

## 7. How to Use the App

1. Start the backend.
2. Start the frontend.
3. Open the frontend URL in your browser.
4. Upload a skin lesion image.
5. View the predicted class and confidence values returned by the API.

## 8. Quick API Test

You can test prediction directly after the backend starts:

```powershell
curl.exe -X POST "http://localhost:8000/predict" -H "accept: application/json" -F "file=@sample.jpg"
```

Replace `sample.jpg` with the image path you want to test.

## 9. Common Issues

### PowerShell blocks virtual environment activation

Run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then activate again:

```powershell
.venv\Scripts\Activate.ps1
```

### `Model not found` error

Make sure this file exists:

```text
backend\artifacts\model.keras
```

### Frontend cannot connect to backend

Make sure:

- backend is running on port `8000`
- `VITE_API_BASE_URL` is set to `http://localhost:8000`
- no other app is blocking port `8000`

### `npm install` or `pip install` fails

Make sure Node.js and Python are installed correctly and available in `PATH`.

## 10. Optional Training Workflow

If you want to train the model again instead of using the included artifact:

```powershell
python ml\training\train.py --data_root <path_to_data_root> --backbone efficientnetb0
```

Dataset structure should be:

```text
data_root/
  train/
    <class_name>/
  val/
    <class_name>/
  test/
    <class_name>/
```

## 11. Recommended Run Order

Use this order every time:

1. Open terminal in project root.
2. Activate `.venv`.
3. Run backend on port `8000`.
4. Open second terminal in `frontend`.
5. Set `VITE_API_BASE_URL`.
6. Run frontend with `npm run dev`.
7. Open the Vite URL in the browser.
