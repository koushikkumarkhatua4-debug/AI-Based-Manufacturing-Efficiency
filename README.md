# AI-Based Manufacturing Efficiency Classification — Streamlit App

## ⚠️ IMPORTANT — Deploy this ENTIRE folder, not just app.py

The `ModuleNotFoundError` / file-not-found errors on Streamlit Cloud happen when only
`app.py` is uploaded on its own. The app needs the `models/`, `data/`, and `outputs/`
folders sitting right next to it. Upload/push **everything in this folder** to your
GitHub repo, keeping this exact structure:

```
your-repo/
├── app.py
├── requirements.txt
├── models/
│   ├── best_model.pkl
│   ├── scaler.pkl
│   ├── op_mode_encoder.pkl
│   ├── target_encoder.pkl
│   └── feature_cols.json
├── data/
│   ├── processed_data.csv
│   └── Thales_Group_Manufacturing.csv
└── outputs/
    └── feature_importance.csv
```

## Deploy on Streamlit Community Cloud

1. Create a new GitHub repository.
2. Copy **all files and folders from this package** into the repo (don't rename or move anything).
3. Commit and push to GitHub.
4. Go to https://share.streamlit.io → "New app" → select your repo → set **Main file path** to `app.py`.
5. Click **Deploy**. Streamlit Cloud will read `requirements.txt` automatically and install everything.

If you already created the app and it's stuck on the old error: open the app →
bottom-right **"Manage app"** → **"Reboot app"** after pushing the corrected files.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open http://localhost:8501

## Notes
- All file paths in `app.py` are resolved relative to the script's own location
  (`Path(__file__).resolve().parent`), so it works the same whether run locally,
  on Streamlit Cloud, or from any other folder — as long as the folder structure
  above is preserved.
- Model used: Random Forest (best of Logistic Regression / Random Forest / XGBoost
  comparison — see the accompanying research report for full metrics).
