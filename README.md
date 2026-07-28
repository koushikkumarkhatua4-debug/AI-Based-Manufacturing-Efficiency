# AI-Based Manufacturing Efficiency Classification

## Contents
- `Manufacturing_Efficiency_Report.docx` — Full research report (EDA, methodology, results, recommendations)
- `train_model.py` — Model training script (run this first to regenerate models/)
- `app/app.py` — Streamlit dashboard application
- `models/` — Trained model, scaler, and encoders (already trained, ready to use)
- `Thales_Group_Manufacturing.csv` — Original dataset
- `processed_data.csv` — Dataset with engineered features + predictions (used by the app)
- `requirements.txt` — Python dependencies

## How to run the Streamlit app

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. From this project's root folder, run:
   ```
   streamlit run app/app.py
   ```

3. The app will open in your browser at http://localhost:8501

## How to retrain the model (optional)

```
python train_model.py
```

This regenerates the model files inside `models/` and refreshes `processed_data.csv`.

## Model Performance

Best model: Random Forest
See the report for full accuracy, F1-score, and feature importance details.
