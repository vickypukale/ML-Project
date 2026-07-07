# Career Path Prediction and Guidance System

A full ML/DL pipeline + Streamlit app that recommends a career path for a
student based on their academic profile, skills, and interests, following
the workflow:

```
Install libraries → Load & preprocess data → EDA → Train ML & DL models
→ Evaluate models → Integrate best model into Streamlit UI
→ Recommend courses based on predicted career
```

## 1. Project structure

```
career_project/
├── data/
│   └── PS2_Dataset.csv          # raw dataset (6901 rows, 20 columns)
├── models/                      # created by train.py
│   ├── best_model.pkl / best_model.keras   # winning model (auto-saved)
│   ├── best_model_meta.json     # which model won + its metrics
│   ├── encoders.pkl             # LabelEncoders (fit on training data)
│   ├── scaler.pkl               # StandardScaler for numeric features
│   └── feature_cols.pkl         # exact feature column order
├── outputs/                     # created by eda.py / train.py
│   ├── eda/*.png                # exploratory plots
│   ├── model_comparison.json    # accuracy/F1/precision/recall per model
│   ├── classification_report_best_model.txt
│   └── confusion_matrix_best_model.csv
├── preprocessing.py              # cleaning + encoding, shared by train & app
├── eda.py                        # exploratory data analysis
├── train.py                      # trains ML + DL models, saves the best
├── app.py                        # Streamlit UI (predict + course recommend)
└── requirements.txt
```

## 2. Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 3. Run the pipeline

```bash
# Step 1 — optional, generates plots in outputs/eda/
python eda.py

# Step 2 — trains 4 ML models (Logistic Regression, Random Forest, SVM,
# XGBoost) + 1 DL model (Keras MLP), evaluates all of them on a held-out
# test set, and saves the best one (by weighted F1) to models/
python train.py

# Step 3 — launch the web app
streamlit run app.py
```

The app loads whatever model `train.py` decided was best (ML or DL) —
you don't need to hardcode which one; `best_model_meta.json` records it
and `app.py` reads it automatically.

## 4. How the model is picked

`train.py` trains all 5 models on an 80/20 stratified split and compares
them on **accuracy, precision, recall, and weighted F1**. The model with
the highest weighted F1 is copied to `models/best_model.pkl` (scikit-learn/
XGBoost) or `models/best_model.keras` (the Keras deep-learning model), and
`app.py` transparently loads whichever one won.

## 5. ⚠️ Important note on model performance

During evaluation, **all five models (including the deep learning model)
score close to the random-guess baseline (~8–10% accuracy, where 12
balanced classes means ~8.3% is chance level)**. A mutual-information
check between every input feature and the target confirms this isn't a
bug — the columns in `PS2_Dataset.csv` carry essentially no statistical
signal about `Suggested Job Role` (this is a widely-used *synthetic*
practice dataset, not real student records).

The full pipeline (preprocessing → training → evaluation → Streamlit
integration → course recommendation) is complete and production-shaped,
so you can drop in a dataset with genuine signal and it will work the
same way. If you want, this can be extended with:
- Feature engineering / combining columns into stronger signals
- Hyperparameter tuning (GridSearchCV / Optuna)
- A larger or real-world labeled dataset

## 6. Course recommendation logic

`app.py` contains a dictionary (`COURSE_RECOMMENDATIONS`) mapping each of
the 12 predicted career labels to 3 relevant courses. After the model
predicts a career, the app looks up that career in the dictionary and
displays the matching courses — this is the "IF-loop / lookup" step called
for in the project workflow.

## 7. Retraining with new data

Replace `data/PS2_Dataset.csv` with a new CSV that has the **same column
names**, then re-run `python train.py`. All encoders/scalers are refit
from scratch so the app stays in sync with the new data automatically.
