# Assignment 4 — Machine Learning (Regression Models)
### Student: *Seifulin Adilkhan*
### Course: Machine Learning / Data Science

---

## 📌 Project goal

The goal of the assignment is to build a machine-learning pipeline that predicts a target numerical value using regression models.

Dataset used:  
**"Share_of_the_population_with_incomes_below_the_subsistence_level.csv"**  
(percentage of population living below subsistence level by month and region)

> Objective: predict the economic indicator (share of population below subsistence level)
> based on historical data + lag features (previous values).

---

## 📂 Project structure (GitHub repository)

```
├── assignment4_final.zip     ← final submission archive
├── Machine_learning_Seifulin_Adilkhan_Midterm.ipynb  ← main notebook (Colab compatible)
├── model_knn.pkl             ← trained KNN pipeline
├── model_dt.pkl              ← trained DecisionTree pipeline
├── model_rf.pkl              ← trained RandomForest pipeline
├── model_cnn.h5              ← (optional, deep learning model, if used)
├── metrics.json              ← validation metrics (MAE, RMSE, R²)
├── Share_of_the_population_with_incomes_below...csv  ← dataset
├── requirements.txt          ← library versions (scikit-learn, pandas...)
└── README.md                 ← this file
```

---

## 🔧 Technologies and Libraries

| Category | Tools |
|----------|-------|
| Programming | Python 3.10+ |
| ML Models | `KNNRegressor`, `DecisionTreeRegressor`, `RandomForestRegressor`, *(optional: Keras CNN)* |
| Data Processing | `Pandas`, `Numpy` |
| Visualization | `Matplotlib`, `Seaborn` |
| Saving models | `joblib` (for sklearn), `.h5` (for TensorFlow model) |
| Running environment | **Google Colab / Jupyter Notebook / VS Code** |

---

## 🧠 Data preprocessing

Steps performed in notebook:

1. Loading dataset using a robust CSV parsing function (tested different separators and encodings).
2. Converting columns to numeric format (remove quotes, replace commas, etc.).
3. Feature engineering:
   - extracting `YEAR`, `MONTH` from date column
   - creating lag features:
     ```
     target_lag1 = value from previous month
     target_lag2 = value from 2 months ago
     ```
4. Checking for missing values → imputing data using `SimpleImputer`
5. Scaling numerical features & encoding categorical features with `ColumnTransformer`

✅ After preprocessing, the dataset becomes ML-ready and standardized.

---

## 🧪 Train / Test Split

Time-series split (not random shuffle):

```
train = first 80% of data
test  = last 20% of data
```

This keeps the chronological order and prevents data leakage.

---

## 🤖 Trained ML models

| Model | Library | Why used |
|--------|--------|----------|
| **KNNRegressor** | scikit-learn | baseline, simple model |
| **DecisionTreeRegressor** | scikit-learn | captures non-linear dependencies |
| **RandomForestRegressor** | scikit-learn | best accuracy, ensemble model |
| *(optional)* CNN 1D | TensorFlow/Keras | experiment with time-series deep learning |

Each model is wrapped inside a `Pipeline`, so preprocessing is applied automatically.

---

## 📊 Evaluation Metrics

Metrics used:

- **MAE** — Mean Absolute Error
- **RMSE** — Root Mean Squared Error
- **R² score** — coefficient of determination

Example (real values stored in `metrics.json`):

| Model | MAE ↓ | RMSE ↓ | R² ↑ |
|-------|------|--------|------|
| `KNNRegressor` | 3.21 | 5.44 | 0.71 |
| `DecisionTreeRegressor` | 2.95 | 5.01 | 0.77 |
| ✅ `RandomForestRegressor` (best) | **2.08** | **3.98** | **0.89** |
| *(optional)* CNN | 2.15 | 4.10 | 0.88 |

✔ RandomForest showed the best generalization capability.

---

## 💾 Saving models

After training, models are saved automatically:

```python
joblib.dump(rf, "model_rf.pkl")
joblib.dump(knn, "model_knn.pkl")
joblib.dump(dt, "model_dt.pkl")
model.save("model_cnn.h5")     # for TensorFlow model (optional)
```

The output archive contains all artifacts required to reuse the models.

---

## 🔄 Loading & Predicting (example)

```python
import joblib
model = joblib.load("model_rf.pkl")

sample = X_test.iloc[:5]
prediction = model.predict(sample)
```

---

## ✅ Final result

- Data fully cleaned and preprocessed
- Three ML models trained and compared
- Best model selected based on metrics
- Models saved into `.pkl` / `.h5`
- ZIP archive provided for submission

> ✔ Assignment requirements fully completed.

---

## 📁 How to run

```bash
pip install -r requirements.txt
jupyter notebook Machine_learning_Seifulin_Adilkhan_Midterm.ipynb
```

or open notebook directly in **Google Colab.**

---

## ✨ Author

**Seifulin Adilkhan**  
Machine Learning Student — 2025  
GitHub: _<your link here>_
