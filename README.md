# Titanic Survival Prediction: Comparing Classification Models

A classification project that predicts passenger survival on the Titanic (Seaborn `titanic` dataset) and compares several machine learning algorithms.

## Project Overview

The notebook walks through a complete, beginner-friendly machine learning workflow:

1. **Data loading and initial exploration**
2. **Exploratory Data Analysis (EDA):** missing values, target distribution, univariate and bivariate analysis, correlation heatmap
3. **Preprocessing:** dropping unnecessary columns, train/test split, feature scaling and one-hot encoding
4. **Model training:** hyperparameter tuning with `GridSearchCV` (5-fold cross-validation)
5. **Model comparison:** accuracy, precision, recall, F1-score, AUC, confusion matrices and ROC curves

## Models Compared

| Model |
|---|
| Logistic Regression |
| K-Nearest Neighbors (KNN) |
| Support Vector Machine (SVM) |
| Decision Tree |
| Random Forest |
| AdaBoost |

## Results

Results on the held-out test set (20% of the data), sorted by AUC:

| Model | Accuracy | Precision | Recall | F1-Score | AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.793 | 0.792 | 0.793 | 0.789 | 0.847 |
| Random Forest | 0.816 | 0.825 | 0.816 | 0.807 | 0.843 |
| KNN | 0.793 | 0.791 | 0.793 | 0.790 | 0.839 |
| AdaBoost | 0.782 | 0.780 | 0.782 | 0.779 | 0.825 |
| SVM | 0.804 | 0.806 | 0.804 | 0.799 | 0.824 |
| Decision Tree | 0.782 | 0.781 | 0.782 | 0.777 | 0.792 |

Random Forest has the best accuracy and Logistic Regression has the best AUC. The models perform within a narrow range, which is expected on a small dataset (891 passengers).

## Project Structure

```
.
├── titanic_classification_comparison.ipynb   # Main notebook (EDA + modeling + comparison)
├── requirements.txt                          # Python dependencies
└── README.md
```

## How to Run

1. Make sure Python 3.9 or newer is installed.
2. (Optional) Create and activate a virtual environment:

```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS / Linux
```

3. Install the dependencies:

```bash
pip install -r requirements.txt
```

4. Launch Jupyter and open the notebook:

```bash
jupyter notebook titanic_classification_comparison.ipynb
```

> **Note:** The dataset is downloaded automatically by `seaborn.load_dataset("titanic")`, so an internet connection is needed the first time you run the notebook.

## Tools Used

Python, NumPy, pandas, Matplotlib, Seaborn, scikit-learn, Jupyter Notebook.

## License

This is a private project. All rights reserved.
