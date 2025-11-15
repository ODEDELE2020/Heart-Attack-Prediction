# Project 5: Heart Attack Prediction

This project focuses on **predicting heart attack occurrences** using clinical and health-related features. The workflow includes exploratory data analysis (EDA), feature engineering, handling missing values and outliers, and building machine learning models to predict heart attack risk.


---

## Visuals & Demo

* **Project Diagram & Visuals:** ![EDA & Features](images/eda_visuals.png)  
* **Documentation Notebook:** [Download PDF](assets/documentation.pdf)

---

## Features
- Comprehensive **Exploratory Data Analysis (EDA)** to understand patterns and relationships
- **Data cleaning and preprocessing**
- **Feature engineering** including encoding and scaling
- **Handling missing values and outliers**
- **Machine learning models:** Logistic Regression, K-Nearest Neighbors (KNN), Decision Tree
- **Model evaluation** using accuracy, precision, recall, and other metrics
- **Baseline model creation** for future enhancements

---

## Dataset
The dataset is available on [Kaggle](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset) and contains clinical and health-related features, including age, sex, chest pain type, resting blood pressure, cholesterol, fasting blood sugar, maximum heart rate, exercise induced angina, slope, number of vessels, thallium stress test results, and the target variable (heart attack occurrence).

---

## Folder Structure

```

Project-05-Heart-Attack-Prediction/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── Heart_Attack_Prediction.ipynb      # Full analysis & ML pipeline
├── final_model.pkl                     # Saved baseline model
├── images/                             # EDA & feature visualization images
│   └── eda_visuals.png
├── assets/
│   ├── demo.mp4                        # Video demonstration
│   └── documentation.pdf               # Notebook/documentation

````

---

## How to Run

1. **Install Dependencies:**
```bash
pip install -r requirements.txt
````

2. **Run Notebook:**

```bash
jupyter notebook Heart_Attack_Prediction.ipynb
```

3. **Use Saved Model:**

```python
import joblib
model = joblib.load("final_model.pkl")
```

---

## Tech Stack

Python | Pandas | NumPy | Matplotlib | Seaborn | Scikit-Learn | Logistic Regression | KNN | Decision Tree

---

## Key Insights

* Explored the correlation between clinical features and heart attack risk
* Created meaningful features to enhance model prediction
* Logistic Regression, KNN, and Decision Tree provided baseline performance
* Provides a foundation for advanced models, hyperparameter tuning, and SHAP-based feature importance

---

## Summary

This project demonstrates a complete ML workflow for heart attack prediction:

* Data cleaning and EDA
* Feature engineering and preprocessing
* Model training and baseline evaluation
* Visualization of feature relationships and distributions

It establishes a **baseline model** to guide future improvements in predictive performance and interpretability.

---

## Author

**Emmanuel Odedele**
Machine Learning | AI | Data Science

```
