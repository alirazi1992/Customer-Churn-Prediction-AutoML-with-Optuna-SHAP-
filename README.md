# 📊 Day 12 — Customer Churn Prediction (XGBoost + Optuna + SHAP)

This project predicts whether a customer is likely to cancel their service (**churn**) based on demographic, financial, and engagement features.  
The goal is to help companies **identify at-risk customers early** and take proactive retention actions.

---

## 🧾 Dataset
**Churn Modelling Dataset**  
https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling

| Rows | Columns | Target | Task Type |
|------|--------|--------|-----------|
| 10,000 | 14 | `Exited` | Binary Classification |

---

## 🛠️ Project Workflow
1. Data Exploration & Cleaning  
2. Feature Encoding (One-Hot for Categoricals)  
3. Train/Test Stratified Split  
4. Baseline Models (LogReg, RandomForest, XGBoost)  
5. **Hyperparameter Optimization using Optuna**  
6. Evaluation (Accuracy, F1, ROC-AUC, Confusion Matrix)  
7. **Model Interpretability with SHAP**  
8. Business Insights & Recommendations

---

## 📈 Model Performance Comparison

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|------|:--------:|:---------:|:------:|:--------:|:-------:|
| **XGBoost (Optuna Tuned)** | **0.863** | **0.74** | 0.50 | 0.60 | **0.867** |
| Random Forest | 0.858 | 0.69 | 0.54 | 0.61 | 0.857 |
| Logistic Regression | 0.713 | 0.39 | 0.70 | 0.49 | 0.777 |

> **Why XGBoost Wins:** Best balance of **ranking ability**, **precision**, and **generalization**.

---

## 🎯 Confusion Matrix — Best Model
Shows classification performance on test data.

True: No Churn → Predicted Correctly: 1544
True: Churn → Predicted Correctly: 194


This model **successfully identifies high-risk churn customers**, while keeping false alarms controlled.

---

## 🔍 SHAP Feature Importance (Model Explainability)

**Top drivers of churn:**
| Feature | Insight |
|--------|---------|
| **NumOfProducts** | Customers with 1 product churn more → upsell bundle opportunities |
| **IsActiveMember** | Inactive customers are at highest risk → re-engagement campaigns |
| **Geography** (Germany > Spain/France) | Region-specific behavior patterns |
| **Age** | Mid-age groups show stronger churn tendencies |
| **Tenure** | Newer customers require proactive onboarding |

---

## 💡 Business Recommendations

| Issue | Recommendation |
|------|---------------|
| Low product usage | Offer bundled plans / loyalty incentives |
| Low customer engagement | Trigger re-engagement campaigns & personalized outreach |
| Region-specific churn patterns | Adjust pricing / service messaging by region |
| Early-tenure churn risk | Build **30–90 day onboarding journey** |

---

## 💾 Model Saving & Inference Example

```python
from joblib import dump, load

# Save
dump(best_model, "churn_model.joblib")

# Predict for new sample
proba = best_model.predict_proba(new_customer)[0][1]
print("Churn probability:", proba)
```

## ⭐ Summary

This project demonstrates:

- End-to-end ML workflow

- Real business use case

- Hyperparameter tuning (Optuna)

- Model explainability (SHAP)

- Actionable strategy recommendations
