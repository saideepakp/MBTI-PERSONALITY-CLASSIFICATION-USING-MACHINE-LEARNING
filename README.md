# MBTI Personality Classification Using Machine Learning
### Built with Python | scikit-learn | XGBoost | Dataset: 60,000 Survey Responses

---

## About the Project

This project applies machine learning to automate MBTI personality type classification using structured survey response data. The Myers-Briggs Type Indicator (MBTI) classifies individuals into 16 personality types across four dichotomies: Introversion vs. Extraversion, Sensing vs. Intuition, Thinking vs. Feeling, and Judging vs. Perceiving. Traditionally reliant on manual evaluation, MBTI classification is time-consuming and difficult to scale. This project replaces that manual process with a multi-class machine learning pipeline capable of predicting personality types with up to 98.77% accuracy.

Five classification models were implemented, tuned, and evaluated: Logistic Regression, K-Nearest Neighbors, Support Vector Machine, Random Forest, and XGBoost. Each model was assessed using accuracy, precision, recall, F1-score, and cross-validation to identify the most reliable classifier for 16-class personality prediction.

**Tools and Technologies Used**

| Tool | Purpose |
|---|---|
| Python | Data processing, modeling, and evaluation |
| scikit-learn | Model implementation, preprocessing, and metrics |
| XGBoost | Gradient boosting classification |
| Pandas / NumPy | Data manipulation and feature engineering |
| Matplotlib / Seaborn | EDA visualizations and confusion matrices |

---

## Problem Statement

Manually administering and scoring MBTI assessments is slow, subjective, and impractical at scale. Organizations in human resources, education, and behavioral research need faster, more reliable ways to derive personality insights from structured data. The challenge was to build a machine learning framework that could accurately classify individuals into one of 16 MBTI personality types using 60 numerically encoded survey questions, while comparing multiple algorithms to determine which approach performs best on this high-dimensional, multi-class problem.

The project addressed the following analytical objectives:

- Can machine learning reliably predict all 16 MBTI personality types from survey responses?
- Which classification algorithm produces the best balance of accuracy, precision, recall, and generalizability?
- Which survey questions carry the most predictive weight for personality classification?
- How do models perform when each MBTI dichotomy is treated as a standalone binary classification target?

---

## Dataset Overview

The dataset contains 60,000 survey responses across 60 numerically encoded questions designed to capture behavioral preferences across the four MBTI dichotomies. Each row represents one individual's complete response profile along with their MBTI personality type as the target variable. The 16 target classes are ENFJ, ENFP, ENTJ, ENTP, ESFJ, ESFP, ESTJ, ESTP, INFJ, INFP, INTJ, INTP, ISFJ, ISFP, ISTJ, and ISTP.

The dataset is well balanced, with class counts ranging from 3,734 (INFP) to 3,761 (INFJ), ensuring minimal class imbalance and supporting fair, unbiased model training without requiring resampling techniques. Responses were encoded on a 7-point scale from Fully Agree (3) to Fully Disagree (-3). No missing values were identified and no duplicate removal was required. Features were standardized using scaling prior to model training, and the data was split 60/40 into training and test sets.

---

## Model Results

**Baseline Model (Stratified Strategy)**

The baseline model assigned predictions based on class distribution in the training data, establishing the lower bound for comparison.

- Accuracy: 6.22%
- F1 Score: 6.21%

**K-Nearest Neighbors (KNN)**

- Best K: 13
- Accuracy: 98.77%
- Precision: 98.77%
- Recall: 98.77%
- F1 Score: 98.77%
- Cross-Validation Accuracy: 98.59% ± 0.11%
- Cross-Validation F1 Score: 98.59% ± 0.11%

**Logistic Regression**

- Best Regularization Parameter (C): 0.046
- Accuracy: 91.82%
- Precision: 91.82%
- Recall: 91.82%
- F1 Score: 91.81%
- Cross-Validation Accuracy: 91.58% ± 0.25%
- Cross-Validation F1 Score: 91.58% ± 0.25%

**Support Vector Machine (SVM)**

- Best Parameters: C=1.0, gamma=scale, kernel=rbf
- Accuracy: 98.74%
- Precision: 98.74%
- Recall: 98.74%
- F1 Score: 98.74%
- Cross-Validation Accuracy: 98.58% ± 0.11%
- Cross-Validation F1 Score: 98.58% ± 0.11%

**Random Forest**

- Best Parameters: max_depth=20, min_samples_split=2, n_estimators=200
- Accuracy: 97.63%
- Precision: 97.64%
- Recall: 97.63%
- F1 Score: 97.63%
- Cross-Validation Accuracy: 97.41% ± 0.06%
- Cross-Validation F1 Score: 97.41% ± 0.06%

**XGBoost**

- Best Parameters: gamma=0, learning_rate=0.2, max_depth=5, n_estimators=200
- Accuracy: 97.63%
- Precision: 97.64%
- Recall: 97.63%
- F1 Score: 97.63%
- Cross-Validation Accuracy: 97.40% ± 0.18%
- Cross-Validation F1 Score: 97.40% ± 0.18%

**Model Performance Summary**

| Model | Accuracy | F1 Score | CV Accuracy |
|---|---|---|---|
| Baseline | 6.22% | 6.21% | N/A |
| Logistic Regression | 91.82% | 91.81% | 91.58% ± 0.25% |
| Random Forest | 97.63% | 97.63% | 97.41% ± 0.06% |
| XGBoost | 97.63% | 97.63% | 97.40% ± 0.18% |
| SVM | 98.74% | 98.74% | 98.58% ± 0.11% |
| KNN | 98.77% | 98.77% | 98.59% ± 0.11% |

---

## Experimental Analysis: Dichotomy-Level Classification

The SVM model was re-run four times, each time treating a single MBTI dichotomy as a standalone binary classification target. All four runs produced accuracy comparable to the full 16-class model, confirming that the model's predictive power holds at both the full type and individual dimension level.

| Dichotomy | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| E vs I | 98.78% | 0.99 | 0.99 | 0.99 |
| N vs S | 98.75% | 0.99 | 0.99 | 0.99 |
| F vs T | 98.58% | 0.98/0.99 | 0.99/0.98 | 0.99 |
| J vs P | 98.68% | 0.99 | 0.98/0.99 | 0.99 |

---

## Key Findings

The stratified baseline achieved only 6.22% accuracy, confirming that meaningful classification requires learned models rather than probability-based random assignment. KNN emerged as the top performer at 98.77% accuracy, marginally outperforming SVM at 98.74%. Both models showed tight cross-validation variance, indicating strong generalizability. Logistic Regression was the weakest advanced model at 91.82%, reflecting the complexity of separating 16 classes in high-dimensional feature space. Random Forest and XGBoost tied at 97.63% and both produced interpretable feature importance rankings. Feature importance analysis across Logistic Regression, Random Forest, and XGBoost consistently highlighted questions Q29, Q12, Q6, Q20, and Q37 as the most influential predictors of personality type.


---

## Connect with Me

**Sai Deepak Poondla**
Data Analyst | Python | Machine Learning | SQL | Power BI | Tableau

[LinkedIn](https://www.linkedin.com/in/your-linkedin-handle) | [Portfolio](https://your-portfolio-link.com)
