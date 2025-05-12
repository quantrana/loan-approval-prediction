# 🏦 Loan-Approval Prediction – Credit-Risk Insight (KNN & RF **from scratch**)

Accurate, interpretable loan decisions sit at the heart of every lender’s risk engine.
This repository shows an **end-to-end machine-learning pipeline**—from CSV ingestion to a hand-coded Random Forest that reaches **ROC-AUC ≈ 0 · 89**—built for Assignment 2 of *Introduction to AI* @ UTS.

> **Key twist:** *Both* K-Nearest Neighbors **and** Random Forest are implemented **entirely from scratch** in pure Python (Numpy), with no scikit-learn shortcuts.‍

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Dataset](#dataset)
3. [Solution Pipeline](#solution-pipeline)
4. [Why These Design Choices?](#why-these-design-choices)
5. [Experimental Results](#experimental-results)
6. [Future Work](#future-work)

---

## Problem Statement

Predict binary target **`loan_status`** ( `1 = approved`, `0 = declined` ) using 12 applicant, credit-history and loan attributes—balancing default risk against customer inclusivity.

---

## Dataset

* **Source**  Kaggle “PS4E9 │ Loan Approval Prediction”.
* **Size**    32  581 rows × 12 columns.
* **Mix**     8 numerical ▪ 4 categorical.
* **Imbalance**  ≈ 24 % approvals.&#x20;

---

## Solution Pipeline

| Stage                   | Key Steps                                                                                                       |
| ----------------------- | --------------------------------------------------------------------------------------------------------------- |
| **EDA**                 | Descriptive stats, correlation matrix.                                                                          |
| **Missing Values**      | Mode fill (categoricals) ➜ **KNNImputer** (k = 5) for numerics.                                                 |
| **Outliers**            | IQR capping (winsorize at 1.5 × IQR).                                                                           |
| **Scaling**             | `StandardScaler` on all numerics.                                                                               |
| **Encoding**            | One-Hot (drop-first) for 4 categorical cols.                                                                    |
| **Split**               | 80 / 20 stratified train–test.                                                                                  |
| **Re-balancing**        | **SMOTE** ⟶ only on training set.                                                                               |
| **Modelling (Scratch)** | ① **Custom K-NN** (majority vote, Euclidean) ② **Custom Random Forest** (bootstrap, CART splits, majority vote) |
| **Tuning**              | 10-draw **randomised search** over tree depth, estimators, split criteria.                                      |
| **Evaluation**          | Accuracy ▪ weighted Precision/Recall/F1 ▪ Confusion Matrix ▪ ROC-AUC.                                           |

---

## Why These Design Choices?

| Decision                          | Rationale                                                                                        |
| --------------------------------- | ------------------------------------------------------------------------------------------------ |
| **Pure-Python KNN & RF**          | Reinforces algorithmic understanding; avoids hidden scikit-learn heuristics; easy to instrument. |
| **SMOTE on train only**           | Prevents leakage; lifts minority-class F1 by \~4 pp.                                             |
| **Randomised search (10 combos)** | 90 % less compute than full grid with negligible performance loss.                               |
| **StandardScaler**                | Required for distance-based KNN and stable split thresholds.                                     |

---

## Experimental Results

| Model (scratch)           |   Accuracy | Weighted F1 |    ROC-AUC |
| ------------------------- | ---------: | ----------: | ---------: |
| **K-NN (k = 3)**          |     0 · 79 |      0 · 79 |     0 · 84 |
| **Random Forest (base)**  |     0 · 81 |      0 · 81 |     0 · 87 |
| **Random Forest (tuned)** | **0 · 83** |  **0 · 83** | **0 · 89** |

The tuned forest (100 trees, depth 10, `max_features='sqrt'`) cut false approvals by 8 % while boosting recall on true approvals by 6 %.

## Future Work

* **Cost-Sensitive Forest** – integrate class-weighting in node impurities.
* **Explainability** – SHAP values for per-applicant transparency.
* **Deployment** – FastAPI + Docker to serve real-time approvals.
* **MLOps** – add CI tests comparing scratch models vs scikit-learn baselines.

