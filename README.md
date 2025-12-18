# 🎵 Music Streaming Churn Prediction

**Authors:** Me & Sila (https://github.com/silabou)  
**Context:** Python for Data Science class @ École Polytechnique

## 🚀 Project Overview

This repository contains a complete Machine Learning pipeline designed to predict user churn for a music streaming service. Using raw user activity logs (listening history, page visits, errors, etc.), we engineered time-series features to classify whether a user is likely to cancel their subscription.

The goal was to build a robust model capable of handling class imbalance and preventing overfitting on the leaderboard metric.

## 🛠 Key Features

* **Advanced Feature Engineering:**
    * **Rolling Window Statistics:** Calculated 14-day and 30-day moving averages (listening time, thumbs up/down, error rates).
    * **Trend Analysis:** Slope and ratio features comparing recent activity (last 7 days) vs. historical averages to capture behavioral shifts.
    * **Gap-Aware Labeling:** Implemented a 2-week "blind" gap between feature computation and target labels to prevent data leakage and simulate real-world prediction scenarios.

* **Dimensionality Reduction:** * Utilized `SelectFromModel` with a base XGBoost estimator to identify and retain only the top 35 predictive features.

* **Ensemble Stacking Architecture:**
    * **Level 1:** Trained a diverse set of base models: **XGBoost, LightGBM, CatBoost, and RandomForest**.
    * **Level 2:** A meta-learner (Logistic Regression) combines the out-of-fold probabilities from the base models to make the final prediction.

* **Automated Tuning:** * Used **Optuna** for Bayesian optimization of hyperparameters for all tree-based models, optimizing specifically for F1-Score.

## 📉 Iterative Modeling Strategy

We approached this problem iteratively to improve performance and stability:

1. **Baseline & Leakage Fix:**
   * Initially identified a "Deathbed" leakage issue where the model was training on the exact moment of cancellation.
   * *Solution:* Implemented a strict time-based split, removing the last 14 days of data for every user during training to force the model to detect early warning signs.

2. **Feature Expansion & Selection:**
   * Engineered over 60+ aggregated features including session intervals and diversity scores.
   * Pruned the feature space using importance-based selection to reduce noise and training time.

3. **Model Diversification (The "Stacking" Shift):**
   * **Problem:** Single XGBoost models were hitting a performance ceiling and struggling with high-variance false positives.
   * **Solution:** We moved to a **Stacking Ensemble**. We trained 4 distinct model architectures (XGB, LGBM, CatBoost, RF) on the same data.
   * **Benefit:** CatBoost handled categorical data natively, while Random Forest provided stability against overfitting. The combination (via a meta-learner) significantly boosted the AUC.

4. **Final Optimization (Optuna):**
   * Replaced manual GridSearch with `Optuna` to efficiently navigate the hyperparameter space for all 3 gradient boosting models.
   * Tuned regularization (L1/L2) and tree depth to prevent overfitting on the validation set.

## 📊 Results

The pipeline generates comprehensive performance visualizations:

1. **Correlation Heatmaps:** To verify feature independence.
2. **ROC & PR Curves:** Evaluating the trade-off between True Positives and False Positives.
3. **Cumulative Gain Curve:** Demonstrating the model's lift over random guessing.
4. **Submission Output:** A probability file (`submission_final_binary.csv`) ready for leaderboard submission, with predictions calibrated to the expected churn rate.

*This project is part of the academic coursework at École Polytechnique.*
