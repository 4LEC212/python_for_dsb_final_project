🎵 Music Streaming Churn Prediction

Authors: Me & @silabou

Context: Python for Data Science class — École Polytechnique

📖 Project Overview

This repository contains a complete Machine Learning pipeline designed to predict user churn for a music streaming service. Using raw user activity logs (listening history, page visits, errors, etc.), we engineered time-series features to classify whether a user is likely to cancel their subscription.

The goal was to build a robust model capable of handling class imbalance and preventing overfitting on the leaderboard metric.

🚀 Key Features

Automated ETL: Aggregates raw logs into daily user snapshots.

Time-Series Feature Engineering: Generates rolling window statistics (7-day, 30-day), trends, and lifetime accumulations.

Robust Validation: Uses Stratified K-Fold Cross-Validation to ensure model stability.

Dynamic Thresholding: Tunes the classification threshold based on Out-Of-Fold (OOF) predictions to maximize the F1-Score.

🧠 Iterative Modeling Strategy

We approached this problem iteratively to improve performance and stability:

Baseline Model (XGBoost): - Initial feature engineering (daily aggregations).

Trained on a simple 80/20 Train/Validation split.

Result: Good initial AUC, but high variance.

Feature Selection & Cleanup:

Implemented SelectFromModel to aggressively prune noisy features (dropping those below 0.5x mean importance).

Reduced dimensionality while maintaining predictive power.

Hyperparameter Optimization:

Performed RandomizedSearchCV to narrow down the search space.

Refined with GridSearchCV to lock in optimal regularization (L1/L2), learning rate, and tree depth.

Final Approach: 5-Fold Ensemble (Current State):

Problem: We observed overfitting to the validation set when tuning thresholds.

Solution: Switched to 5-Fold Cross-Validation. Instead of relying on one model, we train 5 separate models on different data splits and average their predictions.

Outcome: A highly stable score that generalizes better to unseen test data.


Run the full pipeline:
This script loads the data, generates features, runs the 5-Fold CV, determines the optimal threshold, and generates the submission files.

📊 Results

The model outputs two files:

submission_kfold_avg_prob.csv: Raw probabilities (useful for ROC-AUC analysis).

submission_kfold_avg_binary.csv: Final churn predictions (0 or 1) based on the optimal OOF threshold.

This project is part of the academic coursework at École Polytechnique.