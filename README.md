# Credit Card Fraud Detection using Machine Learning

## Background
Fraud detection is a real-world anomaly detection problem. While working in cybersecurity, I observed that identifying suspicious system activity is conceptually similar to identifying fraudulent financial transactions. Both require detecting abnormal patterns within large datasets.

## Objective
The aim of this project is to build a machine learning model capable of classifying transactions as legitimate or fraudulent and to understand the challenges of detecting rare events.

## Dataset
The dataset contains anonymized credit card transactions. Fraud cases represent only a very small portion of the data, making this an imbalanced classification problem.

## Approach
1. Data preprocessing and scaling of transaction amount
2. Train-test split
3. Logistic Regression model training
4. Handling imbalance using class weighting
5. Evaluation using precision, recall and confusion matrix

## Key Observations
- Fraud transactions are extremely rare compared to normal transactions.
- Accuracy alone is misleading for fraud detection.
- Recall is more important because missing fraud has higher cost than false alarms.

## Conclusion
Machine learning can assist real monitoring systems by automatically identifying suspicious patterns. This project strengthened my understanding of anomaly detection and its relation to cybersecurity monitoring systems.
