# 🛡️ Multi-Label Toxic Comment Classification for Online Moderation

This project presents a robust machine learning pipeline to classify user-generated comments into six toxicity categories — enabling better automated moderation on online platforms. The final model achieves high performance across multiple labels using a hybrid of engineered features and TF-IDF vectors with a LinearSVC classifier.

---

## 🔍 Problem Statement

The goal is to detect and categorize toxic comments into the following classes:
- toxic
- severe_toxic
- obscene
- threat
- insult
- identity_hate

The challenge lies in handling overlapping and imbalanced labels, which require a multi-label classification approach rather than a standard multi-class model.

---

## 🎯 Objectives

- Build and compare multi-label classification models for online moderation.
- Apply feature engineering to improve prediction quality.
- Evaluate models on F1 score, accuracy, precision, and recall.

---

## 📊 Dataset

- Source: [Toxicity Data from Kaggle](https://www.kaggle.com/datasets/dkhalidashik/toxicity-data/data)
- Data: Wikipedia talk page comments labeled across 6 toxicity types.
- Class imbalance present (some categories < 1%).

---

## 🔬 Feature Engineering

- Comment length
- Capitalization percentage
- Exclamation & question mark counts
- IP address removal
- Min-max normalization

---

## 🧠 Models Used

- Logistic Regression
- Multinomial Naive Bayes
- LinearSVC
- LinearSVC with Naive Bayes Features (NB-SVM-style)

---

## 🚀 Best Performing Model

| Model Type                          | Avg F1 Score |
|------------------------------------|--------------|
| LinearSVC (NB + Engineered Features) | **0.8023**   |

This model combines:
- Word-level TF-IDF (20,000 features)
- Character-level TF-IDF (10,000 features)
- Naive Bayes feature weighting
- Engineered metadata features

---

## 📈 Final Evaluation Metrics (on holdout set)

| Label         | Accuracy | F1 Score | Precision | Recall  |
|---------------|----------|----------|-----------|---------|
| toxic         | 96.22%   | 0.7834   | 0.8691    | 0.7130  |
| severe_toxic  | 99.02%   | 0.3819   | 0.5187    | 0.3021  |
| obscene       | 98.03%   | 0.7983   | 0.8884    | 0.7247  |
| threat        | 99.80%   | 0.4166   | 0.5434    | 0.3378  |
| insult        | 97.25%   | 0.6936   | 0.7950    | 0.6152  |
| identity_hate | 99.23%   | 0.4409   | 0.6643    | 0.3299  |

- ROC AUC > 0.95 for all labels
- Best results in prevalent classes (toxic, obscene)

---

## 📌 Observations

- TF-IDF (word + char) hybrid outperformed single vectorizers.
- NB-weighted features boosted performance.
- Minority labels suffered due to low recall — data augmentation needed.
- Limiting input to 300 words balanced performance and efficiency.

---

## 🧠 Future Work

- Augment low-frequency labels (e.g., threat, identity_hate)
- Introduce user behavior context (posting history, comment frequency)
- Explore class-weighted loss or focal loss
- Test deep learning models (e.g., BERT for multi-label classification)

---

## 👨‍💻 Contributors

**Bibek Koirala**
- EDA visualizations, correlation matrix
- Model fine-tuning, LinearSVC with NB features

**Surjana Joshi**
- Feature engineering
- Logistic Regression and Naive Bayes implementation

**Shared Work**
- Objective definitions, result analysis, model comparisons, final report

---

## 📜 License

This project was developed as part of **CS 591 - Natural Language Processing**  
Southern Illinois University Carbondale, Fall 2024.

