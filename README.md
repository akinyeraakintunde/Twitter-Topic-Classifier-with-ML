# Twitter Topic Classifier with Machine Learning

A machine learning approach to **thematic classification of Twitter data**.  
This project builds an end-to-end NLP pipeline that classifies social-media style text into three themes:

- 🟣 **Entertainment**
- 🟡 **Politics**
- 🟢 **Sports**

It was developed as part of my **MSc Data Science** dissertation and demonstrates practical skills in:

- Data extraction from Twitter (API + Tweepy)
- Text cleaning and NLP preprocessing
- Feature engineering (TF–IDF and contextual embeddings)
- Supervised model training and evaluation
- Visualisation of results (word clouds, confusion matrices, classification reports)

> 📌 In my MSc dissertation, I further extend this work with a **transformer-based architecture using BERT embeddings** and GPU training, achieving a **macro F1 score of ~0.96** on the test set. The code in this repository focuses on the classical machine-learning pipeline and is structured for clarity and reproducibility.
 
---

## 1. Project Overview

Social media generates huge volumes of noisy, unstructured text.  
Manually sorting tweets into themes (e.g. deciding which tweets are about politics, sports or entertainment) is:

- Time-consuming  
- Error-prone  
- Difficult to scale

This repository implements a **supervised NLP pipeline** that:

1. Loads labelled text data for three thematic classes
2. Cleans and normalises raw text (mentions, URLs, emojis, punctuation)
3. Converts text into numeric features (TF–IDF)
4. Trains multiple classifiers (e.g. SVM, Logistic Regression, Naïve Bayes)
5. Evaluates them using accuracy, precision, recall, F1-score
6. Exposes a simple prediction function for new, unseen text

This project can be adapted for:

- Social listening and trend analysis  
- Targeted content marketing  
- Topic routing and content recommendation  
- Early-warning signals for public opinion or events

---

## 2. Repository Structure

```bash
.
├── README.md                  # Project overview and documentation
├── requirements.txt           # Python dependencies
├── script_model.py            # Main training / evaluation script
├── EDA.py                     # Exploratory data analysis (optional)
├── data/                      # Input datasets (compressed)
│   ├── politics.zip           # Politics class
│   ├── sport.zip              # Sports class
│   └── entertainment.zip      # Entertainment class
├── docs/
│   ├── nlp-pipeline.png       # High-level pipeline diagram
│   └── figures/               # (Add these files)
│       ├── model_architecture.png
│       ├── training_curve.png
│       ├── classification_report.png
│       └── confusion_matrix.png
└── extra_files+code.zip       # Additional notebooks / scripts (archived)