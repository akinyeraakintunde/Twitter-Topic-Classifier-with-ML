# Twitter Topic Classifier with Machine Learning

This repository contains the code and dataset for an MSc Data Science project
by **Ibrahim Akintunde Akinyera**. The goal of the project is to build an
end-to-end natural language processing (NLP) pipeline that classifies
social-media style text into three thematic categories:

- 🏛️ Politics  
- 🏟️ Sports  
- 🎬 Entertainment  

The project demonstrates practical skills in text preprocessing, feature
engineering, model training and evaluation using Python and scikit-learn.

---

## 🌐 Project Overview

Social media generates vast amounts of unstructured text. Understanding what
people are talking about requires automated methods for thematic
classification.

This project implements a supervised learning pipeline that:

1. Loads labelled text data for three classes.
2. Cleans and normalises the raw text.
3. Converts text into numeric features using TF–IDF.
4. Trains multiple classifiers (SVM, Logistic Regression, Naive Bayes).
5. Evaluates their performance using standard metrics.
6. Exposes a simple prediction function for new text.

---

## 📂 Repository Structure

```text
.
├── README.md              # project overview and usage
├── script_model.py        # main training / evaluation script
├── EDA.py                 # exploratory data analysis (optional)
├── docs/
│   └── nlp-pipeline.png   # high-level architecture diagram
└── data/
    ├── politics.zip       # labelled text for politics class
    ├── sport.zip          # labelled text for sports class
    └── entertainment.zip  # labelled text for entertainment class
