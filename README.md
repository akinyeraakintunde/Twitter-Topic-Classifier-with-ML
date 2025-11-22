# Twitter Topic Classifier with Machine Learning

This repository contains the code and dataset for an MSc Data Science project by **Ibrahim Akintunde Akinyera**.  
The goal of the project is to build an end-to-end natural language processing (NLP) pipeline that classifies social-media-style text into three thematic categories:

- 🏛️ Politics  
- 🏟️ Sports  
- 🎬 Entertainment  

The project demonstrates practical skills in text preprocessing, feature engineering, model training and evaluation using Python and scikit-learn.

---

## Project Overview

Social media generates vast amounts of unstructured text. Understanding what people are discussing requires automated methods for thematic classification.  
This project implements a complete supervised learning pipeline that:

1. Loads labelled text data for three categories.  
2. Cleans and normalises the raw text.  
3. Converts text into numerical TF-IDF vectors.  
4. Trains multiple ML models (SVM, Logistic Regression, Naive Bayes).  
5. Evaluates performance using accuracy, F1-score and confusion matrix.  
6. Provides a simple prediction interface for unseen text.

---

## Repository Structure

.
├── README.md
├── script_model.py
├── EDA.py
├── requirements.txt
├── docs/
│   └── nlp-pipeline.png
└── data/
├── politics.zip
├── sport.zip
└── entertainment.zip

The dataset is stored as ZIP archives for efficiency.  
Extract each ZIP locally before training, or update dataset paths in `script_model.py`.

---

## Models & Methods

### Text Preprocessing
- Lowercasing  
- Basic cleaning (punctuation, URLs, numerics)  
- Tokenisation  
- Optional stop-word handling  

### Feature Engineering
- TF-IDF vectorisation using scikit-learn  

### Machine Learning Models
- Support Vector Machine (SVM)  
- Logistic Regression  
- Multinomial Naive Bayes  

### Evaluation Metrics
- Train/test split  
- Accuracy  
- Precision / Recall  
- F1-Score  
- Confusion Matrix  

---

## Architecture

Architecture diagram: `docs/nlp-pipeline.png`

Pipeline stages:

1. Ingestion – load text from `data/`  
2. Preprocessing – clean and normalise  
3. Vectorisation – TF-IDF transformation  
4. Model training – SVM / Logistic Regression / Naive Bayes  
5. Evaluation – accuracy, F1-score  
6. Prediction – classify new text  

---

## Running the Project

### 1. Clone the repository

git clone https://github.com/akinyeraakintunde/Twitter-Topic-Classifier-with-ML.git
cd Twitter-Topic-Classifier-with-ML

### 2. Install dependencies

pip install -r requirements.txt

### 3. Prepare the dataset

Extract the ZIP files into:

data/
politics/
sport/
entertainment/

(Or adjust paths inside `script_model.py`.)

### 4. Run the model

python script_model.py

This will train the models and print evaluation metrics.

---

## Prediction Example

print(predict(“The president announced a new policy today.”))

Output:
Politics

---

## Author

**Ibrahim Akintunde Akinyera**  
MSc Data Science – Ulster University  
LinkedIn: https://www.linkedin.com/in/ibrahimakinyera  
GitHub: https://github.com/akinyeraakintunde
