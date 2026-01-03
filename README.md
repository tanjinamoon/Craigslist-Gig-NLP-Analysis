# Craigslist Gigs NLP: Clustering and Sentiment Analysis

## Project Overview
This project applies Natural Language Processing (NLP) and machine learning techniques to analyze unstructured gig postings from Craigslist’s Boston Gigs section. The goal is to bring structure and insight to free-text listings by categorizing gig types and identifying emotional tone.

---

## Business Problem
Craigslist gig postings are text-heavy and lack structure, making it difficult for users to:
- Quickly identify relevant gig types
- Assess the emotional tone or risk level of postings
- Efficiently search or filter listings

The platform also lacks automated tools to assist moderation and content oversight.

---

## Data
- 939 Craigslist Boston gig postings
- Fields include post title, description, and location
- Data source: Kaggle (public dataset)

---

## Methodology
### Text Processing
- Lowercasing, stopword removal, lemmatization

### Feature Engineering
- TF-IDF vectorization (1–2 grams, min_df = 3)

### Unsupervised Learning
- K-Means clustering (k = 4)
- Identified gig categories:
  - Delivery & Logistics
  - Events & Promotion
  - Research & Testing
  - Manual Labor

### Sentiment Analysis
- VADER sentiment scoring
- Labeled posts as Positive, Neutral, or Negative

### Supervised Modeling
Trained and evaluated:
- Naive Bayes
- Logistic Regression
- Random Forest
- Support Vector Machine (SVM)
- Decision Tree

---

## Model Performance
- Train/Test Split: 70/30 (stratified)
- Best Model: **Decision Tree**
- Accuracy: **~73%**
- Sentiment distribution:
  - Positive: 62%
  - Neutral: 28%
  - Negative: 10%

---

## Key Insights
- Unsupervised clustering successfully grouped gig posts into meaningful categories
- Sentiment analysis helps identify emotionally charged or potentially risky posts
- Decision Tree provided the best balance of accuracy and interpretability
- The approach improves gig discoverability without altering Craigslist’s open-posting model

---

## Tools & Technologies
- Python
- Pandas, NumPy
- Scikit-learn
- NLTK / VADER
- Matplotlib
- Jupyter Notebook

---

## Repository Structure
- `Craigslist Gigs Model.ipynb` – Full NLP and modeling pipeline
- `Data_Craigslist_Gigs_Boston.csv` – Dataset
- Presentation and report PDFs (summary & communication)

---


