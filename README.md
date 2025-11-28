# Craigslist-Gig-NLP-Analysis
Craigslist Gig Analysis: Structuring Unstructured Work Listings with NLP
This project applies natural language processing (NLP), clustering, and classification techniques to enhance user experience and platform intelligence for the Gigs section of Craigslist, a high-traffic but unstructured space for one-off and freelance jobs.

Project Context
The Craigslist Gigs section lacks categorization and sentiment cues, making it difficult for users to find relevant and trustworthy posts. we developed a solution to:

Group posts into meaningful themes using unsupervised learning
Predict emotional tone using sentiment analysis and classification
Add structure to free-text job listings, enabling smarter browsing and content moderation
Objective
Enable Craigslist to:

Organize gig ads into thematic clusters (e.g., Delivery, Events, Research, Labor)
Detect and label tone (positive, neutral, negative) using sentiment analysis
Flag emotionally charged or suspicious posts for moderators
Improve the discovery experience without changing the open-posting interface
Analytical Approach
Data Source: 939 Boston-based Craigslist gig postings
Preprocessing: Stopword removal, lemmatization, n-gram TF-IDF
Clustering: KMeans (k=4) grouped listings into 4 gig types
Sentiment Labeling: VADER lexicon applied to classify post tone
Modeling: Trained 5 classifiers (Naive Bayes, Logistic Regression, Random Forest, SVM, Decision Tree)
Evaluation: Stratified train-test split (70/30) + accuracy, F1-score
Best Model: Decision Tree (Accuracy: 73%)
Machine Learning Components
Unsupervised Learning: KMeans clustering uncovered natural groupings in the text (Delivery, Events, Research, Manual Labor).
Supervised Learning: Used VADER-labeled sentiment as targets and trained five classifiers to predict tone.
Model Evaluation: Compared performance using Accuracy, Precision, Recall, and F1-Score. The Decision Tree model achieved the highest performance and was selected as the final model.
Impact: These ML models enable smarter search filters for users and automated flagging for moderation, creating both user and business value.
Tools Used
Python (pandas, scikit-learn, nltk)
TF-IDF vectorization
VADER Sentiment Analysis
KMeans Clustering
Supervised classification models
Repository Contents
File	Description
Craigslist Gigs Report.pdf	Project report with business context, analysis, and impact
Craigslist Gigs Presentation.pdf	Final slide deck summarizing insights for stakeholders
Craigslist Gigs Model.ipynb	Python notebook with end-to-end preprocessing, modeling, and evaluation
Data_Craigslist_Gigs_Boston.csv	Cleaned dataset of 900+ Boston Craigslist gig listings
This project combines business analysis, natural language processing, and machine learning to solve a real-world problem at scale. It demonstrates my ability to turn unstructured data into actionable insights.
