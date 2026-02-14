# 🎫 AI-Powered Support Ticket Classifier

An intelligent support ticket classification system that automatically categorizes customer support requests using Natural Language Processing and Machine Learning.

## 🌐 Live Demo

**Try it now:** [Support Ticket Classifier App]()

## 🎯 Problem Statement

### The Business Challenge

Support teams receive thousands of tickets daily across multiple categories (Hardware, HR, Access, etc.). Manual categorization:
- ❌ Wastes valuable agent time
- ❌ Causes routing delays
- ❌ Leads to inconsistent categorization
- ❌ Slows down response times

### The Solution

An AI-powered classifier that:
- ✅ **Instantly categorizes tickets** with 85.38% accuracy
- ✅ **Routes to correct department** automatically
- ✅ **Reduces manual work** by 85%+
- ✅ **Improves response times** significantly

  ## 📋 Project Scope

### What This Project Delivers
✅ **Ticket Category Classification** - 8 categories with 85.38% accuracy
✅ **Text Preprocessing Pipeline** - Cleaning, stopword removal, TF-IDF
✅ **Model Evaluation** - Precision, Recall, F1-Score, Confusion Matrix
✅ **Web Deployment** - Live Streamlit application
✅ **Class-wise Performance Analysis** - Detailed breakdown per category

### Note on Priority Prediction
⚠️ The dataset used in this project did not include priority labels (High/Medium/Low/Critical). 

The project focuses on **ticket category classification**, which is the primary business value:
- Automatically routing tickets to the correct department
- 85.38% accuracy across 8 support categories
- Real-time classification via web interface

**Future Enhancement:** Priority prediction can be added when labeled priority data becomes available. The same NLP pipeline and classification approach would apply.

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| **Overall Accuracy** | **85.38%** |
| Weighted Precision | 86% |
| Weighted Recall | 85% |
| Weighted F1-Score | 85% |

### Category-Specific Performance

| Category | Precision | Recall | F1-Score |
|----------|-----------|--------|----------|
| Purchase | 97.5% | 87.4% | **92.2%** ⭐ |
| Access | 91.0% | 87.9% | 89.4% |
| Storage | 95.0% | 82.5% | 88.3% |
| HR Support | 86.4% | 86.5% | 86.5% |
| Internal Project | 90.8% | 81.4% | 85.8% |
| Hardware | 78.9% | 89.1% | 83.7% |
| Miscellaneous | 83.4% | 80.9% | 82.1% |
| Administrative Rights | 88.2% | 63.6% | 73.9% |

## 🚀 Features

### Web Application
- 📝 **Text Input:** Type custom support tickets
- 📋 **Example Templates:** Pre-loaded examples for testing
- 🎯 **Top-3 Predictions:** Shows confidence for multiple categories
- 📊 **Confidence Scores:** Visual progress bars for transparency
- 📱 **Responsive Design:** Works on desktop and mobile

### Model Capabilities
- **8 Categories:** Hardware, HR Support, Access, Storage, Purchase, Internal Project, Administrative Rights, Miscellaneous
- **Real-time Classification:** Instant predictions
- **Multi-class Probability:** Shows likelihood for all categories

## 🛠️ Tech Stack

### Machine Learning
- **scikit-learn** - Model training and evaluation
- **TF-IDF Vectorization** - Text feature extraction
- **Logistic Regression** - Classification algorithm (best performer)

### Natural Language Processing
- **Text Preprocessing:** Lowercasing, punctuation removal, whitespace normalization
- **Stop Word Removal:** English stop words filtered
- **N-grams:** Unigrams and bigrams (1-2 word phrases)
- **Vocabulary:** 5,000 most important features

### Deployment
- **Streamlit** - Web application framework
- **Python 3.x** - Core programming language
- **joblib** - Model serialization
- **Streamlit Cloud** - Hosting platform

### Development Tools
- **pandas** - Data manipulation
- **NumPy** - Numerical computing
- **Matplotlib & Seaborn** - Data visualization
- **Jupyter/Colab** - Development environment

## 📁 Project Structure
```
support-ticket-classifier-app/
├── app.py                          # Streamlit web application
├── requirements.txt                # Python dependencies
├── ticket_classifier_model.pkl     # Trained ML model
├── tfidf_vectorizer.pkl           # Text vectorizer
├── categories.pkl                  # Category labels
└── README.md                       # This file
```



## 📊 Training Data

- **Dataset Size:** 47,837 support tickets
- **Training Split:** 80% (38,269 tickets)
- **Testing Split:** 20% (9,568 tickets)
- **Stratified Sampling:** Maintains category distribution

### Data Preprocessing Pipeline

1. **Text Cleaning**
   - Convert to lowercase
   - Remove punctuation
   - Remove extra whitespace

2. **Feature Extraction**
   - TF-IDF vectorization
   - Max features: 5,000
   - Min document frequency: 5
   - Max document frequency: 80%
   - N-gram range: (1, 2)

3. **Model Training**
   - Algorithm comparison: Naive Bayes, Logistic Regression, Random Forest
   - Best: Logistic Regression (85.38% accuracy)
   - Hyperparameters: max_iter=1000, random_state=42

## 💡 How It Works

### Classification Pipeline
```
User Input
    ↓
Text Cleaning (lowercase, remove punctuation)
    ↓
TF-IDF Vectorization (convert to numerical features)
    ↓
Logistic Regression Model (trained on 38k tickets)
    ↓
Category Prediction + Confidence Scores
    ↓
Display Top 3 Categories with Probabilities
```

### Example Usage

**Input:**
```
"I need access to the HR system to view my payroll information"
```

**Output:**
- **Primary Category:** HR Support (92.3% confidence)
- **Alternative 1:** Access (6.1% confidence)
- **Alternative 2:** Administrative Rights (1.2% confidence)

## 🎓 Key Learnings

### Technical Insights
1. **TF-IDF outperformed simple Bag of Words** for this dataset
2. **Logistic Regression beat Random Forest** (85.4% vs 84.0%)
3. **Bigrams improved accuracy** by ~3% over unigrams alone
4. **Stop word removal was crucial** for reducing noise

### Business Insights
1. **Purchase requests easiest to classify** (97.5% precision)
2. **Administrative Rights most challenging** (limited training data - 352 samples)
3. **Hardware tickets often confused** with other technical categories
4. **85% automation rate** achievable with current model



*Built as part of my machine learning portfolio - demonstrating NLP, classification, and deployment skills.*
