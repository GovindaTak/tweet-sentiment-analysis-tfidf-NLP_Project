# tweet-sentiment-analysis-tfidf-NLP_Project
A sentiment analysis project using TF-IDF vectors and Naïve Bayes classification to analyze and classify tweet sentiments as positive or negative. This project covers text preprocessing, feature extraction, model training, and evaluation to achieve accurate sentiment predictions.

# **Tweet Sentiment Analysis using TF-IDF & Naïve Bayes**  

## **Overview**  
This project applies **Natural Language Processing (NLP)** techniques to classify tweet sentiments as **positive or negative**. Using **TF-IDF (Term Frequency-Inverse Document Frequency)** vectorization and a **Naïve Bayes classifier**, we transform raw tweets into structured data to build an effective sentiment analysis model.  

---

## **Features**  

### **1. Dataset Description**  
The dataset consists of tweets with associated sentiment labels:  
- **0:** Negative Sentiment  
- **4:** Positive Sentiment  

Each tweet contains the following attributes:  
- **Tweet ID**: Unique identifier for each tweet.  
- **Date**: Timestamp when the tweet was posted.  
- **Query**: Search query used (if any).  
- **User**: Twitter username of the person posting the tweet.  
- **Text**: Actual content of the tweet.  

---

### **2. Text Preprocessing**  
- **Lowercasing**: Converts all text to lowercase for uniformity.  
- **Tokenization**: Splits sentences into words.  
- **Removal of Special Characters & Punctuation**: Cleans unnecessary symbols.  
- **Stopword Removal**: Eliminates common words with little meaning.  
- **Lemmatization**: Reduces words to their root forms using **Spacy**.  

---

### **3. Feature Extraction - TF-IDF Vectorization**  
- **Convert text data into numerical vectors using TF-IDF.**  
- **Limits features to the top 5000 most frequent words for efficiency.**  
- **Helps to understand word importance in the dataset.**  

---

### **4. Model Training & Evaluation**  
- **Train-Test Split**:  
  - 80% of data for training.  
  - 20% for testing.  

- **Classifier Used:**  
  - **Naïve Bayes (MultinomialNB)** - A probabilistic model suitable for text classification.  

- **Performance Metrics:**  
  - **Accuracy:** Measures how often the model predicts correctly.  
  - **Precision & Recall:** Evaluates how well the model distinguishes sentiments.  
  - **Confusion Matrix:** Provides a visual representation of true vs. predicted labels.  

✅ **Final Accuracy Achieved:** **80.12%**  

---

## **5. Results & Insights**  
- The **TF-IDF vectorizer** effectively captures word importance for sentiment classification.  
- **Naïve Bayes performed well**, achieving **80.12% accuracy**.  
- **The imbalance in positive & negative tweets impacted recall for some cases.**  
- **Future Enhancements**:
  - Implement **deep learning (LSTM/GRU) models** for improved sentiment classification.  
  - Use **word embeddings (Word2Vec, BERT)** for better contextual understanding.  

---

## **How to Use**  

### **1️⃣ Clone the Repository**  

git clone https://github.com/GovindaTak/tweet-sentiment-analysis-tfidf-NLP_project.git
cd tweet-sentiment-analysis-tfidf

### **2️⃣ Install Dependencies**  

pip install -r requirements.txt


### **3️⃣ Run the Model**  

python train_model.py

### **4️⃣ Evaluate Model Performance**  

python evaluate_model.py


---

## **Technologies Used**  
- **Python**  
- **Pandas, NumPy** (Data Handling)  
- **NLTK, Spacy** (NLP Preprocessing)  
- **Scikit-Learn** (Machine Learning Models)  
- **Matplotlib, Seaborn** (Data Visualization)  

## **Contact**  
For any queries or contributions, reach out at **govindatak19@gmail.com**.  
Check out my other projects: **[Govinda Tak GitHub](https://github.com/GovindaTak)**  
