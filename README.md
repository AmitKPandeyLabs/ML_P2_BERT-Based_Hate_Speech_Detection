# BERT-Based Hate Speech Detection & Text Filtering

<p align="left">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue.svg?style=for-the-badge&logo=python&logoColor=white" alt="Python Version">
  <img src="https://img.shields.io/badge/Hugging%20Face-Transformers-yellow.svg?style=for-the-badge&logo=huggingface&logoColor=white" alt="Hugging Face Transformers">
  <img src="https://img.shields.io/badge/PyTorch-1.13%2B-red.svg?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Pandas-2.0%2B-purple.svg?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas">
</p>

This repository contains an end-to-end NLP project on "Hate Speech Detection Using Sentiment Analysis." The project leverages advanced transformer models (BERT, RoBERTa, and BERTweet) to classify toxic online content and includes a real-time text-filtering pipeline.

## 🎯 Project Overview & Business Problem

The exponential growth of social media has been accompanied by a troubling rise in online toxicity, hate speech, and offensive language. This content not only harms users but also creates significant challenges for platform moderation.

This project aims to build an effective and automated system to promote safer, more respectful online communication. The primary goals are:
* **Analyze & Classify Tweets:** Use advanced NLP to analyze tweets and classify them as **Hate Speech**, **Offensive Language**, or **Neither**.
* **Compare Transformer Models:** Evaluate and compare the performance of three state-of-the-art transformer models (BERT, RoBERTa, and BERTweet) specifically for this task.
* **Develop a Content Filter:** Design a real-time detection pipeline that can automatically identify and filter offensive words from detected hate speech.

## 📊 Dataset

The dataset used is a popular collection of approximately 24,000 tweets, each annotated by multiple contributors. The data is classified into three categories:
* **Class 0:** Hate Speech
* **Class 1:** Offensive Language
* **Class 2:** Neither Hate Speech nor Offensive

## ⚙️ Methodology

The project was divided into three main phases: EDA, Preprocessing, and Predictive Modeling.

### 1. Data Preprocessing
A rigorous text preprocessing pipeline was created to clean and normalize the tweet data for the models:
* Converted all text to lowercase.
* Removed URLs, user mentions (@), hashtags (#), and special characters.
* Applied **Tokenization** to split tweets into individual words (tokens).
* Applied **Lemmatization** to reduce words to their root form (e.g., "running" -> "run").
* Removed common **stop-words** (e.g., "the", "a", "is") to reduce noise and focus on meaningful terms.

### 2. Exploratory Data Analysis (EDA)
EDA was performed to uncover insights into the dataset's composition. Key findings include:
* **Class Imbalance:** The data is highly imbalanced. The "Offensive Language" class is dominant, while the "Hate Speech" class is a small minority. This imbalance presents a challenge for model training.
* **Text Length:** Offensive and hateful tweets tend to have different character length distributions compared to neutral tweets.
* **Common Terms:** Word clouds revealed the most frequent words in each category, helping to understand the distinct vocabularies of hate speech vs. neutral language.

<p align="center">
  <img src="assets/eda_class_distribution.png" width="45%" alt="Distribution of Classes">
  <img src="assets/eda_wordclouds.png" width="45%" alt="Word Clouds by Class">
</p>

### 3. Predictive Modeling
Three powerful transformer models were fine-tuned and evaluated for this classification task:
* **BERT (bert-base-uncased):** The foundational transformer model, providing a strong baseline.
* **BERTweet (vinai/bertweet-base):** A model specifically pre-trained on a massive corpus of English tweets, making it highly effective for informal, context-heavy tweet language.
* **RoBERTa (roberta-base):** A robustly optimized version of BERT that often achieves higher accuracy.

## 📈 Results & Key Findings

All models were evaluated on their Precision, Recall, and F1-Score for the minority "Hate Speech" class. The results clearly show **RoBERTa** as the superior model for this task.

<p align="center">
  <img src="assets/model_comparison_chart.png" width="70%" alt="Model Performance Comparison">
</p>

**Key Finding:** RoBERTa achieved the best balance of Precision (93%) and Recall (93%), making it the most reliable model for accurately identifying hate speech while minimizing false positives.

### Winning Model: RoBERTa
The confusion matrix for RoBERTa shows its strong ability to distinguish between classes, especially for the critical "Hate Speech" (Class 0) category.

<p align="center">
  <img src="assets/roberta_confusion_matrix.png" width="60%" alt="RoBERTa Confusion Matrix">
</p>

## 🛡️ Hate Speech Filter (Final Pipeline)

The final deliverable is a real-time detection and filtering pipeline. Using the trained RoBERTa model, the system predicts the class of a new, unseen tweet. If classified as "Hate Speech" (Class 0) or "Offensive" (Class 1), a profanity filter is applied to censor the offending words.

Here are samples from the final output, showing the filter in action:

| Original Text | Predicted Class | Filtered Text |
| :--- | :--- | :--- |
| denial of normal the con be asked to comment on tragedies an emotional retard | 2 | denial of normal the con be asked to comment on tragedies an emotional retard |
| just by being able to tweet this insufferable bullshit proves trump a nazi you vagina | 0 | just by being able to tweet this insufferable b******t proves trump a n*** you v****a |
| that is retarded you too cute to be single that is life | 0 | that is r******d you too cute to be single that is life |
| afro american basho | 2 | afro american basho |
| straight girls go to hamburger mary once and start thinking they can say faggot | 0 | straight girls go to hamburger mary once and start thinking they can say f****t |

## 📌 Conclusion
This project successfully validated the strength of transformer models in classifying complex, informal tweet language. The **RoBERTa model** proved to be the most effective.

The final pipeline provides a practical, AI-driven solution for social media platforms to automatically detect and moderate toxic content, helping to enforce community guidelines and create a safer online environment.

## 🛠️ Tools & Technologies Used
* **Python**
* **Hugging Face Transformers:** For BERT, RoBERTa, and BERTweet models.
* **PyTorch:** As the backend for the transformer models.
* **Pandas:** For data manipulation and loading.
* **NLTK:** For text preprocessing (tokenization, lemmatization, stop-words).
* **WordCloud:** For EDA visualizations.
* **Scikit-learn (sklearn):** For model evaluation metrics (Confusion Matrix, F1-Score).
* **Jupyter Notebook:** For interactive development and analysis.

## 🚀 How to Run This Project
This project is split into two notebooks: one for EDA and one for modeling.

1.  Clone this repository:
    ```sh
    git clone [https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git)
    ```
2.  Navigate to the project directory:
    ```sh
    cd YOUR_REPO_NAME
    ```
3.  Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```
4.  Run the notebooks:
    * For exploratory analysis: `jupyter notebook BUAN6342_GROUP_3_EDA_Hate_Speech_Detection.ipynb`
    * For model training and filtering: `jupyter notebook BUAN6342_GROUP_3_Predictive_Modelling_BERT_Based_Hate_Speech_Detection.ipynb`
5.  **Required Files:** To run this project, you will need the original dataset (not included) and a `profanity-list.txt` file (used by the filtering script).
