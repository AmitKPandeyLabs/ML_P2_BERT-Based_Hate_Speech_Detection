# BERT-Based Hate Speech Detection & Text Filtering

<p align="left">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue.svg?style=for-the-badge&logo=python&logoColor=white" alt="Python Version">
  <img src="https://img.shields.io/badge/Hugging%20Face-Transformers-yellow.svg?style=for-the-badge&logo=huggingface&logoColor=white" alt="Hugging Face Transformers">
  <img src="https://img.shields.io/badge/PyTorch-1.13%2B-red.svg?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Pandas-2.0%2B-purple.svg?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas">
</p>

This repository contains an end-to-end NLP project on "Hate Speech Detection Using Sentiment Analysis." The project leverages advanced transformer models (BERT, RoBERTa, and BERTweet) to classify toxic online content and concludes with a real-time text-filtering pipeline.

## 🎯 Project Overview & Business Problem

The exponential growth of social media has been accompanied by a troubling rise in online toxicity, hate speech, and offensive language. This content not only harms users but also creates significant challenges for platform moderation, leading to unsafe online environments and reputational damage for platforms.

This project aims to build an effective and automated system to promote safer, more respectful online communication. The primary goals are:
* **Analyze & Classify Tweets:** Develop and apply advanced NLP techniques to analyze social media text (tweets) and accurately classify them as **Hate Speech**, **Offensive Language**, or **Neither**.
* **Compare Transformer Models:** Rigorously evaluate and compare the performance of three state-of-the-art transformer models (BERT, RoBERTa, and BERTweet), specifically assessing their efficacy in detecting subtle nuances of toxic language in a social media context.
* **Develop a Content Filter:** Design and implement a real-time content moderation pipeline that can automatically identify and filter offensive words from tweets predicted to contain hate speech or offensive language, thereby enabling proactive content management.

## 📊 Dataset

The dataset used is a popular collection of approximately 24,000 tweets, each annotated by multiple contributors. The data is classified into three categories:
* **Class 0:** Hate Speech (around 6% of the dataset)
* **Class 1:** Offensive Language (the majority class, representing about 78% of the dataset)
* **Class 2:** Neither Hate Speech nor Offensive (approximately 16% of the dataset)

This dataset presents a significant challenge due to its inherent class imbalance, particularly the small proportion of actual hate speech, requiring careful consideration during model training and evaluation.

## ⚙️ Methodology

The project was divided into three main phases: Preprocessing, Exploratory Data Analysis (EDA), and Predictive Modeling.

### 1. Preprocessing
A rigorous text preprocessing pipeline was created to clean and normalize the raw tweet data for the models. This process is essential for reducing noise, standardizing text, and improving model accuracy and generalization capabilities.

<p align="center">
  <img src="assets/preprocessing_pipeline_new.png" width="80%" alt="Preprocessing Pipeline">
</p>

The key steps included:
* **Text Cleaning (Lowercasing & Regex):** Converted all text to lowercase to ensure consistency and prevent the model from treating "Hate" and "hate" as different words. Regular expressions were used to remove URLs, user mentions (@symbols), hashtags (#symbols), special characters, and extra spaces, which often act as noise rather than meaningful features in this context.
* **Tokenization:** Split tweets into individual words or subword units (tokens). This is a crucial step for transformer models, which operate on sequences of tokens.
* **Lemmatization & Stop-word Removal:** Lemmatization reduced words to their root form (e.g., "running" -> "run," "ran" -> "run") to consolidate their meaning and reduce the vocabulary size. Stop-word removal eliminated common but non-meaningful words (e.g., "the", "a", "is") to help the model focus on important, content-bearing terms.
* **Clean Text / Token ready for model input:** The final output from this pipeline is a clean, normalized, and tokenized sequence of text, perfectly formatted for direct feeding into the transformer models.

### 2. Exploratory Data Analysis (EDA)
EDA was performed to uncover insights into the dataset's composition, identify potential challenges, and guide modeling strategies.

**Class Imbalance:**
The distribution of classes is highly imbalanced, which is a critical characteristic of this dataset. The "Offensive Language" (Class 1) is the dominant class, while "Hate Speech" (Class 0) is a small minority (only ~6% of the data). This imbalance is a critical challenge, as models can become biased towards the majority class, leading to poor performance on the under-represented but crucial hate speech examples.

<p align="center">
  <img src="assets/eda_class_distribution.png" width="80%" alt="Distribution of Classes">
</p>

**Text Length Analysis:**
Offensive and hateful tweets tend to have different character length distributions compared to neutral tweets. The boxplot shows that neutral tweets (Class 2) have a slightly more compact length, while offensive and hateful tweets show a wider range of lengths and more outliers, indicating varied expression patterns in toxic content.

<p align="center">
  <img src="assets/eda_char_plots.png" width="80%" alt="Boxplot of Character distribution by class">
</p>

**Exploratory Data Analysis – Correlation & Density:**
Further analysis of character length, word count, and unique word count revealed interesting correlations within each class. The density plots visually represent the distribution of these features, reinforcing that offensive and hateful tweets often have distinct textual properties.

<p align="center">
  <img src="assets/eda_corr_density.png" width="80%" alt="Correlation and Density Plots of Text Features by Class">
</p>

**Key Term Analysis (Word Clouds):**
Word clouds were generated to visualize the most frequent and unique words in each category. This helps to understand the distinct vocabularies used. The word "bitch" is notably prominent in both "Hate Speech" and "Offensive Language," while "trash" and "pussy" are also common in these categories. In contrast, the "Neither" category features more standard, innocuous terms like "people" and "like," highlighting the clear lexical differences between classes.

<p align="center">
  <img src="assets/eda_wordcloud_hate.png" width="25%" alt="Hate Speech Word Cloud">
  <img src="assets/eda_wordcloud_offensive.png" width="25%" alt="Offensive Language Word Cloud">
  <img src="assets/eda_wordcloud_neither.png" width="50%" alt="Neither Word Cloud">
</p>

### 3. Predictive Modeling
Three powerful, pre-trained transformer models were fine-tuned and evaluated for this classification task. Each model was chosen for its specific strengths in natural language understanding, especially within varied text domains.
* **BERT (bert-base-uncased):** The foundational Bidirectional Encoder Representations from Transformers model. Pre-trained on a large corpus of text (BooksCorpus and English Wikipedia), BERT provides a strong general-purpose contextual understanding of language, serving as a robust baseline for comparison.
* **BERTweet (vinai/bertweet-base):** A BERT-based model specifically pre-trained on a massive corpus of English tweets (850M tweets). Its domain-specific pre-training makes it exceptionally effective for informal, context-heavy social media language, allowing it to capture tweet-specific nuances like slang and abbreviations.
* **RoBERTa (roberta-base):** A Robustly Optimized BERT Pretraining Approach. RoBERTa builds upon BERT's architecture but was trained with a modified strategy (e.g., dynamic masking, larger mini-batches, removal of the next-sentence prediction objective) on an even larger dataset. It often achieves higher accuracy and provides a stronger state-of-the-art performance for various NLP tasks.

## 📈 Results & Key Findings

All models were rigorously evaluated on their Precision, Recall, and F1-Score, with a particular focus on the minority "Hate Speech" (Class 0) category due to its critical importance. The results consistently show **RoBERTa** as the superior model for this task.

<p align="center">
  <img src="assets/model_comparison_chart.png" width="70%" alt="Model Performance Comparison">
</p>

**Key Finding:** RoBERTa achieved the best overall balance of Precision (93%), Recall (93%), and F1-Score (93%), making it the most reliable model for accurately identifying hate speech while effectively minimizing false positives across all classes.

### Model Performance Comparison (Confusion Matrices)
The confusion matrices visually demonstrate the performance of each model, providing a granular view of true positives, true negatives, false positives, and false negatives for each class. The ideal matrix would have high numbers on the diagonal (correct predictions) and low numbers off the diagonal (errors).

<h3 align="center">BERT | BERTTweet | RoBERTa</h3>
<p align="center">
  <img src="assets/bert_confusion_matrix.png" width="30%" alt="BERT Confusion Matrix">
  <img src="assets/bertweet_confusion_matrix.png" width="30%" alt="BERTweet Confusion Matrix">
  <img src="assets/roberta_confusion_matrix.png" width="30%" alt="RoBERTa Confusion Matrix">
</p>

**Interpretation of Confusion Matrices:**
* **BERT:** While performing reasonably well, BERT shows some notable misclassifications, particularly confusing 'Offensive' language with 'Neither' (e.g., 159 'Offensive' predicted as 'Neither'). It also has a relatively higher number of false negatives for 'Hate' speech compared to RoBERTa.
* **BERTweet:** Demonstrates strong performance, especially on 'Offensive' language, likely due to its domain-specific pre-training. It has slightly fewer misclassifications between 'Offensive' and 'Neither' compared to BERT, indicating a better grasp of social media context.
* **RoBERTa:** RoBERTa's matrix clearly shows the strongest performance. It exhibits the highest numbers on the diagonal and the lowest off-diagonal values, especially for 'Hate Speech' (Class 0) and 'Neither' (Class 2). It effectively minimizes false positives for 'Hate' speech (crucial for avoiding over-censorship) and demonstrates robust differentiation between all three classes.

**Summary of Key Performance Metrics (Weighted Averages):**

| Model     | Accuracy | Precision | Recall | F1-Score |
| :-------- | :------- | :-------- | :----- | :------- |
| BERT      | 0.90     | 0.90      | 0.90   | **0.90** |
| BERTTweet | 0.91     | 0.91      | 0.91   | **0.91** |
| RoBERTa   | 0.93     | 0.93      | 0.93   | **0.93** |

**Overall Observations:**
* **Domain-Specific Pre-training Matters:** BERTTweet's slight edge over vanilla BERT highlights the benefits of pre-training on domain-specific data (tweets) when dealing with social media text.
* **Robust Optimization:** RoBERTa's superior performance across all metrics underscores the effectiveness of its robust pre-training approach in achieving a more generalized and accurate understanding of complex language patterns.
* **Challenges in Nuance:** All models still show some difficulty in perfectly distinguishing between 'Hate Speech' and 'Offensive Language', reflecting the inherent ambiguity and overlap in these categories, which often requires subtle contextual understanding.

## 🛡️ Hate Speech Filter (Final Pipeline)

The final deliverable is a real-time detection and filtering pipeline. Using the best-performing trained **RoBERTa model**, the system predicts the class of a new, unseen tweet. If classified as "Hate Speech" (Class 0) or "Offensive" (Class 1), a sophisticated profanity filter is applied to censor the offending words, thus mitigating the spread of harmful content.

Here are samples from the final output CSV, showing the filter in action:

<p align="center">
  <img src="assets/filter_results_table.png" width="80%" alt="Hate Speech Filter Results">
</p>

| Original Text | Predicted Class | Filtered Text |
| :--- | :--- | :--- |
| denial of normal the con be asked to comment on tragedies an emotional retard | 2 | denial of normal the con be asked to comment on tragedies an emotional retard |
| just by being able to tweet this insufferable bullshit proves trump a nazi you vagina | 0 | just by being able to tweet this insufferable b******t proves trump a n*** you v****a |
| that is retarded you too cute to be single that is life | 0 | that is r******d you too cute to be single that is life |
| straight girls go to hamburger mary once and start thinking they can say faggot | 0 | straight girls go to hamburger mary once and start thinking they can say f****t |

## 📌 Conclusion
This project successfully validated the strength and adaptability of transformer models in classifying complex, informal tweet language. The **RoBERTa model** consistently proved to be the most effective, achieving an F1-Score of 0.93 for the critical "Hate Speech" class and demonstrating robust performance across all categories.

The final pipeline provides a practical, AI-driven solution for social media platforms to automatically detect and moderate toxic content, helping to enforce community guidelines, protect users, and create a safer, more respectful online environment.

## 🚀 Future Enhancements
To further improve this project and its practical applicability, several key areas for future work are identified:
* **Explore Advanced Models:** Investigate more recent or specialized transformer architectures (e.g., XLNet, ELECTRA, or larger versions of RoBERTa) to potentially achieve even higher accuracy and capture more nuanced contexts.
* **Bias Detection & Mitigation:** Conduct a thorough analysis of potential biases within the dataset and model predictions. Implement techniques (e.g., re-sampling, adversarial debiasing) to ensure fair and equitable detection across different demographic groups.
* **Ensemble Modeling:** Experiment with combining predictions from multiple models to leverage their individual strengths and potentially achieve a more robust and accurate final classification.
* **Contextual Filtering:** Enhance the content filtering mechanism to be more context-aware, potentially using BERT embeddings to determine if a "profane" word is used in an actually offensive context versus an innocuous one.
* **Explainability (XAI):** Integrate Explainable AI techniques (e.g., LIME, SHAP) to understand *why* the model makes certain predictions, increasing trust and providing insights for further model improvement.
* **Real-time API Deployment:** Develop a lightweight API for the trained model to allow for scalable, real-time hate speech detection and integration into live social media feeds.
* **Multilingual Support:** Extend the project to support hate speech detection in multiple languages using multilingual transformer models.

## 🛠️ Tools & Technologies Used
* **Python:** The core programming language for the project (version 3.9+).
* **Hugging Face Transformers:** A powerful library for state-of-the-art transformer models (BERT, RoBERTa, BERTweet).
* **PyTorch:** The deep learning framework used as the backend for the transformer models (version 1.13+).
* **Pandas:** Essential for efficient data manipulation, loading, and analysis (version 2.0+).
* **NLTK (Natural Language Toolkit):** Utilized for various text preprocessing steps, including tokenization, lemmatization, and stop-word removal.
* **WordCloud:** For generating insightful visualizations of term frequency in different text categories during EDA.
* **Scikit-learn (sklearn):** Employed for comprehensive model evaluation metrics, including Confusion Matrices, F1-Score, Precision, and Recall.
* **Jupyter Notebook:** Used extensively for interactive development, experimentation, and presenting analysis results.

## ▶️ How to Run This Project
This project is split into two notebooks: one for EDA and one for modeling.

1.  Clone this repository:
    ```sh
    git clone [https://github.com/AmitKPandeyLabs/ML_P2_BERT-Based_Hate_Speech_Detection.git](https://github.com/AmitKPandeyLabs/ML_P2_BERT-Based_Hate_Speech_Detection.git)
    ```
2.  Navigate to the project directory:
    ```sh
    cd ML_P2_BERT-Based_Hate_Speech_Detection
    ```
3.  Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```
4.  Run the notebooks:
    * For exploratory analysis: `jupyter notebook EDA_Hate_Speech_Detection.ipynb`
    * For model training and filtering: `jupyter notebook Predictive_Modelling_BERT_Based_Hate_Speech_Detection.ipynb`
5.  **Required Files:** To run this project, you will need the original dataset (included - labeled_data.csv) and a `profanity-list.txt` file (This file is used by the filtering script and add this if you would like the filtering pipeline to work. You can have your own list of custom filters as desired). Please ensure these are placed in the appropriate project directories as referenced in the notebooks.
