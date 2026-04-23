
# 🎬 IMDB Smart Review Analyzer

A full end-to-end NLP pipeline that analyzes IMDB movie reviews, classifies sentiment (positive/negative), and extracts meaningful insights from text using classical and deep learning models.

---

## 📌 Objective

- Build a system that **analyzes user reviews**
- **Classify sentiment** as positive or negative
- **Extract useful insights** such as important keywords, common patterns, and reasons behind predictions

---

## 📂 Project Structure

```
IMDB_Project/
│
├── data/
│   ├── IMDB Dataset.csv           # Raw dataset
│   └── IMDB_cleaned.csv           # Cleaned & preprocessed dataset
│
├── embeddings/
│   ├── X_tfidf.npz                # TF-IDF matrix
│   ├── X_w2v.npy                  # Word2Vec embeddings
│   ├── X_bert.npy                 # BERT embeddings
│   └── y.npy                      # Labels
│
├── models/
│   ├── tfidf_vectorizer.pkl       # Saved TF-IDF vectorizer
│   ├── keras_tokenizer.pkl        # Saved Keras tokenizer
│   ├── w2v_model.model            # Saved Word2Vec model
│   └── lstm_best.keras            # Best saved LSTM model
│
├── outputs/
│   ├── confusion_matrices.png     # Confusion matrix visualization
│   ├── model_comparison.png       # Model comparison chart
│   ├── lstm_training.png          # LSTM training history
│   ├── insights.png               # Insights visualization
│   └── final_results.csv          # Final evaluation results
│
└── README.md
```

---

## 🗃️ Dataset

- **Source**: [IMDB Movie Reviews Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
- **Size**: 50,000 movie reviews
- **Labels**: Positive / Negative (balanced — 25,000 each)

---

## 🔧 Pipeline Overview

```
Raw Reviews
     ↓
1. Preprocessing
     ↓
2. Feature Extraction (TF-IDF, Word2Vec, BERT)
     ↓
3. Baseline Model (Logistic Regression)
     ↓
4. Advanced Model (LSTM)
     ↓
5. Evaluation
     ↓
6. Extract Insights
```

---

## 1️⃣ Preprocessing

Each review goes through the following steps:

| Step | Description |
|------|-------------|
| NER | Named entities replaced with `[PERSON]` and `[MOVIE_NAME]` using `dslim/bert-base-NER` |
| Placeholder Removal | `[PERSON]` and `[MOVIE_NAME]` removed before cleaning |
| Lowercase | All text converted to lowercase |
| HTML Removal | All HTML tags and entities removed |
| Tokenization | Text split into individual tokens using NLTK |
| Punctuation Removal | Non-alphabetic characters removed |
| Stopword Removal | Common English stopwords removed |
| Lemmatization | Words reduced to their base form |

**Example:**
```
ORIGINAL  : "I loved Tom Hanks in Forrest Gump! <br/> It was amazing."
AFTER NER : "I loved [PERSON] in [MOVIE_NAME]! <br/> It was amazing."
CLEANED   : "loved amazing"
```

---

## 2️⃣ Feature Extraction

### TF-IDF
- Converts reviews into numerical vectors based on word frequency
- `max_features=5000`, `ngram_range=(1,2)`, `min_df=3`, `sublinear_tf=True`
- Output shape: `(50000, 5000)`

### Word2Vec
- Trains word embeddings on the IMDB dataset itself
- `vector_size=100`, `window=5`, `min_count=2`, `epochs=10`
- Reviews represented as average of word vectors
- Output shape: `(50000, 100)`

### BERT Embeddings
- Uses `distilbert-base-uncased` as feature extractor
- CLS token used as sentence-level representation
- Output shape: `(50000, 768)`

---

## 3️⃣ Baseline Model — Logistic Regression

Trained on all 3 feature sets:

| Model | Accuracy |
|-------|----------|
| LR + TF-IDF | **89.62%** |
| LR + BERT Embeddings | 88.07% |
| LR + Word2Vec | 87.14% |

> LR + TF-IDF chosen as the best baseline model.

---

## 4️⃣ Advanced Model — LSTM

Architecture:
```
Embedding(10000, 128)
      ↓
SpatialDropout1D(0.2)
      ↓
Bidirectional LSTM(64, dropout=0.2, recurrent_dropout=0.2)
      ↓
Dense(32, relu)
      ↓
Dropout(0.2)
      ↓
Dense(1, sigmoid)
```

Training setup:
- `optimizer`: Adam (lr=0.001)
- `loss`: Binary Crossentropy
- `batch_size`: 128
- `callbacks`: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
- Best model saved automatically to Google Drive

---

## 5️⃣ Evaluation

### Final Results

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| LR + TF-IDF | 89.62% | 88.8% | 90.6% | 89.7% |
| LSTM | ~88% | 89.0% | 84.9%| 86.9% |



### Confusion Matrix
Shows breakdown of correct and incorrect predictions for both models.
<img width="2100" height="750" alt="confusion_matrices" src="https://github.com/user-attachments/assets/10038ab2-6ece-4d38-9e64-17d11af0246e" />


### Comparison Chart
<img width="1800" height="900" alt="model_comparison" src="https://github.com/user-attachments/assets/fd15ce77-4f61-42e0-9654-a6215f800fab" />


---

## 6️⃣ Extracting Insights

### Sentiment + Reason
```
📝 Review   : "This was the worst film I have ever seen"
🏷️  Sentiment : Negative
💡 Reason    : worst, seen
🔑 Keywords  : worst, seen, film
```

### Important Keywords

| Positive Keywords | Negative Keywords |
|------------------|------------------|
| great, brilliant | worst, terrible |
| wonderful, best  | boring, awful   |
| outstanding, love | waste, bad      |

### Common Patterns
```
🟢 Most positive feedback is related to: great, story, performance
🔴 Most complaints are related to      : worst, boring, terrible
```
<img width="2400" height="1800" alt="insights" src="https://github.com/user-attachments/assets/acbef537-5ae7-4835-a2ab-adac418b96dd" />

### Simple Statistics
```
Total reviews    : 50,000
Positive reviews : 25,000 (50%)
Negative reviews : 25,000 (50%)
```

---

## 🛠️ Requirements

```bash
pip install transformers torch scikit-learn nltk pandas gensim scipy tensorflow
```

```python
import nltk
nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('omw-1.4')
```

---

## 🚀 How to Run

1. Open the notebook in **Google Colab**
2. Mount Google Drive
3. Run cells in order from top to bottom
4. If resuming from a saved session:
```python
# Mount drive and load saved files
from google.colab import drive
drive.mount('/content/drive')

SAVE_PATH = '/content/drive/MyDrive/IMDB_Project/'

# Load embeddings, models, and data from Drive
```

---

## 📊 Key Findings

- **TF-IDF + Logistic Regression** achieves 89.62% accuracy — a surprisingly strong baseline
- **LSTM** with Bidirectional layers achieves ~88% accuracy
- **BERT embeddings** as features with Logistic Regression achieve 88.07%
- The dataset is perfectly balanced (50/50) making accuracy a reliable metric
- Most impactful positive words: `great`, `brilliant`, `wonderful`, `best`
- Most impactful negative words: `worst`, `terrible`, `boring`, `awful`

---

## 📚 References

- [IMDB Dataset — Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
- [dslim/bert-base-NER — HuggingFace](https://huggingface.co/dslim/bert-base-NER)
- [distilbert-base-uncased — HuggingFace](https://huggingface.co/distilbert-base-uncased)
- [GloVe Embeddings — Stanford NLP](https://nlp.stanford.edu/projects/glove/)
- [Gensim Word2Vec](https://radimrehurek.com/gensim/models/word2vec.html)

---

## 👤 Author

Built as part of a university NLP project on sentiment analysis using the IMDB dataset.
