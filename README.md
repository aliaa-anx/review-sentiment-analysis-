# 🎬 IMDB Smart Review Analyzer

A full end-to-end NLP pipeline that analyzes IMDB movie reviews,
classifies sentiment (positive/negative), and extracts meaningful
insights using classical and deep learning models.

---

## 📌 Objective
- Build a system that analyzes movie reviews
- Classify sentiment as **positive** or **negative**
- Extract useful insights: keywords, confidence scores,
  attention-based explanations, and model comparisons

---

## 📂 Project Structure

    IMDB_Project/
    │
    ├── data/
    │   ├── IMDB Dataset.csv              # Raw dataset
    │   ├── IMDB_processed_NER.csv        # After NER processing
    │   └── IMDB_cleaned.csv              # Fully cleaned dataset
    │
    ├── outputs/
    │   ├── fig1_lr_insights.png          # LR keyword insights
    │   ├── fig2_confusion_matrices.png   # Confusion matrices
    │   ├── fig3_accuracy_comparison.png  # Accuracy bar chart
    │   ├── fig4_metrics_comparison.png   # Precision/Recall/F1
    │   └── fig5_radar_chart.png          # Radar comparison chart
    │
    └── README.md

---

## 🗃️ Dataset
- **Source:** IMDB Movie Reviews Dataset (Kaggle)
- **Size:** 50,000 movie reviews
- **Labels:** Positive / Negative (perfectly balanced — 25,000 each)

---

## 🔧 Pipeline Overview

    Raw Reviews
         ↓
    1. NER (Named Entity Recognition)
         ↓
    2. Preprocessing (Clean, Tokenize, Lemmatize)
         ↓
    3. Feature Extraction (TF-IDF + BERT Tokenization)
         ↓
    4. Baseline Model (Logistic Regression + TF-IDF)
         ↓
    5. Advanced Model (Fine-Tuned BERT)
         ↓
    6. Evaluation + Visualization
         ↓
    7. Insights + Review Analysis

---

## 1️⃣ Preprocessing

| Step | Description |
|------|-------------|
| NER | Named entities replaced using `dslim/bert-base-NER` |
| Placeholder Removal | `[PERSON]` and `[MOVIE_NAME]` removed |
| Lowercase | All text converted to lowercase |
| HTML Removal | All HTML tags and entities stripped |
| Tokenization | Text split into tokens using NLTK |
| Punctuation Removal | Non-alphabetic characters removed |
| Stopword Removal | Common English stopwords removed |
| Lemmatization | Words reduced to base form using WordNetLemmatizer |

**Example:**

    ORIGINAL  : "I loved Tom Hanks in Forrest Gump! <br/> It was amazing."
    AFTER NER : "I loved [PERSON] in [MOVIE_NAME]! <br/> It was amazing."
    CLEANED   : "loved amazing"

---

## 2️⃣ Feature Extraction

### TF-IDF (for Baseline)
- Converts reviews into numerical vectors based on word frequency
- `max_features=5000`, `ngram_range=(1,2)`, `min_df=3`, `sublinear_tf=True`
- Output shape: `(50000, 5000)`

### BERT Tokenization (for Advanced Model)
- Uses `bert-base-uncased` tokenizer
- `max_length=256`, `padding=True`, `truncation=True`
- Input fed directly into fine-tuned BERT for classification

---

## 3️⃣ Baseline Model — Logistic Regression

- Trained on TF-IDF features
- `max_iter=1000`, `C=1.0`
- Fast, interpretable, strong baseline

**Accuracy: 89.62%**

---

## 4️⃣ Advanced Model — Fine-Tuned BERT

**Architecture:**

    bert-base-uncased (12 transformer layers)
             ↓
      Pooled CLS token output
             ↓
      Linear Classifier (hidden_size → 2)
             ↓
      Softmax → [Negative, Positive]

**Training Setup:**

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning Rate | 2e-5 |
| Epochs | 3 |
| Batch Size | 16 |
| Max Sequence Length | 256 |
| Device | GPU (CUDA) |

---

## 5️⃣ Evaluation

### Final Results

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| LR + TF-IDF (Baseline) | 89.62% | 88.8% | 90.6% | 89.7% |
| Fine-Tuned BERT (Advanced) | ~93%+ | ~93%+ | ~93%+ | ~93%+ |

### Visualizations Produced

| Figure | Description |
|--------|-------------|
| `fig1_lr_insights.png` | Pie chart + top keywords + pattern comparison |
| `fig2_confusion_matrices.png` | Side-by-side confusion matrices |
| `fig3_accuracy_comparison.png` | Accuracy bar chart |
| `fig4_metrics_comparison.png` | Precision / Recall / F1 grouped bars |
| `fig5_radar_chart.png` | Radar chart across all metrics |

---

## 6️⃣ Review Analysis and Insights

Both models analyze any review and output:

    Review     : "The movie was absolutely brilliant and inspiring"
    [AGREE]
                 Logistic Regression        Fine-Tuned BERT
                 -------------------------  -------------------------
    Sentiment  :        Positive                   Positive
    Confidence :         88.5%                      99.9%
    Neg Prob   :          4.4%                       0.1%
    Pos Prob   :         95.6%                      99.9%
    Keywords   : subtle, perfectly, masterpiece   without, subtle, career

### Long Review Test Results

| Review Type | LR Confidence | BERT Confidence |
|-------------|---------------|-----------------|
| Long Positive (149 words) | 95.6% | 99.9% |
| Long Negative (168 words) | 99.9% | 100.0% |

Both models agreed on all test reviews. BERT consistently showed
higher confidence due to its contextual understanding.

---

## 📊 Key Findings

- **Fine-Tuned BERT** outperforms Logistic Regression on confidence
  and contextual understanding
- **LR + TF-IDF** is a surprisingly strong baseline at **89.62%**
  with much faster training time
- BERT correctly understands **negation context**
  (e.g. "cannot believe", "cannot find") which LR misses entirely
- Most impactful **positive** words: `brilliant`, `masterpiece`,
  `outstanding`, `inspiring`
- Most impactful **negative** words: `worst`, `avoid`,
  `disappointing`, `waste`
- Dataset is perfectly balanced (50/50) making accuracy a reliable metric

---

## 🛠️ Requirements

    pip install transformers torch scikit-learn nltk pandas seaborn matplotlib

    import nltk
    nltk.download('stopwords')
    nltk.download('punkt_tab')
    nltk.download('wordnet')

---

## 🚀 How to Run

1. Open the notebook in **Google Colab**
2. Mount Google Drive
3. Upload `IMDB Dataset.csv.zip` to Drive
4. Run all cells **top to bottom**

Mount Drive:

    from google.colab import drive
    drive.mount('/content/drive')

---

## 📚 References
- [IMDB Dataset — Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
- [dslim/bert-base-NER — HuggingFace](https://huggingface.co/dslim/bert-base-NER)
- [bert-base-uncased — HuggingFace](https://huggingface.co/bert-base-uncased)
- [NLTK Library](https://www.nltk.org/)
- [Scikit-Learn](https://scikit-learn.org/)

---

## 👤 Author
Built as part of a university NLP course project —
Capital University (formerly Helwan University),
Faculty of Computing & Artificial Intelligence,
Spring Semester 2025-2026.
