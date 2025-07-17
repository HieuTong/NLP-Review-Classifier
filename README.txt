# Assignment 3: Milestone I Natural Language Processing
## Advanced Programming for Data Science

### Student Information
- **Student Name:** Tong Minh Hieu Le
- **Student ID:** s4098368

### Assignment Overview
This assignment focuses on Natural Language Processing (NLP) techniques for clothing review analysis, including feature representation generation and classification models.

### Environment & Dependencies
- **Python Version:** Python 3
- **Development Environment:** Jupyter Notebook
- **Required Libraries:**
  - pandas
  - numpy
  - scikit-learn (CountVectorizer, TfidfVectorizer, LogisticRegression, train_test_split, etc.)
  - gensim (KeyedVectors)
  - matplotlib
  - seaborn
  - scipy

### Project Structure
```
├── README.txt                  # This file
├── assignment3.csv            # Original dataset
├── processed.csv              # Preprocessed dataset
├── stopwords_en.txt           # English stopwords list
├── vocab.txt                  # Vocabulary file
├── count_vectors.txt          # Generated count vector representations
├── task1.ipynb               # Task 1 notebook
├── task1.py                  # Task 1 Python script
├── task2_3.ipynb             # Task 2 & 3 notebook (main analysis)
├── task2_3.py                # Task 2 & 3 Python script
└── glove/
    └── glove.6B.50d.txt      # GloVe embeddings (50-dimensional)
```

### Task Breakdown

#### Task 1: Data Preprocessing
- Text cleaning and normalization
- Stopword removal
- Tokenization
- Vocabulary generation

#### Task 2: Feature Representation Generation
**2.1 Bag-of-Words Model**
- Count vector generation using CountVectorizer
- Vocabulary-based feature extraction
- Output: count_vectors.txt

**2.2 Word Embeddings (GloVe)**
- Unweighted GloVe embeddings
- TF-IDF weighted GloVe embeddings
- 50-dimensional vector representations

#### Task 3: Classification Analysis
**3.1 Language Model Comparison**
- Logistic Regression classification using:
  - Count vectors (Bag-of-Words)
  - Unweighted GloVe embeddings
  - TF-IDF weighted GloVe embeddings
- 5-fold cross-validation evaluation

**3.2 Information Source Analysis**
- Title-only classification
- Review-only classification
- Combined (title + review) classification
- Performance comparison across different information sources

### Key Findings

#### Model Performance (F1-Score Rankings):
1. **Count Vectors (Bag-of-Words)**: 93.51% (Combined), 92.25% (Review), 92.22% (Title)
2. **Unweighted GloVe**: 91.14% (Combined), 90.13% (Review), 91.80% (Title)
3. **Weighted GloVe**: 90.80% (Combined), 90.09% (Review), 91.88% (Title)

#### Key Insights:
- Bag-of-words consistently outperforms embedding-based methods
- Title information alone provides strong predictive power
- Combining title and review text yields the best performance
- TF-IDF weighting provides minimal improvement over unweighted embeddings
- Class imbalance affects model performance (dataset is skewed toward positive recommendations)

### Dataset Characteristics
- **Total Reviews:** ~23,000 clothing reviews
- **Class Distribution:** Imbalanced (more positive than negative recommendations)
- **Vocabulary Size:** Pre-filtered vocabulary from vocab.txt
- **Average Review Length:** Varies significantly across reviews

### Usage Instructions

1. **Setup Environment:**
   ```bash
   pip install pandas numpy scikit-learn gensim matplotlib seaborn scipy
   ```

2. **Run Notebooks:**
   - Start with `task1.ipynb` for data preprocessing
   - Continue with `task2_3.ipynb` for feature generation and classification

3. **Output Files:**
   - `count_vectors.txt`: Contains bag-of-words representations
   - Various model evaluation results and visualizations in notebooks

### Evaluation Metrics
- **Accuracy:** Overall classification accuracy
- **F1-Score:** Harmonic mean of precision and recall (primary metric due to class imbalance)
- **Confusion Matrix:** Detailed classification performance
- **Cross-Validation:** 5-fold validation for robust performance estimation

### Technical Notes
- **Random Seed:** 0 (for reproducibility)
- **Train/Test Split:** 67%/33%
- **Cross-Validation:** 5-fold stratified sampling
- **Missing Data Handling:** Empty reviews handled with zero vectors
- **Memory Optimization:** Sparse matrices used for count vectors

### Future Improvements
- Address class imbalance with sampling techniques
- Explore advanced embedding models (BERT, Word2Vec)
- Implement ensemble methods
- Feature engineering for domain-specific terms
- Hyperparameter tuning for better model performance

### References
- Course activities and lab materials
- GitHub Copilot for code assistance
- Scikit-learn documentation
- GloVe: Global Vectors for Word Representation

---
**Last Updated:** July 2025
**Assignment Status:** Completed
