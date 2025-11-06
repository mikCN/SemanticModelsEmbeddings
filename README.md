# Semantic Models Embeddings

This repository contains a comprehensive pipeline for extracting semantic embeddings from multiple pre-trained language models for the NatSounds dataset.

## Overview

This project extracts semantic embeddings from 11 different pre-trained language models to create numerical representations of "what" and "how" text data. These embeddings can be used for similarity analysis, clustering, classification, visualization, and other downstream tasks.

## Models Included

1. **Word2Vec** - Google's pre-trained model (300 dimensions)
2. **BERT** - Bidirectional Encoder Representations (768 dimensions)
3. **RoBERTa** - Optimized BERT (768 dimensions)
4. **DistilBERT** - Lightweight BERT (768 dimensions)
5. **Multilingual BERT** - Cross-lingual BERT (768 dimensions)
6. **GPT-2** - Generative Pre-trained Transformer (768 dimensions)
7. **FastText** - Facebook's word embeddings (300 dimensions)
8. **GloVe** - Global Vectors (300 dimensions)
9. **Sentence-Transformer (MPNet)** - Optimized for sentences (768 dimensions)
10. **Sentence-Transformer (MiniLM)** - Smaller version (384 dimensions)
11. **Multilingual Sentence-Transformer** - Cross-lingual (768 dimensions)

## Repository Structure

```
SemanticModelsEmbeddings/
├── scripts/
│   └── extract_semEmbeddings.ipynb  # Main notebook with embedding extraction pipeline
├── data/                            # Dataset (not included in repo - too large)
├── outputs/                         # Generated embeddings (not included - use .pkl file)
├── models/                          # Pre-trained models (not included)
├── README.md                        # This file
└── INFORMAL_REPORT.md               # Informal report explaining the work
```

## Getting Started

### Prerequisites

- Python 3.7+
- Jupyter Notebook
- Required packages (see notebook for full list):
  - pandas
  - numpy
  - transformers
  - sentence-transformers
  - gensim
  - scikit-learn
  - matplotlib
  - torch

### Usage

1. **Prepare your data**: Place your Excel file with "what" and "how" columns in the `data/` folder
2. **Open the notebook**: `scripts/extract_semEmbeddings.ipynb`
3. **Run the cells**: The notebook will:
   - Load and preprocess your data
   - Extract embeddings from all models
   - Save embeddings as `.npy` files
   - Create a dictionary and save it as `.pkl`

### Loading Embeddings

After running the notebook, you can load embeddings in two ways:

#### Option 1: Load from local file (after running the notebook)

```python
import pickle

with open('outputs/all_embeddings_dict.pkl', 'rb') as f:
    all_embeddings = pickle.load(f)

# Access embeddings like:
bert_what = all_embeddings['bert']['what']
word2vec_how = all_embeddings['word2vec']['how']
```

#### Option 2: Download pre-computed embeddings from Google Drive

The complete embeddings dictionary (`all_embeddings_dict_semanticModels.pkl`) is available on Google Drive:

**[Download Embeddings Dictionary](YOUR_GOOGLE_DRIVE_LINK_HERE)**

After downloading, place it in the `outputs/` folder and load it as shown above.

## Key Features

- **11 Different Models**: Compare embeddings from various architectures
- **Comprehensive Documentation**: Markdown cells explain each model and concept
- **t-SNE Visualization**: Built-in function to visualize embeddings in 2D
- **Easy Access**: Dictionary structure for convenient embedding access
- **Educational**: Well-commented code with theory explanations

## Understanding [CLS] Tokens

BERT-based models use a special `[CLS]` token that accumulates information from the entire sequence through self-attention. This token's embedding represents the whole sentence. GPT-2 doesn't have a CLS token, so we use mean pooling instead.

See the notebook for detailed explanations!

## Output

All embeddings are saved in the `outputs/` directory:
- Individual `.npy` files for each model and text type
- `all_embeddings_dict.pkl` - Complete dictionary with all embeddings (created when you run the notebook)
- `all_embeddings_dict_semanticModels.pkl` - Pre-computed embeddings dictionary (available on [Google Drive](YOUR_GOOGLE_DRIVE_LINK_HERE))

## Notes

- Large files (embeddings, models, data) are excluded from git via `.gitignore`
- The notebook is self-contained and well-documented
- All embeddings are saved for easy reuse

## License

[Add your license here]

## Contact

[Add your contact information here]

