# Hey Bruno! 👋

So I've been working on this semantic embeddings extraction pipeline for the NatSounds dataset, and I thought I'd give you a quick rundown of what I've built. It's pretty cool actually!

## What I Did

Basically, I created a comprehensive Jupyter notebook that extracts semantic embeddings from a bunch of different pre-trained language models. The idea is to get numerical representations of our "what" and "how" text data so we can do all sorts of cool analysis later - clustering, similarity comparisons, visualization, you name it.

## The Models I'm Using

I've integrated **11 different embedding models** (yeah, I went a bit overboard 😅):

1. **Word2Vec** - The classic! Using Google's pre-trained model (300M words, 300 dimensions)
2. **BERT** - The transformer that changed everything (768 dims)
3. **RoBERTa** - BERT but better optimized (768 dims)
4. **DistilBERT** - BERT's lightweight cousin (60% smaller, 97% performance, 768 dims)
5. **Multilingual BERT** - Handles 104 languages (768 dims)
6. **GPT-2** - The generative model (uses mean pooling since no CLS token, 768 dims)
7. **FastText** - Facebook's word embeddings with subword info (300 dims)
8. **GloVe** - Stanford's global vectors (300 dims)
9. **Sentence-Transformer (MPNet)** - Optimized for sentence embeddings (768 dims)
10. **Sentence-Transformer (MiniLM)** - Smaller, faster version (384 dims)
11. **Multilingual Sentence-Transformer** - For cross-lingual stuff (768 dims)

## The [CLS] Token Thing

So here's something interesting I learned and documented: BERT and similar models use this special **[CLS] token** at the beginning of each sequence. After the text goes through all the transformer layers, this CLS token basically contains a contextualized representation of the entire input. It's like a summary vector!

For BERT, RoBERTa, DistilBERT, and mBERT, I extract the CLS token embedding (it's at position 0 in the sequence). But GPT-2 doesn't have a CLS token because it's a decoder-only model, so I use **mean pooling** instead - just average all the token embeddings together, excluding padding tokens.

I added a whole markdown section explaining this to students because it's a key concept they need to understand.

## What the Code Does

The pipeline is pretty straightforward:

1. **Load the data** - Reads "what" and "how" columns from a dataframe
2. **Clean the text** - Removes underscores (replaces with spaces) because some models don't handle them well
3. **Extract embeddings** - For each model, creates embeddings for:
   - "what" texts
   - "how" texts  
   - "what_how" (concatenated together)
4. **Save everything** - Stores embeddings as `.npy` files in organized folders
5. **Create a dictionary** - Loads all embeddings into a single Python dictionary for easy access
6. **Save with pickle** - Also saves the dictionary as a pickle file for quick loading later

## The t-SNE Visualization

I also built a t-SNE visualization function that:
- Takes any embedding type
- Reduces dimensions to 2D using t-SNE
- Creates three subplots (what, how, what_how combined)
- Shows unique text labels (I fixed it so duplicates don't clutter the plot!)

## Code Quality & Documentation

I made sure to:
- Add detailed comments explaining what each part does
- Document the CLS token extraction process
- Add markdown cells with theory for students
- Explain why GPT-2 uses mean pooling vs CLS tokens
- Make the code readable and educational

## Output Structure

Everything gets saved in `outputs/`:
```
outputs/
├── embeddings_word2vec/
│   ├── what.npy
│   ├── how.npy
│   └── what_how.npy
├── embeddings_bert/
│   └── ...
├── ... (for each model)
└── all_embeddings_dict.pkl  ← The pickle file with everything!
```

## The Embeddings Dictionary

So I created this big dictionary that contains ALL the embeddings from all models. It's super convenient because you can access everything in one place without having to load individual `.npy` files.

### Dictionary Structure

The dictionary is nested like this:
```python
all_embeddings = {
    'word2vec': {
        'what': numpy_array,      # Shape: (600, 300)
        'how': numpy_array,        # Shape: (600, 300)
        'what_how': numpy_array   # Shape: (600, 300)
    },
    'bert': {
        'what': numpy_array,      # Shape: (600, 768)
        'how': numpy_array,        # Shape: (600, 768)
        'what_how': numpy_array   # Shape: (600, 768)
    },
    'roberta': { ... },
    'distilbert': { ... },
    # ... and so on for all 11 models
}
```

Each model has three keys:
- **'what'**: Embeddings for the "what" column texts
- **'how'**: Embeddings for the "how" column texts  
- **'what_how'**: Embeddings for concatenated "what + how" texts

The arrays are NumPy arrays where:
- First dimension = number of samples (600 in our case)
- Second dimension = embedding dimension (varies by model: 300, 384, or 768)

### How to Load the Dictionary

Super easy! Just use pickle:

```python
import pickle
import numpy as np

# Load the entire dictionary
with open('outputs/all_embeddings_dict.pkl', 'rb') as f:
    all_embeddings = pickle.load(f)

# Now you can access any embedding you want:
# Get BERT embeddings for "what" texts
bert_what = all_embeddings['bert']['what']

# Get Word2Vec embeddings for "how" texts
w2v_how = all_embeddings['word2vec']['how']

# Get RoBERTa embeddings for combined texts
roberta_combined = all_embeddings['roberta']['what_how']

# Check the shape
print(bert_what.shape)  # Should be (600, 768)
```

### Example Use Cases

Once loaded, you can do all sorts of things:

```python
# Compare embeddings from different models
bert_what = all_embeddings['bert']['what']
roberta_what = all_embeddings['roberta']['what']

# Compute similarity between two texts using cosine similarity
from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity([bert_what[0]], [bert_what[1]])[0][0]

# Get all "what" embeddings across all models
what_embeddings = {model: all_embeddings[model]['what'] 
                   for model in all_embeddings.keys()}

# Access a specific embedding
first_text_bert_embedding = all_embeddings['bert']['what'][0]  # First sample
```

The dictionary makes it really easy to:
- Compare different models
- Switch between embedding types for experiments
- Do ensemble methods combining multiple models
- Quickly access embeddings without remembering file paths

## What's Next?

The embeddings are ready for:
- Similarity analysis
- Clustering
- Classification tasks
- Dimensionality reduction and visualization
- Any downstream ML task really!

The notebook is pretty self-contained and well-documented, so students (or future me) should be able to understand what's happening. I tried to make it educational with the theory sections.

Let me know if you want me to run any specific analyses or if you have questions about any part of it!

---

*Quick reference - Loading the dictionary:*
```python
import pickle
all_embeddings = pickle.load(open('outputs/all_embeddings_dict.pkl', 'rb'))
# Access: all_embeddings['model_name']['what'|'how'|'what_how']
```

