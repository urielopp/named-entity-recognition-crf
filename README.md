# Named Entity Recognition with Conditional Random Fields (CRF)

This project implements a **Named Entity Recognition (NER)** system using **Conditional Random Fields (CRF)**, trained and evaluated on the **CoNLL-2002** corpus for **Spanish** and **Dutch**. The system identifies and classifies named entities (persons, organizations, locations, and miscellaneous entities) using `nltk.tag.CRFTagger`, with a custom feature-extraction pipeline, multiple tagging schemes, and a greedy feature-selection search.

## Overview

The goal of this project is to build an automatic NER system with strong precision by:

- Implementing a custom feature-extraction class for the CRF tagger.
- Comparing different **tagging schemes** (BIO, IO, BIOW, BIOE, BIOEW, BIOW+).
- Selecting the most relevant features via a **Greedy Search**, parallelized with `joblib`.
- Evaluating performance with a custom **partial F1-score** metric that rewards partially correct entity boundaries.
- Testing the trained models on real-world, unseen text (extracted from the news) using **Stanza** for tokenization, POS-tagging, and lemmatization.

## Data

- **Corpus:** [CoNLL-2002](https://www.clips.uantwerpen.be/conll2002/ner/) (Spanish and Dutch).
- **Splits:** `train` (training), `testa` (validation / hyperparameter tuning), `testb` (final evaluation).

## Feature Engineering

Instead of using NLTK's default `CRFTagger` feature function, a custom `FeatFunc_Class` was implemented to easily turn individual features on or off for each language. Implemented feature groups include:

- **Word & capitalization attributes:** lowercase word form, all-lowercase / all-uppercase / title-case detection.
- **Alphanumeric information:** digit-only, alphanumeric, or mixed tokens.
- **Punctuation & symbols:** via pre-compiled regular expressions (e.g. `@`, `$`, `€`).
- **Word length:** categorical length buckets (useful for acronyms).
- **Word shape:** full shape (e.g. `Barcelona` → `Xxxxxxxxx`) and short shape (`Xx`).
- **Affixes:** 2–4 character prefixes/suffixes.
- **Morphosyntactic info:** POS-tag and lemma.
- **Context window:** up to ±2 tokens/POS-tags, including POS bigrams.
- **Gazetteers:** frequency-based (built from the training corpus) and static (NLTK's geographical dictionaries).

A set of helper functions handles data preparation, gazetteer generation, scheme conversion, entity grouping, the custom F1 metric, and predicting on raw real-world text — keeping the pipeline consistent and avoiding repeated code across experiments.

## Tagging Schemes

The corpus's standard `BIO` scheme was compared against five alternatives:

| Scheme | Description |
|---|---|
| BIO | Standard Begin / Inside / Outside |
| IO | No explicit "begin" marker |
| BIOW | Adds `W-` for single-token entities |
| BIOE | Adds `E-` for the last token of multi-token entities |
| BIOEW | Combines `W-` and `E-` |
| BIOW+ | Also tags `O` tokens adjacent to an entity with its type (`O-ENT`) |

## Feature Selection: Greedy Search

Rather than an exhaustive grid search, features are selected via a **greedy search**: starting from an empty configuration, at each iteration the candidate feature (boolean flag or context-window size) that most improves the partial F1-score on the validation set is permanently added. The search stops when no remaining candidate improves performance. Each iteration's candidate evaluations are parallelized across CPU cores using `joblib`.

### Results

For **Spanish**, the final F1-score was **0.7862** with 12 features selected. Word shape (`usa_forma`) alone drove the biggest gain (+0.49), since Spanish entities tend to follow regular capitalization patterns. Affixes and a ±2 word-context window followed as the next-largest contributors. Gazetteers entered late with minimal impact.

For **Dutch**, the final F1-score was **0.7719** with 11 features. Here, affixes (`usa_afixos`) drove the largest gain (+0.52) — Dutch relies heavily on characteristic prefixes/suffixes for place names and entities. Word length was also selected (unlike in Spanish), reflecting Dutch's tendency to form long compound words. Gazetteers were **not** selected at all, likely due to English-centric bias in NLTK's static dictionaries.

This contrast highlights a key linguistic difference: word shape is highly discriminative in Spanish but far less useful in Dutch, where compound words rarely repeat in shape — making affixes and length more informative instead.

## Tagging Scheme Comparison (Validation Set)

**Spanish** — best scheme: **BIOE** (F1 = 0.7889), outperforming base BIO (0.7862). Explicit end-of-entity markers help with long multi-word entities.

**Dutch** — best scheme: **BIO** (F1 = 0.7719); no advanced scheme improved on it, consistent with Dutch NER relying more on token-internal morphology than long-range sequence structure.

Across both languages, `IO` and `BIOW+` consistently underperformed — `IO` due to the loss of explicit entity-start boundaries, and `BIOW+` due to class sparsity introduced by tagging adjacent `O` tokens.

## Final Test Set Performance

| Language | Scheme | F1-Score | Precision | Recall |
|---|---|---|---|---|
| Spanish | BIOE | **0.8129** | 0.8243 | 0.8018 |
| Dutch | BIO | **0.7868** | 0.8076 | 0.7670 |

Both models scored *higher* on the test set than on validation, suggesting the greedy search successfully avoided overfitting. In both languages, Precision exceeds Recall — the models are conservative, favoring fewer but more reliable predictions.

## Evaluation on Real-World Text

The trained models were applied to genuine news sentences (Spanish: La Vanguardia, El País, Xataka, BBC News; Dutch: NU.nl, NOS.nl) using a Stanza-based preprocessing pipeline (`preparar_text_real`) to match the CoNLL feature format.

### Common Error Patterns

- **Category confusion** — the most frequent error; an entity is detected but mislabeled (e.g. an organization tagged as a location), since the model relies on morphology/context rather than semantic knowledge.
- **Out-of-domain & English terminology** — modern entities absent from the 2002-era corpus (e.g. *Anthropic*, *ChatGPT*) are misclassified by morphological analogy.
- **Ambiguous multi-word entities** — entities interrupted by common words (e.g. *Lula da Silva*, *New Jersey*) break the expected capitalization pattern.
- **Atypical names** — artistic names and pseudonyms (e.g. *Bad Gyal*) that don't follow standard patterns are often missed.
- **Cross-sentence inconsistency** — the CRF has no memory across sentences, so the same entity (e.g. *Google*) can get different labels in different contexts.

## Challenges

- **Computational cost:** an initial exhaustive grid search over feature combinations was computationally infeasible, motivating the switch to a parallelized greedy search.
- **POS-tag normalization:** mapping Stanza's tagsets to the CoNLL-2002 conventions for both Spanish and Dutch required extensive manual comparison and iterative refinement.

## Conclusions

- A custom, modular feature-extraction design enabled systematic, per-language experimentation.
- Greedy feature search proved an efficient alternative to exhaustive grid search and surfaced clear linguistic differences between Spanish and Dutch.
- A richer tagging scheme is not universally better: **BIOE** is optimal for Spanish, while plain **BIO** performs best for Dutch.
- Final test F1-scores of **0.8129** (Spanish) and **0.7868** (Dutch) confirm good generalization.
- Real-world text evaluation exposed the models' main limitations: dependence on the training corpus's domain/era and difficulty with novel, foreign, or atypical entities.

## Tech Stack

- Python
- [NLTK](https://www.nltk.org/) (`CRFTagger`, static gazetteers)
- [spaCy](https://spacy.io/) (lemmatization during training data preparation)
- [Stanza](https://stanfordnlp.github.io/stanza/) (tokenization, POS-tagging, and lemmatization for raw real-world text)
- [joblib](https://joblib.readthedocs.io/) (parallelized greedy feature search)

## Dataset

- [CoNLL-2002 Shared Task: Language-Independent Named Entity Recognition](https://www.clips.uantwerpen.be/conll2002/ner/)

**Note:** The detailed academic report containing the full implementation methodology, feature engineering, greedy search results, and in-depth error analysis is available in the repository in Catalan as `memoria.pdf`.
