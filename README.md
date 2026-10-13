# MultiEURLEX Multi-Label Legal Text Classification
The paper has this repository associated with it and includes the notebooks required to reproduce the results. This study evaluates baseline(TF-IDF Logistic Regression and linearSVM) and transformer (BERT, mBERT and XLM-R) models for carrying out multilabel, multilingual classification of legal documents using the MultiEURLEX dataset. It compares two different training approaches (one using only English and the other using a cyclic multilingual approach), two different loss functions (unweighted versus class-weighted, in order to deal with the severe imbalance of the labels), and examines transformer models against classical TF-IDF baseline models, assessing all the configurations across all 23 official EU languages in the MultiEURLEX dataset.

## Run settings

- **Platform:** Google Colab
- **Processor:** TPU v5e-1
- **Run `MultiEURLEX_setup.ipynb` first, on CPU.** This notebook downloads and
  prepares the MultiEURLEX dataset and saves it to Google Drive for the other notebooks to
  load. It does not work in a TPU/GPU runtime due to compatibility issues. It is to be run with a plain CPU runtime before
  any of the baseline or transformer notebooks.
- All subsequent notebooks (`Baseline_*.ipynb`, `Transformers_*.ipynb`) load the
  dataset from that saved location and should be run on the TPU v5e-1 runtime.

## Dataset

**MultiEURLEX** (`"all_languages"` config): 55,000 train / 5,000 validation / 5,000
test documents, labeled with 567 level-3 EuroVoc concepts which is a severely imbalanced
multi-label setting (~1.02% positive rate per label) and is split chronologically (train:
1958–2010, validation: 2010–2012, test: 2012–2016).

## Experimental dimensions

The transformer notebooks (BERT, mBERT and XLM-R) vary along two independent axes:

1. **Training regime**:
   - **OTM (One-to-Many)**: train and validate on the **English** version of each
     document only; evaluate on the raw test-set text of all 23 languages. Tests
     zero-shot cross-lingual transfer from an English-only model.
   - **MTM (Many-to-Many, cyclic)**: train on a *different* language per document per
     epoch, cycling through all 23 languages
     (`assign_language(idx, epoch) = LANGUAGES[(idx + epoch) % 23]`, with an English
     fallback when a translation is missing). Validated on a 5-language subset
     (`en, de, fr, bg, hu`). Tests whether multilingual exposure during training
     improves cross-lingual transfer over OTM.
2. **Loss weighting**:
   - **Unweighted (UNW)**: plain `BCEWithLogitsLoss`.
   - **Class-weighted (CW)**: `BCEWithLogitsLoss` with a per-label `pos_weight`
     computed from training-set label frequency and clipped to `[1, 50]`, to counter
     the 1.02% positive rate.

The baseline notebooks (TF-IDF + Logistic Regression / Linear SVM, one-vs-rest,
class-balanced) are included under both OTM and MTM regimes for comparison against
the transformer models.

All transformer training uses HuggingFace `accelerate`, **fp32 precision** throughout
(bf16 was found to cause degenerate collapse under unweighted BCE with this label
distribution), seed 42, max sequence length 256, batch size 16, 10 epochs, learning
rate 2e-5, AdamW with linear warmup/decay.

## Notebooks

| Notebook | Role | Regime | Loss |
|---|---|---|---|
| `MultiEURLEX_setup.ipynb` | Dataset download/prep (**run first, CPU**) | — | — |
| `Baseline_OTM.ipynb` | TF-IDF + Logistic Regression / Linear SVM | OTM | Class-balanced |
| `Baseline_MTM.ipynb` | TF-IDF + Logistic Regression / Linear SVM | MTM | Class-balanced |
| `Transformers_OTM_UNW.ipynb` | Transformer | OTM | Unweighted |
| `Transformers_OTM_CW.ipynb` | Transformer | OTM | Class-weighted |
| `Transformers_MTM_CYCLIC_UNW.ipynb` | Transformer | MTM (cyclic) | Unweighted |

Each notebook's own first markdown cell restates its exact configuration (model,
regime, loss).

## Structure common to every transformer notebook

1. **Setup** — install dependencies, mount Drive, load the pre-saved dataset.
2. **Configuration** — seed, `accelerate` init, hyperparameters, model name.
3. **Metrics** — micro/macro F1, precision, recall, accuracy helper.
4. **Label vocabulary** — builds the 567-label multi-hot encoding.
5. **Preprocessing** — tokenizes the English subset (OTM) or the per-epoch
   cyclically-assigned language (MTM).
6. **DataLoader** — collator and validation loader.
7. **Model, optimizer, scheduler** — `AutoModelForSequenceClassification` with
   `problem_type="multi_label_classification"`, AdamW, linear warmup.
8. **Evaluation function** — runs the English (OTM) or 5-language (MTM) validation set.
9. **Training loop** — epoch loop with running loss, validation pass per epoch,
   history logged for plotting.
10. **Training curves** — train/val loss plot.
11. **Multilingual evaluation** — forward pass over all 23 language versions of the
    test set.
12. **Per-language metrics** — per-language and pooled/averaged metric table.
13. **Inference time** — wall-clock timing over the pooled 23-language forward passes.
14. **Model size** — total/trainable parameter counts.

The baseline notebooks follow the same general shape (setup → vectorize → train OvR
classifier → evaluate per language → save results) but without the accelerate/epoch
machinery, since they are non-neural and train in a single `fit()` call.

## Reproducing results

1. Run `MultiEURLEX_setup.ipynb` on a CPU runtime to download and save the dataset to
   Google Drive.
2. Switch to a TPU v5e-1 runtime.
3. Run any of the `Baseline_*.ipynb` / `Transformers_*.ipynb` notebooks top-to-bottom
   — each loads the dataset saved in step 1 and reproduces its reported metrics,
   training curves, per-language breakdown, and model size/inference-time figures.
