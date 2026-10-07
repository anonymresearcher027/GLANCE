# GLANCE

Code release for the ICLR 2027 submission:

**GLANCE: Global–Local Alignment of Neural and Contextual Embeddings for
Brain–Language Sentence Retrieval**

## Scope

This repository contains the GLANCE neural encoder, global–local matching
model, training objective, checkpoint selection, and retrieval evaluation.
The repository starts from prepared recording-level inputs. 

## Prepared input layout

The training code expects one prepared recording directory:

```text
<recording-root>/
├── features/
│   └── normalized_features.npy
├── text/
│   ├── sentence_embeddings.npy
│   ├── word_embeddings.npy
│   └── word_mask.npy
├── sentence_windows.csv
├── valid_time_mask.npy
└── splits/
    └── seed<N>/
        ├── train.npy
        ├── val.npy
        └── test.npy
