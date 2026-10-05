# GLANCE
code release for the ICLR 2027 submission:

**GLANCE: Global–Local Alignment of Neural and Contextual Embeddings
for Brain–Language Sentence Retrieval**

## Scope

This repository contains the GLANCE model, training loop, loss functions,
checkpoint selection, and retrieval evaluation.

The repository assumes that the Brain Treebank data have already been
downloaded and prepared according to the preprocessing procedure described in
the paper. Raw recordings, filtering code, superlet extraction code, and
dataset files are not included here.

## Input data

The training code expects one prepared recording directory with the following
logical contents:

```text
<recording-root>/
├── neural_features.npy
├── sentence_embeddings.npy
├── word_embeddings.npy
├── word_mask.npy
├── valid_time_mask.npy
├── sentence_windows.csv
├── channel_order.csv
└── splits/
    └── seed<N>/
        ├── train.npy
        ├── val.npy
        └── test.npy
