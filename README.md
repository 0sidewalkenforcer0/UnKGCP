# Certainty in Uncertainty: Reasoning over Uncertain Knowledge Graphs with Statistical Guarantees

This repository contains the official implementation for the EMNLP 2025 paper:  
**Certainty in Uncertainty: Reasoning over Uncertain Knowledge Graphs with Statistical Guarantees**.

## Overview

This project provides implementations and experiments for reasoning over uncertain knowledge graphs, with an emphasis on statistical guarantees.

## Backbone Models

Our work builds on three backbone models for uncertain knowledge graph reasoning:

- [**UKGE**: Embedding Uncertain Knowledge Graphs](https://github.com/stasl0217/UKGE)  
  Probabilistic embeddings for uncertain knowledge graphs.

- [**PASSLEAF**: A Pool-based Semi-supervised Learning Framework for Uncertain Knowledge Graph Embedding](https://github.com/Franklyncc/PASSLEAF)  
  Semi-supervised learning for uncertain knowledge graph embeddings.

- [**BEUrRE**: Probabilistic Box Embeddings for Uncertain Knowledge Graph Reasoning](https://github.com/stasl0217/beurre)  
  Probabilistic box embeddings for modeling uncertainty.

## Repository Structure

- **BEUrRE**: Implementation and experiments for the BEUrRE model.  
- **PASSLEAF**: Implementation and experiments for the PASSLEAF model.  
- **UKGE**: Implementation and experiments for the UKGE model.  

## Data
You can download all datasets used in our experiments from the following link:

https://1drv.ms/u/c/53E5DF60F4E3FFE8/EYFHc7ZRt-NAturvecIjUBMBG2xXgD3Ly0PNhIbzeOMONQ?e=hyC3Dr

The data includes:
- Knowledge graph triples for CN15K, NL27K, PPI5K, etc.
- Uncertainty annotations and calibration sets for each dataset.
- Preprocessed files for direct use in our code.

Please extract the downloaded files into the corresponding `data` folder of each model directory (e.g., `UKGE-master/data`).

## Running the Code

### Option A: Train from scratch  

1. Navigate to the model directory (e.g., for UKGE):  

   ```bash
   cd UKGE-master
   ```

2. Install dependencies (each model folder has its own `requirements.txt`):  

   ```bash
   pip install -r requirements.txt
   ```

3. Run the training script, for example:  

   ```bash
   python run_ppi5k.py
   ```

   You may also run other scripts as described in the corresponding README.

---

### Option B: Use pretrained models and datasets  

1. Navigate to the model directory (e.g., for UKGE):  

   ```bash
   cd UKGE-master
   ```

2. Install dependencies:  

   ```bash
   pip install -r requirements.txt
   ```

3. Create folders for pretrained assets:  

   ```bash
   mkdir -p trained_model
   mkdir -p data
   ```


4. Download the pretrained models and datasets into these folders (**[links to be provided](https://1drv.ms/f/c/53E5DF60F4E3FFE8/ElDL57bsSnhClPiALWlgJJ4Bn5-JLLWX9AZSXMuERRSkag?e=Odu6ZE)**).

---

### Reproducing Our Results

After setting up the environment and preparing the data/model files as above, you can reproduce our main results by running:

```bash
python test_adaptive_CP.py
```


