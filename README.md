# PubMed Candesartan NLP / ML Pipeline

Educational Python project for processing PubMed articles about **Candesartan** with biomedical NLP, a knowledge graph, and a Bayesian network.

## Overview

The pipeline downloads PubMed records through NCBI Entrez, processes article text, extracts predefined biomedical entities and simple relations, builds a knowledge graph, and runs probabilistic experiments with a Bayesian network.

## Pipeline

```text
PubMed / NCBI
     |
     v
Article loader + cache
     |
     v
BioBERT + spaCy + rule-based NER
     |
     +------> Entity / relation extraction
     |
     v
Knowledge graph (NetworkX)
     |
     v
Bayesian network (pgmpy)
     |
     v
Experiment results
```

## Tech Stack

- Python 3.10+
- Biopython / NCBI Entrez
- Hugging Face Transformers
- BioBERT (`dmis-lab/biobert-base-cased-v1.1`)
- spaCy
- NetworkX
- pgmpy
- pandas / NumPy
- Matplotlib

The dependencies are declared in `requirements.txt`. citenone

## What the Project Does

### 1. PubMed data collection

`src/pubmed_loader.py` uses `Bio.Entrez` to search PubMed and retrieve article records. The pipeline stores article data locally and uses hashes to detect changes between runs.

### 2. Biomedical NLP

`src/nlp_pipeline.py` combines BioBERT, spaCy, and transparent rule-based extraction.

The current entity layer covers the educational classes:

- `DRUG`
- `DISEASE`
- `SYMPTOM`
- `DOSAGE`
- `EFFECT`

BioBERT is used as the transformer backbone for contextual features; the repository explicitly uses rules for the task-specific entity labels rather than claiming a task-specific BioBERT NER model.

### 3. Knowledge graph

`src/graph_builder.py` builds a `networkx.MultiDiGraph` from extracted entities and relations. The graph can be exported as a visualization for further analysis.

### 4. Bayesian network

`src/bayesian_network.py` defines a probabilistic model with `pgmpy`, including variables related to treatment, blood pressure, hypertension, stroke risk, diabetes, side effects, and treatment duration.

The probabilities are **demonstration values** and are not intended for medical decision-making.

### 5. Experiments

`src/experiments.py` runs five predefined probabilistic experiments and stores the results as CSV for reporting.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Usage

Run the complete pipeline:

```bash
python src/main.py --email your.email@example.com --retmax 10
```

Use previously downloaded articles:

```bash
python src/main.py --skip-download
```

## Project Structure

```text
PubMed-Project/
├── data/
│   ├── articles/
│   ├── processed/
│   └── cache/
├── src/
│   ├── pubmed_loader.py
│   ├── update_checker.py
│   ├── nlp_pipeline.py
│   ├── graph_builder.py
│   ├── bayesian_network.py
│   ├── experiments.py
│   └── main.py
├── report_examples/
├── requirements.txt
└── README.md
```

## Results

The pipeline produces:

- downloaded PubMed article JSON;
- cached article hashes;
- extracted NLP entities, dependencies, and relations;
- a knowledge-graph visualization;
- CSV results from the Bayesian experiments.

No accuracy or medical performance metric is claimed because the repository does not contain a validated benchmark for these components.

## Notes

- PubMed access requires an internet connection.
- The first BioBERT run downloads model weights from Hugging Face.
- If `en_core_web_sm` is unavailable, the code falls back to a blank English spaCy pipeline with sentence segmentation.
- Bayesian-network probabilities are educational examples, not clinical evidence.
