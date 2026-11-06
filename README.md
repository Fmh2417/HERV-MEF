# HERV-MEF

Code for the HERV-MEF study of human endogenous retrovirus regulatory potential, sequence divergence and activity across biological backgrounds.

## Repository structure

| Directory | Contents |
|---|---|
| [Data_Preprocessing/](Data_Preprocessing/) | HERV catalogue construction and regulatory feature preparation |
| [Modeling/](Modeling/) | Sequence models, evolutionary calibration, global ranking and functional evidence reranking |
| [Biological_Interpretation/](Biological_Interpretation/) | Evolutionary, sequence, chromatin and LUAD analyses |
| [Model_pk/](Model_pk/) | Comparisons with published sequence and enhancer models |
| [Data/](Data/) | Locus annotations, sample groups, frozen models and prediction tables |

## Setup

```bash
conda env create -f environment.yml
conda activate herv-mef
```

Set the reference-data and external-model paths in [config.py](config.py). Model training and sequence embedding extraction require CUDA-enabled PyTorch. Dependencies for comparison models are provided by their original repositories, linked in `config.py`.

## Data

`Data/frozen_models/` contains model weights, the tokenizer and fitted preprocessors. `Data/published_results/` contains model comparison summaries and per-locus predictions. The remaining data folders contain the locus catalogue, Roadmap sample groups and representative-locus annotations.

Reference datasets and external model weights should be obtained from the sources described in the manuscript and `config.py`. DNA sequences are extracted from the hg38 reference genome during preprocessing. The complete HERV priority atlas is provided as Supplementary Data 1.

## Analysis

Run scripts from the repository root. First prepare the Roadmap sample groups and configuration, then build the locus catalogue and evidence features in `Data_Preprocessing/`. After integrating the evidence, run the modeling scripts in this order:

```bash
python Modeling/build_modeling_inputs.py
python Modeling/train_sequence_models.py
python Modeling/freeze_sequence_reference_model.py
python Modeling/calibrate_evolutionary_prior.py
python Modeling/train_global_ranker.py
python Modeling/train_functional_evidence_reranker.py
python Modeling/export_final_ranking.py
```

For biological analyses, run `Biological_Interpretation/build_biological_source_table.py` before the analysis scripts. Representative-locus selection precedes locus attribution.

For the sequence-model comparison:

```bash
python Model_pk/build_sequence_embeddings.py
python Model_pk/run_sequence_model_comparison.py
```

The other `run_*.py` scripts in `Model_pk/` run the enhancer-model comparisons; `summarise_model_comparison.py` combines their results. Sequence models are evaluated on the test split, and full-model comparisons use the validation split. Generated files are written to each module's `output/` directory.
