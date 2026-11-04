# CasSEE

Prediction of Cas9 / Cas9-variant on-target editing efficiency and off-target activity.

## Install

```bash
pip install -r requirements.txt
```

A GPU is used automatically when available, otherwise everything runs on CPU.

## Run

```bash
jupyter lab notebooks/
```

`FOLD` in the first cell of each notebook selects the cross-validation fold
(0–4, default `0`). Set it to `None` to run all five folds and print the mean.

| notebook | reports | metric |
|---|---|---|
| `01_seqtest.ipynb` | 6 on-target benchmark sets | Spearman |
| `02_muttest.ipynb` | 3 Cas9-variant benchmark sets | Spearman |
| `03_offtarget.ipynb` | 66 off-target benchmark sets | Pearson |
| `04_hf1.ipynb` | SpCas9-HF1 on-target and off-target benchmark sets | Spearman / Pearson |
