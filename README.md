# GNN-INCM Replication

Independent practical reproduction of **A graph neural network-based node classification model on class-imbalanced graph data**, Zhenhua Huang, Yinhao Tang and Yunwen Chen, *Knowledge-Based Systems* 244 (2022), 108538. [Paper DOI](https://doi.org/10.1016/j.knosys.2022.108538).

**Status:** the core algorithm runs on real datasets and has genuine experiment artifacts. This is not an exact numerical replication. HSKDM is not implemented. Read the documented deviations before interpreting comparisons.

## Project objective and architecture
A two-layer GCN provides K-dimensional embeddings for supervised node classification, Student-t embedding clustering (ECO) and sampled inner-product graph reconstruction (GRO). Multiplicative hard-node weights affect all three losses. The project compares an ordinary GCN with GNN-INCM and ablations without clustering, reconstruction or hard weights.

**All Python implementation is in `main.py`.** The notebook imports it; no project-specific Python modules are required. The organization is inspired by [the requested reference repository](https://github.com/Vedansh2705/cwgan-gp-replication); its unrelated CWGAN implementation and results are not used.

## Installation
Use Python 3.12 (the tested version) or a compatible Python with available PyTorch wheels. From this folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

A virtual environment is recommended, not required. CPU PyTorch is sufficient. Exact installed versions used for the completed runs are recorded in `results/logs/environment.json`. Requirements provide compatible version ranges; numerical identity across different package versions is not guaranteed.

## Run commands
```bash
python main.py --mode debug
python main.py --mode train --dataset cora --model gcn
python main.py --mode train --dataset cora --model gnn_incm
python main.py --mode evaluate --dataset cora --model gnn_incm
python main.py --mode all --dataset cora
jupyter notebook GNN_INCM_Reproduction.ipynb
```

The first train/evaluate commands default to seed 42. `all` runs both main models and all three ablations for seeds 42, 43, 44. All modes are implemented. Evaluation requires an existing checkpoint and the same input files; it loads the saved masks and validates their input hashes. `--cache PATH` and `--output PATH` work after moving the project to another computer. Training defaults to 1000 maximum epochs and 200-epoch patience, with the best validation macro F1 checkpoint retained.

Commands used for the submitted experiments:
```bash
python main.py --mode all --dataset cora --seeds 42 43 44
python main.py --mode all --dataset citeseer --skip-ablations --seeds 42 43 44
python main.py --mode all --dataset pubmed --skip-ablations --seeds 42 43 44
```

To use a shorter exploratory budget, explicitly pass `--epochs 100 --patience 30 --output results_short`. Do not mix different configurations in one output directory. A repeated dataset/model/seed overwrites that run's files. `--mode summarize` rebuilds comparison tables from existing completed metrics. `FAST_DEBUG=True` controls only the default when `--mode` is omitted; explicit training modes always run real experiments.

## Dataset setup
The exact [Planetoid dataset files](https://github.com/kimiyoung/planetoid/tree/master/data) used in the experiments are included under `data/cora/`, `data/citeseer/` and `data/pubmed/` (eight files each). The default cache is now `data/<dataset>/` within the project; no download is needed. Missing files are downloaded automatically. If download fails, manually save `ind.<dataset>.{x,y,tx,ty,allx,ally,graph,test.index}` there. See `data/README.md` for exact filenames, source and caveats. All 24 input files are included in the updated submission archive and verified against the training-record hashes.

Node/feature/class counts match the paper. Processed unique undirected edge counts differ from its citation-link totals. CiteSeer has 15 unknown-label rows that remain in the graph but are excluded from supervised masks. The split is 3% training followed by 10%/90% of the remainder for validation/test. These are new random masks, not the standard Planetoid public masks.

## Notebook
The notebook includes all 23 requested explanatory sections and calls the same implementation. It loads saved tables and figures by default, so no training or download is needed for a presentation. Set `RUN_DEBUG=True` or `RUN_TRAINING=True` to run those paths; notebook training writes to `results_rerun/` to preserve the submission. The submitted notebook is executed and includes outputs. See the verification record for execution status.

## Completed experiments and results
Twenty-seven non-debug experiments completed: 15 Cora runs (five model configurations times three seeds), six CiteSeer runs and six PubMed runs (GCN and GNN-INCM times three seeds on each). All used the stated 1000-epoch cap and 200-epoch patience; early stopping usually ended training sooner. No extra hyperparameter search was performed after examining test outcomes.

| Dataset | Model | Macro F1 | MCC |
|---|---|---|---|
| citeseer | gcn | 0.5276 +/- 0.0671 | 0.5491 +/- 0.0738 |
| citeseer | gnn_incm | 0.5740 +/- 0.0108 | 0.5385 +/- 0.0194 |
| cora | gcn | 0.4032 +/- 0.2630 | 0.3690 +/- 0.3224 |
| cora | gnn_incm | 0.6432 +/- 0.0207 | 0.6033 +/- 0.0176 |
| cora | no_cluster | 0.6432 +/- 0.0207 | 0.6033 +/- 0.0176 |
| cora | no_hard | 0.6707 +/- 0.0406 | 0.6313 +/- 0.0428 |
| cora | no_reconstruction | 0.6432 +/- 0.0225 | 0.6030 +/- 0.0205 |
| pubmed | gcn | 0.7466 +/- 0.0511 | 0.6347 +/- 0.0630 |
| pubmed | gnn_incm | 0.7555 +/- 0.0046 | 0.6494 +/- 0.0064 |

All summary rows are mean +/- sample standard deviation over actual runs. Full per-seed metrics, accuracy, macro precision/recall and per-class scores are under `results/`. Paper comparison F1 values are not guaranteed to use the same averaging convention; the original paper does not specify it. No missing result is invented.

## Important limitations
- Balanced sampled reconstruction replaces the dense all-pair loss and excludes self-pairs.
- Each loss is averaged independently, changing its relative scale and interaction with weight decay.
- Feature row normalization, fixed initial k-means centers, gamma=1.01 and beta cap=100 are declared assumptions.
- Apparent signs in the paper's class-weight/reconstruction equations are corrected to match its stated minimization objectives.
- Three seeds rather than ten; the baseline uses our controlled common settings rather than the paper's baseline-specific ones.
- Near-zero clustering losses and weak component effects prevent claiming successful numerical reproduction. Cora's no-hard ablation outperforms the core model on mean macro F1.
- HSKDM is analyzed but not implemented; additional dataset ablations are not run. No focal-loss or prediction-averaging substitute is presented as the original method.

## Files and outputs
```text
GNN-INCM-Replication/
  main.py
  GNN_INCM_Reproduction.ipynb
  README.md
  requirements.txt
  .gitignore
  paper_analysis.md
  replicated_research_paper.md
  reproduction_report.md
  reproduction_report.pdf
  submission_manifest.json
  data/README.md
  data/cora/
  data/citeseer/
  data/pubmed/
  data/dataset_manifest.json
  results/
    README.md
    metrics/       # JSON, per-class CSV, per-node test predictions
    tables/        # individual_runs, summary, paper_vs_ours (CSV/Markdown)
    figures/       # confusion, classes, loss/validation curves, comparisons
    logs/          # epoch histories, dataset hashes, environment, verification
    checkpoints/   # best-validation model/optimizer state, masks and settings
    debug/         # isolated debug outputs, never included in main aggregates
```

Submit the supplied project ZIP, or the entire folder above excluding `__pycache__` and virtual environments; keep the included dataset folders. `submission_manifest.json` lists every packaged file and checksum. The manuscript is `replicated_research_paper.md`; the formal report is `reproduction_report.md` and its PDF. Original authors retain credit for the research method; this independent reproduction was prepared with AI assistance. Add your institution-required author identification before submission.

## References
1. Huang, Tang and Chen (2022), [original paper](https://doi.org/10.1016/j.knosys.2022.108538).
2. Kipf and Welling (2017), [GCN](https://arxiv.org/abs/1609.02907).
3. Yang, Cohen and Salakhutdinov (2016), [Planetoid](https://github.com/kimiyoung/planetoid).

## Dataset packaging update
At the user's request, all three exact datasets are now bundled. Statements in the original research report/manuscript about excluding raw data describe the initial package, before this packaging update. Experiment results and checkpoints are unchanged.
