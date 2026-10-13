# Experimental artifacts
Only completed runs belong in the main metrics and summary tables. Debug artifacts are isolated in `debug/` and never included in experiment aggregates.

- `metrics/`: run JSON, test-node predictions, per-class CSVs.
- `tables/`: individual runs, mean/sample SD summaries, paper comparison CSV/Markdown.
- `figures/`: confusion matrices, class distributions, training/validation curves, comparisons.
- `logs/`: per-epoch component losses, validation scores, beta range and input hashes/statistics.
- `checkpoints/`: best-validation PyTorch weights, optimizer state, saved masks, configuration, hard weights and input hashes.

Every test result is measured at the validation-selected checkpoint. An evaluation command reloads that checkpoint and its exact masks. Checkpoints are for evaluation, not an implemented resume-training interface. Re-running the same dataset/model/seed replaces that run's artifacts; use a separate `--output` for a new configuration. Do not mix configurations in one aggregate output directory. No raw graph files are included here.
