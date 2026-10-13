# Dataset setup
The loader downloads the eight official Planetoid files per dataset from:
https://github.com/kimiyoung/planetoid/tree/master/data

All three datasets used in the experiments are included here: `cora/`, `citeseer/`, and `pubmed/`, with eight original Planetoid files per dataset. The default cache is this `data/` directory. Override with `--cache PATH` (the loader appends the dataset name). All 24 files were verified against SHA-256 hashes saved during training; see `dataset_manifest.json`. They are included in the updated submission ZIP, so no download is required.

Run `python main.py --mode debug` to load the bundled Cora data (downloading only if files are missing). CiteSeer and PubMed use `--dataset citeseer` and `--dataset pubmed`. Downloads use a 60-second timeout and atomic file replacement. If downloading fails, manually save these files in the dataset cache directory:
`ind.<dataset>.x`, `.y`, `.tx`, `.ty`, `.allx`, `.ally`, `.graph`, `.test.index`.
For example: https://raw.githubusercontent.com/kimiyoung/planetoid/master/data/ind.cora.allx

Use the raw download, not GitHub's HTML page. The loader reads legacy Python pickles from this research source: do not substitute untrusted pickle files. SHA-256 hashes of every input are logged and checked when evaluating checkpoints. The hashes record the exact inputs used, rather than claiming a pinned upstream release.

Feature/label reordering follows the Planetoid format. CiteSeer isolated rows are restored and unknown labels excluded from supervision. Feature rows are normalized to sum to one (zero rows remain zero). The graph is made undirected, duplicate edges collapsed, and self-loops added only for message passing. Adjacency normalization is symmetric. The paper's citation-link totals may differ from the processed undirected edge count; both are recorded in `results/logs/*_dataset.json`.

The original Planetoid training masks are not used. New random 3% training masks and a 10%/90% validation/test split of the remainder implement the paper's protocol. Training, validation and test node IDs are saved in checkpoints. Class IDs follow the source data's one-hot column order; semantic names are not guessed.
