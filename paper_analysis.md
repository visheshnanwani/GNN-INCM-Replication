# Paper analysis and implementation decisions

## Source and scope
Huang, Z., Tang, Y., and Chen, Y. (2022). *A graph neural network-based node classification model on class-imbalanced graph data*. Knowledge-Based Systems 244, 108538. https://doi.org/10.1016/j.knosys.2022.108538

The supplied 12-page PDF was read, including Sections 3-5, Algorithms 1-2 and Table 2. The algorithms and Table 2 were also checked against rendered pages. This project is an independent practical reproduction, not the authors' code or an exact numerical replication. The authors' linked repository, https://github.com/Geralt-TYH/OurModel, returned HTTP 404 during this work. The requested organizational reference, https://github.com/Vedansh2705/cwgan-gp-replication, was inspected for its separation of data, evaluation, results and documentation; no implementation or results were copied.

## Research question
Can a graph encoder learn useful representations for minority classes when only a small, naturally imbalanced labeled subset is available? GNN-INCM combines class-balanced supervision, embedding clustering, structural reconstruction and adaptive node difficulty. It does not synthesize minority nodes.

## Data and protocol (Table 1 and Section 5.1)
| Dataset | Nodes | Features | Classes | Paper links |
|---|---:|---:|---:|---:|
| Cora | 2708 | 1433 | 7 | 5429 |
| CiteSeer | 3327 | 3703 | 6 | 4732 |
| PubMed | 19717 | 500 | 3 | 44338 |

The paper samples 3% of nodes uniformly for training, then assigns 10% of the remainder to validation and 90% to test. These are approximately 3%, 9.7%, 87.3% of the full dataset, not 3%, 10%, 87%. This code floors the first two sizes, uses a seeded random permutation, and assigns the remainder to test. It does not stratify, rebalance, or silently resample splits. Only training labels enter loss weights and cluster-count calculation. Full graph features and edges participate in unsupervised transductive training, as intended. Validation labels select checkpoints; test labels are evaluated only after selection. Exact original seeds and splits are unavailable.

The Planetoid distribution is used. Symmetrization and duplicate removal can produce fewer unique undirected edges than the citation-link totals in Table 1. Logs distinguish paper links, directed adjacency entries, and unique undirected edges. CiteSeer has isolated missing rows in its Planetoid test range; restored zero-feature nodes with unknown all-zero labels are retained in the graph and excluded from supervised splitting/evaluation. No class labels are invented.

## Architecture and equations
Equation (1): O = A_norm ReLU(A_norm X W0) W1, with A_norm = D^(-1/2)(A+I)D^(-1/2). The first layer has 100 units; the output embedding has K dimensions, the number of classes. The implementation has no bias and applies dropout 0.3 after ReLU. It uses sparse graph multiplication and row-normalized features (the latter is an explicit assumption).

Classification uses softmax(O) and weighted cross-entropy on training nodes. The positive inverse-frequency coefficient is alpha_i = |V_l|/(K |C_yi|). This implements the prose of Eq. (4), whose printed negative sign conflicts with minimizing Eq. (3). Baseline GCN uses ordinary unweighted cross-entropy.

ECO initialization uses k-means on *training-node* embeddings before optimization, with k = ceil(|V_l|/min_c |C_c|), Eq. (5). Centers are fixed buffers: the paper explicitly optimizes W0 and W1 but does not explain center updates. K-means uses 10 initializations and the run seed. A missing training class raises an error rather than silently changing the split.

Equation (6): q_ij is proportional to (1 + ||o_i-u_j||^2)^(-1), normalized over k clusters. Equation (7): p_ij is proportional to q_ij^2/sum_i q_ij, normalized over clusters. Targets are recomputed each epoch and detached from gradient tracking. Clustering minimizes KL(P||Q) across all graph nodes. The paper's Eq. (8) prints an upper limit K despite k centers; the implementation sums over k.

GRO predicts sigmoid(o_i^T o_j), Eq. (9). Binary cross-entropy is implemented using logits for numerical stability. Equations (10)/(14) print a log likelihood without the negative sign, despite describing its minimization as BCE. The implementation minimizes *negative* log likelihood. Instead of materializing n by n probabilities, it samples up to 10,000 positive ordered edges and the same number of uniformly sampled off-diagonal nonedges per epoch. Nonedges are checked against the complete adjacency. Repeated negative samples are permitted. Positives and negatives receive equal aggregate mass. Self-pairs are excluded from the reconstruction objective. These are substantive deviations from the dense all-pair objective and change its class balance.

## Hard samples
Algorithm 1 initializes beta_i uniformly between 0.5 and 1. Eq. (11) divides beta_i by gamma when argmax(q_i) equals argmax(p_i), otherwise multiplies by gamma. Thus difficulty is disagreement between two *cluster* assignments, not a known classification error and not necessarily minority membership. Beta is floored at 0.5. The implementation assumes gamma=1.01 (the paper only specifies gamma>1) and caps beta at 100 to avoid runaway growth. These choices are saved in every checkpoint. Weights are detached, updated once per epoch before computing losses, and applied to classification, node KL and pairwise reconstruction with (beta_i+beta_j)/2. The cap is monitored in histories. The no-hard ablation sets beta=1 everywhere, retaining alpha class weights.

## Joint optimization and numerical scale
The total is L_classification + L_clustering + L_reconstruction. Each component is a mean over its own samples, rather than the paper's unnormalized sums. No extra tuned coefficients are used. This avoids a quadratic reconstruction term overwhelming sparse supervision but changes relative objective scale. Adam uses lr=0.002, weight_decay=0.03, betas=(0.9,0.999), eps=1e-8. The paper specifies 1000 maximum epochs and early stopping of 200, and 10 repetitions. CLI defaults match those epoch limits. Validation macro F1 is our explicit checkpoint/early-stop criterion; the paper does not specify it. Our baseline shares the INCM architecture and hyperparameters for a controlled local comparison; the paper instead uses baseline-specific original settings. Run logs, rather than these defaults, establish which budgets actually ran.

## HSKDM (analyzed, not implemented)
Section 4 jointly trains multiple GNN-INCM networks. Equations (18)/(19) align predictive distributions using teacher-side hard weights and directed KL divergence. Equations (20)/(21) align embeddings using weighted triplets, with anchor=positive as the stated simplification. The combined objectives add both alignment terms to each model's INCM loss. Multiple models can then vote; prediction averaging alone would not reproduce this training procedure. Margin, negative selection, teacher detachment and update sequencing need explicit resolution. HSKDM is intentionally not implemented or claimed as completed. Table 7 reports three-model ensemble F1/MCC: Cora 0.7588/0.7345; CiteSeer 0.6542/0.6354; PubMed 0.7744/0.6552.

## Original quantitative results (Table 2)
Values below are mean +/- SD over ten paper trials.
| Dataset | Model | F1 | MCC |
|---|---|---|---|
| Cora | GCN | 0.6861 +/- 0.0023 | 0.6146 +/- 0.0014 |
| Cora | GNN-INCM | 0.7508 +/- 0.0045 | 0.7237 +/- 0.0051 |
| CiteSeer | GCN | 0.6158 +/- 0.0029 | 0.5549 +/- 0.0014 |
| CiteSeer | GNN-INCM | 0.6490 +/- 0.0048 | 0.6274 +/- 0.0036 |
| PubMed | GCN | 0.7524 +/- 0.0023 | 0.6265 +/- 0.0038 |
| PubMed | GNN-INCM | 0.7704 +/- 0.0039 | 0.6493 +/- 0.0059 |

All three ablation rows for all datasets are transcribed in main.py's PAPER constant and exported to comparison tables. no_cluster maps to GNN-INCM_g; no_reconstruction to GNN-INCM_e; no_hard to GNN-INCM_~h.

## Metric ambiguity and interpretation
The paper defines binary F1/MCC formulas on multiclass datasets but does not identify the multiclass averaging convention. We explicitly report macro precision/recall/F1 across all K classes, accuracy, standard multiclass MCC, and per-class metrics. The F1 table juxtaposes our macro F1 with their reported F1; arithmetic differences do not prove like-for-like replication. Paper SDs use ten runs, our SDs are sample SDs over the available seeds and are undefined for a single run. No significance or equality claim is made from three seeds.
