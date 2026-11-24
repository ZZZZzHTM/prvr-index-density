# Index density in partially relevant video retrieval

This study asks how many video vectors a PRVR index needs to store and which videos change rank when a dense frame index is compressed. Retrained branch controls separate the effects of encoding frames from those of storing individual frame outputs. Fixed-checkpoint comparisons trace changes in retrieval quality, per-video ranks, storage and scoring as the index and its allocation across videos change.

## Supplementary material

[SUPPLEMENTARY.md](SUPPLEMENTARY.md) has the same content as the supplementary PDF, [Supplementary_Material.pdf](paper/supplement/Supplementary_Material.pdf).

| Section | Contents |
|---|---|
| 1–6 | Robustness checks: fusion weights for the retrained controls, per-recall changes, Euclidean Ward, padding, held-out checkpoint selection and expanded retrieval corpora |
| 7 | All four effective-rank quartiles for every GMMFormerV2 seed, with the lowest-minus-highest contrast and rank correlations (Figure S1) |
| 8 | The branch budget split for each dataset separately (Figure S2) |
| 9 | The three links of the TVR chain for every seed (Figure S3) |
| 10 | Ward-8, uniform pooling and k-means on DreamPRVR and HLFormer |
| 11 | Figure 6 of the paper at a larger size (Figure S4) |

## Check the paper values

From the repository root, run the following command with Python 3.9 or later. It uses the standard library and the result files in this repository.

```sh
python paper/build_values.py
```

The expected output is `173/173 match`. A separate line checks the introduction's 229 GiB projection for one million uncompressed fp32 entries. Section 6 uses the measured 14.46 GiB allocation for one million compressed int8 entries. Both calculations use unrounded inputs.

Each row of [values.csv](paper/values.csv) gives the paper location, displayed value, source files and JSON pointer or formula. A `V` identifier denotes a row in that table. An `A` identifier denotes an intermediate field or calculation, defined in [build_values.py](paper/build_values.py). Both use full-precision values until formatting the final display.

The five plotting programs in [paper/figures](paper/figures) read the exported JSON. For example, this command renders Figure 4 to a directory of your choice.

```sh
python paper/figures/fig4_budget.py --output /path/to/plots
```

The supplementary figures use the same options. This command renders Figures S1–S4 as PDF and PNG files.

```sh
python paper/supplement/figures.py --output /path/to/plots
```

## Repository contents

| Directory | Contents |
|---|---|
| [results](results) | Summary results, per-seed records and supporting measurements. Per-seed filenames use decimal seed values, for example `_s20261024.json`. |
| [scripts](scripts) | Evaluation, compression, scoring, statistical analysis and experiment-specific retraining code. |
| [configs](configs) | Experiment settings and preregistrations, including the decision and stopping rules. |
| [paper](paper) | Numerical cross-reference, verification program and plotting code. [paper/supplement](paper/supplement) holds the supplementary PDF, the code for Figures S1–S4 and the rendered figures. |
| [gmmformerv2_patch](gmmformerv2_patch) | Changes to the upstream GMMFormerV2 implementation. `materialize.py` reads six removed configuration lines from the pinned official Git source and emits a complete patch for `git apply`. The command and revision are in [UPSTREAM.md](gmmformerv2_patch/UPSTREAM.md). |
| [SUPPLEMENTARY.md](SUPPLEMENTARY.md) | Supplementary results in eleven sections. |

The verification and plotting programs operate directly on the included results. Running training or evaluation from feature inputs also requires the upstream implementation, the relevant dataset and checkpoint, and paths configured for those assets. Some analysis scripts consume intermediate score matrices or per-query rankings produced by those runs. The release is designed for inspection and numerical verification.

Separate retraining code under `scripts/rq1_branch_controls/` and `scripts/rq1_clip_only_third_branch/` implements the widened feed-forward layers, the one-, two- and three-clip controls, sparse-frame sampling and pooling of trained frame outputs. In the score-fusion control, each dataset has three pairs of independently trained Clip-Only seeds. The two member score matrices are aligned by query and video ID, averaged elementwise with equal weights, and ranked once. The dataset means for the difference from GMMFormerV2 are then averaged with equal weights. The per-pair and per-dataset results appear in [the score-fusion results](results/rq1_score_fusion/RESULTS.json).

## Where each paper item is supported

Experiment names below are paths relative to [results](results). Evaluation code and configurations use the same experiment names under [scripts](scripts) and [configs](configs) where those files are available. Each plotting file is located in [paper/figures](paper/figures). Exact JSON fields for displayed values are listed in [values.csv](paper/values.csv).

| Paper item | Experiment directories and code |
|---|---|
| Table 1. Benchmarks and methods | `index_density_stats`, `budget_split`, `rq2_seeds` and `protocol/youcook2_centered` contain corpus and effective-rank statistics. `generality_six_methods` records available model and dataset combinations. |
| Table 2. Branch and granularity controls | `rq1_branch_controls`, `rq1_clip_only_third_branch`, `rq1_ward_seeds`, `generality_six_methods`, `fusion_weights` and `protocol/param_count`. The control that holds both parameter count and stored slots fixed is under `rq1_slot_matched`. |
| Table 3. Design defaults | `rq1_branch_controls`, `rq1_clip_only_third_branch`, `rq1_ward_seeds`, `budget_split`, `storage_int8`, `storage_int8_seeds` and `storage_pq_ivf`. The retention threshold uses the same underlying measurements as Figure 3. The deployment measurements in the accompanying text use `storage_scale`, `protocol/index_footprint` and `serving_latency`. |
| Figure 1. Retrieval schematic | The paper illustrates encoding, stored slots and maximum-similarity scoring. Slot counts are in `protocol/param_count`; `index_density_stats` contains the frame budgets. |
| Figure 2. Quality versus storage | `index_density_stats`. [Figure 2 script](paper/figures/fig2_quality.py) |
| Figure 3. Compression by model and dataset | `generality_six_methods` and `protocol/youcook2_centered`. The uncompressed scores used for retention also come from `index_density_stats`, `reproduce_released_checkpoints`, `released_gmmformerv2` and `trained_dreamprvr`. [Figure 3 script](paper/figures/fig3_generality.py) |
| Figure 4. Splitting the index budget between branches | `budget_split/FRONTIER.json`. [Figure 4 script](paper/figures/fig4_budget.py) |
| Figure 5. Frame usage and redundancy | `rq2_usage_redundancy`. [Figure 5 script](paper/figures/fig5_redundancy.py) |
| Figure 6. Effective-rank quartiles | `rq2_decomposition`, `rq2_group_ci`. The budget-allocation experiment is in `budget_per_video`. [Figure 6 script](paper/figures/fig6_quartiles.py) |
| Supplement. Matched fusion weights | `rq1_branch_controls` and `check_matched_weights`. |
| Supplement. Standard Euclidean Ward | `check_euclidean_ward`, `rq1_score_fusion` and `check_euclidean_ward_rq2`. |
| Supplement. Padding | `rq2_tvr_frame_branch`. |
| Supplement. Held-out checkpoint selection | `check_heldout_selection`. |
| Supplement. Expanded retrieval corpus | `check_corpus_expansion`. |
| Supplement. Quartiles across seeds | `rq2_seeds` and `rq2_group_ci`. [Figures S1–S4 script](paper/supplement/figures.py) |
| Supplement. Budget split per dataset | `budget_split/FRONTIER.json`. |
| Supplement. TVR chain links | `rq2_tvr_chain`. |
| Supplement. Compression rules on DreamPRVR and HLFormer | `check_crossmodel_compressors`. |
| Supplement. Larger Figure 6 | `rq2_decomposition`, `rq2_group_ci` and `budget_per_video`. |
| Additional checks in Sections 4 and 5 | `rq1_dreamprvr_controls` covers DreamPRVR branch controls. `check_crossmodel_compressors` compares three compression rules for DreamPRVR and HLFormer. `rq2_crossmodel`, `rq2_seeds`, `rq2_tvr_chain` and `rq2_example_query` provide the cross-model, additional-seed, TVR and example-query analyses. |

## Data, features and model weights

For TVR, ActivityNet Captions and Charades-STA, the official release of MS-SL provides the pre-extracted PRVR features and text splits in each dataset's “Required Data” section. [MS-SL dataset instructions](https://github.com/HuiGuanLab/ms-sl)

For YouCook2-PRVR, the official release of CaptAin links the feature and caption package. This study centers features with means estimated on the training split. [CaptAin data instructions](https://github.com/sYYmmEtra/CaptAin-PRVR/blob/main/train/README.md#data-dependencies) · [YouCook2-PRVR files](https://huggingface.co/datasets/e-gf/CaptAin_depends/tree/main/youcook2)

Figure 3 identifies results from our training runs and results from released checkpoints. Use the links below to obtain the original published checkpoints for the latter. For SDM, the authors supply weights for TVR and ActivityNet Captions. The Charades-STA cell is marked unavailable.

| Released checkpoint | Official source |
|---|---|
| MS-SL | [Official release of MS-SL](https://github.com/HuiGuanLab/ms-sl) |
| GMMFormerV2 | [Official release of GMMFormerV2](https://github.com/huangmozhi9527/GMMFormer_v2#trained-models) |
| HLFormer | [Official release of HLFormer](https://github.com/lijun2005/ICCV25-HLFormer#32-retrieval-performance) |
| DreamPRVR | [Official release of DreamPRVR](https://github.com/lijun2005/CVPR26-DreamPRVR#32-retrieval-performance) |
| Holmes | [Official release of Holmes](https://github.com/lijun2005/ICML26-Holmes#32-retrieval-performance) |
| SDM | [Official release of SDM](https://github.com/junwoojinjin/SDM-PRVR#trained-models) |

Our retrained checkpoints will be released upon acceptance.

## Paths in configurations and results

Absolute paths recorded during the original runs are replaced by placeholders. Set them to local locations before running any script.

| Placeholder | Meaning |
|---|---|
| `<DATA>` | Dataset files, including pre-extracted features, text splits and cached inputs |
| `<RELEASED>` | Locations in official model releases, including checkpoints, code and data directories |
| `<WORK>` | Experiment working files, including input checkpoints, configurations, intermediate files and run outputs |
| `<REPO>` | Root of the experiment code |
| `<ENV>` | Python environment |
| `<LOCAL>` | Other machine-specific locations |
| `<REVISION>` | Internal revision identifiers of the original runs |
| `<DEVICE_ID>` | Hardware device identifier recorded during a run |
| `<CONTROL_GROUP>` | Operating system control group or resource scope for a run |
| `<DEVICE_DIAGNOSTICS>` | Device status output collected during a run |
| `<REPOSITORY>` | Branch name recorded in run metadata |
| `<PRIVATE_ID>` | Part of a recorded video identifier |
| `<dataset>`, `<seed>`, `<method>`, `<weights>`, `<level>`, `<row_id>` | Variable fields in example paths and result keys, standing for a dataset, seed, method, fusion weights, analysis level or row identifier |
| `<UPSTREAM_LINE path line>` | Line reference resolved by the patch helper from the pinned upstream source |

Placeholders do not affect `paper/build_values.py`, which reads only the included results.

## Environment

The reference environment used Python 3.11.15, PyTorch 2.11.0 with CUDA 12.8, NumPy 1.26.4, SciPy 1.17.1, h5py 3.16.0, PyYAML 6.0.3, tqdm 4.70.1, easydict 1.13, ipdb 0.13.13, Matplotlib 3.11.2 and seaborn 0.13.2. Install a PyTorch build compatible with the target system for model evaluation. The additional dependencies for running the individual analyses are listed in [requirements.txt](requirements.txt). The numerical verification program runs without these packages.


