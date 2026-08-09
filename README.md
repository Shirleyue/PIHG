# PIHG: Physics-Inspired Heterogeneous Graph Neural Networks for Multi-band Radio Map Prediction

Official implementation of **PIHG**, published in *IEEE Transactions on Wireless Communications*.

> Radio maps are instrumental in optimizing wireless network performance by providing spatial
> distributions of signal power. However, predicting multi-band radio maps across diverse
> environments remains challenging, as propagation behaviors vary significantly with frequency and
> environmental complexity. PIHG incorporates physical knowledge into the learning process for
> coarse-grained radio map estimation, and further refines the prediction using a propagation-aware
> heterogeneous graph that models environment-dependent spatial dependencies among transmitters,
> receivers, and obstacles.

PIHG follows a **coarse-to-fine** paradigm:

1. **Physics-inspired coarse-grained estimation.** A UNet cascade estimates the total electric
   field under a Volume Integral Equation (VIE) consistency constraint and converts it into a
   coarse radio map.
2. **Propagation-aware graph refinement.** A heterogeneous graph built from the Spectral
   Obstruction Map (SOM) and spatial proximity is processed by PAGNet (relational graph attention)
   to produce the fine-grained radio map.

---

## Overview

```mermaid
flowchart TB
    subgraph IN["Inputs — 4 x 128 x 128"]
        direction LR
        I1["Building layout<br/>B"]
        I2["Tx positions<br/>P_tx"]
        I3["Incident field<br/>Re(E_inc), Im(E_inc)"]
    end

    subgraph CG["Stage 1 — Physics-inspired Coarse-grained Estimation"]
        direction TB
        U1["<b>UNet-1</b><br/>4ch -> 2ch"]
        ET["Total field<br/>E_tot (Re, Im)"]
        MAG["Convert to radio map<br/>abs(E_tot) -> 1ch"]
        U2["<b>UNet-2</b><br/>data-driven correction"]
        PC["Coarse radio map<br/>P_c"]
        U1 --> ET --> MAG --> U2 --> PC
    end

    subgraph FG["Stage 2 — Propagation-aware Graph Refinement"]
        direction TB
        NF["Node features<br/>coarse RSS + freq code<br/>+ node-type embedding"]
        HG["Heterogeneous graph<br/>3 node types / 6 relations<br/>16 blocks of 32 x 32"]
        PAG["<b>PAGNet</b><br/>2 x RGATConv + residual<br/>relation-aware attention"]
        PF["Fine radio map<br/>P_f"]
        NF --> PAG
        HG --> PAG
        PAG --> PF
    end

    SOM["Spectral Obstruction Map<br/>radio depth along LoS"]

    IN --> U1
    PC --> NF
    SOM --> HG

    ET -.-> LP["<b>L_p</b> physics consistency<br/>‖ (I + G·chi) E_tot − E_inc ‖²<br/><i>FFT matvec, no dense W</i>"]
    PC -.-> LD["<b>L_d</b> data consistency<br/>‖ P − P_c ‖²"]
    PF -.-> LF["<b>L_fg</b> fine-grained loss<br/>‖ P − P_f ‖²"]
    LP -.-> LCG["L_cg = L_p + lambda · L_d<br/><i>stage-1 objective</i>"]
    LD -.-> LCG

    classDef loss fill:#fff5e6,stroke:#e8a33d,color:#7a4a00;
    class LP,LD,LF,LCG loss;
```

### Where each block lives in the code

| Component in the diagram | Code |
| --- | --- |
| End-to-end model `Phy_CNN_Graph` | [cnn_rgat_modules.py](cnn_rgat_modules.py) |
| UNet-1 / UNet-2 (`MyUNet`) | [myradioUNet_modules.py](myradioUNet_modules.py) |
| PAGNet (`RGAT_HeteroNode_1`) | [cnn_rgat_modules.py](cnn_rgat_modules.py) |
| `L_p` operator `(I + G·chi)E_tot` | [phy_vie_fft.py](phy_vie_fft.py) |
| Incident field `E_inc`, contrast `chi`, dense `W` | [SpectrumNet/gen_incident_field.py](SpectrumNet/gen_incident_field.py) |
| Spectral Obstruction Map (radio depth) | [SpectrumNet/gen_radio_depth.py](SpectrumNet/gen_radio_depth.py) |
| Block splitting + node features | [SpectrumNet/gen_node_feature.py](SpectrumNet/gen_node_feature.py) |
| Edge construction / adjacency | [SpectrumNet/gen_spect_adj.py](SpectrumNet/gen_spect_adj.py), [SpectrumNet/gen_spect_adj_old.py](SpectrumNet/gen_spect_adj_old.py) |
| Node/edge typing (3 types, 6 relations) | `generate_edge_type` in [SpectrumNet/utils.py](SpectrumNet/utils.py) |
| Dataset / graph loading | [SpectrumNet/load_data.py](SpectrumNet/load_data.py) |
| Training (both stages) | [train.py](train.py) |
| Evaluation | [test.py](test.py) |

### Key design points

**Matrix-free VIE operator.** The Green-function coefficient matrix `W` depends only on pairwise
distances, and the grid is regular, so `W` is a BTTB matrix and `W @ v` is exactly a 2-D
convolution. [phy_vie_fft.py](phy_vie_fft.py) evaluates it by circulant embedding + FFT: only a
`256 x 256` complex kernel spectrum per frequency (~0.5 MB) is stored instead of a
`16384 x 16384` complex128 matrix (~4.3 GB **per frequency**). The result is identical to the dense
product up to numerical precision — run `python phy_vie_fft.py` to verify (rel. err ~1e-16 in
complex128, ~1e-7 in complex64).

**Heterogeneous graph.** Node types are `0 = obstacle (building)`, `1 = receiver`,
`2 = transmitter`; the 6 relations are the unordered type pairs
(`0-0, 0-1, 0-2, 1-1, 1-2, 2-2`). An edge exists when two nodes are close in both SOM value and
space (Eq. 17). Each `128 x 128` region is partitioned into `16` blocks of `32 x 32`, PAGNet runs
per block, and the blocks are re-assembled into the full map.

**Model size.** 5.91 M parameters total — UNet-1 2.95 M, UNet-2 2.95 M, PAGNet 4.4 K.

---

## Repository structure

```
PIHG/
├── train.py                        # training entry point (both stages)
├── test.py                         # evaluation entry point (MSE/NMSE/RMSE/SSIM/PSNR)
├── cnn_rgat_modules.py             # Phy_CNN_Graph (PIHG) + PAGNet variants
├── myradioUNet_modules.py          # UNet encoder/decoder used as UNet-1 / UNet-2
├── phy_vie_fft.py                  # matrix-free VIE operator (FFT); self-check in __main__
├── SpectrumNet/                    # dataset construction and loading
│   ├── readPng.py                  # parse radio-map PNG + metadata from the file name
│   ├── readNpz.py                  # parse building/terrain NPZ + metadata
│   ├── tx_process.py               # transmitter coordinates and power maps
│   ├── utils.py                    # path helpers, frequency/height codes, edge typing
│   ├── create_area_dataset.py      # match PNG <-> NPZ <-> Tx, filter by scenario/height
│   ├── gen_radio_depth.py          # Spectral Obstruction Map (radio depth), + data split
│   ├── gen_incident_field.py       # E_inc, chi (and optionally dense W), + data split
│   ├── gen_node_feature.py         # split maps into 32x32 blocks, build node features
│   ├── gen_spect_adj.py            # adjacency from distance + SOM difference (parallel)
│   ├── gen_spect_adj_old.py        # adjacency variants (distance / depth / building / Tx)
│   ├── comb_area_dataset.py        # merge per-scenario splits into area _{train,valid,test}.txt
│   ├── create_block_dataset.py     # block-level dataset lists
│   ├── dataset_split.py            # per-frequency subsets
│   └── load_data.py                # SpectrumDataset / SpectrumDatasetField
├── dataset/SpectrumNet/*.txt       # dataset index files (tab-separated paths)
└── results/<data>/<model>/<scen>/  # checkpoints, logs, metrics
```

---

## Requirements

Tested with Python 3.10 and CUDA 12.1 on NVIDIA GeForce RTX 3090.

| Package | Version |
| --- | --- |
| torch | 2.5.1+cu121 |
| torch-geometric | 2.6.1 |
| torchmetrics | 1.7.1 |
| numpy | 1.26.4 |
| scipy | 1.15.2 |
| numba | 0.61.2 |
| scikit-image | 0.25.2 |
| networkx | 3.2.1 |
| pandas | 2.2.3 |
| matplotlib | 3.8.4 |
| Pillow | 10.2.0 |
| natsort | 8.4.0 |
| tqdm | 4.64.1 |
| bresenham | — |

```bash
conda create -n pihg python=3.10 -y && conda activate pihg
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install torch-geometric==2.6.1
pip install torchmetrics numpy scipy numba scikit-image networkx pandas \
            matplotlib pillow natsort tqdm bresenham tensorboard
```

---

## Dataset preparation

PIHG is evaluated on the public multi-band radio map benchmark
**SpectrumNet** (Zhang *et al.*, *IEEE TCCN* 2025): 15,300 real-world building maps over 11 terrain
scenarios, `1.28 km x 1.28 km` per region at 10 m resolution (`128 x 128` grids), 5 carrier
frequencies (150 MHz, 1.5 GHz, 1.7 GHz, 3.5 GHz, 22 GHz). We use the ground-level (1.5 m) maps.
Received power is normalized to `[0, 1]`, where `0 = -120 dBm` and `1 = 60 dBm`.

The cross-dataset experiments additionally use
[RadioMapSeer](https://radiomapseer.github.io/) and
[UrbanRadio](https://github.com/UNIC-Lab/UrbanRadio3D).

### Expected raw layout

Scripts default to `/root/autodl-tmp/SpectrumNet/`; edit the paths in the `__main__` block of each
script to point at your copy.

```
/root/autodl-tmp/SpectrumNet/
├── png/<scenario>/T06C2D0054_n01_f04_ss_z00.png   # radio map (ground truth)
├── npz/T06C2D0054_n01_bdtr.npz                    # building + terrain
├── tx_info.txt                                    # transmitter metadata
├── tx/ , tx_dB/                                   # generated by tx_process.py
├── depth/                                         # generated by gen_radio_depth.py
├── incField_10.0/                                 # generated by gen_incident_field.py
├── splitRSS_32/                                   # generated by gen_node_feature.py
└── disdepthAdj_32/                                # generated by gen_spect_adj_old.py
```

### Preprocessing pipeline

```mermaid
flowchart LR
    A["tx_process.py<br/>tx/ , tx_dB/"] --> B["create_area_dataset.py<br/>scenario_N_height_0.txt"]
    B --> C["gen_radio_depth.py<br/>depth/ (SOM)"]
    C --> D["gen_incident_field.py<br/>incField_10.0/ (E_inc, chi)"]
    D --> E["gen_node_feature.py<br/>splitRSS_32/ (blocks + nodes)"]
    E --> F["gen_spect_adj_old.py<br/>disdepthAdj_32/ (edges)"]
    F --> G["comb_area_dataset.py<br/>area _train/_valid/_test.txt"]
```

Run every script **from the repository root**:

```bash
python SpectrumNet/tx_process.py            # transmitter coords + power maps
python SpectrumNet/create_area_dataset.py   # pair PNG/NPZ/Tx, filter scenario + height
python SpectrumNet/gen_radio_depth.py       # SOM / radio depth, writes the per-scenario split
python SpectrumNet/gen_incident_field.py    # E_inc + chi, appends the incField column
python SpectrumNet/gen_node_feature.py      # 32x32 blocks and node features
python SpectrumNet/gen_spect_adj_old.py     # adjacency matrices -> disdepthAdj_32/
python SpectrumNet/comb_area_dataset.py     # merge all scenarios into area _*.txt
```

> `gen_spect_adj.py` uses package-style imports and must be run as a module instead:
> `python -m SpectrumNet.gen_spect_adj`. It is the multiprocessing version of the adjacency step;
> note that its `main()` is not called from `__main__`, and it writes to `disAdj<dth>_32/` rather
> than the `disdepthAdj_32/` that the loader expects — enable `main()` and align the output
> directory name if you prefer it over `gen_spect_adj_old.py`.

### Index-file format

Each line of `dataset/SpectrumNet/*.txt` is five tab-separated absolute paths, consumed in this
order by `SpectrumDatasetField.__getitem__`:

| Column | Content |
| --- | --- |
| 0 | radio-map PNG (ground truth; frequency and height are encoded in the file name) |
| 1 | building/terrain NPZ |
| 2 | transmitter power map `.npy` |
| 3 | radio depth / SOM `.npy` |
| 4 | incident-field NPZ (`E_inc_trad_2channel`, `chi`) |

A sample yields `(inputs, rss_map, graph, chi, f_hz, W)`, where `inputs` stacks
`[building, tx, Re(E_inc), Im(E_inc)]` into `4 x 128 x 128`, and `graph` maps each block index to
its `node_obs_axis`, `node_freq`, `node_type_ids`, `edge_index`, `edge_type`, `tx_img`, `building`.

---

## Training

Training is **two-stage**, both driven by [train.py](train.py) via the `isGraph` flag at the top of
`__main__`.

### Stage 1 — physics-inspired coarse estimation (`isGraph = False`)

Trains the UNet cascade with `L_cg = L_p + lambda * L_d`. The `L_p` term needs no dense `W`: FFT
kernels for the 5 frequencies (~2.5 MB total) are built once at startup by `WKernelCache`.

```bash
python train.py     # with isGraph = False
# -> results/SpectrumNet/PIHG<lambda>/area /pretrain/{best_model.pt,train.txt,result.txt}
```

### Stage 2 — graph refinement (`isGraph = True`)

Loads the stage-1 checkpoint through `pt_path`, **freezes both UNets**, and optimizes PAGNet only
with `L_fg`. This is what accelerates convergence and stabilizes optimization.

```bash
python train.py     # with isGraph = True
# -> results/SpectrumNet/PIHG<lambda>/area /train/{best_model.pt,train.txt,result.txt}
```

Optimization defaults: AdamW (`lr = 1e-3`, `weight_decay = 1e-5`) for stage 2 and
(`lr = 1e-4`) for stage 1, 300 epochs, `SequentialLR` = 5-epoch linear warm-up followed by
`CosineAnnealingWarmRestarts(T_0 = 20, T_mult = 2, eta_min = 1e-6)`.

> **`batch_size` must be 1 for the graph stage.** The graph branch of `Phy_CNN_Graph.forward`
> indexes `out[0, 0]` and reshapes to `(1, 1, H, W)`, so it assumes a single sample per batch.

## Evaluation

```bash
python test.py
```

Reports MSE, NMSE, RMSE, SSIM and PSNR over the test set and appends them to
`results/<data_name>/<model_name>/<scen>/test.txt`. Set `train_folder_path` in `test.py` to the
directory holding the stage-2 `best_model.pt`. `fig_show_save()` renders a
*building | prediction | ground truth* panel if you want qualitative figures.

---

## Reproducing the paper configuration

The committed defaults are tuned for fast smoke runs and **do not all match the paper**. Align them
before reproducing the reported numbers:

| Setting | Paper | Committed default | Where |
| --- | --- | --- | --- |
| Data fraction | full dataset | `few_shot_ratio = 0.02` (train), `0.01` (test) | `train.py`, `test.py` |
| `lambda` (loss balance) | `1` (best, Table X) | `args['lamda'] = 0.01` | `train.py` |
| `eta_f` (frequency fading) | `20` | `beta = 10.0` | `gen_radio_depth.py` |
| `tau_d` (spatial threshold) | `45 m` = 4.5 grid cells | `dth = 4.5` / `dth = 9` | `gen_spect_adj.py` / `gen_spect_adj_old.py` |
| `tau_o` (obstruction threshold) | `3.5` | `sigma = obs_cof * 10*log10(5750)` -> `18.8` / `3.76` | `gen_spect_adj.py` / `gen_spect_adj_old.py` |

Also note that `train.py` writes to `model_name = 'PIHG' + str(args.lamda)` while `test.py` reads
from `model_name = 'PIHG'`; point `test.py` at the directory that actually holds your checkpoint.

---

## Results

Overall performance on SpectrumNet (Table II of the paper); best in **bold**:

| Method | MSE ↓ | RMSE ↓ | NMSE ↓ | PSNR ↑ |
| --- | --- | --- | --- | --- |
| RadioUNet | 0.0283 | 0.1666 | 0.5856 | 13.621 |
| AE | 0.0337 | 0.1751 | 0.8387 | 12.420 |
| PEFNet | 0.0284 | 0.1566 | 0.5599 | 13.653 |
| PMNet | 0.0281 | 0.1559 | 0.5870 | 13.646 |
| UNetDCN | 0.0284 | 0.1585 | 0.6081 | 13.407 |
| RadioFormer | 0.0474 | 0.2080 | 1.0238 | 10.874 |
| RadioDiff | 0.0428 | 0.1932 | 0.8887 | 11.793 |
| PhyRMDM | 0.0441 | 0.1921 | 0.7795 | 12.043 |
| **PIHG** | **0.0243** | **0.1459** | **0.4241** | **14.232** |
| *Improvement* | *13.49%* | *6.42%* | *24.25%* | *4.2%* |

Ablation (Table VIII); `PIL` = physics-inspired learning, `Hetero` = heterogeneity:

| Variant | MSE ↓ | RMSE ↓ | NMSE ↓ | PSNR ↑ |
| --- | --- | --- | --- | --- |
| PIHG w/o PIL | 0.0254 | 0.1500 | 0.4923 | 13.843 |
| PIHG w/o PAGNet | 0.0263 | 0.1542 | 0.5103 | 13.686 |
| PIHG w/o (PIL + PAGNet) | 0.0288 | 0.1585 | 0.5867 | 13.571 |
| PIHG w/o Hetero (GAT instead) | 0.0260 | 0.1527 | 0.4545 | 13.876 |
| **PIHG** | **0.0243** | **0.1459** | **0.4241** | **14.232** |

Per-band results, per-scenario results, cross-frequency / cross-scenario / cross-dataset
generalization, noise and transmitter-count robustness, and runtime scaling are reported in
Tables III–X and Figures 6–12 of the paper.

---

## Citation

```bibtex
@article{jiang2025pihg,
  title   = {{PIHG}: Physics-Inspired Heterogeneous Graph Neural Networks for
             Multi-band Radio Map Prediction},
  author  = {Jiang, Xinyue and Li, Tong and Xiao, Zhu and Chen, Ke and
             Tai, Andy Chi Lok and Tang, Zhuo and Li, Kenli},
  journal = {IEEE Transactions on Wireless Communications},
  year    = {2025}
}
```

## Contact

Xinyue Jiang — `xyuejiang@hnu.edu.cn` (College of Computer Science and Electronic Engineering,
Hunan University). Issues and pull requests are welcome.
