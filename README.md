# Recommender System — Collaborative Filtering & CFGAN

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-FF6F00?logo=tensorflow&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

A comparative study of **traditional collaborative filtering** (user-user similarity) and a **GAN-based approach** (CFGAN) for movie recommendation. Built as part of a Recommender Systems design course.

---

## Overview

This project explores two fundamentally different approaches to collaborative filtering:

| Approach | Notebook | Method |
|----------|----------|--------|
| **Traditional CF** | `CollaborativeFiltering.ipynb` | User-user similarity matrices (MSD, Pearson, Cosine) → k-NN prediction |
| **CFGAN** | `CFGAN_model.ipynb` | Generative Adversarial Network that learns to produce realistic user-item interaction vectors |

The traditional approach is interpretable and works well on small datasets, while CFGAN can potentially capture nonlinear patterns that similarity metrics miss.

---

## Project Structure

```
recsys-cfgan/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── data/
│   └── rating_sparse.csv          # Sparse user-item ratings (20 users × 50 items)
├── notebooks/
│   ├── CollaborativeFiltering.ipynb   # Traditional CF with 3 similarity metrics
│   └── CFGAN_model.ipynb              # GAN-based collaborative filtering
└── docs/
    └── RS_Design_Project.pdf          # Course design report
```

---

## Methodology

### Traditional Collaborative Filtering

Three user-user similarity metrics are computed and visualized as heatmaps:

- **Mean Squared Difference (MSD)** — Measures average squared rating difference over co-rated items. Transformed via `1/(1+MSD)` so higher = more similar.
- **Pearson Correlation** — Captures linear correlation between users' rating patterns, normalized to [0, 1].
- **Cosine Similarity** — Measures the angle between user rating vectors (missing ratings filled with 0).

Each metric is evaluated using **k-NN rating prediction** (k=5) with **MAE** and **RMSE**.

### CFGAN (Collaborative Filtering GAN)

Inspired by [Chae et al., 2018](https://dl.acm.org/doi/10.1145/3269206.3271743):

- **Generator**: Takes user embedding + noise → produces a synthetic item-interaction vector
- **Discriminator**: Takes user embedding + item vector → classifies as real or fake
- **Training**: Adversarial min-max game over 100 epochs with Adam optimizer

Evaluated with **Precision@K**, **Recall@K**, and **NDCG@K**.

---

## Datasets

| Dataset | Source | Size | Used In |
|---------|--------|------|---------|
| Custom sparse ratings | `data/rating_sparse.csv` | 20 users, 50 items | CollaborativeFiltering |
| MovieLens 100K | [grouplens.org](https://grouplens.org/datasets/movielens/100k/) | 943 users, 1,682 items, 100K ratings | CFGAN |

---

## Getting Started

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
git clone https://github.com/renidotsh/recsys-cfgan.git
cd recsys-cfgan
pip install -r requirements.txt
```

### Run

```bash
jupyter notebook notebooks/
```

Open either notebook and run all cells.

---

## Key Results

### Traditional CF

Three heatmaps visualize user-user similarity, plus a quantitative k-NN evaluation comparing MAE/RMSE across metrics.

### CFGAN

Training loss curves show the adversarial dynamics, and Top-K metrics evaluate recommendation quality at different cut-offs.

---

## References

- Chae, D.-K., Kang, J.-S., Kim, S.-W., & Lee, J.-T. (2018). *CFGAN: A Generic Collaborative Filtering Framework based on Generative Adversarial Networks*. CIKM '18.
- Harper, F. M. & Konstan, J. A. (2015). *The MovieLens Datasets: History and Context*. ACM TiiS.

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
