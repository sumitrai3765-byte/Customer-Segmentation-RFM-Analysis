# Customer Segmentation & RFM Analysis

**Tools:** Python · Pandas · Scikit-learn · Matplotlib · Seaborn  
**Dataset:** Mall Customers (200 records, 5 features)

---

## Project Overview

This project segments mall customers using **K-Means Clustering** and classifies them using an **RFM (Recency, Frequency, Monetary) framework** to surface actionable retention and marketing strategies.

---

## Dataset

| Column | Description |
|--------|-------------|
| CustomerID | Unique identifier |
| Gender | Male / Female |
| Age | Customer age |
| Annual Income (k$) | Annual income in thousands |
| Spending Score (1-100) | Mall-assigned activity score |

---

## Methodology

### Step 1 — Exploratory Data Analysis (`01_eda.py`)
- Distribution analysis of Age, Income, Spending Score
- Gender split
- Correlation heatmap

### Step 2 — K-Means Clustering (`02_clustering.py`)
- Feature: Annual Income + Spending Score
- Elbow Method → optimal **k = 5**
- Segments identified:
  - 🟢 **Champions** — High Income, High Spenders
  - 🔴 **Impulsive Buyers** — Low Income, High Spenders
  - 🔵 **Conservative** — High Income, Low Spenders
  - 🟠 **At-Risk** — Low Income, Low Spenders
  - 🟣 **Standard** — Average on both axes

### Step 3 — RFM Analysis (`03_rfm_simulation.py`)
Since the dataset lacks transaction history, RFM is derived as:
- **Recency** ← Spending Score proxy (higher = more recent activity)
- **Frequency** ← Age-based proxy (younger = more frequent visits)
- **Monetary** ← Income × Spending interaction

Scores assigned 1–5 via quintile cuts. Tiers:

| RFM Total | Tier |
|-----------|------|
| 13–15 | Champions |
| 10–12 | Loyal Customers |
| 7–9 | At-Risk |
| 3–6 | Lost / Inactive |

### Step 4 — Executive Dashboard (`04_dashboard.py`)
Dark-themed summary dashboard combining all insights.

---

## Key Findings

- 🏆 **Champions (~18% of customers) → ~52% of revenue**
- ⚠️ Lost/Inactive segment needs re-engagement campaigns
- 🔵 Loyal Customers — prime upsell/cross-sell target
- 🟠 At-Risk — retention campaigns (discounts, loyalty points)

---

## How to Run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
mkdir output
python 01_eda.py
python 02_clustering.py
python 03_rfm_simulation.py
python 04_dashboard.py
```

---

## Files

```
customer-segmentation/
├── data/
│   ├── Mall_Customers.csv          ← raw input
│   ├── customers_clustered.csv     ← after Step 2
│   └── customers_rfm_final.csv     ← after Step 3
├── output/
│   ├── 01_eda.png
│   ├── 02_clusters.png
│   ├── 03_rfm.png
│   └── 04_dashboard.png
├── 01_eda.py
├── 02_clustering.py
├── 03_rfm_simulation.py
├── 04_dashboard.py
└── README.md
```
