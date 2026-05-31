"""
Step 2: K-Means Customer Segmentation
- Use Elbow Method to find optimal k
- Cluster on Annual Income + Spending Score
- Label and interpret each cluster
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ── Load & Prepare ──────────────────────────────────────────────────────────────
df = pd.read_csv("data/Mall_Customers.csv")

X = df[["Annual Income (k$)", "Spending Score (1-100)"]].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── Elbow Method ───────────────────────────────────────────────────────────────
print("=" * 55)
print("  STEP 2: K-MEANS CLUSTERING")
print("=" * 55)

inertias = []
K_range = range(1, 11)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)

print("\n📌 Inertia values (for Elbow plot):")
for k, i in zip(K_range, inertias):
    print(f"  k={k}: {i:.2f}")

# ── Fit Final Model (k=5) ───────────────────────────────────────────────────────
k_optimal = 5
km_final = KMeans(n_clusters=k_optimal, random_state=42, n_init=10)
df["Cluster"] = km_final.fit_predict(X_scaled)

# ── Decode cluster labels ───────────────────────────────────────────────────────
# Assign labels by cluster center position (income, spending)
centers_original = scaler.inverse_transform(km_final.cluster_centers_)
cluster_info = {}
for i, (inc, spd) in enumerate(centers_original):
    if spd >= 60 and inc >= 60:
        label = "High Income, High Spender"
        tier = "Champions"
    elif spd >= 60 and inc < 60:
        label = "Low Income, High Spender"
        tier = "Impulsive Buyers"
    elif spd < 40 and inc >= 60:
        label = "High Income, Low Spender"
        tier = "Conservative"
    elif spd < 40 and inc < 60:
        label = "Low Income, Low Spender"
        tier = "At-Risk"
    else:
        label = "Average Income & Spending"
        tier = "Standard"
    cluster_info[i] = {"label": label, "tier": tier, "income": inc, "spending": spd}

df["Segment"] = df["Cluster"].map(lambda c: cluster_info[c]["tier"])
df["SegmentLabel"] = df["Cluster"].map(lambda c: cluster_info[c]["label"])

# ── Print cluster summary ───────────────────────────────────────────────────────
print("\n📌 Cluster Summary:")
summary = df.groupby("Segment").agg(
    Count=("CustomerID", "count"),
    Avg_Income=("Annual Income (k$)", "mean"),
    Avg_Spending=("Spending Score (1-100)", "mean"),
    Avg_Age=("Age", "mean")
).round(2)
print(summary)
print("\n📌 % of customers per segment:")
print((df["Segment"].value_counts(normalize=True) * 100).round(1).to_string())

# ── Plots ───────────────────────────────────────────────────────────────────────
COLORS = {
    "Champions": "#2ecc71",
    "Impulsive Buyers": "#e74c3c",
    "Conservative": "#3498db",
    "At-Risk": "#e67e22",
    "Standard": "#9b59b6"
}

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("K-Means Customer Segmentation (k=5)", fontsize=14, fontweight="bold")

# Left: Elbow curve
axes[0].plot(list(K_range), inertias, "bo-", linewidth=2, markersize=6)
axes[0].axvline(x=k_optimal, color="red", linestyle="--", alpha=0.7, label=f"Optimal k={k_optimal}")
axes[0].set_title("Elbow Method — Optimal k")
axes[0].set_xlabel("Number of Clusters (k)")
axes[0].set_ylabel("Inertia (WCSS)")
axes[0].legend()
axes[0].grid(alpha=0.3)

# Right: Cluster scatter
for seg, grp in df.groupby("Segment"):
    axes[1].scatter(grp["Annual Income (k$)"], grp["Spending Score (1-100)"],
                    c=COLORS.get(seg, "gray"), label=seg, alpha=0.8, s=60, edgecolors="white", linewidth=0.5)

# Plot cluster centers
for i, (inc, spd) in enumerate(centers_original):
    axes[1].scatter(inc, spd, c="black", marker="X", s=200, zorder=5)

axes[1].set_title("Customer Clusters — Income vs Spending")
axes[1].set_xlabel("Annual Income (k$)")
axes[1].set_ylabel("Spending Score (1-100)")
axes[1].legend(loc="upper left", fontsize=9)
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("output/02_clusters.png", dpi=150, bbox_inches="tight")
print("\n✅ Cluster chart saved → output/02_clusters.png")

# Save enriched data for next step
df.to_csv("data/customers_clustered.csv", index=False)
print("✅ Clustered data saved → data/customers_clustered.csv")
