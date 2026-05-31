"""
Step 1: Exploratory Data Analysis
- Understand the data shape, distributions, and relationships
- Identify patterns before modeling
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

# ── Load Data ──────────────────────────────────────────────────────────────────
df = pd.read_csv("data/Mall_Customers.csv")

print("=" * 55)
print("  CUSTOMER SEGMENTATION — EXPLORATORY DATA ANALYSIS")
print("=" * 55)

print("\n📌 Dataset Shape:", df.shape)
print("\n📌 First 5 rows:")
print(df.head())

print("\n📌 Data Types:")
print(df.dtypes)

print("\n📌 Missing Values:")
print(df.isnull().sum())

print("\n📌 Summary Statistics:")
print(df.describe().round(2))

print("\n📌 Gender Distribution:")
print(df["Gender"].value_counts())

# ── Visualizations ──────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 12))
fig.suptitle("Mall Customers — Exploratory Data Analysis", fontsize=16, fontweight="bold", y=0.98)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

# 1. Gender pie chart
ax1 = fig.add_subplot(gs[0, 0])
gender_counts = df["Gender"].value_counts()
ax1.pie(gender_counts, labels=gender_counts.index, autopct="%1.1f%%",
        colors=["#6EC6CA", "#FF9A8B"], startangle=90)
ax1.set_title("Gender Distribution")

# 2. Age distribution
ax2 = fig.add_subplot(gs[0, 1])
ax2.hist(df["Age"], bins=20, color="#A8DADC", edgecolor="white")
ax2.set_title("Age Distribution")
ax2.set_xlabel("Age")
ax2.set_ylabel("Count")

# 3. Annual Income distribution
ax3 = fig.add_subplot(gs[0, 2])
ax3.hist(df["Annual Income (k$)"], bins=20, color="#FFB347", edgecolor="white")
ax3.set_title("Annual Income Distribution")
ax3.set_xlabel("Income (k$)")
ax3.set_ylabel("Count")

# 4. Spending Score distribution
ax4 = fig.add_subplot(gs[1, 0])
ax4.hist(df["Spending Score (1-100)"], bins=20, color="#98D8C8", edgecolor="white")
ax4.set_title("Spending Score Distribution")
ax4.set_xlabel("Spending Score")
ax4.set_ylabel("Count")

# 5. Income vs Spending Score scatter
ax5 = fig.add_subplot(gs[1, 1])
colors = {"Male": "#6EC6CA", "Female": "#FF9A8B"}
for gender, group in df.groupby("Gender"):
    ax5.scatter(group["Annual Income (k$)"], group["Spending Score (1-100)"],
                c=colors[gender], label=gender, alpha=0.7, s=40)
ax5.set_title("Income vs Spending Score")
ax5.set_xlabel("Annual Income (k$)")
ax5.set_ylabel("Spending Score")
ax5.legend()

# 6. Correlation heatmap
ax6 = fig.add_subplot(gs[1, 2])
corr = df[["Age", "Annual Income (k$)", "Spending Score (1-100)"]].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax6,
            linewidths=0.5, square=True)
ax6.set_title("Correlation Heatmap")

plt.savefig("output/01_eda.png", dpi=150, bbox_inches="tight")
print("\n✅ EDA chart saved → output/01_eda.png")
