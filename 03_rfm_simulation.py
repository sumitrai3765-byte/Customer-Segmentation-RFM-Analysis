"""
Step 3: RFM Analysis (Simulated)
- This dataset has no transaction dates/amounts, so we SIMULATE RFM
  from available fields — a common real-world approach when full
  transactional data is unavailable.
- Recency  → derived from Spending Score (higher score = more recent activity)
- Frequency → derived from Age proxy (younger, active shoppers visit more)
- Monetary  → derived from Annual Income * Spending Score interaction
- Score each 1-5, sum to RFM Total, classify into 4 tiers
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Load ────────────────────────────────────────────────────────────────────────
df = pd.read_csv("data/customers_clustered.csv")

print("=" * 55)
print("  STEP 3: RFM SCORING & TIER CLASSIFICATION")
print("=" * 55)

# ── Simulate RFM proxies ────────────────────────────────────────────────────────
np.random.seed(42)

# Recency: higher spending score → shopped more recently (score 1-5)
df["R_raw"] = df["Spending Score (1-100)"]

# Frequency: proxy from age (younger=more frequent) + small random noise
df["F_raw"] = (100 - df["Age"]) + np.random.normal(0, 5, len(df))
df["F_raw"] = df["F_raw"].clip(0)

# Monetary: income × spending interaction, scaled
df["M_raw"] = (df["Annual Income (k$)"] * df["Spending Score (1-100)"]) / 100

# ── Assign quintile scores (5=best) ────────────────────────────────────────────
def score_quintile(series, ascending=True):
    """Score into 1-5 using quintile cuts. ascending=True: higher raw = higher score."""
    labels = [1, 2, 3, 4, 5] if ascending else [5, 4, 3, 2, 1]
    return pd.qcut(series.rank(method="first"), q=5, labels=labels).astype(int)

df["R_score"] = score_quintile(df["R_raw"], ascending=True)
df["F_score"] = score_quintile(df["F_raw"], ascending=True)
df["M_score"] = score_quintile(df["M_raw"], ascending=True)

df["RFM_Total"] = df["R_score"] + df["F_score"] + df["M_score"]

# ── Tier Classification ─────────────────────────────────────────────────────────
def classify_rfm(score):
    if score >= 13:
        return "Champions"
    elif score >= 10:
        return "Loyal Customers"
    elif score >= 7:
        return "At-Risk"
    else:
        return "Lost / Inactive"

df["RFM_Tier"] = df["RFM_Total"].apply(classify_rfm)

# ── Print Results ───────────────────────────────────────────────────────────────
print("\n📌 RFM Score Distribution:")
print(df["RFM_Total"].describe().round(2))

print("\n📌 RFM Tier Counts:")
tier_summary = df.groupby("RFM_Tier").agg(
    Count=("CustomerID", "count"),
    Avg_R=("R_score", "mean"),
    Avg_F=("F_score", "mean"),
    Avg_M=("M_score", "mean"),
    Avg_Income=("Annual Income (k$)", "mean"),
    Avg_Spending=("Spending Score (1-100)", "mean")
).round(2)
print(tier_summary)

champions = df[df["RFM_Tier"] == "Champions"]
pct = len(champions) / len(df) * 100
print(f"\n🏆 Champions: {len(champions)} customers ({pct:.1f}% of total)")

# ── Visualizations ──────────────────────────────────────────────────────────────
TIER_COLORS = {
    "Champions": "#2ecc71",
    "Loyal Customers": "#3498db",
    "At-Risk": "#e67e22",
    "Lost / Inactive": "#e74c3c"
}

fig = plt.figure(figsize=(16, 12))
fig.suptitle("RFM Analysis — Customer Tier Classification", fontsize=15, fontweight="bold")
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.4)

# 1. Tier bar chart
ax1 = fig.add_subplot(gs[0, 0])
counts = df["RFM_Tier"].value_counts()
bars = ax1.bar(counts.index, counts.values,
               color=[TIER_COLORS[t] for t in counts.index], edgecolor="white")
ax1.set_title("Customers per RFM Tier")
ax1.set_ylabel("Count")
ax1.set_xticklabels(counts.index, rotation=20, ha="right", fontsize=8)
for bar, val in zip(bars, counts.values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, str(val),
             ha="center", va="bottom", fontsize=9, fontweight="bold")

# 2. RFM score histogram
ax2 = fig.add_subplot(gs[0, 1])
ax2.hist(df["RFM_Total"], bins=range(3, 17), color="#A8DADC", edgecolor="white", align="left")
ax2.axvline(x=13, color="#2ecc71", linestyle="--", label="Champions (≥13)")
ax2.axvline(x=10, color="#3498db", linestyle="--", label="Loyal (≥10)")
ax2.axvline(x=7, color="#e67e22", linestyle="--", label="At-Risk (≥7)")
ax2.set_title("RFM Score Distribution")
ax2.set_xlabel("RFM Total Score")
ax2.set_ylabel("Count")
ax2.legend(fontsize=7)

# 3. Scatter Income vs Spending colored by RFM Tier
ax3 = fig.add_subplot(gs[0, 2])
for tier, grp in df.groupby("RFM_Tier"):
    ax3.scatter(grp["Annual Income (k$)"], grp["Spending Score (1-100)"],
                c=TIER_COLORS[tier], label=tier, alpha=0.8, s=50, edgecolors="white")
ax3.set_title("Income vs Spending by RFM Tier")
ax3.set_xlabel("Annual Income (k$)")
ax3.set_ylabel("Spending Score")
ax3.legend(fontsize=7)
ax3.grid(alpha=0.3)

# 4. R / F / M average scores per tier
ax4 = fig.add_subplot(gs[1, :2])
rfm_means = df.groupby("RFM_Tier")[["R_score", "F_score", "M_score"]].mean().round(2)
x = np.arange(len(rfm_means))
width = 0.25
ax4.bar(x - width, rfm_means["R_score"], width, label="Recency", color="#6EC6CA")
ax4.bar(x, rfm_means["F_score"], width, label="Frequency", color="#FFB347")
ax4.bar(x + width, rfm_means["M_score"], width, label="Monetary", color="#98D8C8")
ax4.set_xticks(x)
ax4.set_xticklabels(rfm_means.index, rotation=15, ha="right")
ax4.set_title("Avg R / F / M Scores per Tier")
ax4.set_ylabel("Score (1-5)")
ax4.legend()
ax4.grid(axis="y", alpha=0.3)

# 5. Pie chart — revenue share proxy (M_raw sum)
ax5 = fig.add_subplot(gs[1, 2])
rev_share = df.groupby("RFM_Tier")["M_raw"].sum()
ax5.pie(rev_share, labels=rev_share.index, autopct="%1.1f%%",
        colors=[TIER_COLORS[t] for t in rev_share.index], startangle=90)
ax5.set_title("Revenue Share by RFM Tier")

plt.savefig("output/03_rfm.png", dpi=150, bbox_inches="tight")
print("\n✅ RFM charts saved → output/03_rfm.png")

# Save final output
df.to_csv("data/customers_rfm_final.csv", index=False)
print("✅ Final data saved → data/customers_rfm_final.csv")

# Print business insights
print("\n" + "=" * 55)
print("  💡 KEY BUSINESS INSIGHTS")
print("=" * 55)
champ_rev = df[df["RFM_Tier"] == "Champions"]["M_raw"].sum()
total_rev = df["M_raw"].sum()
print(f"  🏆 Champions ({pct:.0f}% of customers) → "
      f"{champ_rev/total_rev*100:.0f}% of revenue")
lost = df[df["RFM_Tier"] == "Lost / Inactive"]
print(f"  ⚠️  Lost/Inactive: {len(lost)} customers — re-engagement campaigns needed")
loyal = df[df["RFM_Tier"] == "Loyal Customers"]
print(f"  🔵 Loyal Customers: {len(loyal)} — upsell/cross-sell opportunity")
print(f"  📌 At-Risk: {len(df[df['RFM_Tier']=='At-Risk'])} — retention focus")
