"""
Step 4: Final Summary Dashboard
- Combines clustering + RFM insights into one executive-level visual
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

df = pd.read_csv("data/customers_rfm_final.csv")

TIER_COLORS = {
    "Champions": "#2ecc71",
    "Loyal Customers": "#3498db",
    "At-Risk": "#e67e22",
    "Lost / Inactive": "#e74c3c"
}
SEG_COLORS = {
    "Champions": "#2ecc71",
    "Impulsive Buyers": "#e74c3c",
    "Conservative": "#3498db",
    "At-Risk": "#e67e22",
    "Standard": "#9b59b6"
}

fig = plt.figure(figsize=(18, 10))
fig.patch.set_facecolor("#1a1a2e")
fig.suptitle("Customer Segmentation & RFM Analysis — Executive Dashboard",
             fontsize=16, fontweight="bold", color="white", y=0.98)

gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.35)

# Shared style
TEXT_COLOR = "white"

def style_ax(ax, title):
    ax.set_facecolor("#16213e")
    ax.set_title(title, color=TEXT_COLOR, fontsize=10, fontweight="bold", pad=8)
    ax.tick_params(colors=TEXT_COLOR, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#0f3460")

# 1. KPI tiles (top row, col 0-3 summary)
kpis = [
    ("Total Customers", f"{len(df):,}", "#A8DADC"),
    ("Champions", f"{(df['RFM_Tier']=='Champions').sum()} ({(df['RFM_Tier']=='Champions').mean()*100:.0f}%)", "#2ecc71"),
    ("Avg Income", f"${df['Annual Income (k$)'].mean():.0f}k", "#FFB347"),
    ("Avg Spend Score", f"{df['Spending Score (1-100)'].mean():.1f}/100", "#98D8C8"),
]
for idx, (label, value, color) in enumerate(kpis):
    ax = fig.add_subplot(gs[0, idx])
    ax.set_facecolor("#0f3460")
    for spine in ax.spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(2)
    ax.text(0.5, 0.65, value, transform=ax.transAxes, fontsize=20, fontweight="bold",
            color=color, ha="center", va="center")
    ax.text(0.5, 0.25, label, transform=ax.transAxes, fontsize=9,
            color="lightgray", ha="center", va="center")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("")

# 2. Cluster scatter (bottom left)
ax5 = fig.add_subplot(gs[1, 0:2])
style_ax(ax5, "K-Means Segments: Income vs Spending Score")
for seg, grp in df.groupby("Segment"):
    ax5.scatter(grp["Annual Income (k$)"], grp["Spending Score (1-100)"],
                c=SEG_COLORS.get(seg, "gray"), label=seg, alpha=0.8, s=45, edgecolors="white", linewidth=0.3)
ax5.set_xlabel("Annual Income (k$)", color=TEXT_COLOR, fontsize=8)
ax5.set_ylabel("Spending Score", color=TEXT_COLOR, fontsize=8)
ax5.legend(fontsize=7, labelcolor="white", facecolor="#1a1a2e", edgecolor="#0f3460")
ax5.grid(alpha=0.15, color="white")

# 3. RFM Tier distribution (bottom mid-left)
ax6 = fig.add_subplot(gs[1, 2])
style_ax(ax6, "RFM Tier Distribution")
tier_counts = df["RFM_Tier"].value_counts()
bars = ax6.barh(tier_counts.index, tier_counts.values,
                color=[TIER_COLORS[t] for t in tier_counts.index], edgecolor="none")
for bar, val in zip(bars, tier_counts.values):
    ax6.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, str(val),
             va="center", fontsize=8, color=TEXT_COLOR)
ax6.set_xlabel("Count", color=TEXT_COLOR, fontsize=8)
ax6.invert_yaxis()
ax6.tick_params(labelsize=8)

# 4. Revenue share donut (bottom right)
ax7 = fig.add_subplot(gs[1, 3])
ax7.set_facecolor("#16213e")
ax7.set_title("Revenue Share by Tier", color=TEXT_COLOR, fontsize=10, fontweight="bold", pad=8)
rev_share = df.groupby("RFM_Tier")["M_raw"].sum()
wedges, texts, autotexts = ax7.pie(
    rev_share, labels=rev_share.index, autopct="%1.0f%%",
    colors=[TIER_COLORS[t] for t in rev_share.index],
    startangle=90, wedgeprops=dict(width=0.55)
)
for t in texts:
    t.set_color(TEXT_COLOR)
    t.set_fontsize(8)
for a in autotexts:
    a.set_color("white")
    a.set_fontsize(8)
    a.set_fontweight("bold")

plt.savefig("output/04_dashboard.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print("✅ Dashboard saved → output/04_dashboard.png")
