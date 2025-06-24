import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from fpdf import FPDF
from ast import literal_eval
from tqdm import tqdm

# === Setup Paths ===
low_path = "openings/top_10_low_elo_openings_dataset"
high_path = "openings/top_10_high_elo_openings_dataset"
results_path = "openings/results"
os.makedirs(results_path, exist_ok=True)

# === Load Games ===
def load_top_opening_data(bracket_folder, bracket_name):
    top_openings = []
    for opening_name in os.listdir(bracket_folder):
        subfolder = os.path.join(bracket_folder, opening_name)
        for file in os.listdir(subfolder):
            if file.endswith(".csv"):
                df = pd.read_csv(os.path.join(subfolder, file))
                df["Bracket"] = bracket_name
                df["OpeningKey"] = opening_name
                df["Moves"] = df["Moves"].apply(literal_eval)
                top_openings.append(df)
    return pd.concat(top_openings, ignore_index=True)

df_low = load_top_opening_data(low_path, "Low Elo")
df_high = load_top_opening_data(high_path, "High Elo")
df_all = pd.concat([df_low, df_high])
df_all["OpeningName"] = df_all["OpeningName"].fillna("Unknown")

# === Win Rate Analysis ===
def calculate_opening_stats(df):
    stats = df.groupby(["OpeningName", "Bracket"]).Result.value_counts().unstack().fillna(0)
    stats["Total"] = stats.sum(axis=1)
    stats["WinRate"] = stats["1-0"] / stats["Total"]
    return stats.reset_index()

stats_df = calculate_opening_stats(df_all)

# === Pivot Table for Comparison ===
pivot_df = stats_df.pivot(index="OpeningName", columns="Bracket", values="WinRate").fillna(0)
pivot_df["WinRateDiff"] = pivot_df.get("High Elo", 0) - pivot_df.get("Low Elo", 0)
pivot_df["PresentIn"] = pivot_df.apply(
    lambda row: "Both" if row["High Elo"] > 0 and row["Low Elo"] > 0
    else ("High Elo only" if row["High Elo"] > 0 else "Low Elo only"), axis=1)
pivot_df = pivot_df.reset_index()

# === Save CSV ===
pivot_df.to_csv(f"{results_path}/opening_analysis.csv", index=False)

# === Plot 1: Frequency of each opening per bracket ===
plt.figure(figsize=(14, 6))
sns.countplot(data=df_all, x="OpeningName", hue="Bracket", palette="pastel")
plt.xticks(rotation=45, ha="right")
plt.title("Opening Frequency by Elo Bracket")
plt.ylabel("Number of Games")
plt.xlabel("Opening Name")
plt.tight_layout()
freq_plot = f"{results_path}/opening_usage_distribution.png"
plt.savefig(freq_plot)
plt.close()

# === Plot 2: Win Rate Comparison ===
plt.figure(figsize=(14, 6))
sns.barplot(data=stats_df, x="OpeningName", y="WinRate", hue="Bracket", palette="Set2")
plt.xticks(rotation=45, ha="right")
plt.title("White Win Rate by Opening and Elo Bracket")
plt.ylabel("Win Rate (White)")
plt.xlabel("Opening Name")
plt.tight_layout()
winrate_plot = f"{results_path}/opening_winrate_comparison.png"
plt.savefig(winrate_plot)
plt.close()

# === Plot 3: Win Rate Differential ===
plt.figure(figsize=(14, 6))
sns.barplot(data=pivot_df, x="OpeningName", y="WinRateDiff", hue="PresentIn", palette="coolwarm", dodge=False)
plt.axhline(0, linestyle="--", color="gray")
plt.xticks(rotation=45, ha="right")
plt.title("Win Rate Difference (High Elo − Low Elo) by Opening")
plt.ylabel("Δ Win Rate")
plt.xlabel("Opening Name")
plt.legend(title="Opening Presence")
plt.tight_layout()
gap_plot = f"{results_path}/opening_winrate_gap.png"
plt.savefig(gap_plot)
plt.close()

# === PDF Report (Graphs Only) ===
pdf = FPDF()
pdf.add_page()
pdf.set_font("Arial", size=14)
pdf.cell(200, 10, txt="Opening Comparison Report (Graphs Only)", ln=True, align="C")

# Add images to PDF
for title, image in [
    ("Opening Frequency by Bracket", freq_plot),
    ("Win Rate by Opening and Bracket", winrate_plot),
    ("Win Rate Difference (High - Low)", gap_plot)
]:
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=title, ln=True)
    pdf.image(image, w=180)

# Save PDF
pdf.output(f"{results_path}/opening_comparison_summary.pdf")
