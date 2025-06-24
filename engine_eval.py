import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import chess
import chess.engine
from tqdm import tqdm
from ast import literal_eval

# === Paths and Setup ===
input_path = "openings/results/opening_analysis.csv"
low_data_path = "openings/top_10_low_elo_openings_dataset"
high_data_path = "openings/top_10_high_elo_openings_dataset"
eval_results_path = "openings/results/evaluation"
eval_games_path = f"{eval_results_path}/all_games_with_eval.csv"
os.makedirs(eval_results_path, exist_ok=True)

# === Load Opening Analysis ===
df_openings = pd.read_csv(input_path)

# === Load Game Data ===
def load_games(bracket_folder, bracket_name):
    games = []
    for opening_name in os.listdir(bracket_folder):
        subfolder = os.path.join(bracket_folder, opening_name)
        for file in os.listdir(subfolder):
            if file.endswith(".csv"):
                df = pd.read_csv(os.path.join(subfolder, file))
                df["OpeningName"] = opening_name
                df["Bracket"] = bracket_name
                df["Moves"] = df["Moves"].apply(literal_eval)
                games.append(df)
    return pd.concat(games, ignore_index=True) if games else pd.DataFrame()

# === Load or Create Evaluation Data ===
if os.path.exists(eval_games_path):
    df_all = pd.read_csv(eval_games_path)
    df_all["Moves"] = df_all["Moves"].apply(literal_eval)
else:
    df_low = load_games(low_data_path, "Low Elo")
    df_high = load_games(high_data_path, "High Elo")
    df_all = pd.concat([df_low, df_high], ignore_index=True)

# === Opening Family Classification ===
def classify_opening_family(name):
    name = name.lower()
    if "gambit" in name:
        return "Gambit"
    elif "queen's pawn" in name or "indian" in name or "benoni" in name or "grünfeld" in name:
        return "d4"
    elif "king's pawn" in name or "e4" in name or "philidor" in name or "ruy lopez" in name or "bishop" in name:
        return "e4"
    elif "caro-kann" in name or "sicilian" in name or "french" in name or "pirc" in name or "modern" in name or "scandinavian" in name:
        return "Semi-open"
    elif "english" in name or "reti" in name or "van't kruijs" in name or "horwitz" in name:
        return "Flank"
    else:
        return "Other"

df_all["OpeningFamily"] = df_all["OpeningName"].apply(classify_opening_family)

# === Stockfish Evaluation after 10 Moves ===
engine_path = "/opt/homebrew/bin/stockfish"
engine = chess.engine.SimpleEngine.popen_uci(engine_path)

def evaluate_after_10_moves(moves):
    board = chess.Board()
    try:
        for move in moves[:10]:
            board.push_san(move)
        info = engine.analyse(board, chess.engine.Limit(time=0.1))
        return info["score"].white().score(mate_score=10000)
    except Exception:
        return None

# === Only evaluate if not already done
if "Eval10" not in df_all.columns:
    df_all["Eval10"] = None

def safe_eval(row):
    if pd.notnull(row["Eval10"]):
        return row["Eval10"]
    return evaluate_after_10_moves(row["Moves"])

tqdm.pandas()
df_all["Eval10"] = df_all.progress_apply(safe_eval, axis=1)
engine.quit()

# === Save evaluated games
df_all.to_csv(eval_games_path, index=False)

# === Aggregate Evaluations by Opening & Bracket
eval_summary = df_all.groupby(["OpeningName", "Bracket"]).agg(
    AvgEval10=("Eval10", "mean"),
    StdEval10=("Eval10", "std"),
    N=("Eval10", "count")
).reset_index()

# === Normalize for merge (convert _ to space, lowercase)
def normalize_opening_name(df):
    return df["OpeningName"].str.replace("_", " ").str.strip().str.lower()

eval_summary["OpeningName"] = normalize_opening_name(eval_summary)
df_openings["OpeningName"] = normalize_opening_name(df_openings)

# === Reshape win rates to long format
df_openings_long = pd.melt(
    df_openings,
    id_vars=["OpeningName"],
    value_vars=["High Elo", "Low Elo"],
    var_name="Bracket",
    value_name="WinRate"
)
df_openings_long["Bracket"] = df_openings_long["Bracket"].str.replace(" Elo", "") + " Elo"
df_openings_long["OpeningName"] = normalize_opening_name(df_openings_long)

# === Merge Evaluation and Win Rate Data
merged = pd.merge(eval_summary, df_openings_long, on=["OpeningName", "Bracket"], how="left")
merged.to_csv(f"{eval_results_path}/eval_by_opening.csv", index=False)

import matplotlib.pyplot as plt
import seaborn as sns

# === Data label helper
def add_labels(ax, data, x_col, y_col, label_col):
    for _, row in data.iterrows():
        ax.text(row[x_col], row[y_col], row[label_col], fontsize=8, alpha=0.7)

# === Merge family info
merged["OpeningFamily"] = merged["OpeningName"].apply(classify_opening_family)
merged["OpeningLabel"] = merged["OpeningName"].str.title().str.replace("'", "").str.replace("_", " ")

# === Plot 1: Eval vs WinRate (Labeled)
plt.figure(figsize=(12, 7))
ax = sns.scatterplot(
    data=merged,
    x="AvgEval10",
    y="WinRate",
    hue="Bracket",
    style="Bracket",
    s=120
)
plt.axvline(0, color='gray', linestyle='--')
add_labels(ax, merged, "AvgEval10", "WinRate", "OpeningLabel")
plt.title("Engine Eval vs. Win Rate (with Labels)")
plt.xlabel("Avg. Stockfish Eval (centipawns)")
plt.ylabel("Empirical Win Rate")
plt.tight_layout()
plt.savefig(f"{eval_results_path}/eval_vs_winrate_labeled.png")
plt.close()

# === Plot 2: Average Eval by Opening Family
plt.figure(figsize=(10, 6))
sns.barplot(
    data=merged,
    x="OpeningFamily",
    y="AvgEval10",
    hue="Bracket"
)
plt.title("Average Eval by Opening Family")
plt.ylabel("Avg. Stockfish Eval (centipawns)")
plt.tight_layout()
plt.savefig(f"{eval_results_path}/avg_eval_by_family.png")
plt.close()

# === Plot 3: Average Win Rate by Family
plt.figure(figsize=(10, 6))
sns.barplot(
    data=merged,
    x="OpeningFamily",
    y="WinRate",
    hue="Bracket"
)
plt.title("Average Win Rate by Opening Family")
plt.ylabel("Empirical Win Rate")
plt.tight_layout()
plt.savefig(f"{eval_results_path}/avg_winrate_by_family.png")
plt.close()

print("✅ Engine evaluations completed and results saved.")
