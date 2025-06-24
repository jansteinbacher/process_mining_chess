import pandas as pd
from ast import literal_eval
from tqdm import tqdm
import chess
import os

# === Step 1: Load games ===
df_low = pd.read_csv("low_elo_games.csv")
df_high = pd.read_csv("high_elo_games.csv")
df_low["Moves"] = df_low["Moves"].apply(literal_eval)
df_high["Moves"] = df_high["Moves"].apply(literal_eval)

# === Step 2: Load opening definitions ===
df_open = pd.read_parquet("openings.parquet")
opening_prefix_map = {}
for _, row in df_open.iterrows():
    uci_seq = tuple(row["uci"].split())
    if uci_seq:
        opening_prefix_map[uci_seq] = row["name"]
sorted_prefixes = sorted(opening_prefix_map.keys(), key=len, reverse=True)

# === Step 3: Classify each game ===
def san_to_uci(moves, max_depth=10):
    board = chess.Board()
    uci_seq = []
    for san in moves[:max_depth]:
        try:
            move = board.parse_san(san)
            uci_seq.append(move.uci())
            board.push(move)
        except:
            break
    return tuple(uci_seq)

def classify_opening(moves):
    uci_seq = san_to_uci(moves)
    for prefix in sorted_prefixes:
        if uci_seq[:len(prefix)] == prefix:
            return opening_prefix_map[prefix]
    return "Unknown"

tqdm.pandas()
df_low["OpeningName"] = df_low["Moves"].progress_apply(classify_opening)
df_high["OpeningName"] = df_high["Moves"].progress_apply(classify_opening)

# === Step 4: Identify top 10 openings in each bracket ===
top_low = df_low["OpeningName"].value_counts().drop("Unknown", errors="ignore").head(10).index.tolist()
top_high = df_high["OpeningName"].value_counts().drop("Unknown", errors="ignore").head(10).index.tolist()

# === Step 5: Save grouped games ===
def save_opening_games(df, top_openings, folder):
    os.makedirs(folder, exist_ok=True)
    for opening in top_openings:
        safe_name = opening.replace("/", "-").replace(":", "").replace(" ", "_")
        subfolder = os.path.join(folder, safe_name)
        os.makedirs(subfolder, exist_ok=True)
        subset = df[df["OpeningName"] == opening]
        out_path = os.path.join(subfolder, f"{safe_name}.csv")
        subset.to_csv(out_path, index=False)

save_opening_games(df_low, top_low, "openings/top_10_low_elo_openings_dataset")
save_opening_games(df_high, top_high, "openings/top_10_high_elo_openings_dataset")

print("✅ Opening-specific game subsets saved.")
