import pandas as pd
import ast
from collections import Counter

# Load data
low_elo_df = pd.read_csv("low_elo_games_openings.csv")
high_elo_df = pd.read_csv("high_elo_games_openings.csv")

# Ensure Moves is parsed as a list
low_elo_df["Moves"] = low_elo_df["Moves"].apply(ast.literal_eval)
high_elo_df["Moves"] = high_elo_df["Moves"].apply(ast.literal_eval)

# Helper function to extract opening family
def extract_opening_family(opening):
    if pd.isnull(opening):
        return "Unknown"
    opening = opening.lower()
    if "indian" in opening:
        return "Indian"
    elif "gambit" in opening:
        return "Gambit"
    elif "semi-" in opening or "semi" in opening:
        return "Semi-Open"
    elif "open" in opening:
        return "Open"
    elif "closed" in opening:
        return "Closed"
    elif "flank" in opening:
        return "Flank"
    elif "king's pawn" in opening or opening.startswith("e4"):
        return "Open"
    elif "queen's pawn" in opening or opening.startswith("d4"):
        return "Closed"
    else:
        return "Other"

# Apply family extraction
low_elo_df["OpeningFamily"] = low_elo_df["OpeningName"].apply(extract_opening_family)
high_elo_df["OpeningFamily"] = high_elo_df["OpeningName"].apply(extract_opening_family)

# Dataset overview stats
def dataset_stats(df):
    num_games = len(df)
    family_distribution = df["OpeningFamily"].value_counts().to_dict()
    top_10_openings = df["OpeningName"].value_counts().head(10).to_dict()
    num_distinct_openings = df["OpeningName"].nunique()
    avg_game_length = df["Moves"].apply(len).mean()
    unique_move_sequences = df["Moves"].apply(lambda x: tuple(x)).nunique()
    elo_distribution = {
        "WhiteElo_mean": df["WhiteElo"].mean(),
        "BlackElo_mean": df["BlackElo"].mean(),
        "WhiteElo_std": df["WhiteElo"].std(),
        "BlackElo_std": df["BlackElo"].std()
    }

    return {
        "num_games": num_games,
        "family_distribution": family_distribution,
        "top_10_openings": top_10_openings,
        "num_distinct_openings": num_distinct_openings,
        "avg_game_length": avg_game_length,
        "unique_move_sequences": unique_move_sequences,
        "elo_distribution": elo_distribution
    }

low_stats = dataset_stats(low_elo_df)
high_stats = dataset_stats(high_elo_df)

print("low_elo:",low_stats)

print("high_elo:",high_stats)
      
