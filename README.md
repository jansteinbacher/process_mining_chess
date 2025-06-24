# ♟️ Process Mining in Chess

This project explores how **process mining techniques** can be applied to chess data to uncover behavioral patterns across player skill levels. Using over 200,000 games from Lichess.org, the project compares low- and high-rated players in terms of opening choices, engine evaluations, and structural differences using Directly-Follows Graphs (DFGs) and conformance checking.

---

## 📁 Project Structure Overview

The repository is organized as follows:

---

### [`create_elo_brackets.py`](./create_elo_brackets.py)

- Decompress and stream PGN games from a large `.zst` file (Lichess dataset)
- Extract basic metadata and the first 10 moves of games
- Select 100,000 games each for low Elo (<1200) and high Elo (>1800)

---

### [`opening_classification.py`](./opening_classification.py)

- Uses the `openings.parquet` file (based on the Lichess ECO database) to assign an opening name to each game
- External resources:
  - [Lichess Openings Dataset (Hugging Face)](https://huggingface.co/datasets/Lichess/chess-openings)
  - [Lichess Chess Openings GitHub](https://github.com/lichess-org/chess-openings?tab=readme-ov-file)

---

### [`analyze_top_openings.py`](./analyze_top_openings.py)

- Loads the classified games
- Extracts the top 10 openings per Elo bracket
- Analyzes win rates
- Creates summary statistics and plots for data interpretation

---

### [`engine_eval.py`](./engine_eval.py)

- Loads games and classifies them into opening families
- Uses Stockfish to evaluate positions after 10 moves
- Generates average and standard deviation of engine evaluations
- Compares win rate with engine evaluation using plots

---

### [`generate_dfg.py`](./generate_dfg.py)

- Loads games for the top 10 openings per bracket
- Computes the Stockfish-optimal move path for each opening
- Transforms move data into event logs suitable for PM4Py
- Generates Directly-Follows Graphs (DFGs) with a frequency threshold
- Computes conformance metrics (fitness, precision, etc.) per opening and bracket


