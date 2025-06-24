# ♟️ Process Mining in Chess

This project explores how **process mining techniques** can be applied to chess data to uncover behavioral patterns across player skill levels. Using over 200,000 games from Lichess.org, the project compares low- and high-rated players in terms of opening choices, engine evaluations, and structural differences using Directly-Follows Graphs (DFGs) and conformance checking.

---

## 📁 Project Structure Overview

The repository is organized as follows:

### `create_elo_brackets.py`

- decompress and stream PGN games from a large .zst file (lichess data set)
- Extract basic info and first 10 moves of games
- 100.000 games low elo / 100.000 games high elo

---

### `opening_classification.py`

- uses openings.parquet file (lichess ECO database) to add an opening name to each game

---
### `analyze_top_openings.py`

- load the classified games
- extract the top 10 openings per ELO bracket
- analyze the win rate
- create graphs for data analysis

---

### `engine_eval.py`

- load games
- classify them into opening family
- generate a Stockfish chess engine evaluation after 10 moves for each game
- generate graphs to compare win rate against engine evaluation


---

### `generate_dfg.py`

- load games (top 10 openings per bracket)
- generate optimal path for opening with stockfish
- transform game data into event log for PM4PY
- create DFGs per opening and bracket with path frequency over 100
- calculate conformance metrics



