import chess.pgn
import zstandard as zstd
import io
import pandas as pd
from tqdm import tqdm

# Set your file path here
FILE_PATH = "lichess_db_standard_rated_2025-04.pgn.zst"

# Target number of games
TARGET_LOW = 100000
TARGET_HIGH = 100000

# Output CSVs
OUT_LOW = "low_elo_games.csv"
OUT_HIGH = "high_elo_games.csv"


def stream_pgn_games(file_path):
    """
    Efficiently decompress and stream PGN games from a large .zst file.
    """
    with open(file_path, "rb") as fh:
        dctx = zstd.ZstdDecompressor(max_window_size=2**31)  # allows large files
        stream_reader = dctx.stream_reader(fh)
        text_stream = io.TextIOWrapper(stream_reader, encoding="utf-8", newline="")
        while True:
            try:
                game = chess.pgn.read_game(text_stream)
                if game is None:
                    break
                yield game
            except Exception:
                continue  # skip broken PGN blocks


def extract_game_data(game):
    """
    Extract basic info and first 10 moves of a game.
    """
    try:
        headers = game.headers
        white_elo = int(headers.get("WhiteElo", 0))
        black_elo = int(headers.get("BlackElo", 0))
        result = headers.get("Result", "")
        if white_elo == 0 or black_elo == 0:
            return None
        if result not in {"1-0", "0-1", "1/2-1/2"}:
            return None

        board = game.board()
        moves = []
        for i, move in enumerate(game.mainline_moves()):
            if i >= 10:
                break
            moves.append(board.san(move))
            board.push(move)

        if len(moves) < 10:
            return None

        return {
            "WhiteElo": white_elo,
            "BlackElo": black_elo,
            "Result": result,
            "Moves": moves
        }
    except:
        return None


def main():
    low_games = []
    high_games = []

    print("Extracting games from file...")
    for game in tqdm(stream_pgn_games(FILE_PATH), desc="Streaming PGN"):
        data = extract_game_data(game)
        if not data:
            continue

        white = data["WhiteElo"]
        black = data["BlackElo"]

        # Add to low Elo bucket
        if white <= 1200 and black <= 1200 and len(low_games) < TARGET_LOW:
            low_games.append(data)

        # Add to high Elo bucket
        elif white >= 1800 and black >= 1800 and len(high_games) < TARGET_HIGH:
            high_games.append(data)

        if len(low_games) >= TARGET_LOW and len(high_games) >= TARGET_HIGH:
            break

    # Save to CSV
    pd.DataFrame(low_games).to_csv(OUT_LOW, index=False)
    pd.DataFrame(high_games).to_csv(OUT_HIGH, index=False)

    print(f"✅ Saved {len(low_games)} low Elo games to {OUT_LOW}")
    print(f"✅ Saved {len(high_games)} high Elo games to {OUT_HIGH}")


if __name__ == "__main__":
    main()
