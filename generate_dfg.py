import pandas as pd
import os
import re
from tqdm import tqdm
from ast import literal_eval
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import networkx as nx
import numpy as np
from collections import Counter
import chess
import chess.engine

from pm4py.objects.conversion.log import converter as log_converter
from pm4py.algo.discovery.dfg import algorithm as dfg_discovery
from pm4py.visualization.dfg import visualizer as dfg_visualizer
from pm4py.objects.log.exporter.xes import exporter as xes_exporter

# Setup paths and configuration
base_input_path = "openings"
output_path = "dfgs"
openings_parquet_path = "openings.parquet"
os.makedirs(output_path, exist_ok=True)

# Stockfish Configuration
STOCKFISH_PATH = "/opt/homebrew/bin/stockfish"
ANALYSIS_DEPTH = 15
ANALYSIS_TIME = 0.5
MAX_MOVES_TOTAL = 10  # Maximum total moves per opening analysis

# Global metrics storage for final report
global_metrics = []

def load_opening_classifications():
    """
    Load opening classifications from the parquet file.
    
    Returns:
        DataFrame: Opening classifications with ECO codes and move sequences
    """
    try:
        openings_df = pd.read_parquet(openings_parquet_path)
        print(f"Loaded {len(openings_df)} opening classifications from parquet file")
        return openings_df
    except Exception as e:
        print(f"Error loading openings.parquet: {e}")
        return pd.DataFrame()


def normalize_name(name):
    """
    Normalize opening name: lowercase, remove punctuation, replace underscores with spaces.
    """
    if not isinstance(name, str):
        return ""
    name = name.replace("_", " ")  # Convert underscores to spaces
    name = name.lower().strip()
    name = re.sub(r'[^a-z0-9 ]', '', name)  # Remove punctuation
    name = re.sub(r'\s+', ' ', name)  # Normalize multiple spaces
    return name

def find_opening_in_classification(opening_name, openings_df):
    """
    Match normalized opening names (case, punctuation, underscores, etc.).
    """
    if openings_df.empty or "name" not in openings_df.columns:
        return None

    normalized_target = normalize_name(opening_name)

    # Normalize once per run
    if "_normalized" not in openings_df.columns:
        openings_df["_normalized"] = openings_df["name"].apply(normalize_name)

    # Exact match
    exact_match = openings_df[openings_df["_normalized"] == normalized_target]
    if not exact_match.empty:
        return exact_match.iloc[0]

    # Partial match fallback
    partial_match = openings_df[openings_df["_normalized"].str.contains(normalized_target.split()[0])]
    if not partial_match.empty:
        return partial_match.iloc[0]

    return None



def get_stockfish_optimal_path(opening_moves, max_moves=MAX_MOVES_TOTAL):
    """
    Analyze an opening with Stockfish and return the optimal continuation.
    
    Args:
        opening_moves (list): List of opening moves in algebraic notation
        max_moves (int): Maximum total number of moves to analyze
        
    Returns:
        list: Complete optimal path according to Stockfish
    """
    try:
        with chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH) as engine:
            board = chess.Board()
            optimal_moves = []
            
            # Play the known opening moves
            for move_str in opening_moves:
                try:
                    move = board.parse_san(move_str)
                    board.push(move)
                    optimal_moves.append(move_str)
                except ValueError:
                    print(f"    Invalid move in opening: {move_str}")
                    break
            
            # Continue with Stockfish's best moves until max_moves reached
            move_count = len(optimal_moves)
            while move_count < max_moves and not board.is_game_over():
                try:
                    result = engine.analyse(board, chess.engine.Limit(
                        depth=ANALYSIS_DEPTH, 
                        time=ANALYSIS_TIME
                    ))
                    
                    if 'pv' in result and result['pv']:
                        best_move = result['pv'][0]
                        san_move = board.san(best_move)
                        board.push(best_move)
                        optimal_moves.append(san_move)
                        move_count += 1
                    else:
                        break
                        
                except Exception as e:
                    print(f"    Stockfish analysis error: {e}")
                    break
                    
            return optimal_moves
            
    except Exception as e:
        print(f"    Stockfish engine error: {e}")
        return opening_moves

def calculate_conformance_metrics(human_dfg, optimal_path, event_log, opening, bracket):
    """
    Calculate comprehensive conformance metrics comparing human play to optimal path.
    
    Args:
        human_dfg (dict): Human player DFG with frequencies
        optimal_path (list): Stockfish optimal path
        event_log: PM4py event log
        opening (str): Opening name
        bracket (str): Elo bracket
        
    Returns:
        dict: Dictionary containing all conformance metrics
    """
    # Create optimal path edges
    optimal_edges = set()
    for i in range(len(optimal_path) - 1):
        optimal_edges.add((optimal_path[i], optimal_path[i + 1]))
    
    # Get human edges
    human_edges = set(human_dfg.keys())
    
    # Basic set operations
    common_edges = human_edges & optimal_edges
    human_only_edges = human_edges - optimal_edges
    optimal_only_edges = optimal_edges - human_edges
    
    # Calculate total human moves and games
    total_human_moves = sum(human_dfg.values())
    total_games = len(event_log)
    
    # Calculate moves on optimal path
    optimal_moves_count = sum(human_dfg.get(edge, 0) for edge in optimal_edges)
    
    # 1. Fitness (recall) - How much of the optimal path is covered by human play
    fitness = len(common_edges) / len(optimal_edges) if optimal_edges else 0
    
    # 2. Precision - How much of human play follows optimal path
    precision = len(common_edges) / len(human_edges) if human_edges else 0
    
    # 3. F1-Score - Harmonic mean of fitness and precision
    f1_score = 2 * (fitness * precision) / (fitness + precision) if (fitness + precision) > 0 else 0
    
    # 4. Jaccard Index - Intersection over union
    jaccard_index = len(common_edges) / len(human_edges | optimal_edges) if (human_edges | optimal_edges) else 0
    
    # 5. Edge Coverage Percentage
    edge_coverage_pct = (len(common_edges) / len(optimal_edges)) * 100 if optimal_edges else 0
    
    # 6. Move-level conformance (weighted by frequency)
    move_conformance_pct = (optimal_moves_count / total_human_moves) * 100 if total_human_moves > 0 else 0
    
    # 7. Deviation metrics
    deviation_rate = len(human_only_edges) / len(human_edges) if human_edges else 0
    missing_optimal_edges = len(optimal_only_edges)
    
    # 8. Path adherence by game
    games_following_optimal = 0
    partial_adherence_scores = []
    
    for trace in event_log:
        trace_moves = [event["concept:name"] for event in trace]
        
        # Check if this game follows optimal path exactly (for first N moves)
        max_check_length = min(len(trace_moves), len(optimal_path))
        if max_check_length > 0:
            exact_match_length = 0
            for i in range(max_check_length):
                if i < len(trace_moves) and i < len(optimal_path) and trace_moves[i] == optimal_path[i]:
                    exact_match_length += 1
                else:
                    break
            
            # Calculate partial adherence score
            adherence_score = exact_match_length / len(optimal_path) if optimal_path else 0
            partial_adherence_scores.append(adherence_score)
            
            # Count as following optimal if matches at least 80% of optimal path
            if adherence_score >= 0.8:
                games_following_optimal += 1
    
    # 9. Game-level metrics
    game_adherence_rate = (games_following_optimal / total_games) * 100 if total_games > 0 else 0
    avg_partial_adherence = np.mean(partial_adherence_scores) if partial_adherence_scores else 0
    std_partial_adherence = np.std(partial_adherence_scores) if partial_adherence_scores else 0
    
    # 10. Complexity metrics
    human_complexity = len(human_edges)  # Number of unique transitions
    optimal_complexity = len(optimal_edges)
    complexity_ratio = human_complexity / optimal_complexity if optimal_complexity > 0 else 0
    
    # Compile all metrics
    metrics = {
        'opening': opening,
        'bracket': bracket,
        'total_games': total_games,
        'total_human_moves': total_human_moves,
        'optimal_path_length': len(optimal_path),
        'fitness': fitness,
        'precision': precision,
        'f1_score': f1_score,
        'jaccard_index': jaccard_index,
        'edge_coverage_pct': edge_coverage_pct,
        'move_conformance_pct': move_conformance_pct,
        'deviation_rate': deviation_rate,
        'missing_optimal_edges': missing_optimal_edges,
        'game_adherence_rate': game_adherence_rate,
        'avg_partial_adherence': avg_partial_adherence,
        'std_partial_adherence': std_partial_adherence,
        'human_complexity': human_complexity,
        'optimal_complexity': optimal_complexity,
        'complexity_ratio': complexity_ratio,
        'common_edges': len(common_edges),
        'human_only_edges': len(human_only_edges),
        'optimal_only_edges': len(optimal_only_edges)
    }
    
    return metrics

def save_conformance_report(metrics, output_dir, bracket):
    """
    Save conformance metrics to CSV and generate a summary report.
    
    Args:
        metrics (dict): Conformance metrics
        output_dir (str): Directory to save the report
        bracket (str): Elo bracket
    """
    # Save detailed metrics as CSV
    metrics_df = pd.DataFrame([metrics])
    csv_path = os.path.join(output_dir, f"{bracket.replace(' ', '_').lower()}_conformance_metrics.csv")
    metrics_df.to_csv(csv_path, index=False)
    
    # Generate human-readable report
    report_path = os.path.join(output_dir, f"{bracket.replace(' ', '_').lower()}_conformance_report.txt")
    
    with open(report_path, 'w') as f:
        f.write(f"CONFORMANCE ANALYSIS REPORT\\n")
        f.write(f"{'='*50}\\n")
        f.write(f"Opening: {metrics['opening']}\\n")
        f.write(f"Bracket: {metrics['bracket']}\\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n\\n")
        
        f.write(f"DATASET OVERVIEW\\n")
        f.write(f"{'-'*20}\\n")
        f.write(f"Total Games Analyzed: {metrics['total_games']:,}\\n")
        f.write(f"Total Human Moves: {metrics['total_human_moves']:,}\\n")
        f.write(f"Optimal Path Length: {metrics['optimal_path_length']} moves\\n\\n")
        
        f.write(f"CORE CONFORMANCE METRICS\\n")
        f.write(f"{'-'*25}\\n")
        f.write(f"Fitness (Recall):        {metrics['fitness']:.3f} ({metrics['fitness']*100:.1f}%)\\n")
        f.write(f"Precision:               {metrics['precision']:.3f} ({metrics['precision']*100:.1f}%)\\n")
        f.write(f"F1-Score:                {metrics['f1_score']:.3f}\\n")
        f.write(f"Jaccard Index:           {metrics['jaccard_index']:.3f}\\n")
        f.write(f"Edge Coverage:           {metrics['edge_coverage_pct']:.1f}%\\n")
        f.write(f"Move Conformance:        {metrics['move_conformance_pct']:.1f}%\\n\\n")
        
        f.write(f"GAME-LEVEL ANALYSIS\\n")
        f.write(f"{'-'*20}\\n")
        f.write(f"Games Following Optimal: {metrics['game_adherence_rate']:.1f}%\\n")
        f.write(f"Avg Partial Adherence:   {metrics['avg_partial_adherence']:.3f}\\n")
        f.write(f"Std Partial Adherence:   {metrics['std_partial_adherence']:.3f}\\n\\n")
        
        f.write(f"DEVIATION ANALYSIS\\n")
        f.write(f"{'-'*18}\\n")
        f.write(f"Deviation Rate:          {metrics['deviation_rate']:.3f} ({metrics['deviation_rate']*100:.1f}%)\\n")
        f.write(f"Missing Optimal Edges:   {metrics['missing_optimal_edges']}\\n")
        f.write(f"Human-Only Edges:        {metrics['human_only_edges']}\\n\\n")
        
        f.write(f"COMPLEXITY ANALYSIS\\n")
        f.write(f"{'-'*19}\\n")
        f.write(f"Human Complexity:        {metrics['human_complexity']} unique transitions\\n")
        f.write(f"Optimal Complexity:      {metrics['optimal_complexity']} unique transitions\\n")
        f.write(f"Complexity Ratio:        {metrics['complexity_ratio']:.2f}\\n")
    
    print(f"    Saved conformance report: {report_path}")
    print(f"    Saved conformance CSV: {csv_path}")

def save_optimal_path_csv(optimal_path, output_dir, bracket):
    """
    Save the optimal path to a CSV file in the opening directory.
    
    Args:
        optimal_path (list): List of optimal moves
        output_dir (str): Directory to save the CSV
        bracket (str): Elo bracket (Low Elo or High Elo)
    """
    optimal_df = pd.DataFrame({
        'move_number': range(1, len(optimal_path) + 1),
        'move': optimal_path,
        'bracket': bracket
    })
    
    csv_path = os.path.join(output_dir, f"{bracket.replace(' ', '_').lower()}_optimal_path.csv")
    optimal_df.to_csv(csv_path, index=False)
    print(f"    Saved optimal path CSV: {csv_path}")

def create_pm4py_dfg_visualization(filtered_dfg, filtered_event_log, output_path):
    """
    Create a standard PM4py DFG visualization.
    Uses filtered event log to ensure only connected nodes are shown.
    
    Args:
        filtered_dfg (dict): Filtered directly-follows graph
        filtered_event_log: PM4py event log with only connected nodes
        output_path (str): Path to save the visualization
    """
    try:
        # Standard PM4py visualization parameters
        parameters = {
            "format": "png",
            "show_edge_labels": True,
            "bgcolor": "white",
            "rankdir": "LR",
            "font_size": 10
        }
        
        # Create visualization using filtered data
        gviz = dfg_visualizer.apply(
            filtered_dfg,
            log=filtered_event_log,
            variant=dfg_visualizer.Variants.FREQUENCY,
            parameters=parameters
        )
        
        # Save the visualization
        dfg_visualizer.save(gviz, output_path)
        print(f"    Saved PM4py DFG: {output_path}")
        
    except Exception as e:
        print(f"    Error creating PM4py visualization: {str(e)}")



def generate_global_conformance_report():
    """
    Generate a comprehensive report comparing all openings and brackets.
    """
    if not global_metrics:
        return
    
    # Convert to DataFrame for analysis
    df = pd.DataFrame(global_metrics)
    
    # Save complete dataset
    complete_csv_path = os.path.join(output_path, "complete_conformance_metrics.csv")
    df.to_csv(complete_csv_path, index=False)
    
    # Generate summary statistics
    summary_path = os.path.join(output_path, "conformance_summary_report.txt")
    
    with open(summary_path, 'w') as f:
        f.write("GLOBAL CONFORMANCE ANALYSIS REPORT\\n")
        f.write("="*60 + "\\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n")
        f.write(f"Total Openings Analyzed: {df['opening'].nunique()}\\n")
        f.write(f"Total Brackets: {df['bracket'].nunique()}\\n")
        f.write(f"Total Games: {df['total_games'].sum():,}\\n\\n")
        
        # Overall statistics
        f.write("OVERALL CONFORMANCE STATISTICS\\n")
        f.write("-" * 35 + "\\n")
        for metric in ['fitness', 'precision', 'f1_score', 'jaccard_index', 'edge_coverage_pct', 'move_conformance_pct']:
            mean_val = df[metric].mean()
            std_val = df[metric].std()
            f.write(f"{metric.replace('_', ' ').title():25s}: {mean_val:.3f} ± {std_val:.3f}\\n")
        f.write("\\n")
        
        # Bracket comparison
        f.write("BRACKET COMPARISON\\n")
        f.write("-" * 18 + "\\n")
        bracket_stats = df.groupby('bracket')[['fitness', 'precision', 'f1_score', 'jaccard_index', 'edge_coverage_pct']].agg(['mean', 'std'])
        f.write(bracket_stats.to_string())
        f.write("\\n\\n")
        
        # Top performing openings
        f.write("TOP 5 OPENINGS BY F1-SCORE\\n")
        f.write("-" * 27 + "\\n")
        top_openings = df.nlargest(5, 'f1_score')[['opening', 'bracket', 'f1_score', 'fitness', 'precision']]
        f.write(top_openings.to_string(index=False))
        f.write("\\n\\n")
        
        # Worst performing openings
        f.write("BOTTOM 5 OPENINGS BY F1-SCORE\\n")
        f.write("-" * 30 + "\\n")
        bottom_openings = df.nsmallest(5, 'f1_score')[['opening', 'bracket', 'f1_score', 'fitness', 'precision']]
        f.write(bottom_openings.to_string(index=False))
        f.write("\\n\\n")
        
        # Complexity analysis
        f.write("COMPLEXITY ANALYSIS\\n")
        f.write("-" * 19 + "\\n")
        f.write(f"Average Human Complexity:  {df['human_complexity'].mean():.1f} transitions\\n")
        f.write(f"Average Optimal Complexity: {df['optimal_complexity'].mean():.1f} transitions\\n")
        f.write(f"Average Complexity Ratio:   {df['complexity_ratio'].mean():.2f}\\n")
        f.write(f"Max Complexity Ratio:       {df['complexity_ratio'].max():.2f}\\n")
        f.write(f"Min Complexity Ratio:       {df['complexity_ratio'].min():.2f}\\n")
    
    # Create visualization comparing brackets
    plt.figure(figsize=(15, 10))
    
    # Subplot 1: Fitness comparison
    plt.subplot(2, 3, 1)
    df.boxplot(column='fitness', by='bracket', ax=plt.gca())
    plt.title('Fitness by Bracket')
    plt.suptitle('')
    
    # Subplot 2: Precision comparison
    plt.subplot(2, 3, 2)
    df.boxplot(column='precision', by='bracket', ax=plt.gca())
    plt.title('Precision by Bracket')
    plt.suptitle('')
    
    # Subplot 3: F1-Score comparison
    plt.subplot(2, 3, 3)
    df.boxplot(column='f1_score', by='bracket', ax=plt.gca())
    plt.title('F1-Score by Bracket')
    plt.suptitle('')
    
    # Subplot 4: Edge Coverage comparison
    plt.subplot(2, 3, 4)
    df.boxplot(column='edge_coverage_pct', by='bracket', ax=plt.gca())
    plt.title('Edge Coverage % by Bracket')
    plt.suptitle('')
    
    # Subplot 5: Move Conformance comparison
    plt.subplot(2, 3, 5)
    df.boxplot(column='move_conformance_pct', by='bracket', ax=plt.gca())
    plt.title('Move Conformance % by Bracket')
    plt.suptitle('')
    
    # Subplot 6: Complexity Ratio comparison
    plt.subplot(2, 3, 6)
    df.boxplot(column='complexity_ratio', by='bracket', ax=plt.gca())
    plt.title('Complexity Ratio by Bracket')
    plt.suptitle('')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "conformance_comparison_by_bracket.png"), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\\nGenerated global conformance reports:")
    print(f"  - Complete metrics: {complete_csv_path}")
    print(f"  - Summary report: {summary_path}")
    print(f"  - Comparison chart: {os.path.join(output_path, 'conformance_comparison_by_bracket.png')}")

def load_all_openings(bracket_folder, bracket_label):
    """
    Load all opening games from a bracket folder.
    
    Args:
        bracket_folder (str): Path to the bracket folder
        bracket_label (str): Label for the bracket (Low Elo/High Elo)
        
    Returns:
        DataFrame: Combined dataframe of all games
    """
    logs = []
    for opening in os.listdir(bracket_folder):
        opening_path = os.path.join(bracket_folder, opening)
        if os.path.isdir(opening_path):
            for file in os.listdir(opening_path):
                if file.endswith(".csv"):
                    df = pd.read_csv(os.path.join(opening_path, file))
                    df["Opening"] = opening
                    df["Bracket"] = bracket_label
                    df["Moves"] = df["Moves"].apply(literal_eval)
                    logs.append(df)
    return pd.concat(logs, ignore_index=True) if logs else pd.DataFrame()

def transform_to_event_log(df, bracket, opening):
    """
    Transform game data into event log format for process mining.
    
    Args:
        df (DataFrame): Game data
        bracket (str): Elo bracket
        opening (str): Opening name
        
    Returns:
        DataFrame: Event log format data
    """
    rows = []
    base_time = datetime(2025, 4, 1, 12, 0, 0)
    
    for i, row in df.iterrows():
        game_id = f"{bracket.replace(' ', '_')}_{opening}_{i}"
        # Limit to first 20 moves for analysis
        for j, move in enumerate(row["Moves"][:20]):
            timestamp = base_time + timedelta(seconds=j)
            rows.append({
                "case:concept:name": game_id,
                "concept:name": move,
                "time:timestamp": timestamp,
                "Bracket": bracket,
                "Opening": opening
            })
    
    return pd.DataFrame(rows)

# Load opening classifications
print("Loading opening classifications...")
openings_df = load_opening_classifications()

# Load game data
print("Loading game data...")
low_df = load_all_openings(os.path.join(base_input_path, "top_10_low_elo_openings_dataset"), "Low Elo")
high_df = load_all_openings(os.path.join(base_input_path, "top_10_high_elo_openings_dataset"), "High Elo")
df_all = pd.concat([low_df, high_df], ignore_index=True)

print(f"Loaded {len(df_all)} total games across {df_all['Opening'].nunique()} openings")

# Cache Stockfish optimal paths per opening
optimal_path_cache = {}

# Main processing loop
for opening in tqdm(df_all["Opening"].unique(), desc="Processing Openings"):
    print(f"\nProcessing opening: {opening}")

    # Find opening in classification database
    opening_info = find_opening_in_classification(opening, openings_df)

    if opening_info is not None:
        print(f"  Found opening classification: {opening_info.get('name', 'Unknown')}")
        if 'moves' in opening_info:
            base_moves = opening_info['moves'].split() if isinstance(opening_info['moves'], str) else []
        else:
            base_moves = None
    else:
        print("  Opening not found in classification database.")
        base_moves = None

    # Use fallback if classification doesn't yield moves
    if base_moves is None:
        fallback_subset = df_all[df_all["Opening"] == opening]
        if fallback_subset.empty:
            print("  No games available for this opening. Skipping.")
            continue
        base_moves = fallback_subset.iloc[0]["Moves"][:4]

    # Get or compute the optimal path (once per opening)
    if opening not in optimal_path_cache:
        print("  Analyzing optimal path with Stockfish...")
        optimal_path_cache[opening] = get_stockfish_optimal_path(base_moves, MAX_MOVES_TOTAL)
    else:
        print("  Using cached optimal path.")

    stockfish_optimal_path = optimal_path_cache[opening]
    print(f"  Stockfish optimal path ({len(stockfish_optimal_path)} moves): {' '.join(stockfish_optimal_path)}")

    # Process both brackets for the same opening
    for bracket in ["Low Elo", "High Elo"]:
        subset = df_all[(df_all["Opening"] == opening) & (df_all["Bracket"] == bracket)]
        if subset.empty:
            print(f"  No data for bracket {bracket}. Skipping.")
            continue

        # ---- Everything else below remains unchanged ----
        # Save path
        opening_dir = os.path.join(output_path, opening.replace(" ", "_"))
        os.makedirs(opening_dir, exist_ok=True)
        save_optimal_path_csv(stockfish_optimal_path, opening_dir, bracket)

        # Transform to event log, filter, analyze DFG, calculate conformance, generate visuals...
        # (use the remaining unchanged code)

        
        # Transform to event log
        event_df = transform_to_event_log(subset, bracket, opening)
        
        # Save event log CSV
        csv_path = os.path.join(opening_dir, f"{bracket.replace(' ', '_').lower()}.csv")
        event_df.to_csv(csv_path, index=False)
        
        # Convert to PM4py event log
        event_df["time:timestamp"] = pd.to_datetime(event_df["time:timestamp"])
        event_log = log_converter.apply(event_df, parameters={"case_id_key": "case:concept:name"})
        
        # Save XES file
        xes_path = os.path.join(opening_dir, f"{bracket.replace(' ', '_').lower()}.xes")
        xes_exporter.apply(event_log, xes_path)
        
        # Discover DFG
        full_dfg = dfg_discovery.apply(event_log)
        min_edge_freq = 100
        
        # Filter edges by frequency
        filtered_dfg = {k: v for k, v in full_dfg.items() if v >= min_edge_freq}
        
        print(f"  DFG edges before filtering: {len(full_dfg)}")
        print(f"  DFG edges after filtering: {len(filtered_dfg)}")
        
        if not filtered_dfg:
            print("  No edges passed filtering. Skipping visualization.")
            continue
        
        # Get only connected nodes (nodes that have at least one edge)
        connected_nodes = set()
        for (source, target) in filtered_dfg.keys():
            connected_nodes.add(source)
            connected_nodes.add(target)
        
        # Filter event log to only include connected nodes
        filtered_event_data = []
        for trace in event_log:
            for event in trace:
                # PM4py events use different key access
                event_name = event.get("concept:name") or event.get("Activity") or str(event.get("concept:name", ""))
                if event_name in connected_nodes:
                    # Create a clean event dictionary
                    clean_event = {
                        "case:concept:name": trace.attributes.get("concept:name", f"case_{len(filtered_event_data)}"),
                        "concept:name": event_name,
                        "time:timestamp": event.get("time:timestamp", datetime.now()),
                        "Bracket": bracket,
                        "Opening": opening
                    }
                    filtered_event_data.append(clean_event)
        
        # Convert filtered data back to event log
        if filtered_event_data:
            filtered_event_df = pd.DataFrame(filtered_event_data)
            filtered_event_log = log_converter.apply(filtered_event_df, parameters={"case_id_key": "case:concept:name"})
        else:
            filtered_event_log = event_log
        
        print(f"  Connected nodes in DFG: {len(connected_nodes)}")
        print(f"  Filtered event log size: {len(filtered_event_data)} events")
        
        # Calculate conformance metrics
        print("  Calculating conformance metrics...")
        conformance_metrics = calculate_conformance_metrics(
            filtered_dfg, stockfish_optimal_path, event_log, opening, bracket
        )
        
        # Add to global metrics
        global_metrics.append(conformance_metrics)
        
        # Save conformance report
        save_conformance_report(conformance_metrics, opening_dir, bracket)
        
        # Print key metrics
        print(f"  Conformance Summary:")
        print(f"    Fitness: {conformance_metrics['fitness']:.3f}")
        print(f"    Precision: {conformance_metrics['precision']:.3f}")
        print(f"    F1-Score: {conformance_metrics['f1_score']:.3f}")
        print(f"    Edge Coverage: {conformance_metrics['edge_coverage_pct']:.1f}%")
        print(f"    Move Conformance: {conformance_metrics['move_conformance_pct']:.1f}%")
        
        # Create PM4py DFG visualization (using filtered event log)
        pm4py_dfg_path = os.path.join(opening_dir, f"{bracket.replace(' ', '_').lower()}_dfg_pm4py.png")
        create_pm4py_dfg_visualization(filtered_dfg, filtered_event_log, pm4py_dfg_path)
        


# Generate global conformance report
print("Generating global conformance analysis...")
generate_global_conformance_report()

print("DFG generation and conformance analysis complete!")