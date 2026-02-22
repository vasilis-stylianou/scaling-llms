from pathlib import Path
import subprocess
import time
from tensorboard import notebook as tb_notebook


def print_tree(root: str | Path, display_max_leaves: int | None = None) -> None:
    root = Path(root)
    print(f"{root.name}/")
    tree = {}
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        # Skip hidden files and directories (e.g., .DS_Store, .git)
        if any(part.startswith('.') for part in rel.parts):
            continue
        parts = rel.parts
        d = tree
        for part in parts[:-1]:
            d = d.setdefault(part, {})
        d.setdefault(parts[-1], {})

    def _print_subtree(node, prefix=""):
        items = list(node.items())

        # Determine how many direct file children this node has
        total_files = 0
        for name, child in items:
            path = root / prefix.replace("│   ", "").replace("    ", "") / name
            if not path.is_dir():
                total_files += 1

        file_limit = None
        if display_max_leaves is not None and total_files > display_max_leaves:
            file_limit = display_max_leaves

        # Build the list of items we will actually print (preserve original order)
        printed_items = []
        printed_files = 0
        for name, child in items:
            path = root / prefix.replace("│   ", "").replace("    ", "") / name
            is_file = not path.is_dir()
            if is_file and (file_limit is not None) and (printed_files >= file_limit):
                continue
            printed_items.append((name, child))
            if is_file:
                printed_files += 1

        # Print the selected items with correct tree connectors
        for i, (name, child) in enumerate(printed_items):
            is_last = i == len(printed_items) - 1
            connector = "└── " if is_last else "├── "
            path = root / prefix.replace("│   ", "").replace("    ", "") / name
            suffix = "/" if path.is_dir() else ""
            print(f"{prefix}{connector}{name}{suffix}")
            extension = "    " if is_last else "│   "
            _print_subtree(child, prefix + extension)

        # If we skipped some files, print a compact summary line
        if (file_limit is not None) and (total_files > file_limit):
            skipped = total_files - file_limit
            # Use a connector matching a final child
            connector = "└── "
            print(f"{prefix}{connector}... {skipped} more files")

    _print_subtree(tree)






# -----------------------------------------------------------------------------
# Configuration: Update this path to match your local setup
# -----------------------------------------------------------------------------
def kill_tb(port: int = 6006) -> None:
    """Kill any process currently using the specified TensorBoard port."""
    try:
        cmd = f"lsof -t -i:{port}"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.stdout.strip():
            pids = result.stdout.strip().split('\n')
            for pid in pids:
                subprocess.run(f"kill {pid}", shell=True)
                print(f"Killed existing process on port {port} (PID: {pid})")
        else:
            print(f"No existing process found on port {port}.")
            
    except Exception as e:
        print(f"Warning during cleanup: {e}")


def launch_tb(log_dir: str, port: int = 6006, display: bool = True) -> None:
    """
    Launches TensorBoard for a specific experiment directory.
    
    Args:
        log_dir (str): The full path to the logging directory.
        port (int): The port to launch TensorBoard on (default: 6006).
    """
    
    # 1. Verify that the logging directory exists
    if not Path(log_dir).exists():
        raise ValueError(f"Logging directory {log_dir} does not exist. Please check the path.")
    log_dir = str(log_dir)  # Ensure it's a string for subprocess

    # 2. Kill any process currently running on the target port
    kill_tb(port=port)

    # 3. Start TensorBoard in the background
    print(f"Launching TensorBoard...")
    print(f"Logging Dir: {log_dir}")
    
    proc = subprocess.Popen(
        ["tensorboard", "--logdir", log_dir, "--port", str(port), "--bind_all"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    print(f"Started TensorBoard with PID {proc.pid}")
    print(f"http://localhost:{port}")
    print("Waiting 3 seconds for startup...")
    time.sleep(3)

    # 4. Display inside the notebook
    if display:
        tb_notebook.display(port=port, height=1000)


def plot_lines(
    df,
    y_col: str,
    x_col: str,
    title: str | None = None,
    color_col: str | None = None,
    smoothing: float = 0.0,
    width: int = 900,
    height: int = 450,
    auto_show: bool = True,
):
    """
    Plots line charts using Plotly Express with optional smoothing.
    
    Args:
        df (pd.DataFrame): DataFrame containing the data to plot.
        y_col (str): Column name for the y-axis.
        x_col (str): Column name for the x-axis.
        title (str | None): Plot title. Defaults to "{y_col} vs {x_col}".
        color_col (str | None): Column name for line colors. Defaults to None.
        smoothing (float): Smoothing factor between 0.0 and 1.0. Defaults to 0.0.
        width (int): Plot width in pixels. Defaults to 900.
        height (int): Plot height in pixels. Defaults to 450.
        auto_show (bool): Whether to automatically display the plot. Defaults to True.

    Returns:
        plotly.graph_objects.Figure: The Plotly figure object.
    """
    import plotly.express as px
    import pandas as pd

    required = {y_col, x_col} 
    if color_col is not None:
        required.add(color_col)
    if not required.issubset(df.columns):
        raise ValueError(f"DataFrame must contain columns: {sorted(required)}")

    if not (0.0 <= float(smoothing) <= 1.0):
        raise ValueError("smoothing must be between 0.0 and 1.0")

    title = title or f"{y_col.upper()} vs {x_col.upper()}"

    # Ensure deterministic ordering
    sort_cols = [x_col] if color_col is None else [x_col, color_col]
    df_sorted = df.sort_values(sort_cols) 

    # If smoothing requested, compute per-run rolling mean of `nll`.
    if float(smoothing) > 0.0:
        y_col_plot = f"{y_col} (smoothing={smoothing})"
        if color_col is not None:
            group_col = color_col
        else:
            # Create artificial group_col to keep smoothing logic the same for both cases
            group_col = "some_group_col"
            df_sorted[group_col] = "some_group_name"
        
        parts = []
        for _, g in df_sorted.groupby(group_col, sort=False):
            g = g.copy()
            # Map smoothing fraction [0,1] to a window size in samples
            window = max(1, int(smoothing * len(g)))
            g[y_col_plot] = g[y_col].rolling(window=window, center=True, min_periods=1).mean()
            parts.append(g)
        df_plot = pd.concat(parts, ignore_index=True)
    else:
        df_plot = df_sorted
        y_col_plot = y_col

    fig = px.line(
        df_plot,
        x=x_col,
        y=y_col_plot,
        color=color_col,
        labels={x_col: x_col, y_col_plot: y_col_plot, color_col: color_col},
        title=title,
    )
    fig.update_layout(width=width, height=height)
    if auto_show:
        fig.show()

    return fig



BASE_PLOT_KWARGS = {
    "train_nll": dict(
        metric_cat="train",
        metric_name="nll",
        smoothing=0.005,
        title="TRAIN NLL vs Step by Run",
    ),
    "eval_nll": dict(
        metric_cat="eval",
        metric_name="nll",
        smoothing=0.005,
        title="EVAL NLL vs Step by Run",
    ),
    "grad_norm": dict(
        metric_cat="network",
        metric_name="grad_norm",
        smoothing=0.0,
        title="Gradient Norm vs Step by Run",
    ),
    "grad_to_param_ratio": dict(
        metric_cat="network",
        metric_name="grad_to_param_ratio",
        smoothing=0.0,
        title="Grad to Param Ratio vs Step by Run",
    ),
    "lr": dict(
        metric_cat="train",
        metric_name="lr",
        smoothing=0.0,
        title="LR vs Step by Run",
    ),
    "param_norm": dict(
        metric_cat="network",
        metric_name="param_norm",
        smoothing=0.0,
        title="Param Norm vs Step by Run",
    ),
    "grad_zero_frac": dict(
        metric_cat="network",
        metric_name="grad_zero_frac",
        smoothing=0.0,
        title="Grad Zero Frac vs Step by Run",
    ),
    "tokens_per_sec": dict(
        metric_cat="train",
        metric_name="tokens_per_sec",
        smoothing=0.005,
        title="Tokens/sec vs Step by Run",
    ),
    "peak_alloc_gb": dict(
        metric_cat="system",
        metric_name="peak_alloc_gb",
        smoothing=0.0,
        title="Peak Alloc GB vs Step by Run"
    )
}
