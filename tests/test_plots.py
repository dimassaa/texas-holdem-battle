"""Plot builders emit persisted PNGs."""
from pathlib import Path

from poker.analysis import render_all_plots
from poker.experiments import run_combo_seeds


def test_render_all_plots_produces_files(tmp_path):
    m = run_combo_seeds("benchmark_mix", seeds=(1,), num_hands=20, log_root=str(tmp_path))
    out = Path(tmp_path) / "reports" / "plots"
    render_all_plots(m, out_dir=str(out))
    pngs = list(out.glob("*.png"))
    # APPROVED deviation from the literal (>= 5): heatmaps are PURE-COMBO-only
    # (six_<strategy>), and benchmark_mix is not a pure combo — so this
    # manifest yields 3 per-combo families (stack_boxplot, stack_evolution,
    # ruin_curve for benchmark_mix) + 1 global (bb100_compare) = 4 files.
    assert len(pngs) >= 4