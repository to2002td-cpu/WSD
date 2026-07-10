"""Interactive per-sense KDE viewer for one word — a self-contained Plotly HTML.

The page is a vertical column of panels, one per hidden layer (deepest at the
bottom), each showing the per-sense Gaussian-KDE density "clouds" in the shared
2-D UMAP space with a light scatter behind them. A checkpoint slider at the
bottom swaps every layer to a different training step at once, so the whole
depth column evolves together as the slider is dragged — you watch senses
separate (or merge) over training, layer by layer.

Same data contract as the other visualizations: all 2-D coordinates come from
the shared UMAP cache, and display filters are applied on top. Colours and the
sense selection match ``density.py`` so the PNG grid and this HTML agree.
"""

from __future__ import annotations

import logging

import numpy as np

from .umapcache import SENSE_PALETTE, displayed_senses, gloss, load_umap

log = logging.getLogger(__name__)

HTML_GRID = 60      # KDE evaluation grid resolution (smaller keeps the HTML light)
HTML_KDE_MAX = 500  # subsample per sense before fitting the KDE (speed + size)
SCATTER_MAX = 250   # points drawn per sense per panel (readability + size)


def _rgb(hexcol: str) -> "tuple[int, int, int]":
    h = hexcol.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def _fill_scale(hexcol: str, amax: float = 0.5):
    """Transparent -> colour ramp, so overlapping sense clouds blend by opacity."""
    r, g, b = _rgb(hexcol)
    return [[0.0, f"rgba({r},{g},{b},0)"], [1.0, f"rgba({r},{g},{b},{amax})"]]


def _kde_grid(pts: np.ndarray, xs: np.ndarray, ys: np.ndarray, rng) -> "np.ndarray | None":
    from scipy.stats import gaussian_kde

    if len(pts) < 15:
        return None
    if len(pts) > HTML_KDE_MAX:
        pts = pts[rng.choice(len(pts), HTML_KDE_MAX, replace=False)]
    X, Y = np.meshgrid(xs, ys)                       # (ny, nx): z[j, i] <-> (xs[i], ys[j])
    try:
        Z = gaussian_kde(pts.T)(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
    except Exception:
        return None
    if Z.max() <= 0:
        return None
    return Z.astype(np.float32)


def build_kde_html(records, emb_dir, word, pos, out_path, cache_dir=None, min_per_sense: int = 0):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    meta, pos, steps, layers, coords = load_umap(emb_dir, records, word, pos, cache_dir)
    senses = displayed_senses(meta, min_per_sense, len(SENSE_PALETTE))
    if len(senses) < 2:
        raise SystemExit(f"'{word}' ({pos}) has <2 senses with >= {min_per_sense} occurrences.")
    color = {s: SENSE_PALETTE[i] for i, s in enumerate(senses)}
    sense_arr = meta["sense"].to_numpy()
    idx = {s: np.where(sense_arr == s)[0] for s in senses}       # row positions per sense
    rng = np.random.default_rng(0)

    nrow = len(layers)
    fig = make_subplots(rows=nrow, cols=1, vertical_spacing=0.015,
                        row_titles=[f"layer {L}" for L in layers])

    # Legend proxies: one always-visible marker per sense. Clicking one toggles
    # the whole sense (its clouds and points) via the shared legend group, and
    # they survive slider moves (which only flip the per-step data traces).
    for s in senses:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers", legendgroup=s, showlegend=True,
            name=f"{s} — {gloss(s, 42)}", marker=dict(size=9, color=color[s])),
            row=1, col=1)
    n_proxy = len(senses)

    default_step = steps[-1]                         # open on the most-trained checkpoint
    trace_step: list[int] = []                       # step owning each data trace (after the proxies)
    for step in steps:
        vis = step == default_step
        for r, layer in enumerate(layers, start=1):
            P = coords[(step, layer)]
            x0, x1 = float(P[:, 0].min()), float(P[:, 0].max())
            y0, y1 = float(P[:, 1].min()), float(P[:, 1].max())
            mx, my = 0.05 * (x1 - x0 + 1e-9), 0.05 * (y1 - y0 + 1e-9)
            xs = np.linspace(x0 - mx, x1 + mx, HTML_GRID, dtype=np.float32)
            ys = np.linspace(y0 - my, y1 + my, HTML_GRID, dtype=np.float32)
            for s in senses:
                pts = P[idx[s]]
                Z = _kde_grid(pts, xs, ys, rng)
                if Z is not None:
                    fig.add_trace(go.Contour(
                        x=xs, y=ys, z=Z, zmin=0.0, showscale=False, ncontours=6,
                        colorscale=_fill_scale(color[s], 0.5), line_width=0,
                        contours=dict(coloring="fill", showlines=False),
                        hoverinfo="skip", legendgroup=s, showlegend=False, visible=vis),
                        row=r, col=1)
                    trace_step.append(step)
                sp = pts if len(pts) <= SCATTER_MAX else pts[rng.choice(len(pts), SCATTER_MAX, replace=False)]
                fig.add_trace(go.Scattergl(
                    x=sp[:, 0], y=sp[:, 1], mode="markers", hoverinfo="skip",
                    marker=dict(size=3, color=color[s], opacity=0.35, line=dict(width=0)),
                    legendgroup=s, showlegend=False, visible=vis),
                    row=r, col=1)
                trace_step.append(step)
        log.info("KDE HTML step %d done (%d layers)", step, len(layers))

    slider_steps = [dict(method="update", label=f"{step:,}",
                         args=[{"visible": [True] * n_proxy + [ts == step for ts in trace_step]}])
                    for step in steps]

    for r in range(1, nrow + 1):
        xref = "x" if r == 1 else f"x{r}"
        fig.update_xaxes(visible=False, row=r, col=1)
        fig.update_yaxes(visible=False, scaleanchor=xref, row=r, col=1)

    fig.update_layout(
        title=f"“{word}” ({pos}) — per-sense KDE by layer; drag the slider to scrub checkpoints",
        width=680, height=300 * nrow + 140, template="plotly_white",
        legend=dict(title="sense (click to toggle)", itemsizing="constant",
                    yanchor="top", y=1.0, xanchor="left", x=1.02),
        margin=dict(l=20, r=20, t=90, b=70),
        sliders=[dict(active=steps.index(default_step), steps=slider_steps,
                      currentvalue=dict(prefix="checkpoint step: "),
                      pad=dict(t=20), x=0.05, len=0.9)],
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path), include_plotlyjs=True, full_html=True)
    log.info("Wrote %s (%d senses, %d layers, %d steps, %d traces)",
             out_path, len(senses), nrow, len(steps), len(fig.data))


def kde_slider(cfg: dict, word: str, pos: str | None, only: "list[str] | None" = None) -> None:
    """Config-driven entry point: interactive per-layer KDE slider HTML for one word."""
    from .config import run_paths, store
    from .dataset import build_dataset

    p = cfg["plot"]
    _, emb_dir = run_paths(cfg, only)
    cache_dir = (emb_dir / "_npy_cache") if p["npy_cache"] else None
    records = build_dataset(cfg, only)
    out = store(cfg, p["fig_dir"]) / f"{word}_kde_slider.html"
    build_kde_html(records, emb_dir, word, pos, out,
                   cache_dir=cache_dir, min_per_sense=p["min_per_sense"])
