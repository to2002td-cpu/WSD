"""Interactive Plotly UMAP for one word — a self-contained HTML.

Controls in the page:
  * a "layer · checkpoint" dropdown that swaps the 2-D coordinates,
  * a "style" dropdown that dims every point not written in that style,
  * the legend, which toggles individual senses on/off,
  * hover, showing the sense, style and sentence of each point.

All 2-D coordinates come from the shared UMAP cache; the same points are simply
re-projected per panel, so the file stays reasonably small.
"""

from __future__ import annotations

import logging

import numpy as np

from .umapcache import SENSE_PALETTE, displayed_senses, gloss, load_umap

log = logging.getLogger(__name__)


def _trunc(s: str, n: int = 120) -> str:
    s = str(s)
    return s if len(s) <= n else s[: n - 1] + "…"


def build_html(records, emb_dir, word, pos, out_path, cache_dir=None, min_per_sense: int = 0):
    import plotly.graph_objects as go

    meta, pos, steps, layers, coords = load_umap(emb_dir, records, word, pos, cache_dir)
    senses = displayed_senses(meta, min_per_sense, len(SENSE_PALETTE))
    if len(senses) < 2:
        raise SystemExit(f"'{word}' ({pos}) has <2 senses with >= {min_per_sense} occurrences.")

    sense_arr = meta["sense"].to_numpy()
    style_arr = meta["style"].fillna("?").to_numpy()
    sent_arr = meta["sentence"].to_numpy()
    idx = {s: np.where(sense_arr == s)[0] for s in senses}       # row positions per sense
    styles = sorted({str(v) for v in style_arr[np.concatenate(list(idx.values()))]})

    default = (steps[-1], layers[-1])                            # most-trained, deepest
    fig = go.Figure()
    for i, s in enumerate(senses):
        rows = idx[s]
        P = coords[default][rows]
        custom = np.column_stack([style_arr[rows], [_trunc(t) for t in sent_arr[rows]]])
        fig.add_trace(go.Scattergl(
            x=P[:, 0], y=P[:, 1], mode="markers", name=f"{s} — {gloss(s, 40)}",
            marker=dict(size=4, color=SENSE_PALETTE[i], opacity=0.6, line=dict(width=0)),
            customdata=custom,
            hovertemplate=f"<b>{s}</b><br>style=%{{customdata[0]}}<br>%{{customdata[1]}}<extra></extra>",
            selected=dict(marker=dict(opacity=0.9)),
            unselected=dict(marker=dict(opacity=0.05)),
        ))

    panel_buttons = []
    for layer in layers:
        for step in steps:
            xs = [coords[(step, layer)][idx[s], 0] for s in senses]
            ys = [coords[(step, layer)][idx[s], 1] for s in senses]
            panel_buttons.append(dict(method="restyle", args=[{"x": xs, "y": ys}],
                                      label=f"layer {layer} · step {step:,}"))
    default_panel = layers.index(default[1]) * len(steps) + steps.index(default[0])

    style_buttons = [dict(method="restyle", args=[{"selectedpoints": [None] * len(senses)}],
                          label="all styles")]
    for st in styles:
        sel = [np.where(style_arr[idx[s]] == st)[0].tolist() for s in senses]
        style_buttons.append(dict(method="restyle", args=[{"selectedpoints": sel}], label=st))

    fig.update_layout(
        title=f"“{word}” ({pos}) — {len(senses)} senses in UMAP space "
              f"(pick layer·checkpoint; legend toggles senses)",
        width=1050, height=760, template="plotly_white",
        legend=dict(title="sense (click to toggle)", itemsizing="constant"),
        xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor="x"),
        margin=dict(l=20, r=20, t=120, b=20),
        updatemenus=[
            dict(buttons=panel_buttons, active=default_panel, direction="down",
                 x=0.0, xanchor="left", y=1.16, yanchor="top", showactive=True),
            dict(buttons=style_buttons, active=0, direction="down",
                 x=0.32, xanchor="left", y=1.16, yanchor="top", showactive=True),
        ],
        annotations=[
            dict(text="panel:", x=0.0, xref="paper", y=1.20, yref="paper",
                 showarrow=False, xanchor="left"),
            dict(text="style:", x=0.32, xref="paper", y=1.20, yref="paper",
                 showarrow=False, xanchor="left"),
        ],
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path), include_plotlyjs=True, full_html=True)
    log.info("Wrote %s (%d senses, %d panels, %d styles)",
             out_path, len(senses), len(panel_buttons), len(styles))


def interactive(cfg: dict, word: str, pos: str | None, only: "list[str] | None" = None) -> None:
    """Config-driven entry point: interactive Plotly HTML for one word."""
    from .config import run_paths, store
    from .dataset import build_dataset

    p = cfg["plot"]
    _, emb_dir = run_paths(cfg, only)
    cache_dir = (emb_dir / "_npy_cache") if p["npy_cache"] else None
    records = build_dataset(cfg, only)
    out = store(cfg, p["fig_dir"]) / f"{word}_interactive.html"
    build_html(records, emb_dir, word, pos, out,
               cache_dir=cache_dir, min_per_sense=p["min_per_sense"])
