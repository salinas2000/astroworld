"""
Interactive 3D visualization of gravitational systems using Plotly.

Generates self-contained HTML files that can be opened in any browser
for interactive exploration (rotate, zoom, pan) of orbital trajectories.

Supports two modes:
  - Static: full trajectory trails + markers at final positions
  - Animated: Play/Pause animation with accumulating trails and time slider

Performance notes:
  - Static trails are downsampled to ~5000 points per body max
  - Animation trails use the same downsampled points (not full data)
  - This keeps HTML files under 10 MB even for long simulations
"""

from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from astroworld.engine.simulation import SimulationResult
from astroworld.data.state import SystemState


# ── Deep-space theme ──────────────────────────────────────────────

_AXIS_STYLE = dict(
    backgroundcolor="black",
    gridcolor="rgba(60, 60, 80, 0.4)",
    showbackground=True,
    zerolinecolor="rgba(80, 80, 100, 0.3)",
    tickfont=dict(color="gray", size=10),
    title_font=dict(color="rgba(200, 200, 220, 0.8)", size=12),
)

_DEFAULT_COLORS = [
    "#FDB813",  # gold (star)
    "#E74C3C", "#3498DB", "#27AE60", "#F39C12",
    "#9B59B6", "#E67E22", "#1ABC9C", "#95A5A6",
    "#E91E63", "#00BCD4", "#FF9800", "#8BC34A",
]

# Max points per body in static trails (keeps HTML < 5 MB)
_MAX_TRAIL_POINTS = 5000


# ── Helper functions ──────────────────────────────────────────────

def _get_colors(state: SystemState, n: int) -> list[str]:
    """Get colors from state or use defaults."""
    if state.colors and len(state.colors) == n:
        return list(state.colors)
    return [_DEFAULT_COLORS[i % len(_DEFAULT_COLORS)] for i in range(n)]


def _compute_marker_sizes(gm_values: np.ndarray, star_index: int) -> np.ndarray:
    """
    Compute marker sizes proportional to log(GM).

    Star gets the largest marker; planets scale from 5 to 12.
    """
    sizes = np.full(len(gm_values), 8.0)

    planet_mask = np.ones(len(gm_values), dtype=bool)
    planet_mask[star_index] = False
    planet_gm = gm_values[planet_mask]

    if len(planet_gm) > 0 and planet_gm.max() > 0:
        log_gm = np.log1p(planet_gm / planet_gm.max())
        scaled = 5.0 + 7.0 * log_gm / np.log1p(1.0)
        sizes[planet_mask] = scaled

    sizes[star_index] = 18.0
    return sizes


def _downsample_indices(n_total: int, max_points: int) -> np.ndarray:
    """Generate evenly spaced indices for downsampling."""
    if n_total <= max_points:
        return np.arange(n_total)
    return np.unique(np.linspace(0, n_total - 1, max_points, dtype=int))


def _make_deep_space_layout(title: str, axis_label: str = "AU") -> go.Layout:
    """Create a deep-space themed 3D layout."""
    return go.Layout(
        title=dict(
            text=title,
            font=dict(color="white", size=18),
            x=0.5,
        ),
        paper_bgcolor="black",
        plot_bgcolor="black",
        scene=dict(
            xaxis=dict(**_AXIS_STYLE, title=f"x ({axis_label})"),
            yaxis=dict(**_AXIS_STYLE, title=f"y ({axis_label})"),
            zaxis=dict(**_AXIS_STYLE, title=f"z ({axis_label})"),
            aspectmode="data",
        ),
        legend=dict(
            font=dict(color="white", size=11),
            bgcolor="rgba(0, 0, 0, 0.6)",
            bordercolor="rgba(100, 100, 120, 0.4)",
            borderwidth=1,
        ),
        margin=dict(l=0, r=0, t=50, b=0),
    )


# ── Static 3D plot ────────────────────────────────────────────────

def _build_static_traces(
    result: SimulationResult,
    state: SystemState,
    trail_opacity: float,
) -> list[go.Scatter3d]:
    """Build Plotly traces for static (non-animated) 3D visualization."""
    star_idx = state.star_index
    colors = _get_colors(state, state.n)
    sizes = _compute_marker_sizes(state.gm_values, star_idx)
    traces = []

    # Downsample trail indices (shared across all bodies)
    trail_idx = _downsample_indices(len(result.times), _MAX_TRAIL_POINTS)

    # Star at center
    star_name = result.body_names[star_idx]
    traces.append(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode="markers+text",
        marker=dict(
            size=sizes[star_idx],
            color=colors[star_idx],
            symbol="diamond",
            line=dict(width=1, color="rgba(255,255,255,0.5)"),
        ),
        text=[star_name],
        textposition="top center",
        textfont=dict(color="white", size=11),
        name=star_name,
        hovertemplate=(
            f"<b>{star_name}</b><br>"
            "x: 0.000 AU<br>y: 0.000 AU<br>z: 0.000 AU"
            "<extra></extra>"
        ),
    ))

    # Planet trajectories + final markers
    for i, name in enumerate(result.body_names):
        if i == star_idx:
            continue

        # Star-relative positions (downsampled for trail)
        rel_pos = result.positions[trail_idx, i, :] - result.positions[trail_idx, star_idx, :]
        x, y, z = rel_pos[:, 0], rel_pos[:, 1], rel_pos[:, 2]

        # Orbital trail
        traces.append(go.Scatter3d(
            x=x, y=y, z=z,
            mode="lines",
            line=dict(color=colors[i], width=2),
            opacity=trail_opacity,
            name=name,
            showlegend=False,
            hoverinfo="skip",
        ))

        # Final position marker with label (from LAST frame, not downsampled)
        rel_final = result.positions[-1, i, :] - result.positions[-1, star_idx, :]
        xf, yf, zf = float(rel_final[0]), float(rel_final[1]), float(rel_final[2])
        # Clean display name: "trappist_1_b" → "b", "earth" → "earth"
        display_name = name.split("_")[-1] if "_" in name else name
        traces.append(go.Scatter3d(
            x=[xf], y=[yf], z=[zf],
            mode="markers+text",
            marker=dict(
                size=sizes[i],
                color=colors[i],
                line=dict(width=0.5, color="rgba(255,255,255,0.4)"),
            ),
            text=[display_name],
            textposition="top center",
            textfont=dict(color=colors[i], size=11),
            name=name,
            hovertemplate=(
                f"<b>{name}</b><br>"
                f"x: {xf:.6f} AU<br>"
                f"y: {yf:.6f} AU<br>"
                f"z: {zf:.6f} AU"
                "<extra></extra>"
            ),
        ))

    return traces


# ── Animated 3D plot ──────────────────────────────────────────────

def _build_animated_figure(
    result: SimulationResult,
    state: SystemState,
    trail_opacity: float,
    max_frames: int,
) -> go.Figure:
    """
    Build a Plotly figure with animation frames.

    The key optimization: both the animation frames AND the trail data
    use the same downsampled indices. Each frame k shows trail points
    0..k (from the downsampled set), not from the full dataset.
    This keeps the HTML small regardless of simulation length.
    """
    star_idx = state.star_index
    colors = _get_colors(state, state.n)
    sizes = _compute_marker_sizes(state.gm_values, star_idx)

    # Downsample to max_frames points — these ARE the animation keyframes
    frame_indices = _downsample_indices(len(result.times), max_frames)
    n_frames = len(frame_indices)

    planet_indices = [i for i in range(state.n) if i != star_idx]

    # Pre-compute downsampled star-relative positions for all planets
    # Shape: (n_frames, 3) per planet
    rel_ds = {}
    for i in planet_indices:
        rel_ds[i] = (
            result.positions[frame_indices, i, :]
            - result.positions[frame_indices, star_idx, :]
        )

    times_ds = result.times[frame_indices]

    # Display name mapping: "trappist_1_b" → "b", "earth" → "earth"
    display_names = {}
    for i in planet_indices:
        n = result.body_names[i]
        display_names[i] = n.split("_")[-1] if "_" in n else n

    star_name = result.body_names[star_idx]

    # ── Initial frame traces ──
    initial_traces = [go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode="markers+text",
        marker=dict(size=sizes[star_idx], color=colors[star_idx], symbol="diamond",
                    line=dict(width=1, color="rgba(255,255,255,0.5)")),
        text=[star_name],
        textposition="top center",
        textfont=dict(color="white", size=11),
        name=star_name,
        hoverinfo="name",
    )]

    for i in planet_indices:
        name = result.body_names[i]
        dname = display_names[i]
        # Trail (initially just the first point)
        initial_traces.append(go.Scatter3d(
            x=[float(rel_ds[i][0, 0])],
            y=[float(rel_ds[i][0, 1])],
            z=[float(rel_ds[i][0, 2])],
            mode="lines",
            line=dict(color=colors[i], width=2),
            opacity=trail_opacity,
            name=name,
            showlegend=False,
            hoverinfo="skip",
        ))
        # Current position marker with label
        initial_traces.append(go.Scatter3d(
            x=[float(rel_ds[i][0, 0])],
            y=[float(rel_ds[i][0, 1])],
            z=[float(rel_ds[i][0, 2])],
            mode="markers+text",
            marker=dict(size=sizes[i], color=colors[i],
                        line=dict(width=0.5, color="rgba(255,255,255,0.4)")),
            text=[dname],
            textposition="top center",
            textfont=dict(color=colors[i], size=11),
            name=name,
            hovertemplate=f"<b>{name}</b><br>t: {times_ds[0]:.1f} days<extra></extra>",
        ))

    fig = go.Figure(data=initial_traces)

    # ── Animation frames ──
    # Each frame k shows trail from downsampled points 0..k
    frames = []
    for k in range(n_frames):
        frame_data = [initial_traces[0]]  # Star (static)

        for i in planet_indices:
            name = result.body_names[i]
            # Trail: downsampled points 0..k (inclusive)
            trail = rel_ds[i][:k + 1]
            frame_data.append(go.Scatter3d(
                x=trail[:, 0].tolist(),
                y=trail[:, 1].tolist(),
                z=trail[:, 2].tolist(),
                mode="lines",
                line=dict(color=colors[i], width=2),
                opacity=trail_opacity,
                hoverinfo="skip",
            ))
            # Current position with label
            dname = display_names[i]
            frame_data.append(go.Scatter3d(
                x=[float(rel_ds[i][k, 0])],
                y=[float(rel_ds[i][k, 1])],
                z=[float(rel_ds[i][k, 2])],
                mode="markers+text",
                marker=dict(size=sizes[i], color=colors[i],
                            line=dict(width=0.5, color="rgba(255,255,255,0.4)")),
                text=[dname],
                textposition="top center",
                textfont=dict(color=colors[i], size=11),
                hovertemplate=f"<b>{name}</b><br>t: {times_ds[k]:.1f} days<extra></extra>",
            ))

        frames.append(go.Frame(data=frame_data, name=f"f{k}"))

    fig.frames = frames

    # ── Animation controls ──
    fig.update_layout(
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            x=0.05, y=0.02,
            xanchor="left", yanchor="bottom",
            font=dict(color="white"),
            bgcolor="rgba(40, 40, 60, 0.8)",
            buttons=[
                dict(
                    label="&#9654; Play",
                    method="animate",
                    args=[None, dict(
                        frame=dict(duration=50, redraw=True),
                        fromcurrent=True,
                        transition=dict(duration=0),
                    )],
                ),
                dict(
                    label="&#9724; Pause",
                    method="animate",
                    args=[[None], dict(
                        frame=dict(duration=0, redraw=False),
                        mode="immediate",
                        transition=dict(duration=0),
                    )],
                ),
            ],
        )],
        sliders=[dict(
            active=0,
            yanchor="top", y=0,
            xanchor="left", x=0.1,
            len=0.8,
            currentvalue=dict(
                prefix="Time: ",
                suffix=" days",
                font=dict(color="white", size=12),
                visible=True,
            ),
            font=dict(color="gray"),
            tickcolor="gray",
            steps=[
                dict(
                    args=[[f"f{k}"], dict(
                        frame=dict(duration=0, redraw=True),
                        mode="immediate",
                        transition=dict(duration=0),
                    )],
                    label=f"{times_ds[k]:.1f}",
                    method="animate",
                )
                for k in range(n_frames)
            ],
        )],
    )

    return fig


# ── Public API ────────────────────────────────────────────────────

def plot_system_3d(
    result: SimulationResult,
    state: SystemState,
    save_path: str | Path,
    animate: bool = False,
    trail_opacity: float = 0.4,
    max_animation_frames: int = 200,
) -> None:
    """
    Generate an interactive 3D HTML visualization of the simulation.

    Parameters
    ----------
    result : SimulationResult from a completed simulation run.
    state : SystemState with body metadata (colors, star_index, etc.)
    save_path : Output path for the HTML file.
    animate : If True, add Play/Pause animation with time slider.
    trail_opacity : Opacity for orbital trail lines (0.0 - 1.0).
    max_animation_frames : Maximum number of animation frames
                           (downsampled if needed to keep HTML small).
    """
    save_path = Path(save_path)
    duration = result.times[-1]
    system_name = result.body_names[state.star_index]
    n_bodies = len(result.body_names)

    title = (
        f"{system_name} System "
        f"({n_bodies} bodies, {duration:.1f} days)"
    )

    if animate:
        fig = _build_animated_figure(result, state, trail_opacity, max_animation_frames)
    else:
        traces = _build_static_traces(result, state, trail_opacity)
        fig = go.Figure(data=traces)

    layout = _make_deep_space_layout(title)
    fig.update_layout(layout)

    fig.write_html(
        str(save_path),
        include_plotlyjs=True,
        full_html=True,
    )
