"""
Tests for 3D Plotly visualization module.

These tests verify that HTML files are generated correctly without
requiring a browser. They check file existence, content structure,
and that body names appear in the output.
"""

import numpy as np
import pytest
from pathlib import Path

from astroworld.data.state import SystemState
from astroworld.engine.simulation import NBodySimulation
from astroworld.analysis.visualization_3d import plot_system_3d
from astroworld.constants import GM_SUN


def _make_simple_state():
    """Create a minimal 2-body system for testing."""
    a = 1.0
    v_circ = np.sqrt(GM_SUN / a)
    return SystemState(
        names=["star", "planet_x"],
        gm_values=np.array([GM_SUN, GM_SUN * 3e-6]),
        positions=np.array([[0, 0, 0], [a, 0, 0]], dtype=float),
        velocities=np.array([[0, 0, 0], [0, v_circ, 0]], dtype=float),
        masses_kg=np.array([1.989e30, 5.972e24]),
        star_index=0,
        colors=["#FDB813", "#2E86C1"],
    )


def _run_short_sim(state, duration=10.0, dt=0.5):
    """Run a short simulation for testing."""
    sim = NBodySimulation(relativistic=False)
    sim.set_initial_state(state)
    return sim.run(duration_days=duration, dt=dt, integrator="verlet")


class TestStatic3D:

    def test_generates_html_file(self, tmp_path):
        """Static 3D plot generates an HTML file."""
        state = _make_simple_state()
        result = _run_short_sim(state)
        out = tmp_path / "test_static.html"

        plot_system_3d(result, state, out, animate=False)

        assert out.exists()
        assert out.stat().st_size > 1000  # Non-trivial HTML

    def test_html_contains_plotly(self, tmp_path):
        """Generated HTML includes Plotly.js library."""
        state = _make_simple_state()
        result = _run_short_sim(state)
        out = tmp_path / "test_plotly.html"

        plot_system_3d(result, state, out)

        content = out.read_text(encoding="utf-8")
        assert "plotly" in content.lower()
        assert "<script" in content

    def test_html_contains_body_names(self, tmp_path):
        """Body names appear in the generated HTML."""
        state = _make_simple_state()
        result = _run_short_sim(state)
        out = tmp_path / "test_names.html"

        plot_system_3d(result, state, out)

        content = out.read_text(encoding="utf-8")
        assert "star" in content
        assert "planet_x" in content

    def test_html_has_deep_space_theme(self, tmp_path):
        """HTML includes black background styling."""
        state = _make_simple_state()
        result = _run_short_sim(state)
        out = tmp_path / "test_theme.html"

        plot_system_3d(result, state, out)

        content = out.read_text(encoding="utf-8")
        assert "black" in content


class TestAnimated3D:

    def test_generates_html_file(self, tmp_path):
        """Animated 3D plot generates an HTML file."""
        state = _make_simple_state()
        result = _run_short_sim(state)
        out = tmp_path / "test_animated.html"

        plot_system_3d(result, state, out, animate=True, max_animation_frames=10)

        assert out.exists()
        assert out.stat().st_size > 1000

    def test_animated_has_play_button(self, tmp_path):
        """Animated HTML includes Play button."""
        state = _make_simple_state()
        result = _run_short_sim(state)
        out = tmp_path / "test_play.html"

        plot_system_3d(result, state, out, animate=True, max_animation_frames=10)

        content = out.read_text(encoding="utf-8")
        assert "Play" in content

    def test_animated_larger_than_static(self, tmp_path):
        """Animated HTML is larger due to frame data."""
        state = _make_simple_state()
        result = _run_short_sim(state)
        static_out = tmp_path / "static.html"
        anim_out = tmp_path / "animated.html"

        plot_system_3d(result, state, static_out, animate=False)
        plot_system_3d(result, state, anim_out, animate=True, max_animation_frames=10)

        assert anim_out.stat().st_size > static_out.stat().st_size


class TestTrappist1_3D:

    def test_trappist1_no_error(self, tmp_path):
        """Full TRAPPIST-1 system renders without error."""
        from astroworld.catalogs.trappist_1 import make_trappist1_state

        state = make_trappist1_state(include_planets=["b", "c", "d"])
        result = _run_short_sim(state, duration=5.0, dt=0.001)
        out = tmp_path / "trappist1.html"

        plot_system_3d(result, state, out)

        assert out.exists()
        content = out.read_text(encoding="utf-8")
        assert "trappist_1" in content
        assert "trappist_1_b" in content
        assert "trappist_1_c" in content
        assert "trappist_1_d" in content
