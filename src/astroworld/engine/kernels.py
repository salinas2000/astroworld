"""
Numba JIT-compiled kernels for N-body integration.

These kernels fuse the entire step loop (gravity + Verlet update + energy)
into compiled code, eliminating per-step Python overhead.  When Numba is
not installed, all functions degrade to plain NumPy / Python loops
(identical results, slower execution on long runs).

Architecture
------------
The key insight is that for small N (8-10 bodies), explicit double loops
compiled by Numba are *faster* than NumPy broadcasting + einsum because:
  1. No allocation of (N,N,3) temporary tensors per step
  2. Newton's 3rd law halves the pair count (N*(N-1)/2 vs N*(N-1))
  3. All data fits in L1 cache; no BLAS dispatch overhead

The fused kernels (verlet_loop_*) run the entire Velocity Verlet integration
inside a single compiled function, recording snapshots at the specified
interval.  This eliminates ~36M Python interpreter round-trips for a
10,000-year outer-system run.

Performance: ~20-100x speedup over the Python loop for typical use cases.
"""

import numpy as np

try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    # No-op decorator: functions run as plain Python (slower, same results)
    def njit(*args, **kwargs):
        def decorator(fn):
            return fn
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator


# ── Gravity kernel ────────────────────────────────────────────────

@njit(cache=True)
def _compute_accelerations_jit(positions, gm_values, n):
    """
    Compute gravitational accelerations via explicit pairwise loop.

    Exploits Newton's 3rd law: each (i,j) pair computed once.
    For N=9 this means 36 pairs instead of 72.

    Parameters
    ----------
    positions : ndarray (N, 3) — positions in AU
    gm_values : ndarray (N,) — GM values in AU³/day²
    n : int — number of bodies

    Returns
    -------
    acc : ndarray (N, 3) — accelerations in AU/day²
    """
    acc = np.zeros((n, 3))
    for i in range(n):
        for j in range(i + 1, n):
            dx = positions[j, 0] - positions[i, 0]
            dy = positions[j, 1] - positions[i, 1]
            dz = positions[j, 2] - positions[i, 2]
            r_sq = dx * dx + dy * dy + dz * dz
            inv_r3 = r_sq ** (-1.5)
            # Force direction (j→i)
            fx = inv_r3 * dx
            fy = inv_r3 * dy
            fz = inv_r3 * dz
            # Accumulate: a_i += GM_j * f, a_j -= GM_i * f (Newton 3rd)
            acc[i, 0] += gm_values[j] * fx
            acc[i, 1] += gm_values[j] * fy
            acc[i, 2] += gm_values[j] * fz
            acc[j, 0] -= gm_values[i] * fx
            acc[j, 1] -= gm_values[i] * fy
            acc[j, 2] -= gm_values[i] * fz
    return acc


# ── GR correction kernel ─────────────────────────────────────────

@njit(cache=True)
def _compute_gr_correction_jit(positions, velocities, gm_sun, sun_index, c2, n):
    """
    1PN General Relativity correction (EIH simplified, solar term only).

    a_GR_i = (GM / (c² r³)) × [(4GM/r - v²) r⃗ + 4(v⃗·r⃗) v⃗]

    Parameters
    ----------
    positions : ndarray (N, 3)
    velocities : ndarray (N, 3)
    gm_sun : float — GM of the Sun
    sun_index : int — index of Sun in arrays
    c2 : float — speed of light squared (AU/day)²
    n : int — number of bodies

    Returns
    -------
    acc_gr : ndarray (N, 3)
    """
    acc_gr = np.zeros((n, 3))

    for i in range(n):
        if i == sun_index:
            continue

        # Heliocentric position and velocity
        rx = positions[i, 0] - positions[sun_index, 0]
        ry = positions[i, 1] - positions[sun_index, 1]
        rz = positions[i, 2] - positions[sun_index, 2]

        vx = velocities[i, 0] - velocities[sun_index, 0]
        vy = velocities[i, 1] - velocities[sun_index, 1]
        vz = velocities[i, 2] - velocities[sun_index, 2]

        r_sq = rx * rx + ry * ry + rz * rz
        r = r_sq ** 0.5
        v_sq = vx * vx + vy * vy + vz * vz
        v_dot_r = vx * rx + vy * ry + vz * rz

        # prefactor = GM / (c² r³)
        prefactor = gm_sun / (c2 * r * r_sq)

        coeff_r = 4.0 * gm_sun / r - v_sq
        coeff_v = 4.0 * v_dot_r

        acc_gr[i, 0] = prefactor * (coeff_r * rx + coeff_v * vx)
        acc_gr[i, 1] = prefactor * (coeff_r * ry + coeff_v * vy)
        acc_gr[i, 2] = prefactor * (coeff_r * rz + coeff_v * vz)

    return acc_gr


# ── Energy kernel ─────────────────────────────────────────────────

@njit(cache=True)
def _compute_total_energy_jit(positions, velocities, gm_values, masses, n):
    """
    Total mechanical energy (KE + PE).

    KE = 0.5 Σ mᵢ |vᵢ|²
    PE = -Σᵢ<ⱼ GMᵢ mⱼ / |rⱼ - rᵢ|

    Returns float.
    """
    # Kinetic energy
    ke = 0.0
    for i in range(n):
        v_sq = (velocities[i, 0] ** 2 +
                velocities[i, 1] ** 2 +
                velocities[i, 2] ** 2)
        ke += 0.5 * masses[i] * v_sq

    # Potential energy (upper triangle)
    pe = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dx = positions[j, 0] - positions[i, 0]
            dy = positions[j, 1] - positions[i, 1]
            dz = positions[j, 2] - positions[i, 2]
            r = (dx * dx + dy * dy + dz * dz) ** 0.5
            pe -= gm_values[i] * masses[j] / r

    return ke + pe


# ── Fused Verlet loop: Newtonian ──────────────────────────────────

@njit(cache=True)
def verlet_loop_newtonian(
    pos, vel, gm_values, masses,
    n, dt, n_steps, record_interval, n_records,
):
    """
    Fused Velocity Verlet loop (Newtonian gravity only).

    Runs n_steps of integration entirely inside compiled code.
    Records positions, velocities, times, and energies every
    record_interval steps into pre-allocated arrays.

    Parameters
    ----------
    pos : ndarray (N, 3) — initial positions (modified in-place)
    vel : ndarray (N, 3) — initial velocities (modified in-place)
    gm_values : ndarray (N,)
    masses : ndarray (N,)
    n : int — number of bodies
    dt : float — time step (days)
    n_steps : int — total steps
    record_interval : int — record every N steps
    n_records : int — expected number of records

    Returns
    -------
    (times, pos_history, vel_history, energy_history)
    """
    dt2 = dt * dt

    # Pre-allocate output
    times = np.zeros(n_records)
    pos_history = np.zeros((n_records, n, 3))
    vel_history = np.zeros((n_records, n, 3))
    energy_history = np.zeros(n_records)

    # Initial accelerations
    acc = _compute_accelerations_jit(pos, gm_values, n)

    # Record initial state
    record_idx = 0
    t = 0.0
    times[0] = t
    for i in range(n):
        for k in range(3):
            pos_history[0, i, k] = pos[i, k]
            vel_history[0, i, k] = vel[i, k]
    energy_history[0] = _compute_total_energy_jit(pos, vel, gm_values, masses, n)

    # Main integration loop
    for step in range(1, n_steps + 1):
        # Position update: r(t+dt) = r(t) + v(t)·dt + 0.5·a(t)·dt²
        for i in range(n):
            pos[i, 0] += vel[i, 0] * dt + 0.5 * acc[i, 0] * dt2
            pos[i, 1] += vel[i, 1] * dt + 0.5 * acc[i, 1] * dt2
            pos[i, 2] += vel[i, 2] * dt + 0.5 * acc[i, 2] * dt2

        # New accelerations at updated positions
        new_acc = _compute_accelerations_jit(pos, gm_values, n)

        # Velocity update: v(t+dt) = v(t) + 0.5·(a(t) + a(t+dt))·dt
        for i in range(n):
            vel[i, 0] += 0.5 * (acc[i, 0] + new_acc[i, 0]) * dt
            vel[i, 1] += 0.5 * (acc[i, 1] + new_acc[i, 1]) * dt
            vel[i, 2] += 0.5 * (acc[i, 2] + new_acc[i, 2]) * dt

        # Swap accelerations (Numba: just reassign reference)
        acc = new_acc

        t += dt

        # Record snapshot
        if step % record_interval == 0:
            record_idx += 1
            if record_idx < n_records:
                times[record_idx] = t
                for i in range(n):
                    for k in range(3):
                        pos_history[record_idx, i, k] = pos[i, k]
                        vel_history[record_idx, i, k] = vel[i, k]
                energy_history[record_idx] = _compute_total_energy_jit(
                    pos, vel, gm_values, masses, n
                )

    actual = record_idx + 1
    return times[:actual], pos_history[:actual], vel_history[:actual], energy_history[:actual]


# ── Fused Verlet loop: Relativistic ──────────────────────────────

@njit(cache=True)
def verlet_loop_relativistic(
    pos, vel, gm_values, masses,
    gm_sun, sun_index, c2,
    n, dt, n_steps, record_interval, n_records,
):
    """
    Fused Velocity Verlet loop with 1PN GR correction.

    Uses previous-step velocities for the GR term, consistent with
    the existing Python implementation and O(dt²) Verlet accuracy.

    Parameters are the same as verlet_loop_newtonian plus:
    gm_sun : float — GM of the Sun
    sun_index : int — index of Sun
    c2 : float — speed of light squared
    """
    dt2 = dt * dt

    times = np.zeros(n_records)
    pos_history = np.zeros((n_records, n, 3))
    vel_history = np.zeros((n_records, n, 3))
    energy_history = np.zeros(n_records)

    # Initial accelerations (Newton + GR)
    acc_newton = _compute_accelerations_jit(pos, gm_values, n)
    acc_gr = _compute_gr_correction_jit(pos, vel, gm_sun, sun_index, c2, n)
    acc = np.zeros((n, 3))
    for i in range(n):
        for k in range(3):
            acc[i, k] = acc_newton[i, k] + acc_gr[i, k]

    # Record initial state
    record_idx = 0
    t = 0.0
    times[0] = t
    for i in range(n):
        for k in range(3):
            pos_history[0, i, k] = pos[i, k]
            vel_history[0, i, k] = vel[i, k]
    energy_history[0] = _compute_total_energy_jit(pos, vel, gm_values, masses, n)

    for step in range(1, n_steps + 1):
        # Save previous velocities for GR (before position/velocity update)
        prev_vel = vel.copy()

        # Position update
        for i in range(n):
            pos[i, 0] += vel[i, 0] * dt + 0.5 * acc[i, 0] * dt2
            pos[i, 1] += vel[i, 1] * dt + 0.5 * acc[i, 1] * dt2
            pos[i, 2] += vel[i, 2] * dt + 0.5 * acc[i, 2] * dt2

        # New accelerations with GR using previous velocities
        new_newton = _compute_accelerations_jit(pos, gm_values, n)
        new_gr = _compute_gr_correction_jit(pos, prev_vel, gm_sun, sun_index, c2, n)
        new_acc = np.zeros((n, 3))
        for i in range(n):
            for k in range(3):
                new_acc[i, k] = new_newton[i, k] + new_gr[i, k]

        # Velocity update
        for i in range(n):
            vel[i, 0] += 0.5 * (acc[i, 0] + new_acc[i, 0]) * dt
            vel[i, 1] += 0.5 * (acc[i, 1] + new_acc[i, 1]) * dt
            vel[i, 2] += 0.5 * (acc[i, 2] + new_acc[i, 2]) * dt

        acc = new_acc
        t += dt

        if step % record_interval == 0:
            record_idx += 1
            if record_idx < n_records:
                times[record_idx] = t
                for i in range(n):
                    for k in range(3):
                        pos_history[record_idx, i, k] = pos[i, k]
                        vel_history[record_idx, i, k] = vel[i, k]
                energy_history[record_idx] = _compute_total_energy_jit(
                    pos, vel, gm_values, masses, n
                )

    actual = record_idx + 1
    return times[:actual], pos_history[:actual], vel_history[:actual], energy_history[:actual]


# ── Public API ────────────────────────────────────────────────────

def run_verlet_jit(
    pos, vel, gm_values, masses, dt, n_steps,
    record_interval=1, relativistic=False,
    gm_sun=0.0, sun_index=0, c2=0.0,
):
    """
    Run a full Velocity Verlet simulation using JIT-compiled kernels.

    This is the entry point called by NBodySimulation.run() when JIT
    dispatch is active.  Copies input arrays to avoid mutating caller state.

    Parameters
    ----------
    pos, vel : ndarray (N, 3) — initial state
    gm_values : ndarray (N,) — gravitational parameters
    masses : ndarray (N,) — body masses (kg) for energy computation
    dt : float — time step (days)
    n_steps : int — total number of integration steps
    record_interval : int — record state every N steps
    relativistic : bool — include 1PN GR correction
    gm_sun : float — GM of the Sun (only used if relativistic)
    sun_index : int — index of Sun in arrays
    c2 : float — c² in (AU/day)² (only used if relativistic)

    Returns
    -------
    (times, pos_history, vel_history, energy_history)
    """
    n = len(pos)
    n_records = n_steps // record_interval + 1

    # Copy inputs (fused kernels modify in-place)
    pos_work = pos.astype(np.float64).copy()
    vel_work = vel.astype(np.float64).copy()
    gm_work = gm_values.astype(np.float64)
    masses_work = masses.astype(np.float64)

    if relativistic:
        return verlet_loop_relativistic(
            pos_work, vel_work, gm_work, masses_work,
            float(gm_sun), int(sun_index), float(c2),
            n, float(dt), int(n_steps), int(record_interval), int(n_records),
        )
    else:
        return verlet_loop_newtonian(
            pos_work, vel_work, gm_work, masses_work,
            n, float(dt), int(n_steps), int(record_interval), int(n_records),
        )
