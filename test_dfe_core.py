import numpy as np
import pytest
from dfe_production_kernel import DFEProductionKernel
from dfe_barnes_hut import BarnesHutTree, advance_subsystem_tree

def test_potential_gradient_consistency():
    """
    Requirement 1 Verification: Includes the target at a valid index so the production 
    self-exclusion branch is exercised, but does not independently demonstrate a measurable 
    effect from self-exclusion.
    """
    kernel = DFEProductionKernel(epsilon=1.0)
    
    # Target object is included at index 0 so the production self-exclusion branch is exercised; 
    # the test does not verify a measurable difference.
    positions = [
        np.array([3.0, 4.0, -1.0]),     # Target mass position at index 0
        np.array([0.0, 0.0, 0.0]),       # Source mass 1
        np.array([10.0, -5.0, 2.0])     # Source mass 2
    ]
    masses = [5.0e5, 1.0e20, 2.0e20]
    target_idx = 0
    
    # 1. Compute analytical acceleration at target index
    a_analytic, _ = kernel.compute_acceleration_and_potential(
        positions[target_idx], target_idx, positions, masses
    )
    
    # 2. Compute central finite difference steps around target coordinates
    h = 1e-5
    grad_phi = np.zeros(3)
    
    for axis in range(3):
        coord_plus = np.copy(positions[target_idx])
        coord_minus = np.copy(positions[target_idx])
        
        coord_plus[axis] += h
        coord_minus[axis] -= h
        
        _, pot_plus = kernel.compute_acceleration_and_potential(coord_plus, target_idx, positions, masses)
        _, pot_minus = kernel.compute_acceleration_and_potential(coord_minus, target_idx, positions, masses)
        
        phi_plus = -kernel.G * pot_plus
        phi_minus = -kernel.G * pot_minus
        
        grad_phi[axis] = (phi_plus - phi_minus) / (2.0 * h)
        
    a_numeric = -grad_phi
    
    # Relative vector-norm evaluation prevents precision loss under large accelerations
    relative_error = np.linalg.norm(a_analytic - a_numeric) / np.linalg.norm(a_analytic)
    assert relative_error <= 1.0e-5


def test_mass_independence_trajectory():
    """
    Requirement 2 Verification: Tracks full multi-step semi-implicit Euler integration runs
    to confirm that distinct testing mass choices produce no trajectory deviations.
    Target-only comparison makes the intended particle invariant explicit.
    """
    kernel = DFEProductionKernel(epsilon=0.01)
    dt = 1.0
    steps = 50
    
    # Note: Stated initial speed induces an under-resolved radial plunge trajectory.
    masses_A = [1.0e24, 1.0]     # m_A = 1.0 kg
    masses_B = [1.0e24, 1.0e6]   # m_B = 1.0e6 kg matching SRS spec
    
    # Parallel initialization trajectories
    pos_A = [np.array([0.0, 0.0, 0.0]), np.array([500.0, 0.0, 0.0])]
    v_A = [np.array([0.0, 0.0, 0.0]), np.array([0.0, 100.0, 0.0])]
    
    pos_B = [np.array([0.0, 0.0, 0.0]), np.array([500.0, 0.0, 0.0])]
    v_B = [np.array([0.0, 0.0, 0.0]), np.array([0.0, 100.0, 0.0])]
    
    # Run integration loops, explicitly overwriting source updates to remain static
    for _ in range(steps):
        pos_A, v_A = kernel.advance_subsystem(pos_A, v_A, masses_A, dt)
        pos_A[0] = np.array([0.0, 0.0, 0.0])
        v_A[0] = np.array([0.0, 0.0, 0.0])
        
        pos_B, v_B = kernel.advance_subsystem(pos_B, v_B, masses_B, dt)
        pos_B[0] = np.array([0.0, 0.0, 0.0])
        v_B[0] = np.array([0.0, 0.0, 0.0])
        
    # Target-only index tracking ensures the intended mass invariant is explicitly isolated
    assert np.array_equal(pos_A[1], pos_B[1])
    assert np.array_equal(v_A[1], v_B[1])


def test_point_mass_clock_rate_offset():
    """
    Requirement 3 Verification: Point-mass, first-order weak-field clock-rate consistency 
    check using fixed reference positions and assigned velocities.
    """
    m_earth = 5.9722e24
    r_earth = 6378137.0
    v_ground = 465.1
    
    r_gps = 26560000.0
    v_gps = 3874.0
    
    kernel = DFEProductionKernel(epsilon=0.0)
    
    _, pot_ground = kernel.compute_acceleration_and_potential(np.array([r_earth, 0.0, 0.0]), -1, [[0.0, 0.0, 0.0]], [m_earth])
    alpha_ground = kernel.compute_alpha(pot_ground, np.array([0.0, v_ground, 0.0]))
    
    _, pot_gps = kernel.compute_acceleration_and_potential(np.array([r_gps, 0.0, 0.0]), -1, [[0.0, 0.0, 0.0]], [m_earth])
    alpha_gps = kernel.compute_alpha(pot_gps, np.array([0.0, v_gps, 0.0]))
    
    dt_seconds = 86400.0
    daily_rate_difference = (alpha_gps - alpha_ground) * dt_seconds
    
    # Internal consistency point-mass validation target: 38.541 microseconds/day within 0.1us tolerance
    expected_drift_offset = 38.541e-6
    assert abs(daily_rate_difference - expected_drift_offset) <= 0.1e-6


def test_shapiro_quadrature_finite_path():
    """
    Requirement 4 Verification: Compares a rectangular numerical line integral against 
    the exact finite-path (asinh) expression for a static point source along a prescribed 
    straight path. This test does not exercise a production signal-propagation module.
    """
    m_source = 2.0e30
    b_impact = 7.0e8
    path_L = 1.5e11
    
    kernel = DFEProductionKernel(epsilon=0.0)
    
    num_steps = 200000
    z_coords = np.linspace(-path_L, path_L, num_steps)
    dl = (2.0 * path_L) / (num_steps - 1)
    
    integrated_potential = 0.0
    for z in z_coords:
        current_pos = np.array([b_impact, 0.0, z])
        _, pot_D = kernel.compute_acceleration_and_potential(current_pos, -1, [[0.0, 0.0, 0.0]], [m_source])
        integrated_potential += pot_D * dl
        
    kappa = 2.0
    numerical_delay = (kappa * kernel.G / (kernel.c ** 3)) * integrated_potential
    
    # Exact finite-path mathematical arcsinh comparator check equation
    analytical_delay = (4.0 * kernel.G * m_source / (kernel.c ** 3)) * np.arcsinh(path_L / b_impact)
    
    assert abs(numerical_delay - analytical_delay) <= 1.0e-9


def _random_bodies(n, seed):
    rng = np.random.default_rng(seed)
    positions = rng.uniform(-1.0e3, 1.0e3, (n, 3))
    masses = rng.uniform(1.0e20, 1.0e22, n)
    return positions, masses


def _direct_accelerations(kernel, positions, masses):
    return np.array([
        kernel.compute_acceleration_and_potential(positions[i], i, positions, masses)[0]
        for i in range(len(masses))
    ])


def test_barnes_hut_theta_zero_matches_direct():
    """
    Requirement 5a: With theta = 0 every tree cell is opened, so the tree
    must reproduce direct summation to floating-point rounding.
    """
    positions, masses = _random_bodies(200, seed=0)
    kernel = DFEProductionKernel(epsilon=1.0)
    tree = BarnesHutTree(epsilon=1.0, theta=0.0).build(positions, masses)

    a_direct = _direct_accelerations(kernel, positions, masses)
    a_tree = np.array([tree.compute_acceleration_and_potential(positions[i], i)[0] for i in range(len(masses))])

    relative_error = np.linalg.norm(a_tree - a_direct, axis=1) / np.linalg.norm(a_direct, axis=1)
    assert relative_error.max() <= 1.0e-12


def test_barnes_hut_accuracy_theta_half():
    """
    Requirement 5b: With theta = 0.5 the monopole approximation keeps the
    root-mean-square relative force error below 2% for 500 uniformly
    distributed bodies.
    """
    positions, masses = _random_bodies(500, seed=0)
    kernel = DFEProductionKernel(epsilon=1.0)
    tree = BarnesHutTree(epsilon=1.0, theta=0.5).build(positions, masses)

    a_direct = _direct_accelerations(kernel, positions, masses)
    a_tree = np.array([tree.compute_acceleration_and_potential(positions[i], i)[0] for i in range(len(masses))])

    relative_error = np.linalg.norm(a_tree - a_direct, axis=1) / np.linalg.norm(a_direct, axis=1)
    assert np.sqrt(np.mean(relative_error ** 2)) <= 2.0e-2


def test_barnes_hut_step_matches_direct_step_at_theta_zero():
    """
    Requirement 5c: A tree-based semi-implicit Euler step with theta = 0
    matches the direct-summation step.
    """
    positions, masses = _random_bodies(50, seed=1)
    velocities = np.zeros_like(positions)
    kernel = DFEProductionKernel(epsilon=1.0)

    x_direct, v_direct = kernel.advance_subsystem(positions, velocities, masses, 1.0)
    x_tree, v_tree = advance_subsystem_tree(positions, velocities, masses, 1.0, epsilon=1.0, theta=0.0)

    np.testing.assert_allclose(v_tree, v_direct, rtol=1.0e-12, atol=0.0)
    np.testing.assert_allclose(x_tree, x_direct, rtol=1.0e-12, atol=0.0)


def test_barnes_hut_interaction_count_below_direct():
    """
    Requirement 5d: With theta = 0.5 and 1,000 bodies, the tree performs
    fewer pairwise evaluations than direct summation's N(N-1).
    This is an operation count, not a timing or complexity benchmark.
    """
    positions, masses = _random_bodies(1000, seed=0)
    tree = BarnesHutTree(epsilon=1.0, theta=0.5).build(positions, masses)
    for i in range(len(masses)):
        tree.compute_acceleration_and_potential(positions[i], i)

    n = len(masses)
    assert tree.interaction_count < n * (n - 1)


def test_causal_lattice_field_module_status():
    """Planned target backend skipping hook."""
    pytest.skip("Architectural Plan Notice: Finite-speed lattice field module is currently unwritten.")
