# df-engine

A discrete-frame approximate gravitational dynamics simulator implementing direct-sum softened Newtonian mechanics and a Barnes–Hut octree backend.

This is the reference code for Document 2 (the Discrete Frame Engine) of the Discrete Frame Portfolio. It is an engineering prototype in flat computational coordinates. It is not a replacement for General Relativity, and it is not evidence for the Discrete Frame Hypothesis.

## Files

| File | Contents |
|---|---|
| `dfe_production_kernel.py` | Direct-summation softened-Newtonian force and potential, the first-order weak-field clock-rate proxy (α), and a semi-implicit Euler position/velocity update |
| `dfe_barnes_hut.py` | Barnes–Hut octree force evaluation (monopole approximation) and a tree-based semi-implicit Euler step |
| `test_dfe_core.py` | Automated verification suite (Requirements 1–4 and 5a–5d) |

## Requirements

- Python 3
- NumPy
- pytest

## Running the tests

```
pip install numpy pytest
pytest test_dfe_core.py -v
```

Expected result: **8 passed, 1 skipped**. The skipped test is a placeholder for the planned finite-speed lattice module.

## What the tests check

| Test | Requirement |
|---|---|
| `test_potential_gradient_consistency` | 1 — analytical acceleration matches a finite-difference gradient of the potential |
| `test_mass_independence_trajectory` | 2 — test-body trajectory is independent of its mass (harness-level check) |
| `test_point_mass_clock_rate_offset` | 3 — point-mass clock-rate difference of 38.541 µs/day between a ground and an orbital reference |
| `test_shapiro_quadrature_finite_path` | 4 — numerical line integral matches the exact finite-path asinh expression |
| `test_barnes_hut_theta_zero_matches_direct` | 5a — tree equals direct summation at θ = 0 |
| `test_barnes_hut_accuracy_theta_half` | 5b — RMS relative force error ≤ 2% at θ = 0.5 |
| `test_barnes_hut_step_matches_direct_step_at_theta_zero` | 5c — tree-based step equals direct step at θ = 0 |
| `test_barnes_hut_interaction_count_below_direct` | 5d — tree uses fewer pairwise evaluations than direct summation |

These are internal consistency checks of the implemented code. They do not validate physical truth.

## Not implemented

- Clock-register (τ) integration
- Finite-speed lattice field and visual signal subsystem
- Benchmarks for long-term energy and momentum behaviour, timestep refinement and runtime scaling
