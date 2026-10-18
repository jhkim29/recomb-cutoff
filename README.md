# Recombination cutoff

Python companion to [*Cutoff Phenomenon for Inhomogeneous Nonlinear Recombination in Arbitrary Finite Product Spaces*](https://doi.org/10.1007/s10955-026-03640-z), illustrated on 262,144 four-state sites (A, T, C, G). Marginals are drawn independently once, uniformly subject to pᵢ(a) ≥ δ = 0.08, then coupled comonotonically.

The plots compare Q₂-based Monte Carlo lower confidence bounds (99% simultaneous coverage) with the paper's analytic upper bounds on total-variation distance. An integral representation evaluates the quadratic statistic in O(n) time per sequence after O(n log n) preprocessing.

[Demo and explanation](https://jhkim29.github.io/recomb-cutoff/) · [Code](cutoff.py)

```sh
pip install numpy scipy
python cutoff.py
```

Run from a copy of this repository. The script saves or reuses `marginals.npy` and updates the plots in `index.html`.
