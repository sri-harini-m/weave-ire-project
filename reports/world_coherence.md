# News World Coherence Verification

* **Articles sampled**: 500
* **Embedding reduction**: PCA to 8 dimensions
* **Correlation parameter rho**: 0.5

| Metric | Measured Value | Baseline / Target | Status |
|---|---|---|---|
| Nearest-Neighbour Category Match | 50.08% | 16.31% (random) | **3.1x higher than chance** |
| NN Correlation | 0.506 | Target ≈ 0.50 | PASS |
| Min Eigenvalue (C) | -1.40e-15 | >= 0 | PASS (Valid covariance) |
