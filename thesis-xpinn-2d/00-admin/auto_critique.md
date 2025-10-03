# Auto-Critique and Next Steps

* **Absorbing boundaries**: The current sponge mask is simple and may underperform compared with CPML. Future work should implement CPML and tighten the reflection gate accordingly.
* **Finite-difference accuracy**: Spatial derivatives rely on first-order approximations, which can introduce dispersion. Upgrading to staggered second-order (or higher) operators will improve fidelity while remaining CI-friendly.
* **XPINN losses**: The Phase-2 scaffold should incorporate adaptive weighting between data misfit, boundary conditions, and PDE residuals to stabilise training.
