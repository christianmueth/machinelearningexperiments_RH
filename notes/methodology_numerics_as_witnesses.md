# Methodology note: numerics as hypothesis witnesses

Numerical experiments are used solely to witness the hypotheses of the analytic statements (invertibility margins, spectral gaps, and divisor cleanliness on safe rectangles); no step of the analytic proof relies on numerical computation.

Concretely, the computation produces auditable artifacts under a frozen run contract (see `RUN_CONTRACT.md`) that certify the *premises* required to apply the local convergence, selector admissibility, and divisor-control arguments. The analytic conclusions (meromorphic continuation, functional equation/involution, growth control, and rigidity of the quotient) are supplied by standard operator theory and representation theory, and are referenced in the Task A/B notes rather than inferred numerically.

## Policy: tracked vs pointwise branches

Whenever a numerical claim is used as a hypothesis witness for a divisor / argument-principle premise (e.g. “winding is zero on ∂R”), it must be supported by the **tracked/continued branch** output (`wind_track*`).

The **pointwise minimizer** output (`wind_pointwise*`) is recorded only as a diagnostic of selector instability; it may legitimately hit integers due to discontinuous branch switching and must not be used to support divisor-clean claims.
