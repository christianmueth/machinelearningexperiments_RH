# Dyadic seed → Möbius → Hecke/BC propagation refactor plan (exploration)

This note is an implementation-facing summary of the intended architecture:

1) **Dyadic seed tower (2^k)**
   - There exists an upstream object indexed by k with intrinsic additive lengths
     \(\ell_k = k\log 2\).
   - This seed tower is *not* “the prime p=2 inserted downstream”; it is the initial
     exponential ladder grammar.

2) **Aggregate tower observable**
   - The raw tower-level data typically mixes primitive and imprimitive contributions.
   - A common signature is a divisor-sum identity on an index set \(n\):

     \[ F(n)=\sum_{d\mid n} f(d). \]

3) **Möbius inversion = primitive extraction**
   - Primitive data is recovered by:

     \[ f(n)=\sum_{d\mid n} \mu(d)\,F(n/d). \]

   - This is the step that should become nontrivial for higher powers and should
     be a *real code path*, not just a conceptual word.

4) **Arithmetic propagation host (Hecke / Bost–Connes)**
   - After primitive extraction, the data should sit inside an arithmetic semigroup
     dynamical system.
   - In code terms, this should mean: a single object that owns the propagation law
     (coprime multiplicativity + prime-power recurrences).

5) **Packet construction**
   - Only after propagation, build packet/scattering objects and only then apply
     Schur/Cayley reductions.

## Current repo status

  directly via `p_mode` and uses `p_eff^(gamma*k)`.
  generator path.

## Step 1.5 — Primitive/imprimitive separation (minimal, honest)

Before attempting any dyadic→primes propagation story, we can add a small, explicit “primitive extractor” layer:

- Treat an aggregate tower observable $F(k)$ (period-$k$ data) as containing both primitive and imprimitive contributions.
- Define primitive contributions via Möbius inversion on the divisor lattice:

  $$P(k) = \sum_{d\mid k} \mu(d)\,F(k/d).$$

This is structurally aligned with dynamical-zeta/Euler-product grammar (primitive orbits), without claiming it uniquely generates primes.

Artifacts:
- tools/dyadic_primitive_extraction_demo.py (computes $P(k)$ from simple dyadic choices like $F(k)=2^k-1$)
- src/arithmetic/mobius.py now includes a type-generic `mobius_invert_divisor_sum_linear` suitable for scalars or small matrices.

## What was added (scaffolding)

- `src/arithmetic/mobius.py`: μ(n), divisors, and Möbius inversion helper.
- `src/arithmetic/seed_tower.py`: dyadic seed tower container + divisor-sum aggregate.
- `src/arithmetic/propagation.py`: propagation interface + Hecke prime-power family helper.
- `tools/seed_tower_refactor_demo.py`: small CLI demo of seed → aggregate → Möbius.

## Next concrete integration steps (proposed)

A) Add a new packet source mode for FE runs that does **not** accept `--primes`.
   - Instead, accept a seed spec (k_max, seed parameters) and a propagation spec.

B) Define the precise object that Möbius inversion acts on in the existing 6×6 model.
   Candidates:
   - a scalar derived from the 2×2 block (trace/det/eigenphase),
   - a scalar derived from the reduced packet (trace of S, logdet(I-K), etc.).

C) Define the propagation map from primitives to prime-indexed data.
   - If the target is a Hecke algebra host, map primitive data into a consistent
     family {T_p} and then use Hecke relations to generate prime powers.
   - If the target is BC, define the semigroup action/time-evolution and show how
     the same primitive data yields a canonical arithmetic indexing.

D) Only then reintroduce Schur/Cayley reduction as an observation map from the
   propagated arithmetic object into local scattering packets.
