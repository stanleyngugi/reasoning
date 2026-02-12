**

# Pattern Mining for Verified Competition Mathematics

  

> **Scope:** Combinatorics and Number Theory for AIMO 3  

> **Status:** Consolidated reference — replaces all prior docs in this directory  

> **Last updated:** February 2026

  

---

  

## 1. Purpose and Honest Framing

**Execution contract note:** Stage orchestration, confidence tiers, and fallback policy are defined in `ideas/proof_tir/V2_EXECUTION_SPEC.md`. Research-backed assumptions and decision boundaries are captured in `ideas/proof_tir/V2_RESEARCH_BASELINE.md`. This file remains the algorithmic source for combo/NT mining tiers.

  

This document is the single reference for the pattern mining layer of the Trace-to-Lean pipeline. It covers detection, mining, computation, and verification of integer sequence patterns — the core machinery that converts raw execution traces into candidate formulas for Lean 4 verification.

  

**What this layer does:** Given a sequence of integers `[a₁, a₂, ..., aₖ]` produced by an LLM-generated Python trace, it identifies the generative law (recurrence, polynomial, cycle, etc.) and outputs a candidate formula suitable for Lean translation and `native_decide` verification.

  

**What this layer does NOT do:** It does not solve the problem. It does not produce proofs. It produces *hypotheses* that are then checked computationally. The distinction matters — see §11 for an honest assessment of what "verified" actually means here.

  

### 1.1 Realistic Coverage (Combinatorics + Number Theory Only)

  

Based on analysis of AIME 2015–2025 problems, AIMO 1–2 public problem sets, and OEIS sequence classification:

  

| Mining Tier | Target | Coverage of Combo | Coverage of NT | Hallucination Risk |

|---|---|---|---|---|

| Polynomial (Lagrange) | Degree-d sequences | ~15–20% | ~5–10% | **Zero** — deterministic |

| C-finite (Berlekamp-Massey) | Linear recurrences | ~40–50% | ~10–15% | **Zero** — deterministic |

| Holonomic (P-recursive guesser) | Polynomial-coefficient recurrences | ~15–20% | ~5% | **Zero** — deterministic |

| Modular Cycles (Floyd/Brent + Carmichael) | Periodic mod-m sequences | ~5% | ~30–35% | **Zero** — deterministic |

| OEIS Lookup | Named / known sequences | +5–10% | +5–10% | **Low** — database lookup |

| LLM Pattern Recognition | Unstructured | Fallback | Fallback | **Medium** — constrained |

  

**Aggregate estimate:** ~70–80% of combinatorics problems and ~65–75% of number theory problems have their core sequence successfully mined. Of these, only the deterministic tiers (polynomial through modular cycles) produce artifacts suitable for formal verification — roughly **55–65% with formal guarantees**.

  

The remaining 20–35% resist pattern mining because they involve:

- Floor/ceiling/GCD/valuation functions

- Piecewise or parity-dependent formulas

- Constructive arguments, bijections, pigeonhole

- Multi-parameter sequences (sums over divisors, a(n,k) tables)

- Problems requiring genuine mathematical insight, not computation

  

These fall back to standard TIR.

  

### 1.2 The Sequence Hierarchy

  

Every competition math sequence lives somewhere in this hierarchy:

  

```

Polynomial ⊂ C-finite ⊂ Holonomic ⊂ X-recursive ⊂ Computable

```

  

| Class | Definition | Closed Under | Examples | Mining Method |

|---|---|---|---|---|

| **Polynomial** | f(n) is polynomial in n | +, ×, Σ, Δ | n², C(n,k) for fixed k, triangular numbers | Lagrange interpolation |

| **C-finite** | Constant-coefficient linear recurrence | +, ×, Hadamard | Fibonacci, Lucas, tiling counts, 2ⁿ | Berlekamp-Massey |

| **Holonomic** | Polynomial-coefficient linear recurrence | +, ×, Cauchy, Σ | n!, Catalan, Motzkin, derangements, central binomials | Holonomic guesser |

| **X-recursive** | C-finite-coefficient recurrences | Unknown | F(n²), Perrin(n²) | Research frontier |

  

**Key fact:** Polynomial ⊂ C-finite. Every polynomial sequence is C-finite with connection polynomial (1−x)^(d+1). This means BM handles polynomials too — but Lagrange gives you the explicit formula directly, which is better for Lean translation.

  

**Key fact:** C-finite ⊂ Holonomic. Every C-finite sequence is trivially holonomic (the polynomial coefficients happen to be constants). The holonomic guesser subsumes BM but is slower and needs more terms.

  

**Detection strategy:** Try the simplest class first. If it fails, move up the hierarchy.

  

---

  

## 2. Type Detection: Classifying a Sequence Before Mining

  

Before running any mining algorithm, we classify the sequence to route it to the right tier. This is a deterministic pre-processing step, not an LLM call.

  

### 2.1 The Finite Difference Test (Polynomial Detection)

  

Given sequence `[a₀, a₁, ..., aₖ]`, compute successive differences:

  

```

Δ⁰ = [a₀, a₁, a₂, ...]

Δ¹ = [a₁−a₀, a₂−a₁, ...]

Δ² = [Δ¹₁−Δ¹₀, Δ¹₂−Δ¹₁, ...]

...

```

  

If `Δᵈ` is constant (all equal) and `Δᵈ⁺¹` is zero, the sequence is polynomial of degree d.

  

**Terms needed:** d+2 terms detect a degree-d polynomial.

  

**Implementation:**

```python

def detect_polynomial(seq):

    """Returns degree d if polynomial, else None."""

    diffs = list(seq)

    for d in range(len(seq) - 1):

        diffs = [diffs[i+1] - diffs[i] for i in range(len(diffs) - 1)]

        if len(diffs) < 2:

            return None

        if all(x == diffs[0] for x in diffs):

            return d + 1

    return None

```

  

**Why check this first:** Polynomial sequences are also C-finite (BM will find them), but Lagrange gives you the explicit polynomial f(n) = ... directly, which is trivially translatable to Lean. BM gives you only the recurrence, requiring an additional solve step.

  

### 2.2 The Linear Complexity Profile (C-finite Detection)

  

Run BM on the sequence and track the history of the linear complexity L at each step.

  

- **C-finite sequence:** L stabilizes early and stays constant. For a degree-L recurrence, L stabilizes by step 2L and never changes again.

- **Non-C-finite sequence:** L grows roughly as N/2. This is the hallmark of sequences that are holonomic, algebraic, or unstructured.

  

**Diagnostic rule:**

```python

def is_likely_c_finite(seq, threshold=0.4):

    L_values = berlekamp_massey_profile(seq)

    final_L = L_values[-1]

    N = len(seq)

    # If final complexity is less than 40% of sequence length,

    # and hasn't changed in the last N/3 steps, it's likely C-finite

    stable_region = L_values[2*final_L:]

    if final_L < threshold * N and all(l == final_L for l in stable_region):

        return True, final_L

    return False, final_L

```

  

### 2.3 The Ratio Test (Hypergeometric / Holonomic Hint)

  

Compute consecutive ratios `r(n) = a(n+1) / a(n)`:

  

- **Constant ratio:** Geometric sequence (C-finite of order 1)

- **Ratio is linear in n:** Hypergeometric term (e.g., n! has ratio n+1)

- **Ratio is rational in n:** Holonomic candidate (e.g., Catalan has ratio (4n−2)/(n+1))

- **Ratio is irregular:** Not simply classifiable

  

```python

from fractions import Fraction

  

def ratio_test(seq):

    """Compute a(n+1)/a(n) and check if it's rational in n."""

    ratios = []

    for i in range(len(seq) - 1):

        if seq[i] == 0:

            return None  # Can't divide by zero

        ratios.append(Fraction(seq[i+1], seq[i]))

    return ratios

```

  

### 2.4 Modular Cycle Detection (for mod-m problems)

  

If the problem asks for "f(N) mod m" where N is huge, the sequence f(n) mod m is periodic by the pigeonhole principle (the state space is finite). Detect the cycle directly.

  

**Use Brent's algorithm** (36% fewer function evaluations than Floyd):

  

```python

def brent_cycle(f, x0):

    """Returns (preperiod μ, period λ)."""

    power = lam = 1

    tortoise = x0

    hare = f(x0)

    while tortoise != hare:

        if power == lam:

            tortoise = hare

            power *= 2

            lam = 0

        hare = f(hare)

        lam += 1

    # Find preperiod

    tortoise = hare = x0

    for _ in range(lam):

        hare = f(hare)

    mu = 0

    while tortoise != hare:

        tortoise = f(tortoise)

        hare = f(hare)

        mu += 1

    return mu, lam

```

  

**When to use direct math instead of detection:** When m is huge (10¹⁸) but the structure is known (linear recurrence → matrix exponentiation, exponential → Carmichael function). Detection requires iterating up to the period length; direct math takes O(log n).

  

### 2.5 The Detection Cascade

  

```

Input: sequence [a₁, ..., aₖ] (typically k = 20-40)

  

1. POLYNOMIAL? Run finite differences.

   → If Δᵈ constant for d ≤ 8: ROUTE TO TIER 1 (Lagrange)

  

2. C-FINITE? Run Berlekamp-Massey.

   → If L stabilizes at L < k/3 and k ≥ 2L: ROUTE TO TIER 2 (BM)

  

3. MODULAR CYCLE? (If problem is "mod m" type)

   → Run cycle detection on trace mod m: ROUTE TO TIER 4 (Modular)

  

4. HOLONOMIC? Run holonomic guesser.

   → If recurrence found with polynomial coefficients: ROUTE TO TIER 3

  

5. OEIS? Hash first 8-10 terms, lookup in offline database.

   → If unique match found: ROUTE TO TIER 5

  

6. LLM FALLBACK. Ask LLM to recognize the pattern.

   → Constrained output, unverified. ROUTE TO TIER 6

```

  

### 2.6 Hankel Matrix Pre-Routing

  

Before choosing between BM and the holonomic guesser, compute the Hankel determinants of the trace to characterize its generating function structure.

  

Given a sequence `[a₀, a₁, ..., aₖ]`, the order-r Hankel matrix is:

  

```
H_r = [[a_i+j]]  for i,j = 0,...,r-1
```

  

Compute `det(H_r)` for increasing r. The behavior determines the routing:

  

| Hankel Behavior | Interpretation | Route |
|---|---|---|
| `det(H_r) = 0` for all `r > L` | Finite-rank → linear recurrence of order L | **BM (Tier 2)** |
| `det(H_r)` grows regularly (polynomial/exponential) | Rational generating function structure | **Padé approximants** |
| `det(H_r)` chaotic / no pattern | Non-linear or non-rational | **OEIS / LLM / TIR fallback** |

  

```python
from fractions import Fraction

def hankel_determinants(seq, max_order=None):
    """Compute Hankel determinants for pre-routing diagnostic."""
    if max_order is None:
        max_order = len(seq) // 2
    dets = []
    for r in range(1, max_order + 1):
        H = [[Fraction(seq[i + j]) for j in range(r)] for i in range(r)]
        dets.append(_det_exact(H))
    return dets

def hankel_route(seq):
    """Route sequence based on Hankel determinant behavior."""
    dets = hankel_determinants(seq)
    # Find rank: first r where det vanishes and stays zero
    for r, d in enumerate(dets):
        if d == 0 and all(dd == 0 for dd in dets[r:]):
            return "BM", r  # Linear recurrence of order r
    # Check for regular growth (Padé candidate)
    if all(d != 0 for d in dets[:len(dets)//2]):
        return "PADE", None
    return "FALLBACK", None
```

  

**Cost:** O(r³) per determinant, negligible for competition-size traces (r ≤ 20). This diagnostic runs in <1ms and avoids wasting time on the wrong miner.

  

---

  

## 3. Tier 1: Polynomial Mining (Lagrange Interpolation)

  

### 3.1 When This Applies

  

~15–20% of combinatorics problems and ~5–10% of number theory problems produce sequences that are polynomial in n. Common sources:

  

- Path counting on small grids

- Sums of consecutive integers/squares/cubes (Faulhaber)

- Binomial coefficients C(n, k) for fixed small k

- Direct formulas from counting arguments

- Chromatic polynomials of small graphs

  

Typical degree in competition math: ≤ 5. Degree > 8 is extremely rare.

  

### 3.2 The Algorithm

  

Given points (1, a₁), (2, a₂), ..., (k, aₖ), Lagrange interpolation constructs the unique polynomial of degree < k passing through all points.

  

**Use SymPy's exact rational interpolation**, not NumPy's floating-point polyfit:

  

```python

from sympy import interpolate, Symbol, Rational

  

n = Symbol('n')

points = [(i+1, seq[i]) for i in range(len(seq))]

poly = interpolate(points, n)

```

  

**Why not NumPy:** For degree ≥ 6, floating-point errors in coefficient reconstruction cause catastrophic failures. A degree-8 polynomial with integer values can have rational coefficients with denominators in the hundreds of thousands. `numpy.polyfit` will silently corrupt these.

  

### 3.3 Integer-Valued Polynomials and the Binomial Basis

  

Many competition formulas are integer-valued polynomials with non-integer coefficients. The canonical example: f(n) = n(n+1)/2. The polynomial has coefficient 1/2, but always produces integers.

  

For Lean translation, integer-valued polynomials are better expressed in the **binomial basis**:

  

$$f(n) = \sum_{k=0}^{d} c_k \binom{n}{k}$$

  

where all cₖ are integers. This avoids rational arithmetic in Lean entirely.

  

**Conversion:** The cₖ are exactly the finite differences Δᵏf(0):

  

```python

def to_binomial_basis(seq):

    """Convert polynomial sequence to binomial basis coefficients."""

    coeffs = []

    diffs = list(seq)

    for k in range(len(seq)):

        coeffs.append(diffs[0])

        diffs = [diffs[i+1] - diffs[i] for i in range(len(diffs)-1)]

        if not diffs:

            break

    return coeffs  # f(n) = Σ coeffs[k] * C(n, k)

```

  

### 3.4 Overfitting Guard

  

Lagrange interpolation **always** produces a polynomial of degree k-1 from k points. If the sequence is not actually polynomial (e.g., it's exponential), the interpolation will "succeed" but the formula will be wrong for n > k.

  

**Guard:** After interpolation of degree d using d+1 points, check the formula against the remaining trace points. If it matches all k points (where k > d+1), the polynomial is likely correct. If k ≤ d+1, the result is speculative.

  

**Rule of thumb:** For a degree-d polynomial, use at least 2d+2 trace points (d+1 for interpolation, d+1 for validation).

  

### 3.5 Lean Translation

  

```python

# Python: f(n) = n*(n+1)*(2*n+1)//6  (sum of squares)

# Lean:   fun n => n * (n + 1) * (2 * n + 1) / 6

```

  

For binomial basis:

```python

# Python: f(n) = 3*C(n,2) + C(n,1) + 1

# Lean:   fun n => 3 * Nat.choose n 2 + n + 1

```

  

**Integer division safety:** If the polynomial is expressed as p(n)/d where d divides p(n) for all natural n, Lean's natural number division `a / b` will produce correct results. But verify this computationally — a wrong divisor produces silently wrong Lean code.

  

---

  

## 4. Tier 2: C-finite Mining (Berlekamp-Massey)

  

This is the workhorse. ~40–50% of combinatorics problems produce C-finite sequences. These are sequences satisfying constant-coefficient linear recurrences: tiling problems, Fibonacci-style DP, partition-into-parts with fixed structure, linear recurrence formulas from generating functions.

  

### 4.1 The Algorithm

  

**Core guarantee (Massey, 1969):** Given a C-finite sequence with minimal recurrence of degree L, BM uniquely identifies the recurrence from N ≥ 2L terms.

  

**The iterative update:**

  

Maintain: connection polynomial C(x), backup polynomial B(x), complexity L, last-change index m, last discrepancy b.

  

Initialize: C(x) = 1, B(x) = 1, L = 0, m = −1, b = 1.

  

For n = 0, 1, ..., N−1:

1. Compute discrepancy: d = sₙ + Σᵢ₌₁ᴸ Cᵢ sₙ₋ᵢ

2. If d = 0: no update needed

3. If d ≠ 0:

   - **Fraction-free update:** C_new(x) = b·C(x) − d·x^(n−m)·B(x)

   - If 2L ≤ n: increase complexity

     - L_new = n + 1 − L

     - B(x) ← C_old(x), b ← d, m ← n

  

**Implementation:** Use the fraction-free variant. Never use floating-point.

  

```python

def berlekamp_massey(seq):

    """Fraction-free Berlekamp-Massey. Returns recurrence coefficients."""

    N = len(seq)

    C = [1]  # Connection polynomial

    B = [1]  # Backup polynomial

    L = 0    # Current complexity

    m = -1   # Last length change

    b = 1    # Last discrepancy

    for n in range(N):

        # Compute discrepancy

        d = seq[n]

        for i in range(1, L + 1):

            if i < len(C):

                d += C[i] * seq[n - i]

        if d == 0:

            continue

        # Compute update

        shift = n - m

        T = [0] * (shift) + [b_coeff * (-d) for b_coeff in B]

        # Pad to same length

        new_C = [0] * max(len(C), len(T))

        for i in range(len(C)):

            new_C[i] += C[i] * b

        for i in range(len(T)):

            new_C[i] += T[i]

        if 2 * L <= n:

            B = C[:]

            L = n + 1 - L

            b = d

            m = n

        C = new_C

    # Extract recurrence: a(n) = -C[1]/C[0]*a(n-1) - C[2]/C[0]*a(n-2) - ...

    return C, L

```

  

### 4.2 Complexity

  

- **Time:** O(N²) — negligible for N ≤ 1000

- **Space:** O(N)

- **Optimized (Half-GCD):** O(N log² N) — overkill for competition math

  

### 4.3 Failure Modes

  

| Sequence Type | BM Behavior | Diagnostic | Action |

|---|---|---|---|

| Polynomial (n², n³) | **Succeeds.** Finds (1−x)^(d+1) | Connection poly is power of (1−x) | Use Lagrange instead for direct formula |

| Exponential (2ⁿ) | **Succeeds.** Finds 1−2x | Degree 1, clean | Proceed normally |

| Factorial (n!) | **Fails.** L grows as N/2 | Linear complexity profile | Route to holonomic guesser |

| Catalan numbers | **Fails.** L ≈ N/2 | Algebraic GF, not rational | Route to holonomic guesser |

| Moser's circle (1,2,4,8,16,31,...) | **Misleading at N=5.** Finds wrong degree, self-corrects at N=6 | Degree jumps after stabilizing | Always validate with k > 2L terms |

| Partition numbers | **Fails.** L ≈ N/2 | Non-holonomic GF | Route to OEIS/LLM |

  

**The Moser trap is the canonical warning:** The sequence 1, 2, 4, 8, 16, 31, 57, ... looks like 2ⁿ for the first 5 terms. BM finds degree 1. The 6th term (31 ≠ 32) forces a massive degree update. The true recurrence has degree 5 (it's a sum of binomial coefficients, hence polynomial of degree 4).

  

**Pipeline rule:** Never accept a BM result unless k ≥ 2L. If k < 2L, generate more trace terms before proceeding.

  

### 4.4 Composite Moduli (Reeds-Sloane)

  

Standard BM requires field arithmetic (division). Over ℤₘ with composite m, division can fail (2 has no inverse mod 10).

  

**Practical strategy for the pipeline:**

- If m is **prime**: Standard BM over ℤₚ works perfectly.

- If m is **square-free** (p₁·p₂·...·pₖ): Run BM modulo each prime separately, reconstruct via CRT.

- If m has **prime powers** (pᵏ): Use fraction-free BM over ℤ, then reduce. Accept possibly non-minimal degree.

  

For AIMO 3, answers are integers 0–99999. Common moduli in problems: 10, 100, 1000, 10000, primes. The CRT decomposition approach handles all practical cases.

  

### 4.5 Computing the N-th Term Efficiently

  

Once BM gives recurrence a(n) = c₁a(n−1) + ... + cₗa(n−L):

  

**Matrix Exponentiation (L < 50):**

Encode as companion matrix M. Compute M^N in O(L³ log N).

  

```python

def matrix_pow_mod(M, n, mod):

    """Binary exponentiation for k×k matrix."""

    k = len(M)

    result = [[1 if i == j else 0 for j in range(k)] for i in range(k)]

    while n > 0:

        if n % 2 == 1:

            result = mat_mul_mod(result, M, mod)

        M = mat_mul_mod(M, M, mod)

        n //= 2

    return result

```

  

**Bostan-Mori (L ≥ 50):**

Uses the rational function P(x)/Q(x) representation. Halves the target index at each step via even/odd splitting. Runs in O(L log L log N) with FFT-based polynomial multiplication. Vastly superior for high-degree recurrences.

  

**Recommendation:** Always start with matrix exponentiation. Switch to Bostan-Mori only if L > 50 (rare in competition math — typical degree is 2–5, occasionally up to 10–15).

  

---

  

## 5. Tier 3: Holonomic Mining (P-Recursive Guesser)

  

**This is the critical gap in the current pipeline.** The holonomic guesser captures ~15–20% of combinatorics sequences that BM misses: Catalan numbers, factorial-based counts, derangements, Motzkin numbers, central binomial coefficients, Apéry numbers, most DP sequences with non-constant transition rules, and binomial sums.

  

### 5.1 What Holonomic Sequences Are

  

A sequence (aₙ) is **holonomic** (P-recursive, D-finite) if it satisfies a linear recurrence with polynomial coefficients:

  

$$p_r(n) \cdot a_{n+r} + p_{r-1}(n) \cdot a_{n+r-1} + \dots + p_0(n) \cdot a_n = 0$$

  

where p₀, ..., pᵣ are polynomials in n (not constants).

  

| Sequence | Recurrence | Type |

|---|---|---|

| Fibonacci | F(n+2) − F(n+1) − F(n) = 0 | C-finite (constant coefficients) |

| Factorial | a(n+1) − (n+1)·a(n) = 0 | Holonomic, not C-finite |

| Catalan | (n+2)·C(n+1) − 2(2n+1)·C(n) = 0 | Holonomic, not C-finite |

| Derangements | D(n+1) − n·D(n) − (−1)ⁿ = 0 | Holonomic (order 2 after clearing) |

| Central binomials C(2n,n) | (n+1)·a(n+1) − 2(2n+1)·a(n) = 0 | Holonomic |

  

**Prevalence:** ~25% of OEIS sequences are holonomic (Salvy, 2005). In competition combinatorics, estimated 30–50% of counting problems produce holonomic sequences.

  

### 5.2 The Guessing Algorithm (Kauers Ansatz)

  

**The algorithm is conceptually simple.** It reduces holonomic guessing to solving a homogeneous linear system over ℚ.

  

**Input:** Sequence a₀, a₁, ..., aₙ.

**Parameters:** Maximum recurrence order r, maximum polynomial degree d.

**Output:** Polynomials p₀(n), ..., pᵣ(n) of degree ≤ d such that Σ pᵢ(n)·a(n+i) = 0.

  

**Method:**

  

1. **Make an ansatz:** Write pᵢ(n) = Σⱼ₌₀ᵈ cᵢⱼ nʲ. This introduces (r+1)(d+1) unknown coefficients cᵢⱼ.

  

2. **Substitute known terms:** For each value of n = 0, 1, 2, ..., the recurrence equation becomes one linear equation in the unknowns cᵢⱼ. We get as many equations as we have valid evaluation points (roughly N − r).

  

3. **Solve the linear system:** Find the kernel (null space) of the resulting matrix over ℚ. If the kernel is non-trivial (dimension ≥ 1), we have a candidate recurrence. If the kernel is trivial, no holonomic recurrence of order ≤ r and degree ≤ d exists for this data.

  

4. **Uniqueness:** Need at least (r+1)(d+2) − 2 terms for the system to be sufficiently overdetermined.

  

**Search strategy:** Iterate over increasing (r, d) pairs in a reasonable order:

  

```

(r=1, d=0), (r=1, d=1), (r=2, d=0), (r=1, d=2), (r=2, d=1), (r=3, d=0), ...

```

  

Stop at the first (r, d) that yields a non-trivial kernel.

  

### 5.3 Implementation

  

**No pure-Python implementation exists as of February 2026.** The gold standard is `ore_algebra` for SageMath (~8 GB, unusable on Kaggle). SymPy's holonomic module handles closure properties but does NOT do guessing. Mathematica's `Guess.m` (Kauers) works but is not Python.

  

**We must build this.** The core is ~200–400 lines of Python:

  

```python

from fractions import Fraction

  

def guess_holonomic(seq, max_order=4, max_degree=3):

    """

    Guess a holonomic recurrence for seq.

    Returns list of polynomial coefficients [(d, [(power, coeff), ...]), ...]

    or None if no recurrence found.

    """

    N = len(seq)

    for r in range(1, max_order + 1):

        for d in range(0, max_degree + 1):

            num_unknowns = (r + 1) * (d + 1)

            num_equations = N - r

            if num_equations < num_unknowns + 1:

                continue  # Not enough data

            # Build the matrix: each row is one evaluation of the ansatz

            # Column for c_{i,j} corresponds to n^j * a(n+i)

            matrix = []

            for n_val in range(num_equations):

                row = []

                for i in range(r + 1):

                    a_val = seq[n_val + i]

                    for j in range(d + 1):

                        row.append(Fraction(n_val ** j * a_val))

                matrix.append(row)

            # Find kernel of this matrix over Q

            kernel = rational_kernel(matrix)

            if kernel:

                # Decode the kernel vector into polynomial coefficients

                vec = kernel[0]

                polys = []

                idx = 0

                for i in range(r + 1):

                    coeffs = {}

                    for j in range(d + 1):

                        if vec[idx] != 0:

                            coeffs[j] = vec[idx]

                        idx += 1

                    polys.append(coeffs)

                # Validate: check against extra terms

                if validate_holonomic(seq, polys, r):

                    return r, d, polys

    return None

  

def rational_kernel(matrix):

    """Find null space of a matrix over Q using Gaussian elimination."""

    # Standard row reduction over Fraction, return basis of kernel

    # ... (standard linear algebra, ~80 lines)

    pass

```

  

**Performance:** For typical competition parameters (r ≤ 4, d ≤ 3, N ≤ 40), the matrix is at most 36 × 20. Gaussian elimination over ℚ is instantaneous. The entire guessing loop runs in < 100ms.

  

### 5.4 Verification of Holonomic Recurrences

  

Once we have p₀(n)·a(n) + p₁(n)·a(n+1) + ... + pᵣ(n)·a(n+r) = 0, verification in Lean is straightforward:

  

```lean

def holonomic_check (n : Nat) : Bool :=

  let p0 := (3 * n + 2)  -- example polynomial coefficient

  let p1 := -(n + 2)     -- example

  p0 * a n + p1 * a (n + 1) == 0

  

theorem recurrence_holds : (List.range 100).all holonomic_check = true := by

  native_decide

```

  

This verifies that the recurrence holds for n = 0, ..., 99. Since the recurrence has polynomial coefficients, `native_decide` evaluates polynomial arithmetic — well within its capabilities.

  

### 5.5 Computing the N-th Term of a Holonomic Sequence

  

Unlike C-finite sequences, holonomic sequences don't have constant-coefficient recurrences, so matrix exponentiation doesn't directly apply. Options:

  

1. **Direct iteration:** Compute a(0), a(1), ..., a(N) using the recurrence. O(N·r) time. Fine if N ≤ 10⁶.

2. **Fast evaluation (advanced):** Binary splitting methods can compute a(N) in O(M(N) log²N) where M(N) is multiplication cost. Only needed for N > 10⁶.

3. **For AIMO:** The answer itself is 0–99999, so even if N is large, we compute a(N) mod m. Holonomic recurrences mod m are still holonomic. For very large N, combine with modular periodicity (the sequence mod m must eventually become periodic, though the period can be large).

  

---

  

## 6. Tier 4: Modular Cycle Mining

  

~30–35% of number theory problems reduce to computing f(N) mod m for astronomically large N. The fundamental insight: any deterministic recurrence over a finite state space must eventually cycle.

  

### 6.1 Theory: Why Modular Sequences Cycle

  

**Pigeonhole:** For a k-th order recurrence over ℤₘ, the state (aₙ, aₙ₊₁, ..., aₙ₊ₖ₋₁) lives in ℤₘᵏ. There are mᵏ possible states. After mᵏ + 1 steps, some state must repeat, and the sequence is periodic from that point.

  

**Much tighter bounds exist for specific structures:**

  

| Structure | Sequence | Period Bound | Key Function |

|---|---|---|---|

| Exponential aⁿ mod m | 7ⁿ mod 100 | Divides λ(m) | Carmichael function |

| Fibonacci mod m | F(n) mod m | π(m) ≤ 6m | Pisano period |

| General order-k recurrence mod m | Various | Divides m^k − 1 | Matrix order |

| Polynomial P(n) mod m | n² mod 8 | Divides m | Trivial |

| Tower aᵇᶜ mod m | 7^(7^7) mod 100 | Recursive φ/λ | Generalized Euler |

  

### 6.2 The Carmichael Function λ(m)

  

The Carmichael function λ(m) is the smallest positive integer such that aᴸ ≡ 1 (mod m) for all a coprime to m. It provides a tighter bound than Euler's φ(m).

  

| m | φ(m) | λ(m) | Ratio |

|---|---|---|---|

| 100 | 40 | 20 | 2× tighter |

| 1000 | 400 | 100 | 4× tighter |

| 10000 | 4000 | 500 | 8× tighter |

  

**Computation:**

- λ(2) = 1, λ(4) = 2, λ(2ᵏ) = 2ᵏ⁻² for k ≥ 3

- λ(pᵏ) = φ(pᵏ) = pᵏ⁻¹(p−1) for odd prime p

- λ(m) = lcm(λ(p₁ᵉ¹), ..., λ(pₖᵉᵏ)) for m = p₁ᵉ¹ · ... · pₖᵉᵏ

  

```python

from sympy.ntheory import reduced_totient  # This IS Carmichael's λ

```

  

### 6.3 The Pisano Period π(m)

  

For Fibonacci numbers mod m:

- π(10) = 60 (last digit of Fibonacci repeats every 60)

- π(100) = 300

- π(10ᵏ) = 15 · 10ᵏ⁻¹ for k ≥ 1

  

**Computation:** Factor m, compute π(pᵏ) for each prime power, take lcm. For π(p), check divisors of p−1 (if 5 is a QR mod p) or 2(p+1) (otherwise).

  

### 6.4 Tower Exponentiation (a↑↑k mod m)

  

"Find 7^(7^7) mod 100" — the classic competition pattern.

  

**Algorithm (recursive Euler reduction):**

1. To compute aᵇ mod m: reduce b mod λ(m), then compute aᵇ ᵐᵒᵈ λ⁽ᵐ⁾ mod m

2. But b itself might be huge, so compute b mod λ(m) by recursing: compute b mod λ(m), which requires computing the exponent mod λ(λ(m)), etc.

3. The recursion terminates because λ(λ(...λ(m)...)) reaches 1 after O(log m) steps.

  

**The non-coprime trap:** When gcd(a, m) ≠ 1, Euler's theorem doesn't directly apply. Use the **Generalized Euler Theorem:**

  

$$a^n \equiv a^{(n \bmod \lambda(m)) + \lambda(m)} \pmod{m} \quad \text{for } n \geq \log_2 m$$

  

The "+λ(m)" in the exponent ensures we stay in the periodic region, past any preperiod.

  

### 6.5 Lean Verification for Modular Patterns

  

**Critical safety measure:** Naive `native_decide` on 7^10000 will compute the full integer before reducing mod m, potentially consuming all available RAM.

  

**Fix:** Always inject efficient `powMod` into every Lean file:

  

```lean

def powMod (base exp mod : Nat) : Nat :=

  if mod == 0 then 0 else

  if mod == 1 then 0 else

  match exp with

  | 0 => 1 % mod

  | _ =>

    let half := powMod base (exp / 2) mod

    let result := (half * half) % mod

    if exp % 2 == 1 then (result * base) % mod else result

```

  

**Verification template:**

```lean

set_option maxRecDepth 10000

set_option maxHeartbeats 0

  

-- Verify: 7^n mod 100 has period 4

theorem period_check :

  (List.range 100).all (fun n =>

    powMod 7 n 100 == powMod 7 (n % 4) 100) = true := by

  native_decide

```

  

---

  

## 7. Tier 5: OEIS Lookup (Offline)

  

The OEIS contains ~393,000 sequences with formulas, recurrences, and generating functions. It is **trivially deployable offline** for AIMO 3.

  

### 7.1 Deployment

  

Upload `stripped.gz` (~15 MB) and `names.gz` (~6 MB) as a Kaggle dataset. Total: ~21 MB compressed, ~107 MB uncompressed.

  

### 7.2 Hash-Based Lookup

  

```python

import gzip

  

def build_oeis_index(stripped_path, n_terms=8):

    """Build dict: tuple of first n terms → (A-number, full_terms)."""

    index = {}

    with open(stripped_path) as f:

        for line in f:

            if line.startswith('#') or ',' not in line:

                continue

            parts = line.strip().split(',')

            a_num = parts[0].strip()

            terms = [int(x) for x in parts[1:] if x.strip()]

            if len(terms) >= n_terms:

                key = tuple(terms[:n_terms])

                index.setdefault(key, []).append((a_num, terms))

    return index

  

# Usage:

# index = build_oeis_index('stripped')

# matches = index.get(tuple(my_sequence[:8]), [])

```

  

**Match statistics:**

- 3–4 common terms: hundreds of matches (useless)

- 6–8 distinctive terms: 1–5 matches (usually sufficient)

- 10+ terms: almost always unique (if the sequence is in OEIS)

  

### 7.3 What OEIS Gives You

  

Each entry has: name, formula, recurrence, generating function, Maple/Mathematica/Python code, and references. For the pipeline, the most valuable field is the **formula** — it can often be parsed directly into a candidate for Lean translation.

  

### 7.4 Limitations

  

- OEIS formulas are community-curated, not formally verified (~85% semantically correct per Luzhnica & Kohlhase study)

- OEIS matches tell you what the sequence IS, not WHY it appears in this problem

- Shifted/scaled variants require multiple lookups (try offsets ±1, divide by GCD, take differences)

- This tier provides **hypotheses**, not **certificates** — the formula must still pass Lean verification

  

---

  

## 8. Tier 6: LLM Pattern Recognition (Fallback)

  

When all deterministic tiers fail, ask the LLM to recognize the pattern from the trace and problem context.

  

### 8.1 Prompting Strategy

  

```

Given the sequence [1, 1, 2, 5, 14, 42, 132, ...] arising from [problem context]:

1. What named sequence is this? (e.g., Catalan, Bell, Motzkin)

2. What is the closed-form formula?

3. What recurrence does it satisfy?

Output ONLY the formula in Python syntax.

```

  

### 8.2 When This Works

  

LLMs have memorized the ~500–1000 most famous sequences (Fibonacci, Catalan, primes, factorials, Bell numbers, etc.) with reasonable reliability. For the long tail of 390K+ OEIS sequences, LLM recall is unreliable.

  

### 8.3 Verification Requirement

  

LLM outputs are **untrusted hypotheses**. They must be:

1. Evaluated against the trace (all terms must match exactly)

2. Checked against additional computed terms (compute f(n) for n beyond the trace)

3. If both pass, submitted to Lean for `native_decide` verification

  

If the LLM's formula fails trace validation, discard it and fall back to TIR.

  

---

  

## 9. From Pattern to Lean: The Translation Layer

  

### 9.1 What We Translate

  

We translate **formulas**, not **proofs**. Both Python and Lean are formal languages with precise syntax. The translation is mechanical.

  

| Python | Lean |

|---|---|

| `*` | `*` |

| `+`, `-` | `+`, `-` |

| `//` (integer division) | `/` (for Nat) |

| `**` | `^` |

| `% m` | `% m` |

| `math.factorial(n)` | `Nat.factorial n` |

| `math.comb(n, k)` | `Nat.choose n k` |

  

### 9.2 The SymPy-to-Lean Printer

  

SymPy expression trees have structural nuances that must be handled:

  

- **Subtraction:** SymPy represents a − b as Add(a, Mul(−1, b)). Detect and print as subtraction.

- **Division:** SymPy represents a/b as Mul(a, Pow(b, −1)). Detect and print with appropriate type casting.

- **Rational numbers:** SymPy's Rational(p, q) must become `(p : ℚ) / q` in Lean — or better, rewrite to avoid rationals entirely using the binomial basis or clearing denominators.

- **N-ary operators:** SymPy's Add and Mul are n-ary. Lean operators are binary. Fold them: Add(a, b, c) → `a + (b + c)`.

  

```python

class Lean4Printer:

    def print_expr(self, expr):

        if isinstance(expr, sympy.Integer):

            return str(expr)

        elif isinstance(expr, sympy.Add):

            terms = [self.print_expr(a) for a in expr.args]

            return ' + '.join(terms)

        elif isinstance(expr, sympy.Mul):

            factors = [self.print_expr(a) for a in expr.args]

            return ' * '.join(factors)

        elif isinstance(expr, sympy.Pow):

            base = self.print_expr(expr.args[0])

            exp = self.print_expr(expr.args[1])

            return f'{base} ^ {exp}'

        elif isinstance(expr, sympy.Symbol):

            return str(expr)

        else:

            raise ValueError(f"Unsupported: {type(expr)}")

```

  

### 9.3 Type Safety

  

The most dangerous translation failure: SymPy is dynamically typed. Lean is statically typed.

  

- **Nat vs Int:** If the formula can produce negative intermediate values, use Int, not Nat. Nat subtraction in Lean is truncating (3 − 5 = 0).

- **Integer division:** Lean's Nat division truncates. If your formula is n(n+1)/2, this works because n(n+1) is always even. But n(n+2)/3 does NOT always produce integers. Verify integrality computationally before assuming Nat division is safe.

- **Modular arithmetic:** Use Lean's `%` operator. Ensure consistency with Python's `%` (both return non-negative results for non-negative inputs).

  

### 9.4 Pre-Verification in Python

  

Before generating Lean code, verify the candidate formula against the trace in Python:

  

```python

def pre_verify(formula, trace, var='n'):

    """Check formula matches all trace values."""

    n = sympy.Symbol(var)

    for i, expected in enumerate(trace):

        actual = formula.subs(n, i + 1)  # or i, depending on offset

        if actual != expected:

            return False, i + 1

    return True, len(trace)

```

  

If pre-verification fails, don't waste time on Lean. Go back to mining.

  

---

  

## 10. The Verification Layer: native_decide

  

### 10.1 What native_decide Actually Does

  

When you write `theorem check : P = true := by native_decide`, Lean:

  

1. **Synthesizes** a `Decidable P` instance (type class inference)

2. **Compiles** the decision procedure to C++ via the Lean compiler pipeline:

   - Lean IR → LCNF (A-normal form) → Mono IR → C/bytecode

3. **Executes** the compiled code using GMP for big integer arithmetic

4. If it returns `true`, accepts the proof via the `Lean.ofReduceBool` axiom

  

### 10.2 The Trusted Code Base (TCB) Expansion

  

Standard Lean proofs trust only the kernel (~few thousand lines of C++). `native_decide` expands the TCB to include:

  

- The Lean compiler (30K+ lines)

- The Lean interpreter/VM

- The runtime system (GC, object model)

- GMP (big integer arithmetic)

- Any `@[implemented_by]` or `@[extern]` code

  

**For competition math, this is acceptable.** The probability of a compiler bug causing a false positive is negligible compared to LLM reasoning errors (5–10%). But it means we should say "computationally verified" rather than "formally proved."

  

**Known soundness risk:** `native_decide` can theoretically prove `False` via compiler bugs or non-deterministic I/O exploits. Seed-Prover 1.5 explicitly avoids it as "unsafe." For AIMO, this risk is acceptable. For publishing to Mathlib, it is not.

  

### 10.3 Performance Characteristics

  

| Scenario | decide (kernel) | native_decide | norm_num |

|---|---|---|---|

| Check 2 + 2 = 4 | Instant | Overkill | Instant |

| Check property for n < 100 | Slow/timeout | Fast (~ms) | N/A |

| Check property for n < 10000 | Timeout | Fast (~100ms) | N/A |

| Large integer arithmetic (10¹⁰⁰) | OOM (unary repr.) | Instant (GMP) | Good (binary) |

| Graph search on 100 nodes | Timeout | Works with Array | N/A |

  

### 10.4 Critical Implementation Details

  

**Use Array, not List.** Lists are linked structures with O(n) access and cache misses. Arrays are contiguous memory with O(1) access. For `native_decide`, Array gives 20× speedup on collections over 100 elements.

  

**Ensure tail recursion.** Non-tail-recursive functions in `native_decide` will stack-overflow on large inputs. The Lean compiler performs TCO, but only for functions in tail position. Use accumulator-passing style:

  

```lean

-- UNSAFE: Stack overflow for large n

def sum_bad : List Nat → Nat

  | [] => 0

  | h :: t => h + sum_bad t

  

-- SAFE: Tail recursive

def sum_good (acc : Nat) : List Nat → Nat

  | [] => acc

  | h :: t => sum_good (acc + h) t

```

  

**Add fuel for potentially non-terminating computations.** Unlike the kernel (which has `maxHeartbeats`), `native_decide` runs compiled code that can loop forever. Always ensure termination structurally or via a fuel parameter.

  

**Set options for heavy computation:**

```lean

set_option maxRecDepth 10000

set_option maxHeartbeats 0  -- Disable heartbeat limit for native_decide

```

  

### 10.5 Lean 4 on Kaggle: Practical Concerns

  

Research reveals that Lean 4 installation on Kaggle GPU instances can fail due to CUDA/LLVM conflicts. Mitigation:

  

1. Pre-build Lean 4 binaries as a Kaggle dataset (~500 MB without Mathlib)

2. Use the Lean interpreter, not native compilation (avoids clang dependency)

3. Test the specific Kaggle H100 environment before competition submission

4. Have a fallback plan if Lean doesn't work (verified Python computation is better than no answer)

  

---

  

## 11. What "Verified" Actually Means (Honest Assessment)

  

The pipeline verifies: "the candidate formula f(n) matches the trace-computed values g(n) for n = 1, ..., K."

  

This is **not** a proof that f(n) is the correct answer to the original problem for all n. The gap has three components:

  

### 11.1 Gap 1: Trace Correctness

  

The trace is produced by LLM-generated Python code. If the LLM misparses the problem (off-by-one, wrong constraints, missing edge cases), the trace is wrong, and Lean will happily certify a formula that matches the wrong trace. This is the **most dangerous failure mode** — a certified wrong answer.

  

**Mitigation:** Generate multiple independent traces (different LLM prompts, different approaches). If they agree, confidence increases. If they disagree, flag for manual inspection.

  

### 11.2 Gap 2: Finite vs. Infinite

  

Matching f(n) = g(n) for n = 1, ..., 100 does not prove f(n) = g(n) for all n. Infinitely many distinct functions agree on any finite prefix.

  

**Mitigation:** This is strong evidence, not proof. For competition math, where sequences are well-behaved and formulas have low complexity, a formula that matches 100 terms and was produced by a deterministic algorithm (BM, Lagrange) is overwhelmingly likely to be correct. The probability of a "coincidence" decreases exponentially with the number of verified terms.

  

**Stronger mitigation (future):** Build a certificate layer. For C-finite sequences, proving that both the trace and the formula satisfy the same recurrence with the same initial conditions would give a true proof. This requires formalizing the trace generator's logic in Lean — a significant engineering effort.

  

### 11.3 Gap 3: TCB Trust

  

`native_decide` expands the TCB. The verification is conditional on the correctness of the Lean compiler, runtime, and GMP.

  

**Mitigation:** For competition math, this is a non-issue. The probability of a compiler bug is astronomically lower than the probability of an LLM reasoning error.

  

### 11.4 What We Can Honestly Claim

  

- "The formula matches the trace for n = 1, ..., 100, as verified by Lean 4's `native_decide`." ✅

- "The formula is correct for all n." ❌ (Not without an inductive proof or recurrence certificate)

- "The answer to the original problem is X." ❌ (Not without trusting the trace generator)

  

For AIMO 3, the first claim is sufficient. It's vastly stronger than TIR's "Python ran without errors," and for problems where the trace is correct and the mining is deterministic, the end-to-end probability of a wrong answer is near zero.

  

---

  

## 12. SymPy as Infrastructure

  

SymPy is the backbone of the pipeline's symbolic computation. Key capabilities and limitations:

  

### 12.1 What SymPy Handles Well

  

| Operation | Function | Notes |

|---|---|---|

| Polynomial interpolation | `interpolate(points, x)` | Exact rational arithmetic |

| Linear recurrence finding | `SeqFormula.find_linear_recurrence()` | Uses Berlekamp-Massey internally |

| Recurrence solving | `rsolve(equation, seq)` | Finds closed forms for C-finite + some holonomic |

| Symbolic summation | `Sum(expr, (k, a, b)).doit()` | Gosper's algorithm for hypergeometric |

| Factorization | `factorint(n)` | Needed for Carmichael function |

| Modular arithmetic | `mod_inverse(a, m)`, `crt()` | Chinese Remainder Theorem |

| Combinatorial functions | `binomial(n,k)`, `catalan(n)`, `bell(n)`, `stirling(n,k)` | Symbolic, not just numeric |

| Expression simplification | `simplify()`, `factor()`, `expand()` | For cleaning up formulas |

  

### 12.2 What SymPy Cannot Do

  

- **Holonomic guessing:** SymPy's holonomic module handles closure properties but does NOT guess recurrences from data. We must build this ourselves.

- **Non-linear recurrences:** `rsolve` cannot handle aₙ = aₙ₋₁² + 1 or similar.

- **Large-scale numeric:** SymPy is pure Python. For evaluating formulas on 10⁶ data points, use `lambdify` to compile to NumPy, or use raw Python `int` arithmetic.

  

### 12.3 Performance Tips

  

- Install `gmpy2` for 10–100× speedup on integer arithmetic. SymPy auto-detects it.

- Use `lambdify(n, formula, 'math')` for fast numeric evaluation during pre-verification.

- Clear SymPy caches periodically (`sympy.core.cache.clear_cache()`) in long-running pipelines.

- Use `Poly(expr, domain='ZZ')` when working with integer polynomials to avoid unnecessary rational promotion.

  

### 12.4 SymPy's Carmichael Function

  

```python

from sympy.ntheory import reduced_totient, totient

  

# Carmichael's λ(m) = reduced_totient(m)

# Euler's φ(m) = totient(m)

  

lambda_100 = reduced_totient(100)  # Returns 20

phi_100 = totient(100)             # Returns 40

```

  

---

  

## 13. Implementation Priorities

  

Ordered by impact-to-effort ratio for AIMO 3:

  

### Priority 1: OEIS Offline Lookup (1 day)

- Upload `stripped.gz` + `names.gz` as Kaggle dataset

- Build hash-based index on first 8 terms

- Trivial to implement, immediately useful

  

### Priority 2: Holonomic Guesser (2–3 days)

- Pure Python, ~300 lines, exact Fraction arithmetic

- Captures Catalan, factorial, derangements, central binomials

- Largest single coverage gain (~15–20% of combinatorics)

  

### Priority 3: Validate Lean 4 on Kaggle H100s (1 day)

- Test compilation, native_decide, timing

- Discover and fix CUDA/LLVM conflicts early

- If Lean doesn't work on Kaggle, the entire verification layer needs a Plan B

  

### Priority 4: Detection Cascade + Router (2 days)

- Finite difference test → Lagrange

- BM with linear complexity profile → C-finite

- Holonomic guesser → P-recursive

- Cycle detection → Modular

- OEIS lookup → Named sequences

- Full pipeline integration

  

### Priority 5: Lean Template Library (3–5 days)

- Verification templates for each pattern type

- powMod injection for modular arithmetic

- SymPy-to-Lean printer

- Pre-verification checks

  

### Priority 6: Certificate Layer (5+ days, stretch goal)

- Prove recurrence holds, not just formula matches trace

- Requires Lean lemma library for recurrence reasoning

- Transforms "strong testing" into "actual proof"

- This is what makes the approach genuinely novel vs. just "fancy TIR"

  

---

  

## 14. Trace Generation Prompting

  

The quality of the mined sequence depends entirely on the trace. These are the prompting strategies for generating reliable traces.

  

### 14.1 TraceGen System Prompt

  

```
# Role
You are an expert Research Software Engineer specializing in Experimental Mathematics.

# Objective
Write a robust, defensive Python script to compute the first N terms of the sequence f(n).

# Protocol
1. **Plan:** Analyze the problem. Is it combinatorial, number-theoretic, or geometric?
2. **Strategy:** Choose a brute-force algorithm. Correctness > Speed for small N.
3. **Libraries:** Use `itertools` (combinatorics), `sympy` (number theory), `networkx` (graphs).
4. **Implementation:** Write the function `compute_sequence(limit)`.
   - Use a Generator (`yield`).
   - Use `@functools.cache` for recursion.
   - Assert types and constraints (Defensive Coding).
   - Implement a Timeout mechanism (10s limit).
5. **Output:** The script must print the sequence as a list of STRINGS to avoid precision loss.

# Output Format
Return ONLY the Python code block.
```

  

### 14.2 Consensus Verification (AlphaCode-style)

  

To mitigate trace errors, generate multiple independent scripts and cluster by output:

  

1. Generate K scripts (10–50) at temperature ~0.7
2. Execute all for n = 1..5 (fast filter — discard crashes/timeouts)
3. Cluster surviving scripts by output trace
4. Select most efficient script from the largest cluster
5. Use the consensus trace as ground truth for mining

  

### 14.3 Key Failure Mitigations

  

| Failure Mode | Cause | Fix |
|---|---|---|
| **JSON precision loss** | Large integers (>2⁵³) corrupt in float-based JSON | Output as list of strings |
| **Labeled vs unlabeled** | LLMs confuse ordered/unordered counting | Prompt must specify explicitly |
| **Symmetry breaking** | Overcounting in grid/geometry enumeration | Fix first element to break symmetry |

  

---

  

## 15. Verification Retry Protocol

  

When Lean verification fails, a structured retry loop diagnoses the failure and selects the correct recovery strategy.

  

### 15.1 Failure Diagnosis Matrix

  

| Error Class | Regex Signature | Root Cause | Recovery |
|---|---|---|---|
| **Syntax** | `unexpected token`, `function expected` | Lean 3 syntax / hallucination | Prompt LLM with error for fix |
| **Type** | `failed to synthesize instance DivisionRing` | `/` on Nat/Int without Ring | Cast to Rat or use `Int.fdiv` |
| **Type** | `type mismatch.*expected.*Int.*got.*Rat` | Domain mixing | Inject coercion |
| **Resource** | `(deterministic) timeout at 'whnf'` | Calc too heavy / inefficient recursion | Reduce N, optimize to tail recursion |
| **Logic** | `tactic 'native_decide' failed` (False) | Formula doesn't match trace | Extract counterexample index, re-mine |
| **System** | `failed to compile definition` | Nested `let rec` / C gen bug | Rewrite to simple structural recursion |

  

### 15.2 Diagnostic native_decide (Counterexample Extraction)

  

Instead of treating `native_decide` as binary pass/fail, extract which indices fail:

  

```lean
-- Returns the specific n values where formula disagrees with trace
#eval (Array.range 100).filter (fun n => f n != expected[n]!)
```

  

This guides mining refinement: off-by-one errors, wrong initial conditions, or parity-dependent formulas become immediately visible.

  

### 15.3 Resource Budget Annealing

  

Each problem gets a three-phase time budget:

  

| Phase | Time Window | Strategy |
|---|---|---|
| **Exploration** | 0–120s | Generate 3 traces. Run fast miners. Quick verification (N=10). |
| **Exploitation** | 120–300s | Lock best candidate. Full verification (N=100). Syntax repair loops. |
| **Panic** | 300–360s | Abandon verification. Fall back to TIR. Python answer > no answer. |

  

---

  

## References

  

### Algorithms

- Berlekamp, 1968. "Nonbinary BCH decoding." IEEE Trans. Inform. Theory.

- Massey, 1969. "Shift-register synthesis and BCH decoding." IEEE Trans. Inform. Theory.

- Reeds & Sloane, 1985. "Shift-register synthesis (modulo m)." SIAM J. Discrete Math.

- Bostan & Mori, 2020. "A Simple and Fast Algorithm for Computing Exponentials and Logarithms of Formal Power Series." FOCS 2020.

- Kauers, 2009. "Guessing Handbook." RISC Report 09-07, JKU Linz.

- Berthomieu & Faugère, 2016. "Guessing Linear Recurrence Relations of Sequence Tuples and P-recursive Sequences with Linear Algebra." ISSAC 2016.

  

### Lean 4

- Lean 4 documentation: native_decide, ofReduceBool axiom

- Mathlib: LinearRecurrence, norm_num, omega, ring

- Compfiles: 228 competition math problems in Lean 4

- miniF2F: 488 theorems from AMC/AIME/IMO

  

### Competition Math

- AIMO Progress Prize 1 (NuminaMath), 2 (NemoSkills), 3 (in progress)

- OEIS: ~393K sequences, stripped.gz ~15 MB

- Kauers & Paule, "The Concrete Tetrahedron: Symbolic Sums, Recurrence Equations, Generating Functions, Asymptotic Estimates." Springer, 2011.

- Salvy, 2005. "D-finiteness: Algorithms and applications." ISSAC 2005.

  

### Software

- SymPy: sympy.series.sequences, sympy.ntheory, sympy.polys

- ore_algebra (SageMath): gold standard for holonomic computation

- mpmath: arbitrary precision arithmetic, PSLQ algorithm

- qalle2/oeis-search: offline OEIS search tool

**
