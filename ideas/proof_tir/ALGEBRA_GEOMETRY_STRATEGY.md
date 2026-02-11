# Algebra & Geometry: Near-Perfect Assurance Through Computational Redundancy

> **Core Thesis:** The LLM should never SOLVE. It should only FORMALIZE. Once a problem is polynomial equations, deterministic algorithms solve and redundant verification makes error impossible.

---

## Part 0: The Principle — Why Trace-to-Lean Works and What We're Generalizing

Trace-to-Lean achieves near-perfect assurance for combinatorics/NT through a specific trick:

```
LLM writes code to compute f(1)..f(10)     → EASY task for LLM (code generation)
Berlekamp-Massey finds the recurrence       → DETERMINISTIC (zero hallucination)
Lean checks f(1)..f(100) match the formula  → TRUSTED (native_decide)
```

The magic is that **the LLM never reasons about math**. It only writes a Python loop. The hard work (pattern finding, verification) is done by algorithms that cannot hallucinate.

The question: what is the equivalent for algebra and geometry?

**The answer is not one trick. It's a hierarchy of five independent assurance mechanisms that stack multiplicatively.** If each mechanism has a failure probability of 10⁻³, five independent mechanisms give 10⁻¹⁵ — equivalent to formal proof for all practical purposes.

---

## Part 1: The Five Assurance Mechanisms

### Mechanism 1: The Formalization Inversion

**The key insight the documents get right but don't push hard enough:**

> The LLM should not SOLVE the problem. The LLM should TRANSLATE it into polynomial constraints. These are two fundamentally different tasks with fundamentally different difficulty and reliability.

**Why this is near-perfect for algebra:**

An algebra problem says: "Let x, y, z be positive reals with x + y + z = 1 and x² + y² + z² = 1/3. Find x³ + y³ + z³."

The LLM's job is NOT to solve this. It's to write:

```python
# Constraints
# e1 = x + y + z = 1
# e2 = x^2 + y^2 + z^2 = 1/3
# Target: x^3 + y^3 + z^3 = ?
```

This is **pattern matching on natural language**, not mathematical reasoning. LLMs are excellent at this. The error rate for this specific subtask is low — and crucially, errors are detectable (the constraints are checkable against the problem statement by a second LLM pass or by human inspection of a short structured output).

Once we have constraints, Newton's identities give us the answer deterministically:
- p₁ = e₁ = 1
- p₂ = e₁p₁ - 2e₂ = 1 - 2/3 = 1/3
- p₃ = e₁p₂ - e₂p₁ + 3e₃ = ...

No LLM involvement in the computation. Zero hallucination risk.

**Why this is near-perfect for geometry:**

A geometry problem says: "Triangle ABC has AB=13, BC=14, CA=15. Find the area."

The LLM's job is to write:

```python
constraints = [
    dist_sq(A, B) == 169,
    dist_sq(B, C) == 196,
    dist_sq(C, A) == 225,
]
# WLOG: A = (0, 0), B = (13, 0)
# Solve for C
target = area(A, B, C)
```

Again — **translation, not reasoning**. The LLM maps "AB = 13" to `dist_sq(A, B) == 169`. This is a lookup table, not creative mathematics.

**The formalization inversion**: Instead of asking "what is the area?" (hard), we ask "what are the equations?" (easy). Then deterministic solvers find the area from the equations.

### Mechanism 2: Exact Symbolic Solving (No Numerics Needed for Most Problems)

**A critical insight the existing documents underweight:**

The current docs propose: numerical optimization → mpmath → PSLQ → hope. This is the wrong order for competition math.

**Most competition algebra/geometry problems are EXACTLY SOLVABLE by SymPy.**

Why? Because competition problems are designed by humans to have nice answers. The polynomial systems that arise are:
- Low degree (≤ 8)
- Few variables (≤ 6)
- Designed to have rational or simple algebraic solutions

SymPy's `solve()` handles these directly. No numerics, no PSLQ, no floating-point uncertainty.

**Proposed hierarchy:**

```
Tier 1: SymPy exact solve                     → Success rate: ~60-70%
Tier 2: SymPy + resultant elimination          → Success rate: ~80%
Tier 3: Numerical (mpmath) + PSLQ             → Success rate: ~95%
Tier 4: Numerical + integer snapping           → Success rate: ~99% (for AIMO integer answers)
```

**Why Tier 1 first?** Because when SymPy gives you an exact answer, verification is trivial — substitute back and simplify. No PSLQ uncertainty, no precision questions, no false positives. The answer is either right or SymPy throws an error.

**Novel trick — The "Resultant Cascade" for geometry:**

When SymPy's `solve()` fails on a multivariate polynomial system (which happens for geometry with ≥4 free points), use resultants to eliminate variables one at a time:

```python
from sympy import resultant

# System: f1(x,y,z) = 0, f2(x,y,z) = 0, f3(x,y,z) = 0
# Eliminate z from f1, f2:
g1 = resultant(f1, f2, z)  # Polynomial in x, y only
# Eliminate z from f1, f3:
g2 = resultant(f1, f3, z)  # Polynomial in x, y only
# Eliminate y from g1, g2:
h = resultant(g1, g2, y)   # Polynomial in x only!
# Now solve univariate polynomial
x_values = solve(h, x)
```

This is deterministic, guaranteed to terminate, and produces the MINIMAL POLYNOMIAL of the answer variable. It's the algebraic geometry version of Gaussian elimination.

**Critical advantage over Gröbner bases:** Resultant computation doesn't suffer from intermediate coefficient swell in the same way (it's still polynomial degree explosion, but the coefficients stay bounded by the input). For competition-sized problems, it's fast.

### Mechanism 3: The "Over-Verification" Principle

**This is the direct analog of Trace-to-Lean's "verify n=1..100 when you only needed n=1..10."**

For algebra: After finding the answer, verify it satisfies not just the original constraints but ADDITIONAL INDEPENDENT consequences.

**Example:**

Problem: "x + y = 5, xy = 6. Find x² + y²."

Solution: x² + y² = (x+y)² - 2xy = 25 - 12 = 13. ✓

Over-verification:
- Check: x³ + y³ = (x+y)³ - 3xy(x+y) = 125 - 90 = 35 ✓
- Check: x⁴ + y⁴ = (x²+y²)² - 2(xy)² = 169 - 72 = 97 ✓
- Check: (x-y)² = (x+y)² - 4xy = 25 - 24 = 1, so x-y = ±1 ✓
- Check: x, y are roots of t² - 5t + 6 = 0, so x=2, y=3 (or vice versa) ✓
- Check: 2² + 3² = 4 + 9 = 13 ✓

Each check is independent. If the answer were wrong, it would be astronomically unlikely to pass all 5 checks.

**For geometry, over-verification is even more powerful:**

After computing coordinates for a geometry problem:
1. Check ALL stated constraints (distances, angles, incidences) — not just the ones used to solve
2. Check UNSTATED but implied constraints:
   - Triangle inequality for all triangles
   - Angle sum = π for all triangles
   - Heron's formula consistency (area via sides vs. area via coordinates)
   - Stewart's theorem for all cevians
   - Ptolemy's theorem for all cyclic quads
   - Power of a point for all circles
   - Cross-ratio invariance for all projective configurations
3. Each check is a polynomial identity that evaluates to either 0 (pass) or nonzero (fail)

**How many checks do we need?**

By Schwartz-Zippel: if a polynomial identity of degree d is FALSE, evaluating at a random point gives nonzero with probability ≥ 1 - d/|F|. For degree-100 polynomials over 50-digit rationals, one evaluation gives error probability < 10⁻⁴⁸. So even ONE over-verification check is overkill — but doing 10 makes error probability < 10⁻⁴⁸⁰.

### Mechanism 4: Dual Computation (Independent Methods, Same Answer)

**The most powerful practical assurance mechanism, and the one the documents barely mention.**

For any competition problem, generate TWO independent solution methods:

```
Method A: LLM writes coordinate geometry solution (Python)
Method B: LLM writes trigonometric/synthetic solution (Python)
```

If both return the same answer to 50 digits: **the answer is correct with near-certainty.**

Why? The error modes are INDEPENDENT:
- A coordinate geometry bug (wrong constraint translation) has nothing to do with a trigonometric bug (wrong identity application)
- The probability of both methods producing the same wrong answer is the PRODUCT of their individual error probabilities
- If each has 5% error rate, the dual computation has 0.25% error rate
- With 3 independent methods: 0.0125%

**This is essentially what SC-TIR (self-consistency TIR) does, but we can make it much stronger:**

SC-TIR generates N solutions with the SAME prompt and votes. The solutions are NOT independent — they share the same biases, the same training data, the same failure modes.

**Our version: Structurally Independent Dual Computation (SIDC)**

```
Method A: Coordinate geometry (place A at origin, B on x-axis)
Method B: Complex number representation (A, B, C as complex numbers)
Method C: Trigonometric (law of cosines, law of sines)
Method D: Vector approach (position vectors, dot/cross products)
Method E: Barycentric coordinates
```

These five methods have STRUCTURALLY different error modes. A coordinate geometry off-by-one (e.g., using squared distance instead of distance) will NOT produce the same wrong answer as a complex number phase error. The methods are truly independent.

**For AIMO: generate all 5, take majority vote among those that agree to 10 digits.** If 4/5 agree, confidence is astronomical.

### Mechanism 5: Lean Verification of the Final Answer

This is the cherry on top. After computing the answer k and finding its minimal polynomial P(x):

```lean
-- Core Lean 4, no Mathlib needed
-- Verify that k satisfies the elimination polynomial
example : (84 : Nat) * (84 : Nat) = 7056 := by native_decide

-- Or for the minimal polynomial check:
def P (x : Int) : Int := x^4 - 10*x^2 + 1
example : P 3 = 52 := by native_decide  -- Not a root
example : P 0 = 1 := by native_decide   -- Not a root
```

For integer answers (AIMO), this is trivially implementable.

For polynomial identity verification (geometry):

```lean
-- Core Lean 4 with grind (no Mathlib)
-- Verify that Heron's formula matches cross-product area
example [CommRing α] (a b c s : α) :
    s = (a + b + c) / 2 →
    16 * s * (s - a) * (s - b) * (s - c) =
    2*a^2*b^2 + 2*b^2*c^2 + 2*c^2*a^2 - a^4 - b^4 - c^4 := by
  intro h; subst h; ring
```

The `grind` tactic with its built-in `ring` solver handles this **without any Mathlib dependency**. This is the cleanest path to formal verification on Kaggle.

---

## Part 2: Novel Tricks and Ideas

### Trick 1: The "Elimination Polynomial" — Complete Algebraic Proof for Geometry

**This is potentially the most important new idea.**

For a geometry problem asking "find length d given constraints C₁, ..., Cₙ":

1. LLM translates to polynomial constraints in variables (x₁, ..., xₘ, d) where d is the target
2. SymPy computes `resultant(C₁, C₂, x₁)` to eliminate x₁
3. Continue eliminating until we have a SINGLE polynomial Q(d) = 0
4. Solve Q(d) = 0 (it's univariate!)
5. Lean verifies: `Q(answer) = 0` via `native_decide`

**Why this is a complete proof:**

The elimination polynomial Q(d) is DETERMINED by the constraints. If the constraints are correct (checkable by re-reading the problem), then Q(d) = 0 is a NECESSARY condition for d. If Q has a unique positive real root (typical for geometry "find the length" problems), that root IS the answer.

**What the LLM does:** ONLY translate the problem to constraints. Everything else is deterministic.

**What can go wrong:** The LLM misformulates a constraint. But we catch this by:
- Having a second LLM independently translate the problem (dual formalization)
- Checking the constraints against each other for consistency
- Verifying that the computed coordinates satisfy ALL constraints (over-verification)

**Failure probability:** If two independent LLM translations produce the same constraints, and those constraints are internally consistent, and the computed answer satisfies all constraints — the probability of error is effectively zero.

**Implementation complexity:** LOW. This is just SymPy resultants + mpmath root-finding + one Lean `native_decide` call. Maybe 200 lines of Python.

### Trick 2: The "Random Witness" for Universal Geometry Statements

For "prove that property P holds for ALL triangles":

1. Generate 1000 random triangles (rational coordinates, to avoid floating-point issues)
2. Check P for each triangle EXACTLY (rational arithmetic in SymPy)
3. By Schwartz-Zippel: if P is a polynomial identity of degree d, and we test at 1000 random points over ℚ with coordinates in [-10⁶, 10⁶], the probability of a false identity passing is ≤ d/10⁶ per test
4. For 1000 tests: probability ≤ (d/10⁶)^1000 ≈ 0

**This is not a heuristic. It's a rigorous probabilistic proof.**

Schwartz-Zippel is a theorem. For polynomial identity testing, random evaluation IS proof (with quantifiable error bounds). The error bound for 1000 evaluations of a degree-100 polynomial is < 10⁻³⁰⁰⁰.

**Key requirement:** We must work over EXACT arithmetic (SymPy rationals or Python integers), not floating-point. This eliminates all concerns about numerical precision.

**What Lean adds:** We can even implement this IN Lean:

```lean
-- Check polynomial identity at 100 random rational points
-- If it passes all, native_decide accepts it
theorem geometric_identity :
    ∀ a b c : Fin 100,
    let x := (a.val : Int) - 50
    let y := (b.val : Int) - 50
    let z := (c.val : Int) - 50
    P x y z = Q x y z := by native_decide
```

This checks the identity at 100³ = 10⁶ integer points. For a degree-d polynomial, this guarantees correctness if d < 100 (which covers all competition math).

### Trick 3: The "Newton's Identity Cascade" for Symmetric Functions

**Most competition algebra problems reduce to symmetric function evaluation.**

Given elementary symmetric polynomials e₁, e₂, ..., eₙ, Newton's identities give power sums:
- p₁ = e₁
- p₂ = e₁p₁ - 2e₂
- p₃ = e₁p₂ - e₂p₁ + 3e₃
- ...

This is a DETERMINISTIC RECURRENCE. Once the LLM extracts (e₁, e₂, ..., eₙ) from the problem statement, the entire computation is hallucination-free.

**The trick:** Most "find the value" algebra problems are secretly asking for pₖ given (e₁, ..., eₙ). The LLM's job is to recognize this pattern and extract the elementary symmetric polynomials. The rest is pure computation.

**Verification:** Newton's identities can be verified in Lean via `ring`:

```lean
-- Newton's identity: p₂ = e₁² - 2e₂
example [CommRing α] (e1 e2 : α) :
    e1^2 - 2*e2 = e1 * e1 - 2 * e2 := by ring
```

### Trick 4: The "Constructive Decomposition" for Inequalities

The current docs propose SOS via SDP (fragile). Here's a better approach:

**Step 1:** LLM GUESSES the SOS decomposition.

This is the key inversion. Instead of using an SDP solver (fragile, floating-point, hard to rationalize), we ask the LLM:

```
"Express x⁴ - 4x³ + 6x² - 4x + 1 as a sum of squares of polynomials."
```

LLMs are GOOD at this! They've seen hundreds of SOS decompositions in training data. And the answer is easy to verify:

```
x⁴ - 4x³ + 6x² - 4x + 1 = (x² - 2x + 1)² = ((x-1)²)²
```

**Step 2:** Lean verifies via `ring` + `positivity`.

```lean
example (x : ℝ) : x^4 - 4*x^3 + 6*x^2 - 4*x + 1 ≥ 0 := by
  have h : x^4 - 4*x^3 + 6*x^2 - 4*x + 1 = (x^2 - 2*x + 1)^2 := by ring
  rw [h]; positivity
```

**Step 3:** If the LLM's guess is WRONG, `ring` rejects it. No harm done. Try again with a different prompt or a different decomposition.

**Why this beats SDP:**
- No floating-point issues (the LLM outputs integer/rational coefficients)
- No rationalization step
- No SDP solver dependency
- `ring` verification is kernel-checked in Lean
- The LLM can try Schur, AM-GM, Cauchy-Schwarz, and Muirhead-type decompositions
- Multiple attempts cost seconds, not the minutes an SDP solver takes

**For harder inequalities (non-SOS, e.g., Motzkin):**

Use the Schur-SOS technique: for symmetric 3-variable inequalities, every symmetric inequality can be written as:

```
S_a(b-c)² + S_b(c-a)² + S_c(a-b)² ≥ 0
```

where S_a, S_b, S_c are expressions in a, b, c. The LLM finds S_a, S_b, S_c; Lean verifies the identity via `ring` and the non-negativity via `positivity` or manual case analysis.

### Trick 5: The "Parametric Trace" — Turning Geometry into Sequences

**Some geometry problems CAN be reduced to the Trace-to-Lean framework.**

Example: "Regular n-gon inscribed in unit circle. Find the product of distances from one vertex to all others."

This is parameterized by n! We can:
1. Compute for n = 3, 4, 5, ..., 20 (Python, exact)
2. Apply Berlekamp-Massey or pattern recognition
3. Find: product = n (!)
4. Verify in Lean for n = 3..100

**When does this work?**
- Problems involving regular polygons
- Problems with an integer parameter
- Recursive geometric constructions (fractals, nested figures)
- Problems involving lattice points

**This is the "Neuro-Symbolic Cascading" from the router doc, elevated to a primary strategy:**

```
LLM identifies the integer parameter in a geometry problem
    → Computes the answer for parameter = 1, 2, ..., 20
    → Berlekamp-Massey finds the pattern
    → Lean verifies for parameter = 1..100
```

The LLM's job is just to identify the parameter and write the computation code. Pattern mining and verification are deterministic.

### Trick 6: The "Reverse Verification" for "Find" Problems

**Instead of proving the answer is correct, prove all other answers are wrong.**

For AIMO (answers are integers 0-99999):

1. Compute numerical answer to 50 digits → a ≈ 42.00000000...
2. Verify f(42) = 0 in Lean (native_decide)
3. Verify f(41) ≠ 0 and f(43) ≠ 0 in Lean (native_decide)

If f is continuous and f(42) = 0 while f(41) ≠ 0 and f(43) ≠ 0, then 42 is the unique integer root in [41, 43].

For problems where the answer could be any of a small set (e.g., "how many triangles?"):

1. The answer is at most ~1000 (AIMO range)
2. Build the elimination polynomial Q(d)
3. Check Q(0), Q(1), ..., Q(999) in Lean
4. Report which ones are roots

This is brute-force but **formally verified**. And it's fast — `native_decide` can check 1000 polynomial evaluations in milliseconds.

### Trick 7: The "Dual Formalization" — Catching Constraint Translation Errors

**The biggest risk in our approach is the LLM misformulating the constraints. Here's how to catch it.**

Run TWO independent LLM calls with DIFFERENT prompts:

```
Prompt A: "Translate this geometry problem into polynomial constraints 
           using Cartesian coordinates. Place A at origin, B on x-axis."

Prompt B: "Translate this geometry problem into polynomial constraints 
           using complex numbers. Let A = 0, B = 1."
```

If both produce constraints that yield the same answer, the formalization is correct with high probability. The error modes are independent:
- Cartesian bugs (wrong distance formula, sign error in perpendicularity)
- Complex number bugs (wrong conjugate, wrong rotation formula)

**Even stronger:** Generate the constraints, then ask a THIRD LLM to verify them against the original problem text:

```
"Here is a geometry problem: [original text]
 Here are the proposed constraints: [constraints]
 Does each constraint correctly correspond to a condition in the problem? 
 Are any conditions missing?"
```

This is a VERIFICATION task (checking someone else's work), which LLMs are measurably better at than generation (the "self-correction blind spot" research shows they're good at finding errors in OTHERS' work, just not their own).

### Trick 8: The "Computable Algebraic Number" for Lean Verification

**The documents propose using Mathlib's `AdjoinRoot` for Q(√2, √3). This doesn't work (noncomputable). Here's what does work:**

Define a custom computable type directly in Lean:

```lean
-- Q(√2): elements are a + b√2 with a, b : ℚ
structure QSqrt2 where
  a : Int  -- rational part (numerator, denominator tracked separately)
  b : Int  -- √2 coefficient
  d : Nat  -- common denominator (> 0)
  deriving DecidableEq, Repr

-- Multiplication: (a₁ + b₁√2)(a₂ + b₂√2) = (a₁a₂ + 2b₁b₂) + (a₁b₂ + a₂b₁)√2
def QSqrt2.mul (x y : QSqrt2) : QSqrt2 :=
  { a := x.a * y.a + 2 * x.b * y.b
  , b := x.a * y.b + x.b * y.a
  , d := x.d * y.d }
```

This is fully computable. `native_decide` can evaluate arithmetic in this field. We don't need Mathlib at all — just a 50-line Lean file defining the field operations and `DecidableEq`.

**For Q(√2, √3, √5):** Elements are 8-tuples of rationals. The multiplication table is fixed and hardcoded. Still fully computable, still fast with `native_decide`.

**For arbitrary Q(√d):** Parameterize by d. One definition covers all quadratic extensions.

This is a few hundred lines of Lean, not a Mathlib dependency. And it gives us formal verification of geometry answers involving surds.

### Trick 9: The "Confidence Cascade"

**Not all problems need the same level of assurance. Use a cascade that stops at the first sufficient level:**

```
Level 0 (Cost: ~0.1s): Integer snapping
  - Compute answer numerically to 30 digits
  - If |answer - round(answer)| < 10⁻²⁰, return the integer
  - Confidence: 99.99% (for AIMO problems)

Level 1 (Cost: ~1s): Exact symbolic solve
  - SymPy solve the system exactly
  - Substitute answer back into ALL constraints
  - If all residuals are exactly 0: verified
  - Confidence: 100% (modulo SymPy bugs, which are rare)

Level 2 (Cost: ~5s): Dual computation
  - Run two independent solution methods
  - If they agree to 50 digits: verified
  - Confidence: 99.9999% (independent error modes)

Level 3 (Cost: ~10s): Over-verification
  - Check 10+ independent geometric/algebraic consequences
  - If all pass: verified
  - Confidence: 99.99999999% (Schwartz-Zippel bound)

Level 4 (Cost: ~30s): Lean formal verification
  - Build elimination polynomial
  - Verify Q(answer) = 0 in Lean via native_decide
  - Confidence: ~100% (formal proof, modulo compiler trust)
```

**The cascade saves time.** Most AIMO problems terminate at Level 0 or 1. Only hard problems need Level 3-4. Average time: 2-3 seconds per problem, well within the 6-minute budget.

### Trick 10: The "Problem Inverter" for Geometry

**Instead of solving the problem forward, solve it backward.**

Forward: "Given triangle with sides 13, 14, 15, find the area."
Backward: "Is there a triangle with sides 13, 14, 15 and area 84?"

The backward question is a DECISION problem, not a search problem. For Lean:

```lean
-- Verify: Heron's formula gives area² = s(s-a)(s-b)(s-c)
-- s = (13+14+15)/2 = 21
-- area² = 21 * 8 * 7 * 6 = 7056
-- area = 84 ✓
example : 21 * 8 * 7 * 6 = 7056 := by native_decide
example : 84 * 84 = 7056 := by native_decide
```

**How do we get the candidate answer 84?** Any method — numerical, symbolic, LLM guess, even brute force over integers 1..999. Once we HAVE the candidate, verification is trivial.

**This is the deepest version of the Oracle-Checker paradigm.** The Oracle can be wrong, buggy, hallucinating — it doesn't matter. The Checker (Lean) has the final word.

---

## Part 3: The Unified Architecture

### 3.1 The Pipeline

```
PROBLEM (natural language)
    │
    ▼
┌──────────────────────────────────┐
│   DUAL FORMALIZATION (LLM × 2)  │  Two independent constraint translations
│   Prompt A: Cartesian coords     │
│   Prompt B: Complex numbers      │
│   → Check agreement of constraints│
└──────────────┬───────────────────┘
               │
               ▼
┌──────────────────────────────────┐
│   EXACT SYMBOLIC SOLVE (SymPy)   │  Tier 1: solve(), Tier 2: resultants
│   → If success: exact answer     │  Tier 3: mpmath + PSLQ (fallback)
│   → If integer: snap to integer  │
└──────────────┬───────────────────┘
               │
               ▼
┌──────────────────────────────────┐
│   OVER-VERIFICATION              │  Check ALL constraints + extra theorems
│   → Substitute answer back       │  Stewart's, Ptolemy's, Heron's, etc.
│   → All residuals must be 0      │  (exact rational arithmetic)
│   → 10+ independent checks       │
└──────────────┬───────────────────┘
               │
               ▼
┌──────────────────────────────────┐
│   LEAN VERIFICATION              │  For high-confidence answers:
│   → Build elimination polynomial │  native_decide on Q(answer) = 0
│   → Verify Q(answer) = 0        │  OR: grind/ring for polynomial identity
│   → Core Lean 4, no Mathlib     │  
└──────────────┬───────────────────┘
               │
               ▼
           FINAL ANSWER
     (with assurance level)
```

### 3.2 What Each Component Does and What Can Go Wrong

| Component | What it does | What LLM does | Hallucination risk | Error detection |
|-----------|-------------|---------------|-------------------|-----------------|
| Dual Formalization | Translate NL → polynomial constraints | Constraint extraction (easy) | Low (~5%) | Cross-check between two translations |
| Exact Solve | Find answer from constraints | Nothing | Zero (deterministic) | N/A |
| Numerical Fallback | mpmath + PSLQ | Nothing | Zero (PSLQ is deterministic) | RoI ≥ 2 check |
| Over-Verification | Check redundant conditions | Nothing | Zero (polynomial evaluation) | Any failure = alarm |
| Lean Check | Formal verification | Nothing | Zero (kernel-checked) | native_decide is binary |

**Critical observation:** The LLM is involved in ONLY ONE step (formalization). Everything else is deterministic. And that one step is cross-checked by dual formalization. The overall system hallucination risk is near zero.

### 3.3 Time Budget (per problem)

| Step | Time | Notes |
|------|------|-------|
| Dual formalization (2 LLM calls) | 10s | Parallel inference |
| SymPy exact solve | 2-5s | Most competition systems |
| Resultant elimination (if needed) | 5-15s | For larger systems |
| mpmath + PSLQ (if needed) | 1s | 100 dps is fast |
| Over-verification (10 checks) | 0.5s | Rational arithmetic |
| Lean verification | 1-5s | subprocess + native_decide |
| **Total** | **15-35s** | Well within 6-minute budget |

---

## Part 4: What About Problems We Can't Formalize?

### The Hard Cases

Not every problem reduces to "polynomial constraints → solve → verify." The hard cases are:

1. **Functional equations** ("find all f: ℝ → ℝ such that..."): These ask for a FUNCTION, not a number. The numerical sniper can't fire because there's no single target.

   **Our trick:** Discretize. Compute f(1), f(2), ..., f(20) by treating the functional equation as a recurrence/constraint system. Use Berlekamp-Massey or Lagrange interpolation to guess f(x). Verify the guess by substitution.

   This works because competition functional equations almost always have polynomial or simple solutions (f(x) = x², f(x) = cx, etc.).

2. **Optimization with non-polynomial constraints** ("minimize sin(x) + cos(y) subject to..."): Transcendental functions break the polynomial framework.

   **Our trick:** For competition math, transcendental values are almost always at "nice" points (multiples of π/6, π/4, π/3). Enumerate these candidates and check each. This is finite brute force over a small candidate set.

3. **Existence proofs** ("prove there exists a configuration with property P"): We need to FIND the configuration, not just verify it.

   **Our trick:** This IS what numerical optimization is for. Coordinate descent finds a witness, then we verify it exactly. The optimization is untrusted; the verification is trusted.

4. **Counting with large parameters** ("how many lattice points in a circle of radius 10⁹?"): Too large for brute force.

   **Our trick:** This is a number theory problem in disguise. Route to Trace-to-Lean. Compute for small radii, find the pattern, extrapolate.

### The Unsolvable Cases (Fallback to TIR)

Some problems genuinely resist our approach:
- Abstract algebra / group theory
- Problems requiring creative auxiliary constructions
- Problems requiring deep structural insight (not computation)

For these: fall back to standard TIR (LLM chain-of-thought with Python REPL). Accept lower confidence. This is ~10-15% of competition math.

---

## Part 5: Implementation Priority

### Week 1-2: The Minimal Viable Pipeline

```python
# 1. LLM formalization (one prompt, not dual yet)
constraints = llm_formalize(problem_text)

# 2. SymPy exact solve
answer = sympy.solve(constraints)

# 3. If integer: snap and verify
if is_close_to_integer(answer):
    k = round(answer)
    # Lean: native_decide on constraints evaluated at k
    verified = lean_verify(constraints, k)
```

Test on 50 AIME geometry + algebra problems. Measure success rate.

### Week 3-4: Add Redundancy

- Dual formalization (two LLM prompts)
- Over-verification (10 independent checks)
- Numerical fallback (mpmath + PSLQ)
- Resultant elimination for harder geometry

### Week 5-6: Lean Integration

- Package Lean 4 for offline use
- Implement QSqrt2, QSqrt3 computable types
- Elimination polynomial → native_decide pipeline
- `grind` for polynomial identity verification

### Week 7-8: Inequality Engine

- LLM-guessed SOS decompositions
- `ring` + `positivity` verification in Lean
- Schur-SOS for symmetric 3-variable inequalities
- DSOS (LP-based) as fallback for when LLM can't guess

---

## Part 6: Why This Achieves "Near-Perfect Assurance"

The error in our system can only come from ONE source: **the LLM mistranslating the problem into constraints.** Every other step is deterministic.

We catch translation errors via:
1. **Dual formalization** (two independent translations must agree)
2. **Over-verification** (wrong constraints → inconsistent consequences)  
3. **Cross-checking** (third LLM verifies the constraints against the problem text)

The probability of:
- Both LLM translations having the SAME error: < 0.5% × 0.5% = 0.0025%
- That error producing a consistent answer: < 50% of the time
- That answer passing 10 over-verification checks: < 10⁻¹⁰ (Schwartz-Zippel)
- That answer passing Lean verification: < 10⁻¹⁵ (compiler bugs only)

**Combined error probability: < 10⁻¹⁵**

This is comparable to formal proof. For all practical purposes, it IS formal proof — just with a different trust model (we trust the Lean compiler rather than trusting the LLM's reasoning).

---

## Summary

| Innovation | What it replaces | Why it's better |
|-----------|-----------------|-----------------|
| **Formalization Inversion** | LLM solves the problem | LLM only translates; deterministic algorithms solve |
| **Exact Symbolic Solve** | Numerical optimization + PSLQ | No floating-point; no false positives |
| **Resultant Cascade** | Gröbner bases | Simpler, no coefficient swell, same result |
| **Over-Verification** | Trust the computation | 10+ independent checks catch any error |
| **Dual Computation** | Single solution path | Independent error modes multiply confidence |
| **LLM-Guessed SOS** | SDP + rationalization | No SDP solver; no floating-point; verified by `ring` |
| **Random Witness** | Symbolic proof of universals | Schwartz-Zippel gives 10⁻³⁰⁰⁰ error probability |
| **Computable Algebraic Fields** | Mathlib AdjoinRoot | 50 lines of Lean; fully decidable; no dependencies |
| **Confidence Cascade** | One-size-fits-all verification | Stops at cheapest sufficient level |
| **Dual Formalization** | Trust the LLM's translation | Two independent translations must agree |

**The deepest principle:** We never ask the LLM to be RIGHT. We ask it to be CREATIVE — to guess, to translate, to propose decompositions. Then we verify everything it says with deterministic algorithms. The LLM is the Oracle. Math is the Checker. And the Checker never lies.
