# Key Research Findings: Extracted from Early Research Phase

> **Purpose:** Consolidated reference of concrete data, novel framings, and prior art worth preserving. Everything else from the original research docs is either already in TRACE_TO_LEAN.md / ALGEBRA_GEOMETRY_STRATEGY.md or is standard background.

---

## 1. The "Hard Core" — What Trace-to-Lean Cannot Solve

~30% of Combinatorics/Number Theory problems resist trace mining entirely. These require:
- **Bijective proofs** — proving |A| = |B| by constructing a mapping (equivalent to program synthesis)
- **Invariant discovery** — finding a conserved quantity across transformations
- **Constructive existence proofs** — proving a configuration exists on large discrete sets
- **Extremal combinatorics** — bounding Ramsey numbers, proving optimal bounds

These are the "Boss" problems. No amount of Berlekamp-Massey or Lagrange interpolation helps. They require semantic insight that current LLMs cannot reliably produce. This is the honest ceiling of the architecture.

**Per-category tractability (empirically grounded):**

| Domain | Traceable/Verifiable | Numerical/Search | Hard Core |
|--------|---------------------|------------------|-----------|
| Combinatorics | 40% | 30% | **30%** |
| Number Theory | 50% | 20% | **30%** |
| Algebra | 40% | 45% | **15%** |
| Geometry | 10% (synthetic) | 85% (coordinate) | **5%** |

---

## 2. Concrete TIR Failure Examples

These are the strongest rhetorical evidence for why verification matters. Each is a real failure mode where Python exits 0 but the answer is wrong.

### Example 1: Silent NumPy Overflow
```python
import numpy as np
arr = np.arange(1, 10**6 + 1, dtype=np.int32)
squares = arr ** 2          # (10^6)^2 = 10^12 overflows int32
print(np.sum(squares) % 10) # Garbage — NumPy wraps silently
```

### Example 2: Float Factorial
```python
import math
res = math.factorial(200) / math.factorial(198)  # Float division
print(res % 47)  # Precision destroyed — modulo is meaningless
```

### Example 3: Off-By-One
```python
count = 0
for i in range(1, 100):  # Stops at 99, excludes 100
    if i % 3 == 0 or i % 5 == 0:
        count += 1         # Off by 1 (100 is divisible by 5)
```

### Example 4: Empty Set Base Case
```python
if k == 0: return 0  # Hallucinated — C(n,0) = 1, not 0
```

### Example 5: Float Comparison in Geometry
```python
if slope1 == slope2:  # Float epsilon failure — 1.0000000000000001 ≠ 1.0
```

### Example 6: NumPy Silent Wraparound
```python
val = np.power(2, 64, dtype=np.int64)  # Overflows, returns 0
# LLM: "The result is 0, implying impossibility..."
```

### Example 7: Permutation Float Check
```python
# AIMO 2024 problem — checking condition on 10! permutations
if ratio == 1.5:  # Float: 1.499999999 fails — count off by 4
```

---

## 3. Why Majority Voting (SC-TIR) Fails

**The Condorcet Jury Theorem requires uncorrelated errors.** LLM errors are systematically correlated:

- All N samples share the same training data biases
- If a problem feature (e.g., empty set edge case) triggers a wrong algorithm, it triggers it across ALL samples
- SC-TIR converges confidently on the wrong answer — "consensus of error" indistinguishable from consensus of truth

**Concrete data:**
- False positive rates (code runs, answer wrong): **40.6%** for LLaMA-13B, **~21.8%** for larger models
- LLMs show **22% accuracy** on loop boundary structural reasoning — off-by-one is systematic, not random
- LLMs fail to self-correct **64.5%** of the time (but CAN identify errors in other models' outputs)
- "Reasoning gap" of **58-80%** — performance drops when problem parameters are changed, suggesting memorization not reasoning

---

## 4. Academic Framing and Prior Art

### 4.1 The Architecture IS a CEGIS Loop

Counter-Example Guided Inductive Synthesis (SyGuS/CEGIS) maps exactly:

| CEGIS Component | Trace-to-Lean Component |
|----------------|------------------------|
| Examples | Execution traces |
| Synthesizer | Berlekamp-Massey / Lagrange / LLM pattern mining |
| Verifier | Lean `native_decide` |
| Counter-example feedback | Lean rejection → retry with different formula |

This is the right framing for a paper. We are doing CEGIS for mathematical formula discovery.

### 4.2 "Daikon for Math"

Daikon (dynamic analysis tool) runs programs, records variable values at checkpoints, and checks template invariants (e.g., `x = a*y + b`, `array is sorted`). Output: "likely invariants."

Our system is Daikon for math — but we upgrade "likely invariants" to verified theorems by passing mined formulas to `native_decide`. The analogy is clean and citable.

### 4.3 Key Prior Art to Cite

| System | What it does | Gap we fill |
|--------|-------------|-------------|
| **MathCheck** | SAT solver + CAS for graph theory conjectures | Closest prior art for Oracle-Checker in math. We add LLMs as the Oracle. |
| **LeanCert** | Uses `native_decide` to verify neural network properties | Direct precedent for computation-as-verification in Lean 4. We apply it to competition math. |
| **Four Color Theorem (Gonthier, 2005)** | Computation-as-proof inside Coq kernel | 20-year pedigree for the approach. Checked thousands of graph configurations via execution, not deduction. |
| **Ramanujan Machine** | Brute-force discovers continued fraction formulas for constants | Discovery without verification. We add the verification step. |
| **Inverse Symbolic Calculator** | Numerical input → closed-form symbolic expression | Pattern matching against known constants. We do this for sequences, not constants. |

### 4.4 Why Now (The Convergence)

Three technologies converged post-2021:
1. **Lean 4's `native_decide`** — compiles to C++, makes verification fast enough for competition use
2. **LLM code generation quality** — 85%+ compilation success makes trace generation reliable
3. **AIMO competition** — creates the evaluation framework and incentive structure

---

## 5. Evaluation Metrics Worth Using

### 5.1 Formalization Tax (L_form)

The percentage of problems where the system finds the correct answer via informal computation but fails to produce a valid Lean verification:

```
L_form = |Correct_Trace ∩ Invalid_Verification| / |Correct_Trace|
```

This measures the cost of requiring formal verification.

### 5.2 Rigor Bonus (G_rigor)

The percentage of problems where TIR yields an incorrect answer (false positive) while Trace-to-Lean correctly rejects or finds the true solution:

```
G_rigor = |TIR_Fail ∩ Lean_Success| / |Total_Problems|
```

This measures the gain from formal verification.

### 5.3 The Crossover Hypothesis

- **Easy problems (AIME 1-10):** TIR outperforms due to Formalization Tax
- **Hard problems (AIME 11-15, IMO):** Trace-to-Lean outperforms due to Rigor Bonus
- The system's value is proportional to problem difficulty

---

## 6. Failure Taxonomy for Autoformalization

| Error Category | Sub-Type | Description |
|---------------|----------|-------------|
| **Translation** | Syntax Invalidity | Generated Lean violates syntax |
| **Translation** | Hallucinated Lemma | References nonexistent Mathlib function |
| **Semantic** | Goal Drift | Proves a different theorem than intended |
| **Semantic** | Premise Omission | Missing constraints (e.g., n > 0) |
| **Logical** | Tactic Failure | Specific tactic can't close the goal |
| **Resource** | Timeout | Lean kernel hangs during elaboration |

**Domain correlation hypothesis:**
- Algebra → high Syntax Invalidity (complex equation formatting)
- Combinatorics → high Premise Omission (forgetting set finiteness)
- Geometry → high Hallucinated Lemma (inventing geometric theorems)

---

## 7. Problem Labeling: The "Trap Type" Taxonomy

For adversarial benchmark construction, label problems by trap type:

| Trap Type | Description | Example |
|-----------|-------------|---------|
| **Boundary** | Edge cases (n=0, n=1, empty set) | C(n,0) = 1 not 0 |
| **Over-counting** | Symmetry/distinctness confusion | Circular vs. linear arrangements |
| **Fake Pattern** | Pattern holds for n < k then breaks | Sequence matches polynomial for small n, diverges |

---

## 8. Competition Trends Affecting Strategy

- **Combinatorics rising** from ~25% to ~35% of high-tier competition problems (2015-2025)
- **Pure synthetic geometry declining** — problems now invite algebraic approaches
- **~15% of problems are hybrid** (combinatorial NT, algebraic combinatorics)
- **~10-15% are deliberately "computer-resistant"** — problem setters aware of AI, designing against computation
- **AIMO C/NT overrepresented at 54%** — the Hard Core is where points are lost
- **IMO problem committee actively moving away** from coordinate-bashable geometry
