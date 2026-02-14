# Combinatorics & Number Theory: Critical Review and Path to 50/50

> **Purpose:** Challenge the conservative coverage claims in `pattern_mining.md` and `V2_EXECUTION_SPEC.md`, identify what's overstated and understated, and lay out a strategy to extend verification coverage to ace combo+NT.
>
> **Core thesis:** The competitive edge is verification, not computation. Everyone can compute. The question is: can we VERIFY every answer we produce? The docs claim 20-35% of problems resist verification. That claim is the problem to solve.
>
> **Premise:** We do not assume any problem is unverifiable. The question is never "can an LLM solve this?" but "can we design a verification method that covers this problem type?"
>
> **Date:** February 2026

---

## 0. Why Verification Is the Edge, Not Computation

### 0.1 The TIR Trap

The temptation is to look at the evidence — o3 scoring 47/50, frontier models hitting 100% on AIME with code execution — and conclude that computation is solved. Just generate 64 Python scripts, majority-vote the answer, done.

This is exactly what Nvidia's NemoSkills does. They won AIMO2 with TIR + GenSelect: 93.3% on AIME 2024 at maj@64. They trained on 540K curated problems with 3.2M solutions. They have the data moat, the compute, the infrastructure. Competing on TIR ground is a losing strategy.

**The deeper problem:** Multiple code samples from the same LLM share the same systematic biases. If the model confuses "labeled" vs "unlabeled" in a counting problem, all 64 samples make the same mistake. Consensus on correlated outputs is not verification — it's an echo chamber. NemoSkills at 93.3% still has 6.7% confident wrong answers with no way to flag them.

### 0.2 What Verification Actually Gives

The Trace-to-Lean pipeline makes a fundamentally different claim: **"If the verification passes, the answer IS correct."**

This creates something TIR cannot: a positive VG gap (Verification-Generation gap). When `Verify(Acc) > Generate(Acc)`, scaling compute always helps — generate 1000 candidates, verify each, the one that passes IS right. TIR can't scale this way because it has no reliable filter.

The verification chain creates semantic independence:
- **Python trace:** Brute-force computation, LLM-generated, potentially biased
- **Mined formula:** Deterministic algorithm (BM, Lagrange, holonomic), zero LLM involvement, zero hallucination
- **Lean check:** Verifies trace-formula agreement, zero LLM involvement

When an LLM-written trace and a deterministically-mined formula agree on 100 terms, the confidence is qualitatively different from "64 LLM outputs agree." The trace and the formula are produced by independent processes. Their agreement is genuine evidence, not correlated noise.

### 0.3 The Real Problem Statement

The docs claim 20-35% of combo/NT problems resist pattern mining and therefore fall back to unverified TIR. That 20-35% is where verification's edge is lost. The task is to shrink that number — not by abandoning verification, but by extending the verification layer to cover the "resistant" categories.

**Target:** Reduce the unverifiable residual from 20-35% to under 5%.

---

## 1. Evidence That 50/50 Is Realistic

What frontier systems have demonstrated on the computation side:

| System | Benchmark | Score | Notes |
|---|---|---|---|
| o3-preview (high compute) | AIMO2 public LB (50 problems, national olympiad) | **47/50** pass@1, **50/50** top-2 | March 2025 |
| o4-mini (with Python) | AIME 2025 (30 problems) | **99.5%** pass@1, **100%** consensus@8 | April 2025 |
| Gemini 3 Pro (with code exec) | AIME 2025 | **100%** | Late 2025 |
| Claude Sonnet 4.5 (with code exec) | AIME 2025 | **100%** | Late 2025 |
| NemoSkills 32B TIR + GenSelect | AIME 2024 | **93.3%** maj@64 | April 2025 |
| AlphaProof | IMO 2024 | 4/6 (28/42 pts), silver medal | July 2024 |

The 47/50 was o3-preview without a specialized pipeline. The 3 missed problems on AIMO2 were 2 geometry, 2 algebra, 3 combinatorics. On AIME 2025, multiple frontier models hit 100% with code execution.

**Key fact:** Computation is near-solved for AIME-level math. The remaining gap is not "can we get the answer?" but "can we KNOW we got the answer?" That's verification. That's our edge.

**AlphaProof at IMO 2024:** Solved 4/6 but failed on BOTH combinatorics problems (P3 and P5). This is the strongest empirical evidence for combo being hard — but specifically hard for *Lean proof search*, not for computation. AlphaProof tries to produce full proofs in Lean. Our pipeline only needs to verify trace-formula agreement, which is a much weaker (and more tractable) claim.

### 1.1 AIMO3 Is Significantly Harder Than AIMO2

The evidence table above is mostly AIME-level or national-olympiad-level (AIMO2). AIMO3 is a different beast:

- **110 problems** (up from 50 in AIMO2), **IMO-level difficulty** (up from national olympiad)
- **5-digit answers** (0-99999), up from smaller ranges — larger answer space means majority voting is less reliable
- **H100 GPUs available** — more compute, but harder problems eat that budget
- **GRIDDI was unsolved by ALL 2000+ AIMO2 teams** AND o3-preview at maximum compute. If AIMO3 has problems of similar difficulty, frontier TIR will fail on some of them.

The AIMO2→AIMO3 difficulty jump means the evidence table above is an **upper bound** on raw LLM accuracy. At IMO level, even o3/o4-mini accuracy drops well below 100%. This makes verification MORE important, not less — the VG gap widens when generation accuracy falls.

---

## 2. What's Overstated in pattern_mining.md

### 2.1 The "20-35% resists pattern mining" figure

The doc lists five resistance categories (§1.1):
- Floor/ceiling/GCD/valuation functions
- Piecewise or parity-dependent formulas
- Constructive arguments, bijections, pigeonhole
- Multi-parameter sequences
- Problems requiring genuine mathematical insight

The problem is not that these categories exist — they do. The problem is that the doc treats them as terminal: "These fall back to standard TIR." That's the line where the verification edge is surrendered. Each of these categories has a verifiable path that the doc doesn't explore.

### 2.2 The coverage table counts only existing tiers

The 55-65% "with formal guarantees" figure counts Tiers 1-4 (polynomial, C-finite, holonomic, modular cycles). It excludes:
- Floor/Beatty sequence verification (verifiable with known formulas)
- Piecewise formula verification (split + mine + verify each piece)
- Multiplicative function verification (identify Euler product, verify on prime powers)
- Direct Lean computation verification (write the algorithm in BOTH Python and Lean)
- SAT-backed enumeration with Lean witnessing

These aren't speculative — they're well-defined algorithmic extensions to the existing verification layer.

### 2.3 The holonomic guesser framing

The doc says it's a "critical gap" needing 2-3 days of implementation. This is correct, but `ore_algebra` via passagemath now makes it a 1-day deployment instead of a custom build:

```bash
pip install "ore_algebra[passagemath] @ git+https://github.com/mkauers/ore_algebra.git"
```

~200-500 MB of modularized SageMath components. Fits on Kaggle. Gives the gold-standard Kauers guessing algorithm. This is a strict upgrade over building a custom guesser.

### 2.4 The "genuine mathematical insight" escape clause

This is the most dangerous phrase in the document. It's unfalsifiable — any problem that hasn't been solved yet can be labeled as requiring "genuine mathematical insight." It gives the system permission to give up.

In practice, for AIMO's format (compute a specific integer 0-99999), there is always a finite computation that produces the answer. The question is whether we can verify that computation. The phrase should be replaced with: "Problems where the verification method has not yet been identified."

---

## 3. What's Overstated in V2_EXECUTION_SPEC.md

### 3.1 The 9-stage pipeline latency

S0-S8 with strict ordering is expensive. The spec says easy problems should take ≤45s, medium ≤120s, hard ≤240s (§15.3). With 110 AIMO3 problems, the total budget is tight. Two observations:

- For combo/NT specifically, the stages that matter are S1 (trace generation), S4 (certificate synthesis via mining), and S5 (Lean check). S2 (audit) and S3 (deterministic solve) are primarily for algebra/geometry. A fast path that skips S2-S3 for combo/NT problems would save significant time.
- The spec should allow S1 and S4 to run in parallel: generate traces while simultaneously mining and verifying partial results.

### 3.2 K=4 speculative parallelism is too low for trace consensus

The spec says K=4 traces (§19.3). This is for trace generation, not for TIR voting — the distinction matters. We're not voting on answers; we're generating independent traces and checking which ones agree before feeding them into mining.

But K=4 is still low for trace consensus. With K=4, a single biased trace can dominate. K=8-16 gives much stronger trace consensus — if 12 of 16 traces agree on the same sequence values, the trace is almost certainly correct, addressing Gap 1 (trace correctness) from §11.

The cost model is favorable: on H100 with vLLM, generating 16 traces shares the KV cache prefix and runs in ~2-3x the time of generating 1 trace, not 16x.

### 3.3 The Sage/Singular emphasis is misplaced for combo/NT

The solver decision boundary (§4) triggers Sage when "variables >= 3 and max degree >= 2." This almost never fires for combo/NT — these are sequence/recurrence problems, not polynomial systems. The engineering cost of deploying Sage on Kaggle is better spent on extending mining tiers.

### 3.4 The formalization tax is real but the framing is fatalistic

The spec defines L_form (formalization tax) and G_rigor (rigor bonus) but treats the crossover point as something to measure, not something to shift. The entire purpose of extending mining coverage is to reduce L_form. Every new verification tier we add moves the crossover point toward easier problems, expanding the regime where verification helps.

---

## 4. What's Understated in Both Docs

### 4.1 The trace consensus is the strongest weapon against Gap 1

Gap 1 (trace correctness) is flagged as the "most dangerous failure mode" in §11 of `pattern_mining.md`. The mitigation ("generate multiple independent traces") is mentioned but not operationalized.

Trace consensus is qualitatively different from answer consensus:
- **Answer consensus (TIR):** 64 scripts compute the same wrong answer because the LLM makes the same systematic error → false confidence
- **Trace consensus:** 16 scripts compute f(1)..f(50) independently. If 12+ agree on ALL 50 values, the trace is almost certainly correct. A systematic LLM bias would need to produce identical wrong values at all 50 points across 12 independent implementations — astronomically unlikely.

**Operationalization:**
1. Generate K=16 trace-generating scripts using diverse prompts (different algorithmic approaches: enumeration, DP, recursion, generating functions)
2. Execute all, producing 16 candidate traces
3. Cluster traces by exact match on first 20 terms
4. Accept the largest cluster if size >= 10/16
5. Feed the consensus trace to the mining cascade

This should be a mandatory stage, not an optional mitigation.

### 4.2 Dual-computation Lean verification for the mining residual

For problems where mining fails (no polynomial, C-finite, holonomic, or periodic structure), there's a verification method the docs don't consider: **write the computation in both Python and Lean, and verify they agree.**

This is weaker than formula-vs-trace verification (both computations could have the same bug), but it's stronger than pure TIR because:
- Python and Lean are structurally different languages — the same logical error rarely manifests identically in both
- Lean's type system catches entire classes of bugs (integer overflow, off-by-one in Nat subtraction, etc.) that Python silently swallows
- `native_decide` in Lean compiles to C++ with GMP — it's a genuinely independent implementation

**Template:**
```lean
-- Python computed f(target_n) = 42
-- Lean independently computes f(target_n) using the same algorithm
def f (n : Nat) : Nat := ... -- translate the algorithm, not just the answer
theorem answer_check : f target_n = 42 := by native_decide
```

This verifies that the algorithm produces 42 at the target, not just that the number 42 is stated. If the Python and Lean implementations agree, confidence is high.

### 4.3 SAT/CSP solvers as both computation AND verification

SAT solvers provide both the answer and a proof certificate (UNSAT proofs, model witnesses). For combinatorics problems encodable as SAT:
- The solver computes the answer (or a count, via model counting)
- The witness IS the verification — it's a concrete assignment satisfying all constraints
- This can be checked in Lean by verifying the witness against the constraints

PySAT (pip-installable, works on Kaggle) provides access to CaDiCaL, Kissat, and other state-of-the-art solvers.

### 4.4 The problem format enables a key trick the docs miss

AIMO3 answers are 5-digit integers (0-99999). This means:
- For any candidate answer A, we can verify it by checking A against independently computed values
- If a problem asks "find f(n) mod m," and we can verify f(k) mod m for k=1..100 via mining, then f(n) mod m follows from the verified pattern
- The 5-digit constraint means we can sometimes exhaustively verify: if f(n) must equal one of {0, 1, ..., 99999}, and our formula yields a value in this range that matches the trace, that's strong evidence

### 4.5 The Most Dangerous Failure Mode: Wrong Answers That Mine Cleanly

**This is the single most important finding from the research phase.** The current pipeline assumes that formula-vs-trace verification catches errors. It does catch mining/translation errors. But it does NOT catch the case where the **trace itself is wrong AND the wrong trace has perfectly clean mathematical structure.**

This is not hypothetical. Here are concrete examples:

**Labeled vs unlabeled counting.** A problem asks to count unlabeled graphs on n vertices. The LLM's brute-force code counts labeled graphs instead. Labeled graph counts are polynomial in nature (n choose 2 subsets). This sequence mines perfectly as a polynomial via Lagrange interpolation, passes BM, and the Lean `native_decide` check confirms the formula matches the (wrong) trace. The entire pipeline reports Tier A confidence on a wrong answer.

**Ordered vs unordered partitions.** Compositions (ordered) vs partitions (unordered). Compositions of n have 2^(n-1) elements — a clean C-finite sequence with BM order 1. If the LLM counts compositions when the problem asks for partitions, the pipeline mines the composition formula flawlessly.

**Off-by-one Catalan variants.** Catalan(n) vs Catalan(n-1) vs Catalan(n+1). All are holonomic, all mine perfectly with `ore_algebra`, all produce valid Lean verification. The pipeline can't distinguish which offset matches the problem statement.

**Signed vs unsigned Stirling numbers.** These differ by a sign pattern but are both C-finite for fixed k. Mining works perfectly for either.

**AIMO2 empirical evidence:** Wrong answers on AIMO2 cluster around specific values — not uniformly distributed. This is exactly what you'd expect if wrong answers arise from coherent misinterpretations (different but valid mathematical objects) rather than random computational errors. The clustering confirms that wrong traces have clean structure.

**Implication for the pipeline:** Formula-vs-trace verification is a **necessary but not sufficient** condition for answer correctness. It catches one class of errors (mining bugs, Lean translation bugs) but is blind to another class (trace misinterpretation). A wrong trace that mines cleanly is the worst-case scenario — it produces a certified wrong answer with high confidence.

**What's needed:** A second verification layer — **trace-vs-problem verification** — that checks whether the trace is actually computing what the problem asks. See §5.6.

---

## 5. Extending Verification to the "Resistant" Categories

This is the core technical contribution. For each "resistant" category, we define a verification method that maintains the independence property (deterministic verification of LLM-generated traces).

### 5.1 Floor/Ceiling/GCD/Valuation Sequences — Tier 2.5: Structural Type Verification

**The problem:** These sequences are not polynomial, C-finite, or holonomic. The mining cascade (§2.5 of `pattern_mining.md`) routes them to OEIS or LLM fallback.

**The solution:** Don't mine a recurrence. Instead, identify the structural type and verify the type-specific formula.

**Detection:**
```python
def detect_floor_type(seq):
    """Identify floor/ceiling/GCD structural types."""

    # Test 1: Beatty sequence (floor(alpha*n + beta))
    # Signature: first differences take exactly 2 values (Sturmian word)
    diffs = [seq[i+1] - seq[i] for i in range(len(seq)-1)]
    unique_diffs = set(diffs)
    if len(unique_diffs) == 2:
        # Recover alpha via convergents of continued fraction
        alpha_approx = seq[-1] / len(seq)
        return "beatty", alpha_approx

    # Test 2: Legendre-type (v_p(n!) = (n - s_p(n)) / (p-1))
    for p in [2, 3, 5, 7, 11, 13]:
        def digit_sum_base(n, base):
            s = 0
            while n > 0:
                s += n % base
                n //= base
            return s
        match = all(
            seq[n] == (n - digit_sum_base(n, p)) // (p - 1)
            for n in range(1, len(seq))
            if n < len(seq) and seq[n] is not None
        )
        if match:
            return "legendre", p

    # Test 3: Digit sum / digital root
    for base in [10, 2, 3]:
        def digit_sum(n, b):
            s = 0
            while n > 0:
                s += n % b
                n //= b
            return s
        if all(seq[n] == digit_sum(n, base) for n in range(1, min(len(seq), 100))):
            return "digit_sum", base

    # Test 4: GCD-based (gcd(f(n), g(n)))
    # Check if seq[n] divides n for all n (common for gcd(n, k))
    for k in range(2, 20):
        from math import gcd
        if all(seq[n] == gcd(n, k) for n in range(1, min(len(seq), 100))):
            return "gcd_fixed", k

    return None, None
```

**Lean verification for Beatty sequences:**
```lean
-- Verify: a(n) = floor(alpha * n) where alpha = p/q (rational approx)
-- For exact verification, check |a(n) - (p*n)/q| <= 1 for all test values
theorem beatty_check :
  (List.range 200).all (fun n =>
    let approx := p * n / q  -- integer division
    a n == approx || a n == approx + 1) = true := by
  native_decide
```

**Lean verification for Legendre formula:**
```lean
def digitSumBase (n p : Nat) : Nat :=
  if n == 0 then 0
  else (n % p) + digitSumBase (n / p) p

theorem legendre_check :
  (List.range 200).all (fun n =>
    vp_factorial n p == (n - digitSumBase n p) / (p - 1)) = true := by
  native_decide
```

**Coverage gain:** This tier handles ~5% of combo and ~15-20% of NT problems that currently fall through to TIR. Verification is deterministic — the structural type is identified algorithmically and verified computationally.

### 5.2 Piecewise/Parity-Dependent Formulas — Case-Splitting Miner

**The problem:** The sequence has no single closed form, but behaves differently for even/odd n, or for n mod 3, etc.

**The solution:** Automatically detect the modulus, split the trace, mine each sub-sequence through the existing tiers, and verify each piece in Lean.

**Detection and mining:**
```python
def case_split_mine(seq, max_modulus=6):
    """Try mining with modular case-splitting."""
    for m in range(2, max_modulus + 1):
        sub_seqs = [[] for _ in range(m)]
        for i, val in enumerate(seq):
            sub_seqs[i % m].append(val)

        # Try mining each sub-sequence
        results = []
        all_mined = True
        for r in range(m):
            sub = sub_seqs[r]
            # Try polynomial
            deg = detect_polynomial(sub)
            if deg is not None:
                poly = lagrange_interpolate(sub)
                results.append(("poly", r, poly))
                continue
            # Try C-finite
            C, L = berlekamp_massey(sub)
            if L < len(sub) / 3:
                results.append(("c_finite", r, C, L))
                continue
            # Try holonomic (via ore_algebra)
            # ...
            all_mined = False
            break

        if all_mined:
            return m, results
    return None, None
```

**Lean verification for piecewise formulas:**
```lean
-- Verify: f(n) = g(n) when n % 2 == 0, h(n) when n % 2 == 1
def f_piecewise (n : Nat) : Nat :=
  if n % 2 == 0 then g (n / 2) else h (n / 2)

theorem piecewise_check :
  (List.range 200).all (fun n =>
    f_piecewise n == trace_values[n]!) = true := by
  native_decide
```

**Coverage gain:** +5% combo, +3% NT. The verification remains deterministic — each piece is mined by standard tiers and verified by `native_decide`.

### 5.3 Constructive/Bijective/Pigeonhole — Aggressive Trace Consensus + Mining

**The problem:** The "intended" solution uses a bijection or pigeonhole argument. The LLM's brute-force code might get the construction wrong.

**The solution:** This is a trace correctness problem (Gap 1), not a mining problem. If the trace is correct, the underlying sequence is typically polynomial or C-finite (counting problems almost always produce well-structured sequences). The fix is:

1. **Generate K=16 diverse traces** using structurally different approaches:
   - Direct enumeration prompt
   - DP prompt
   - Inclusion-exclusion prompt
   - Generating function prompt
2. **Cluster by exact match** on first 20 terms
3. **Accept largest cluster** if size >= 10/16
4. **Mine the consensus trace** through standard tiers (it will usually be polynomial or C-finite)
5. **Verify in Lean** as usual

**Why this works:** Constructive problems produce sequences that ARE polynomial, C-finite, or holonomic — the resistance is not in the sequence structure but in the risk of wrong traces. Trace consensus neutralizes that risk. Once the trace is correct, standard mining applies.

**Coverage gain:** +5-8% combo (the largest single gain). The verification is standard — the innovation is in trace robustness.

### 5.4 Multi-Parameter/Divisor Sum Sequences — Slice + Mine

**The problem:** The sequence depends on number-theoretic functions like sigma(n), tau(n), phi(n), or is a slice of a 2D table.

**The solution:**

**For multiplicative functions:**
```python
def detect_and_verify_multiplicative(seq):
    """If multiplicative, identify by values at prime powers."""
    from math import gcd

    # Test multiplicativity
    for m in range(2, len(seq)//2):
        for n in range(2, len(seq)//m):
            if gcd(m, n) == 1 and m*n < len(seq):
                if seq[m*n] != seq[m] * seq[n]:
                    return None

    # Multiplicative! Identify by prime power values
    # seq[p] for small primes identifies the Dirichlet character
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    signature = [seq[p] for p in primes if p < len(seq)]

    # Match against known multiplicative functions
    known = {
        "tau": [2, 2, 2, 2, 2, 2, 2, 2, 2],        # number of divisors
        "sigma": [3, 4, 6, 8, 12, 14, 18, 20, 24],  # sum of divisors
        "phi": [1, 2, 4, 6, 10, 12, 16, 18, 22],    # Euler totient
        "id": [2, 3, 5, 7, 11, 13, 17, 19, 23],     # identity
    }

    for name, expected in known.items():
        if signature[:len(expected)] == expected[:len(signature)]:
            return name

    return "unknown_multiplicative"
```

**For 2D table slices:**
Fix one parameter and mine the resulting 1D sequence:
```python
# If a(n) = Stirling(n, k) for fixed k, mine a(n) for that k
# Stirling numbers of the 2nd kind for fixed k are polynomial in n of degree 2k
# → Lagrange interpolation recovers the polynomial
```

**Lean verification:** Once the multiplicative function or table formula is identified, verify it against the trace via `native_decide` as usual.

**Coverage gain:** +0% combo, +5-8% NT. Multiplicativity detection is pure number theory.

### 5.5 Dual-Computation Verification — The Last Resort Before TIR

For the residual that truly resists all mining tiers (estimated 3-5% after extensions), there's one more verification method before falling back to unverified TIR:

**Write the computation in BOTH Python and Lean. Verify they agree.**

```lean
-- The Python brute-force gave answer 42 for f(target_n)
-- Translate the ALGORITHM (not just the answer) into Lean
def f_lean (n : Nat) : Nat :=
  -- ... (algorithmic translation of the Python code)

-- Verify the algorithm produces 42
theorem answer_verified : f_lean target_n = 42 := by native_decide
```

**Why this is weaker but still valuable:**
- Both implementations use the same algorithm, so a logical error could appear in both
- But: Python and Lean are structurally different (dynamic vs static typing, arbitrary precision vs Nat, different evaluation models). Implementation bugs rarely manifest identically in both
- Lean's type system catches truncating Nat subtraction, division errors, and type mismatches that Python silently swallows
- `native_decide` compiles to C++ with GMP — a genuinely independent execution environment

**When to use:** After all mining tiers fail AND the problem has a small enough computation for `native_decide` to handle (typically n ≤ 10^4).

**Coverage gain:** +2-3% across both domains. Converts "unverified TIR" into "dual-computation verified."

### 5.6 Trace-vs-Problem Verification — The Missing Dimension

**Motivation:** §4.5 established that formula-vs-trace verification is blind to trace misinterpretation. A wrong trace that mines cleanly produces a certified wrong answer. We need a second, orthogonal verification axis: checking the trace against the PROBLEM STATEMENT, not just against the mined formula.

**The cascade of filters:** Verification should not be a single binary gate. It should be a cascade, where each layer catches a different error class:

1. **Layer 1: Trace consensus** (§4.1) — catches implementation bugs and random LLM errors
2. **Layer 2: Formula-vs-trace** (§5.1-5.5) — catches mining errors and Lean translation bugs
3. **Layer 3: Trace-vs-problem** (this section) — catches systematic misinterpretation of the problem

**Layer 3 methods:**

#### 5.6.1 Boundary Condition Verification

Most competition problems have small cases that are trivially verifiable by hand — or can be extracted from the problem statement.

```python
def extract_boundary_checks(problem_text, trace):
    """Extract and verify boundary conditions from the problem statement."""
    checks = []

    # Many problems state: "For n=1, there is exactly 1 such object"
    # or: "Show that f(1) = 0, f(2) = 1, ..."
    # Extract these from the problem text via LLM + regex

    # Also: if the problem says "arrangements of n objects"
    # then f(0) should be 0 or 1 (empty arrangement)
    # f(1) should be 0 or 1 (trivial case)

    for n, expected in extracted_values:
        if trace[n] != expected:
            return False, f"Boundary mismatch: f({n})={trace[n]}, expected {expected}"

    return True, "All boundary conditions match"
```

This is weak individually but catches a surprising number of off-by-one and labeled/unlabeled errors, because the small cases are where these distinctions are most visible. Labeled vs unlabeled graphs: at n=1, both give 1. At n=2, labeled gives 2, unlabeled gives 2. But at n=3, labeled gives 8, unlabeled gives 4. If the problem text implies a small answer at n=3, the labeled trace fails.

#### 5.6.2 Monotonicity and Growth Rate Checks

Competition combinatorics problems rarely produce non-monotonic counting sequences. If the trace is non-monotonic for a counting problem, something is wrong.

```python
def growth_rate_check(trace, problem_type):
    """Verify trace growth rate matches problem type expectations."""
    if problem_type == "counting":
        # Counting sequences are typically monotonically non-decreasing
        if not all(trace[i+1] >= trace[i] for i in range(len(trace)-1)):
            return False, "Non-monotonic counting sequence"

        # Check growth rate: polynomial, exponential, or factorial?
        # If the problem counts subsets → expect exponential growth
        # If the problem counts permutations → expect factorial growth
        # If the problem counts lattice paths → expect Catalan-like growth
        ratios = [trace[i+1]/trace[i] for i in range(5, len(trace)-1) if trace[i] > 0]
        if problem_type_implies_exponential and max(ratios)/min(ratios) > 2:
            return False, "Growth rate inconsistent with exponential counting"

    return True, "Growth rate consistent"
```

#### 5.6.3 OEIS Semantic Matching

When the mined sequence matches an OEIS entry, compare the OEIS description against the problem statement:

```python
def oeis_semantic_check(oeis_id, oeis_description, problem_text):
    """Check if OEIS description is semantically compatible with the problem."""
    # Use LLM to judge: "Does this OEIS description match this problem?"
    # This is LLM-assisted verification, but it's checking a DIFFERENT thing
    # than the LLM that generated the trace — semantic independence.

    # Example: Problem asks for "unlabeled graphs on n vertices"
    # OEIS match is A000088 (unlabeled graphs) → compatible
    # OEIS match is A006125 (labeled graphs) → INCOMPATIBLE → flag

    prompt = f"""
    Problem: {problem_text}
    OEIS sequence {oeis_id}: {oeis_description}
    Question: Is this OEIS sequence computing the same thing the problem asks for?
    Answer YES or NO with a one-sentence explanation.
    """
    return llm_judge(prompt)
```

This is the most powerful trace-vs-problem check because OEIS descriptions are human-written, precise mathematical English. When our sequence IS in OEIS, the semantic match between the OEIS description and the problem statement is strong evidence for or against trace correctness.

#### 5.6.4 Perturbation Consistency

Modify the problem slightly (change a parameter, add/remove a constraint) and check that the answers change in the expected direction:

```python
def perturbation_check(problem, trace, answer):
    """Verify answer stability under problem perturbation."""
    perturbations = generate_perturbations(problem)
    # E.g., for "count paths on n×n grid", try (n-1)×n and n×(n+1)
    # The answers should satisfy: f(n-1, n) < f(n, n) < f(n, n+1)

    for perturbed_problem, expected_relation in perturbations:
        perturbed_answer = solve(perturbed_problem)  # quick TIR
        if not expected_relation(answer, perturbed_answer):
            return False, f"Perturbation inconsistency: {expected_relation}"

    return True, "Perturbation consistent"
```

#### 5.6.5 Divisibility and Modular Constraints

Many competition problems have structural divisibility constraints extractable from the problem statement:

- "Divide n objects into k equal groups" → f(n) should be 0 when k ∤ n
- "Pairs of ..." → f(n) = 0 when n is odd (for certain problems)
- "Mod p" in the problem → the answer should satisfy specific modular relationships

```python
def divisibility_check(trace, problem_constraints):
    """Verify trace satisfies structural divisibility constraints."""
    for constraint in problem_constraints:
        if constraint.type == "zero_when":
            for n in constraint.zero_indices:
                if n < len(trace) and trace[n] != 0:
                    return False, f"f({n}) should be 0 by problem structure"
        if constraint.type == "divisible_by":
            for n in range(len(trace)):
                if trace[n] % constraint.divisor != 0:
                    return False, f"f({n}) should be divisible by {constraint.divisor}"
    return True, "Divisibility constraints satisfied"
```

**Combined trace-vs-problem confidence:** No single check is strong enough alone. But together, they form a soft filter that catches the most common misinterpretation modes:

| Check | Catches | False positive rate |
|---|---|---|
| Boundary conditions | Off-by-one, labeled/unlabeled at small n | Low (if boundary is stated in problem) |
| Monotonicity/growth | Completely wrong counting approach | Very low |
| OEIS semantic match | Labeled/unlabeled, Catalan variants | Low (when OEIS match exists) |
| Perturbation | Systematic misinterpretation | Medium (perturbation itself may fail) |
| Divisibility | Structural constraint violations | Very low |

**Key distinction — deterministic vs LLM-assisted checks within Layer 3:**

Not all trace-vs-problem checks are created equal. Some are deterministic and genuinely independent of the trace-generating LLM; others use LLM judgment and therefore share correlated biases with the trace generator:

| Check | Type | Independence from trace LLM |
|---|---|---|
| Monotonicity/growth rate | **Deterministic** | Full — pure arithmetic on the trace values |
| Divisibility constraints | **Deterministic** | Full — extracted from problem structure, not LLM |
| Boundary conditions (when stated in problem text) | **Semi-deterministic** | High — regex/parsing, minimal LLM involvement |
| OEIS semantic match | **LLM-assisted** | Low — uses same model family, shares systematic biases |
| Perturbation consistency | **LLM-assisted** | Low — requires LLM to understand "nearby problem" |
| Boundary extraction (when NOT stated explicitly) | **LLM-assisted** | Low — LLM interprets problem, same biases as trace gen |

The deterministic checks (monotonicity, divisibility, explicit boundary values) provide genuine independent evidence. The LLM-assisted checks (OEIS matching, perturbation, implicit boundary extraction) are correlated with the trace-generating LLM — if the model misunderstands "unlabeled" vs "labeled" when generating traces, it will likely make the same conceptual error when judging OEIS descriptions or extracting boundaries. These checks still have value (they catch a different class of errors — the LLM is doing a different TASK even if it shares the same biases), but they should not be treated as independent evidence in the probabilistic sense.

The cascade is:

```
Trace consensus (statistical) → Formula-vs-trace (deterministic) → Trace-vs-problem (mixed)
```

The deterministic Layer 3 checks are strong. The LLM-assisted ones are supplementary, not load-bearing.

---

## 6. Revised Verification Coverage Estimates

**Important caveat from §4.5:** The coverage estimates below measure how many problems can be *formula-verified* (formula matches trace). They do NOT measure how many problems produce correct answers. A problem where the trace is wrong but mines cleanly counts as "verified" in the formula-vs-trace sense but produces a wrong answer. The trace-vs-problem layer (§5.6) is a separate, softer filter that reduces this risk but cannot eliminate it.

### 6.1 Formula-vs-Trace Coverage (Layer 2)

After extending the mining cascade:

| Verification Tier | Coverage (Combo) | Coverage (NT) | Verification Type |
|---|---|---|---|
| **Existing:** Polynomial (Lagrange) | ~15-20% | ~5-10% | Formula vs trace (deterministic) |
| **Existing:** C-finite (BM) | ~40-50% | ~10-15% | Formula vs trace (deterministic) |
| **Existing:** Holonomic (ore_algebra) | ~15-20% | ~5% | Formula vs trace (deterministic) |
| **Existing:** Modular cycles | ~5% | ~30-35% | Period verification (deterministic) |
| **New:** Floor/GCD/Valuation (§5.1) | ~0% | ~10-15% | Structural type verification |
| **New:** Piecewise case-split (§5.2) | ~5% | ~3% | Split formula vs trace |
| **New:** Trace consensus + standard mining (§5.3) | ~5-8% | ~2% | Consensus + formula vs trace |
| **New:** Multiplicativity/slice (§5.4) | ~0% | ~5-8% | Multiplicative identity verification |
| **New:** Dual-computation Lean (§5.5) | ~2-3% | ~2-3% | Algorithm agreement |
| OEIS + formula extraction | ~3-5% | ~3-5% | Formula vs trace |

**Formula-vs-trace totals (optimistic):**
- Combinatorics: ~85-95% can be formula-verified
- Number Theory: ~85-95% can be formula-verified

### 6.2 End-to-End Correctness Estimate (All Layers)

Formula verification is necessary but not sufficient. End-to-end correctness requires:
1. The trace is correct (addressed by trace consensus, Layer 1)
2. The formula matches the trace (addressed by mining cascade, Layer 2)
3. The trace actually computes what the problem asks (addressed by trace-vs-problem checks, Layer 3)

Estimated failure rates at each layer (for problems that pass the previous layer):
- **Layer 1 failure (wrong trace, even with consensus):** ~5-10% of problems. This is the labeled/unlabeled, off-by-one class. Trace consensus catches implementation bugs but not systematic misinterpretation shared across prompts.
- **Layer 2 failure (mining fails):** ~5-15% of problems, depending on difficulty tier.
- **Layer 3 failure (trace-vs-problem checks miss a misinterpretation):** ~50% of Layer 1 failures (some misinterpretations are caught by boundary checks, OEIS matching, etc.)

**Honest combined estimate:**
- Problems with correct, verified answers: **~80-90%**
- Problems with wrong answers that passed verification (certified wrong): **~2-5%** — this is the dangerous zone
- Problems where verification correctly flags uncertainty: **~5-10%**
- Problems where mining fails and we fall back to TIR: **~5-10%**

### 6.3 Why This Is Not an Accuracy Race Against TIR

The comparison to TIR is not "who gets more answers right on easy problems." The pipeline operates on a fundamentally different axis. The right framing:

**What TIR requires to compete:**
- A curated training dataset (NemoSkills: 540K problems, 3.2M solutions)
- Massive compute to fine-tune models on that data
- A base model that already knows how to write competition math code
- Luck that the training distribution covers the test problems
- Access to gated/proprietary datasets that provide the data moat

**What this pipeline requires:**
- An LLM that can write Python code to enumerate small cases (a dramatically lower bar — not competition math, just enumeration)
- Deterministic algorithms (BM, Lagrange, ore_algebra) — no training, no data
- Lean's kernel — a mathematical truth engine, not a learned model
- Engineering time (months, not training compute)

**The structural advantage is not incremental — it's categorical:**

| Property | Trace-to-Lean Pipeline | Pure TIR (K=64) |
|---|---|---|
| Training data required | None | 100K+ curated problems |
| Fine-tuning compute | None | Significant |
| Mathematical guarantees | Yes (native_decide) | None |
| Can detect wrong answers | Yes (verification failure) | No (all answers look the same) |
| Scales with more compute | Yes (VG gap: generate more, verify) | Diminishing (correlated LLM errors) |
| LLM quality dependency | Low (write enumeration code) | High (must solve competition math) |

The pipeline doesn't compete with TIR. It bypasses the bottleneck that makes TIR competition brutal: the need for training data, fine-tuning, and a model that deeply understands competition math. The LLM here is an untrusted proposal generator. The math comes from algorithms and proofs, not from the neural network.

No one else in the competition has this. The closest system (AlphaProof) requires MCTS over Lean proof space, takes days per problem, and failed on both combinatorics problems at IMO 2024. This pipeline makes a weaker verification claim (formula-trace agreement, not full proof) but one that's tractable at competition time scales.

---

## 7. The Danger of Gap 1 and How Trace Consensus Addresses It

Gap 1 (trace correctness) is the most dangerous failure mode. A certified wrong answer is worse than an uncertain right answer. The docs acknowledge this but don't operationalize the fix.

**Trace consensus protocol:**

```
For each problem:

1. GENERATE: K=16 trace scripts using diverse prompting strategies:
   - 4x direct enumeration (vary algorithm: for-loop, itertools, recursion, DP)
   - 4x formula-based (LLM proposes formula, script evaluates)
   - 4x cross-validation (compute by 2+ methods, assert agreement)
   - 4x adversarial (include edge-case checks, sanity assertions)

2. EXECUTE: Run all 16 scripts for n=1..50 (or problem-appropriate range)

3. CLUSTER: Group scripts by exact match on output trace
   - If largest cluster has >= 10/16 scripts: HIGH confidence trace
   - If largest cluster has 6-9/16: MEDIUM confidence, generate more scripts
   - If no cluster has > 5/16: LOW confidence, flag for manual review

4. VALIDATE: Before mining, run sanity checks on consensus trace:
   - All values are non-negative integers (for counting problems)
   - Values are monotonically increasing/decreasing (if expected)
   - Known small cases match (n=1,2,3 often have easily verifiable answers)
   - No obvious artifacts (all zeros, all same value, obviously wrong growth rate)

5. MINE: Feed the consensus trace to the extended mining cascade
```

**Why this works:** The diversity of algorithmic approaches means correlated failures require the LLM to make the same conceptual error across 4 different algorithmic frameworks. This is much harder than making the same error 16 times in the same framework.

**Cost:** ~4x the compute of K=4, but the trace is the foundation of everything downstream. Getting it wrong wastes all subsequent computation. The investment is justified.

---

## 8. Specific Corrections to pattern_mining.md

### 8.1 Rewrite the coverage table (§1.1)

Replace the current 6-row table with the extended table from §6 above. Change the "remaining 20-35%" paragraph to acknowledge the extensions and honestly state the reduced residual.

### 8.2 Add Tier 0: Trace Consensus (before Tier 1)

Before any mining, run the trace consensus protocol. This addresses Gap 1 directly and improves the quality of input to all subsequent tiers.

### 8.3 Add Tier 2.5: Structural Type Verification (§5.1)

Between C-finite and Holonomic, add detection and verification for floor/GCD/valuation sequences. These are NOT holonomic but have known closed forms that are verifiable.

### 8.4 Add Tier 2.7: Case-Splitting Miner (§5.2)

After standard BM but before holonomic guessing, try case-splitting. Many sequences that fail BM are actually piecewise C-finite or piecewise polynomial.

### 8.5 Update §5.3: ore_algebra is deployable

Replace "No pure-Python implementation exists" with: "The gold-standard `ore_algebra` is now pip-installable via passagemath modular distributions (~200-500 MB). Recommended for Kaggle deployment."

### 8.6 Add Tier 7: Dual-Computation Lean Verification (§5.5)

After LLM fallback (Tier 6), add dual-computation as a verification method for the residual. This converts some Tier C answers to Tier B.

### 8.7 Remove the "genuine mathematical insight" phrase

Replace with: "Problems where no verification template has been identified yet." The distinction matters — the first framing gives up, the second invites solutions.

---

## 9. Specific Corrections to V2_EXECUTION_SPEC.md

### 9.1 Add a fast path for combo/NT

For problems routed as combo/NT at S0:
1. Run Tier 0 (trace consensus, K=16) — this replaces both S1 and partial S2
2. Run extended mining cascade (Tiers 1-7) — this IS S4
3. Run Lean verification — this IS S5
4. If all pass: Tier A answer in ~30-60s

Stages S2 (audit) and S3 (deterministic solve) can be skipped for combo/NT — they're designed for algebra/geometry constraint systems.

### 9.2 Increase K from 4 to 16 for traces

§19.3 should specify K=16 for trace generation with consensus clustering. The cost increase is sublinear (shared-prefix KV cache) and the trace quality improvement is the single strongest defense against certified wrong answers.

### 9.3 Add verification tier extensions to §6

The Domain Minimum Acceptance Rules (§6) for combo/NT should include:
- Case-splitting miner for piecewise detection
- Multiplicativity test for NT sequences
- Structural type detection for floor/GCD/valuation
- Dual-computation fallback before TIR

### 9.4 Add SAT/CSP to the solver stack

The deterministic solve stage (S3) should include PySAT/Z3 for combinatorial constraint satisfaction. This is both a computation method and a verification method (SAT witnesses are machine-checkable).

### 9.5 Reframe the formalization tax

L_form should be reported PER VERIFICATION TIER, not as a single aggregate number. Tiers 1-4 likely have L_form < 5% (the mining-to-Lean translation is mechanical). The new tiers (5.1-5.5) may have higher L_form initially, which is where engineering effort should focus.

---

## 10. Implementation Priorities (Reordered)

Ordered by impact on **verification coverage**, not computation coverage:

### Priority 1: Trace Consensus Protocol (2 days)
Implement K=16 diverse trace generation + clustering + validation. This is the foundation — all downstream mining is only as good as the trace.

### Priority 2: ore_algebra via passagemath (1 day)
Deploy and validate on Kaggle. Test holonomic guessing on 20 representative sequences. Replaces 2-3 days of custom guesser development and extends Tier 3 to full holonomic coverage.

### Priority 3: Case-Splitting Miner (1 day)
Implement the modular case-splitting wrapper around existing miners. ~50 lines of Python. Catches the entire "piecewise/parity-dependent" resistance category.

### Priority 4: Structural Type Verification (2 days)
Implement Beatty/floor, Legendre/valuation, multiplicativity, and digit-sum detectors with corresponding Lean verification templates. The detection code is ~200 lines; the Lean templates are ~100 lines.

### Priority 5: Validate Lean on Kaggle H100 (1 day)
Go/no-go gate for the entire verification layer. If this fails, dual-computation verification (§5.5) becomes the ceiling.

### Priority 6: Dual-Computation Lean Verification (2 days)
Build infrastructure for translating Python algorithms to Lean and verifying agreement via `native_decide`. Templates for common patterns: loops, recursion, array operations.

### Priority 7: OEIS Offline Lookup (1 day)
Upload stripped.gz, build hash index. Useful for sequence identification (what IS this sequence?) which guides the choice of mining tier.

### Priority 8: SAT Solver Integration (1 day)
Add PySAT as a fallback for structured combinatorial problems. Test on representative AIME problems.

**Total: ~8-10 weeks of focused implementation, with empirical validation at each stage.** The day estimates per priority are engineering time, not calendar time — testing, debugging, and iteration on real problem sets will dominate the schedule.

---

## 11. The Architecture After Corrections

### 11.1 Three-Layer Verification Cascade

The revised architecture replaces the single-gate verification model with a cascade of three layers, each catching different error classes:

```
┌─────────────────────────────────────────────────────────────────────┐
│ LAYER 1: TRACE CONSENSUS (statistical)                             │
│   Catches: implementation bugs, random LLM errors                  │
│   Method: K=16 diverse scripts, cluster by exact match on trace    │
│   Pass criterion: largest cluster ≥ 10/16                          │
│   Failure mode: systematic misinterpretation shared across prompts │
└────────────────────────────┬────────────────────────────────────────┘
                             │ consensus trace
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│ LAYER 2: FORMULA-vs-TRACE (deterministic)                          │
│   Catches: mining errors, Lean translation bugs                    │
│   Method: mining cascade (Tiers 1-7) + native_decide               │
│   Pass criterion: formula matches trace on all computed terms      │
│   Failure mode: wrong trace that mines cleanly (§4.5)              │
└────────────────────────────┬────────────────────────────────────────┘
                             │ verified formula
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│ LAYER 3: TRACE-vs-PROBLEM (heuristic)                              │
│   Catches: labeled/unlabeled, off-by-one, Catalan variants         │
│   Method: boundary checks, monotonicity, OEIS semantic matching,   │
│           perturbation consistency, divisibility constraints        │
│   Pass criterion: no flags raised by any check                     │
│   Failure mode: misinterpretation too subtle for heuristic checks  │
└────────────────────────────┬────────────────────────────────────────┘
                             │ confidence tier
                             ▼
                    ┌────────┴────────┐
                    │  ANSWER OUTPUT  │
                    │  Tier A/B/C     │
                    └─────────────────┘
```

### 11.2 Mining Cascade (Layer 2 Detail)

```
DETECTION CASCADE:
  │
  ├── Polynomial? (finite differences) ──────────► TIER 1: Lagrange → Lean verify
  │
  ├── C-finite? (BM + stability check) ─────────► TIER 2: BM recurrence → Lean verify
  │
  ├── Floor/GCD/valuation? (structural tests) ──► TIER 2.5: Type-specific formula → Lean verify
  │
  ├── Piecewise? (case-split, re-mine) ─────────► TIER 2.7: Per-piece formulas → Lean verify
  │
  ├── Holonomic? (ore_algebra guess) ───────────► TIER 3: P-recursive → Lean verify
  │
  ├── Modular cycle? (Brent + Carmichael) ──────► TIER 4: Period → Lean verify
  │
  ├── Multiplicative? (multiplicativity test) ──► TIER 4.5: Euler product → Lean verify
  │
  ├── OEIS match? (hash lookup) ────────────────► TIER 5: OEIS formula → Lean verify
  │
  ├── LLM recognizes? (constrained prompt) ─────► TIER 6: LLM formula → Lean verify
  │
  ├── Dual-computation? (Python + Lean agree) ──► TIER 7: Algorithm agreement → Lean verify
  │
  └── None of the above ────────────────────────► TIER C: TIR fallback (unverified, honestly labeled)
```

### 11.3 Confidence Tiers (Revised)

| Tier | Layers Passed | Meaning | Estimated accuracy |
|---|---|---|---|
| **A** | All 3 layers, deterministic mining | Verified formula matches correct trace | ~99%+ |
| **A-** | Layers 1+2, Layer 3 partial | Formula verified, some trace-vs-problem checks passed | ~95-98% |
| **B** | Layer 1 + dual-computation (Tier 7) | Algorithm agreement, no formula | ~90-95% |
| **B-** | Layer 1 only, mining failed | Trace consensus but no formula verification | ~85-90% |
| **C** | No layers passed fully | TIR fallback, honestly uncertain | ~70-85% |

**Key properties:**
- Every tier from 1 through 7 produces a Layer 2 verified answer
- Layer 3 (trace-vs-problem) is applied to ALL answers, but its checks are heuristic, not deterministic
- Tier A requires passing ALL three layers with no flags
- Only the final fallback (Tier C) is unverified
- The cascade tries cheapest/most-common patterns first
- Detection is deterministic at every stage — no LLM involvement until Tier 6

---

## 12. What 100% Actually Requires — Honest Assessment

### 12.1 What This System Is

No deployed system provides mathematical guarantees on competition math answers at scale. AlphaProof is the closest — it uses Lean for formal verification — but it requires MCTS proof search that takes days per problem, fails on combinatorics entirely (0/2 at IMO 2024), and requires massive bootstrapping to generate Lean training data from a model that doesn't know Lean. TIR systems (NemoSkills, etc.) provide no guarantees at all — they output an answer and a confidence score from a neural network, which is a prediction, not a proof.

This pipeline is the only system that:
- Produces answers with deterministic mathematical verification (`native_decide`)
- Works at competition time scales (minutes, not days)
- Requires no training data, no fine-tuning, no data moat
- Can detect and flag its own wrong answers (verification failure = uncertainty signal)

The remaining risks are operational, not architectural:
1. **Trace misinterpretation** — the LLM may systematically misunderstand the problem. This is real but addressable: trace consensus, diverse interpretations (§12.2), and deterministic Layer 3 checks (monotonicity, divisibility, boundary conditions) catch the most common failure modes.
2. **Mining gaps** — some sequences don't fit any tier. The extended cascade (7 tiers + dual computation) reduces this to an estimated 5-10%.
3. **Deployment risk** — Lean + native_decide on Kaggle H100, ore_algebra via passagemath, time budgets. These are testable before competition day.

### 12.2 Strategies for the Last 10-20%

**For the 5-10% that resists all verification:** Run K=64 TIR with trace consensus. This won't give Tier A confidence, but it maximizes the probability of a correct answer.

**For reducing certified-wrong answers (2-5%):**

1. **Diverse problem interpretation.** Instead of just generating diverse algorithms, generate diverse INTERPRETATIONS. Prompt the LLM to enumerate possible readings of the problem (labeled/unlabeled, ordered/unordered, with/without repetition). Generate traces for each interpretation. If only one interpretation mines cleanly AND passes the deterministic Layer 3 checks, that's strong evidence. If multiple interpretations mine cleanly, the OEIS descriptions (when available) or explicit boundary values in the problem statement can discriminate.

2. **Cross-problem validation.** If similar problems appear in the set (common in AIMO), check that our approach is consistent across them. If we count labeled graphs for problem 17 but unlabeled for problem 42, something is wrong.

**For hard combinatorics (the GRIDDI class):**

4. **SAT/CSP encoding** when the problem structure permits. SAT witnesses are machine-checkable and bypass the entire trace-interpretation problem.

5. **Detect failure early** (low trace consensus, Layer 3 flags), allocate more compute (time bank algorithm from §19.2 of the spec), and try radically different approaches (different model, different prompt structure, different algorithmic paradigm).

### 12.3 The VG Gap — Separating Verification from Scoring

A critical distinction that the research literature often blurs: **learned scoring is not verification.** PRMs, ORMs, and o3's "internal verifier" are all neural networks — trained statistical models that predict whether an answer is correct. They can be confidently wrong. They provide no mathematical guarantee. Calling them "verifiers" is a misnomer.

| System | Mechanism | Category | Can be wrong? |
|---|---|---|---|
| **PRM** (Lightman et al.) | Neural net trained on human step annotations | **Learned scorer** | Yes — predicts, doesn't prove |
| **ORM** (Cobbe et al.) | Neural net trained on outcome correctness | **Learned scorer** | Yes — predicts, doesn't prove |
| **o3 "internal verifier"** | Neural reward model (details unknown) | **Learned scorer** | Yes — predicts, doesn't prove |
| **AlphaProof** | Full Lean 4 proofs via MCTS | **Formal verification** | No — if proof checks, it's correct |
| **This pipeline** | `native_decide` on formula-trace agreement | **Formal verification** | No — if `native_decide` accepts, the formula matches the trace |

PRMs improve performance by ~6% over majority voting (Lightman et al. 2023). ORMs improve by ~8-12% at N=100 (Cobbe et al. 2021). These are useful but they are in a fundamentally different category from `native_decide`, which is a kernel of mathematical truth. A PRM can assign high confidence to a wrong answer. `native_decide` cannot — if it says `f(n) = 42`, that is a theorem, not a prediction.

LLM self-verification (asking the same model to check its own work) provides ~0% improvement (Stechly et al. 2023). This confirms the core thesis: the verification must be EXTERNAL to the generating model. This pipeline's verification chain (deterministic mining + Lean kernel) is as external as it gets — zero learned parameters, zero LLM involvement in the verification itself.

### 12.4 Targets

Without this pipeline, you have zero mathematical guarantees on any answer and you're competing on TIR ground against teams with massive training data and compute advantages. With it, you have `native_decide` backing every verified answer, confidence calibration that tells you where to spend more compute, and a VG gap that makes scaling productive.

The architecture is ready for implementation. The remaining questions are operational:
- Does Lean + `native_decide` work on Kaggle H100s within time budget?
- Does ore_algebra via passagemath install and run cleanly?
- How many IMO-level problems produce sequences that mine within the 7-tier cascade?

These are answerable by building and testing, not by more design documents.

---

## References

- o3-preview AIMO2 results: https://aimoprize.com/updates/2025-09-05-the-gap-is-shrinking
- AIMO3 launch: https://aimoprize.com/updates/2025-11-19-third-progress-prize-launched
- OpenMathReasoning / NemoSkills: arXiv:2504.16891
- AlphaProof at IMO 2024: Nature (2025), doi:10.1038/s41586-025-09833-y
- AlphaProof blog: https://deepmind.google/discover/blog/ai-solves-imo-problems-at-silver-medal-level/
- o3/o4-mini release: https://openai.com/index/introducing-o3-and-o4-mini/
- Gemini 3 Pro: https://deepmind.google/technologies/gemini/pro/
- ore_algebra: https://github.com/mkauers/ore_algebra
- passagemath: https://github.com/passagemath/passagemath
- Large Language Monkeys (scaling): arXiv:2407.21787
- rStar-Math: arXiv:2501.04519
- Kauers, "D-Finite Functions" (2023), Springer
- Wu's Method for geometry: arXiv:2404.06405
- mpmath PSLQ: https://mpmath.org/doc/current/identification.html
- Beatty sequences: https://en.wikipedia.org/wiki/Beatty_sequence
- Integer relation algorithms: https://en.wikipedia.org/wiki/Integer_relation_algorithm
- SymPy guess module: https://github.com/sympy/sympy/blob/master/sympy/concrete/guess.py
- OEIS: https://oeis.org
- Lightman et al. (2023), "Let's Verify Step by Step" (PRM vs ORM): arXiv:2305.20050
- Cobbe et al. (2021), "Training Verifiers to Solve Math Word Problems" (ORM): arXiv:2110.14168
- Stechly et al. (2023), "GPT-4 Doesn't Know It's Wrong" (self-verification failure): arXiv:2310.01798
- Kimina-Prover (auto-formalization SOTA): https://arxiv.org/abs/2504.11354
- FIMO benchmark (IMO-level formalization): arXiv:2309.04295
