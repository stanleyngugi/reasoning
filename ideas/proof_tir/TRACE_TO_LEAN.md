# Trace-to-Lean: A Neuro-Symbolic Architecture for Verified Mathematical Reasoning

> **"Truth is the Reward."**
> 
> **Status:** Research Concept / Early Prototype  
> **Target:** AIMO 3 Competition (First Proving Ground)  
> **Potential:** General-purpose verified mathematical discovery

---

## Executive Summary

This document describes **Trace-to-Lean**, a novel neuro-symbolic architecture for mathematical reasoning that addresses a fundamental flaw in current AI systems: **they cannot distinguish between code that runs and code that is correct**.

The Artificial Intelligence Mathematical Olympiad (AIMO) represents the vanguard of modern AI evaluation, shifting the focus from broad language competence to deep, multi-step logical reasoning. While AIMO 1 and 2 were dominated by Tool-Integrated Reasoning (TIR) systems—most notably NuminaMath and NemoSkills—analysis suggests that this paradigm is approaching an asymptote. Current TIR systems, which rely on Python execution to verify numerical answers, suffer from a fundamental semantic disconnect: they verify the _calculation_, not the _logic_. This results in "silent execution failures," where models generate correct answers through flawed reasoning.

The pursuit of reliable mathematical AI has traditionally been bifurcated into two camps: neural systems (LLMs) that generate plausible answers but cannot guarantee correctness, and symbolic systems (theorem provers) that provide certainty but require expensive proof search. The Trace-to-Lean architecture represents a **third path** that avoids the limitations of both.

### The Core Insight

Current Tool-Integrated Reasoning (TIR) systems like NuminaMath assume:
```
Code executes successfully → Answer is correct
```

This assumption is **demonstrably false**. Research shows that 5% of test cases produce correct answers with wrong reasoning, and silent logic bugs (off-by-one, precision errors) cause wrong answers without any runtime errors.

**Our approach inverts the paradigm:**
```
Python GUESSES the formula → Lean VERIFIES it is correct
```

We are not asking the LLM to solve problems. We are asking it to:
1. **Experiment** (write Python to compute small cases)
2. **Observe** (collect execution traces)
3. **Synthesize** (mine patterns from traces using deterministic algorithms)
4. **Verify** (prove the pattern in Lean 4 using `native_decide`)

The key insight is that **we are computing, not proving**. The `native_decide` tactic compiles formulas to C++ and executes them — it's a fast computation check, not a slow proof search. This transforms formal verification from an NP-hard search problem into a linear-time computation, making it competition-viable without requiring the massive training or compute of systems like AlphaProof.

### What Trace-to-Lean Is and Is Not

**Trace-to-Lean is NOT:**
- An autoformalization system (we don't translate natural language to formal proofs)
- A proof search system (we don't explore tactic spaces)
- A trained formal reasoning system (we don't fine-tune on Mathlib)
- Using RAG to retrieve lemmas

**Trace-to-Lean IS:**
- A computational verification system
- An orchestration of LLM code generation + deterministic pattern mining + verified computation
- A way to get formal guarantees without proof search
- Using LLMs to generate Python code that computes f(n) for small n
- Using deterministic algorithms (Berlekamp-Massey) to mine patterns from traces
- Translating simple formulas (not proofs) from Python syntax to Lean syntax
- Using `native_decide` to verify the formula by computation

### Why This Is Novel

| Existing Approach | What It Does | Our Difference |
|-------------------|--------------|----------------|
| **TIR (NuminaMath)** | Code solves → assume correct | Code guesses → Lean verifies |
| **AlphaProof** | Train autoformalizer + RL proof search | No training, just prompting + computation |
| **Ramanujan Machine** | Brute-force discovers formulas | We add automated formal verification |
| **LeanConjecturer** | LLM generates conjectures | We mine conjectures from execution traces |
| **DeepSeek-Prover** | Generate proof tactics via MCTS | Generate formula, verify by computation |

**The key distinction:** We are **computing**, not **proving**. `native_decide` compiles the formula to C++ and executes it—it's a fast computation check, not a slow proof search.

### Why No One Has Done This Before

1. **Lean 4 is new (2021).** The `native_decide` tactic that makes this fast didn't exist before Lean 4. Lean 4's ecosystem is still maturing. The insight that you can use it for competition math verification is recent.

2. **Neuro-symbolic researchers think in proofs.** The entire field is focused on natural language → formal proof tactics. The insight that you can use computation-as-verification is overlooked.

3. **The insight crosses communities.** It requires understanding LLM code generation, formal verification, AND competition math simultaneously:
   - **ML researchers** focus on training better reasoners
   - **Formal methods researchers** focus on proof search
   - **Competition math practitioners** focus on heuristics

4. **"Python is formal" is obvious but ignored.** Everyone focuses on the hard problem (natural language → formal) rather than the easy one (code → code).

5. **"Too Simple to Be Novel."** The approach is embarrassingly simple: compute small cases, find the pattern deterministically, verify by computation. It looks like "just running code," but the key is that the verification happens in a trusted environment (Lean) rather than an untrusted one (Python).

This isn't a "no one has tried" warning signal—it's explained by the recency of the enabling technology and the specific cross-domain insight required.

### Why Formal Methods Existed But No One Combined Them This Way

The individual components are NOT new:
- Berlekamp-Massey (1968) — 57 years old
- Gröbner bases (1965) — 60 years old
- Sturm's theorem (1829) — 196 years old
- Sum-of-Squares / Positivstellensatz (1888/1974) — well-established
- PSLQ algorithm (1991) — 34 years old
- Computer algebra systems (1970s) — 50+ years old

**So why hasn't anyone done this?**

1. **The inversion is non-obvious.** Everyone uses these tools to SOLVE problems. We use them to VERIFY solutions that an LLM guessed. The asymmetry (discovery is hard, verification is easy) is known in complexity theory but not applied this way.

2. **LLMs didn't exist.** Before 2020, there was no cheap way to generate problem-specific code/constraints. You'd need a human to write the trace generator for each problem. LLMs make the "Oracle" cheap.

3. **Lean 4's `native_decide` is new (2021).** Previous theorem provers couldn't run fast enough for competition use. Lean 4 compiles to C++, making millisecond verification possible.

4. **The communities don't talk:**
   - Formal methods people build provers, not competition solvers
   - ML people train models, not verification pipelines
   - Competition math people use heuristics, not formal systems

5. **"Verification without proof" sounds wrong.** The formal methods community is proof-obsessed. The idea that you can get formal guarantees by just COMPUTING (not proving) feels like cheating. But for decidable properties, computation IS proof.

6. **The Oracle-Checker split is obvious in retrospect.** Once you see it, you can't unsee it: let untrusted systems find answers, let trusted systems check them. But it requires abandoning the goal of making AI "reason correctly" and accepting AI as a heuristic search engine.

**The architecture is simple precisely because it combines well-understood components in a novel configuration.**

### Why No Training Is Required

This is the potential "moat." Everyone else is:
- Fine-tuning on Mathlib (DeepSeek-Prover)
- Training autoformalizers (AlphaProof: 80,000 TPU-days)
- Curating massive TIR datasets (NuminaMath: 860K problems)

We leverage what LLMs are **already excellent at**:
- Code generation (85%+ compilation success in TIR benchmarks)
- Simple formula translation (translating `n*(n+1)/2` to Lean is easy)

The architecture is **orchestration**, not training. Few-shot prompting guides each step.

---

## Part 1: The Problem — Why TIR Fails

### 1.1 The "Silent Execution" Fallacy

The dominant architecture for AIMO 1 (NuminaMath) and AIMO 2 (NemoSkills) was Tool-Integrated Reasoning (TIR). In this paradigm, the LLM generates natural language mixed with Python code blocks. The code is executed, and the output is fed back into the context. While this solves arithmetic errors, it introduces a more insidious class of error: the "Silent Execution Failure."

A silent execution failure occurs when a program executes without throwing an exception but produces a semantically incorrect result due to logic bugs or unhandled edge cases. The script runs, calculates a number, and the LLM accepts this number as truth because the execution was "successful."

The most dangerous failure mode is when code **runs successfully** but produces the **wrong answer**.

**Evidence from research:**
- 5% of test cases in math benchmarks produce correct answers with wrong reasoning
- Larger models make "far fewer syntactic mistakes than non-syntactic mistakes"
- Non-syntactic mistakes (logic bugs) don't produce errors—they silently corrupt results

**Example: The Off-By-One Trap**
```python
def count_arrangements(n):
    return sum(range(1, n))  # BUG: should be range(1, n+1)
```
- Code runs perfectly, returns a number
- Model outputs it as final answer
- No mechanism to detect the logical error

**Example: The Precision Trap**
```python
import math
result = math.factorial(100) / (math.factorial(50) * math.factorial(50))
# Returns 1.0089...e+29 (float approximation, not exact integer)
```

### 1.2 Correct Answer, Wrong Reasoning (CAWR)

Recent studies reveal that 5-10% of "correct" responses from reasoning models are derived from logically flawed reasoning chains. This "unfaithful reasoning" arises because LLMs are trained on datasets where certain answers are statistically over-represented. A model might "guess" the answer based on surface features and hallucinate a reasoning chain to justify it.

### 1.3 The "Brute Force" Wall

TIR tries to *simulate* problems rather than *solve* them. This hits scaling limits.

**Example: The N=10^50 Cliff**
```python
print(7**7**7 % 10)  # Times out or crashes for large exponents
```

The correct approach requires *mathematical insight* (Euler's Totient Theorem), not more computation.

**Trace-to-Lean approach:**
1. Run `7^1 % 10`, `7^2 % 10`, `7^3 % 10`...
2. Observe trace: `[7, 9, 3, 1, 7, 9, 3, 1...]` (period = 4)
3. Derive formula: Answer = `7^(N mod 4)`
4. Verify in Lean for N=1..100
5. Compute for N=2025

### 1.4 The "Black Box" Trust Issue / Self-Correction Blind Spot

TIR models often "pretend" to verify:
```python
ans = calculate()
if ans > 0:  # Weak, tautological check
    print("Verified")
```

The model generates both solution and test—sharing the same blind spots.

**Research finding:** LLMs fail to correct their own errors in **64.5% of cases** (the "Self-Correction Blind Spot"). They can identify errors in *other* models' outputs but not their own. This means asking an LLM to "check its work" is fundamentally unreliable.

**Trace-to-Lean solves this** by providing an external, deterministic verification signal. Lean doesn't care what the LLM thinks — it either confirms the formula matches the expected sequence, or it doesn't. Lean is the ultimate external critic.

### 1.5 Summary: The Structural Gap

| Failure Mode | TIR (Current SOTA) | Trace-to-Lean |
|--------------|-------------------|---------------|
| **Logic Bugs** | Silent failure (Exit 0, wrong answer) | Compilation error (Lean rejects) |
| **Scaling** | Hits compute/memory wall O(N) | Constant time via formula O(1) |
| **Precision** | Float approximation errors | Symbolic exactness (GMP backend) |
| **Trust** | "I ran code" | "I proved it" |

---

## Part 2: The Solution — Trace-to-Lean Architecture

### 2.1 The Core Workflow

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  PROBLEM        │────▶│  TRACE          │────▶│  INVARIANT      │────▶│  LEAN           │
│  Statement      │     │  GENERATOR      │     │  MINER          │     │  VERIFIER       │
│                 │     │  (LLM → Python) │     │  (SymPy/B-M)    │     │  (native_decide)│
└─────────────────┘     └─────────────────┘     └─────────────────┘     └─────────────────┘
                                │                       │                       │
                                ▼                       ▼                       ▼
                         [1, 3, 6, 10...]         n(n+1)/2              ✓ Verified
                         (Ground Truth)          (Candidate)           (Binary Signal)
```

**Detailed Pipeline:**

```
Problem Statement
       │
       ▼
┌──────────────────┐
│ TRACE GENERATOR  │  LLM writes Python to compute f(n) for n=1..20
│ (LLM + Python)   │  Output: [1, 3, 6, 10, 15, 21, ...]
└──────────────────┘
       │
       ▼
┌──────────────────┐
│ INVARIANT MINER  │  Berlekamp-Massey finds recurrence (deterministic)
│ (SymPy)          │  Or: Lagrange, LLM pattern recognition, OEIS
└──────────────────┘  Output: f(n) = n(n+1)/2
       │
       ▼
┌──────────────────┐
│ FORMULA TRANS.   │  Python syntax → Lean syntax (trivial mapping)
│ (Template)       │  "n*(n+1)//2" → "fun n => n*(n+1)/2"
└──────────────────┘
       │
       ▼
┌──────────────────┐
│ LEAN VERIFIER    │  native_decide checks: does f(n) = expected(n) for n=1..100?
│ (native_decide)  │  Output: true (verified) or false (rejected)
└──────────────────┘
       │
       ▼
   FINAL ANSWER (with verification guarantee)
```

### 2.2 Phase 1: Trace Generator (Python)

**Input:** Math problem statement  
**Output:** Sequence of concrete values `[(1, 1), (2, 3), (3, 6), ...]`

**Method:**
1. Prompt LLM to write Python script that computes f(n) for n=1..15
2. Execute script in sandbox with timeout
3. Capture output trace

**Key prompt pattern:**
```
"Write a Python script to EXPERIMENTALLY determine the answer for small N.
Output a list of integers. Do NOT attempt to solve for N=2025 directly."
```

**Why this works:**
- LLMs excel at code generation (85%+ compilation success, proven by TIR success)
- We're asking for experimentation, not mathematical insight
- Small-trace execution is cheap compared to long reasoning-token rollouts
- Provides "ground truth" data that anchors reasoning in reality

The LLM is NOT solving the problem — it's running an experiment. This leverages what LLMs are good at (code generation) and avoids what they're bad at (mathematical reasoning).

### 2.3 Phase 2: Invariant Miner (Hybrid)

**Input:** Data trace  
**Output:** Mathematical formula f(n)

**Tiered strategy (updated from `pattern_mining.md`):**

| Tier | Method | Target | Implementation | Hallucination Risk |
|------|--------|--------|----------------|-------------------|
| **1** | Polynomial detection + interpolation | Degree-d sequences | Finite differences + `sympy.interpolate` (exact rational) | **Zero** — deterministic algorithm |
| **2** | Berlekamp-Massey (C-finite) | Constant-coefficient recurrences | Fraction-free BM over integers/rationals | **Zero** — deterministic algorithm |
| **3** | Holonomic guesser (P-recursive) | Polynomial-coefficient recurrences (Catalan, factorial-like) | Kauers-style linear ansatz + rational nullspace | **Zero** — deterministic algorithm |
| **4** | Modular cycle mining | Huge-N mod-m problems | Brent/Floyd + Carmichael/Pisano reductions | **Zero** — deterministic algorithm |
| **5** | OEIS lookup (offline) | Named sequences / known recurrences | Hash first 8-10 terms + validate extension | Low — database lookup |
| **6** | LLM pattern fallback | Residual unstructured cases | Constrained prompting + strict validation | Medium — untrusted hypothesis |

**Detection cascade:**
1. Finite-difference polynomial test
2. C-finite test via BM complexity profile
3. Modular-cycle route for explicit mod-m tasks
4. Holonomic guesser
5. Offline OEIS lookup
6. LLM fallback

**Why tiered:**
- Tier 1-4 are deterministic and hallucination-free.
- Tier 5-6 are retrieval/heuristic fallbacks and must be validated against extra terms.
- Candidate formulas are pre-verified in Python before Lean translation.

**Critical implementation corrections (from latest doc):**
- Use `sympy.interpolate` with exact arithmetic, not `numpy.polyfit`, for polynomial mining.
- Never accept BM unless enough terms are available (`k >= 2L` and validated on holdout terms).
- Treat holonomic guessing as a first-class tier; it is a major combinatorics coverage gain over BM+Lagrange alone.

**Berlekamp-Massey specifics:**
- Finds shortest linear recurrence that generates a sequence
- Given sequence [a₁, a₂, ..., aₙ], find the shortest linear recurrence
- Requires 2k terms for degree-k recurrence
- 100% accurate when sequence is linearly recurrent
- Works in O(n²) time

The actual deterministic mining (polynomial/BM/holonomic/modular) does not hallucinate. It either finds a candidate relation consistent with observed data or fails. The remaining risk is not miner hallucination — it is trace correctness and finite-prefix overfitting.

### 2.4 Phase 3: Formula Translation

**Input:** Formula in Python-like syntax  
**Output:** Lean expression

**Actor:** Template-based translator or simple LLM call

**The critical distinction:** We translate **formulas**, not **proofs**.

| Task | Difficulty | Why |
|------|------------|-----|
| Translate formula to Lean | **Trivial** | Syntax mapping between formal languages |
| Generate full proof | Hard | Requires tactic search, lemma selection |

**Example translation:**
```
Python: n * (n + 1) // 2
Lean:   fun n => n * (n + 1) / 2
```

This is **syntax mapping**, not autoformalization. Both Python and Lean are formal languages. The translation is mechanical:
- `*` → `*`
- `//` → `/` (for Nat)
- `**` → `^`
- `factorial(n)` → `Nat.factorial n`

Everyone focuses on the hard problem:
```
Natural language → Formal proof (HARD)
```

We focus on the easy problem:
```
Python formula → Lean formula (EASY)
```

**Critical insight:** If the translation IS wrong, Lean catches it. A mistranslated formula won't match the expected sequence, and `native_decide` rejects it. The verification step IS the translation correctness check.

### 2.5 Phase 4: Lean Verifier (native_decide)

**Input:** Candidate formula + expected sequence  
**Output:** Boolean (Verified / Rejected)

**Actor:** Lean 4 compiler + native_decide

**The verification:**
```lean
def f := fun n => n * (n + 1) / 2
def expected := [1, 3, 6, 10, 15, 21, 28, 36, 45, 55]  -- from trace

theorem verify : (List.range 100).all (fun n => f (n+1) = expected[n]!) = true := by
  native_decide
```

Lean compiles this to C++, runs a loop checking f(n) = expected(n) for n=0..99, and returns true or false. **This is computation, not proof search.**

`native_decide` compiles this to C++ and runs it. It's a loop, not a proof search.

**Template-based translation:**
```lean
-- Prelude (injected into every file)
set_option maxRecDepth 10000
set_option maxHeartbeats 0

def powMod (base exp mod : Nat) : Nat := ...  -- Efficient modular exponentiation

-- The verification
def f := fun n => n * (n + 1) / 2

theorem check : (List.range 100).all (fun n => f n = expected n) = true := by
  native_decide
```

The LLM only fills in the formula slot. The template handles imports, options, and structure.

**Why `native_decide`:**
- Compiles Lean to C++ via the Lean compiler
- Orders of magnitude faster than kernel reduction (`decide`)
- Can verify properties of massive numbers (10^100) limited only by RAM
- Provides binary truth signal in milliseconds

### 2.6 The "Strong Law of Small Numbers" Problem

**Risk:** Pattern fits N=1..10 perfectly but is wrong for N=100.

**Solution:** The Lean verification step checks at **higher N** than Python traced.

```
Python traces N=1..10 → Formula guessed → Lean verifies N=1..100 → Submit
```

This is the **key innovation** that catches overfitting. No other competition system does this.

---

## Part 3: Why This Works — Key Insights

### 3.1 Computation IS Verification (for Decidable Properties)

Traditional theorem proving asks: "Can you derive P from axioms?"

We ask: "Does f(n) = g(n) for n = 1, 2, ..., 100?"

The second question is decidable by computation. `native_decide` performs that computation within Lean's trusted kernel (via C++ compilation).

### 3.2 Formula Translation is Trivial

Everyone focuses on the hard problem (natural language → formal proof). We focus on the easy problem (Python formula → Lean formula).

Both Python and Lean have precise, unambiguous syntax. If we get it wrong, Lean catches it — the formula won't verify.

### 3.3 Deterministic Mining Eliminates Hallucination

The LLM is only used for:
1. Generating trace code (code generation — LLMs are good at this)
2. Final fallback pattern guessing when deterministic tiers fail

The core pattern mining (polynomial/BM/holonomic/modular) is done by **deterministic algorithms**. They don't hallucinate. They either find a valid relation for the observed data or report failure.

### 3.4 External Verification Breaks the Blind Spot

LLMs fail to correct their own errors 64.5% of the time. They can identify errors in others but not themselves.

Lean is the ultimate external critic. It doesn't know what the LLM thinks — it just checks whether the formula matches the expected values. This breaks the self-correction blind spot.

---

## Part 4: Lean 4 and native_decide

### 4.1 How native_decide Works

When you write:
```lean
theorem check : P = true := by native_decide
```

Lean:
1. Synthesizes a `Decidable P` instance
2. **Compiles** the decision procedure to C++ (via Lean's compiler)
3. **Executes** the compiled code
4. If it returns `true`, accepts the proof via the `Lean.ofReduceBool` axiom

This is orders of magnitude faster than kernel reduction (`decide`) because it runs compiled C++ rather than interpreting Lean terms.

### 4.2 Trusted Code Base (TCB) Considerations

Using `native_decide` expands the TCB to include:
- Lean compiler
- C++ compiler
- GMP library (for big integers)

**For competition use, this is acceptable.** The probability of a compiler bug causing a false positive is negligible compared to LLM reasoning errors (5-10%).

**For Mathlib, this is NOT acceptable** — they want proofs that depend only on logical axioms. But we're not publishing to Mathlib; we're winning competitions.

### 4.3 What We DON'T Use

In the **core combinatorics/number-theory path**, we do NOT use:
- `simp` (simplification tactic)
- `rw` (rewrite tactic)
- `omega` (Presburger arithmetic)
- `aesop` (proof search)
- `sorry` (proof holes)
- Mathlib lemmas

We primarily use `native_decide` with bounded computational checks.

For geometry/algebra certificates in the core path, use `grind` for polynomial identities plus `native_decide` for bounded computation checks.

### 4.4 Why Lean 4 Over Alternatives

- **Z3 (SMT):** Fails on recursive number theory (induction)
- **Isabelle:** Too heavy (compilation >100ms)
- **Coq:** Comparable, but Lean's C-compilation is smoother

---

## Part 5: Comparison with Other Approaches

### 5.1 vs. Tool-Integrated Reasoning (TIR)

| Aspect | TIR (NuminaMath) | Trace-to-Lean |
|--------|------------------|---------------|
| Verification | Python executes → assume correct | Lean verifies formula against trace |
| Silent failures | Common (5-10%) | Greatly reduced in deterministic tiers; residual risk remains if trace is wrong |
| Trust model | Trust LLM's code is correct | Trust formula matches verified trace |

### 5.2 vs. AlphaProof

| Aspect | AlphaProof | Trace-to-Lean |
|--------|------------|---------------|
| **Approach** | Autoformalization + RL proof search | Trace mining + computational verification |
| **What Lean does** | Proof search (tactics, lemmas) | Mostly computation checks (`native_decide`) plus lightweight certificate checking (`grind`) |
| **Training required** | 80,000+ TPU-days | **None** — prompting only |
| **Time per problem** | Hours to days | Milliseconds to seconds |
| **Tactics used** | simp, rw, induction, etc. | **Primarily `native_decide` + `grind` (core path)** |
| **Mathlib required** | Yes (massive library) | Core combo+NT path: No (core Lean suffices) |
| **Autoformalization** | Yes (hard problem) | No (trivial syntax mapping) |
| **Failure mode** | Can't find proof | Wrong formula → Lean rejects → retry |

### 5.3 vs. DeepSeek-Prover

| Aspect | DeepSeek-Prover | Trace-to-Lean |
|--------|-----------------|---------------|
| Approach | Generate proof tactics | Generate formula, verify by computation |
| Search | MCTS over tactic space | No search — just run |
| Training | Extensive on formal proofs | None |

**The fundamental distinction:** Other systems search for proofs. We run computations.

**We are not in the proof-search business.** We are in the formula-confirming business.

### 5.4 Full Comparison Table

|**Feature**|**TIR (NuminaMath)**|**AlphaProof**|**Trace-to-Lean**|
|---|---|---|---|
|**Verification**|Python execution|Lean proof search|**Lean computation**|
|**What's verified**|Code runs|Proof compiles|**Formula matches trace**|
|**Training required**|Fine-tuning|Massive RL|**None**|
|**Mathlib required**|No|Yes|**Core path: No (extended tracks may need it)**|
|**Tactics used**|N/A|Many|**Mostly `native_decide` + `grind` in certificate-heavy tracks**|
|**Time per verification**|Milliseconds|Hours|**Milliseconds**|
|**Hallucination risk**|High|Low|**Very low**|
|**Offline feasibility**|High|Low|**High**|

---

## Part 6: The Unified Verification Framework — All Four Domains

### 6.1 The Oracle-Checker Paradigm

The extended Trace-to-Lean architecture rests on a fundamental insight: **verification is polynomial, discovery is exponential**. We separate these concerns:

- **The Oracle (Untrusted):** LLMs, Python, SymPy, numerical optimizers — they FIND answers using any method
- **The Checker (Trusted):** Lean 4 with `native_decide` — it VERIFIES the answer is correct

The Oracle can be wrong, buggy, or hallucinate. The checker reliably catches certificate/computation errors; formalization errors are handled by dual formalization, constraint audits, and over-verification.

### 6.2 Coverage: Updated Realistic Scope

For now, the production-ready claim is strongest for **Combinatorics + Number Theory** (the pattern-mining stack).

| Domain | Status | Current Strategy | Realistic Outcome |
|--------|--------|------------------|-------------------|
| **Combinatorics** | Primary | Polynomial + BM + Holonomic + OEIS + Lean check | Core sequence mined in ~70-80%; deterministic tiers yield formal-style computational guarantees on a substantial subset |
| **Number Theory** | Primary | BM + modular cycles (Carmichael/Pisano) + OEIS + Lean check | Core sequence mined in ~65-75%; deterministic tiers yield formal-style computational guarantees on a substantial subset |
| **Geometry** | Secondary (research) | Coordinate constraints + polynomial checking | Promising, but not yet claim-stable at scale |
| **Algebra** | Secondary (research) | Substitution/certificates + polynomial identities | Promising, but not yet claim-stable at scale |

**Important:** Based on the latest mining analysis, deterministic tiers in combo+NT currently support roughly **55-65% formally checked outputs** end-to-end. Anything above that is fallback or future work, not guaranteed throughput.

### 6.3 The Unified Architecture

```
PROBLEM
   │
   ▼
┌─────────────────────────────────────────┐
│              ROUTER (LLM)               │
│  "Classify: Combo/NT/Geometry/Algebra"  │
└────────────────┬────────────────────────┘
                 │
    ┌────────────┼────────────┬────────────┐
    ▼            ▼            ▼            ▼
┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐
│ COMBO  │  │  NT    │  │  GEOM  │  │ ALGEBRA│
└───┬────┘  └───┬────┘  └───┬────┘  └───┬────┘
    │           │           │           │
    ▼           ▼           ▼           ▼
┌─────────────────────────────────────────┐
│     LLM GENERATES (Problem-Specific):   │
│  • Trace code (10 lines Python)         │
│  • Constraint equations (geometry)      │
│  • Coordinate assignments (geometry)    │
│  • Equation setup (algebra)             │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│     HARDCODED INFRASTRUCTURE:           │
│  • Execute trace, collect output        │
│  • Berlekamp-Massey / Lagrange          │
│  • Coordinate solver (SymPy exact)      │
│  • PSLQ reconstruction                  │
│  • Gröbner cofactor computation         │
│  • SOS decomposition (for inequalities) │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│     CERTIFICATE GENERATOR:              │
│  • Formula (for sequences)              │
│  • Coordinates + Constraints (geometry) │
│  • Solution + Equation (algebra)        │
│  • SOS Decomposition (inequalities)     │
│  • Gröbner Cofactors (universal thms)   │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│     LEAN 4 VERIFIER (Trusted):          │
│  Level 0: native_decide on ℤ/ℚ          │
│  Level 1: native_decide on computable    │
│           algebraic encodings (√d fields)│
│  Level 2: grind for polynomial identities│
│  Level 3: bounded root-isolation checks  │
│           (native_decide on certificates)│
└────────────────┬────────────────────────┘
                 │
                 ▼
           ✓ VERIFIED ANSWER
```

**Geometry/algebra refinement:** In current implementation planning, the GEOM/ALG branch begins with dual formalization and a constraint audit before symbolic solve; treat the diagram as high-level, not strict execution order.

### 6.4 Domain-Specific Verification Strategies

| Domain | Discovery (Oracle) | Certificate | Verification (Checker) |
|--------|-------------------|-------------|----------------------|
| **Combinatorics** | LLM → Python trace | Recurrence/Formula | `native_decide` on formula |
| **Number Theory** | LLM → Python trace | Recurrence/Formula | `native_decide` on formula |
| **Geometry** | Dual LLM formalization → constraints | Coordinates + elimination polynomial + residual checks | `native_decide` + `grind` on certificates |
| **Algebra** | Constraint extraction → exact symbolic solve | Exact root/certificate + substitution residuals | `native_decide` + `grind` |
| **Inequalities** | LLM-guessed decomposition (multi-attempt) | SOS/Schur-style identity certificate | `grind` identity + core nonnegativity lemmas |

### 6.5 Geometry: Constraint Satisfaction Verification

**Updated strategy (from `ALGEBRA_GEOMETRY_STRATEGY (1).md`):**
1. **Dual formalization:** Two independent LLM translations (e.g., Cartesian + complex form).
2. **Constraint audit:** Structural checks before solve (degrees of freedom, sign/domain checks, triangle sanity checks).
3. **Exact-first solve stack:** Sage/Singular (if available) → hardened SymPy exact solve → resultant cascade → numerical+PSLQ fallback.
4. **Over-verification:** Validate ALL stated constraints plus independent geometric consequences not used in solving.
5. **Lean certificate check:** Verify elimination-polynomial/root certificate or polynomial identities with `native_decide`/`grind`.

This keeps the LLM on the translation task and pushes solving/verification into deterministic infrastructure.

**Constraint Translation Table:**

| Geometric Constraint | Polynomial Equation |
|---------------------|---------------------|
| Collinear(A, B, C) | (xB - xA)(yC - yA) - (yB - yA)(xC - xA) = 0 |
| Perpendicular(AB, CD) | (xB - xA)(xD - xC) + (yB - yA)(yD - yC) = 0 |
| Distance(A, B) = k | (xB - xA)² + (yB - yA)² - k² = 0 |
| On Circle(P, center O, radius r) | (xP - xO)² + (yP - yO)² - r² = 0 |
| Midpoint(M, A, B) | xM = (xA + xB)/2 ∧ yM = (yA + yB)/2 |

**Computable algebraic numbers (practical note):**
- For surd-heavy geometry, prefer computable encodings (e.g., explicit `a + b√d` representations) for `native_decide` checks.
- Treat this as an engineering track; do not assume arbitrary algebraic-field support is free in core Lean.

**Failure mode to track explicitly:** incorrect constraint translation remains the dominant risk; dual formalization + audits + independent consequences are the main mitigation.

### 6.6 Algebra: Solution Substitution Verification

**The insight:** solving can be fragile; checking is cheap and deterministic.

**Updated solve hierarchy:**
1. Exact symbolic solve first (Sage/Singular if available, then hardened SymPy exact).
2. Resultant elimination / triangular decomposition when direct multivariate solve stalls.
3. Numerical+PSLQ only as fallback, with strict residual checks.
4. Lean checks candidate against polynomial certificate and original constraints (`native_decide`/`grind`).

**For integer answers (AIMO exploit):**
- AIMO answers are 0-999
- If answer is integer k, just verify f(k) = 0 directly
- Bypass all algebraic number machinery

**Root disambiguation requirement:** whenever multiple algebraic candidates exist, branch/domain constraints must be applied before accepting an answer.

### 6.7 Inequalities: Sum-of-Squares Certificates

**Updated default:** LLM proposes SOS/Schur-style decompositions; Lean verifies identity and nonnegativity lemmas.

**Workflow:**
1. Multi-attempt decomposition proposals (SOS, Schur-SOS, AM-GM style).
2. Deterministic identity verification (`grind`) for `P = Σ qᵢ²` (or equivalent decomposition).
3. Conclude nonnegativity using small core lemmas (avoid Mathlib-only `positivity` dependency in core path).

SDP-based decomposition remains an optional fallback, not the default first line.

### 6.8 Gröbner Basis Certificates (Universal Geometry Theorems)

For universal statements, use a layered strategy:
1. Exact algebraic certificates when feasible (elimination/Gröbner/cofactor style).
2. Bounded/random-witness polynomial checks as high-confidence filters (explicitly probabilistic, not full proof).
3. Promote to formal certificate only when a deterministic symbolic certificate is available.

This keeps universal claims honest: high-confidence screening is separated from true formal certification.

### 6.9 The Router

**Implementation:** Zero-shot LLM classification
```
"Classify this math problem: [Number Theory, Geometry, Algebra, Combinatorics]"
```

**Why it works:** LLMs excel at meta-cognitive recognition even when they can't solve the problem.

### 6.10 Control Flow: Pipeline, Not Agent

**Decision:** Rigid orchestrator beats autonomous agent for math.

**Evidence:**
- NuminaMath (AIMO 1 winner): Self-Consistency loop (not ReAct)
- AlphaProof (IMO silver): MCTS controls everything (LLM only proposes)
- Anthropic research: "Flow Engineering" beats agents for well-defined tasks

**The pipeline:**
1. **Route:** Classify problem domain
2. **Generate:** LLM produces problem-specific code/constraints
3. **Solve:** Hardcoded infrastructure finds answer
4. **Certify:** Generate verification certificate
5. **Verify:** Lean confirms certificate
6. **Vote:** If K samples, use majority among verified answers

### 6.11 What Is Hardcoded vs LLM-Generated

| Component | Hardcoded (Build Once) | LLM-Generated (Per Problem) |
|-----------|----------------------|---------------------------|
| Berlekamp-Massey | ✅ | |
| Lagrange interpolation | ✅ | |
| Holonomic guesser | ✅ | |
| PSLQ reconstruction | ✅ | |
| Constraint audit | ✅ | |
| Hardened SymPy/Sage solve stack | ✅ | |
| Resultant cascade | ✅ | |
| Over-verification library | ✅ | |
| Sturm chain | ✅ | |
| Python-Lean bridge | ✅ | |
| Lean verification templates | ✅ | |
| Trace code | | ✅ (~10 lines) |
| Constraint extraction | | ✅ (geometry/algebra) |
| Solve-order hint (optional) | | ✅ |
| SOS guess/decomposition proposals | | ✅ |

**Updated principle:** the LLM proposes structure (constraints/decompositions/solve order); deterministic code executes, checks, and rejects if inconsistent.

---

## Part 7: Coverage and Formal Guarantee Levels

### 7.1 The Verification Hierarchy

| Level | Domain | Arithmetic Type | Tactic | Speed |
|-------|--------|-----------------|--------|-------|
| 0 | Integer/Modular | ℤ, ℤ/nℤ | `native_decide` | Fastest |
| 1 | Rational | ℚ | `native_decide` | Fast |
| 2 | Computable algebraic encodings | Surd-style fields (engineered types) | `native_decide` | Medium |
| 3 | Polynomial Identity | Commutative ring goals | `grind` (core) | Medium |
| 4 | Root/branch disambiguation | Real algebraic candidates | Certificate checks + `native_decide` | Slower |
| 5 | Numerical fallback | ℝ approximations | High-precision residual checks | Slowest |

### 7.2 What Trace-to-Lean Now Handles

**Combinatorics (current realistic range):**
- Tiling problems (linear recurrence)
- Counting arrangements (polynomial or recurrence)
- Subset counting (binomial sums)
- Catalan/derangement/factorial-style families via holonomic guessing

Estimated outcome:
- Core pattern mined in ~70-80% of problems
- Deterministic tiers produce formally checked outputs for a significant subset

**Number Theory (current realistic range):**
- Modular arithmetic cycles
- Divisibility sequences
- Fibonacci-style recurrences
- Huge-exponent modulo tasks via Carmichael/Pisano reductions

Estimated outcome:
- Core pattern mined in ~65-75% of problems
- Deterministic tiers produce formally checked outputs for a significant subset

**Geometry (research track):**
- Coordinate geometry (rational or algebraic coordinates)
- Distance/angle problems (via polynomial constraints)
- Circle problems (algebraic fields for √)
- Triangle/polygon problems
- Dual formalization + constraint audit + over-verification workflow

**Algebra (research track):**
- Polynomial equations (substitution verification)
- Systems of equations (exact solutions)
- Inequalities (SOS certificates)
- Optimization (verified bounds)
- Exact-first solve hierarchy with numerical fallback only when needed

### 7.3 What Still Falls Back to High-Confidence Numerical

**Current hard slice in geometry/algebra (difficulty-dependent):**
- True transcendentals (sin(1), e^π) — rare in competitions
- Problems requiring synthetic proof (not just computation)
- Highly degenerate edge cases
- High-difficulty olympiad problems with non-polynomial structural insight

**Fallback strategy:** Use high-confidence numerical checks or standard TIR when deterministic certificate paths fail. Keep this explicitly labeled as non-formal.

### 7.4 The Integer Answer Exploit

**Critical observation:** AIMO answers are integers 0-999.

Even if the intermediate computation involves complex algebraic numbers, the FINAL answer is an integer. This massively simplifies verification:

1. Compute algebraic expression for answer
2. If it simplifies to integer k, just verify f(k) = 0
3. Bypass all field arithmetic complexity

This covers the majority of competition problems.

### 7.5 What "Verified" Means in This Pipeline

The current pipeline verifies:
- Candidate formula matches the generated trace on a bounded range checked by Lean.

It does **not** by itself prove:
- The trace program encoded the original problem perfectly.
- The formula is correct for all n without an additional recurrence/induction certificate.

So the strongest honest claim is bounded computational verification plus cross-checking safeguards (independent traces, holdout terms, deterministic miners).

---

## Part 8: Research Validation

### 8.1 Evidence That TIR's Assumption Is Broken

| Finding | Source |
|---------|--------|
| 5% of test cases: right answer, wrong reasoning | arXiv:2502.11574 |
| Non-syntactic bugs far exceed syntactic bugs in large models | arXiv:2411.01414 |
| LLMs fail to self-correct in 64.5% of cases | Self-Correction Bench |
| Prompted self-correction often **degrades** performance | GSM8K studies |

### 8.2 Evidence That Formula Translation Works

| Metric | Value |
|--------|-------|
| Formula/statement compilation to Lean | 85%+ |
| Full proof verification (end-to-end) | <1% |

**Key insight:** We stay in the easy zone (formulas), not the hard zone (proofs).

### 8.3 Evidence That native_decide Is Fast Enough

- `(List.range 1000).length = 1000` works with `native_decide`, times out with `decide`
- Orders of magnitude faster than kernel reduction
- Limited by RAM, not time, for bounded verification

### 8.4 Evidence That Pattern Mining Has Coverage

| Mining Tier | Target | Estimated Coverage Contribution | Verification Quality |
|-------------|--------|-------------------------------|----------------------|
| Polynomial interpolation | Degree-d sequences | Combo ~15-20%, NT ~5-10% | Deterministic |
| C-finite (Berlekamp-Massey) | Constant-coefficient recurrences | Combo ~40-50%, NT ~10-15% | Deterministic |
| Holonomic guesser | Polynomial-coefficient recurrences | Combo ~15-20%, NT ~5% | Deterministic |
| Modular cycle mining | mod-m periodic structures | Combo ~5%, NT ~30-35% | Deterministic |
| OEIS lookup (offline) | Named sequences | +5-10% | Retrieval + validation |
| LLM fallback | Residual cases | Variable | Heuristic only |

Net from current analysis: sequence mining covers roughly ~70-80% of combinatorics and ~65-75% of number theory, with deterministic tiers supporting roughly ~55-65% end-to-end computationally verified outputs.

### 8.5 Novelty Assessment

**No existing system combines:**
1. Python trace generation (using LLM code ability)
2. Deterministic tiered pattern mining (polynomial + BM + holonomic + modular)
3. Few-shot Lean translation (no fine-tuning)
4. Computational verification via `native_decide` (not proof search)

The closest systems either discover formulas without proving (Ramanujan Machine) or prove given statements without discovering (LeanDojo/AlphaProof).

---

## Part 9: Engineering Implementation

### 9.1 Minimal Dependencies

**Python side:**
- `sympy` — polynomial interpolation, recurrence utilities, number theory
- `mpmath` — Arbitrary precision arithmetic (50+ digits)
- `scipy` — Numerical optimization (falls back from mpmath for non-smooth)
- Offline OEIS data (`stripped.gz`, `names.gz`) + hash index
- Custom holonomic guesser (pure Python, exact `Fraction` arithmetic)
- `hypothesis` — Optional property-based testing for trace robustness
- Standard LLM inference library

**Lean side:**
- Lean 4 binary (core combo+NT path: no Mathlib needed)
- ~500 MB total for the core path

**Package size comparison:**
- Full Mathlib: ~5 GB
- Core Lean only: ~500 MB

This dramatically simplifies offline deployment.

### 9.2 Offline Deployment (Kaggle)

For the core combo+NT path (no Mathlib), offline deployment is simple:
1. Pre-package Lean 4 binaries
2. Pre-package standard library `.olean` files
3. Set `PATH` to include Lean binaries
4. Done

No need for the complex Mathlib cache setup that full formal provers require.

If you enable advanced theorem-prover workflows beyond the core `native_decide`/`grind` path, include a Mathlib cache and adjust disk budget accordingly.

**Feasible.** Mathlib `.olean` cache is ~2-3GB. Upload `lean4` binaries as Kaggle Dataset.

### 9.3 Directory Structure

```
/kaggle/input/lean-offline/
├── lean4/                    # Lean 4 binaries
│   └── bin/
│       ├── lean
│       └── leanc
├── lib/                      # Standard library .olean files
└── verify_template.lean      # Our verification template
```

### 9.4 Python-Lean Integration

Simple subprocess call — no complex IPC needed:

```python
def verify_formula(formula_lean: str, expected: list[int]) -> bool:
    lean_code = TEMPLATE.format(formula=formula_lean, expected=expected)
    with open("verify.lean", "w") as f:
        f.write(lean_code)
    result = subprocess.run(["lean", "--run", "verify.lean"], capture_output=True)
    return result.returncode == 0
```

### 9.5 Engineering Constraints (AIMO 3)

**Hardware:**
- H100 GPUs available (major upgrade from T4/L4)
- 5 hours runtime limit
- No internet access
- 20GB disk, 30GB RAM on Kaggle

### 9.6 Time Budget

5 hours / 50 problems = 6 minutes per problem

| Stage | Time Budget |
|-------|-------------|
| LLM trace generation | 30s |
| Python execution | 5s |
| Berlekamp-Massey | <1s |
| Lean verification | 1-5s |
| Retry loop (if needed) | 2 min |
| Fallback to TIR | remaining |

**Alternative breakdown:**

| Phase | Budget |
|-------|--------|
| Reasoning / Python generation | 1 min |
| Execution / Search / Verification | 4 min |
| Buffer | 1 min |

### 9.7 Engineering Considerations

**Small-N Overfitting:**
- **Concern:** Pattern fits N=10 but is wrong for N=100.
- **Solution:** Lean verification checks at higher N than Python traced. This is the **primary purpose** of the Lean step—catching overfitting to small samples.

**Formula Translation:**
- **Non-issue.** This is syntax mapping between formal languages, not autoformalization.
- If the translation is wrong, Lean rejects it (the formula won't match the expected sequence). The verification step catches translation errors automatically.

**native_decide Bounds:**
- **Consideration:** Verification for ∀ n < N runs in O(N) time.
- **Practice:** N=100 to N=1000 is fast (milliseconds). Sufficient to catch overfitting while staying within time budget.

**Router Accuracy:**
- **Consideration:** Misclassified problem goes to wrong module.
- **Practice:** LLMs have excellent meta-cognitive ability to recognize problem types. Zero-shot classification is high-accuracy.

**Geometry/Algebra Formalization Risk:**
- **Consideration:** Constraint mistranslation is the dominant failure mode in non-sequence domains.
- **Practice:** Use dual formalization (independent prompts/representations), then run a deterministic constraint audit before solve.

**Symbolic Solver Fragility:**
- **Consideration:** Exact CAS pipelines can hang or return incomplete branches on hard multivariate systems.
- **Practice:** Use hardened solve order: exact solve → resultant/triangular decomposition → numerical+PSLQ fallback, with strict residual checks.

**Branch Ambiguity:**
- **Consideration:** Multiple algebraic candidates may satisfy intermediate equations.
- **Practice:** Keep candidates until final pruning by all original constraints + domain constraints, then certify selected root in Lean.

### 9.8 Lean Safety Protocol

**Critical trap:** `native_decide` computes `7^10000` before taking mod, causing RAM crash.

**Fix:** Inject efficient `powMod` function into every Lean file:
```lean
def powMod (base exp mod : Nat) : Nat :=
  if mod == 0 then 0 else
  if mod == 1 then 0 else
  let rec loop (b e acc : Nat) : Nat :=
    if e == 0 then acc else
    let acc' := if e % 2 == 1 then (acc * b) % mod else acc
    loop ((b * b) % mod) (e / 2) acc'
  loop (base % mod) exp 1
```

**Config requirements:**
```lean
set_option maxRecDepth 10000
set_option maxHeartbeats 0
```

---

## Part 10: Performance Projections

| Metric | TIR (NuminaMath) | Trace-to-Lean |
|--------|------------------|---------------|
| **AIMO 1 Score** | 29/50 | — |
| **AIMO 2 Score** | 34/50 | — |
| **Precision (verified answers)** | Unknown | Very high on deterministic verified subset |
| **Coverage (problems attempted)** | 100% | ~70-80% combo mining / ~65-75% NT mining; full coverage via fallback |
| **Silent failure rate** | 5-10% | Greatly reduced on deterministic verified subset |

**Hybrid Strategy:**
1. Attempt Trace-to-Lean verification
2. If verified, output with confidence = 1.0
3. If verification fails after N retries, fall back to TIR
4. Output TIR answer with confidence = 0.5

This ensures we never score zero while "locking in" verified answers.

---

## Part 11: Key Arguments and Rebuttals

### "Isn't this just TIR with extra steps?"

No. TIR assumes execution validates correctness. We **reject that assumption** and add formal verification. The execution is for **discovery**, not **validation**.

### "AlphaProof already does neuro-symbolic math."

AlphaProof requires:
- Trained autoformalizer
- 80,000 TPU-days of training
- 2-3 days per problem

We require:
- Off-the-shelf LLMs
- Few-shot prompting
- Milliseconds per verification

Different paradigm entirely.

### "Berlekamp-Massey only works on linear recurrences."

True for one tier. But:
1. Many competition counting problems ARE linearly recurrent (tiling, Fibonacci-style DP)
2. Polynomial tier catches degree-d sequences directly (finite differences + interpolation)
3. Holonomic tier captures Catalan/factorial-like families BM misses
4. Modular cycle tier captures a large fraction of huge-N number-theory mod tasks
5. OEIS/LLM are fallbacks, not the core guarantee path

---

## Part 12: Summary

### The Thesis

LLMs are good at code generation. Deterministic algorithms are good at pattern mining. Lean is good at verification. **Combine them.**

### The Architecture

```
Problem → LLM (Python trace) → SymPy (mine pattern) → LLM (Lean formula) → native_decide (verify)
```

### The Innovation

1. **Computing, not proving** — `native_decide` is fast
2. **Formulas, not proofs** — translation is easy
3. **Prompting, not training** — no fine-tuning required
4. **Deterministic mining stack** — polynomial/BM/holonomic/modular tiers don't hallucinate
5. **Verification catches TIR's silent failures** — the key differentiator

### The Paradigm

```
Python GUESSES the formula → Lean VERIFIES it is correct
```

### The Guarantee

If Lean accepts the check, it certifies that the candidate formula matches the generated trace on the verified range (e.g., n = 1..100). This is a strong bounded computational guarantee, not a universal proof over all n.

### The Coverage

- **Combinatorics (primary target):** ~70-80% core sequence mining with current tiers; deterministic tiers provide the formal-style computationally verified subset.
- **Number Theory (primary target):** ~65-75% core sequence mining; strongest wins are modular-cycle and recurrence classes.
- **Geometry/Algebra (secondary):** active research directions with promising certificate workflows, but not yet coverage-stable claims.

Across combo+NT deterministic tiers, current realistic end-to-end computational verification is roughly in the ~55-65% range.

### AIMO Competition Context

| Competition | Winner     | Key Technique                             |
| ----------- | ---------- | ----------------------------------------- |
| AIMO 1      | NuminaMath | SC-TIR (code + voting)                    |
| AIMO 2      | NemoSkills | GenSelect + TIR                           |
| AIMO 3      | TBD        | H100 available, 120B models hitting 42/50 |

No winner has used formal verification. This could be a differentiator.

### Hallucination Risk Summary

| Component | What It Does | Hallucination Risk |
|-----------|--------------|-------------------|
| Trace Generator | LLM writes Python to compute f(n) | Low (code gen is reliable) |
| Deterministic mining tiers | Polynomial/BM/Holonomic/Modular discovery | **Zero** (algorithmic) |
| OEIS lookup | Named sequence retrieval | Low (must validate against extra terms) |
| LLM fallback | Pattern guess on residual cases | Medium (untrusted hypothesis) |
| Formula Translation | Python syntax → Lean syntax | Low (if wrong, Lean rejects) |
| native_decide | Runs verification as computation | **Zero** (trusted execution) |

**End-to-end risk note:** The dominant residual risks are trace correctness and finite-prefix overfitting, not miner hallucination.

---

Trace-to-Lean represents a paradigm shift from **"trusting execution"** to **"verifying computation."** By using Lean 4's `native_decide` as a computational oracle — not a proof assistant — we achieve the benefits of formal verification without the costs of proof search.

This architecture is competition-viable today with off-the-shelf LLMs, deterministic mining infrastructure, and Lean 4 binaries. It targets verified-by-computation reasoning without the cost of training autoformalizers or running large proof-search loops.

---

## Part 13: V2 Operational Contract

To prevent architecture drift between strategy documents and implementation, the stage-level execution contract now lives in:

- `ideas/proof_tir/V2_EXECUTION_SPEC.md`
- `ideas/proof_tir/V2_RESEARCH_BASELINE.md` (evidence-backed rationale and decision boundaries)

Use that file as the source of truth for:
- stage inputs/outputs and pass/fail gates
- certificate requirements per domain
- confidence-tier policy and fallback rules
- failure taxonomy and logging requirements

This document (`TRACE_TO_LEAN.md`) is the conceptual architecture. `V2_EXECUTION_SPEC.md` is the deployable controller contract.
`V2_RESEARCH_BASELINE.md` captures the durable evidence-backed assumptions behind that contract.
