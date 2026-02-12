## Part V: 2026 Corrections and Contract Alignment (Authoritative Addendum)

This addendum supersedes conflicting policy-level statements earlier in this file.

Normative precedence:
1. `ideas/proof_tir/V2_EXECUTION_SPEC.md` controls runtime behavior.
2. `ideas/proof_tir/V2_RESEARCH_BASELINE.md` captures research assumptions and uncertainty boundaries.
3. This document remains a verifier handbook and technical deep dive.

### 42. Critical Corrections

### 42.1 \"No false positive\" wording

Earlier wording implies an absolute no-false-positive property.

Corrected statement:
- Untrusted generators cannot force acceptance when certificate/residual checks fail.
- But wrong formalization can still produce internally consistent wrong-world certificates.
- Therefore formalization is a first-class risk and must be mitigated explicitly.

Mandatory mitigation set:
- dual formalization (or equivalent independent extraction paths)
- deterministic constraint audit
- independent consequence checks

### 42.2 Core tactic strategy

Earlier sections over-emphasize `ring` as the primary continuous-domain primitive.

Corrected baseline:
- Core path: `native_decide` + `grind`.
- Mathlib path: `ring`, `field_simp`, `linarith`, `nlinarith`, `positivity`, `linear_combination`, `norm_num`.

Interpretation:
- Kernel-checked Mathlib proofs remain strongest where available.
- Core path remains viable for deployment-constrained settings.

### 42.3 Sanitization hard gate

Sanitization is mandatory before checker invocation for generated Lean artifacts.

Minimum hard rules:
- strip `@[implemented_by]`
- strip `@[extern]`
- reject non-allowlisted imports
- reject checker code outside template contracts

No checker invocation without sanitizer pass.

### 43. Geometry/Algebra Risk Model Update

Dominant residual risks:
1. formalization mismatch
2. branch ambiguity
3. insufficient certificate class for claim type

Tier A (geometry/algebra) requires:
- audit-strength formalization path
- branch tracking (no implicit first-root acceptance)
- full residual checks against original constraints
- certificate class appropriate to domain
- checker pass

### 44. Elimination Workflow Correction

Elimination roots are necessary conditions, not sufficient.

Safe acceptance protocol:
1. domain sieve
2. back-substitution extension existence
3. exact residual checks on original constraints
4. branch/domain disambiguation
5. bounded-answer policy checks (when contest format requires bounds)

If any step fails, reject or downgrade.

### 45. Inequality Discovery and Verification Update

Do not treat SDP as the only default discovery path.

Preferred sequence:
1. decomposition-first proposal search (SOS/Schur/AM-GM structure)
2. deterministic identity checking (`grind`/`ring` by profile)
3. deterministic nonnegativity path via lemmas/checker
4. SDP-based search as fallback

If decomposition identity cannot be certified, downgrade or fallback.

### 46. Corrected Template B (Array-Only)

Earlier Template B uses `List.range`, conflicting with Array-only performance guidance.
Use this corrected version:

```lean
def fib_arr (n : Nat) : Nat :=
  if n == 0 then 0
  else if n == 1 then 1
  else
    let arr := (Array.range (n - 1)).foldl
      (init := #[(0 : Nat), 1])
      (fun memo _ =>
        let k := memo.size
        let v := memo[k - 1]! + memo[k - 2]!
        memo.push v)
    arr[n]!
```

### 47. Bounded vs Universal Claim Discipline

Rules:
- bounded `native_decide` checks certify bounded claims only
- universal claims require symbolic certificate checking (or explicit probabilistic label)
- never promote finite-range checks to universal correctness without proof obligations

### 48. Deployment Clarifications

1. persistent Lean and symbolic workers are preferred
2. repeated cold starts in scoring loops are non-competitive
3. offline packaging assumptions must be validated before competition freeze

### 49. Unified Trust Statement (Revised)

Trusted:
- kernel for proof-term validation
- runtime/compiler path when using `native_decide` (competition-acceptable trust expansion)

Untrusted:
- LLM generation
- solver proposals
- decomposition proposals
- reconstruction guesses

Enforcement:
- untrusted outputs become candidates only
- acceptance requires audit + certificate + checker pass at target tier

### 50. Immediate Implementation Checklist

- [ ] enforce `S1.5` sanitizer gate before checker calls
- [ ] enforce branch tracking (ban implicit `sol[0]` acceptance)
- [ ] enforce elimination safe acceptance protocol for algebra/geometry Tier A
- [ ] use Array-only templates for `native_decide` data-heavy checks
- [ ] enforce bounded/universal claim labeling discipline
- [ ] enforce correlation-aware confidence (no naive independence multiplication)

This addendum is the authoritative correction layer for this handbook.

# Core Verification Engine: `native_decide`, `ring`, and Python-to-Lean Translation

  

This document is the unified technical reference for the verification engine of the Trace-to-Lean architecture: (1) the Lean 4 `native_decide` tactic for computational verification of discrete formulas and algebraic number field arithmetic, (2) the `ring`/`grobner`/`nlinarith` tactic ecosystem for polynomial identity and inequality verification with zero TCB expansion, (3) the Python-to-Lean formula translation layer, and (4) the extension to geometry and algebra via coordinate constraint satisfaction, SOS certificates, and Gröbner basis certificate checking. Every claim in this document has been verified against primary sources: the Lean 4 language reference, the Lean 4 source code, Lean Zulip archives, the Lean 4 system paper by de Moura & Ullrich, the FPiL textbook, Mathlib documentation, and the CPP 2025 paper by Baanen et al. on certifying rings of integers.

  

---

  

## Part I: `native_decide` — Computation as Verification

  

### 1. What `native_decide` Is

  

`native_decide` is a tactic that discharges a goal `⊢ P` by:

  

1. Synthesizing a `Decidable P` instance — a computable function that returns `true` or `false` for `P`.

2. Evaluating that function via the Lean interpreter (or compiled native code).

3. If the result is `true`, accepting the proof via the `Lean.ofReduceBool` axiom.

  

It is a synonym for `decide +native`. The key distinction from `decide` (which reduces the term inside the kernel) is that `native_decide` delegates computation to the runtime, bypassing the kernel's slow symbolic reduction.

  

**Source:** [Lean Tactic Reference](https://lean-lang.org/doc/reference/latest/Tactic-Proofs/Tactic-Reference/)

  

### 2. The Theoretical Basis: Proof by Reflection

  

To prove a proposition `P`, one can define a decision procedure `f : Input → Bool` and a soundness theorem `∀ x, f(x) = true → P(x)`. To prove `P(a)`, compute `f(a)`. If it returns `true`, the soundness theorem justifies `P(a)`.

  

- **`decide`** (small-scale reflection): The kernel symbolically evaluates `f(a)`. Secure but slow — the kernel represents `Nat` as unary chains of `succ` constructors. A number like 1,000,000 is a linked list of one million constructors.

- **`native_decide`** (large-scale reflection): The runtime evaluates `f(a)` using GMP-backed integers, compiled C code, and contiguous arrays. Orders of magnitude faster, but trusts the compiler.

  

### 3. The Compilation Pipeline

  

When `native_decide` is invoked, the elaborated `Decidable` instance flows through:

  

```

Lean Source → Elaboration → LCNF → LCNF (optimized) → IR → C code → Binary execution

```

  

#### 3.1 LCNF (Lean Compiler Normal Form)

  

A functional intermediate representation based on A-normal form (ANF). Complex expressions are decomposed into sequences of let-bindings. The compiler performs inlining, specialization (monomorphization of polymorphic functions), and dead code elimination at this stage.

  

**Source:** [LCNF Types documentation](https://leanprover-community.github.io/mathlib4_docs/Lean/Compiler/LCNF/Types.html), confirmed as ANF-based: "The code generator uses a format based on A-normal form." Design also draws from the paper "Compiling without continuations."

  

#### 3.2 IR (Intermediate Representation)

  

LCNF is lowered to an imperative IR that handles:

  

- **Reference counting:** Insertion of `inc`/`dec` instructions (Perceus algorithm).

- **Boxing/Unboxing:** Deciding whether values are heap-allocated pointers or raw machine scalars.

- **Destructor reuse:** `reset`/`reuse` instructions that enable memory recycling for constructors.

  

**Source:** Ullrich & de Moura, "Counting Immutable Beans" (arXiv:1908.05647); Huisinga master thesis (KIT, 2023).

  

#### 3.3 The Backend: Interpreter vs. Native Code

  

**By default, `native_decide` uses the Lean interpreter.** The interpreter is a highly optimized bytecode executor written in C++. It executes the IR directly, avoiding the latency of invoking an external C compiler.

  

When the `precompileModules` Lake option is set, Lean generates **C code** (not C++) for each module. This C code is compiled by a **bundled `clang`** (currently clang 19.x) into a shared object (`.so`/`.dylib`) that is dynamically loaded. This path unlocks LLVM optimizations (loop unrolling, vectorization) for maximum performance.

  

**Critical correction from prior documentation:** The generated code is **C**, not C++. The Lean runtime and kernel are written in C++, but the per-module compiler output is C. The Lean 4 system paper explicitly states: "The new compiler produces C code." Lean also bundles `lld` (LLVM linker) and `libc++`.

  

**Sources:** [Lean Language Reference — Elaboration and Compilation](https://lean-lang.org/doc/reference/latest/Elaboration-and-Compilation/); [Lean 4 system paper](https://lean-lang.org/papers/lean4.pdf)

  

### 4. The Axiom: `Lean.ofReduceBool` and `Lean.trustCompiler`

  

The logical validity of `native_decide` rests on:

  

```lean

axiom Lean.ofReduceBool (a b : Bool) (h : Lean.reduceBool a = b) : a = b

```

  

`Lean.reduceBool` is an `opaque` definition. When the kernel encounters `Lean.reduceBool c`, it invokes the Lean interpreter to evaluate `c`. If the result is `true`, the axiom bridges the gap between runtime computation and logical truth.

  

Since Lean v4.2.0-rc2 ([PR #2654](https://github.com/leanprover/lean4/pull/2654)), this axiom transitively depends on:

  

```lean

axiom Lean.trustCompiler : True

```

  

This axiom explicitly marks that the proof depends on the correctness of the Lean compiler, interpreter, and all `@[implemented_by]` and `@[extern]` annotations. Any theorem using `native_decide` will show both `Lean.ofReduceBool` and `Lean.trustCompiler` in `#print axioms` output.

  

**Note:** Prior to Lean 4.23.0, a bug ([issue #8840](https://github.com/leanprover/lean4/issues/8840), fixed in [PR #8842](https://github.com/leanprover/lean4/pull/8842)) caused `collectAxioms` to miss `Lean.trustCompiler` in the transitive closure. This means older Lean versions might show `Lean.ofReduceBool` but not `Lean.trustCompiler`.

  

### 5. Runtime Representation: How Lean Executes Math

  

#### 5.1 `Nat` and `Int`: GMP-Backed Arithmetic

  

At runtime, `Nat` and `Int` are **not** unary inductive types. Both are implemented using **GMP (GNU Multiple Precision Arithmetic Library)**.

  

- **Small integers** (values < 2^63 on 64-bit systems) are stored directly as tagged scalars — no heap allocation. The lowest-order bit of the object pointer is used as a tag to distinguish scalars from heap pointers. This means natural numbers smaller than 9,223,372,036,854,775,808 require zero allocation.

- **Large integers** (values ≥ 2^63) are automatically promoted to heap-allocated `mpz_t` structures managed by GMP.

- **Kernel primitives:** Basic operations (`Nat.add`, `Nat.mul`, `Nat.beq`, `Nat.ble`, `Nat.gcd`, etc.) are overridden by C++ primitives in the kernel. The official docs note: "Because they are part of the kernel, if these primitives did not correspond to their definitions as Lean functions, it could undermine soundness."

  

**Sources:** [Lean Language Reference — Natural Numbers](https://lean-lang.org/doc/reference/latest/Basic-Types/Natural-Numbers/); [Lean Language Reference — Integers](https://lean-lang.org/doc/reference/latest/Basic-Types/Integers/)

  

#### 5.2 `Array`: Contiguous Memory with In-Place Mutation

  

Lean's `Array` type compiles to a dynamic array — a contiguous block of memory with O(1) random access. The C FFI type is:

  

```c

typedef struct {

    lean_object   m_header;

    size_t        m_size;

    size_t        m_capacity;

    lean_object * m_data[];

} lean_array_object;

```

  

**Functional But In-Place (FBIP):** When an array has a reference count of 1 (uniquely owned), operations like `push`, `set!`, and `swap!` are performed as **destructive in-place mutations**, avoiding the O(n) copy cost of naive functional updates. This is a runtime optimization based on reference counting, not a static linear types system. Lean provides `dbgTraceIfShared` to verify at runtime that arrays are being mutated in place.

  

**Sources:** [Lean Language Reference — Arrays](https://lean-lang.org/doc/reference/latest/Basic-Types/Arrays/); [FPiL — Insertion Sort and Array Mutation](https://leanprover.github.io/functional_programming_in_lean/programs-proofs/insertion-sort.html)

  

#### 5.3 `List`: Linked List — Avoid for `native_decide`

  

Lean's `List` is a linked list with O(n) random access and dispersed memory layout (pointer chasing, cache misses). A nested loop over a `List` with index access turns an O(n²) algorithm into O(n³).

  

**Mandate for our architecture:** Always use `Array` for data collections passed to `native_decide`. Use `Array.range`, `Array.foldl`, `Array.map`.

  

#### 5.4 Tail Call Optimization

  

Lean 4 eliminates **direct self-recursive tail calls only**. A recursive call in the tail position is compiled into a loop, reusing the current stack frame. However, **mutual recursion and tail calls to other functions are NOT optimized** — they consume stack space normally.

  

This means:

- `def sum_tco (acc : Nat) : List Nat → Nat | [] => acc | h :: t => sum_tco (acc + h) t` — **safe**, compiled to a loop.

- A function calling a different function in tail position — **not optimized**, stack overflow risk at depth >200K.

  

**Source:** [FPiL — Tail Recursion](https://leanprover.github.io/functional_programming_in_lean/programs-proofs/tail-recursion.html): "At the time of writing, Lean only eliminates direct tail calls in recursive functions."

  

### 6. The Trusted Code Base (TCB)

  

Standard Lean proofs (kernel-checked) trust only:

1. The kernel (~few thousand lines C++)

2. The C++ compiler used to build Lean

3. The hardware

  

When `native_decide` is used, the TCB expands to include:

  

1. **The Lean compiler** (30k+ lines) — LCNF transformations, IR generation, C code emission.

2. **The Lean interpreter/VM** — the C++ bytecode executor.

3. **The runtime system** — garbage collector, object model, reference counting.

4. **GMP** — arbitrary precision arithmetic.

5. **Bundled `clang`** and `lld` — C compilation and linking.

6. **All `@[implemented_by]` annotations** — runtime replacements for logical definitions.

7. **All `@[extern]` annotations** — FFI calls to C code.

  

The `Lean.reduceBool` docstring explicitly quantifies this: "by using this feature, the Lean compiler and interpreter become part of your trusted code base. This is extra 30k lines of code."

  

**For competition use, this is acceptable.** `native_decide` expands the runtime TCB (compiler/interpreter path), while Mathlib proof-term paths minimize that runtime trust expansion. This is a trust-model tradeoff, not a replacement for formalization quality controls.

  

### 7. Known Soundness Issues and Historical Bugs

  

#### 7.1 The `IO.getRandomBytes` Exploit (Real, Documented)

  

Discovered by Mario Carneiro on [Lean Zulip](https://leanprover-community.github.io/archive/stream/270676-lean4/topic/soundness.20bug.3A.20native_decide.20leakage.html):

  

```lean

def foo : Bool := match IO.getRandomBytes 1 () with

  | .ok bs _ => bs[0]! >= 128 | _ => false

theorem T1 : false = Lean.reduceBool foo := rfl

theorem T2 : Lean.reduceBool foo = true := rfl

theorem contradiction : False := nomatch T1.trans T2

```

  

This exploited two flaws: (a) `IO.RealWorld` was not opaque, allowing construction of a `RealWorld` value from `()`; (b) `reduceBool` was declared as `opaque` rather than `axiom`, so `#print axioms contradiction` reported no axiom dependencies. Both were fixed:

- [lean4 PR #2654](https://github.com/leanprover/lean4/pull/2654): Added `axiom Lean.trustCompiler : True`.

- [mathlib4 PR #2656](https://github.com/leanprover-community/mathlib4/pull/2656): Made `IO.RealWorld` opaque.

  

Mario also demonstrated a **deterministic variant** using `@[implemented_by]` with `unsafeBaseIO` that works 100% of the time, proving the issue is fundamental to the `reduceBool` mechanism.

  

#### 7.2 `@[implemented_by]` — The "Shadow Logic"

  

The `@[implemented_by]` attribute replaces a function's runtime implementation with a different (possibly `unsafe`) function. The kernel sees the logical definition; the compiler sees the replacement. If they diverge, soundness breaks under `native_decide`.

  

Mario Carneiro demonstrated:

```lean

def six := 6

@[implemented_by six] def zero := 0

#eval #[1, 2, 3].get ⟨zero, by decide⟩  -- memory unsafe!

```

  

As Mario stated: "`implemented_by` is only unsafe in conjunction with `reduceBool`, but either way it's still signposted."

  

Additionally, `@[csimp]` (the "safe" version of `@[implemented_by]`) can also smuggle axioms into proofs via `native_decide` because axioms used in the `csimp` proof are not propagated through `native_decide`'s axiom tracking ([GitHub issue #7463](https://github.com/leanprover/lean4/issues/7463)).

  

#### 7.3 GMP-Related Bugs

  

According to Jason Rute on [Proof Assistants Stack Exchange](https://proofassistants.stackexchange.com/questions/5252/): GMP-based `native_decide` bugs have been "common in Lean 4." The bugs were in Lean's C++ code that implements functions like `Nat.mod` incorrectly (not matching the Lean mathematical definition), not in GMP itself. These allow reducing an expression two different ways and proving `False`. Mario Carneiro confirmed on Hacker News: "there have been 3 soundness bugs in lean 4 so far. (They were all fixed within hours.)"

  

#### 7.4 Competition Relevance

  

For our architecture, these bugs are irrelevant because:

1. We never use `unsafe`, `@[implemented_by]`, `@[extern]`, or `IO` in our verification code.

2. We use only standard arithmetic operations on `Nat` and `Int`.

3. The probability of a GMP arithmetic bug (already rare and quickly patched) is vastly lower than the probability of an LLM reasoning error.

  

### 8. `maxHeartbeats` and `native_decide` — A Critical Distinction

  

`maxHeartbeats` limits the elaborator's **deterministic WHNF reduction steps**. It does NOT limit the execution time of the compiled binary that `native_decide` runs.

  

An infinite loop inside `native_decide` will hang the process until the external supervisor (Kaggle/Docker timeout) kills it. There is no built-in Lean timeout for the native execution phase.

  

**Defensive pattern:** Use a `fuel` parameter that decrements on each recursive call, ensuring structural termination regardless of input:

  

```lean

def check (n : Nat) (fuel : Nat := 100000) : Bool :=

  match fuel with

  | 0 => false

  | fuel' + 1 => -- actual computation, recursing with fuel'

```

  

### 9. Comparison with Other Tactics

  

| Tactic | Evaluation Method | Axioms Introduced | Proof Object | Primary Use |

|---|---|---|---|---|

| **`decide`** | Elaborator reduction | None | Full `Expr` tree | Small finite domains |

| **`decide +kernel`** | Kernel reduction (ignores transparency) | None | Full `Expr` tree | When elaborator fails |

| **`native_decide`** | Interpreter/compiled native code | `Lean.ofReduceBool`, `Lean.trustCompiler` | `ofReduceBool` application | Heavy computation |

| **`norm_num`** | Tactic-driven rewriting (Mathlib) | None | Full `Expr` tree (kernel-checkable) | Concrete arithmetic |

| **`omega`** | Presburger arithmetic solver | None | Full `Expr` tree | Linear integer arithmetic |

| **`bv_decide`** | SAT solver + LRAT certificate | `Lean.ofReduceBool` (via reflection) | LRAT-verified certificate | Bitvector/Bool |

  

**`norm_num`** produces full kernel-checkable proof terms with no TCB expansion. It uses a binary representation for numbers and is fast for concrete arithmetic. Requires Mathlib.

  

**`omega`** is a (partial) Presburger arithmetic decision procedure following William Pugh's "The omega test." It handles `+`, `-`, `*` (by constants), `/`, `%`, divisibility, and comparisons over `Nat` and `Int`. It omits the "dark" and "grey" shadows, making it incomplete in rare edge cases. Produces full proof terms.

  

**`bv_decide`** bitblasts `BitVec`/`Bool` goals into CNF, calls the CaDiCaL SAT solver, gets an LRAT UNSAT proof, and verifies it using a Lean-implemented LRAT checker. The LRAT certificate is verified via `ofReduceBool`, so the compiler is still in the TCB.

  

### 10. Offline Deployment (Kaggle/AIMO)

  

`native_decide` requires only the bundled Lean toolchain. No internet access needed.

  

- The **interpreter** (default mode) has zero external toolchain dependencies — it executes IR directly.

- If using native compilation, the bundled `clang` must be accessible via `PATH`. The standard Lean installation (`elan`) includes it at `~/.elan/toolchains/.../bin/clang`.

- **Profile A (discrete + algebraic geometry via custom types):** Pre-package Lean 4 toolchain (~500-875 MB). No Mathlib needed. Tactics: `native_decide`, `omega`, `grobner`.

- **Profile B (full pipeline including `ring`/`nlinarith`):** Pre-package Lean 4 toolchain + Mathlib `.olean` cache (~5-6 GB total). Required for inequality verification, universal geometry theorems, and `norm_num`. On Kaggle (20 GB disk), this fits with room to spare. Pre-build with `lake exe cache get` on an internet-connected machine, bundle `.lake/` and `.elan/toolchains/<version>/` as Kaggle datasets.

  

See Section 34 for detailed deployment profile comparison.

  

### 11. The "Freezing" Pattern for Competition

  

1. **Discovery phase:** Use `#eval` or `native_decide` in a scratch file to find the answer.

2. **Freezing phase:** Hardcode the discovered value into the submission. Instead of `example : ∃ n, P n := by native_decide`, write `example : P 12345 := by native_decide`. This dramatically reduces verification time.

3. **Optional TCB reduction:** If the frozen witness can be verified cheaply, use `norm_num` or `decide` for the final step, relegating `native_decide` to discovery only.

  

### 12. Advanced: What `native_decide` Can Verify

  

Beyond sequence equality, `native_decide` can verify:

  

- **Bounded universal quantification:** `∀ n < 1000, P(n)` — iterate and check.

- **Polynomial non-negativity** (bounded): `∀ x ∈ [0..1000], P(x) ≥ 0` — evaluate at each integer point.

- **Graph properties** for finite graphs: connectivity, coloring, path existence.

- **SAT instances:** Via `bv_decide` for `BitVec`/`Bool`, or by encoding directly and using `native_decide`.

  

### 13. Lean4Lean: The External Verifier

  

[Lean4Lean](https://github.com/digama0/lean4lean) is a complete type checker for Lean 4 written in Lean 4 itself, by Mario Carneiro (arXiv:2403.14064). It can verify all of Mathlib, running 20-50% slower than the C++ kernel. It has already caught kernel bugs. This project represents the future path toward reducing the TCB — eventually, a Lean-verified Lean compiler would remove the compiler trust assumption entirely.

  

---

  

## Part II: Python-to-Lean Formula Translation

  

### 14. The Translation Problem

  

Our invariant miner produces formulas in Python syntax:

- `n * (n + 1) // 2`

- `pow(2, n, mod) * factorial(n) % mod`

- `sum(comb(n, k) for k in range(n+1))`

- Recurrences: `f(n) = 2*f(n-1) + f(n-2)`

  

These must be translated to Lean 4 for verification via `native_decide`. The common assumption of "trivial syntax mapping" is **partially correct but has critical semantic pitfalls** in three areas: subtraction, division/modulo, and recursion.

  

### 15. What Is Trivially Correct

  

Addition (`+`), multiplication (`*`), exponentiation (`**` → `^`), comparison operators (`<`, `>`, `<=`, `>=`, `==` → `BEq`), boolean operators (`and` → `&&`, `or` → `||`, `not` → `!`), and conditionals (`x if c else y` → `if c then x else y`) all map directly with no semantic divergence.

  

Both Python and Lean support arbitrary-precision integers. `Array.range n` and `List.range n` both produce `[0, 1, ..., n-1]`, matching Python's `range(n)` semantically; generated checker code should prefer `Array.range` for performance and contract consistency.

  

### 16. The Three Semantic Pitfalls

  

#### 16.1 Subtraction: Nat Saturates at Zero

  

In Python, `3 - 5 = -2`. In Lean 4, `(3 : Nat) - 5 = 0` — Nat subtraction is **saturating** (also called monus): `a - b = max(0, a - b)`.

  

**Source:** [Lean Language Reference — Natural Numbers](https://lean-lang.org/doc/reference/latest/Basic-Types/Natural-Numbers/): "Subtraction of natural numbers, truncated at 0."

  

**Rule:** Any formula involving subtraction where the result could be negative must use `Int`, not `Nat`. This is the single most dangerous translation pitfall — it produces no error, just silently wrong results.

  

#### 16.2 Division: Lean's Default Matches Python (for Positive Divisors)

  

**Critical correction from prior documentation:** The original research claimed that Lean 4's `/` on `Int` performs truncated division (rounds toward zero). This is **wrong**.

  

**The actual behavior:** `/` on `Int` resolves to `Int.ediv` (Euclidean division), which for **positive divisors** is equivalent to floor division — the same as Python's `//`.

  

- `(-5 : Int) / 2 = -3` in Lean (Euclidean, positive divisor → floors)

- `(-5) // 2 = -3` in Python (floor division)

- **These match for positive divisors.**

  

For **negative divisors**, they diverge:

- `5 // (-2) = -3` in Python (floor)

- `Int.ediv 5 (-2) = -2` in Lean (Euclidean)

  

Lean also has `Int.tdiv` (truncated, rounds toward zero) and `Int.fdiv` (floor, rounds toward −∞). These are **not** the default `/` operator.

  

**Practical implication for our architecture:** Since competition math formulas almost never involve division by negative numbers, the default Lean `/` on `Int` matches Python's `//` with no helper function needed. The `py_div` helper proposed in earlier documentation is **unnecessary for our use case**.

  

**Sources:** [Lean Language Reference — Integers](https://lean-lang.org/doc/reference/latest/Basic-Types/Integers/): shows `(-12 : Int) / 7 = -2`, confirming Euclidean semantics.

  

#### 16.3 Modulo: Use `Int.emod` (It's Already the Default)

  

**Another correction:** The original research described a separate `Int.mod` function that takes the sign of the dividend. This does not exist under that name. The actual situation:

  

- `%` on `Int` resolves to **`Int.emod`** (Euclidean modulo): result is **always non-negative**.

  - `(-5 : Int) % 2 = 1` in Lean

  - `-5 % 2 = 1` in Python

  - **These match for positive divisors.**

- `Int.tmod` (truncated modulo): result takes sign of dividend.

  - `(-5 : Int).tmod 2 = -1`

  - This is C's `%` behavior, NOT Python's.

  

**Practical implication:** For positive moduli (which is every case in competition math), Lean's default `%` and Python's `%` produce identical results. No helper function needed.

  

### 17. Type Strategy

  

| Concept | Python | Lean 4 | Translation Rule |

|---|---|---|---|

| **General integer** | `int` | `Int` | Default to `Int` when subtraction is possible |

| **Non-negative integer** | `int` (positive) | `Nat` | Use `Nat` when values are provably ≥ 0 and no subtraction occurs |

| **Array index** | `int` | `Nat` | Cast with `.toNat` when needed |

| **Boolean** | `bool` | `Bool` | Use `Bool`, not `Prop`. `native_decide` requires `BEq` (`==`), not `Eq` (`=`) |

| **Float** | `float` | — | Reject. Use `Rat` if exact rationals needed |

| **Rational** | `fractions.Fraction` | `Rat` | `Rat` is in core Lean 4 (Init.Data.Rat), not Mathlib. However, `Rat.add` is `@[irreducible]`, so `decide` fails — use `native_decide` or `norm_num` |

  

**`Bool` vs `Prop` distinction:** Lean 4 separates computable booleans (`Bool`: `true | false`, in `Type`) from logical propositions (`Prop`: `Sort 0`). `native_decide` operates on `Bool` via `Decidable`. When comparing values, use `==` (`BEq` instance, returns `Bool`), not `=` (`Eq`, returns `Prop`). The `Decidable` typeclass bridges the gap: if `Decidable p` exists, `native_decide` can evaluate `p`.

  

### 18. What's in Core Lean 4 vs. Mathlib

  

For offline competition deployment without Mathlib, we need to know exactly what's available:

  

| Function | Core Lean 4 | Mathlib Only | Notes |

|---|---|---|---|

| `Nat.add`, `Nat.mul`, `Nat.sub`, `Nat.div`, `Nat.mod` | ✅ | | |

| `Nat.gcd` | ✅ | | In `Init.Data.Nat.Gcd` |

| `Nat.lcm` | ✅ | | In `Init.Data.Nat.Gcd` |

| `Nat.sqrt` | ✅ | | In `Init.Data.Nat.Sqrt` |

| `Nat.factorial` | | ✅ | **Not in core.** Must implement ourselves |

| `Nat.choose` (binomial coefficient) | | ✅ | **Not in core.** Must implement ourselves |

| `Int.ediv`, `Int.emod` | ✅ | | Default `/` and `%` on `Int` |

| `Int.tdiv`, `Int.tmod` | ✅ | | Truncated variants |

| `Rat` | ✅ | | In `Init.Data.Rat`. But `norm_num` for `Rat` requires Mathlib |

| `Array.range` | ✅ | | Preferred for generated checker templates |
| `List.range` | ✅ | | Semantically valid, but avoid in `native_decide` checker templates |

| `List.map`, `Array.map`, `foldl`, etc. | ✅ | | |

| `Finset.range`, `∑` (big operators) | | ✅ | Use `Array.foldl` instead |

| `norm_num` tactic | | ✅ | |

| `omega` tactic | ✅ | | In core since recent Lean versions |

| `DecidableEq` deriving | ✅ | | Works for structures and inductives |

  

**Implication:** We must include in our Lean prelude: `factorial`, `choose` (binomial coefficient), and `powMod` (modular exponentiation). Everything else is available in core Lean 4.

  

### 19. The Lean Prelude: Required Helper Functions

  

Since we avoid Mathlib, our verification template must include:

  

```lean

set_option maxRecDepth 10000

set_option maxHeartbeats 0

  

def factorial : Nat → Nat

  | 0 => 1

  | n + 1 => (n + 1) * factorial n

  

def choose (n k : Nat) : Nat :=

  if k > n then 0

  else factorial n / (factorial k * factorial (n - k))

  

def powMod (base exp mod : Nat) : Nat :=

  if mod == 0 then 0 else

  if mod == 1 then 0 else

  let rec loop (b e acc : Nat) : Nat :=

    if e == 0 then acc else

    let acc' := if e % 2 == 1 then (acc * b) % mod else acc

    loop ((b * b) % mod) (e / 2) acc'

  loop (base % mod) exp 1

```

  

The `powMod` function is critical: naive `base ^ exp % mod` computes `base ^ exp` first (potentially astronomical), then takes the modulus. For large exponents this crashes on RAM. Binary exponentiation with modular reduction at each step keeps intermediate values bounded.

  

### 20. Translation Templates

  

#### Template A: Closed-Form Formula

  

For formulas depending only on `n` with standard operations:

  

```lean

def f (n : Nat) : Nat := n * (n + 1) / 2

  

def expected : Array Nat := #[0, 1, 3, 6, 10, 15, 21, 28, 36, 45]

  

theorem verify : (Array.range 10).all (fun n => f n == expected[n]!) = true := by

  native_decide

```

  

#### Template B: Linear Recurrence (Array Accumulator)

  

For recurrences like `a(n) = c₁·a(n-1) + c₂·a(n-2)`:

  

```lean

def fib_arr (n : Nat) : Nat :=

  if n == 0 then 0

  else if n == 1 then 1

  else

let arr := (Array.range (n - 1)).foldl

      (init := #[(0 : Nat), 1])

      (fun memo _ =>

        let k := memo.size

        let v := memo[k - 1]! + memo[k - 2]!

        memo.push v)

    arr[n]!

```

  

This is O(n) time and O(n) space, using Array's in-place mutation optimization. The naive recursive definition (`fib (n+2) = fib (n+1) + fib n`) is O(2^n) and will time out for n > 30 under `native_decide`.

  

#### Template C: Fuel-Guarded Recursion

  

For Python `while` loops or recursions where termination is not structurally obvious:

  

```lean

def collatz (n : Nat) (fuel : Nat := 1000000) : Nat :=

  match fuel with

  | 0 => 0

  | fuel' + 1 =>

    if n <= 1 then 0

    else if n % 2 == 0 then 1 + collatz (n / 2) fuel'

    else 1 + collatz (3 * n + 1) fuel'

```

  

Fuel ensures structural termination on the `fuel` argument, satisfying Lean's totality checker without requiring a termination proof.

  

#### Template D: Modular Arithmetic

  

```lean

def f (n : Nat) : Nat := powMod 7 n 100

  

theorem verify : (Array.range 50).all (fun n => f n == expected[n]!) = true := by

  native_decide

```

  

### 21. Complex Translation Patterns

  

#### 21.1 Nested Summations

  

Python:

```python

sum(sum(f(i, j) for j in range(i)) for i in range(n))

```

  

Lean (computational, using Array):

```lean

(Array.range n).foldl (init := (0 : Int)) fun acc i =>

  acc + (Array.range i).foldl (init := (0 : Int)) fun inner j =>

    inner + f i j

```

  

#### 21.2 Binomial Sums with Alternating Signs

  

Python:

```python

sum((-1)**k * comb(n, k) for k in range(n+1))

```

  

Lean:

```lean

(Array.range (n + 1)).foldl (init := (0 : Int)) fun acc k =>

  acc + ((-1 : Int) ^ k) * (choose n k : Int)

```

  

Note: `(-1)^k` forces `Int` context. `choose n k` returns `Nat` and must be cast to `Int` explicitly.

  

#### 21.3 Python Chained Comparisons

  

Python `a < b < c` must be expanded to `a < b && b < c` in Lean.

  

#### 21.4 Python "Truthy" Values

  

Python allows `if 5: ...` (non-zero integers are truthy). Lean requires strict `Bool`: translate to `if 5 != 0 then ...`.

  

### 22. The Verification Harness

  

The complete verification template:

  

```lean

-- Prelude (injected into every verification file)

set_option maxRecDepth 10000

set_option maxHeartbeats 0

  

-- Helper functions

def powMod (base exp mod : Nat) : Nat :=

  if mod == 0 then 0 else

  if mod == 1 then 0 else

  let rec loop (b e acc : Nat) : Nat :=

    if e == 0 then acc else

    let acc' := if e % 2 == 1 then (acc * b) % mod else acc

    loop ((b * b) % mod) (e / 2) acc'

  loop (base % mod) exp 1

  

def factorial : Nat → Nat

  | 0 => 1

  | n + 1 => (n + 1) * factorial n

  

def choose (n k : Nat) : Nat :=

  if k > n then 0

  else factorial n / (factorial k * factorial (n - k))

  

-- [FORMULA SLOT: LLM or template fills this]

def f (n : Nat) : Nat := sorry

  

-- [EXPECTED VALUES: from Python trace]

def expected : Array Nat := #[sorry]

  

-- [VERIFICATION: checks formula matches trace for n=0..N]

theorem verify :

  (Array.range expected.size).all (fun n => f n == expected[n]!) = true := by

  native_decide

```

  

The Python orchestrator fills the `sorry` slots with the mined formula and the computed trace values. If `lean --run verify.lean` exits with code 0, the formula is verified.

  

### 23. Validation Strategy: Rosetta Stone Testing

  

Before trusting the translation layer in competition, validate it:

  

1. Generate random inputs (various `n` values including edge cases: 0, 1, large n).

2. Run the Python formula and the Lean formula (`#eval`) on the same inputs.

3. Assert outputs match.

  

Edge cases to test:

- `n = 0` (empty products, base cases)

- `n = 1` (boundary)

- Subtraction that would go negative in `Nat` context

- Division where both dividend and divisor have the same sign vs. different signs

- Modular arithmetic with large numbers (> 2^64)

- Empty sums: `sum([])` in Python = 0, `foldl (init := 0) ... #[]` in Lean = 0

  

### 24. Error Recovery: Compiler-in-the-Loop

  

If Lean rejects a translated formula:

  

1. **Type mismatch** (most common): Usually missing `Int` cast. Feed the Lean error back to the LLM with the prompt: "Fix this Lean 4 type error. Use Int types. Cast Nat to Int where needed."

2. **Termination failure**: The function isn't structurally recursive. Add fuel parameter.

3. **`native_decide` fails** (returns `false`): The formula is wrong for the checked range. This is correct behavior — the invariant miner should try a different candidate.

4. **`native_decide` hangs**: Non-terminating computation. The fuel parameter wasn't added, or the computation is genuinely too expensive. Kill the process and fall back to TIR.

  

---

  

## Part III: Extending to Geometry and Algebra — Continuous Domain Verification

  

The discrete pipeline (Parts I–II) covers combinatorics and number theory — roughly half of competition math. The other half is geometry and algebra, which involve continuous variables, irrational numbers, and polynomial constraints. This part extends the verification engine to these domains.

  

### 28. The Core Insight: Discovery vs. Verification Asymmetry

  

The same Oracle-Checker paradigm applies. Finding geometric coordinates or algebraic roots is hard (exponential search). Verifying that found coordinates satisfy constraints is easy (polynomial evaluation). We use untrusted Python solvers (SymPy, CVXPY, PSLQ) for discovery and trusted Lean verification for checking.

  

### 29. The `ring` Tactic — A Second Verification Primitive

  

For geometry and algebra, `ring` is as important as `native_decide`. It verifies polynomial identities over commutative (semi)rings.

  

**How `ring` works:** It normalizes both sides of an equality to a canonical sum-of-products form (based on Grégoire & Mahboubi's "Proving Equalities in a Commutative Ring Done Right in Coq"), then checks syntactic equality. The normalization produces a **full kernel-checkable proof term** — no TCB expansion, no `Lean.ofReduceBool`.

  

**What `ring` can prove:** Any identity that follows from commutative ring axioms: `(a + b)² = a² + 2ab + b²`, `x⁴ - 2x² + 1 = (x² - 1)²`, etc. It handles addition, subtraction, multiplication, and `Nat` exponents. It works over any type with a `CommSemiring` or `CommRing` instance, including `ℝ`, `ℚ`, `ℤ`, and custom algebraic number types.

  

**What `ring` cannot do:** Division, field inverses, or identities that require knowing that `(√2)² = 2` (it treats `√2` as an opaque atom). For division, use `field_simp` first to clear denominators, then `ring`. For algebraic relations, rewrite with the minimal polynomial first.

  

**`ring` is in Mathlib** (`Mathlib.Tactic.Ring`). However, Lean 4 core (since ~v4.22+) ships a separate ring solver embedded in the `grind` tactic, and a standalone `grobner` tactic that uses Gröbner bases for polynomial equation goals. The `grobner` tactic is available **without Mathlib**.

  

**Trust model:** `ring` and `grobner` produce full proof terms checked by the kernel (minimal runtime trust expansion). `native_decide` uses a larger runtime TCB. This trust distinction does not remove formalization-risk obligations.

  

**Source:** [Mathlib Ring/Basic docs](https://leanprover-community.github.io/mathlib4_docs/Mathlib/Tactic/Ring/Basic.html)

  

### 30. The Full Tactic Ecosystem for Continuous Domains

  

| Tactic | Location | What It Does | TCB Expansion | Use Case |

|---|---|---|---|---|

| **`ring`** | Mathlib | Proves polynomial equalities via normalization | None | `P(x) = Q(x)` identities |

| **`grobner`** | Core Lean 4 | Proves polynomial equations using Gröbner bases | None | Polynomial identities with hypotheses |

| **`field_simp`** | Mathlib | Clears denominators in field expressions | None | Prepare for `ring` when division present |

| **`linarith`** | Mathlib | Linear arithmetic over ordered fields (Fourier-Motzkin) | None | Linear inequalities over ℝ, ℚ |

| **`nlinarith`** | Mathlib | Nonlinear extension of `linarith` (adds `0 ≤ a²` etc.) | None | Nonlinear inequalities |

| **`positivity`** | Mathlib | Proves `0 ≤ x`, `0 < x`, `x ≠ 0` by structure analysis | None | Non-negativity goals |

| **`norm_num`** | Mathlib | Evaluates concrete numerical expressions | None | `37 is prime`, `2 + 3 = 5` |

| **`linear_combination`** | Mathlib | Certificate checker: verifies given coefficients via `ring` | None | Gröbner-style certificate checking |

| **`native_decide`** | Core | Compiles and executes `Decidable` instance | Compiler + GMP | Bounded computation |

  

**Key insight:** For geometry/algebra, the primary tactics (`ring`, `grobner`, `linarith`, `nlinarith`, `positivity`) all produce kernel-checkable proofs with **zero TCB expansion**. `native_decide` is the fallback for bounded computation in algebraic number fields.

  

**`polyrith` is defunct.** It relied on an external SageMath web server that has been shut down. The replacement is `grobner` (core Lean) or `linear_combination` with externally-computed coefficients.

  

### 31. Geometry Verification: Coordinate Constraint Satisfaction

  

The strategy: map geometric propositions to polynomial equations, then verify computationally.

  

#### 31.1 Constraint Translation Table

  

| Geometric Constraint | Polynomial Equation |

|---|---|

| Collinear(A, B, C) | `(xB - xA)*(yC - yA) - (yB - yA)*(xC - xA) = 0` |

| Parallel(AB, CD) | `(yB - yA)*(xD - xC) - (xB - xA)*(yD - yC) = 0` |

| Perpendicular(AB, CD) | `(xB - xA)*(xD - xC) + (yB - yA)*(yD - yC) = 0` |

| Distance(A, B) = k | `(xB - xA)² + (yB - yA)² - k² = 0` |

| On Circle(P, O, r) | `(xP - xO)² + (yP - yO)² - r² = 0` |

| Midpoint(M, A, B) | `xM = (xA + xB)/2 ∧ yM = (yA + yB)/2` |

| Angle(ABC) = 60° | `dot(BA, BC) = |BA|·|BC|/2` (via cos 60° = 1/2) |

  

#### 31.2 Rational Geometry (Level 1)

  

Problems involving only lines, intersections, and ratios can often be solved entirely in `ℚ`. If all coordinates are rational, verification is trivial:

  

```lean

def A : Rat × Rat := (0, 0)

def B : Rat × Rat := (2, 0)

def C : Rat × Rat := (1, 3/2)

  

theorem perpendicular_check :

  let (ax, ay) := A; let (bx, by_) := B; let (cx, cy) := C

  (bx - ax) * (cx - ax) + (by_ - ay) * (cy - ay) == 0 = false := by

  native_decide

```

  

`Rat` is in core Lean 4 (`Init.Data.Rat`). `Rat` has `DecidableEq`, so `native_decide` works. However, `Rat.add` is `@[irreducible]`, which means `decide` (kernel reduction) fails — `native_decide` is required for `Rat` computation.

  

#### 31.3 Algebraic Number Fields (Level 2) — The Critical Engineering Question

  

Most geometry problems introduce square roots (circles, distances, angles), requiring computation in fields like `ℚ(√2)`, `ℚ(√3)`, or `ℚ(√2, √3)`.

  

**Mathlib's `AdjoinRoot` cannot compute with `native_decide`.** `AdjoinRoot` is defined as `R[X] ⧸ Ideal.span {f}`, built on `Polynomial` which uses `Finsupp` — a largely **noncomputable** type. The `DecidableEq` instance exists logically but the underlying arithmetic operations are noncomputable. This was confirmed by Baanen, Chavarri Villarello & Dahmen in their CPP 2025 paper "Certifying Rings of Integers in Number Fields" — they had to build a separate `List R` representation for computation because Mathlib's `Polynomial` type cannot compute.

  

**The solution: custom lightweight algebraic number types with `deriving DecidableEq`.**

  

For `ℚ(√d)` (single quadratic extension):

```lean

structure QSqrt (d : Rat) where

  a : Rat  -- rational part

  b : Rat  -- coefficient of √d

  deriving DecidableEq, Repr

  

namespace QSqrt

def add (x y : QSqrt d) : QSqrt d := ⟨x.a + y.a, x.b + y.b⟩

def mul (x y : QSqrt d) : QSqrt d := ⟨x.a * y.a + x.b * y.b * d, x.a * y.b + x.b * y.a⟩

def neg (x : QSqrt d) : QSqrt d := ⟨-x.a, -x.b⟩

def sub (x y : QSqrt d) : QSqrt d := add x (neg y)

-- Instance declarations for CommRing, etc.

end QSqrt

```

  

For `ℚ(√2, √3)` (double quadratic extension, 4-dimensional over ℚ):

```lean

structure QSqrt2Sqrt3 where

  a : Rat  -- coefficient of 1

  b : Rat  -- coefficient of √2

  c : Rat  -- coefficient of √3

  d : Rat  -- coefficient of √6

  deriving DecidableEq, Repr

```

  

Multiplication uses the relations `(√2)² = 2`, `(√3)² = 3`, `(√6)² = 6`:

```

(a₁ + b₁√2 + c₁√3 + d₁√6)(a₂ + b₂√2 + c₂√3 + d₂√6)

```

Each product of basis elements reduces: `√2·√3 = √6`, `√2·√6 = 2√3`, `√3·√6 = 3√2`, `√6·√6 = 6`.

  

These custom types:

- Get automatic `DecidableEq` from Lean's deriving mechanism (since `Rat` has `DecidableEq`)

- Are fully computable — all field operations are defined as computable `def`s

- Work with `native_decide` for equality and comparison checks

- Are lightweight — no Mathlib dependency needed for the type itself

  

**This approach is novel.** No public work was found using `native_decide` on custom algebraic number types for geometry verification. The CPP 2025 paper by Baanen et al. used a similar strategy (computable list-based polynomials) for number field certification.

  

#### 31.4 Handling Trigonometric Angles

  

Competition geometry specifies angles (30°, 45°, 60°, 72°) that yield algebraic values:

- cos 60° = 1/2, sin 60° = √3/2 → lives in `ℚ(√3)`

- cos 45° = √2/2, sin 45° = √2/2 → lives in `ℚ(√2)`

- cos 72° involves √5 → lives in `ℚ(√5)`

  

Instead of using transcendental `Real.sin`/`Real.cos` (non-computable, blocks `native_decide`), translate angle constraints to algebraic ones via the dot product formula:

  

```

∠ABC = 60° → (BA⃗ · BC⃗) / (|BA⃗| · |BC⃗|) = 1/2

```

  

This becomes a polynomial constraint involving square roots (for magnitudes), which falls into the algebraic number field case above.

  

### 32. Algebra Verification: Certificate Checking

  

#### 32.1 Polynomial Substitution (`ring`)

  

For "find x such that f(x) = 0": if the answer is an algebraic expression, substitute and check.

  

```lean

-- Prove x² - 2 = 0 when x = √2

-- Using custom type QSqrt where √2 is represented as ⟨0, 1⟩

theorem root_check : 

  let x : QSqrt 2 := ⟨0, 1⟩  -- √2

  QSqrt.mul x x == ⟨2, 0⟩ = true := by

  native_decide

```

  

For the common case where the answer is an integer (AIMO answers are 0-999), bypass all field arithmetic:

  

```lean

-- If answer is integer k, just check P(k) = 0

theorem answer_check : (5 : Int)^3 - 125 == 0 = true := by native_decide

```

  

#### 32.2 Sum-of-Squares (SOS) Certificates for Inequalities

  

To prove `P(x) ≥ 0` for all `x`, find polynomials `q₁, ..., qₖ` such that `P = Σqᵢ²`. Squares are non-negative by axiom (`sq_nonneg` in Mathlib).

  

**Discovery (Python):** Use SDP solvers (CVXPY with SCS/MOSEK) to find the SOS decomposition.

  

**Verification (Lean):**

```lean

-- Prove x⁴ - 2x² + 1 ≥ 0

-- Python discovers: x⁴ - 2x² + 1 = (x² - 1)²

theorem ineq (x : ℝ) : x^4 - 2*x^2 + 1 ≥ 0 := by

  have h : x^4 - 2*x^2 + 1 = (x^2 - 1)^2 := by ring

  linarith [sq_nonneg (x^2 - 1)]

```

  

The pattern: `ring` verifies the polynomial identity (the certificate), `sq_nonneg` establishes non-negativity of each square, `linarith` combines them. All three produce kernel-checkable proofs — **zero TCB expansion**.

  

Mathlib does **not** have a general-purpose SOS certificate checker tactic. The verification must be assembled from `ring` + `sq_nonneg` + `linarith`/`nlinarith`. For more complex cases, `nlinarith` can handle products of hypothesis differences automatically.

  

**Constrained inequalities** use the Positivstellensatz: to prove `P(x) ≥ 0` assuming `g(x) ≥ 0`, find SOS polynomials `s₀, s₁` such that `P = s₀ + s₁·g`. Verify `P - s₀ - s₁·g = 0` via `ring`.

  

#### 32.3 Gröbner Basis Certificates for Universal Geometry

  

For "prove for all triangles" problems (universal quantification over coordinates), we use the Ideal Membership method:

  

**Discovery (Python):** SymPy/SageMath computes cofactors `c₁, ..., cₖ` such that the conclusion `g = Σcᵢ·fᵢ` where `fᵢ` are hypothesis polynomials.

  

**Verification (Lean):** The `linear_combination` tactic checks the certificate:

```lean

-- Hypotheses: h1 : f₁ = 0, h2 : f₂ = 0

-- Certificate: g = c₁·f₁ + c₂·f₂

theorem conclusion : g = 0 := by linear_combination c₁ * h1 + c₂ * h2

```

  

`linear_combination` internally uses `ring` to verify the resulting polynomial identity. The `grobner` tactic (core Lean 4) can also close such goals directly, but requires the hypotheses to be in the goal context.

  

#### 32.4 Root Isolation (Sturm's Theorem)

  

When a polynomial has multiple roots, we need to specify which root is the answer. The standard method: represent the algebraic number as `(P, [a, b])` where `P` is the minimal polynomial and `[a, b]` is an isolating rational interval.

  

**Sturm's theorem** counts distinct real roots in an interval `(a, b]` using sign variations of the Sturm chain. The computation involves only polynomial division and evaluation at rational points — fully decidable.

  

**Mathlib does NOT implement Sturm's theorem.** This would need to be implemented as a custom verified function in our prelude. The algorithm is straightforward (Euclidean algorithm on polynomials, sign variation counting) and works well with `native_decide` on a computable polynomial representation (e.g., `Array Rat` for coefficients).

  

### 33. The Verification Hierarchy

  

Not all problems need heavy machinery. The architecture dispatches to the simplest sufficient level:

  

| Level | Domain | Arithmetic | Primary Tactic | TCB | Mathlib? |

|---|---|---|---|---|---|

| **0** | Combinatorics, Number Theory | `Nat`, `Int`, `ZMod` | `native_decide` | Compiler | No |

| **1** | Rational geometry, linear algebra | `Rat` | `native_decide` | Compiler | No |

| **2** | Euclidean geometry, radicals | Custom `QSqrt d`, `QSqrt2Sqrt3` | `native_decide` | Compiler | No (custom types) |

| **3** | Polynomial identities, certificates | `ℝ`, `ℚ`, variables | `ring`, `grobner`, `linear_combination` | None | Yes (Mathlib) |

| **4** | Inequalities, optimization | `ℝ` with ordering | `nlinarith`, `positivity`, `linarith` | None | Yes (Mathlib) |

| **5** | Transcendental / approximation | Verified intervals | Interval arithmetic | Depends | Custom |

  

**Levels 0–2** use `native_decide` and require no Mathlib — the discrete pipeline plus custom algebraic number types. Deployment is ~500-875 MB (core Lean toolchain).

  

**Levels 3–4** use `ring`/`linarith`/`nlinarith` and require Mathlib. These produce kernel-checkable proof terms with minimal runtime trust expansion relative to `native_decide`. Deployment requires Mathlib `.olean` cache (~5 GB total with toolchain).

  

**Level 5** is a fallback for rare problems with no algebraic solution form.

  

### 34. The Mathlib Dependency Split

  

The architecture has two deployment profiles:

  

**Profile A: Discrete-only (no Mathlib)**

- Covers: Combinatorics, Number Theory, rational/algebraic geometry via custom types

- Tactics: `native_decide`, `omega`, `grobner` (core)

- Size: ~500-875 MB (Lean toolchain)

- Suitable for Kaggle with tight disk constraints

  

**Profile B: Full pipeline (with Mathlib)**

- Covers: All of Profile A plus polynomial identity certificates, inequality verification, universal geometry theorems

- Tactics: All of Profile A plus `ring`, `field_simp`, `linarith`, `nlinarith`, `positivity`, `linear_combination`, `norm_num`

- Size: ~5-6 GB total (toolchain + Mathlib `.olean` cache)

- Pre-package entire `.lake/` directory and `.elan/toolchains/<version>` for offline use

- `lake exe cache get` takes ~1-5 minutes to decompress pre-cached oleans

  

**The strategic choice:** Profile A alone covers ~70% of competition problems. Profile B adds the remaining ~30% (inequalities, universal geometry) with stronger guarantees. On Kaggle (20 GB disk), both profiles fit with room to spare.

  

**Offline Mathlib deployment:**

1. On internet-connected machine: `lake exe cache get` to download all oleans

2. Bundle `.lake/` and `.elan/toolchains/<version>/` as Kaggle datasets

3. On Kaggle: set `PATH` to include Lean binaries, point Lake to pre-built packages

4. No recompilation needed — pre-compiled `.olean` files load directly

  

### 35. Geometry/Algebra Verification Templates

  

#### Template E: Rational Coordinate Geometry

  

```lean

-- All coordinates rational — direct native_decide

def dist_sq (p q : Rat × Rat) : Rat :=

  (q.1 - p.1)^2 + (q.2 - p.2)^2

  

def A : Rat × Rat := (0, 0)

def B : Rat × Rat := (3, 0)

def C : Rat × Rat := (0, 4)

  

theorem right_triangle : dist_sq A B + dist_sq A C == dist_sq B C = true := by

  native_decide

```

  

#### Template F: Algebraic Number Geometry (Custom Type)

  

```lean

-- Geometry in Q(√3)

structure QSqrt3 where

  a : Rat; b : Rat  -- represents a + b√3

  deriving DecidableEq, Repr

  

instance : BEq QSqrt3 where beq x y := x.a == y.a && x.b == y.b

  

def QSqrt3.mul (x y : QSqrt3) : QSqrt3 :=

  ⟨x.a * y.a + x.b * y.b * 3, x.a * y.b + x.b * y.a⟩

  

def QSqrt3.add (x y : QSqrt3) : QSqrt3 := ⟨x.a + y.a, x.b + y.b⟩

def QSqrt3.sub (x y : QSqrt3) : QSqrt3 := ⟨x.a - y.a, x.b - y.b⟩

  

def dist_sq_alg (p q : QSqrt3 × QSqrt3) : QSqrt3 :=

  QSqrt3.add (QSqrt3.mul (QSqrt3.sub q.1 p.1) (QSqrt3.sub q.1 p.1))

             (QSqrt3.mul (QSqrt3.sub q.2 p.2) (QSqrt3.sub q.2 p.2))

  

-- Equilateral triangle: A=(0,0), B=(2,0), C=(1, √3)

def A : QSqrt3 × QSqrt3 := (⟨0, 0⟩, ⟨0, 0⟩)

def B : QSqrt3 × QSqrt3 := (⟨2, 0⟩, ⟨0, 0⟩)

def C : QSqrt3 × QSqrt3 := (⟨1, 0⟩, ⟨0, 1⟩)  -- (1, √3)

  

theorem equilateral :

  dist_sq_alg A B == ⟨4, 0⟩ &&

  dist_sq_alg A C == ⟨4, 0⟩ &&

  dist_sq_alg B C == ⟨4, 0⟩ = true := by

  native_decide

```

  

#### Template G: SOS Inequality Certificate (Mathlib required)

  

```lean

import Mathlib.Tactic

  

-- Prove AM-GM variant: a² + b² ≥ 2ab

theorem am_gm_sq (a b : ℝ) : a^2 + b^2 ≥ 2 * a * b := by

  nlinarith [sq_nonneg (a - b)]

  

-- More complex: x⁴ + y⁴ ≥ x²y²  (Python finds decomposition)

theorem quartic_ineq (x y : ℝ) : x^4 + y^4 ≥ x^2 * y^2 := by

  nlinarith [sq_nonneg (x^2 - y^2), sq_nonneg (x*y)]

```

  

#### Template H: Gröbner Certificate (Mathlib required)

  

```lean

import Mathlib.Tactic

  

-- Given: a + b = 5, a * b = 6. Prove: a² + b² = 13

theorem sum_sq (a b : ℝ) (h1 : a + b = 5) (h2 : a * b = 6) :

    a^2 + b^2 = 13 := by

  -- Certificate: a² + b² = (a+b)² - 2ab = 25 - 12 = 13

  nlinarith [sq_nonneg a, sq_nonneg b]

  -- Alternative: linear_combination (a + b) * h1 - 2 * h2

```

  

### 36. The Integer Answer Exploit

  

AIMO answers are integers 0-999. Even when intermediate computation involves complex algebraic numbers, the final answer is an integer. This massively simplifies verification:

  

1. Compute the algebraic expression for the answer (in Python, using SymPy with exact arithmetic)

2. If it simplifies to integer `k`, just verify `f(k) = 0` directly via `native_decide`

3. Bypass all algebraic field arithmetic complexity

  

For example, if the area of a triangle involves `√3` intermediately but the answer is `12`:

```lean

-- Skip the algebraic number field machinery entirely

theorem answer_is_12 : computed_area == 12 = true := by native_decide

```

  

This covers the majority of competition problems and eliminates the need for custom algebraic number types in many cases.

  

### 37. What We Trust — Extended for Geometry/Algebra

  

| Component | Trusted? | Why |

|---|---|---|

| LLM constraint generation | **No** | LLM may produce wrong constraints. If wrong, Lean rejects. |

| SymPy coordinate solving | **No** | May find wrong coordinates. If wrong, Lean rejects. |

| SDP solver (SOS decomposition) | **No** | May find approximate/wrong decomposition. `ring` rejects if identity doesn't hold exactly. |

| PSLQ reconstruction | **No** | May guess wrong algebraic form. Lean rejects. |

| `ring` / `grobner` / `linarith` | **Yes** (kernel-verified) | Produce full proof terms. **Zero TCB expansion.** |

| `native_decide` on custom types | **Yes** (modulo TCB) | Trusts Lean compiler + GMP. Same as discrete pipeline. |

  

The geometry/algebra pipeline preserves a key verifier property: untrusted generators (LLM/SDP/PSLQ) cannot force acceptance when certificates/residual checks fail. However, formalization mismatch can still yield internally consistent wrong-world certificates unless mitigated by dual formalization, deterministic audits, and consequence checks. For the `ring`-based verification path (Levels 3-4), proof terms are kernel-checked with minimal runtime trust expansion.

  

---

  

## Part IV: Putting It Together — The Unified Verification Protocol

  

### 38. End-to-End Flow (Discrete Pipeline)

  

```

Python trace [1, 3, 6, 10, 15, ...]

        │

        ▼

Berlekamp-Massey / Lagrange finds: f(n) = n*(n+1)/2

        │

        ▼

Template fills Lean file:

  def f (n : Nat) : Nat := n * (n + 1) / 2

  def expected : Array Nat := #[0, 1, 3, 6, 10, 15, 21, 28, 36, 45, ...]

  theorem verify : ... := by native_decide

        │

        ▼

subprocess.run(["lean", "--run", "verify.lean"])

        │

        ├── exit code 0 → VERIFIED. Submit answer.

        └── exit code ≠ 0 → REJECTED. Try next candidate or fall back to TIR.

```

  

### 39. End-to-End Flow (Geometry/Algebra Pipeline)

  

```

Problem: "Find the area of triangle ABC where AB=2, ∠A=60°, AC=3"

        │

        ▼

LLM generates Python: SymPy solves for coordinates

  → A=(0,0), B=(2,0), C=(3/2, 3√3/2)

  → Area = 3√3/2 → Integer answer = 3√3/2... not integer

  → But AIMO answer is integer → area mod 1000 = ...

        │

        ▼

Certificate generated:

  coordinates + constraint polynomial evaluations

        │

        ▼

Template fills Lean file (using QSqrt3 custom type):

  theorem verify : area == ⟨0, 3/2⟩ = true := by native_decide

        │

        ▼

OR (if Mathlib available, for universal theorem):

  theorem verify : ... := by nlinarith [sq_nonneg ...]

```

  

### 40. What We Trust and What We Don't (Unified)

  

| Component | Trusted? | Why |

|---|---|---|

| LLM (trace gen, constraint gen) | **No** | May be wrong. Lean rejects wrong output. |

| Python (trace execution, SymPy) | **No** | May have silent bugs. Lean rejects wrong output. |

| Berlekamp-Massey / Lagrange | **Yes** (deterministic) | Exact algorithms. Given correct input → correct output. |

| SDP solver / PSLQ | **No** | May find approximate/wrong result. Lean rejects. |

| Formula/constraint translation | **No** | May be wrong. Lean rejects. |

| `native_decide` | **Yes** (modulo TCB) | Trusts Lean compiler + GMP. Acceptable for competition. |

| `ring` / `grobner` / `linarith` | **Yes** (kernel-verified) | Full proof terms. **Zero TCB expansion.** |

  

**Untrusted generators cannot force acceptance when checks fail.** Acceptance still depends on correct formalization and constraint extraction. The `ring`-based path uses kernel-checked proof terms with lower runtime trust expansion than `native_decide`.

  

### 41. Performance Budget

  

For AIMO 3 (5 hours, 50 problems, 6 minutes per problem):

  

| Stage | Time Budget | Notes |

|---|---|---|

| LLM trace/constraint generation | ~30s | |

| Python execution (trace, SymPy, SDP) | ~5-15s | SDP solving may take longer |

| Berlekamp-Massey / Lagrange | <1s | |

| Formula/certificate translation | <1s | |

| Lean verification (Level 0-2: `native_decide`) | 1-5s | Milliseconds for simple formulas |

| Lean verification (Level 3-4: `ring`/`nlinarith`) | 2-15s | Polynomial identity checking |

| Retry loop (if verification fails) | ~2 min | |

| Fallback to TIR | remaining | |

  

Lean verification is fast at all levels. `native_decide` uses GMP arithmetic and runs tight loops. `ring` normalizes polynomial expressions efficiently. `nlinarith` with SOS hints terminates quickly for competition-sized problems.

  

---

  

## Appendix A: Safe Usage Checklist

  

- [ ] **Tail recursion:** All recursive functions use accumulator passing style (direct self-recursion only).

- [ ] **Data structures:** `Array`, never `List`, for any collection used in `native_decide`.

- [ ] **Termination:** All functions are total. Use `fuel` parameter if termination proof is non-trivial.

- [ ] **Purity:** No `unsafe`, `partial`, `@[extern]`, `@[implemented_by]`, or `IO` in verification code.

- [ ] **Sanitizer gate:** Enforce `S1.5` policy (strip banned attributes, enforce import allowlist, reject non-template checker code).

- [ ] **Types:** Default to `Int` when subtraction is possible. Use `Nat` only for provably non-negative values.

- [ ] **`powMod`:** Always use modular exponentiation helper. Never `base ^ exp % mod` for large exponents.

- [ ] **Test with `#eval` first:** Before wrapping in `native_decide`, run `#eval f 10` to catch runtime errors.

- [ ] **Axiom audit:** `#print axioms verify` should show only `Lean.ofReduceBool` and `Lean.trustCompiler`.

  

## Appendix B: Complete Syntax Mapping Table

  

| Python | Lean 4 | Notes |

|---|---|---|

| `+` | `+` | |

| `-` | `-` | **Use `Int` context if result can be negative** |

| `*` | `*` | |

| `**` | `^` | Lean `^` requires `Nat` exponent |

| `//` | `/` | On `Int`, this is `Int.ediv` — matches Python for positive divisors |

| `%` | `%` | On `Int`, this is `Int.emod` — matches Python for positive divisors |

| `/` (float) | — | Reject or use `Rat` |

| `pow(a, b, m)` | `powMod a b m` | Must use custom helper (not in core) |

| `factorial(n)` | `factorial n` | Must use custom helper (not in core) |

| `comb(n, k)` | `choose n k` | Must use custom helper (not in core) |

| `gcd(a, b)` | `Nat.gcd a b` | In core |

| `abs(x)` | `Int.natAbs x` | Returns `Nat` |

| `math.isqrt(n)` | `Nat.sqrt n` | In core |

| `==` | `==` | Uses `BEq` instance, returns `Bool` |

| `!=` | `!=` | |

| `<`, `>`, `<=`, `>=` | `<`, `>`, `<=`, `>=` | |

| `and` | `&&` | Python `and` returns operand value; Lean `&&` returns `Bool` |

| `or` | `\|\|` | Same caveat |

| `not` | `!` | |

| `x if c else y` | `if c then x else y` | `c` must be `Decidable` |

| `a < b < c` | `a < b && b < c` | Must expand chained comparisons |

| `range(n)` | `Array.range n` | Both produce `[0, ..., n-1]` |

| `sum(L)` | `L.foldl (· + ·) 0` | |

| `[f(x) for x in L]` | `L.map (fun x => f x)` | |

| `[x for x in L if p(x)]` | `L.filter (fun x => p x)` | |

| `L.append(x)` | `L.push x` | `Array` only |

| `L[i] = v` | `L.set! i v` | Functional update (in-place if RC=1) |

| `L[i]` | `L[i]!` or `L.get! i` | |

| `len(L)` | `L.size` | |

| `True` / `False` | `true` / `false` | Lowercase in Lean |

| `def f(x): return ...` | `def f (x : Int) : Int := ...` | Must annotate types |

| `lambda x: ...` | `fun x => ...` | |

| `for x in range(n): ...` | `(Array.range n).foldl ...` or recursion | No imperative loops in Lean |

  

## Appendix C: References

  

1. de Moura, L. & Ullrich, S. "The Lean 4 Theorem Prover and Programming Language." [lean-lang.org/papers/lean4.pdf](https://lean-lang.org/papers/lean4.pdf)

2. Ullrich, S. & de Moura, L. "Counting Immutable Beans: Reference Counting Optimized for Purely Functional Programming." arXiv:1908.05647

3. Carneiro, M. "Lean4Lean: Verifying a Typechecker for Lean, in Lean." arXiv:2403.14064

4. Lean Language Reference. [lean-lang.org/doc/reference/latest/](https://lean-lang.org/doc/reference/latest/)

5. Functional Programming in Lean. [leanprover.github.io/functional_programming_in_lean/](https://leanprover.github.io/functional_programming_in_lean/)

6. Lean Zulip — `native_decide` soundness bug discussion. [leanprover-community.github.io/archive/stream/270676-lean4/topic/soundness.20bug.3A.20native_decide.20leakage.html](https://leanprover-community.github.io/archive/stream/270676-lean4/topic/soundness.20bug.3A.20native_decide.20leakage.html)

7. Huisinga, S. "Static Uniqueness Analysis for the Lean 4 Compiler." Master thesis, KIT, 2023.

8. Lean4 PR #2654 — Added `Lean.trustCompiler` axiom. [github.com/leanprover/lean4/pull/2654](https://github.com/leanprover/lean4/pull/2654)

9. Lean4 issue #8840 / PR #8842 — Fixed `collectAxioms` transitivity. [github.com/leanprover/lean4/issues/8840](https://github.com/leanprover/lean4/issues/8840)

10. Lean4 issue #7463 — `@[csimp]` axiom leakage via `native_decide`. [github.com/leanprover/lean4/issues/7463](https://github.com/leanprover/lean4/issues/7463)

11. Baanen, A., Chavarri Villarello, X. & Dahmen, S. "Certifying Rings of Integers in Number Fields." CPP 2025. [arxiv.org/abs/2409.18030](https://arxiv.org/abs/2409.18030)

12. Grégoire, B. & Mahboubi, A. "Proving Equalities in a Commutative Ring Done Right in Coq." TPHOLs 2005.

13. Mathlib Ring tactic documentation. [leanprover-community.github.io/mathlib4_docs/Mathlib/Tactic/Ring/Basic.html](https://leanprover-community.github.io/mathlib4_docs/Mathlib/Tactic/Ring/Basic.html)

14. Mathlib computation models for polynomials (wiki). [github.com/leanprover-community/mathlib4/wiki/Computation-models-for-polynomials-and-finitely-supported-functions](https://github.com/leanprover-community/mathlib4/wiki/Computation-models-for-polynomials-and-finitely-supported-functions)

15. Lean `grind` tactic — Algebraic Solver. [lean-lang.org/doc/reference/latest/The--grind--tactic/Algebraic-Solver-_LPAR_Commutative-Rings___-Fields_RPAR_/](https://lean-lang.org/doc/reference/latest/The--grind--tactic/Algebraic-Solver-_LPAR_Commutative-Rings___-Fields_RPAR_/)

**
