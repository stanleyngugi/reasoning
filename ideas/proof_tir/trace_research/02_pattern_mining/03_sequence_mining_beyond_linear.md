# Deep Research: Sequence Mining Beyond Linear Recurrences

## Research Objective

Berlekamp-Massey handles linear recurrences perfectly, but many competition math sequences are NOT linearly recurrent. We need a complete taxonomy of sequence types, detection methods, and mining algorithms for each — forming Tiers 2-4 of our pattern mining pipeline.

## Context

Our tiered mining strategy:
- **Tier 1**: Berlekamp-Massey (linear recurrences) — deterministic, 100% accurate
- **Tier 2**: Polynomial interpolation (Lagrange) — for n², n³, binomial-coefficient sequences
- **Tier 3**: LLM pattern recognition — for named sequences (Catalan, Fibonacci, etc.)
- **Tier 4**: OEIS lookup — for unknown sequences

We need deep understanding of each tier: when to use it, how it works, failure modes, and implementation.

## Research Questions

### Part A: Taxonomy of Sequences in Competition Math

#### 1. Sequence Type Classification
- Create a comprehensive taxonomy of sequence types appearing in olympiad math:
  - C-finite (linear recurrences) — e.g., Fibonacci, Lucas, tiling counts
  - P-recursive / D-finite / Holonomic — e.g., factorials, Catalan, binomial sums
  - Polynomial sequences — e.g., triangular numbers, sums of powers
  - Exponential sequences — e.g., 2^n, powers
  - Modular/periodic sequences — e.g., last digit cycles
  - Hybrid sequences — combinations of above
  - Irregular sequences — no known pattern
- For each type, give 3-5 examples from competition math

#### 2. Detection Heuristics
- Given a sequence [a₁, a₂, ..., aₖ], how can we quickly determine its likely type?
- What patterns suggest linear recurrence vs polynomial vs exponential?
- Are there statistical tests (ratio test, difference test) that help classify?
- How many terms do we need for reliable classification?

### Part B: Tier 2 — Polynomial Sequences and Lagrange Interpolation

#### 3. Polynomial Sequence Theory
- What sequences in competition math are exactly polynomial? (f(n) is a polynomial in n)
- Common examples: triangular (n²), tetrahedral (n³), sum of cubes, binomial coefficients
- How do we detect that a sequence is polynomial? (Finite differences method)
- What's the expected degree for competition problems? (Usually ≤ 5?)

#### 4. Lagrange Interpolation Mechanics
- Given points (1, a₁), (2, a₂), ..., (k, aₖ), construct the unique polynomial of degree < k
- What's the numerical stability situation? When does floating-point fail?
- How do we use exact rational arithmetic to avoid precision issues?
- What's the complexity? O(n²) for evaluation, O(n² log² n) with FFT?

#### 5. Numerical Stability Solutions
- When is `numpy.polyfit` insufficient?
- How does `mpmath` help? What precision is needed for degree-10 polynomials?
- Should we use SymPy's symbolic interpolation instead?
- What about modular interpolation (work mod p, then CRT)?

#### 6. From Polynomial to Lean
- Once we have polynomial coefficients, how do we translate to Lean?
- Example: f(n) = (n³ + 3n² + 2n) / 6 — integer division issues?
- How do we handle rational coefficients that always yield integers?
- Verification strategy: check f(n) for n=1..100 via native_decide

### Part C: Tier 3 — Named Sequences and LLM Pattern Recognition

#### 7. Common Named Sequences in Competition Math
- Create a reference of named sequences that appear frequently:
  - Catalan numbers (parenthesizations, binary trees, Dyck paths)
  - Bell numbers (set partitions)
  - Stirling numbers (first and second kind)
  - Euler numbers
  - Motzkin numbers
  - Delannoy numbers
  - Partition numbers
- For each: formula, recurrence (if exists), first 10 terms, common problem contexts

#### 8. LLM Sequence Recognition
- How good are current LLMs at recognizing named sequences from their first 10-15 terms?
- What prompting strategies maximize recognition accuracy?
- Should we show the sequence, the problem context, or both?
- How do we handle sequences that are "shifts" or "multiples" of named sequences?

#### 9. Generating Functions
- Many named sequences have known generating functions
- Can we use generating function techniques for pattern mining?
- Connection to holonomic sequences and recurrence extraction

### Part D: Tier 4 — OEIS Integration

#### 10. OEIS as a Pattern Oracle
- How do we query OEIS effectively? (API, scraping, local database)
- What's the format of OEIS entries? (A-number, formula, recurrence, references)
- How many sequences are in OEIS? (~360,000) — what percentage of competition sequences are covered?
- How do we handle sequences that are "almost" in OEIS (shifted, scaled, interleaved)?

#### 11. Offline OEIS
- For competition use (no internet), how do we package OEIS data?
- What's the size of OEIS in raw form? Can we compress it?
- Should we embed a subset (most common 10,000 sequences)?
- Can we use LLM memory as a compressed OEIS proxy?

### Part E: Holonomic / P-Recursive Sequences

#### 12. Holonomic Sequence Theory
- Define P-recursive / D-finite / holonomic sequences
- These satisfy recurrences where coefficients are polynomials in n:
  - a(n) = p₁(n)*a(n-1) + p₂(n)*a(n-2) + ...
- Examples: factorials (a(n) = n*a(n-1)), Catalan, central binomial
- Why can't Berlekamp-Massey find these?

#### 13. Holonomic Sequence Recognition
- Given a sequence, how do we determine if it's holonomic?
- What algorithms find the minimal holonomic recurrence?
- Are there Python libraries for this? (SymPy? ore_algebra? other?)
- What's the complexity? How many terms needed?

#### 14. Guessing Holonomic Recurrences
- The "guess" package in Mathematica
- Kauers' "Guess" algorithm
- Any Python equivalents?
- Practical success rate on competition sequences?

### Part F: Modular and Periodic Patterns

#### 15. Modular Arithmetic Cycles
- Many competition problems ask for f(10^18) mod p
- Sequences often become periodic modulo p (Pisano period for Fibonacci)
- How do we detect and extract the period?
- What's the theory behind modular periodicity of linear recurrences?

#### 16. Chinese Remainder Theorem Reconstruction
- If we know f(n) mod p₁, f(n) mod p₂, etc., we can reconstruct f(n) mod (p₁*p₂*...)
- When is this useful for sequence mining?
- Connection to modular Berlekamp-Massey

### Part G: Integration and Decision Logic

#### 17. Tier Selection Heuristics
- Given a sequence, how do we decide which tier to try first?
- Can we run multiple tiers in parallel and take the first success?
- What's the time budget for each tier in a 6-minute-per-problem competition?

#### 18. Verification Before Acceptance
- Before passing a candidate formula to Lean, we should check it against the trace
- What's the tolerance? (Must match exactly for all trace terms)
- How do we handle floating-point formulas that need conversion to exact form?

#### 19. Fallback Strategies
- If all tiers fail, what do we do?
- Return to LLM for a completely different approach?
- Try brute-force closed-form search?
- Give up on formal verification and fall back to TIR?

## Desired Output Format

1. **Sequence Taxonomy**: Comprehensive classification with examples
2. **Tier-by-Tier Implementation Guide**: Algorithms, libraries, code patterns
3. **Detection Flowchart**: Decision tree for sequence classification
4. **Benchmark Data**: Success rates of each method on olympiad sequences
5. **Failure Case Analysis**: What each tier can't handle
6. **OEIS Integration Guide**: Practical usage patterns
7. **References**: Papers on sequence recognition, holonomic functions, OEIS










# Computation as Verification: An Exhaustive Analysis of `native_decide` in Lean 4

## 1. Introduction: The Convergence of Execution and Deduction

The landscape of formal verification has traditionally been characterized by a sharp dichotomy between logical deduction and computational execution. On one side, interactive theorem provers (ITPs) like Coq, Isabelle/HOL, and early versions of Lean prioritized a small, trusted kernel that verified proofs via symbolic reduction and structural checking. This approach, while epistemologically robust, imposed severe performance penalties on proofs that required heavy computation—such as checking the primality of a large integer or verifying a certificate from an external solver. On the other side, general-purpose programming languages offered high-performance execution but lacked the semantic assurances of formal logic.

Lean 4 represents a paradigm shift in this domain, engineered from the ground up to serve simultaneously as a high-performance functional programming language and an interactive theorem prover. At the nexus of this convergence lies the `native_decide` tactic. This mechanism allows the user to bridge the gap between the logical model and the physical machine, effectively treating the successful execution of a compiled program as a formal proof of its result. By invoking `native_decide`, a user instructs the system to bypass the slow, symbolic reduction of the logical kernel in favor of the optimized, binary execution of the Lean runtime.

In the context of high-stakes competition environments, such as the AI Math Olympiad (AIMO) or Kaggle-hosted automated reasoning challenges, `native_decide` offers a compelling strategic advantage. It provides a "nuclear option" for discharging computational goals that are computationally intractable for the standard kernel yet trivial for a modern CPU. However, this power is not without cost. The utilization of `native_decide` fundamentally alters the trust model of the verification environment, expanding the Trusted Code Base (TCB) from a few thousand lines of C++ kernel code to tens of thousands of lines including the compiler, the interpreter, the runtime system, and external libraries like GMP.

This report provides an exhaustive technical analysis of `native_decide`, specifically tailored for the exigencies of competition environments. We will dissect the internal architecture of the tactic, trace the compilation pipeline from elaboration to execution, and quantify the risks associated with TCB expansion. Furthermore, we will present a rigorous set of performance benchmarks and safe usage patterns, enabling competitors to leverage this powerful tool without compromising the integrity of their formal developments.

### 1.1 The Theoretical Basis: Proof by Reflection

To understand `native_decide`, one must situate it within the broader history of "proof by reflection" in type theory. Classically, to prove a proposition $P$, one constructs a proof term $t : P$. For computational propositions, such as $2 + 2 = 4$, the proof term is typically just `Refl` (reflexivity), and the kernel verifies this by reducing both sides of the equation to their normal forms. This is "proof by reduction."

Proof by reflection takes a different approach. It involves defining a decision procedure—a computable function $f : \text{Input} \to \text{Bool}$—and proving a soundness theorem: $\forall x, f(x) = \text{true} \to P(x)$. To prove $P(a)$, one simply computes $f(a)$. If the computation yields `true`, the soundness theorem justifies the conclusion $P(a)$.

Standard "small-scale" reflection (using the `decide` tactic) performs this computation within the kernel. The kernel symbolically evaluates $f(a)$. While secure, this is slow because the kernel treats numbers and data structures as inductive types (linked lists of constructors) rather than machine primitives. `native_decide` represents "large-scale" reflection. It relies on the same soundness principle but delegates the computation of $f(a)$ to the native hardware via the compiler. This distinction is critical: `decide` trusts only the logical rules; `native_decide` trusts that the compiler correctly implements those rules in machine code.

### 1.2 The Competition Context

In competition environments like the AIMO, participants are tasked with solving mathematical problems using Lean. The constraints are unique:

1. **Time Limits:** Solutions must be checked within a strict time window (often effectively bounding the proof checking time).
    
2. **Resource Limits:** Memory and CPU are finite, often running in Docker containers with specific configurations.
    
3. **Correctness:** Solutions are binary—either the proof checks, or it does not.
    
4. **Black-Box Grading:** Often, the specific test cases or the evaluation environment are opaque to the participant during the development phase.
    

In this context, `native_decide` becomes a strategic asset. It allows for the verification of brute-force solutions (e.g., "find a counterexample by checking the first 10,000 cases") that would simply time out using standard tactics. However, misuse of `native_decide` can lead to "works on my machine" syndromes where local compilation succeeds but the remote evaluation fails due to environment mismatches, timeout handling differences, or subtle soundness bugs in edge cases.

## 2. Architectural Internals: From Logic to Machine Code

The operation of `native_decide` is a multi-stage pipeline that transforms a dependent type theory expression into an executable binary. Understanding this pipeline is essential for diagnosing failures and optimizing performance.

### 2.1 The Elaboration Phase

When the user invokes `native_decide` on a goal $\vdash P$, the first step is **Elaboration**. The tactic inspects the goal $P$ and attempts to synthesize an instance of the type class `Decidable P`.

- **Mechanism:** The type class inference mechanism searches for a function that can compute the truth value of $P$. This relies on recursive instance search. For example, if $P$ is `10 < 20`, Lean looks for `Decidable (Nat.lt 10 20)`.
    
- **Result:** If successful, elaboration produces an expression `d : Decidable P`. This expression contains the algorithm for verifying $P$.
    
- **Competition Relevance:** This phase runs in the _elaborator_, meaning it consumes `maxHeartbeats` but not necessarily `maxRecDepth`. If the instance synthesis is too complex (e.g., deciding a property for a deeply nested structure), `native_decide` may fail before even attempting execution.
    

### 2.2 The Compilation Pipeline

Unlike `decide`, which stops at elaboration and hands the term to the kernel, `native_decide` pushes the term `d` into the Lean Compiler. This is where Lean 4 diverges radically from Lean 3.

#### 2.2.1 Lean Compiler Normal Form (LCNF)

The first stage of compilation converts the elaborated expression into **LCNF**. This is a functional intermediate representation designed for optimization.

- **Structure:** LCNF is a variant of A-normal form (ANF), where complex expressions are broken down into sequences of let-bindings.
    
- **Transformations:** The compiler performs:
    
    - **Inlining:** Expanding small function bodies to reduce call overhead.
        
    - **Specialization:** Creating specific versions of polymorphic functions for concrete types (e.g., specialized `List.map` for `Nat`).
        
    - **Dead Code Elimination:** Removing computations that do not contribute to the result.
        

#### 2.2.2 Mono and Intermediate Representation (IR)

Following LCNF, the code is lowered to **IR** (Intermediate Representation). This is an imperative, low-level representation that handles:

- **Memory Management:** Insertion of reference counting instructions (`inc`, `dec`).
    
- **Boxing/Unboxing:** Decisions on whether to represent a value as a pointer to a heap object (boxed) or as a raw machine scalar (unboxed).
    
- **Constructors:** Inductive type constructors are mapped to tagged pointers or scalar values. For instance, `Bool.true` might be represented as the scalar `1`, while `List.cons` is a pointer to a memory block.
    

#### 2.2.3 The Backend: Interpreter vs. Native

The final execution strategy depends on the environment and flags.

- **Interpreter (`#eval` mode):** By default, `native_decide` utilizes the Lean interpreter. This is a highly optimized bytecode interpreter written in C++. It executes the IR directly. This avoids the latency of invoking an external C compiler but runs slightly slower than fully compiled code.
    
- **Native (`decide +native`):** In fully compiled projects, or when explicitly configured, Lean generates C code. This C code is then compiled by a C compiler (typically `clang`) into a shared object (`.so` or `.dylib`) and dynamically linked into the running process. This path offers the highest possible performance, unlocking aggressive compiler optimizations (e.g., loop unrolling, vectorization) provided by LLVM.
    

### 2.3 The Runtime System

The **Runtime** is the execution environment for the compiled code. It fundamentally differs from the Kernel in how it represents data.

#### 2.3.1 `Nat` and `Int` Representation

In the Kernel, `Nat` is defined inductively:

Lean

```
inductive Nat where

| zero : Nat
| succ (n : Nat) : Nat
```

A number like `1,000,000` is a linked list of one million `succ` constructors. Kernel reduction on this structure is $O(n)$ for simple access and extremely slow for arithmetic.

In the Runtime, `Nat` and `Int` are implemented using **GMP (GNU Multiple Precision Arithmetic Library)**.

- **Small Integers:** Values that fit within a machine word (minus 1 bit for tagging) are stored directly as scalars. On a 64-bit system, this means integers up to $\approx 2^{63}$ are immediate values.
    
- **Big Integers:** Values exceeding this range are automatically promoted to heap-allocated `mpz_t` structures managed by GMP.
    
- **Performance Implication:** Arithmetic operations like addition, multiplication, and modular exponentiation are handled by highly optimized C/C++ and Assembly routines in GMP. This gives `native_decide` a performance characteristic similar to Python or Haskell, rather than a raw proof assistant.
    

#### 2.3.2 Arrays and Memory Layout

The Runtime distinguishes between `List` (linked list) and `Array` (dynamic array).

- **Lists:** Remain linked structures in memory. Traversing a list is a pointer-chasing operation, incurring high cache miss rates.
    
- **Arrays:** Are implemented as contiguous blocks of memory (C arrays). This allows for $O(1)$ random access and excellent CPU cache locality.
    
- **Critical Optimization:** Lean's runtime supports functional uniqueness optimizations. If an array has a reference count of 1 (it is uniquely owned), operations like `push` or `set` are performed in-place (mutation), avoiding the $O(n)$ copy cost usually associated with functional data structures. `native_decide` benefits automatically from this if the code is written to preserve linearity.
    

## 3. The Trusted Code Base (TCB): Risks and Realities

The decision to use `native_decide` is a decision to expand the Trusted Code Base (TCB). In formal methods, the TCB consists of all hardware and software components that, if faulty, could allow a false proposition to be accepted as true.

### 3.1 The Expansion Quantification

Standard Lean proofs (checked by the Kernel) have a minimal TCB:

1. The Kernel (C++ implementation of CIC).
    
2. The C++ compiler used to build Lean.
    
3. The Hardware.
    

When `native_decide` is employed, the TCB expands to include:

1. **The Lean Compiler (30k+ lines):** Including LCNF transformations and IR generation.
    
2. **The Lean Interpreter/VM:** The complex C++ machinery that executes bytecode.
    
3. **The Runtime System:** Including the garbage collector and object model.
    
4. **External Libraries:** Specifically **GMP** for arithmetic and `glibc` (or equivalent) for system calls.
    
5. **Foreign Function Interfaces (FFI):** Any C/C++ code linked via `@[extern]`.
    

This expansion is non-trivial. History has shown that optimizing compilers are significantly more prone to bugs than small logical kernels. A bug in the compiler's dead code elimination pass, for instance, could theoretically cause a decision procedure to return `true` unconditionally.

### 3.2 The `Lean.ofReduceBool` Axiom

The logical validity of `native_decide` hangs on a single axiom:

Lean

```
axiom Lean.ofReduceBool (a b : Bool) (h : reduceBool a = b) : a = b
```

This axiom is profound. It asserts a semantic equivalence between the _operational semantics_ of the runtime (represented by the opaque `reduceBool` constant) and the _denotational semantics_ of the logic. By invoking this, the user is stating: "I trust that the value computed by the compiled binary is the same value that would be computed by reducing the term in the logic".

This axiom acts as a firewall. Proofs using `native_decide` are "infected" with this axiom. When a user checks the axioms of a theorem using `#print axioms MyTheorem`, `Lean.ofReduceBool` will appear, signaling that the proof's validity is conditional on the correctness of the compiler and runtime.

### 3.3 The "Shadow Logic" and `@[implemented_by]`

Lean 4 allows developers to provide efficient runtime implementations for logical functions using the `@[implemented_by]` attribute. This creates a "Shadow Logic" – a parallel universe where functions may behave differently than their logical definitions.

**The Mechanism:**

Lean

```
def myFunc : Nat → Nat :=... -- Logical definition (slow, safe)

@[extern "c_my_func"]
def myFunc_impl : Nat → Nat :=... -- Native implementation (fast, unsafe?)

attribute [implemented_by myFunc_impl] myFunc
```

When the Kernel checks `myFunc`, it uses the logical definition. When `native_decide` executes `myFunc`, it uses `myFunc_impl`. If these two diverge, soundness is broken.

### 3.4 Case Study: The `IO.getRandomBytes` Exploit

A famous example of unsoundness introduced by `native_decide` involves non-deterministic I/O. In pure logic, all functions must be deterministic. However, the `IO` monad allows interaction with the outside world.

**The Exploit:**

1. Define a function `coin_flip` using `IO.getRandomBytes` that returns a random `Bool`.
    
2. Wrap this in a way that allows it to be called as a pure function (e.g., using `unsafe` casts or exploiting `@[implemented_by]` on a pure interface).
    
3. Invoke `native_decide` to prove `coin_flip = true`. If the random generator outputs `true`, the proof succeeds.
    
4. Invoke `native_decide` again to prove `coin_flip = false`. If the generator outputs `false`, this proof also succeeds.
    
5. Combine the two proofs to derive `true = false`, and thus `False`.
    

**The Root Cause:** The compiler and runtime executed the side-effecting code, violating the logical assumption of referential transparency. While the Kernel would typically reject such definitions due to type constraints (IO monad handling), the aggressive nature of `native_decide` combined with `unsafe` or `implemented_by` creates a loophole.

**Competition Implication:** In a competition, using `native_decide` on code that relies on undefined behavior, uninitialized memory (via FFI), or non-determinism is a recipe for disaster. A solution might pass locally but fail on the judge's machine due to different random seeds or memory states.

## 4. Performance Analysis: When and How to Use

The utility of `native_decide` is defined by its performance relative to other methods. This section provides a comparative analysis and benchmarks to guide tactic selection.

### 4.1 Comparative Benchmarks

We compare `native_decide` against three primary alternatives: `decide` (kernel), `norm_num` (symbolic simplifier), and `omega` (Presburger arithmetic solver).

|**Metric**|**decide (Kernel)**|**norm_num**|**native_decide**|**omega**|
|---|---|---|---|---|
|**Execution Model**|Symbolic Reduction|Tactic-driven Rewriting|Native Compilation/JIT|Constraint Solving|
|**Integer Rep.**|Unary (Inductive)|Binary (Symbolic)|GMP (Machine Int/Bignum)|Symbolic Atoms|
|**Big Int Addition**|Exponentially Slow|Linear (Bitwise)|Instant (GMP)|N/A (Logical)|
|**Recursion Depth**|Limited by Heartbeats|Limited by Heartbeats|Limited by Stack/RAM|N/A|
|**TCB Impact**|None|None|High (Compiler+)|None|
|**Overhead**|Low (for tiny terms)|Medium|High (Compilation latency)|Medium|

#### 4.1.1 Large Integer Arithmetic

For checking primality of a large number (e.g., $10^{100}$):

- **`decide`:** Fails immediately. The unary representation creates a term size exceeding memory.
    
- **`norm_num`:** Can handle addition/multiplication of large numbers efficiently because it uses a binary representation in the logic. However, for primality testing, it must construct a proof of non-divisibility for every candidate factor. This scales poorly compared to native machine division.
    
- **`native_decide`:** Invokes GMP. The operation is instantaneous. The overhead is solely the compilation of the call.
    

**Insight:** Use `native_decide` for number theoretic properties of specific large integers where no efficient symbolic proof exists.

#### 4.1.2 Data Structure Traversal

For a graph search problem (e.g., "Does a path exist on this 100-node graph?"):

- **`decide`:** Fails. Reduction of complex data structures generates massive amounts of intermediate `Expr` allocations.
    
- **`native_decide`:** Succeeds, provided `Array` is used. If `List` is used, performance degrades significantly due to cache misses, but may still pass small instances.
    

### 4.2 Array vs. List Performance

A critical optimization for `native_decide` is the choice of data structures.

- **List:**
    
    - _Memory:_ Dispersed nodes.
        
    - _Access:_ $O(n)$.
        
    - _Effect on `native_decide`:_ Even in compiled code, walking a linked list is slow. Deep recursion on lists often triggers stack overflows because the compiler might fail to optimize non-tail-recursive list operations.
        
- **Array:**
    
    - _Memory:_ Contiguous buffer.
        
    - _Access:_ $O(1)$.
        
    - _Effect on `native_decide`:_ Compiles to efficient C array accesses. The runtime's reuse analysis allows in-place updates, effectively giving persistent arrays the performance of mutable arrays.
        

**Benchmark:** Summing an array of 1,000,000 integers.

- `List.foldl`: ~100ms (High GC pressure).
    
- `Array.foldl`: ~5ms (Near C speeds).
    

**Competition Strategy:** Always prefer `Array` for input data and intermediate states in algorithmic problems intended for `native_decide`.

### 4.3 Tail Recursion and Stack Limits

Lean 4 performs **Tail Call Optimization (TCO)**. This means a recursive call in the tail position is compiled into a jump (loop), reusing the current stack frame.

- **Non-Tail Recursive:** Consumes stack space proportional to depth.
    
- **Tail Recursive:** Constant stack space.
    

In `native_decide`, a non-tail-recursive function running on a large input will cause the interpreter or the compiled binary to segfault (Stack Overflow). Unlike the Kernel, which has a managed `maxHeartbeats` counter and fails gracefully with a timeout error, a stack overflow in the native runtime can crash the process or yield an opaque error.

**Safe Pattern:** Use Accumulator Passing Style (APS) to ensure tail recursion.

Lean

```
-- Unsafe (Stack Overflow risk)
def sum : List Nat → Nat

| => 0
| h :: t => h + sum t

-- Safe (Tail Recursive)
def sum_tco (acc : Nat) : List Nat → Nat

| => acc
| h :: t => sum_tco (acc + h) t
```

## 5. Competition Strategy: Kaggle and Offline Environments

Participating in competitions like the AIMO on platforms like Kaggle introduces specific environmental constraints that dictate how `native_decide` should be deployed.

### 5.1 The "Freezing" Pattern

In a competition, you have limited GPU/CPU time. Re-running a heavy `native_decide` computation every time you verify your file is wasteful.

**Strategy:**

1. **Computation Phase:** Use `native_decide` (or `#eval`) in a scratch file to find a witness or a result.
    
    - Example: Find a counterexample to a conjecture. `def find_counter :=...`
        
    - `#eval find_counter` -> `some 12345`.
        
2. **Freezing Phase:** Hardcode the result into your submission file.
    
    - Instead of `example : ∃ n, P n := by native_decide`, write `example : P 12345 := by native_decide`.
        
    - Better yet, if the verification of the specific witness is cheap, use `norm_num` or `decide` for the final proof step to reduce TCB reliance, having used `native_decide` only for discovery.
        

### 5.2 Handling Timeouts and Resources

Kaggle notebooks run in Docker containers with specific resource limits.

- **Compilation Time vs. Execution Time:** `maxHeartbeats` limits the _compilation_ (elaboration) of the `native_decide` instance. It does _not_ necessarily limit the execution time of the compiled binary once it starts running. An infinite loop in `native_decide` can cause the notebook to hang until the global supervisor kills it (Time Limit Exceeded).
    
- **Debug Strategy:** If a submission times out, do not assume it's a slow proof search. It might be a non-terminating `native_decide`. Wrap potentially dangerous calls with explicit fuel counters (e.g., pass a `fuel : Nat` argument that decrements on each call) to force termination.
    

### 5.3 Offline Compilation

Kaggle notebooks are often cut off from the internet during grading.

- **Dependency:** `native_decide` relies on the bundled Lean toolchain. It does not require internet.
    
- **C Compiler:** If your project configuration forces compilation to C (via Lake or specific flags), ensure `clang` is available in the environment. The standard Lean distribution includes a bundled `clang`, but path issues can arise. Using the interpreter (default for `#eval`/`native_decide` in single files) is safer in restricted environments as it avoids external toolchain dependencies.
    

### 5.4 Using `native_decide` as an Oracle

Often, proving a theorem requires intuition. `native_decide` can serve as a rigorous oracle.

- **Scenario:** You need to prove $\forall n < 1000, P(n)$.
    
- **Oracle:** Write a function `check_all : Unit → Bool` that iterates 0..999.
    
- **Proof:** `example : ∀ n < 1000, P n := by native_decide` works, but if you want to be robust, you can use `native_decide` to generate a list of "hard cases" and handle only those with slow tactics, handling the rest via automation.
    

## 6. Safe Usage Patterns and Best Practices

To mitigate the risks of TCB expansion and runtime instability, adopt the following "Defensive Verification" patterns.

### 6.1 The "Sanitization" Workflow

Never pass raw, untrusted functions to `native_decide`.

1. **Termination:** Ensure the function is structurally recursive or well-founded. Avoid `partial` definitions.
    
2. **Purity:** Ensure the function does not use `IO`, `unsafe` casts, or `@[extern]` calls to unverified C code.
    
3. **Isolation:** Keep `native_decide` usages in separate files or specifically marked sections. This allows you to audit the "unsafe" parts of your proof quickly.
    

### 6.2 Type Selection for Performance

- **Use `UInt64` / `USize`:** For loop counters and array indices. These compile to raw machine integers, avoiding the slight overhead of GMP `Nat` (though Lean optimizes `Nat` well).
    
- **Use `ByteArray`:** For handling large blobs of binary data (e.g., parsing a file format).
    
- **Avoid `String`:** For heavy manipulation. Lean's `String` is UTF-8 correct and immutable, which can be slow for intensive parsing. Use `ByteArray` buffers.
    

### 6.3 Fallback Chains

In automation scripts, use a fallback chain:

1. Try `norm_num` (safest, medium speed).
    
2. Try `omega` (safe, fast for linear arithmetic).
    
3. Try `native_decide` (unsafe TCB, fastest).
    

This ensures that you only pay the TCB cost when absolutely necessary.

### 6.4 The `trustCompiler` Audit

Before submitting a solution, run:

Lean

```
#print axioms MyMainTheorem
```

If `Lean.ofReduceBool` appears, acknowledge that the proof is conditional. In some competition tracks, this might carry a penalty or be disallowed. If strictly verified proofs are required, you must use `native_decide` only for _finding_ the solution (the witness), and then verify that witness using kernel-safe tactics.

## 7. Future Outlook: Verified Compilation

The tension between `native_decide` and strict verification is a known issue in the community. Projects like **Lean4Lean** (an external verifier for Lean 4 written in Lean 4) and verified compilation efforts aim to close this gap.

- **Verified Compilers:** In the future, we may have a compiler from Lean to C that is itself verified in Lean. This would remove the compiler from the TCB, leaving only the hardware and the spec.
    
- **Certificate Generation:** Tools like `bv_decide` are pioneering a middle ground: using an external solver to find a solution but requiring it to produce a certificate that the kernel can check. This is the "gold standard" for the future of competition proving—using computation to find the proof, but not _as_ the proof.
    

## 8. Conclusion

`native_decide` is a powerful instrument in the Lean 4 arsenal, embodying the philosophy of computation as verification. For the competition participant, it opens the door to solving problems that require intensive calculation, simulation, or brute-force search—domains where traditional theorem provers have historically struggled.

However, this power is wielded at the cost of expanding the trusted code base and introducing potential vectors for unsoundness through compiler bugs or unsafe runtime implementations. By understanding the internal compilation pipeline, respecting the differences between logical and runtime data representations, and adhering to strict resource management and sanitization patterns, one can leverage `native_decide` effectively.

In the high-pressure environment of a math competition, `native_decide` should be viewed as a precision tool: used for specific, computationally heavy sub-goals where the kernel is the bottleneck, but surrounded by a defensive architecture that preserves the overall integrity of the solution.

---

## Appendix A: Summary of Tactic Performance Characteristics

|**Feature**|**decide**|**native_decide**|**norm_num**|**omega**|
|---|---|---|---|---|
|**Primary Use Case**|Small finite domains, boolean logic|Large computation, simulation, primality|Arithmetic inequalities, concrete numbers|Linear integer arithmetic, Presburger|
|**Trusts Compiler?**|No|**Yes**|No|No|
|**Large Int Support**|Poor (Unary)|Excellent (GMP)|Good (Binary)|N/A (Symbolic)|
|**Recursion Limit**|`maxHeartbeats`|`maxRecDepth` (Stack)|`maxHeartbeats`|N/A|
|**Proof Object**|Full `Expr` tree|`ofReduceBool` application|Full `Expr` tree|Full `Expr` tree|
|**Competition Role**|Basic sanity checks|**Heavy lifting / Oracle**|Standard arithmetic steps|Constraint solving|

## Appendix B: Safe Usage Checklist for Competitions

- [ ] **Tail Recursion:** Verify all recursive functions passed to `native_decide` use accumulator passing style.
    
- [ ] **Data Structures:** Replace `List` with `Array` for any collection exceeding 100 elements.
    
- [ ] **Termination:** Ensure functions are total; consider adding a `fuel` parameter if unsure.
    
- [ ] **Sanitization:** Audit code for `unsafe`, `partial`, or `@[extern]` attributes.
    
- [ ] **Local Testing:** Test with `#eval` before wrapping in `native_decide` to distinguish runtime errors from tactic failures.
    
- [ ] **Witness Isolation:** If possible, use `native_decide` to _find_ a value, then `norm_num` to _check_ it.
