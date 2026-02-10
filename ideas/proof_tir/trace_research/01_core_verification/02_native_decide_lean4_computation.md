# Deep Research: Lean 4's native_decide — Computation as Verification

## Research Objective

Achieve complete understanding of Lean 4's `native_decide` tactic — how it works internally, its performance characteristics, its limitations, and how to use it safely for verified computation in a competition environment. This is the core enabler of our architecture: the ability to verify formulas via computation rather than proof search.

## Context

Our system uses Lean 4 NOT for theorem proving in the traditional sense. We use it as a verified computation engine:
```lean
def f := fun n => n * (n + 1) / 2
theorem check : (List.range 100).all (fun n => f n = expected n) = true := by native_decide
```

This compiles to C++, runs a loop, and returns true/false. We need to understand every aspect of this mechanism.

## Research Questions

### 1. Internal Mechanics
- What exactly happens when `native_decide` is invoked?
- Walk through the compilation pipeline: Lean → IR → C++ → Binary → Execution → Result
- What is the `Lean.ofReduceBool` axiom? What does `Lean.trustCompiler` mean?
- How does the kernel accept the result? Is there any proof term generated, or is it purely axiomatic?
- What's the difference between `native_decide`, `decide`, and `decide +kernel`?

### 2. Performance Characteristics
- How fast is `native_decide` compared to kernel reduction (`decide`)?
- What's the overhead of compilation? Is the C++ code cached between invocations?
- For our use case (checking f(n) = g(n) for n=1..100), what's the expected latency?
- At what problem size does `native_decide` become necessary? (n=10? n=100? n=1000?)
- Memory usage: how much RAM does verification of large computations require?
- Can `native_decide` handle Nat operations on numbers like 10^100? 10^1000?

### 3. What Can Be Computed
- What types have `Decidable` instances that work with `native_decide`?
- Can we use it for:
  - Nat arithmetic (addition, multiplication, division, modulo)
  - Int arithmetic (negative numbers)
  - List operations (map, filter, all, any)
  - Array operations
  - Nested function calls
  - Recursive functions (with proven termination)
- What CANNOT be computed via `native_decide`?
- How do we handle modular exponentiation efficiently? (Our `powMod` function)

### 4. Trusted Code Base (TCB) Analysis
- What exactly is added to the TCB when using `native_decide`?
  - Lean compiler
  - C++ compiler (which one? GCC? Clang?)
  - LLVM (if applicable)
  - GMP library
  - Hardware/CPU
- How many lines of code is the TCB expansion approximately?
- What are the historical bugs found in this pipeline? List specific CVEs or GitHub issues.
- Has anyone proven `False` via `native_decide` bugs? Document these cases.

### 5. GMP Integration Deep Dive
- How does Lean 4 use GMP for `Nat` and `Int`?
- What operations are delegated to GMP?
- Are there known edge cases where GMP's implementation differs from Lean's mathematical definition?
- What about division semantics? Modulo with negative numbers? Edge cases at 0?
- How do `@[implemented_by]` overrides work? What's the risk of mismatch?

### 6. Configuration and Tuning
- What do these options do and what values should we use?
  - `set_option maxRecDepth 10000`
  - `set_option maxHeartbeats 0`
  - Any other relevant options?
- How do we set timeouts for `native_decide` computations?
- Can we limit memory usage?
- What happens if computation exceeds limits — graceful failure or crash?

### 7. Error Handling
- If `native_decide` returns `false`, what error message do we get?
- Can we get diagnostic information about WHY it failed (which n failed the check)?
- How do we distinguish between:
  - Formula is wrong (correct behavior)
  - Computation timed out
  - Memory exhausted
  - Compilation error

### 8. Comparison with Alternatives
- How does `native_decide` compare to:
  - `decide` (kernel reduction)
  - `norm_num` (numeric normalization)
  - `omega` (linear arithmetic)
  - `#eval` (evaluation without proof)
- When should we use each?
- Can we combine them (e.g., `native_decide` for the loop, `omega` for subgoals)?

### 9. Offline Deployment
- Does `native_decide` require any runtime dependencies beyond the Lean binary?
- In a Kaggle environment with no internet, what needs to be pre-packaged?
- Are compiled binaries cached? Where? Can we pre-warm the cache?
- Does the `.olean` cache include native code, or is recompilation needed?

### 10. Patterns for Our Use Case
- What's the optimal Lean template for verifying "formula f equals expected sequence"?
- Should we use `List.range` or `Array.range` or manual recursion?
- How do we efficiently pass the expected sequence (hardcode? compute from recurrence?)?
- Example patterns for:
  - Checking closed-form formula: f(n) = n*(n+1)/2
  - Checking recurrence: a(n) = a(n-1) + a(n-2) with base cases
  - Checking modular formula: f(n) = 7^n mod 100
  - Checking with multiple conditions/branches

### 11. Safety and Soundness
- For competition use (not Mathlib publication), is `native_decide` "safe enough"?
- What's the practical probability of a false positive (accepting wrong formula)?
- How does this compare to the probability of LLM reasoning errors (~10%)?
- Should we run verification twice with different random seeds or orderings?

### 12. Advanced Usage
- Can we use `native_decide` to verify properties of functions, not just sequences?
- Can we verify "for all x in [0,1000], polynomial P(x) ≥ 0"?
- Can we verify graph properties (connectivity, coloring) for finite graphs?
- What about verifying SAT instances? (Connection to `bv_decide`)

## Desired Output Format

1. **Technical Deep Dive**: Complete explanation of `native_decide` internals
2. **Performance Benchmarks**: Concrete numbers for various computation sizes
3. **TCB Risk Assessment**: Detailed analysis of trust assumptions
4. **Configuration Guide**: Recommended settings for competition use
5. **Code Templates**: Optimal Lean patterns for formula verification
6. **Bug Compendium**: Historical issues and their resolutions
7. **References**: Lean documentation, Zulip discussions, academic papers




















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