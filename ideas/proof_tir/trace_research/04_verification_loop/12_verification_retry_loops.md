# Deep Research: Verification Retry Loops — Error Recovery and Iteration Strategy

## Research Objective

When Lean rejects a formula, the system needs to recover. The verification step provides a binary signal (accept/reject) and potentially error messages. We need a strategy for iterating: adjusting the formula, trying alternative mining strategies, or falling back to TIR.

## Context

The pipeline:
```
Trace → Mine Pattern → Translate to Lean → native_decide → Accept/Reject
```

If rejected:
- The formula might be wrong (mining failed)
- The translation might be wrong (syntax error)
- The trace might be wrong (LLM code bug)

We need systematic retry logic to maximize success rate within time constraints (6 min/problem).

## Research Questions

### Part A: Failure Mode Classification

#### 1. Lean Rejection Types
Classify all possible rejection scenarios:
- **Type Error**: Formula doesn't type-check (syntax/translation issue)
- **Compilation Error**: Lean can't compile the theorem
- **native_decide False**: Formula compiles but is mathematically wrong
- **Timeout**: Verification takes too long
- **Memory Error**: Out of memory during verification

#### 2. Distinguishing Failure Types
- How do we tell which type of failure occurred?
- What does Lean's error message say for each?
- Can we parse error messages automatically?

#### 3. Root Cause Analysis
For each failure type, what's the likely root cause?
- Type error → translation bug
- native_decide false → mining error OR translation error OR trace error
- Timeout → formula too complex OR verification range too large

### Part B: Translation Error Recovery

#### 4. Common Translation Bugs
- Division: Python `//` vs Lean `/`
- Power: Python `**` vs Lean `^` or `Nat.pow`
- Modulo: Semantic differences for negative numbers
- Missing parentheses
- Type mismatches (Nat vs Int)

#### 5. Automatic Fix Strategies
For each common bug pattern:
- Detection heuristic from error message
- Automatic fix template
- Example before/after

#### 6. LLM-Assisted Translation Repair
- Pass error message to LLM with original formula
- Few-shot examples of error → fix
- What's the success rate of LLM repair?

### Part C: Mining Error Recovery

#### 7. When Mining Fails
Scenarios where the mined pattern is wrong:
- Sequence not linearly recurrent (B-M gives garbage)
- Insufficient trace terms
- Polynomial degree higher than assumed
- Named sequence not recognized

#### 8. Tier Escalation
If Tier 1 (B-M) fails:
- Move to Tier 2 (Lagrange interpolation)
- Check if polynomial fits
- Move to Tier 3 (LLM pattern recognition)
- Move to Tier 4 (OEIS)

#### 9. Alternative Mining Strategies
- Try B-M with more terms
- Try polynomial of higher degree
- Ask LLM to guess the pattern from trace
- Search OEIS for the sequence

#### 10. Ensemble Mining
- Run multiple miners in parallel
- Each produces a candidate formula
- Verify each against trace BEFORE Lean
- Only Lean-verify candidates that pass trace check

### Part D: Trace Error Recovery

#### 11. Detecting Trace Bugs
If all reasonable formulas fail verification:
- Maybe the trace itself is wrong
- Signs: no formula fits, obvious anomaly in trace
- Statistical outlier detection?

#### 12. Trace Regeneration
- Prompt LLM again for trace code
- Use different prompt / temperature
- Ask for more verbose code with comments
- Request assertions in the code

#### 13. Trace Validation
Before mining, validate the trace:
- Monotonicity check (if expected)
- Positivity check
- Consistency with problem constraints
- Cross-check first few terms manually

### Part E: Retry Strategy

#### 14. Time Budget Allocation
6 minutes per problem, how to allocate:
- 1 min: Trace generation (with 2 retries)
- 2 min: Mining + Lean verification (with 3-5 retry iterations)
- 2 min: Fallback strategies (TIR, alternative approaches)
- 1 min: Buffer

#### 15. Retry Limits
- Maximum retries per phase?
- When to give up on formal verification?
- Confidence threshold for accepting TIR fallback?

#### 16. Parallel vs Sequential Retries
- Run multiple strategies in parallel?
- Sequential with early termination on success?
- Trade-off: resource usage vs speed

### Part F: Error-Driven Learning

#### 17. Error Message Parsing
- What information can we extract from Lean errors?
- "type mismatch" → which part of the formula?
- "failed to synthesize Decidable" → what's not decidable?
- "native_decide failed" → which n violated the check?

#### 18. Diagnostic native_decide
Can we modify verification to give more info?
```lean
-- Instead of:
theorem check : (List.range 100).all (fun n => f n = g n) = true := by native_decide

-- Use:
#eval (List.range 100).filter (fun n => f n ≠ g n)  -- Which n fail?
```

#### 19. Feedback to Mining
If we know which n fails:
- n=0 fails → base case issue
- n=100 fails → formula diverges for large n
- n=17 fails → specific edge case
- Use this to refine formula

### Part G: Fallback to TIR

#### 20. When to Fall Back
Triggers for giving up on formal verification:
- Exceeded retry budget
- No viable formula candidates remain
- Problem type is inherently non-verifiable

#### 21. TIR Execution
When falling back:
- Use the trace code directly to compute answer
- Or generate new TIR solution from scratch
- Confidence marking (lower than verified answer)

#### 22. Hybrid Confidence Scoring
Output format:
```python
{
    "answer": 42,
    "method": "trace-to-lean",  # or "tir-fallback"
    "confidence": 0.99,  # 0.99 for verified, 0.7 for TIR
    "retries": 3
}
```

### Part H: Learning from Failures

#### 23. Failure Logging
Log every failure with:
- Problem ID
- Failure type
- Error message
- Failed formula
- Retry count
- Final outcome

#### 24. Post-Competition Analysis
Use logs to:
- Identify systematic failure patterns
- Improve prompts
- Improve mining algorithms
- Tune retry parameters

#### 25. Online Learning (If Permitted)
If we could update system during competition:
- Learn from early problems
- Adjust strategies for later problems
- Not allowed in AIMO, but useful for development

### Part I: Implementation Patterns

#### 26. Retry Loop Structure
```python
def solve_with_verification(problem, max_retries=5):
    trace = generate_trace(problem)
    for tier in [berlekamp_massey, lagrange, llm_pattern, oeis]:
        formula = tier(trace)
        if formula is None:
            continue
        for attempt in range(max_retries):
            lean_code = translate(formula)
            result = verify(lean_code)
            if result.success:
                return formula.compute(N)
            elif result.is_translation_error:
                formula = fix_translation(formula, result.error)
            elif result.is_wrong_formula:
                break  # Try next tier
    return tir_fallback(problem)
```

#### 27. Error Handler Dispatch
```python
def handle_lean_error(error):
    if "type mismatch" in error:
        return {"action": "fix_types", "info": parse_type_error(error)}
    elif "native_decide failed" in error:
        return {"action": "next_tier", "info": None}
    elif "timeout" in error:
        return {"action": "reduce_range", "info": None}
    ...
```

#### 28. State Machine for Retry Logic
- States: TRACE, MINE_TIER1, MINE_TIER2, ..., VERIFY, FIX, FALLBACK, DONE
- Transitions based on outcomes
- Clear termination conditions

### Part J: Testing and Validation

#### 29. Retry System Test Cases
Create test cases that exercise:
- Successful first-try
- Successful after translation fix
- Successful after tier escalation
- Successful after trace regeneration
- Graceful fallback to TIR
- Timeout handling

#### 30. Performance Metrics
Track:
- Average retries per problem
- Success rate by retry count
- Time spent in each phase
- Failure type distribution

## Desired Output Format

1. **Failure Taxonomy**: Complete classification of all failure modes
2. **Recovery Strategies**: What to do for each failure type
3. **Retry Loop Design**: State machine and flowchart
4. **Error Message Guide**: How to parse and act on Lean errors
5. **Time Budget Analysis**: Optimal allocation of retry budget
6. **Implementation Code**: Python patterns for retry logic
7. **Testing Protocol**: How to validate the retry system
8. **Metrics Dashboard**: What to track and optimize


























# Verification Retry Loops: Comprehensive Error Recovery and Iteration Strategies in Automated Mathematical Discovery

## 1. Introduction: The Neuro-Symbolic Verification Gap

The integration of Large Language Models (LLMs) with formal verification systems represents a pivotal advancement in the field of automated reasoning. This report examines a specific, high-stakes domain within this field: the automated mining and verification of integer sequences using the Lean 4 theorem prover. The core pipeline under analysis—transitioning from **Trace Generation** to **Pattern Mining**, **Translation**, and finally **Formal Verification**—promises to bridge the gap between probabilistic intuition and provable truth. However, the operational reality of this pipeline is characterized by extreme fragility. The interface between the stochastic output of an LLM and the rigid, unforgiving semantics of a kernel-based proof assistant like Lean 4 creates a "Neuro-Symbolic Gap" where the majority of candidate solutions fail.

The "happy path"—where an LLM generates a correct trace, an algorithm mines the perfect formula, and Lean accepts it on the first attempt—is statistically rare. In high-throughput environments, such as those constrained by a six-minute execution window per problem, the success of the system depends less on the brilliance of the initial guess and more on the robustness of the **Verification Retry Loop**. This report argues that the retry mechanism must be elevated from a simple exception-handling routine to a sophisticated, decision-theoretic agent capable of diagnosing failure modes, managing computational budgets, and navigating a hierarchy of formal and empirical reasoning strategies.

We analyze the role of the `native_decide` tactic in Lean 4, which offers a unique trade-off between performance and trust, compiling Lean definitions into optimized C++ binaries for execution. While this enables the verification of computationally intensive properties (such as the 10,000th term of a sequence), it introduces complex failure modes ranging from Foreign Function Interface (FFI) crashes to deterministic timeouts. Furthermore, we explore the semantic dissonance between Python (the language of discovery) and Lean (the language of verification), particularly in integer arithmetic, which serves as a frequent, silent killer of valid proofs.

The objective of this research is to define a comprehensive strategy for iteration. We propose a taxonomy of failure modes that decomposes the binary "Accept/Reject" signal into actionable intelligence. We detail algorithmic fallback strategies, moving from linear recurrence mining (Berlekamp-Massey) to rational interpolation (Pade approximants) and finally to Tool Integrated Reasoning (TIR). By synthesizing insights from recent literature on LLM self-correction , agentic loops , and formal library design , this report provides a blueprint for a resilient, autonomous mathematical discovery engine.

---

## 2. The Verification Pipeline Architecture

To understand the necessity of a complex retry loop, one must first appreciate the architectural dependencies of the generation pipeline. The workflow consists of four distinct stages, each acting as a filter that can introduce, amplify, or expose errors.

### 2.1 Stage 1: Trace Generation (The Probabilistic Foundation)

The process begins with an LLM generating a Python script to produce the first $N$ terms of an integer sequence. This stage is inherently probabilistic. The LLM functions as a "guess generator," leveraging its training on the Online Encyclopedia of Integer Sequences (OEIS) to predict sequence behavior.

- **Primary Failure Mode:** **Hallucination.** The LLM may generate a script that produces numbers resembling the target sequence but diverging at the $k$-th term due to an off-by-one error or a misunderstanding of the mathematical definition.
    
- **Secondary Failure Mode:** **Cheating/Lookup.** As noted in benchmarks like finding OEIS sequences, models may attempt to hardcode values rather than implementing the algorithm. A robust verification loop must detect this by validating the formula against terms _outside_ the hardcoded range.
    

### 2.2 Stage 2: Pattern Mining (The Algorithmic Bridge)

Once a numerical trace (e.g., `[1, 1, 2, 3, 5, 8...]`) is generated, the system employs classical algorithms to hypothesize a closed-form solution or recurrence relation. This transforms raw data into a mathematical object.

- **Algorithm A: Berlekamp-Massey.** This algorithm synthesizes the shortest linear feedback shift register (LFSR) capable of generating the sequence. It is the standard tool for linear recurrences.
    
- **Algorithm B: Rational Interpolation.** If the sequence is not strictly linear but grows regularly, rational approximation (Pade approximants) is used to find a generating function $P(x)/Q(x)$.
    
- **Risk:** These algorithms are sensitive to noise. A single wrong digit in the trace (from Stage 1) can cause the miner to produce a vastly more complex, incorrect formula, or fail to converge entirely.
    

### 2.3 Stage 3: Translation to Lean (The Semantic Transformation)

The mined mathematical object (usually represented in Python or SymPy syntax) must be translated into Lean 4 syntax. This is not a mere string replacement; it is a compilation from an untyped, dynamic environment (Python) to a dependently typed, rigid environment (Lean).

- **The Semantic Gap:** Python handles large integers and mixed types transparently. Lean requires explicit type handling (`Nat` vs. `Int` vs. `Rat`) and rigorous proofs of termination for recursive functions.
    
- **Critical Friction:** A formula that evaluates correctly in Python may fail to even compile in Lean due to type inference failures (`failed to synthesize instance`) or universe level constraints.
    

### 2.4 Stage 4: Formal Verification (The Deterministic Gatekeeper)

The final stage is the execution of the proof. In this specific pipeline, we rely on `native_decide` to verify that the translated formula matches the generated trace for a set of test indices.

- **Mechanism:** `native_decide` attempts to prove a proposition $P$ by synthesizing a `Decidable P` instance and evaluating it to `isTrue`. Unlike standard kernel reduction (`rfl`), `native_decide` compiles the Lean definition into C++, links it against the Lean runtime, and executes it.
    
- **Implication:** This allows for extremely fast evaluation of complex functions but adds the C++ compiler and the Foreign Function Interface (FFI) to the Trusted Code Base (TCB). It also introduces unique error modes related to binary compilation and system resource limits that do not exist in pure kernel verification.
    

---

## 3. Phenomenology of Verification Failure: A Deep Taxonomy

A robust retry loop cannot treat "failure" as a monolith. The strategy for recovering from a syntax error is fundamentally different from recovering from a logical refutation. We classify failures into a four-stratum taxonomy based on the depth of execution reached before the error occurs.

### 3.1 Stratum 1: Compilation and Elaboration Failures

These errors occur before any mathematical logic is checked. They represent an inability of the Lean environment to understand the question being asked.

#### 3.1.1 Syntax and Parsing Hallucinations

LLMs trained on mixed codebases often "hallucinate" syntax, blending Lean 3 and Lean 4, or inventing Python-like constructs within Lean.

- **Indicators:** "unexpected token," "function expected," "unknown identifier."
    
- **Root Cause:** The translation layer failed to strictly adhere to Lean 4's grammar. For example, using `begin... end` blocks (Lean 3) instead of `by...` (Lean 4), or misplacing parentheses in function applications.
    

#### 3.1.2 Type Synthesis Failures

Lean 4 uses a mechanism called Type Class Synthesis to handle polymorphism (e.g., `+` working on both `Nat` and `Int`).

- **Indicators:** `failed to synthesize instance`.
    
    - _Example 1:_ `failed to synthesize instance DivisionRing ℕ`. This occurs if the formula uses field division (`/`) on Natural numbers, which form a Semiring, not a Division Ring.
        
    - _Example 2:_ `failed to synthesize instance Decidable (x < y)`. This suggests the proposition involves types (like Real numbers) where equality or inequality is not computationally decidable without additional axioms or approximations.
        
- **Implication:** The formula might be mathematically sound, but it is expressed in types that do not support the required operations. Recovery requires **Type Casting** (Horizon 1).
    

#### 3.1.3 Universe and Metavariable Errors

- **Indicators:** "stuck at metavariable," "universe level too large".
    
- **Context:** These errors often arise in recursive definitions where Lean cannot infer the type of an intermediate term. For example, a `let rec` binding inside a tactic block might fail to generalize properly if it references variables from the outer scope.
    

### 3.2 Stratum 2: Tactic Execution and Resource Failures

In this stratum, the code is valid Lean, but the `native_decide` tactic fails to complete the verification due to resource constraints or implementation bugs.

#### 3.2.1 Deterministic Timeouts (Heartbeat Exhaustion)

Lean 4 measures computation not in seconds but in "heartbeats" (abstract units of memory allocation/reduction steps).

- **Indicators:** `(deterministic) timeout at 'whnf', maximum number of heartbeats (200000) has been reached`.
    
- **Mechanism:** `whnf` (Weak Head Normal Form) reduction is the process of evaluating terms. If the sequence formula is inefficient (e.g., a naive exponential Fibonacci implementation: $F_n = F_{n-1} + F_{n-2}$), calculating $F_{100}$ will generate a call tree exceeding the heartbeat limit.
    
- **Differentiation:** A timeout at `whnf` implies calculation cost. A timeout at `isDefEq` implies type-checking complexity (often due to deep type aliases or heavy coercion chains).
    
- **Recovery:** This is a strong signal to optimize the definition (e.g., enabling `@implemented_by` with a tail-recursive version) or to reduce the verification index $N$.
    

#### 3.2.2 Native Compilation Crashes

Since `native_decide` involves C++ code generation, it is susceptible to lower-level failures.

- **Indicators:** `failed to compile definition`, `error in C++ code generation`.
    
- **Context:** Bugs in the interaction between Lean's intermediate representation (IR) and the C++ compiler can cause this. Specifically, nested `let rec` declarations or complex partial definitions have historically triggered compiler bugs.
    
- **Security Risk:** Unsound logic or GMP (arbitrary precision integer) implementation bugs can lead to crashes or, worse, "False" proving "True" in rare cases.
    

### 3.3 Stratum 3: Logical Refutations

This is the most critical failure mode: the system works, the verification runs, and it proves that the formula is **incorrect**.

#### 3.3.1 Binary Rejection

- **Indicator:** `native_decide` reduces the proposition to `false`.
    
- **Context:** The formula predicts $f(n) = X$, but the actual calculation yields $Y$.
    
- **Ambiguity:** A simple `false` does not tell us _where_ the failure occurred. It could be at $n=0$ or $n=1000$.
    

#### 3.3.2 Counterexample Generation

- **Indicator:** Advanced tactics (like `bv_decide` or custom implementations) providing a variable assignment that falsifies the goal.
    
- **Strategy:** By wrapping `native_decide` in a search loop (e.g., checking indices $0..N$ sequentially), the retry loop can extract the **Index of Divergence**. This is high-value feedback for the pattern miner (see Section 6).
    

### 3.4 Stratum 4: Trace Ground Truth Failures

Here, the verification succeeds (the formula matches the trace), but the result is rejected by an external oracle or downstream validation because the trace itself was wrong.

- **Context:** The LLM hallucinated the sequence values in Stage 1.
    
- **Detection:** This requires external validation (checking against OEIS data or a second independent trace generator).
    

---

## 4. The Semantic Rift: Translation and Type Theory

One of the most persistent sources of error in this pipeline is the **Semantic Rift** between Python (the language of generation) and Lean (the language of verification). This rift is most treacherous in the domain of integer arithmetic.

### 4.1 The Integer Division Trap

In the domain of integer sequences, division is ubiquitous. However, "division" is not a universal concept across programming languages.

- **Python (`//`):** Performs **Floor Division** ($ \lfloor x/y \rfloor $). The result is always rounded towards $-\infty$.
    
    - Example: $-5 // 2 = -3$.
        
- **Lean (`/` on `Int`):** Performs **Truncated Division** ($ \text{sgn}(x/y) \lfloor |x/y| \rfloor $). The result is rounded towards $0$.
    
    - Example: `-5 / 2 = -2`.
        

**Impact on Verification:**

If an LLM mines a formula in Python involving negative numbers and division (e.g., a linear recurrence running backwards), and simply translates `//` to `/` in Lean, the formula will be mathematically distinct. `native_decide` will correctly return `false`, but the "error" is purely translational, not mathematical.

**Recovery Pattern:** The translation layer must detect division operations. If the operands are Integers (`Int`), the LLM must be prompted to use `Int.fdiv` (floor division) to match Python semantics, rather than the default `/` operator. If the operands are Naturals (`Nat`), the behavior is identical, but the type class `DivisionRing` will fail synthesis (as `Nat` is not a ring), requiring a cast to `Rat` or `Int`.

### 4.2 The Modulo Mismatch

Similarly, the modulo operator `%` differs in behavior regarding negative numbers.

- **Python (`%`):** Result has the same sign as the **divisor**.
    
    - $-5 \% 3 = 1$.
        
- **Lean (`%`):** Result has the same sign as the **dividend**.
    
    - $-5 \% 3 = -2$.
        

**Impact:**

This discrepancy is fatal for modular arithmetic sequences. A formula mined correctly in Python will fail verification in Lean.

**Recovery Pattern:**

The retry loop must rewrite modular operations using `Int.emod` (Euclidean modulo) in Lean, which guarantees non-negative remainders matching Python's behavior for positive divisors. The diagnostic system should parse the formula for `%` and automatically suggest this substitution if a logical failure occurs on a sequence involving negative terms.

### 4.3 Type Casting and Coercion Chains

LLMs often struggle with Lean's strict separation of `Nat` (non-negative) and `Int`.

- **Scenario:** A sequence is defined as $a_n = a_{n-1} - 5$.
    
- **Lean `Nat`:** If $a_{n-1} = 3$, then $3 - 5 = 0$ (saturation subtraction).
    
- **Lean `Int`:** $3 - 5 = -2$.
    
- **Failure:** If the LLM declares the sequence type as `Nat` but the logic requires negative intermediates, the verification will fail logically.
    
- **Recovery:** The Feedback Loop must analyze the error. If the error is a logical mismatch and subtraction is involved, the system should propose lifting the entire definition to `Int` or `Rat`.
    

---

## 5. Algorithmic Recovery: Mining Strategy Iteration

When the failure is truly mathematical (Stratum 3)—meaning the formula does not describe the trace—the system must iterate on the mining strategy itself. We employ a hierarchy of mining algorithms, moving from simple to complex.

### 5.1 The Primary Miner: Berlekamp-Massey

The Berlekamp-Massey (BM) algorithm is the gold standard for recovering **Linear Recurrence Relations**. It interprets the sequence as the output of a Linear Feedback Shift Register.

- **Mechanism:** It processes terms one by one. When a discrepancy occurs (the current recurrence fails to predict the next term), it updates the connection polynomial to correct the error while maintaining minimality.
    
- **Constraint:** BM requires the sequence to be strictly linear. If the sequence is "almost linear" or has a transient pre-period that behaves differently, BM will either produce a recurrence of degree $N/2$ (fitting noise) or fail.
    
- **Failure Signal:** If the degree of the mined recurrence is $> N/2$, the result is likely spurious.
    

### 5.2 The Secondary Miner: Rational Interpolation (Pade)

If BM fails, the sequence might be generated by a **Rational Function**, which can be represented as the ratio of two polynomials $P(x)/Q(x)$.

- **Algorithm:** Pade Approximants or the AAA algorithm.
    
- **Relation to BM:** There is a deep theoretical connection between BM and Pade approximants. BM can be viewed as an algorithm for computing the Pade approximant of the formal power series of the sequence. However, explicit rational interpolation (using Python libraries like `scipy.interpolate` or `sympy.pade`) is often more robust to "wiggles" or non-standard index offsets.
    
- **Implementation:** The retry loop triggers a Python worker to compute the Pade approximant of the sequence. If the resulting polynomials $P$ and $Q$ have small degrees relative to the trace length, this candidate is promoted to translation.
    

### 5.3 Advanced Diagnostics: The Hankel Matrix

To intelligently choose between BM and Pade, the system can compute the **Hankel Matrix** of the sequence (a matrix where $H_{i,j} = a_{i+j}$).

- **Heuristic:** The determinant of the Hankel matrix serves as a fingerprint. If the determinants of successive Hankel matrices vanish (become zero) after a certain size, the sequence satisfies a linear recurrence of that order.
    
- **Python Check:** Before running expensive verifications, the system calculates the Hankel determinants.
    
    - _Determinants are zero:_ High confidence in Linear Recurrence $\to$ Prioritize BM.
        
    - _Determinants grow regularly:_ Possible Rational structure $\to$ Prioritize Pade.
        
    - _Determinants are chaotic:_ Sequence is likely non-linear/non-rational $\to$ Fallback to TIR.
        

### 5.4 Fallback Miner: Symbolic Regression

If both linear and rational miners fail, the sequence may be non-linear (e.g., $a_n = a_{n-1}^2 + 1$).

- **Tools:** `gplearn` or specialized integer sequence miners like Diofantos.
    
- **Cost:** High. This is a search over expression trees.
    
- **Strategy:** Due to the 6-minute constraint, this is usually skipped in favor of TIR unless the trace shows strong non-linear characteristics (super-exponential growth).
    

---

## 6. The Retry State Machine: Logic and Control Flow

To manage these complex decisions within a tight time budget, we architect the retry loop as a **Finite State Machine (FSM)**. This agentic approach replaces rigid scripts with dynamic state transitions based on error feedback.

### 6.1 State Definitions and Transitions

We define the following states for the verification agent:

|**State**|**Description**|**Transition on Success**|**Transition on Failure**|
|---|---|---|---|
|**S0: GEN_TRACE**|LLM generates Python trace.|**S1: MINE**|**S0** (Retry generation)|
|**S1: MINE**|BM / Pade Algorithms.|**S2: TRANSLATE**|**S0** (Trace un-minable)|
|**S2: TRANSLATE**|LLM writes Lean code.|**S3: VERIFY**|**S4: REPAIR_SYNTAX**|
|**S3: VERIFY**|Run `native_decide`.|**DONE** (Success)|**S4** / **S5** / **S6** (Diagnose)|
|**S4: REPAIR_SYNTAX**|Fix Compilation/Type errors.|**S3: VERIFY**|**S4** (Loop) or **S6** (Escalate)|
|**S5: REFINE_MINING**|Logical failure $\to$ Adjust Miner.|**S2: TRANSLATE**|**S6: FALLBACK**|
|**S6: FALLBACK_TIR**|Python-based Verification.|**DONE** (Partial)|**FAIL**|

### 6.2 The Diagnostic Decision Logic (The "Brain")

When **S3: VERIFY** fails, the Diagnostic Logic routes the flow:

1. **Input:** Lean Log Output + Execution Metadata (Time).
    
2. **Parser:** Apply Regex from Section 3.1.
    
3. **Routing Rules:**
    
    - IF `Error == Syntax/Type` AND `Retries < 3` $\to$ **S4: REPAIR_SYNTAX**.
        
        - _Action:_ Send code + error to LLM. "Fix this type mismatch."
            
    - IF `Error == Timeout` AND `Heartbeats < Max` $\to$ **S3: VERIFY** (Retry).
        
        - _Action:_ Double `maxHeartbeats`, try again.
            
    - IF `Error == Timeout` AND `Heartbeats >= Max` $\to$ **S5: REFINE_MINING**.
        
        - _Action:_ The definition is too slow. Switch miner or reduce verification range.
            
    - IF `Error == Logic (False)` $\to$ **S5: REFINE_MINING**.
        
        - _Action:_ Extract counterexample index. Re-run BM with sub-trace ending before that index.
            
    - IF `Time_Remaining < 60s` $\to$ **S6: FALLBACK_TIR**.
        
        - _Action:_ Abort Lean. Generate Python verifier.
            

### 6.3 Resource Budgeting: The Simulated Annealing Schedule

We treat the 6-minute window as a computational budget that must be allocated dynamically.

- **Phase 1: Exploration (0s - 120s)**
    
    - _Strategy:_ High Temperature. Generate 3 distinct traces. Run fast miners (BM) on all. Try fast verification ($N=10$).
        
    - _Goal:_ Find the "path of least resistance."
        
- **Phase 2: Exploitation (120s - 300s)**
    
    - _Strategy:_ Low Temperature. Select the single most promising candidate (e.g., one that compiled but failed logically).
        
    - _Action:_ Allocate massive heartbeats. engage Syntax Repair loops. Attempt Pade interpolation.
        
- **Phase 3: Panic / Fallback (300s - 360s)**
    
    - _Strategy:_ Survival.
        
    - _Action:_ Switch entirely to TIR (Section 7). Generate a Python solution and verify it empirically. A Python solution is better than a timeout.
        

---

## 7. Fallback to Tool Integrated Reasoning (TIR)

When the rigor of Formal Verification becomes a blocker—either due to undecidability, compilation bugs, or time constraints—the system must degrade gracefully. **Tool Integrated Reasoning (TIR)** provides this safety net.

### 7.1 Defining TIR in this Context

TIR involves using an external tool (the Python interpreter) to augment the reasoning process. In our context, it means replacing the Lean kernel with the Python runtime as the "verifier."

### 7.2 The TIR Verification Loop

Instead of proving `theorem : ∀ n, f n = trace n`, we generate a Python script `verify.py`:

Python

```
def verify_candidate(formula_func, ground_truth_func, limit=10000):
    for n in range(limit):
        if formula_func(n)!= ground_truth_func(n):
            return False, n
    return True, limit
```

This script checks the formula against the ground truth (generated by the trace logic) for a much larger range than is feasible in Lean (e.g., $N=10,000$).

### 7.3 Confidence Scoring

While TIR does not provide mathematical certainty, it provides statistical certainty. We calculate a **Confidence Score** ($C$) for the user.

$$C = \min(1.0, \frac{\log(N_{verified})}{\log(N_{threshold})} \times (1 - \delta_{complexity}))$$

- $N_{verified}$: Number of terms matched (e.g., 10,000).
    
- $N_{threshold}$: Empirical threshold for integer sequence uniqueness (usually ~100-500).
    
- $\delta_{complexity}$: A penalty for formula complexity (a 100-term polynomial matching 100 points has low confidence; a 2-term recurrence matching 100 points has high confidence).
    

By outputting this score, the system allows the user to gauge the reliability of the result even in the absence of a green Lean checkmark.

---

## 8. Error-Driven Learning: Closing the Loop

A sophisticated agent does not just retry; it learns. **Error-Driven Learning** utilizes the rich signals from the compiler to improve the LLM's performance within the session and across sessions.

### 8.1 In-Context Learning (The "Repair" Prompt)

When `native_decide` fails, we do not simply ask the LLM to "try again." We construct a prompt that includes:

1. **The Code:** The failing Lean definition.
    
2. **The Error:** The exact compiler message (e.g., "type mismatch").
    
3. **The Context:** A retrieved example of a similar error being fixed (One-Shot Learning).
    

- _Effect:_ This transforms the LLM from a generator into a **debugger**. Research shows that providing compiler feedback significantly boosts "pass@k" rates for code generation.
    

### 8.2 Persistent Learning (Future Outlook)

While the current scope focuses on a single problem session, a production system should maintain a database of **Successful Repairs**.

- _Mechanism:_ If the system fixes a "DivisionRing" error by casting to `Int`, store this pair (Error $\to$ Fix).
    
- _Retrieval:_ On future errors, query this database to inject the "perfect fix" into the prompt context, effectively creating a self-improving "Auto-Formalization" capability.
    

---

## 9. Conclusion and Implementation Roadmap

The verification of mined integer sequences in Lean 4 is a problem of **Signal Management**. The "Reject" signal from the kernel is noisy and multi-modal. Success requires a system that parses this noise into specific failure taxonomies—distinguishing between the semantic mismatches of integer division, the resource exhaustion of recursive reductions, and the true falsity of a wrong hypothesis.

We have outlined a **Verification Retry Loop** that operates as a hierarchical State Machine. It prioritizes low-cost syntax repairs, escalates to sophisticated algorithmic adjustments (BM $\to$ Pade), and intelligently falls back to empirical verification (TIR) when formal proofs are intractable. By implementing this architecture, developers can maximize the "Time-to-Insight" within the 6-minute constraints, turning the fragility of `native_decide` into a robust engine for mathematical discovery.

### Summary of Recommendations

1. **Implement Diagnostic Parsing:** Use Regex to classify Lean output into `Syntax`, `Type`, `Resource`, and `Logic` buckets.
    
2. **Explicit Semantics:** Hardcode translation rules for `//` (Python) to `Int.fdiv` (Lean) to avoid semantic traps.
    
3. **Algorithmic Diversity:** Do not rely solely on linear recurrence; integrate Rational Interpolation (Pade) as a secondary miner.
    
4. **Resource annealing:** Use a tiered budget strategy, spending the first 2 minutes on exploration and the last 4 on deep verification/repair.
    
5. **TIR Fallback:** Always generate a Python verification script as a fail-safe, providing a confidence-scored answer when formal proof fails.
    

This report provides the theoretical and practical foundation for building the next generation of Neuro-Symbolic mathematical assistants.

---

## 10. Appendix: Data Tables and Reference Heuristics

### Table 1: Failure Mode Diagnosis Matrix

|**Error Class**|**Regex Signature**|**Root Cause**|**Recovery Horizon**|
|---|---|---|---|
|**Syntax**|`unexpected token`, `function expected`|LLM hallucination / Lean 3 syntax|**Horizon 1:** Prompt LLM with error for syntax fix.|
|**Type**|`failed to synthesize instance DivisionRing`|Use of `/` on `Nat` or `Int` without Ring structure|**Horizon 1:** Prompt LLM to cast to `Rat` or use `Int.fdiv`.|
|**Type**|`type mismatch.*expected.*Int.*got.*Rat`|Domain mixing|**Horizon 1:** Inject coercion `(n : Int)` or `Int.ofNat`.|
|**Resource**|`(deterministic) timeout at 'whnf'`|Calculation too heavy / Inefficient recursion|**Horizon 2:** Reduce test index $N$. Optimize definition (tail recursion).|
|**Resource**|`timeout at 'isDefEq'`|Type inference too complex|**Horizon 1:** Add explicit type annotations to intermediate terms.|
|**Logic**|`tactic 'native_decide' failed` (False)|Formula does not match trace|**Horizon 2:** Extract counterexample index. Rerun Miner (Pade).|
|**System**|`failed to compile definition`|C++ Gen bug / Nested `let rec`|**Horizon 1:** Rewrite recursion to be simple/structural.|

### Table 2: Mining Strategy Hierarchy

|**Algorithm**|**Complexity**|**Scope**|**Trigger Condition**|
|---|---|---|---|
|**Berlekamp-Massey**|$O(N^2)$|Linear Recurrences|Initial attempt. Hankel Det = 0.|
|**Rational Interpolation (Pade)**|$O(N^3)$|Rational Functions|BM fails. Hankel Det grows regularly.|
|**Modulus Search**|$O(N \cdot P)$|Modular Linear|Large numbers, timeouts.|
|**TIR (Python Check)**|$O(N)$|Empirical|Time < 60s or Logic Refutation.|

### Table 3: Python vs. Lean Semantic Map

|**Operation**|**Python Syntax**|**Python Semantics**|**Lean Syntax**|**Lean Semantics**|**Translation Fix**|
|---|---|---|---|---|---|
|**Division**|`a // b`|Floor ($\lfloor x/y \rfloor$)|`a / b` (Int)|Truncate ($\to 0$)|Use `Int.fdiv a b`|
|**Modulo**|`a % b`|Sign of Divisor|`a % b` (Int)|Sign of Dividend|Use `Int.emod a b`|
|**Exponent**|`a ** b`|Arbitrary|`a ^ b`|Nat/Nat|Cast to `Nat` for exp|
|**Bitwise**|`^`, `&`|Integers|`^^^`, `&&&`|BitVec / UInt|Cast to `BitVec`|