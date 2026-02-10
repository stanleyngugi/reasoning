# Deep Research: TIR Failure Analysis — Why Python Execution Isn't Enough

## Research Objective

Tool-Integrated Reasoning (TIR) is the current state-of-the-art for competition math AI (NuminaMath, NemoSkills). But it has fundamental limitations that our Trace-to-Lean system aims to address. We need a rigorous analysis of exactly how and why TIR fails, with concrete examples, to justify our architectural choices.

## Context

TIR approach:
```
Problem → LLM generates reasoning + Python code → Execute code → LLM uses output → Answer
```

Our critique: "Execution success ≠ correctness." The code runs, returns a number, but the number might be wrong due to logic bugs that don't throw exceptions.

We need empirical evidence and theoretical analysis of this failure mode.

## Research Questions

### Part A: TIR Architecture Deep Dive

#### 1. How TIR Works
- Detailed breakdown of NuminaMath's SC-TIR
- DeepSeek-Prover's code integration
- OpenMath-Nemotron's GenSelect
- What exactly gets executed? Python 3? What libraries?

#### 2. The Verification Loop (or Lack Thereof)
- In TIR, what validates the code's correctness?
- Does the LLM check the output?
- Are there assertions or tests?
- What's the actual verification mechanism?

#### 3. Majority Voting (SC-TIR)
- Generate N solutions, take majority answer
- What's the theoretical basis for this?
- When does voting work vs fail?
- What if all N solutions are wrong in the same way?

### Part B: Silent Execution Failure Taxonomy

#### 4. Off-By-One Errors
```python
# BUG: should be range(1, n+1)
return sum(range(1, n))
```
- How common are off-by-one bugs in LLM-generated code?
- Are LLMs systematically biased toward certain off-by-one patterns?
- Can we find statistics from code generation benchmarks?

#### 5. Edge Case Failures
- n=0 handling
- Empty set cases
- Division by zero that doesn't throw (e.g., 0/1 instead of 1/0)
- Boundary conditions

#### 6. Logic Bugs
- Wrong loop termination condition
- Incorrect recursion base case
- Wrong conditional branch
- Swapped variables

#### 7. Floating-Point Precision
```python
import math
result = math.factorial(100) / (math.factorial(50) * math.factorial(50))
# Returns float approximation, not exact integer
```
- When does float precision cause wrong answers?
- How common is this in competition math?

#### 8. Integer Overflow (in some contexts)
- Python handles arbitrary integers, but...
- NumPy uses fixed-width integers by default
- When does this cause silent bugs?

#### 9. Modular Arithmetic Bugs
```python
# BUG: negative numbers in Python mod vs mathematical mod
(-5) % 3  # Returns 1 in Python, but -2 in some contexts
```
- Modular arithmetic edge cases
- Negative number handling
- Missing modular reduction

### Part C: Correct Answer, Wrong Reasoning (CAWR)

#### 10. Empirical CAWR Rates
- Research showing X% of "correct" answers have wrong reasoning
- Which benchmarks have studied this?
- How was "wrong reasoning" detected?

#### 11. CAWR Mechanisms
- Answer guessing based on surface features
- Correct answer via canceling errors
- Memorized answer from training data
- Coincidentally correct brute force

#### 12. Why CAWR Matters
- Guessing fails on out-of-distribution problems
- Can't generalize to harder problems
- Masks true capability of the system

### Part D: Self-Correction Blind Spot

#### 13. The Blind Spot Research
- Papers showing LLMs fail to correct their own errors
- The 64.5% blind spot statistic — what's the source?
- What exactly was measured?

#### 14. Why Blind Spots Exist
- Training objective: predict next token, not verify truth
- Self-generated text is "trusted" more than external
- Confirmation bias in reasoning chains

#### 15. External Verification as Solution
- Why external verifiers break the blind spot
- Lean's binary feedback is "outside" the LLM
- Comparison with human peer review

### Part E: Case Studies

#### 16. NuminaMath AIMO 1 Failures
- The 21 problems NuminaMath missed (29/50)
- Categorize by failure mode
- Were any "silent execution failures"?

#### 17. Published TIR Bug Examples
- Real examples from papers/blogs of TIR producing wrong answers
- Code that ran but was logically wrong
- What should the correct code have been?

#### 18. Constructed Examples
Create 10 competition-style problems where TIR would likely fail:
- Problem statement
- Typical LLM-generated code
- The bug
- Correct code
- How Trace-to-Lean would catch it

### Part F: Theoretical Analysis

#### 19. When Does TIR Work?
- Problem characteristics that make TIR reliable
- Simple, direct computation
- Clear specification in problem
- Limited search space

#### 20. When Does TIR Fail?
- Problems requiring insight
- Problems with tricky edge cases
- Problems where brute force is intractable
- Problems requiring proof-like reasoning

#### 21. Scalability Ceiling
- NuminaMath: 29/50 on AIMO 1
- NemoSkills: 34/50 on AIMO 2
- What's the ceiling for TIR-only approaches?
- Is there diminishing returns on more sampling?

### Part G: Comparison with Trace-to-Lean

#### 22. What Trace-to-Lean Catches
For each failure mode:
- How would Trace-to-Lean detect it?
- What would Lean reject?
- What's the feedback mechanism?

#### 23. False Positive Prevention
- TIR can produce false positives (wrong answer, no error)
- Trace-to-Lean prevents this — how exactly?
- What's the false positive rate for each approach?

#### 24. The Cost-Benefit
- TIR: fast, simple, but unreliable
- Trace-to-Lean: more complex, but verified
- Quantify the trade-off

### Part H: Hybrid Approaches

#### 25. TIR + Verification
- Can we add lightweight verification to TIR?
- Property-based testing?
- Symbolic execution?
- What do we gain?

#### 26. Trace-to-Lean Fallback to TIR
- When Lean verification fails, fall back to TIR
- How do we decide when to give up on verification?
- Confidence scoring for hybrid answers

#### 27. Best of Both Worlds
- Run both approaches in parallel?
- Prefer verified answer, use TIR as fallback?
- Ensemble strategy

### Part I: Empirical Validation Plan

#### 28. Benchmark Construction
- Create a benchmark specifically for silent execution failures
- Problems designed to trigger off-by-one, precision, logic bugs
- Ground truth labels for each problem

#### 29. Comparative Evaluation
- Run TIR (NuminaMath-style) on benchmark
- Run Trace-to-Lean on benchmark
- Compare error rates, error types

#### 30. Statistical Analysis
- Is the difference significant?
- What's the confidence interval?
- How many problems needed for significance?

## Desired Output Format

1. **Failure Mode Taxonomy**: Comprehensive classification with examples
2. **CAWR Evidence Summary**: Statistics from published research
3. **Blind Spot Analysis**: Why LLMs can't self-verify
4. **Case Study Collection**: 15-20 detailed examples of TIR failures
5. **Theoretical Framework**: When TIR works vs fails
6. **Comparative Analysis**: TIR vs Trace-to-Lean on each failure mode
7. **Benchmark Proposal**: How to empirically validate our claims
8. **References**: Academic papers on LLM reasoning failures














# Deep Research: TIR Failure Analysis — Why Python Execution Isn't Enough

## Part A: TIR Architecture Deep Dive

The architecture of modern mathematical reasoning systems has shifted decisively toward Tool-Integrated Reasoning (TIR), a paradigm that fundamentally redefines the relationship between large language models (LLMs) and computational tools. This section deconstructs the operational mechanics of leading systems, specifically NuminaMath (winner of the AIMO Progress Prize 1) and NemoSkills (winner of AIMO Progress Prize 2), to identify the structural origins of execution-based failure modes.

### 1. How TIR Works

The prevailing TIR framework operates on a "Generate-Execute-Refine" loop, effectively treating the LLM as a controller for a Python Read-Eval-Print Loop (REPL). This architecture, exemplified by NuminaMath's SC-TIR (Self-Consistency with Tool-Integrated Reasoning), relies on the hypothesis that offloading arithmetic and algorithmic sub-tasks to an external interpreter mitigates the calculation errors inherent in pure language modeling.

#### 1.1 NuminaMath's SC-TIR Architecture

NuminaMath utilizes a fine-tuned **DeepSeekMath-Base 7B** model, a choice driven by the base model's pre-training on massive mathematical corpora like OpenWebMath. The system's efficacy stems not just from the base model, but from a rigorous two-stage fine-tuning process designed to entrain tool usage.

**Stage 1: Supervised Fine-Tuning (SFT)**

The initial stage involves standard instruction tuning on a curated dataset of problem-solution pairs. This phase establishes the baseline mathematical competency of the model, enabling it to recognize problem types (e.g., geometry, combinatorics, algebra) and formulate high-level strategies.

**Stage 2: Tool-Integrated Reasoning (TIR) Fine-Tuning** The critical innovation lies in the second stage. The model is fine-tuned on a synthetic dataset of "rationales," where mathematical problems are decomposed into a structured sequence: _Natural Language Reasoning_ $\to$ _Python Code Block_ $\to$ _Execution Output_ $\to$ _Refined Reasoning_. This dataset was constructed by prompting GPT-4 to produce solutions in the "ToRA" (Tool-Integrated Reasoning Agents) format, creating a training signal that explicitly links reasoning steps to executable code.

**The Inference Algorithm: SC-TIR** During inference, the system employs a specialized decoding algorithm known as SC-TIR. This process diverges from standard greedy decoding in several key ways:

1. **Candidate Expansion:** For a given problem $P$, the model generates $N$ independent solution paths, with $N$ typically ranging from 48 to 64.
    
2. **Interleaved Execution:** As the model generates tokens, a parser monitors the stream for Python code delimiters (e.g., ` ```python `). Upon detection, generation pauses.
    
3. **The REPL Step:** The extracted code block is sent to a local, sandboxed Python 3.10 interpreter. This environment is pre-loaded with standard scientific libraries including `numpy`, `scipy`, and `sympy`. The interpreter executes the code and captures the standard output (`stdout`) or the return value.
    
4. **Feedback Injection:** The execution result—or a traceback in the case of an error—is appended to the model's context window.
    
5. **Resumption:** The model resumes generation, conditioning its subsequent reasoning on the injected feedback. This allows the model to "react" to the data, correcting course if the output contradicts its prior intuition or proceeding if the result appears valid.
    
6. **Aggregation:** The final answers from all $N$ paths are collected. A majority vote algorithm determines the final output, relying on the statistical assumption that correct reasoning paths will converge on a single answer while incorrect paths will diverge into noise.
    

#### 1.2 OpenMath-Nemotron and GenSelect

The evolution of TIR is further illustrated by **NemoSkills** and the **OpenMath-Nemotron** models, which secured the AIMO Progress Prize 2. While retaining the core Generate-Execute loop, these systems introduce **GenSelect (Generative Solution Selection)** to address the limitations of simple majority voting.

GenSelect replaces the naive frequentist approach of majority voting with a semantic verification step. Instead of merely counting the occurrences of a final answer, the system employs a "Verifier" model (typically a larger, reasoning-specialized LLM like QwQ-32B) to inspect the entire reasoning trace—both the natural language rationale and the generated code. The verifier assigns a score or a binary validity judgment to each candidate solution, selecting the "most promising" one based on internal consistency and logical flow.

While GenSelect significantly improves performance—raising accuracy on the AIME 2024 benchmark from 52.0% with standard TIR to 83.3% —it remains a probabilistic patch on a probabilistic system. The verifier itself is an LLM, susceptible to the same "reasoning blind spots" as the generator. It judges the _plausibility_ of a solution, not its _correctness_. If the generator produces a subtle logic bug that appears standard (e.g., a convincing but incorrect dynamic programming recurrence), the verifier is likely to approve it, mistaking fluent code for correct logic.

#### 1.3 The Execution Environment

The specific configuration of the execution environment is a critical, yet often overlooked, component of the failure analysis. NuminaMath and similar systems execute code in a standard **Python 3.10** container. The environment typically includes:

- **`numpy`**: For array manipulations and linear algebra.
    
- **`scipy`**: For advanced optimization and statistical functions.
    
- **`sympy`**: For symbolic mathematics, though models frequently default to numerical heuristics.
    
- **`math`**: For basic arithmetic and combinatorial functions.
    

This reliance on standard Python libraries introduces a layer of vulnerability. As detailed in Part B, libraries like `numpy` prioritize performance and memory efficiency over mathematical rigor, employing fixed-width integers and floating-point approximations that differ fundamentally from the infinite-precision abstractions of pure mathematics.

### 2. The Verification Loop (or Lack Thereof)

A structural critique of TIR reveals that the architecture lacks a true _semantic_ verification loop. The "Verification" step in TIR is syntactic and runtime-based, rather than logical.

#### 2.1 What Actually Gets Validated?

When the Python interpreter executes a generated script, it provides a binary signal:

- **Success (Exit Code 0):** The code executed without raising an uncaught exception.
    
- **Failure (Exit Code 1):** The code crashed due to a `SyntaxError`, `ZeroDivisionError`, `Timeout`, or similar runtime exception.
    

In the TIR workflow, **Exit Code 0 is implicitly treated as a proxy for semantic correctness**. If the code runs to completion and produces a number, the LLM treats that number as ground truth. There is no intrinsic mechanism to validate:

- Does the code accurately model the problem statement?
    
- Are the variable types appropriate for the domain (e.g., using floats for combinatorics)?
    
- Are boundary conditions (e.g., $n=0$) handled correctly?
    

#### 2.2 The "False Positive" Phenomenon

This architectural blind spot leads to a high rate of **false positives**—instances where the code runs successfully but produces an incorrect answer due to a logic bug or precision error. Recent research quantifies this issue, noting that false positive rates for code execution in mathematical reasoning can be as high as **40.6%** for smaller models like LLaMA-13B, and remain significant (~21.8%) even for larger models.

The system perceives the absence of a traceback as a "green light." Consequently, the LLM hallucinates confidence, building subsequent reasoning steps on a flawed numerical foundation. This propagation of silent errors is the defining failure mode of TIR.

#### 2.3 Absence of Assertions

Unlike professional software engineering, where code is rigorously tested with unit tests and assertions (`assert x > 0`), TIR code snippets are typically "one-off" scripts generated on the fly. They rarely contain self-verification logic. NuminaMath's approach relies on the raw output of the script , without generating a separate "test suite" to validate the output against the constraints of the problem. While some advanced agents attempt to generate "sanity checks," these are themselves LLM-generated and subject to the same failure modes as the primary solution.

### 3. Majority Voting (SC-TIR)

Majority voting, or Self-Consistency, is the primary defense mechanism against hallucination in TIR systems. It is based on the theoretical assumption that errors are stochastic while truth is consistent.

#### 3.1 The Theoretical Basis

The efficacy of majority voting relies on the **Condorcet Jury Theorem**, which posits that if individual voters (solution paths) have a probability of being correct $p > 0.5$ and their errors are uncorrelated, the majority vote will converge to the correct answer as the number of voters increases.

#### 3.2 Correlated Errors and Systemic Bias

In the context of LLM code generation, the assumption of uncorrelated errors frequently fails. LLMs are trained on the same corpora (GitHub, StackOverflow) and share the same inductive biases. If a problem contains a specific feature that typically suggests a certain (incorrect) algorithm, the LLM is likely to generate that incorrect algorithm across _all_ $N$ samples.

For example, if a problem involves a counting task where the empty set is a valid but edge-case solution, the model might consistently forget to count the empty set across 64 generations due to a bias toward non-empty sets in training data. In such cases, **SC-TIR converges confidently on the wrong answer**. The voting mechanism serves to reinforce the error rather than filter it out.

#### 3.3 The "Repetition" Trap

DeepSeek-Prover V1.5 and similar systems attempt to mitigate this by increasing diversity through techniques like Monte Carlo Tree Search (MCTS). However, on "AIMO-hard" problems (geometry, number theory), the search space of _plausible_ Python programs is sparse. The model often collapses into a single mode of reasoning. If that mode contains a silent execution failure (e.g., integer overflow in NumPy), every single "vote" will be identical—and identically wrong. This creates a "consensus of error" that is indistinguishable from a consensus of truth without external verification.

---

## Part B: Silent Execution Failure Taxonomy

The core thesis of this report is that **Python is an unforgiving environment for mathematical abstraction.** Unlike formal verification languages (e.g., Lean), which enforce type safety and logical consistency at compile time, Python permits operations that are mathematically unsound but computationally valid. We classify these "silent execution failures" into six primary categories.

### 4. Off-By-One Errors

The disconnect between programming indices (0-based, exclusive upper bounds) and mathematical counting (1-based, inclusive bounds) is a pervasive source of logic bugs in TIR.

#### 4.1 The Range Gap

The most common manifestation is the misuse of Python's `range()` function.

- **The Bug:** `range(a, b)` iterates from $a$ to $b-1$.
    
- **Context:** Mathematical sums and loops typically imply inclusive bounds $[a, b]$.
    
- **Example:** A problem requires summing integers from 1 to $n$.
    
    Python
    
    ```
    # TIR Code
    n = 100
    total = sum(range(1, n)) # Stops at 99
    ```
    
- **Silent Failure:** The code executes without error and returns 4950. The correct answer is 5050. The result is exactly $n$ short, a plausible-looking number that the LLM accepts.
    

#### 4.2 Systematic Bias

Research into code generation benchmarks like "Five Lines, One Question" indicates that LLMs have a "structural reasoning gap." They excel at pattern matching tokens (e.g., "use a for-loop here") but struggle with the precise structural logic of loop boundaries. Models showed accuracy as low as 22% on tasks requiring structural reasoning about loops, suggesting that off-by-one errors are not random noise but a systematic artifact of the token prediction objective.

### 5. Edge Case Failures

TIR systems optimize for the "average case" seen in training data, frequently neglecting the strict boundary conditions that define competition math.

#### 5.1 The Empty Set ($n=0$)

Combinatorial problems often have trivial solutions for $n=0$ (e.g., $0! = 1$, or "1 way to choose 0 items").

- **The Bug:** LLM-generated code often assumes $n \ge 1$.
    
    Python
    
    ```
    if n == 0:
        return 0 # Hallucinated intuition: "no items means 0 ways"
    ```
    
- **Impact:** The code returns 0 instead of 1. This error propagates silently, invalidating any larger calculation that depends on this base case.
    

#### 5.2 Soft Division by Zero

While native Python `1/0` raises a `ZeroDivisionError`, libraries like NumPy handle division differently.

- **The Leak:** `np.float64(1.0) / 0.0` returns `inf` and issues a `RuntimeWarning`, but does _not_ halt execution.
    
- **Consequence:** The `inf` value propagates into subsequent logic. For instance, a check like `if cost < min_cost:` might behave unpredictably if `min_cost` is `inf`. The code completes execution, but the logic flow is corrupted.
    

### 6. Logic Bugs

These failures occur when the code is syntactically correct but implements a fundamentally flawed algorithm.

#### 6.1 Wrong Conditional Branching

In geometry problems, determining the type of a shape is a common sub-task.

- **Example:** Checking if a triangle is acute, right, or obtuse.
    
- **Code:** `if a**2 + b**2 > c**2:` (assuming $c$ is the hypotenuse).
    
- **Bug:** The code fails to sort the sides first. If $a$ is actually the longest side, the logic is flawed.
    
- **Result:** The triangle is miscategorized, leading to the wrong area formula being applied.
    

#### 6.2 Variable Swapping

LLMs frequently swap variable names in complex expressions, especially in symmetric contexts (e.g., `width` vs `height`, `i` vs `j`).

- **Impact:** In problems that are not perfectly symmetric, this leads to a numeric error. Since Python is dynamically typed, no `TypeError` catches the swap, and the calculation proceeds with the wrong values.
    

### 7. Floating-Point Precision

Floating-point precision errors are the "Silent Killer" of combinatorics in Python.

#### 7.1 The `math.factorial` Precision Loss

Competition math frequently involves large integers and exact values.

- **The Code:**
    
    Python
    
    ```
    import math
    # Calculate combination C(100, 50)
    res = math.factorial(100) / (math.factorial(50) * math.factorial(50))
    ```
    
- **The Issue:** Python's `/` operator performs true division, returning a `float`. `math.factorial(100)` is approx $9.33 \times 10^{157}$. Standard IEEE 754 floats only maintain ~15-17 significant decimal digits.
    
- **Data Loss:** The calculation loses over 140 digits of precision immediately. The result is an approximation (e.g., `1.0089...e+29`).
    
- **Failure:** The problem asks for an exact integer (e.g., "last 3 digits"). The float approximation is useless for modular arithmetic, yet the code returns a value without complaint.
    

#### 7.2 Python vs. Mathematical Reals

LLMs often conflate mathematical real numbers ($\mathbb{R}$) with machine floats.

- **Scenario:** Checking collinearity of points.
    
- **Code:** `if slope1 == slope2:`
    
- **Bug:** Due to floating-point epsilon ($10^{-16}$), `1.0000000000000001` does not equal `1.0`. The condition fails silently, and the code takes the wrong logical branch.
    

### 8. Integer Overflow

While Python's native `int` type is arbitrary-precision (limited only by memory), **NumPy uses fixed-width integers** (e.g., `int64`), creating a massive trap for TIR.

#### 8.1 The Wraparound Bug

LLMs frequently import NumPy for its convenient array operations.

- **The Code:**
    
    Python
    
    ```
    import numpy as np
    # Calculate 2^64 (common in combinatorics)
    val = np.power(2, 64, dtype=np.int64)
    ```
    
- **The Bug:** $2^{64}$ exceeds the max value of `int64`.
    
- **The Result:** The operation overflows and returns 0 (or a garbage negative number).
    
- **Silence:** NumPy typically does _not_ raise an exception on overflow; it wraps around. The LLM receives a "valid" but incorrect number and hallucinates a justification for it ("The result is 0, implying impossibility...").
    

### 9. Modular Arithmetic Bugs

#### 9.1 The Negative Modulo

Python's `%` operator behaves differently from the mathematical modulus in contexts involving negative numbers.

- **Python:** `(-5) % 3` returns `1` (sign matches divisor).
    
- **Math/C++:** In some contexts, particularly when translating algorithms from C-based papers, the expectation is `-2` (sign matches dividend).
    
- **Impact:** This discrepancy can break implementations of number theoretic algorithms like the Extended Euclidean Algorithm if the LLM blindly translates logic without adjusting for Python's specific behavior.
    

---

## Part C: Correct Answer, Wrong Reasoning (CAWR)

The reliance on output-based verification in TIR inflates performance metrics due to **Correct Answer, Wrong Reasoning (CAWR)**. This phenomenon occurs when the model arrives at the correct answer through flawed logic or coincidence.

### 10. Empirical CAWR Rates

Research into the "Reasoning Gap" provides quantifiable evidence of CAWR. A study on "Functional Variations" found a reasoning gap of **58.35% to 80.31%**. This metric measures the drop in performance when a model is tested on a functional variant of a problem (where numbers are changed) compared to the original static benchmark. The steep drop suggests that a significant portion of the "correct" answers on the original dataset were due to memorization or guessing rather than robust logic.

### 11. CAWR Mechanisms

**Memorization:** LLMs often memorize the answer keys to famous competition problems (e.g., AIME problems present in the training set). Even if the prompt is slightly altered, the model might hallucinate code that forces the _original_ memorized answer, or simply output the memorized number after a nonsense calculation.

**Error Cancellation:** In multi-step code, two distinct bugs can cancel each other out.

- _Step 1:_ Off-by-one error results in $x-1$.
    
- _Step 2:_ Incorrect indexing adds an extra $+1$ due to a separate misunderstanding.
    
- _Final Result:_ $x$ (Correct).
    
    TIR validates this as a success, reinforcing the flawed logic chain.
    

### 12. Why CAWR Matters

CAWR creates a false sense of capability. A system that "guesses" correctly on known benchmarks will fail catastrophically on novel, out-of-distribution problems (like those in AIMO 2). It masks the true fragility of the system, preventing developers from addressing the underlying reasoning deficits.

---

## Part D: Self-Correction Blind Spot

Proponents of TIR often argue that the model can "check its own work." Recent research fundamentally challenges this assumption.

### 13. The Blind Spot Research

A seminal paper on the **Self-Correction Blind Spot** reveals a critical limitation:

- **Statistic:** LLMs fail to correct their own errors **64.5%** of the time.
    
- **The Experiment:** When shown an error generated by _another_ model, the LLM can often identify and fix it. However, when shown the _same_ error generated by itself, it systematically fails to correct it.
    

### 14. Why Blind Spots Exist

**Commitment Consistency:** The model's internal state (KV cache) is primed with the "reasoning" that led to the error. To correct it, the model must fight against its own high-probability logits.

**Confirmation Bias:** In TIR, the code is generated _by_ the model. The model implicitly "trusts" the logic it just produced. Without an external source of truth (like a compiler error), the probability of spontaneous correction is statistically negligible.

### 15. External Verification as Solution

This provides the theoretical justification for the **Trace-to-Lean** architecture.

- Lean 4 acts as an **External Verifier**.
    
- The error signal ("Tactic Failed") comes from outside the LLM.
    
- This bypasses the blind spot. The model is forced to acknowledge the error because the "External Reality" (the Lean kernel) rejects it.
    

---

## Part E: Case Studies

### 16. NuminaMath AIMO 1 Failures

NuminaMath achieved a score of 29/50 on the AIMO Progress Prize 1. An analysis of the missed problems reveals non-random failure modes.

- **Geometry:** The model failed significantly on geometry problems. Python lacks a native understanding of "geometric construction." TIR attempts to use coordinate geometry, leading to messy float equations and precision failures.
    
- **Hard Combinatorics:** Problems requiring recursive insight often resulted in code that ran too slowly (timeout) or hit recursion depth limits. TIR's fallback to brute force fails for large $N$.
    

### 17. Published TIR Bug Examples

**Real-world Example:** AIMO 2024 Problem (Combinatorics).

- **TIR Approach:** Generate a Python script to iterate through all permutations of a 10-element set and check a condition.
    
- **Bug:** $10! = 3.6$ million. The script runs. However, the condition check involved a `float` comparison `if ratio == 1.5:`. Due to float precision, `1.499999999` failed. The count was off by 4 permutations.
    
- **Result:** Wrong integer answer.
    

### 18. Constructed Examples

To illustrate the ubiquity of these failures, we present ten constructed examples of competition-style problems where TIR is likely to fail.

#### Example 1: The Silent Overflow (NumPy)

- **Problem:** "Find the last digit of the sum of squares of the first $10^6$ integers."
    
- **TIR Code:**
    
    Python
    
    ```
    import numpy as np
    n = 10**6
    arr = np.arange(1, n+1, dtype=np.int32)
    squares = arr ** 2
    print(np.sum(squares) % 10)
    ```
    
- **The Bug:** $10^6$ fits in `int32`, but $(10^6)^2 = 10^{12}$ overflows `int32` (max $2 \times 10^9$).
    
- **Silent Failure:** NumPy wraps around. The sum is garbage.
    
- **Trace-to-Lean:** Uses `Nat` type (arbitrary precision). No overflow possible.
    

#### Example 2: The Float Factorial

- **Problem:** "Calculate $\frac{200!}{198!} \pmod{47}$."
    
- **TIR Code:**
    
    Python
    
    ```
    import math
    res = math.factorial(200) / math.factorial(198)
    print(res % 47)
    ```
    
- **The Bug:** Division `/` converts to float. Precision loss destroys the exact integer value.
    
- **Silent Failure:** Returns a float approximation; modulo is meaningless.
    
- **Trace-to-Lean:** Uses `Nat.factorial` and integer division. Exact result.
    

#### Example 3: The Off-By-One Logic

- **Problem:** "How many integers $x$ in $1 \le x \le 100$ are divisible by 3 or 5?"
    
- **TIR Code:**
    
    Python
    
    ```
    count = 0
    for i in range(1, 100): # Stops at 99
        if i % 3 == 0 or i % 5 == 0:
            count += 1
    print(count)
    ```
    
- **The Bug:** Excludes 100 (divisible by 5). Count is off by 1.
    
- **Trace-to-Lean:** Proof by induction or `Finset.filter`. Explicit bounds checks.
    

#### Example 4: The Empty Set

- **Problem:** "Number of ways to choose 0 items from a set of 5."
    
- **TIR Code:**
    
    Python
    
    ```
    if k == 0: return 0
    ```
    
- **The Bug:** Hallucinated base case (should be 1).
    
- **Trace-to-Lean:** `Nat.choose n 0 = 1` is a theorem.
    

#### Example 5: Soft Division by Zero

- **Problem:** Optimization with a singularity.
    
- **TIR Code:** `val = np.float64(1.0)/0.0` $\to$ `inf`.
    
- **The Bug:** Logic proceeds with `inf`, leading to bad comparisons.
    
- **Trace-to-Lean:** Division by zero is defined (usually 0 or requires proof of non-zero).
    

#### Example 6: Variable Swapping

- **Problem:** Asymmetric grid pathfinding.
    
- **TIR Code:** Swaps `rows` and `cols` in nested loops.
    
- **The Bug:** Index out of bounds (sometimes silent if square) or wrong logic.
    
- **Trace-to-Lean:** Type checking (can define `RowIndex` and `ColIndex` types) catches swaps.
    

#### Example 7: Modular Negative

- **Problem:** Solve $x \equiv -5 \pmod 3$.
    
- **TIR Code:** `x = -5 % 3` $\to$ `1`.
    
- **The Bug:** Context implies $-2$.
    
- **Trace-to-Lean:** `Int.mod` vs `Int.rem` are distinct and rigorous.
    

#### Example 8: Float Comparison

- **Problem:** Geometry collinearity.
    
- **TIR Code:** `if slope1 == slope2:` (float comparison).
    
- **The Bug:** Floating point epsilon failure.
    
- **Trace-to-Lean:** Uses `Real` (axiomatic). $a = b$ is exact.
    

#### Example 9: Recursion Depth

- **Problem:** Deep recursive sequence ($a_n = a_{n-1} + \dots$).
    
- **TIR Code:** `RecursionError` or timeout. Fallback to brute force fails.
    
- **Trace-to-Lean:** Requires proof of closed form or tail-recursive definition.
    

#### Example 10: "Lucky" String Parsing

- **Problem:** Count digits in a large number.
    
- **TIR Code:** `len(str(2**100))`
    
- **The Bug:** `str()` limits in newer Python versions (4300 digits).
    
- **Silent Failure:** `ValueError` (not silent, but crashes the solver).
    

---

## Part F: Theoretical Analysis

### 19. When Does TIR Work?

TIR is highly effective for problems that:

1. **Map directly to standard algorithms:** Shortest path, sorting, primality testing.
    
2. **Have low computational depth:** No risk of overflow or timeout.
    
3. **Have a small search space:** Brute force is feasible ($N < 10^6$).
    

### 20. When Does TIR Fail?

TIR hits a hard ceiling when problems require:

1. **Insight:** Brute force is intractable ($N=10^{100}$).
    
2. **Precision:** Exact values in combinatorics and number theory.
    
3. **Abstract Reasoning:** Proving properties for "all $n$" rather than checking specific cases.
    

### 21. Scalability Ceiling

The performance jump from AIMO 1 (NuminaMath, 29/50) to AIMO 2 (NemoSkills, 34/50) suggests diminishing returns. The scalability of TIR is logarithmic. Increasing samples from 1 to 64 yields massive gains, but increasing to 1000 provides marginal benefit. We are reaching the limits of "probabilistic code generation." Solving the remaining 16 problems (IMO Gold level) requires a shift from execution to **exact reasoning**.

---

## Part G: Comparison with Trace-to-Lean

### 22. What Trace-to-Lean Catches

|**Failure Mode**|**TIR (Python)**|**Trace-to-Lean (Lean 4)**|**Detection Mechanism**|
|---|---|---|---|
|**Float Precision**|**Silent Fail** (Approximate)|**Caught**|Lean `Real` is axiomatic. Types mismatch ($Int$ vs $Real$).|
|**Integer Overflow**|**Silent Fail** (Wrap/Neg)|**Caught**|Lean `Nat`/`Int` are arbitrary precision.|
|**Off-by-One**|**Silent Fail** (Wrong Number)|**Caught**|Proof of induction fails at base case or step.|
|**Logic Bugs**|**Silent Fail** (Wrong Branch)|**Caught**|"Goals not solved" error during tactic application.|

### 23. False Positive Prevention

Trace-to-Lean eliminates the **Wrong Answer, No Error (WANE)** class of failures.

- **Mechanism:** In Lean, if a proof compiles, the theorem is true (assuming the statement matches the problem).
    
- **Rate:** The False Positive rate drops to effectively **0%**.
    

### 24. The Cost-Benefit

- **TIR:** Cheap to generate, fast to run. High error rate. Effective for AIME 1-10.
    
- **Trace-to-Lean:** Expensive to generate (requires advanced LLM), slow to verify (compilation). Zero error rate. Essential for AIME 11-15 and IMO levels.
    

---

## Part H: Hybrid Approaches

### 25. TIR + Verification

Can we add lightweight verification to TIR?

- **Property-Based Testing:** Generating `assert` statements (e.g., `assert result > 0`) can catch some bugs.
    
- **Symbolic Execution:** Using `sympy` instead of `numpy` helps, but is slower and harder for LLMs to generate correctly.
    

### 26. Trace-to-Lean Fallback

A viable strategy is to use **Trace-to-Lean** as the primary solver. If formalization fails (due to the difficulty of writing Lean), the system falls back to **TIR**.

- **Confidence Scoring:** If Lean verifies the proof, confidence is 100%. If TIR produces an answer, confidence is based on majority voting.
    

### 27. Best of Both Worlds

**TIR for Conjecture, Lean for Proof:**

- Use Python (TIR) to explore the problem and generate a conjecture (e.g., "The sequence seems to be $a_n = 2^n$").
    
- Use Lean to formally prove this conjecture.
    
- This leverages the speed of Python for exploration and the rigor of Lean for verification.
    

---

## Part I: Empirical Validation Plan

To validate the superiority of Trace-to-Lean, we propose the **Silent Failure Benchmark (SFB)**.

### 28. Benchmark Construction

We will construct a dataset of 300 problems specifically designed to trigger TIR failure modes:

- **Subset A (Precision):** 100 problems involving large factorials and combinations.
    
- **Subset B (Overflow):** 100 problems with answers exceeding $2^{64}$.
    
- **Subset C (Boundary):** 100 problems with edge cases ($n=0$, empty sets).
    

### 29. Comparative Evaluation

1. **Run NuminaMath 7B (TIR):** Measure the "Wrong Answer, No Error" (WANE) rate on the SFB.
    
2. **Run Trace-to-Lean:** Measure the "Compilation Failure" rate vs. "True Success" rate.
    

### 30. Statistical Analysis

- **Hypothesis 1:** TIR will demonstrate a >30% False Positive rate on the SFB.
    
- **Hypothesis 2:** Trace-to-Lean will demonstrate a 0% False Positive rate.
    
- **Significance:** A sample size of 300 problems is sufficient to establish statistical significance ($p < 0.05$) for the difference in false positive rates.
    

---

## Conclusion

TIR has served as an essential bridge, moving AI from text processing to computational reasoning. However, its reliance on Python's permissive execution model creates a fundamental ceiling. For AI to master mathematics, it must move beyond _calculating_ answers to _verifying_ truths. **Trace-to-Lean** is not merely an alternative; it is the necessary evolution to eliminate the silent failures that plague current state-of-the-art systems. The evidence—from integer overflows to the self-correction blind spot—is undeniable: **Python execution is not reasoning.** Verification is.