# Deep Research: Berlekamp-Massey Algorithm — Complete Mastery

## Research Objective

Achieve complete understanding of the Berlekamp-Massey algorithm for use in a competition math pipeline where it serves as the primary deterministic pattern miner for integer sequences. We need to know every edge case, every failure mode, every extension, and every optimization.

## Context

We are building a system where:
1. An LLM generates Python code that computes f(n) for n=1..k (typically k=15-30)
2. The output sequence is fed to Berlekamp-Massey to find the minimal linear recurrence
3. The recurrence is translated to Lean 4 and verified via `native_decide`

B-M is our Tier 1 pattern miner. Its success rate directly determines system performance on combinatorics and number theory problems (~50% of competition math).

## Research Questions

### 1. Core Algorithm Mechanics
- What is the precise mathematical guarantee of Berlekamp-Massey? Under what conditions is it 100% correct?
- How many terms are required to guarantee finding a degree-k recurrence? (I've seen 2k cited — verify this and explain why)
- What is the algorithm's behavior when given a sequence that is NOT linearly recurrent? Does it return a "best fit" or fail explicitly?
- What happens with sequences that have multiple valid recurrences of the same length?

### 2. Field Arithmetic Requirements
- B-M requires field arithmetic (multiplicative inverses). What exactly fails when working over Z_n for composite n?
- Explain the Reeds-Sloane extension for sequences over Z_n (composite moduli). How does it work? What's the complexity? Are there implementations available?
- For competition math, sequences are often over Z (integers) or Z_p (prime moduli). What's the recommended approach for each?
- How do we handle sequences with very large integers (10^100+)? Does B-M work with arbitrary-precision integers?

### 3. Practical Failure Modes
- What sequences commonly appear in competition math that B-M CANNOT handle?
  - Polynomial sequences (n², n³, binomial coefficients)
  - Exponential sequences (2^n, but note 2^n alone satisfies a(n) = 2*a(n-1))
  - Factorial sequences
  - Catalan numbers (are these linearly recurrent?)
  - Partition numbers
- For each failure case, explain WHY B-M fails and what alternative should be used

### 4. Numerical Stability
- When using B-M over rationals or reals (not modular arithmetic), what precision issues arise?
- How does floating-point error accumulate during the algorithm?
- What's the recommended approach: exact rational arithmetic, or modular arithmetic with CRT reconstruction?

### 5. Detecting Linear Recurrence
- Before running B-M, can we quickly test whether a sequence IS linearly recurrent?
- What heuristics indicate a sequence is likely to have a short linear recurrence?
- How can we estimate the degree of the recurrence from partial data?

### 6. From Recurrence to Closed Form
- Once B-M gives us coefficients [c1, c2, ..., ck] for a(n) = c1*a(n-1) + ... + ck*a(n-k), how do we:
  - Compute a(10^18) efficiently? (Matrix exponentiation? Polynomial exponentiation?)
  - Convert to a closed-form expression (if possible)?
  - Handle cases where the characteristic polynomial has repeated roots?
- What's the complexity of computing a(N) given the recurrence of length k? O(k³ log N) for matrix exp, or O(k² log N) for polynomial exp?

### 7. Implementation Details
- What is the best Python implementation of B-M for competition use?
- Does SymPy's `find_linear_recurrence` use B-M internally? What are its limitations?
- Are there optimized implementations using FFT for O(n log n) instead of O(n²)?
- What's the memory footprint for sequences of length 1000? 10000?

### 8. Competition Math Specific
- Survey AIME/IMO problems from 2015-2025 that involve integer sequences
- What percentage have solutions that are linearly recurrent?
- Are there competition problems where B-M would find a recurrence but it's "coincidental" (breaks at larger n)?
- What's the typical recurrence degree for competition problems? (Most seem to be degree 2-5)

### 9. Extensions and Alternatives
- What is the Berlekamp-Massey-Sakata algorithm for multidimensional sequences?
- How does B-M relate to Prony's method for exponential fitting?
- What's the connection between B-M and the Extended Euclidean Algorithm?
- Are there quantum or approximate versions for noisy sequences?

### 10. Integration with Lean Verification
- Once we have the recurrence a(n) = c1*a(n-1) + ... + ck*a(n-k), what's the most efficient Lean representation?
- How do we verify the recurrence for n=1..100 when we need the first k terms as base cases?
- Are there Lean/Mathlib libraries for linear recurrences we should know about?

## Desired Output Format

1. **Technical Summary** (1-2 pages): Core algorithm, guarantees, complexity
2. **Failure Mode Taxonomy**: Table of what B-M can't handle and why
3. **Implementation Recommendations**: Best libraries, code patterns
4. **Competition Math Analysis**: Statistics on recurrence types in olympiad problems
5. **Edge Case Compendium**: Specific sequences that break or challenge B-M
6. **References**: Academic papers, competitive programming resources, implementations



















# Berlekamp-Massey Algorithm: A Definitive Guide for Automated Mathematical Reasoning

## 1. Introduction and Theoretical Foundations

The automation of high-level mathematical reasoning, particularly within the domain of competitive mathematics (IMO, AIME, Putnam), necessitates a deterministic and rigorously verifiable mechanism for pattern discovery. While Large Language Models (LLMs) excel at semantic translation and heuristic guessing, they lack the arithmetic precision and algorithmic guarantees required to definitively identify generating functions for integer sequences. The Berlekamp-Massey (BM) algorithm serves as the cornerstone of this pipeline, bridging the gap between raw data generation and formal verification in Lean 4.

This report provides an exhaustive analysis of the Berlekamp-Massey algorithm, positioning it not merely as a sequence prediction tool, but as a fundamental algebraic operator on linear feedback shift registers (LFSRs). We explore its mathematical guarantees, its extension to complex arithmetic structures via the Reeds-Sloane algorithm, and its integration into a modern neuro-symbolic theorem proving stack.

### 1.1 The Algebra of Linear Recurrences

To leverage the Berlekamp-Massey algorithm effectively, one must first rigorously define the algebraic object it synthesizes: the linear recurrence relation. A sequence $S = (s_0, s_1, s_2, \dots)$ over a field $\mathbb{F}$ is said to be **linearly recurrent** (or $C$-finite) if there exists a finite degree $L$ and coefficients $c_1, c_2, \dots, c_L \in \mathbb{F}$ such that for all $n \ge L$:

$$s_n + \sum_{i=1}^{L} c_i s_{n-i} = 0$$

This equation defines the feedback mechanism of an LFSR. In the context of generating functions, this property is equivalent to the assertion that the ordinary generating function (OGF) $S(x) = \sum_{n=0}^{\infty} s_n x^n$ is a rational function of the form:

$$S(x) = \frac{P(x)}{C(x)}$$

where $C(x) = 1 + c_1 x + c_2 x^2 + \dots + c_L x^L$ is the **connection polynomial** (also known as the reciprocal characteristic polynomial), and $P(x)$ is an initialization polynomial with $\deg(P) < L$.

The Berlekamp-Massey algorithm can be fundamentally understood as a solver for the **Padé approximation problem**. Given a finite prefix of the sequence $s_0, \dots, s_{N-1}$, the algorithm seeks to find the pair of polynomials $(P(x), C(x))$ such that:

$$S(x) C(x) \equiv P(x) \pmod{x^N}$$

subject to the constraint that the degree of $C(x)$, denoted $L$, is minimized. This minimization is crucial; infinitely many recurrences can explain a finite prefix, but only the minimal recurrence captures the intrinsic structure of the sequence.

### 1.2 Mathematical Guarantees and The Uniqueness Bound

The utility of BM in a deterministic pipeline rests on its provable correctness. The most critical theorem governing its application is the relationship between the sequence length $N$ and the recurrence degree $L$.

**Theorem (Massey, 1969):** Let $S$ be a linearly recurrent sequence with minimal connection polynomial $C(x)$ of degree $L$. If the algorithm is provided with a prefix of length $N \ge 2L$, it uniquely identifies $C(x)$.

**Proof Insight:** The requirement $N \ge 2L$ is not arbitrary. It stems from the theory of Hankel matrices. The linear recurrence relation can be viewed as a dependency among the rows of the Hankel matrix formed by the sequence. For a recurrence of order $L$ to be uniquely determined, the underlying system of linear equations must have full rank. If fewer than $2L$ terms are provided, the system is underdetermined, meaning multiple distinct polynomials of degree $L$ (or smaller) could generate the observed prefix. Specifically, if two distinct recurrences of length $L_1$ and $L_2$ generate the same sequence of length $N$, then the sequence must satisfy a recurrence of length at most $L_1 + L_2$. If $N < L_1 + L_2$, ambiguity persists. The bound $2L$ ensures that any alternative explanation would require a complexity that contradicts the minimality assumption.

For the competition math pipeline, this theorem provides a strict stopping condition:

1. **Generation Phase:** The LLM generates terms $s_0, \dots, s_k$.
    
2. **Synthesis Phase:** BM finds a recurrence of degree $L$.
    
3. **Verification Phase:** If $k \ge 2L$, the recurrence is mathematically unique _for that prefix_. If $k < 2L$, the result is speculative and requires generating more terms ($s_{k+1}, \dots$) to confirm stability.
    

### 1.3 Behavior on Non-Recurrent Sequences

A vital aspect of "complete mastery" is understanding the algorithm's behavior when the premise of linear recurrence is false. Many competition problems involve sequences that are not $C$-finite, such as the Catalan numbers or the partition function.

When BM is applied to a non-recurrent sequence, it does not halt with an error. Instead, it produces a sequence of "best-fit" recurrences for the data seen so far. The hallmark of a non-linearly recurrent sequence is the behavior of its **linear complexity profile** $L_n$.

For a truly random sequence or a non-rational algebraic sequence (like Catalan numbers), the linear complexity $L_n$ grows linearly with $n$, typically satisfying:

$$L_n \approx \frac{n}{2}$$

This growth pattern serves as a powerful heuristic discriminator. If the pipeline observes that the degree of the discovered recurrence scales with the number of inputs ($L \approx N/2$), it should reject the hypothesis of linear recurrence and switch to alternative solvers (e.g., holonomic guessers).

## 2. Core Algorithm Mechanics

The Berlekamp-Massey algorithm is an iterative procedure that processes the sequence one term at a time, updating the current connection polynomial $C(x)$ to account for any discrepancies.

### 2.1 The Iterative Update Logic

Let $S = (s_0, s_1, \dots)$ be the input sequence. The algorithm maintains two primary polynomials:

- $C(x)$: The current connection polynomial.
    
- $B(x)$: The previous connection polynomial (used for corrections).
    
    It also tracks the current linear complexity $L$, and the index $m$ of the last length change.
    

**Initialization:**

$C(x) = 1$, $B(x) = 1$, $L = 0$, $m = -1$, $b = 1$ (discrepancy at step $m$).

**Iteration (for $n = 0, 1, \dots, N-1$):**

1. **Calculate Discrepancy:** The algorithm predicts the next term $s_n$ using the current recurrence and compares it to the actual value. The discrepancy $d$ is:
    
    $$d = s_n + \sum_{i=1}^{L} C_i s_{n-i}$$
    
    where $C_i$ is the coefficient of $x^i$ in $C(x)$.
    
2. **Evaluate Condition:**
    
    - **Case 1: $d = 0$.** The current recurrence correctly predicts the new term. No update is needed. $m$ and $B(x)$ remain unchanged.
        
    - **Case 2: $d \neq 0$.** The recurrence fails. We must adjust $C(x)$. The update formula uses the previous polynomial $B(x)$ to cancel the error:
        
        $$C_{new}(x) = C(x) - \frac{d}{b} x^{n-m} B(x)$$
        
        Here, $x^{n-m}$ shifts the previous correction polynomial to align with the current position. The scalar factor $d/b$ scales the correction to exactly negate the discrepancy $d$.
        
3. **Update Length:**
    
    If $2L \le n$, the current register length $L$ is insufficient to handle the sequence complexity. We must increase the length.
    
    - $L_{new} = n + 1 - L$
        
    - Update the "previous" polynomial: $B(x) \leftarrow C(x)$ (the old $C(x)$ before the update).
        
    - Update the "previous" discrepancy: $b \leftarrow d$.
        
    - Update the "previous" index: $m \leftarrow n$.
        
        If $2L > n$, the length is sufficient. We update $C(x)$ but keep $L$, $B(x)$, $b$, and $m$ unchanged (effectively treating the current step as a transient error within the capacity of the current length).
        

### 2.2 Time and Space Complexity

- **Time Complexity:** The calculation of the discrepancy takes $O(L)$ operations. The update of $C(x)$ involves polynomial addition/scaling of degree $L$. Since $L$ can grow up to $N$, the work per step is $O(N)$. Over $N$ steps, the total complexity is **$O(N^2)$**.
    
- **Space Complexity:** The algorithm stores $C(x)$ and $B(x)$, both of degree at most $N$. Thus, space complexity is **$O(N)$**.
    
- **Optimized Variations:** Using the connection between BM and the Euclidean Algorithm, it is possible to implement a "Half-GCD" (HGCD) version that runs in **$O(N \log^2 N)$**. However, for competition math where $N$ is typically small ($<1000$), the overhead of HGCD makes the $O(N^2)$ implementation superior in practice.
    

### 2.3 Fraction-Free Implementation

Standard BM requires division by $b$ (the previous discrepancy). When working with integer sequences, this introduces rational numbers, leading to coefficient explosion and potential precision issues if floating-point arithmetic is naively substituted.

For exact integer arithmetic, the **Fraction-Free Berlekamp-Massey** algorithm is preferred. It avoids division by cross-multiplying.

**Update Rule (Fraction-Free):**

$$C_{new}(x) = b \cdot C(x) - d \cdot x^{n-m} B(x)$$

This ensures that if the input sequence consists of integers, all intermediate polynomials also have integer coefficients. This is the **standard recommendation** for the competition math pipeline, as Python handles arbitrary-precision integers natively, avoiding the complexities of rational number arithmetic or modular inverses (unless working over $\mathbb{Z}_p$).

## 3. Field Arithmetic and Composite Moduli

The standard formulation of BM assumes the coefficients lie in a field $\mathbb{F}$ (e.g., $\mathbb{Q}$, $\mathbb{R}$, $\mathbb{Z}_p$). The operation $d/b$ requires the existence of a multiplicative inverse $b^{-1}$. This creates significant challenges when analyzing sequences over rings that are not fields, particularly $\mathbb{Z}_m$ where $m$ is composite.

### 3.1 Failure in $\mathbb{Z}_m$ (Composite Modulus)

Consider a sequence over $\mathbb{Z}_{10}$. If the discrepancy at the previous step was $b=2$, and the current discrepancy is $d=3$, the update requires computing $3 \cdot 2^{-1} \pmod{10}$. Since 2 has no inverse modulo 10 (it is a zero divisor), the standard algorithm crashes or produces undefined behavior.

In competition math, sequences modulo $m$ are common (e.g., "Find the last 2 digits of..."). Simply applying BM over the integers $\mathbb{Z}$ and then reducing modulo $m$ is often insufficient because the minimal recurrence over $\mathbb{Z}$ might have a much higher degree than the minimal recurrence over $\mathbb{Z}_m$.

### 3.2 The Reeds-Sloane Extension

To handle sequences over $\mathbb{Z}_m$, we employ the **Reeds-Sloane algorithm**. This algorithm generalizes BM to rings by replacing division with a more sophisticated update strategy that respects the ring structure.

The strategy involves **Prime Factorization** and the **Chinese Remainder Theorem (CRT)**.

1. **Decomposition:** Factor the modulus $m = p_1^{e_1} p_2^{e_2} \dots p_k^{e_k}$. The problem of finding the minimal recurrence modulo $m$ is decomposed into $k$ independent sub-problems: finding the recurrence modulo $p_i^{e_i}$.
    
2. **Prime Power Synthesis:** The core of Reeds-Sloane is solving the synthesis problem over a ring $\mathbb{Z}_{p^e}$.
    
    - Unlike a field, non-zero elements in $\mathbb{Z}_{p^e}$ may not be invertible (specifically, multiples of $p$).
        
    - The algorithm defines a **$p$-adic valuation** for the discrepancy. Instead of simply checking if $d=0$, it tracks how "divisible by $p$" the discrepancy is.
        
    - It maintains a _set_ of candidate polynomials rather than a single one, effectively performing a lattice reduction to find the shortest generator that satisfies the congruence modulo $p^e$.
        
    - The update step involves choosing a candidate that minimizes the increase in linear complexity, penalized by the $p$-adic valuation.
        
3. **Reconstruction:** Once the minimal polynomials $C_i(x)$ for each $p_i^{e_i}$ are found, they are combined using the Chinese Remainder Theorem for polynomials to produce the global connection polynomial $C(x) \in \mathbb{Z}_m[x]$.
    

**Implementation Note:** Writing a full Reeds-Sloane solver from scratch is error-prone. A practical compromise for the pipeline is:

- If $m$ is prime: Use standard BM over $\mathbb{Z}_p$.
    
- If $m$ is square-free ($p_1 \dots p_k$): Run BM modulo each $p_i$ and CRT the results.
    
- If $m$ has prime powers ($p^k$): Use the Reeds-Sloane logic or, if $N$ is large enough, solve over $\mathbb{Z}$ (fraction-free) and reduce the result, accepting that the degree might be non-minimal.
    

### 3.3 Large Integers and Arbitrary Precision

Competition math often involves sequences with rapid growth (e.g., $10^{100}$). Python's `int` type supports arbitrary precision automatically. The Berlekamp-Massey algorithm works transparently with these large numbers.

- **Memory:** Storing a sequence of 1000 terms where the $N$-th term has $N$ digits requires roughly $O(N^2)$ bits. For $N=1000$, this is trivial (~1 MB).
    
- **Speed:** Multiplication of large integers uses the Karatsuba algorithm ($O(D^{1.58})$ where $D$ is the number of digits). For $N=30$ (typical competition pipeline), operations are instantaneous. The pipeline need not implement custom big-integer arithmetic.
    

## 4. Practical Failure Modes and Analysis

Recognizing when BM fails is as important as using it when it succeeds. "Failure" here refers to the inability to find a _low-degree_ recurrence, often indicating the sequence belongs to a more complex class (P-recursive, Modular, or Hypergeometric).

### 4.1 Polynomial Sequences ($n^2, n^3$)

- **Sequence:** $0, 1, 4, 9, 16, 25, \dots$ ($a_n = n^2$)
    
- **Generating Function:** $\sum n^2 x^n = \frac{x(1+x)}{(1-x)^3}$.
    
- **BM Result:** BM will successfully find the connection polynomial $C(x) = (1-x)^3 = 1 - 3x + 3x^2 - x^3$.
    
- **Observation:** Polynomial sequences of degree $d$ are linearly recurrent with connection polynomial $(1-x)^{d+1}$. BM handles these perfectly.
    

### 4.2 Exponential Sequences ($2^n$)

- **Sequence:** $1, 2, 4, 8, 16, \dots$
    
- **BM Result:** $C(x) = 1 - 2x$. Success.
    
- **Failure Case:** $a_n = 2^n \pmod{3}$. This sequence is $1, 2, 1, 2, \dots$. BM over $\mathbb{Z}$ might struggle if not using modular arithmetic, but BM over $\mathbb{Z}_3$ finds $1+x$.
    

### 4.3 The "Coincidence" Trap: Moser's Circle Problem

This is the canonical example of why $N \ge 2L$ is vital.

- **Sequence:** $1, 2, 4, 8, 16, 31, 57, 99, \dots$
    
- **First 5 Terms:** $1, 2, 4, 8, 16$. BM finds $C(x) = 1 - 2x$ (Degree 1).
    
- **Adding 6th Term (31):** The discrepancy is non-zero ($31 \neq 32$). BM performs a massive degree update. The true recurrence involves a polynomial of degree 5: $(1-x)^5$.
    
- **Algebraic Reason:** The closed form is a sum of binomial coefficients $\sum_{k=0}^4 \binom{n-1}{k}$. Any sum of binomials (polynomial in $n$) leads to a recurrence power of $(1-x)$.
    
- **Pipeline Heuristic:** If the discovered recurrence predicts the next term $s_{next}$ and the actual term deviates slightly (e.g., 31 vs 32), checking for polynomial corrections (like adding a term $\binom{n}{k}$) is a valid secondary strategy.
    

### 4.4 Factorials and Catalan Numbers

- **Factorials ($n!$):** $1, 1, 2, 6, 24, 120, \dots$.
    
    The ratio $s_n / s_{n-1} = n$ is not constant. There is no constant-coefficient linear recurrence.
    
    **BM Behavior:** The algorithm will generate a sequence of polynomials with degrees $L \approx N/2$. The "linear complexity" grows linearly.
    
    **Alternative:** Use a **Holonomic Recurrence finder** (guessing coefficients for $c_1(n)s_n + c_0(n)s_{n-1} = 0$).
    
- **Catalan Numbers:** $1, 1, 2, 5, 14, 42, \dots$. The generating function involves $\sqrt{1-4x}$. It is algebraic, not rational. **BM Behavior:** Similar to factorials, $L \approx N/2$. **Diagnostic:** If BM returns a recurrence of degree 15 for a sequence of 30 terms, the sequence is almost certainly **not** linearly recurrent. The pipeline should flag this as "Likely Holonomic or Non-Elementary".
    

## 5. From Recurrence to Closed Form: The $N$-th Term

Once BM identifies a recurrence $s_n = \sum_{i=1}^L c_i s_{n-i}$, the pipeline must compute $s_N$ for large $N$ (e.g., $N=10^{18}$ in combinatorics problems).

### 5.1 Matrix Exponentiation

The recurrence can be encoded in a Companion Matrix $M$ of size $L \times L$.

$$\begin{pmatrix} s_n \\ s_{n-1} \\ \vdots \end{pmatrix} = M \begin{pmatrix} s_{n-1} \\ s_{n-2} \\ \vdots \end{pmatrix}$$

Computing $M^N$ takes $O(L^3 \log N)$ operations.

- **Pros:** Simple to implement.
    
- **Cons:** $O(L^3)$ is prohibitively slow if $L$ is large (e.g., $L=500$).
    

### 5.2 The Bostan-Mori Algorithm (2020)

For high-degree recurrences, the **Bostan-Mori algorithm** is asymptotically superior ($O(L \log L \log N)$). It exploits the rational function representation $S(x) = P(x)/Q(x)$.

To find the $N$-th coefficient $[x^N] \frac{P(x)}{Q(x)}$:

1. Multiply numerator and denominator by $Q(-x)$:
    
    $$\frac{P(x)Q(-x)}{Q(x)Q(-x)} = \frac{P(x)Q(-x)}{V(x^2)}$$
    
    where $V(y) = Q(\sqrt{y})Q(-\sqrt{y})$ is a polynomial in $x^2$.
    
2. **Even/Odd Splitting:** If $N$ is even, the $N$-th coefficient comes only from the even powers of $x$ in the numerator. If $N$ is odd, only from the odd powers.
    
    - If $N$ is even ($N=2k$): $[x^{2k}] \frac{P(x)Q(-x)}{V(x^2)} = [x^k] \frac{P_{even}(x)}{V(x)}$
        
    - If $N$ is odd ($N=2k+1$): $[x^{2k+1}] \frac{P(x)Q(-x)}{V(x^2)} = [x^k] \frac{P_{odd}(x)}{V(x)}$
        
3. **Recursive Step:** This reduces the problem to finding the $k$-th coefficient of a new rational function with the same degree but half the target index.
    
4. **Base Case:** When $N=0$, return $P(0)/Q(0)$.
    

This algorithm effectively doubles the step size at each iteration using polynomial multiplication (FFT-based), making it vastly more efficient than matrix multiplication for large $L$.

**Recommendation:**

- For $L < 50$: Use Matrix Exponentiation (lower constant overhead).
    
- For $L \ge 50$: Use Bostan-Mori.
    

## 6. Numerical Stability and Implementation Strategy

### 6.1 Numerical Stability

- **Floating Point:** Using `float` or `double` in BM is catastrophic. The subtraction $C(x) - \frac{d}{b} x^{n-m} B(x)$ involves cancellation of large terms. Errors accumulate exponentially, leading to incorrect degree updates.
    
- **Rational Arithmetic:** Python's `fractions.Fraction` is exact but suffers from "coefficient swell." The number of bits required to represent coefficients doubles roughly every step. For $N=100$, numerators can have $2^{100}$ bits.
    
- **Modular Arithmetic:** Extremely stable. If the problem asks for the answer modulo $p$, perform all BM operations in $\mathbb{Z}_p$. This bounds coefficient size to $\log p$.
    
- **Fraction-Free (Integer):** Best compromise for general integer sequences. It grows coefficients but avoids the GCD overhead of `Fraction`.
    

### 6.2 Implementation Detail: SymPy vs. Custom

SymPy's `find_linear_recurrence` is robust but opaque. A custom implementation is recommended for the pipeline to:

1. Expose the **Linear Complexity Profile** (history of $L$ values) for failure analysis.
    
2. Implement the **Fraction-Free** update explicitly to handle huge integers.
    
3. Integrate **Bostan-Mori** directly for the evaluation step.
    

## 7. Advanced Variations and Extensions

### 7.1 Berlekamp-Massey-Sakata (BMS)

When the data is not a sequence but a 2D array (e.g., a table of values $s_{i,j}$), the **Berlekamp-Massey-Sakata algorithm** finds the minimal set of 2D recurrence relations.

- It computes a **Gröbner basis** for the ideal of linear recurrence relations annihilating the array.
    
- While powerful, 2D recurrences are rare in competition math. A simpler heuristic is to flatten the array (e.g., anti-diagonals) into a 1D sequence and run standard BM.
    

### 7.2 Prony's Method

Prony's method is the continuous analogue of BM. It is used to recover parameters of a sum of exponentials $f(t) = \sum A_i e^{\lambda_i t}$ from sampled data.

- **Connection:** If we sample $f(t)$ at integers $t=0, 1, \dots$, the sequence $s_n = \sum A_i (e^{\lambda_i})^n$ is linearly recurrent (sum of geometric series). BM finds the polynomial whose roots are $e^{\lambda_i}$. This is useful if the competition problem involves finding the continuous function generating a sequence.
    

## 8. Integration with Lean 4 Verification

The final step is formally verifying the discovered pattern.

### 8.1 Data Structures in Lean

The recurrence should be defined using `Mathlib.Algebra.LinearRecurrence`.

Lean

```
import Mathlib.Algebra.LinearRecurrence

def my_recurrence : LinearRecurrence ℤ := {
  order := 3,
  coeffs :=![1, -2, 1] -- Coefficients for s(n-1), s(n-2), s(n-3)
}
```

### 8.2 Verification via `native_decide`

The pipeline generates a Lean file that asserts the sequence computed by the code matches the recurrence.

Since we cannot verify the _infinite_ sequence without an inductive proof (which requires insight into the _source_ of the sequence, not just the numbers), the attainable goal is **Bounded Verification**.

We prove: $\forall n < K, \text{generated\_seq}(n) = \text{recurrence\_seq}(n)$.

Lean

```
def check_recurrence (n : ℕ) : Bool :=
  generated_seq n == recurrence_eval n

theorem valid_prefix : (List.range 100).all check_recurrence = true := by
  native_decide
```

**`native_decide`** is crucial here. It compiles the Lean definitions into binary code and executes them. This avoids the slowness of kernel-based reduction (`rfl`) when checking 100 terms of complex arithmetic. It relies on the consistency of the Lean compiler, which is generally accepted for computational verification in this context.

### 8.3 Decidability Issues

The `LinearRecurrence` structure usually requires a `CommSemiring`. For `native_decide` to work, the equality on the ring must be **decidable**.

- **Safe:** `Int`, `Nat`, `Rat`, `ZMod p`.
    
- **Unsafe:** `Real` (equality is undecidable). The pipeline must restrict BM outputs to coefficients in $\mathbb{Q}$ or $\mathbb{Z}$ to ensure the generated Lean code compiles.
    

## 9. Conclusion

The Berlekamp-Massey algorithm is a robust "truth engine" for linear recurrences. By implementing the **Fraction-Free** variant for integer stability, integrating **Bostan-Mori** for efficient term computation, and wrapping the logic in a **Reeds-Sloane** handler for composite moduli, the system can achieve deterministic mastery over $C$-finite sequences. The integration with **Lean 4** via `native_decide` provides the final layer of rigor, converting heuristic discovery into machine-checked proof for finite prefixes, establishing a high-confidence starting point for full inductive proofs.

### Summary of Recommendations

|**Component**|**Recommendation**|**Reason**|
|---|---|---|
|**Algorithm**|Fraction-Free Berlekamp-Massey|Avoids float errors & rational swell.|
|**Evaluation**|Bostan-Mori ($N > 50$), Matrix Exp ($N < 50$)|Optimal asymptotic complexity.|
|**Moduli**|Reeds-Sloane or CRT Wrapper|Handles non-invertible elements in $\mathbb{Z}_m$.|
|**Heuristic**|Linear Complexity Profile ($L \approx N/2$ check)|Detects non-recurrent sequences early.|
|**Verification**|`native_decide` on prefix|Efficient computational proof in Lean.|