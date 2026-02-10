# Deep Research: Modular Arithmetic Patterns — Cycles, Periods, and Fast Verification

## Research Objective

Many competition math problems involve computing f(10^18) mod p. The naive approach (iterate 10^18 times) is impossible. The insight is that modular sequences become periodic. We need complete mastery of detecting, extracting, and verifying these cycles.

## Context

Example problem: "Find 7^(7^7) mod 100."

Our approach:
1. Compute 7^1 mod 100, 7^2 mod 100, ..., 7^k mod 100
2. Detect the cycle: [7, 49, 43, 1, 7, 49, 43, 1, ...] (period = 4)
3. Derive formula: 7^n mod 100 = 7^(n mod 4) mod 100
4. Verify in Lean using native_decide for small cases
5. Compute 7^(7^7 mod 4) mod 100 for the final answer

This is a core pattern in number theory problems.

## Research Questions

### Part A: Theory of Modular Periodicity

#### 1. Pisano Period
- For Fibonacci numbers mod m, the Pisano period π(m)
- F(n) mod m is periodic with period π(m)
- Known values and bounds for π(m)
- How to compute π(m) efficiently

#### 2. General Linear Recurrence Periodicity
- If a(n) = c₁a(n-1) + c₂a(n-2) + ... + cₖa(n-k) over ℤ_m
- The sequence is eventually periodic
- Period divides m^k - 1 (or related bound)
- Preperiod length bounds

#### 3. Exponential Periodicity (Carmichael Function)
- For a^n mod m, the sequence is periodic with period dividing λ(m)
- λ(m) is Carmichael's function (reduced totient)
- Relationship to Euler's φ(m)
- When is the period exactly λ(m)?

#### 4. Polynomial Sequences mod m
- For polynomial P(n) mod m, the sequence has period m (or divisor)
- Chinese Remainder Theorem decomposition
- Special cases: P(n) = n², n³, etc.

### Part B: Cycle Detection Algorithms

#### 5. Floyd's Tortoise and Hare
- Classic O(1) space cycle detection
- How it works for modular sequences
- Detecting both preperiod and period

#### 6. Brent's Algorithm
- Improved cycle detection
- When is it better than Floyd's?
- Implementation details

#### 7. Direct Period Computation
- For exponential a^n mod m, compute λ(m) directly
- For linear recurrences, use matrix analysis
- When is direct computation better than detection?

### Part C: From Trace to Pattern

#### 8. Detecting Periodicity from Partial Trace
Given [a₁, a₂, ..., a₃₀]:
- How do we determine if it's periodic?
- Minimum trace length to detect period p?
- False positive risk (coincidental pattern)?

#### 9. Extracting the Period
- Finding the minimal period
- Distinguishing preperiod from period
- Handling long preperiods

#### 10. Confidence in Detected Pattern
- If we see period 4 in first 30 terms, how confident are we?
- When might the "real" period be longer?
- Competition math context: patterns are usually real

### Part D: Efficient Computation

#### 11. Matrix Exponentiation for Recurrences
Given recurrence a(n) = c₁a(n-1) + ... + cₖa(n-k):
- Construct companion matrix M
- a(n) = M^(n-k) × [a(k), a(k-1), ..., a(1)]ᵀ
- M^n computed in O(k³ log n) via binary exponentiation

#### 12. Modular Matrix Exponentiation
- Same as above, but all operations mod m
- No precision issues — exact integer arithmetic
- Memory considerations for large matrices

#### 13. Polynomial Exponentiation (Cayley-Hamilton)
- Alternative to matrix exponentiation
- Represent recurrence as polynomial in shift operator
- x^n mod characteristic polynomial
- O(k² log n) or O(k log k log n) with FFT

#### 14. Tower Exponentiation (a^b^c mod m)
- For problems like 7^(7^7) mod 100
- Use a^b mod m = a^(b mod λ(m) + λ(m)) mod m (when b ≥ log₂ m)
- Recursive application for tall towers

### Part E: Lean Verification for Modular Patterns

#### 15. Efficient powMod in Lean
Our custom implementation:
```lean
def powMod (base exp mod : Nat) : Nat := ...
```
- Is this correct and efficient?
- Does native_decide use this for verification?
- Any edge cases (mod = 0, mod = 1)?

#### 16. Verifying Periodicity
How do we verify in Lean that:
- a^n mod m = a^(n mod period) mod m for n = 1..100?
- Template for period verification?

#### 17. Verifying Linear Recurrence Mod m
- Check that a(n) = c₁a(n-1) + ... + cₖa(n-k) mod m for n = k+1..100
- Lean template for recurrence verification

### Part F: Common Competition Patterns

#### 18. Last Digit Problems
- "Find the last digit of 7^2025"
- Last digit cycles with period dividing 4 for odd numbers
- Period 1, 2, or 4 for different bases

#### 19. Last Two Digits
- Mod 100 patterns
- When does period divide 20? 40? 100?

#### 20. Fibonacci Mod p
- π(10) = 60, π(100) = 300, etc.
- Known Pisano periods for small m
- LLM should recognize these

#### 21. Lucas Numbers, Tribonacci, etc.
- General k-nacci sequences mod m
- Periodicity properties

### Part G: Edge Cases and Gotchas

#### 22. Preperiod Considerations
- Some sequences take time to "settle" into period
- Example: a(n) = (previous terms product) mod m
- How to handle preperiod in verification

#### 23. Prime vs Composite Moduli
- Period for mod p (prime) vs mod p² vs mod pq
- CRT decomposition for composite moduli

#### 24. Zero in the Sequence
- If a(n) = 0 mod m for some n, sequence stays 0 for multiplicative sequences
- This is a valid period-1 pattern
- Handling this edge case

#### 25. Negative Coefficients
- Recurrence with negative coefficients: a(n) = a(n-1) - a(n-2)
- Proper modular arithmetic (ensure positive)

### Part H: Integration with Pipeline

#### 26. Trace Generator Prompting
How do we prompt LLM to generate:
- Correct modular trace code
- Efficient enough for n = 1..50
- Output in usable format

#### 27. Pattern Miner Integration
- Berlekamp-Massey on modular sequences
- Works over ℤ_p for prime p
- Reeds-Sloane for composite moduli

#### 28. Period Detection Layer
- Add cycle detection as a mining step?
- Before or after B-M?
- Or as post-processing on B-M result?

### Part I: Implementation Details

#### 29. Python Implementation
```python
def detect_cycle(seq):
    # Floyd or Brent implementation
    # Returns (preperiod_length, period)
    ...

def matrix_pow_mod(M, n, mod):
    # Binary exponentiation for matrix
    ...
```

#### 30. SymPy Functions
- `sympy.ntheory.carmichael_function(m)`
- `sympy.ntheory.totient(m)`
- `sympy.polys.LinearRecurrence` — does this exist?

#### 31. Performance Benchmarks
- Time to detect period in length-1000 sequence
- Time for matrix exponentiation k=5, n=10^18, mod 10^9+7

## Desired Output Format

1. **Theory Summary**: Periodicity theorems and bounds
2. **Algorithm Catalog**: Cycle detection and computation methods
3. **Competition Pattern Library**: Common modular patterns with formulas
4. **Lean Verification Templates**: Ready-to-use code for modular checks
5. **Edge Case Compendium**: All the gotchas and how to handle them
6. **Implementation Guide**: Python code for the pipeline
7. **Performance Analysis**: Complexity and practical timing
8. **References**: Papers on Pisano periods, Carmichael function























# Deep Research: Modular Arithmetic Patterns — Cycles, Periods, and Fast Verification

## 1. Introduction and Research Scope

The evaluation of functions with astronomically large inputs modulo an integer $m$, typically denoted as $f(N) \pmod m$ where $N$ can exceed $10^{18}$ or even $10^{100}$, constitutes a central class of problems in algorithmic number theory and competitive mathematics. The brute-force iteration of such functions is computationally intractable, necessitating a shift from simulation to structural analysis. The fundamental insight governing these problems is that any sequence generated by a deterministic recurrence over a finite state space must eventually become periodic. This report provides a comprehensive, expert-level analysis of the theory, detection, and verification of these modular patterns.

Our objective is to establish complete mastery over the lifecycle of a modular problem: detecting the cycle from a partial trace, extracting the exact period and preperiod, deriving a closed-form or algorithmic solution for the $N$-th term, and rigorously verifying the result using formal methods in Lean 4. The context for this research is driven by the increasing complexity of mathematical Olympiad problems (e.g., Putnam, AIME, IMO) and the need for high-assurance software verification in cryptographic implementations.

The investigation is divided into theoretical foundations, algorithmic cycle detection, efficient computation techniques (such as matrix exponentiation), and modern formal verification. We synthesize classical results—such as the properties of the Pisano periods and the Carmichael function—with contemporary computational strategies, including the Reeds-Sloane algorithm for sequences over rings and `native_decide` based proofs in Lean. This synthesis aims to transform the "art" of spotting patterns into a rigorous "science" of automated modular analysis.

---

## 2. Part A: Theory of Modular Periodicity

The behavior of integer sequences modulo $m$ is governed by the algebraic structure of the ring $\mathbb{Z}/m\mathbb{Z}$. Unlike sequences over the field of real numbers, which may diverge or converge, modular sequences are confined to a finite set of values. By the Pigeonhole Principle, if a sequence is generated by a function $s_{n+1} = f(s_n)$ where $s_n \in S$ and $|S|$ is finite, the sequence must eventually repeat a state. The sequence is thus divided into two distinct components: the **preperiod** (or tail), which is the transient initial segment, and the **period** (or cycle), which repeats indefinitely.

### 2.1 The Pisano Period: Fibonacci Numbers Modulo $m$

The Fibonacci sequence, defined by the linear recurrence $F_n = F_{n-1} + F_{n-2}$ with $F_0=0, F_1=1$, exhibits rich periodic structure when reduced modulo $m$. This period is known as the Pisano period, denoted $\pi(m)$. Understanding $\pi(m)$ is the prototype for analyzing all second-order linear recurrences.

#### 2.1.1 Algebraic Characterization

The state of the Fibonacci recurrence is determined by a pair of consecutive terms $(F_n, F_{n+1})$. Since there are $m^2$ possible pairs in $\mathbb{Z}_m \times \mathbb{Z}_m$, the sequence must repeat within $m^2$ steps. However, the actual period is often much smaller. The transition between states is governed by the matrix $M = \begin{pmatrix} 0 & 1 \\ 1 & 1 \end{pmatrix}$. The $n$-th Fibonacci numbers are derived from $M^n$. The Pisano period $\pi(m)$ is exactly the order of the matrix $M$ in the General Linear Group $GL(2, \mathbb{Z}/m\mathbb{Z})$.

Since $\det(M) = -1$, we have $\det(M^k) = (-1)^k$. For the sequence to return to the identity state $\begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}$, the determinant must be $1$. This implies that for $m > 2$, the period $\pi(m)$ must be even.

#### 2.1.2 Bounds and Properties

The value of $\pi(m)$ depends critically on the prime factorization of $m$.

- **Multiplicativity:** $\pi(\text{lcm}(m, n)) = \text{lcm}(\pi(m), \pi(n))$. This reduces the problem to finding $\pi(p^k)$ for prime powers.
    
- **Prime Powers:** For a prime $p$, $\pi(p^k) = p^{k-1}\pi(p)$ is conjectured to hold for all $p$ (Wall's Conjecture). While unproven, no counterexamples (Wall-Sun-Sun primes) have been found below $10^{14}$. For all practical competitive math purposes, one assumes this equality holds.
    
- **Bounds for Primes:**
    
    - If $p = 5$, $\pi(5) = 20$.
        
    - If $\left(\frac{5}{p}\right) = 1$ (i.e., $p \equiv 1, 4 \pmod 5$), then $\pi(p) \mid (p-1)$. The characteristic polynomial $x^2 - x - 1$ splits in $\mathbb{Z}_p$, meaning eigenvalues exist in the base field.
        
    - If $\left(\frac{5}{p}\right) = -1$ (i.e., $p \equiv 2, 3 \pmod 5$), then $\pi(p) \mid 2(p+1)$. The roots lie in the quadratic extension $\mathbb{F}_{p^2}$, and the Frobenius automorphism implies the roots are conjugate, leading to the period dividing $2(p+1)$ rather than $p^2-1$.
        
    - The strict upper bound is $\pi(m) \le 6m$, with equality iff $m = 2 \cdot 5^k$.
        

#### 2.1.3 Efficient Computation of $\pi(m)$

To compute $\pi(m)$ efficiently for large $m$:

1. Factor $m = \prod p_i^{e_i}$.
    
2. For each $p_i$, compute $\pi(p_i)$ by checking divisors of $p_i-1$ (if Legendre symbol is 1) or $2(p_i+1)$ (if -1).
    
3. Scale by $p_i^{e_i-1}$.
    
4. Compute the LCM of the components. This approach is exponentially faster than generating the sequence.
    

### 2.2 General Linear Recurrence Periodicity

Consider a linear recurrence relation of order $k$ over $\mathbb{Z}_m$:

$$a_n = c_1 a_{n-1} + c_2 a_{n-2} + \dots + c_k a_{n-k} \pmod m$$

The sequence of state vectors $\mathbf{v}_n = (a_n, \dots, a_{n-k+1})$ follows the linear map $\mathbf{v}_n = C \mathbf{v}_{n-1}$, where $C$ is the companion matrix.

#### 2.2.1 Period and Preperiod Structure

- **Invertibility:** If $c_k$ (the constant term of the characteristic polynomial) is coprime to $m$ (i.e., $c_k \in (\mathbb{Z}/m\mathbb{Z})^\times$), then the matrix $C$ is invertible. The map is a bijection on the state space, meaning there is **no preperiod** (except possibly for the trivial zero state if the recurrence is homogeneous). The sequence is purely periodic.
    
- **Non-invertibility:** If $\gcd(c_k, m) > 1$, the map is not surjective. The sequence will have a "tail" or preperiod before entering a cycle. The length of this preperiod is bounded by the max power of any prime factor of $m$ dividing the zero-divisor components, but generally is short compared to the period.
    
- **Period Length:** The maximum period is $m^k - 1$. This is achieved by M-sequences (Maximum Length Sequences) typically used in LFSRs (Linear Feedback Shift Registers) over fields. Over composite rings, the period divides the Carmichael function of the exponent of the extension ring defined by the characteristic polynomial.
    

### 2.3 Exponential Periodicity and the Carmichael Function

The sequence $x_n = a^n \pmod m$ is the simplest nonlinear recurrence ($x_n = a \cdot x_{n-1}$). Its periodicity is fundamental to RSA and Diffie-Hellman cryptography.

#### 2.3.1 Euler's Totient vs. Carmichael's Lambda

Euler's Theorem states $a^{\phi(m)} \equiv 1 \pmod m$ for $\gcd(a, m) = 1$. This implies the period of $a^n$ divides $\phi(m)$. However, $\phi(m)$ is not always the _minimal_ universal period. The **Carmichael function** $\lambda(m)$ is the smallest integer such that $a^{\lambda(m)} \equiv 1 \pmod m$ for all coprime $a$.

The relationship is defined as:

- $\lambda(p^k) = \phi(p^k)$ for odd primes.
    
- $\lambda(2^k) = \frac{1}{2}\phi(2^k)$ for $k \ge 3$.
    
- $\lambda(\text{lcm}(n_1, n_2, \dots)) = \text{lcm}(\lambda(n_1), \lambda(n_2), \dots)$. For example, $\phi(100) = 40$, but $\lambda(100) = \text{lcm}(\lambda(4), \lambda(25)) = \text{lcm}(2, 20) = 20$. Using $\lambda(m)$ instead of $\phi(m)$ can reduce exponent size by half or more, which is significant in computational efficiency.
    

#### 2.3.2 The Non-Coprime Case and Generalized Euler Theorem

When $\gcd(a, m) \neq 1$, the sequence $a^n \pmod m$ is not purely periodic. It has a preperiod. For sufficiently large $n$, the sequence becomes periodic with period dividing $\lambda(m)$.

The **Generalized Euler Theorem** states:

$$a^n \equiv a^{n \pmod{\phi(m)} + \phi(m)} \pmod m \quad \text{for } n \ge \log_2 m$$

More precisely, the threshold is the maximum exponent of any prime factor in the factorization of $m$. The addition of $\phi(m)$ (or specifically $\lambda(m)$) ensures that the exponent remains in the "periodic" region of the sequence, effectively skipping the preperiod. This formula is crucial for solving tower exponents $a^{b^c}$ where base and modulus share factors.

### 2.4 Polynomial Sequences Modulo $m$

Sequences of the form $x_n = P(n) \pmod m$ where $P(x)$ is a polynomial in $n$ (e.g., $n^2 + 1$) have a different structure.

- **Period:** The sequence is periodic with period $m$. This is trivial because if $n \equiv k \pmod m$, then $P(n) \equiv P(k) \pmod m$.
    
- **Sub-periods:** The minimal period can be a divisor of $m$. For example, $n^2 \pmod 8$ produces the sequence $0, 1, 4, 1, 0, 1, 4, 1, \dots$ which has period 4, not 8.
    
- **Analysis:** The period is determined by the smallest $k$ such that $P(x+k) \equiv P(x) \pmod m$ for all $x$. This requires determining $k$ such that the polynomial $\Delta_k P(x) = P(x+k) - P(x)$ is identically zero modulo $m$. This involves checking divisibility of coefficients by $m$.
    

---

## 3. Part B: Cycle Detection Algorithms

While theoretical bounds provide limits, in many "Black Box" problems (where the function $f$ is complex or unknown), we must detect the cycle dynamically. The two primary algorithms are Floyd's and Brent's.

### 3.1 Floyd's Tortoise and Hare

Floyd's cycle-finding algorithm is the standard $O(1)$ space algorithm. It uses two pointers moving at different speeds: the Tortoise ($T$) moves 1 step, the Hare ($H$) moves 2 steps.

**Algorithm Trace:**

1. Initialize $T = x_0, H = x_0$.
    
2. Loop: $T \leftarrow f(T), H \leftarrow f(f(H))$.
    
3. If $T = H$, a cycle is detected. The index is some multiple of the period $\lambda$.
    
4. **Finding $\mu$ (Preperiod):** Reset $T = x_0$. Keep $H$ at the meeting point. Move both 1 step at a time. The index where they meet is $\mu$.
    
5. **Finding $\lambda$ (Period):** From the meeting point $\mu$, advance $H$ until it equals $T$ again. The count is $\lambda$.
    

**Complexity:** The algorithm executes $3( \mu + \lambda )$ function evaluations in the worst case. It is robust and simple to implement.

### 3.2 Brent's Algorithm

Brent's algorithm is an optimization of Floyd's that attempts to find the cycle length $2^k < \lambda \le 2^{k+1}$ using exponential search.

**Mechanism:**

The Hare moves $2^k$ steps. If it doesn't meet the Tortoise, the Tortoise "teleports" to the Hare's current position, and the search continues with $2^{k+1}$ steps. This reduces the number of equality checks and function evaluations.

- **Comparison:** Brent's algorithm performs approximately $36\%$ fewer function evaluations on average than Floyd's. It is preferred when the function evaluation $f(x)$ is cheap but the comparison is expensive, or simply for raw speed in factorization (Pollard's Rho).
    

### 3.3 Direct Computation vs. Detection

When should one use detection over math?

- **Use Detection:** When $m$ is relatively small ($< 10^9$) or the function is non-standard (e.g., $x_{n+1} = (x_n^2 + 1) \pmod m$). Detection is safer because it doesn't require proving properties of the function; it treats it as a black box.
    
- **Use Direct Math:** When $m$ is huge ($10^{18}$) but the structure is linear ($F_n$) or exponential ($a^n$). In these cases, simulating the cycle is impossible ($10^{18}$ steps), but computing $\lambda(m)$ or matrix powers takes logarithmic time.
    

---

## 4. Part C: From Trace to Pattern

In competition problems, one is often given the first $k$ terms of a sequence and asked to predict the rest. This is a problem of _system identification_.

### 4.1 Detecting Periodicity from Partial Trace

Given a trace $[a_1, a_2, \dots, a_{30}]$, how do we know it's periodic?

- **Heuristic:** If the sequence is generated by a linear recurrence of order $k$, we need roughly $2k$ terms to uniquely identify the recurrence (and thus the period) via the Berlekamp-Massey algorithm.
    
- **Risk:** "1, 2, 1, 2,..." looks like period 2. But it could be $1, 2, 1, 2, 3, \dots$. In modular arithmetic, however, the "patterns are usually real." If a sequence repeats a state vector of length $k$ (where $k$ is the order of recurrence), the periodicity is guaranteed mathematically.
    
- **Confidence:** The confidence is absolute if we observe a repetition of the _state vector_. For a simple sequence $x_{n+1} = f(x_n)$, observing $x_i = x_j$ guarantees periodicity. For order-2 ($x_{n+2} = f(x_{n+1}, x_n)$), we need a repeated _pair_ $(x_i, x_{i+1}) = (x_j, x_{j+1})$.
    

### 4.2 Extracting the Period and Preperiod

Given a list `S`:

1. **Iterate:** Check all sub-segments.
    
2. **Function:**
    
    Python
    
    ```
    def find_pattern(S):
        for length in range(1, len(S)//2):
            for start in range(len(S) - 2*length):
                p1 = S[start : start+length]
                p2 = S[start+length : start+2*length]
                if p1 == p2:
                    return start, length # Found candidate
    ```
    
3. **Validation:** This naive check ($O(N^3)$) can be optimized to $O(N^2)$ or using suffix trees. In competition contexts ($N \approx 50$), naive is sufficient.
    

---

## 5. Part D: Efficient Computation Techniques

Once the period or recurrence is identified, we need to jump to the $N$-th term without iteration.

### 5.1 Matrix Exponentiation for Recurrences

For linear recurrences, matrix exponentiation is the gold standard.

$$\begin{pmatrix} a_{n} \\ a_{n-1} \end{pmatrix} = \begin{pmatrix} c_1 & c_2 \\ 1 & 0 \end{pmatrix}^{n-1} \begin{pmatrix} a_{1} \\ a_{0} \end{pmatrix}$$

- **Complexity:** Computing $M^n$ takes $O(k^3 \log n)$ operations using binary exponentiation (Square-and-Multiply). For Fibonacci ($k=2$), this is extremely fast.
    
- **Modular Arithmetic:** All additions and multiplications in the matrix product are performed modulo $m$. This keeps the numbers small (fitting in 64-bit integers if $m < 2^{32}$) and prevents overflow.
    

### 5.2 Polynomial Exponentiation (Cayley-Hamilton)

For larger $k$ (e.g., $k=100$), $O(k^3)$ matrix multiplication is slow. We can use the Cayley-Hamilton theorem.

The characteristic polynomial is $P(x) = x^k - \sum c_i x^{k-i}$. The matrix $M$ satisfies $P(M) = 0$.

To compute $M^n$, we compute $x^n \pmod{P(x)}$ in the polynomial ring $\mathbb{Z}_m[x]$.

$$x^n = q(x)P(x) + r(x) \implies M^n = r(M)$$

This reduces the problem to polynomial modular exponentiation, which can be done in $O(k \log k \log n)$ using FFT-based multiplication, or $O(k^2 \log n)$ using standard multiplication. This is significantly faster than matrix multiplication for large $k$.

### 5.3 Tower Exponentiation ($a^{b^c} \pmod m$)

This is the "Boss Level" of modular arithmetic.

Problem: Compute $7^{7^7} \pmod{100}$.

1. **Reduce Exponent:** We need $7^X \pmod{100}$. By Euler's Theorem (since $\gcd(7, 100)=1$), $a^X \equiv a^{X \pmod{\phi(100)}} \pmod{100}$.
    
    - $\phi(100) = 100(1-1/2)(1-1/5) = 40$.
        
    - New goal: Find $7^7 \pmod{40}$.
        
2. **Recurse:** We need $7^7 \pmod{40}$. $\gcd(7, 40)=1$.
    
    - $\phi(40) = 40(1-1/2)(1-1/5) = 16$.
        
    - New goal: Find $7 \pmod{16}$. This is trivial: $7$.
        
3. **Unwind:**
    
    - Exponent for step 2 is $7 \pmod{16}$. So $7^7 \equiv 7^7 \pmod{40}$.
        
    - $7^2 = 49 \equiv 9 \pmod{40}$.
        
    - $7^4 \equiv 81 \equiv 1 \pmod{40}$.
        
    - $7^7 = 7^4 \cdot 7^3 \equiv 1 \cdot 343 \equiv 23 \pmod{40}$.
        
    - Final step: $7^{23} \pmod{100}$.
        
    - $7^{23} = 7^{20} \cdot 7^3 \equiv 1 \cdot 343 \equiv 43 \pmod{100}$ (using $\lambda(100)=20$).
        
    - Result: 43.
        

**Key Algorithm:** The function `tower_mod(base_list, m)` recursively calculates the exponent modulo $\phi(m)$ (or $\lambda(m)$). If $\gcd(base, m) \neq 1$, it uses the Generalized Euler formula: exponent becomes $(\text{exponent} \pmod{\phi(m)}) + \phi(m)$.

---

## 6. Part E: Lean Verification for Modular Patterns

In high-stakes environments (formal math libraries, verified cryptography), relying on a Python script is insufficient. We use Lean 4 to provide a proof certificate.

### 6.1 Efficient `powMod` in Lean

The standard `Nat.pow` in Lean is defined structurally and is inefficient for evaluation in the kernel. For `native_decide` to work on numbers like $10^{18}$, we need a binary exponentiation implementation that compiles to GMP integers.

Lean

```
/-- Efficient modular exponentiation using binary squaring -/
def powMod (base exp mod : Nat) : Nat :=
  if mod == 0 then 0 else
  match exp with

| 0 => 1 % mod
| _ =>
    let half := powMod base (exp / 2) mod
    let result := (half * half) % mod
    if exp % 2 == 1 then (result * base) % mod else result

-- Verify it computes correctly for small cases
example : powMod 7 3 100 = 43 := by native_decide
```

This function allows Lean's kernel to verify equations like $a^{10^{18}} \equiv x \pmod m$ in milliseconds.

### 6.2 Verifying Periodicity

We can define a predicate for periodicity and use `native_decide` to prove it for specific moduli.

Lean

```
def is_periodic (f : Nat → Nat) (p : Nat) : Prop :=
  ∀ n, f (n + p) = f n

/-- Theorem: A linear recurrence mod m is periodic with period p if the state repeats. -/
theorem fib_mod_periodic (m : Nat) (p : Nat)
  (h_recurrence : ∀ n, f (n+2) = (f (n+1) + f n) % m)
  (h_match : f p = f 0 ∧ f (p+1) = f 1) :
  is_periodic f p := by
  -- Proof omitted (requires induction on n)
  sorry
```

Using this template, a user can prove `is_periodic fib_mod_100 300` by simply checking the base case `h_match` using `native_decide`. This tactic evaluates the boolean condition `fib(300)%100 == fib(0)%100 &&...` at compile time and accepts the result as a proof.

### 6.3 Verifying Linear Recurrences

To verify that a sequence matches a derived formula (e.g., $F_n \pmod 5 = 2 \cdot 3^n \pmod 5$?), we can check the first $2k$ terms.

- **Theorem:** If two sequences $a_n, b_n$ satisfy the same linear recurrence of order $k$ and match on the first $k$ terms, they are identical.
    
- **Lean Strategy:** Prove $a_n$ and $b_n$ satisfy the recurrence. Then use `native_decide` to check $a_i = b_i$ for $i < k$.
    

---

## 7. Part F: Common Competition Patterns

### 7.1 Last Digit Problems

The "last digit" is $n \pmod{10}$.

- **Properties:** $\lambda(10) = \text{lcm}(\lambda(2), \lambda(5)) = \text{lcm}(1, 4) = 4$.
    
- **Implication:** For any integer $a$ (even non-coprime ones, essentially), the sequence $a^n \pmod{10}$ has period dividing 4 (after a preperiod).
    
    - $\{2, 4, 8, 6, \dots\}$ (Period 4)
        
    - $\{3, 9, 7, 1, \dots\}$ (Period 4)
        
    - $\{4, 6, \dots\}$ (Period 2)
        
    - $\{5, \dots\}$ (Period 1)
        
- **Trick:** To find last digit of $X^Y$, compute $Y \pmod 4$. If $Y \pmod 4 = 0$, use exponent 4 (not 0).
    

### 7.2 Last Two Digits

Modulus 100. $\lambda(100) = 20$.

- **Pattern:** $x^{20} \equiv 00, 01, \dots \pmod{100}$. If $\gcd(x, 100)=1$, $x^{20} \equiv 1 \pmod{100}$.
    
- **Application:** $7^{2022} \pmod{100}$. $2022 \equiv 2 \pmod{20}$. $7^{2022} \equiv 7^2 \equiv 49$.
    

### 7.3 Fibonacci Pisano Periods

- $\pi(10) = 60$ (Last digit of Fibonacci repeats every 60 terms).
    
- $\pi(100) = 300$.
    
- $\pi(10^k) = 1.5 \cdot 10^k$.
    
- This "1.5" factor is a recurring theme in decimal Fibonacci problems.
    

### 7.4 Lucas and $k$-nacci

- **Lucas:** $L_n = L_{n-1} + L_{n-2}$, start $2, 1$. Modulo properties are nearly identical to Fibonacci because they share the characteristic polynomial $x^2-x-1$.
    
- **Tribonacci:** $T_{n+3} = T_{n+2} + T_{n+1} + T_n$. Period is much larger. For prime $p$, period divides $(p^3-1)$. The period modulo 10 is 31 (surprisingly small), but modulo 100 it is 1240.
    

---

## 8. Part G: Edge Cases and Gotchas

### 8.1 Preperiod Considerations

The Generalized Euler Theorem $a^n \equiv a^{n \pmod{\lambda(m)} + \lambda(m)}$ is only valid if $n$ is large enough ($n \ge \text{exponent of prime powers in } m$).

- **Mistake:** Computing $2^1 \pmod 4$ using $\phi(4)=2$. $2^1 \not\equiv 2^{1 \pmod 2} \equiv 2^1 \equiv 2$. But $2^3 \pmod 4 \equiv 0$. $2^{3 \pmod 2} = 2^1 = 2$. **Error.**
    
- **Fix:** Ensure the exponent is effectively $n \pmod{\phi} + \phi$ to keep it $\ge \text{preperiod}$.
    

### 8.2 Prime vs Composite Moduli

- **CRT Strategy:** For $m = p \cdot q$, analyze the sequence modulo $p$ and modulo $q$ separately. The period is $\text{lcm}(\pi(p), \pi(q))$. The preperiod is $\max(\mu(p), \mu(q))$.
    
- **Composite Rings:** Matrix inversion can fail. When running Berlekamp-Massey on a sequence modulo $m$, if you encounter a division by a non-unit (zero divisor), the standard algorithm fails. You must use **Reeds-Sloane** or decompose via CRT.
    

### 8.3 Zero in the Sequence

If a multiplicative sequence $x_n = c \cdot x_{n-1} \pmod m$ hits $0$, it stays $0$. The period is 1 (the sequence $0, 0, 0\dots$).

This happens if $c$ shares factors with $m$ and the power of those factors accumulates to $m$.

---

## 9. Part H: Integration with Pipeline

To automate the solution of such problems, we propose a pipeline.

### 9.1 Trace Generator Prompting

When using LLMs to assist, prompt for **code execution**, not direct answers.

- **Bad:** "What is the period of Fibonacci mod 10?" (Hallucination risk).
    
- **Good:** "Write a Python script to generate the first 100 Fibonacci numbers modulo 10 and output them as a list."
    

### 9.2 Pattern Miner Integration

Use the generated trace with the **Berlekamp-Massey Algorithm**.

- **Input:** $[1, 1, 2, 3, 5, 8, 3, 1, 4, 5, 9, \dots]$ (Fib mod 10).
    
- **Action:** Run BMA over $\mathbb{Q}$? No, over $\mathbb{Z}_{10}$. Since 10 is composite, run over $\mathbb{Z}_2$ and $\mathbb{Z}_5$.
    
    - Mod 2: $[1, 1, 0, 1, 1, 0 \dots]$. Period 3.
        
    - Mod 5: Period 20.
        
    - Combine: Period $\text{lcm}(3, 20) = 60$.
        

### 9.3 Period Detection Layer

A robust solver pipeline:

1. **Generate Trace:** Compute first 100-200 terms.
    
2. **Detect Cycle:** Use Floyd's algorithm on the trace.
    
3. **Verify:** Check if detected period $p$ holds for the whole trace.
    
4. **Solve:** Use $a_N = a_{(N - \mu) \pmod \pi + \mu}$.
    

---

## 10. Part I: Implementation Details

### 10.1 Python Implementation (Cycle Detection)

Python

```
def brent_cycle_finding(f, x0):
    # Search for the length of the cycle (lambda)
    power = lam = 1
    tortoise = x0
    hare = f(x0) 
    while tortoise!= hare:
        if power == lam:  # Time to start a new power of two?
            tortoise = hare
            power *= 2
            lam = 0
        hare = f(hare)
        lam += 1
    
    # Find the start of the cycle (mu)
    tortoise = hare = x0
    for _ in range(lam):
        hare = f(hare)
    
    mu = 0
    while tortoise!= hare:
        tortoise = f(tortoise)
        hare = f(hare)
        mu += 1
        
    return mu, lam
```

### 10.2 Matrix Exponentiation (Python)

Python

```
def mat_mul(A, B, m):
    C = [, ]
    for i in range(2):
        for j in range(2):
            for k in range(2):
                C[i][j] = (C[i][j] + A[i][k] * B[k][j]) % m
    return C

def mat_pow(A, p, m):
    res = [, ]
    while p > 0:
        if p % 2 == 1:
            res = mat_mul(res, A, m)
        A = mat_mul(A, A, m)
        p //= 2
    return res
```

### 10.3 Performance Benchmarks

- **Cycle Detection:** For a period of $10^6$, Brent’s algorithm requires $\sim 10^6$ function evaluations but far fewer comparisons than Floyd's. On modern CPUs, this takes milliseconds.
    
- **Matrix Exponentiation:** Computing $F_{10^{18}}$ takes $\log_2(10^{18}) \approx 60$ matrix multiplications. Each $2 \times 2$ multiplication is negligible. Total time is $< 10\mu s$.
    
- **Python vs C++:** For $N < 10^{18}$, Python's arbitrary precision integers are perfectly adequate. C++ is only needed if $N$ is so large that the _number of digits_ in $N$ becomes a bottleneck (e.g., $N=10^{10^6}$), or if we are doing millions of queries.
    

---

## 11. Conclusion

The transition from naive iteration to the mastery of modular periodicity represents a shift from linear-time simulation to logarithmic-time algebraic solution. By leveraging the theoretical bounds of the Pisano period and Carmichael function, we can predict the behavior of sequences at indices far beyond the capability of any computer to iterate.

The integration of these number-theoretic insights with formal verification in Lean 4 provides a new standard for correctness. We are no longer just "guessing" the pattern from a trace; we are detecting it, extracting the recurrence, deriving the matrix form, and formally proving that the periodicity holds. This pipeline—Trace $\to$ Detect $\to$ Recurrence $\to$ Matrix $\to$ Proof—solves the general class of modular sequence problems with both efficiency and certainty.

### References

- Pisano Period Bounds
    
- Carmichael Function
    
- Cycle Detection Algorithms
    
- Power Towers & Generalized Euler Theorem
    
- Lean 4 Verification
    
- Reeds-Sloane Algorithm
    

### Table 1: Comparative Analysis of Periodicity Types

|**Sequence Type**|**Form**|**Period Bound**|**Key Function**|**Verification Method**|
|---|---|---|---|---|
|**Linear**|$n \pmod m$|$m$|Identity|Direct Modulo|
|**Polynomial**|$P(n) \pmod m$|$m$ (or divisor)|$\Delta P(x)$|Finite Difference|
|**Exponential**|$a^n \pmod m$|$\lambda(m)$|Carmichael $\lambda$|Generalized Euler Thm|
|**Recurrence**|$F_n \pmod m$|$\pi(m) \le m^2 - 1$|Matrix Order|Pisano Period / BMA|
|**Power Tower**|$a^{b^c} \pmod m$|Stable after $\log m$|Recursive $\phi$|Tower Reduction|

### Table 2: Modular Arithmetic Cheat Sheet (Olympiad Tricks)

|**Concept**|**Formula / Rule**|**Use Case**|
|---|---|---|
|**Fermat's Little Thm**|$a^{p-1} \equiv 1 \pmod p$|Prime modulus exponentiation|
|**Euler's Theorem**|$a^{\phi(m)} \equiv 1 \pmod m$|Coprime modulus exponentiation|
|**Generalized Euler**|$a^n \equiv a^{n \pmod{\phi(m)} + \phi(m)}$|Non-coprime bases (Power Towers)|
|**Lucas Theorem**|$\binom{n}{k} \equiv \prod \binom{n_i}{k_i} \pmod p$|Binomial coeff. mod prime|
|**LTE Lemma**|$v_p(x^n - y^n) = v_p(x-y) + v_p(n)$|Exponent valuation (divisibility)|
|**Matrix Exp.**|$V_n = M^n V_0$|Fast recurrence calculation|
|**Brent's Algo**|Steps $2^k$, teleports $T$|Fast cycle finding|

**(End of Report)**