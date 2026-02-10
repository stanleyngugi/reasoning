# Formal Verification of Geometry and Algebra in Competition Mathematics: Extending the Trace-to-Lean Architecture to Continuous Domains

## Executive Summary

The intersection of large language models (LLMs) and formal theorem proving has birthed new architectures for mathematical reasoning, most notably the "Trace-to-Lean" paradigm. This architecture, which currently excels in discrete domains such as combinatorics and number theory by leveraging deterministic pattern mining and computational verification, faces significant hurdles when applied to continuous domains like geometry and algebra. These fields, constituting approximately half of competition mathematics (e.g., AIME, IMO), involve continuous variables, irrational numbers, and complex constraints that resist simple sequence-based pattern recognition.

This research report presents a comprehensive framework for extending Trace-to-Lean to cover geometry and algebra with 100% formal guarantees. The core thesis driving this architecture is the asymmetry between discovery and verification: while finding a solution may require heuristic, numerical, or probabilistic methods (such as coordinate descent, PSLQ, or neural guidance), verifying that solution can often be reduced to a deterministic, computationally decidable task. By treating geometric proofs as constraint satisfaction problems over algebraic fields and algebraic proofs as certificate checking (via methods like Sum-of-Squares or Sturm's theorem), we can leverage Lean 4’s `native_decide` tactic and high-performance kernel to achieve formal correctness without requiring the solver to generate human-readable proofs.

This document details the theoretical foundations, implementation strategies, and complexity analyses required to build this unified verification framework. It explores the use of Gröbner bases for geometric theorems, Sturm's theorem for algebraic root isolation, and Sum-of-Squares certificates for inequalities, all integrated into a hierarchical verification system that balances computational cost with formal rigor.

---

## 1. Introduction: The Verification Gap in Neuro-Symbolic Mathematics

### 1.1 The Current State of Neuro-Symbolic Reasoning

Recent advancements in neuro-symbolic AI have demonstrated that LLMs can effectively generate "traces"—intermediate reasoning steps or executable code—that lead to correct answers in mathematical problems. The Trace-to-Lean architecture capitalizes on this by using Python to generate these traces (e.g., computing the first few terms of a sequence) and then employing deterministic algorithms (like Berlekamp-Massey) to infer general formulas. These formulas are then formally verified in Lean 4 using computational tactics like `native_decide`.

This approach has proven highly effective for discrete mathematics. In number theory and combinatorics, answers are often integers or simple rationals, and the relationships are often recursive or satisfying simple modular arithmetic properties. The verification of a closed-form recurrence or a modular identity is computationally inexpensive and typically decidable within the logic of standard integer arithmetic.

### 1.2 The Challenge of Continuous Domains

Geometry and algebra present a fundamentally different set of challenges.

- **Continuous Search Space:** Unlike the discrete grids of combinatorics, geometric problems live in $\mathbb{R}^2$ or $\mathbb{R}^3$. Solutions are often irrational numbers (e.g., $4\sqrt{3}$, $\frac{5+\sqrt{5}}{2}$), making exact pattern matching from small integer traces impossible.
    
- **Representation Complexity:** A geometric configuration is defined by a system of non-linear polynomial equations. The "answer" might be a length or area that relies on the precise intersection of these manifolds.
    
- **Lack of Canonical Forms:** Algebraic expressions can take many equivalent forms (e.g., $\frac{1}{\sqrt{2}} = \frac{\sqrt{2}}{2}$). Verification requires symbolic engines capable of recognizing these equivalences, which is harder than checking $5 = 5$.
    

Currently, solvers resort to numerical approximations (e.g., `mpmath` with 50+ digits) or heuristic reconstruction (PSLQ). While these yield high confidence, they are mathematically "unsound" in the formal sense—they do not constitute a proof. A numerical coincidence at the 50th decimal place, while rare, is not impossible, and floating-point errors can propagate in unstable systems.

### 1.3 The Research Objective

The objective is to close this gap by designing a system where _any_ answer found by _any_ method can be formally verified. If a neural network guesses that the area of a triangle is $3\sqrt{2}$, we need a formal pipeline that:

1. Constructs the exact algebraic context of the problem in Lean 4.
    
2. Formally defines the triangle based on the problem constraints.
    
3. Proves, via computation, that the area is indeed $3\sqrt{2}$.
    

This report proposes a unified framework that utilizes **Computational Algebraic Geometry** and **Real Closed Field theory** to translate these continuous problems into discrete, verifiable certificates.

---

## 2. Geometric Verification via Coordinate Constraint Satisfaction

The most robust pathway to automating geometry verification is not through synthetic axioms (Euclid's style) but through analytic geometry (Descartes' style). By mapping geometric propositions to algebraic systems, we transform theorem proving into polynomial identity testing.

### 2.1 The Coordinate Method as a Decision Procedure

The "Coordinate Method" involves assigning coordinates to a subset of points in a geometric configuration and defining the remaining points via equations. If a problem states "Let $M$ be the midpoint of $AB$", and we have $A=(x_A, y_A)$ and $B=(x_B, y_B)$, this introduces the constraints $x_M = \frac{x_A+x_B}{2}$ and $y_M = \frac{y_A+y_B}{2}$.

#### 2.1.1 Constraint Translation

Every standard geometric constraint translates to a polynomial equation over the field of coordinates $K$:

- **Collinearity:** Three points $A, B, C$ are collinear iff the determinant of their coordinates (augmented with a column of 1s) is zero, or equivalently via the cross product: $(x_B - x_A)(y_C - y_A) - (y_B - y_A)(x_C - x_A) = 0$.
    
- **Parallelism:** Lines $AB$ and $CD$ are parallel iff $(y_B - y_A)(x_D - x_C) - (x_B - x_A)(y_D - y_C) = 0$.
    
- **Perpendicularity:** Lines $AB$ and $CD$ are perpendicular iff the dot product of their displacement vectors is zero: $(x_B - x_A)(x_D - x_C) + (y_B - y_A)(y_D - y_C) = 0$.
    
- **Distance:** $d(A, B) = k$ translates to $(x_B - x_A)^2 + (y_B - y_A)^2 - k^2 = 0$.
    
- **Circle Membership:** Point $P$ lies on a circle with center $O$ and radius $r$ iff $(x_P - x_O)^2 + (y_P - y_O)^2 - r^2 = 0$.
    
- **Tangency:** A line is tangent to a circle if the distance from the center to the line equals the radius, or algebraically, if the discriminant of the intersection equation is zero.
    

The verification task is then: Given a set of coordinates for all points, verify that all constraint polynomials evaluate to zero.

#### 2.1.2 Exact Rational Geometry

A surprising number of competition problems, particularly those involving only lines, intersections, and ratios, can be solved entirely within the rational numbers $\mathbb{Q}$. If the "givens" of the problem can be placed on rational coordinates (e.g., a unit square with vertices at $(0,0), (1,0), (1,1), (0,1)$), and all subsequent points are constructed via linear intersections, the resulting coordinates remain in $\mathbb{Q}$.

For such problems, Lean 4’s verification strategy is trivial:

1. Define the points as `Rat` types (pairs of integers).
    
2. Define the constraint polynomials.
    
3. Use `native_decide` to evaluate the polynomials. Since rational arithmetic is decidable and exact, evaluating `poly.eval points == 0` is a computable boolean check.
    

### 2.2 Algebraic Number Coordinates and Field Extensions

Most geometry problems involve circles, distances, or angles that introduce square roots, moving the problem from $\mathbb{Q}$ to algebraic number fields.

#### 2.2.1 Constructible Numbers and Quadratic Towers

Compass-and-straightedge constructions generate "constructible numbers." These numbers lie in a field $K$ that can be reached via a tower of quadratic extensions: $\mathbb{Q} = F_0 \subset F_1 \subset \dots \subset F_n = K$, where $F_{i+1} = F_i(\sqrt{d_i})$ for some $d_i \in F_i$ and $d_i > 0$. To verify geometry in such fields, Lean needs a representation of $F_n$. Since the degree of the extension is $2^n$, elements can be represented as vectors of length $2^n$ over $\mathbb{Q}$. Addition is component-wise; multiplication is determined by the defining polynomials of the extensions.

**Lean Implementation Strategy:**

We can define a recursive structure for these fields.

Lean

```
inductive Constructible : Type

| base : Rat -> Constructible
| extension : Constructible -> Constructible -> Constructible -- a + b√d
```

However, a more efficient approach for `native_decide` is to define specific instances for required fields. If a problem involves $\sqrt{2}$ and $\sqrt{3}$, we work in $\mathbb{Q}(\sqrt{2}, \sqrt{3})$. This is a vector space of dimension 4 over $\mathbb{Q}$ with basis $\{1, \sqrt{2}, \sqrt{3}, \sqrt{6}\}$. Lean can verify identities in this field by performing exact arithmetic on the coefficients.

#### 2.2.2 General Algebraic Numbers

For problems involving cube roots or higher-order radicals (less common in standard geometry but possible), we represent numbers via their minimal polynomials. An algebraic number $\alpha$ is represented by a tuple $(P, I)$, where $P \in \mathbb{Q}[x]$ is irreducible and $I = [a, b]$ is a rational interval containing exactly one root of $P$. Equality checking $\alpha = \beta$ becomes checking if they have the same minimal polynomial and overlapping intervals consistent with uniqueness. Arithmetic operations (addition, multiplication) are performed using resultants (e.g., the polynomial for $\alpha + \beta$ is computed via $\text{Res}_y(P(x-y), Q(y))$). This allows Lean to verify identities like $\sqrt{2} + \sqrt{4} = \dots$ without explicit radical denesting.

### 2.3 Handling Transcendental Angles and Trigonometry

Geometry problems often specify angles (e.g., "$\angle ABC = 60^\circ$"). While trigonometric functions are transcendental, the specific values used in competitions ($30^\circ, 45^\circ, 60^\circ, 72^\circ$, etc.) yield algebraic numbers for their sine and cosine.

- $30^\circ, 60^\circ$: Involve $\sqrt{3}$.
    
- $45^\circ$: Involve $\sqrt{2}$.
    
- $72^\circ$ (pentagon): Involve $\sqrt{5}$.
    

Instead of using the real `sin` and `cos` functions (which block `native_decide` due to non-computability), we verify using the algebraic counterparts. We map angle constraints to algebraic ones:

- "$\angle ABC = 60^\circ$" $\rightarrow$ $\cos(\angle ABC) = 1/2$.
    
- Using the dot product formula: $\frac{\vec{BA} \cdot \vec{BC}}{|\vec{BA}| |\vec{BC}|} = \frac{1}{2}$. This converts the transcendental angle constraint into a polynomial constraint involving square roots (for the magnitudes), which falls back to the algebraic number verification case.
    

**Complex Numbers and Roots of Unity:** Alternatively, plane geometry can be verified using complex numbers. A rotation by $\theta$ is multiplication by $e^{i\theta}$. If $\theta$ is a rational multiple of $\pi$, $e^{i\theta}$ is a root of unity, which is an algebraic integer. We can represent points as elements of a cyclotomic field $\mathbb{Q}(\zeta_n)$. Lean's Mathlib has extensive support for cyclotomic polynomials, allowing us to verify rotational symmetries and regular polygon properties purely algebraically.

### 2.4 Computational Algebraic Geometry: Gröbner Bases

When we cannot simply "plug in" coordinates (e.g., the coordinates are variables $u, v$ satisfying some constraints), we need to prove that a conclusion polynomial $g(u, v)$ vanishes whenever the hypothesis polynomials $f_1(u, v), \dots, f_k(u, v)$ vanish.

This is the **Ideal Membership Problem**: Is $g \in \sqrt{\langle f_1, \dots, f_k \rangle}$?

By Hilbert's Nullstellensatz, this is equivalent to $1 \in \langle f_1, \dots, f_k, 1 - y \cdot g \rangle$.

**The Verification Workflow:**

1. **Discovery (Python):** Use a computer algebra system (SageMath/SymPy) to compute a Gröbner basis for the ideal generated by the hypotheses.
    
2. **Certificate Generation:** The CAS finds polynomials $c_1, \dots, c_k$ such that $g^N = \sum c_i f_i$ (or simply $g = \sum c_i f_i$ if the ideal is radical).
    
3. **Verification (Lean):** Lean receives the coefficients $c_i$. It does _not_ need to run Buchberger's algorithm (which is slow). It only needs to verify the polynomial identity $g - \sum c_i f_i = 0$. This is a simple ring check, efficiently handled by `ring` or `native_decide`.
    

This method, often called the "Wu's Method" or "Gröbner Basis Method" in automated geometry proving, is incredibly powerful. The Trace-to-Lean extension essentially acts as a "certificate checker" for Wu's method.

---

## 3. Algebraic Verification via Certificate Checking

Algebra problems typically ask to find a value satisfying an equation, prove an inequality, or solve a system.

### 3.1 Polynomial Identity and Substitution Verification

For problems asking to "Find $x$ such that...", verification is substitution. If the answer is a polynomial expression (e.g., $P(k) = k^2 + 1$), and the problem states $f(P(k)) = 0$, we substitute and check. Lean's `ring` tactic is a powerful normalizer for commutative semirings. It can verify any identity built from addition, multiplication, and integer powers. For fields (division), `field_simp` clears denominators, reducing the problem to a ring identity.

**Large Expression Verification:** Sometimes the expansion is too large for the kernel (e.g., degree 100 polynomials). In these cases, we can use **modular arithmetic** or **randomized checking** as a probabilistic proof (though not 100% formal), OR we can break the verification into smaller, lemma-based steps. However, `native_decide` is surprisingly fast at raw computation and can often handle expansions that choke the symbolic simplifier.

### 3.2 Root Isolation and Sturm's Theorem

When the solution to an algebra problem is a specific real root of a polynomial $P(x)$, simply stating "let $\alpha$ be a root of $P$" is insufficient if $P$ has multiple roots. We must specify _which_ root.

The standard way to identify a real algebraic number is by a pair $(P, [a, b])$ where $P \in \mathbb{Q}[x]$ and $[a, b]$ is an isolating interval containing exactly one root.

**Sturm's Theorem in Lean:** To verify that $\alpha$ is the unique root in $[a, b]$, Lean implements Sturm's Theorem :

1. Define the Sturm chain $S_0 = P, S_1 = P', S_{i} = -\text{rem}(S_{i-2}, S_{i-1})$.
    
2. Let $V(x)$ be the number of sign variations in the sequence $S_0(x), S_1(x), \dots$.
    
3. The number of distinct real roots in $(a, b]$ is $V(a) - V(b)$. Since the Sturm chain computation involves only polynomial division and evaluation at rational points $a, b$, it is entirely computable within `native_decide`. This provides a formally verified method to distinguish, say, $\sqrt{2}$ from $-\sqrt{2}$.
    

### 3.3 Sum-of-Squares (SOS) and Semidefinite Programming Certificates

Inequalities are notoriously hard to verify. "Prove $x^4 - x^2 - 2x + 3 \ge 0$ for all $x$." Simply sampling points is not a proof. The standard certificate for non-negativity of a polynomial $P(x)$ is a **Sum-of-Squares (SOS) decomposition**: finding polynomials $q_1, \dots, q_k$ such that $P(x) = \sum q_i(x)^2$.

**The Architecture:**

1. **Discovery (Python):** Use Semidefinite Programming (SDP) solvers (like Mosek or SCS via CVXPY) to find the coefficients of $q_i$.
    
2. **Certificate:** The list of polynomials $[q_1, \dots, q_k]$ is the certificate.
    
3. **Verification (Lean):** Lean verifies the identity $P - \sum q_i^2 = 0$. Since squares are axiomatically non-negative in ordered fields, this formally proves $P(x) \ge 0$.
    

This extends to constrained inequalities via the **Positivstellensatz**. If we need to prove $P(x) \ge 0$ assuming $g(x) \ge 0$, we look for SOS polynomials $s_0, s_1$ such that $P = s_0 + s_1 g$. The existence of such $s_i$ is a certificate of the implication.

### 3.4 The PSLQ Connection: Integer Relation Finding

Many competition answers involving constants like $e, \pi, \sqrt{2}$ are actually algebraic numbers or simple rational combinations. The PSLQ algorithm (in Python) can find integer coefficients $a_i$ such that $a_0 + a_1 x_1 + \dots + a_n x_n \approx 0$. If PSLQ suggests an identity, say $\alpha^2 + \beta^2 = 7$, Lean can verify this exact algebraic relation. This allows the solver to "snap" numerical approximations to exact algebraic forms which are then amenable to the `ring` tactic.

---

## 4. The Trace-to-Lean Architecture Extension

### 4.1 The Hybrid Solver-Verifier Loop

The proposed architecture splits the problem-solving process into two distinct phases: **Untrusted Discovery** and **Trusted Verification**.

1. **Input:** Natural language problem statement.
    
2. **Translation (LLM):** Convert problem to Python (for solving) and Lean (for verification statement).
    
3. **Solver Phase (Python):**
    
    - For Geometry: Use SymPy/Coordinate Descent to find coordinates. Use Gröbner bases to find dependency coefficients.
        
    - For Algebra: Use numerical root finding, PSLQ, or SDP to find roots and certificates.
        
    - **Output:** A "Proof Certificate" JSON containing exact coordinates, minimal polynomials, and auxiliary polynomials (SOS terms, Gröbner coefficients).
        
4. **Verifier Phase (Lean 4):**
    
    - Parse the JSON certificate.
        
    - Reconstruct the algebraic objects (polynomials, fields).
        
    - Execute `native_decide` or tactic scripts (`ring`, `linarith`) to check constraints.
        
    - **Output:** `True` (Verified) or Error message.
        

### 4.2 Data Serialization and The Python-Lean Bridge

Data transfer is a critical link. We cannot pass floating point numbers due to precision loss. We must pass **exact descriptions**.

**JSON Schema for Geometry:**

JSON

```
{
  "type": "geometry_certificate",
  "field_extension": {
    "base": "Q",
    "generators": ["sqrt(2)", "sqrt(3)"]
  },
  "points": {
    "A": ,
    "B": ,
    "C": ["5", "5*sqrt(3)"] 
  },
  "target_value": "25*sqrt(3)", // Area
  "proof_method": "direct_evaluation"
}
```

**Lean 4 Parser:** Lean 4 has robust JSON parsing capabilities (`Lean.Data.Json`). We implement a custom parser that reads this JSON and utilizes metaprogramming to construct the corresponding Lean expressions (e.g., building the term `(5 : ℚ) + (5 : ℚ) * Real.sqrt 3`).

### 4.3 Lean 4 Implementation: `native_decide` and FFI

`native_decide` is the workhorse. It compiles the verification goal into C++ and executes it. This is orders of magnitude faster than kernel reduction (`rfl`).

- **GMP Integration:** Lean 4 uses GMP for arbitrary-precision integers, ensuring that checking $x = 10^{100}$ is efficient and correct.
    
- **FFI (Foreign Function Interface):** For extremely heavy algebraic computations (e.g., factoring degree 50 polynomials), Lean can link to external C libraries like **Arb** (for rigorous interval arithmetic) or **FLINT** (for number theory). However, for the goal of "100% formal guarantees," we prefer pure Lean implementations of the verification logic (like Horner's method for polynomial evaluation) to minimize the trusted code base, relying on `native_decide` to execute them efficiently.
    

---

## 5. A Unified Verification Hierarchy

Not all problems require the heavy machinery of algebraic geometry. We propose a hierarchical dispatcher that selects the simplest sufficient verification level.

### 5.1 Level 0: Integer/Modular Arithmetic

- **Domain:** Combinatorics, Number Theory.
    
- **Tools:** `Nat`, `Int`, `ZMod`.
    
- **Tactic:** `native_decide`.
    
- **Example:** "Find the remainder of $2^{100}$ mod 7." $\to$ Direct computation.
    

### 5.2 Level 1: Rational Arithmetic

- **Domain:** Metric Geometry with rational coordinates, simple linear algebra.
    
- **Tools:** `Rat`.
    
- **Tactic:** `norm_num`, `ring`.
    
- **Example:** "Intersection of two lines with rational equations."
    

### 5.3 Level 2: Algebraic Number Fields

- **Domain:** Euclidean Geometry (circles, angles), Algebra (radicals).
    
- **Tools:** `NumberField`, `AdjoinRoot`.
    
- **Method:** Computation in $K = \mathbb{Q}(\alpha_1, \dots, \alpha_k)$. Verification via field arithmetic.
    
- **Example:** "Area of a triangle with side lengths $\sqrt{2}, \sqrt{3}, \sqrt{5}$."
    

### 5.4 Level 3: Real Algebraic Geometry

- **Domain:** Inequalities, Optimization, Non-linear constraints.
    
- **Tools:** SOS Certificates, Positivstellensatz.
    
- **Method:** Verify $P(x) = \sum q_i(x)^2$.
    
- **Example:** "Minimum value of $x^4 + y^4 - x^2 y^2$."
    

### 5.5 Level 4: Transcendental / Approximation Fallback

- **Domain:** Problems with no algebraic solution form (rare in competitions).
    
- **Tools:** Verified Interval Arithmetic (`LeanCert`).
    
- **Method:** Prove answer $\in$ where $|L-R| < \epsilon$.
    
- **Example:** "Estimate $\sin(1)$ to 3 decimal places."
    

---

## 6. Implementation Feasibility and Complexity Analysis

### 6.1 Computational Complexity

The bottleneck is almost always in the _generation_ of the certificate (finding the Gröbner basis or SOS decomposition), which happens in Python/C++. The _verification_ in Lean is generally polynomial time in the size of the certificate.

- **Polynomial Evaluation:** Linear in the number of terms (with Horner's method).
    
- **Sturm's Theorem:** Euclidean algorithm is quadratic in degree.
    
- **SOS Check:** Expanding $\sum q_i^2$ is quadratic in the size of $q_i$.
    

Given competition limits (e.g., 6 minutes per problem), verification that takes even 10-20 seconds is acceptable. `native_decide` can perform millions of ops per second, making this entirely feasible for competition-level complexities.

### 6.2 The "Integer Answer" Exploit

A crucial heuristic for AIME/AIMO is that the final answer is often an integer $0-999$. This simplifies verification enormously. Instead of proving $x = \sqrt{3 + 2\sqrt{2}}$, we can prove $x^2 = 3+2\sqrt{2}$ and $x \approx 2.41$ via interval arithmetic, then refine to prove $x$ is not an integer—or more likely, if the answer is an integer, the algebraic expression simplifies to a rational. If we suspect the answer is 5, we just check $P(5) = 0$ and $5 \in [a, b]$. This bypasses complex field arithmetic in favor of integer verification.

---

## 7. Templates and Prototype Logic

### 7.1 Template: Geometry Constraint Verification

**Python Side (Discovery):**

Python

```
import sympy as sp
# Define constraint: Point C is intersection of circle A and circle B
# Solve for C.x, C.y symbolically
# Output: C = (3/5, 4/5)
```

**Lean Side (Verification):**

Lean

```
import Mathlib.Tactic

-- Define the algebraic context (Rational for this example)
def Ax : ℚ := 0
def Ay : ℚ := 0
def Bx : ℚ := 2
def By : ℚ := 0
def Cx : ℚ := 1
def Cy : ℚ := 1.732 -- Wait, this is approximate. 
-- Correct approach: Use Algebraic Number Field
def K := AdjoinRoot (X^2 - 3) -- Field Q(sqrt(3))
def sqrt3 : K := AdjoinRoot.root
def Cy_alg : K := sqrt3

theorem verify_triangle_equilateral : 
  let dist_sq (x1 y1 x2 y2 : K) := (x1 - x2)^2 + (y1 - y2)^2
  dist_sq Ax Ay Bx By = 4 ∧ 
  dist_sq Ax Ay Cx Cy_alg = 4 ∧
  dist_sq Bx By Cx Cy_alg = 4 := by
  native_decide -- Computes in Q(sqrt(3))
```

### 7.2 Template: Algebraic Inequality (SOS)

**Lean Side:**

Lean

```
import Mathlib.Tactic

-- Problem: Prove x^4 - 2x^2 + 2 >= 1
-- Python discovers: x^4 - 2x^2 + 1 = (x^2 - 1)^2
theorem algebra_inequality (x : ℝ) : x^4 - 2*x^2 + 2 ≥ 1 := by
  have h : x^4 - 2*x^2 + 2 - 1 = (x^2 - 1)^2 := by ring
  rw [h]
  exact sq_nonneg (x^2 - 1)
```

This template shows how `ring` acts as the certificate checker for the identity discovered by the external solver.

---

## 8. Conclusion

Extending Trace-to-Lean to geometry and algebra is not only feasible but represents the next logical leap in automated reasoning. By shifting the burden of _finding_ complex geometric configurations and algebraic roots to external, untrusted solvers (Python/SymPy), and reserving Lean 4 strictly for _verifying_ the resulting certificates (coordinates, identities, SOS decompositions), we can achieve 100% formal guarantees.

The integration of specific mathematical technologies—**Sturm's theorem** for root isolation, **Gröbner bases** for ideal membership, **SOS** for inequalities, and **constructible field arithmetic** for geometry—creates a robust pipeline. The `native_decide` tactic serves as the computational engine that makes this verification tractable within competition time limits. This Unified Verification Framework essentially treats Lean 4 not just as a logic checker, but as a verified computer algebra system, capable of "calculating" the truth of continuous mathematical statements through rigorous discrete approximations and algebraic identities.

### Recommendations for Development

1. **Prioritize the "Integer Answer" Exploit:** Implement tactics that specifically check if a complex algebraic number simplifies to a target integer.
    
2. **Build a `Constructible` Number Type:** Create a streamlined Lean 4 library for quadratic extension towers optimized for `native_decide`.
    
3. **Develop the JSON Bridge:** Standardize the serialization format for geometric certificates to decouple the Python solver from the Lean verifier.
    

This architecture paves the way for a neuro-symbolic solver capable of achieving SOTA performance on benchmarks like AIMO, with the unique differentiator of providing absolute mathematical certainty.