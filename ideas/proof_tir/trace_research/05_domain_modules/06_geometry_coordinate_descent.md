# Deep Research: Geometry — Coordinate Descent and Numerical Verification

## Research Objective

Geometry problems constitute ~25% of competition math and are NOT amenable to the Berlekamp-Massey approach. We need a completely different strategy: numerical optimization to find coordinates, followed by computational verification. This research explores the "Coordinate Descent" module for geometry.

## Context

The approach for geometry:
1. LLM translates geometric constraints into algebraic equations
2. Coordinate Descent (or other optimizer) finds numerical coordinates satisfying constraints
3. With concrete coordinates, we compute the answer numerically
4. Verification: check that constraints are satisfied to high precision, and answer is consistent

This is NOT formal proof — it's "proof by computation with high confidence."

## Research Questions

### Part A: Geometry Problem Classification

#### 1. Types of Geometry Problems in Competitions
- Euclidean plane geometry (triangles, circles, polygons)
- Coordinate geometry (explicit coordinates given)
- Constructive geometry (prove existence of a point/line/circle)
- Measurement problems (find length, area, angle)
- Locus problems (find set of points satisfying condition)
- Olympiad-style "prove X = Y" or "show collinear/concurrent"

#### 2. What's Computationally Tractable
- Which geometry problems can be solved by numerical methods?
- What requires symbolic/algebraic approaches?
- What's inherently impossible to verify numerically?

#### 3. Common Geometric Objects
- Points, lines, circles, triangles, polygons
- How do we represent each numerically?
- What precision is needed for competition-level accuracy?

### Part B: Constraint Translation

#### 4. LLM as Constraint Translator
- How do we prompt an LLM to extract geometric constraints from problem text?
- Example: "Let ABC be a triangle with AB = 5, BC = 7, CA = 8" → 
  - `dist(A, B) = 5`
  - `dist(B, C) = 7`
  - `dist(C, A) = 8`
- What's the success rate of this translation?

#### 5. Constraint Types
- Distance constraints: `dist(P, Q) = d`
- Angle constraints: `angle(A, B, C) = θ`
- Collinearity: `collinear(A, B, C)`
- Concurrency: `concurrent(line1, line2, line3)`
- Incidence: `P on line L` or `P on circle C`
- Perpendicularity: `perp(line1, line2)`
- Parallelism: `parallel(line1, line2)`
- Tangency: `tangent(circle, line)` or `tangent(circle1, circle2)`

#### 6. Constraint Representation
- How do we encode each constraint as an algebraic equation?
- What's the best form for optimization (squared residuals)?
- Handling angular constraints (periodicity issues)?

### Part C: Numerical Optimization

#### 7. Coordinate Descent Deep Dive
- How does Coordinate Descent work?
- Why might it be preferred over gradient descent for geometry?
- Convergence guarantees for convex vs non-convex constraint sets?
- When does it get stuck in local minima?

#### 8. Alternative Optimizers
- Gradient Descent with momentum
- L-BFGS (quasi-Newton)
- Nelder-Mead (derivative-free)
- Basin-hopping (global optimization)
- Particle Swarm Optimization
- When to use each?

#### 9. Initialization Strategies
- How do we initialize coordinates for geometric objects?
- Random initialization vs structured (e.g., unit circle)?
- Multiple restarts to escape local minima?
- Using LLM to suggest initial configuration?

#### 10. Constraint Satisfaction vs Optimization
- Some constraints are hard (must be satisfied exactly)
- Some are soft (minimize violation)
- How do we formulate the objective function?
- Lagrangian methods? Penalty methods?

### Part D: Precision and Verification

#### 11. Numerical Precision Requirements
- What precision do we need for competition geometry?
- When is 64-bit float sufficient?
- When do we need `mpmath` or extended precision?
- How does error accumulate in geometric computations?

#### 12. Verification Criteria
- How do we know the solution is "correct enough"?
- Tolerance thresholds: 1e-6? 1e-10? 1e-15?
- What if the problem has multiple valid solutions?
- Detecting degeneracy (collinear points that shouldn't be, etc.)

#### 13. Answer Extraction
- Once we have coordinates, how do we compute the answer?
  - Length: distance formula
  - Area: shoelace formula, cross product
  - Angle: dot product, atan2
  - Ratio: direct computation
- Rounding to competition format (usually integer or simple fraction)

#### 14. Confidence Scoring
- How confident are we in the numerical answer?
- Can we bound the error?
- When should we distrust the result?

### Part E: Special Geometric Techniques

#### 15. Circle Geometry
- Representing circles: center + radius vs equation
- Circle-circle intersection
- Circle-line intersection
- Power of a point
- Inversion (does numerical optimization handle this?)

#### 16. Triangle Centers
- Centroid, incenter, circumcenter, orthocenter
- Computing from vertex coordinates
- When are these the answer vs intermediate constructions?

#### 17. Projective and Affine Geometry
- Homogeneous coordinates
- Cross-ratio (projective invariant)
- Does coordinate descent work in projective space?

#### 18. Algebraic Geometry Techniques
- Gröbner bases for polynomial constraint systems
- When is symbolic solving better than numerical?
- Hybrid approaches?

### Part F: Integration with Verification

#### 19. From Numerical to Verified
- Can we use the numerical solution to guide a formal proof?
- "Witness" approach: numerical solution as existence proof
- Can we verify geometric facts with `native_decide`?

#### 20. Lean Geometry
- What geometry is formalized in Mathlib?
- Can we verify geometric computations in Lean?
- Or is geometry inherently the "fallback to TIR" category?

### Part G: AlphaGeometry Connection

#### 21. AlphaGeometry Architecture
- How does DeepMind's AlphaGeometry work?
- Symbolic engine + LLM for auxiliary point generation
- What can we learn from their approach?
- Is it feasible in a competition setting?

#### 22. Synthetic Geometry vs Coordinate Geometry
- When is synthetic reasoning superior?
- When is coordinate bashing the right call?
- Can we detect which approach fits the problem?

### Part H: Implementation

#### 23. Python Implementation
- Best libraries: scipy.optimize, cvxpy, casadi?
- Custom coordinate descent vs off-the-shelf?
- Performance: how many iterations typical? How fast?

#### 24. Edge Cases and Failure Modes
- Parallel lines (no intersection)
- Tangent circles (single intersection)
- Degenerate triangles
- Points at infinity
- How do we detect and handle these?

## Desired Output Format

1. **Geometry Problem Taxonomy**: Classification with examples
2. **Constraint Encoding Guide**: How to translate geometry to algebra
3. **Optimizer Comparison**: Which algorithm for which situation
4. **Precision Analysis**: When each precision level is needed
5. **Verification Protocol**: How to validate numerical solutions
6. **Implementation Guide**: Libraries, code patterns, best practices
7. **AlphaGeometry Comparison**: What we can adopt
8. **Failure Mode Catalog**: What can go wrong and how to handle it

















# Deep Research: Geometry — Coordinate Descent and Numerical Verification

## 1. Introduction

The pursuit of automated reasoning in geometry has traditionally bifurcated into two distinct methodological lineages: the symbolic and the numerical. The symbolic lineage, exemplified by the pioneering work of Wu Wen-Tsun on the characteristic set method and the application of Gröbner bases, treats geometric entities as algebraic variables within polynomial rings. These methods seek to establish the truth of a proposition through rigorous algebraic manipulation, verifying whether the conclusion belongs to the ideal generated by the hypotheses. While theoretically complete for proving equality-type theorems in complex projective geometry, symbolic methods frequently encounter insurmountable barriers in computational complexity—often double exponential in the worst case—when applied to metric problems involving inequalities, transcendental functions, or the complex constructive requirements typical of high-level competitions like the International Mathematical Olympiad (IMO).

In contrast, the numerical lineage views geometry through the lens of analysis and optimization. Here, a geometric configuration is not a set of polynomial ideals but a concrete realization in $\mathbb{R}^n$, a vector of coordinates that minimizes a "loss function" representing the violation of geometric constraints. This approach, often termed "proof by computation," has historically been viewed as less rigorous due to the inherent approximations of floating-point arithmetic. However, recent advancements in neuro-symbolic AI, such as Google DeepMind's AlphaGeometry, have demonstrated that integrating statistical intuition with deductive engines can solve Olympiad-level problems. Yet, a significant subset of competition mathematics—approximately 25%—remains resistant to both pure symbolic deduction and sequence prediction strategies like Berlekamp-Massey. These problems typically require the determination of specific metric values (lengths, areas, angles) or the construction of loci, tasks for which symbolic engines are ill-suited due to the "intermediate expression swell" phenomena.

This research report explores a third paradigm: a high-precision **Coordinate Descent (CD) Module** designed to solve geometry problems through numerical optimization followed by computational verification. The central thesis of this work is that geometric problem solving can be effectively modeled as a non-convex optimization problem where the objective is to find a coordinate configuration that satisfies all constraints to a precision sufficient for symbolic recovery. This framework leverages the capabilities of Large Language Models (LLMs) as constraint translators, transforming natural language problem statements into algebraic loss functions. Subsequently, robust optimization algorithms—specifically Coordinate Descent and its variants—navigate the energy landscape to locate valid geometric configurations. Finally, integer relation algorithms such as PSLQ and LLL bridge the gap between floating-point approximations and exact symbolic answers.

This document provides an exhaustive analysis of this "Coordinate Descent" strategy. It categorizes the geometric problem space, details the algebraic formulation of constraints, evaluates the convergence dynamics of various optimization algorithms in non-convex geometric landscapes, and establishes rigorous protocols for numerical precision and verification. The goal is to define a system capable of "proof by computation with high confidence," offering a pragmatic and scalable alternative to purely symbolic theorem provers for the metric-heavy segment of competition geometry.

## Part A: Geometry Problem Classification

To design an effective numerical solver, one must first map the terrain of competition geometry. The efficacy of numerical optimization is highly dependent on the problem type; what works for a constructive problem may fail for a general proof.

### 2. Types of Geometry Problems in Competitions

The taxonomy of geometry problems in competitions such as the IMO, AIME, and AMC can be broadly classified by their goal structure and the nature of their constraints.

#### 2.1 Euclidean Plane Geometry

These are the classical problems involving the fundamental objects of the Euclidean plane: triangles, circles, and polygons, governed by axioms of incidence, congruence, and similarity.

- **Structure:** Typically stated in coordinate-free terms (e.g., "Let $ABC$ be a triangle with circumcenter $O$").
    
- **Numerical Suitability:** High. While phrased synthetically, these problems implicitly define a rigid or semi-rigid system of constraints. A numerical solver can arbitrarily fix the coordinate system (e.g., placing $A$ at the origin and $B$ on the x-axis) to resolve the translational and rotational degrees of freedom, reducing the problem to finding the remaining coordinates.
    

#### 2.2 Coordinate Geometry

These problems are explicitly defined in the Cartesian plane, often involving parabolas, ellipses, or specific point coordinates.

- **Structure:** "Find the distance from point $(2,3)$ to the line $y=2x+5$."
    
- **Numerical Suitability:** Maximal. These problems are the "native tongue" of a Coordinate Descent solver. The translation step is minimal as the constraints are already algebraic. The challenge here often lies in the precision required to differentiate between very close solutions in high-degree curves.
    

#### 2.3 Constructive Geometry

These problems require proving the existence of a geometric configuration or constructing a specific object that satisfies a set of conditions.

- **Structure:** "Construct a circle tangent to two given lines and passing through a point $P$."
    
- **Numerical Suitability:** Moderate to High. Numerical methods treat construction as a satisfaction problem. If the optimizer reaches a zero-loss state, the object "exists" in the numerical sense. The difficulty arises when the construction is impossible (no solution exists); numerical optimizers may simply oscillate or converge to a local minimum with non-zero loss, making it difficult to distinguish "impossible" from "hard to find".
    

#### 2.4 Measurement Problems

This category constitutes the majority of "short answer" competition problems (e.g., AIME).

- **Structure:** "Find length $AB$," "Find the area of $\triangle ABC$," "Find $\cos \angle BAC$."
    
- **Numerical Suitability:** This is the primary target for the proposed module. Symbolic engines often struggle here because they maintain complex nested radicals. A numerical solver computes the final state and simply measures the quantity. The critical success factor is the ability to recover the exact symbolic value (e.g., identifying $1.73205\dots$ as $\sqrt{3}$) from the floating-point result.
    

#### 2.5 Locus Problems

- **Structure:** "Find the set of all points $P$ such that..."
    
- **Numerical Suitability:** High, via sampling. A numerical solver can trace a locus by discretizing one degree of freedom (e.g., moving a driving point $M$ along a line in small steps) and re-optimizing the system at each step. The resulting trail of points can be analyzed using regression to fit curves (lines, circles, conics), effectively "discovering" the locus equation.
    

#### 2.6 Olympiad-Style Proofs

- **Structure:** "Prove that lines $AP$, $BQ$, and $CR$ are concurrent."
    
- **Numerical Suitability:** High for verification, low for formal proof. A numerical solver verifies the theorem for a _specific_ random instance. If the distance between the intersection points of $AP \cap BQ$ and $AP \cap CR$ is $10^{-16}$, the theorem holds with extremely high probability for that instance. Repeated testing on random initializations provides "probabilistic proof," which is often sufficient for generating the insight needed to construct a formal proof.
    

### 3. Computational Tractability

#### 3.1 Solvable by Numerical Methods

Problems involving **fixed geometric configurations** (rigid graphs) are ideally suited for numerical optimization. If the number of constraints equals the number of degrees of freedom (minus the rigid body motions), the system usually has a finite number of discrete solutions (roots). Numerical methods like Newton-Raphson or Coordinate Descent can converge to these roots rapidly. Inequalities (e.g., "point $P$ is inside the triangle") are also naturally handled by numerical optimizers through penalty functions (e.g., $\max(0, -d(P, \text{boundary}))^2$), whereas they pose significant difficulties for algebraic methods like Wu's method which rely on equality constraints.

#### 3.2 Requiring Symbolic Approaches

Problems asking for general relationships in **under-constrained systems** are difficult to "prove" numerically. For instance, "For any triangle $ABC$, prove $X$ lies on $Y$." A numerical solver can only show this for specific instances of $ABC$. While running 100 random instances suggests truth, it does not provide the derivation or the logical "why," which is the essence of a formal proof. Furthermore, problems involving **combinatorial geometry** (e.g., "Find the maximum number of regions...") or discrete point sets often lack the smooth gradients required for efficient descent algorithms.

#### 3.3 Inherently Impossible to Verify Numerically

Problems where the distinction between cases relies on precision exceeding the machine's capability are problematic. For example, distinguishing whether a point is _exactly_ on a line or just $10^{-30}$ away requires arbitrary precision arithmetic, which becomes computationally prohibitive. Additionally, **singularities** and **bifurcation points**—where the number of solutions changes or the Jacobian becomes singular—can cause numerical solvers to diverge or behave erratically, failing to verify the geometry reliably.

### 4. Common Geometric Objects and Representation

The transition from Euclidean axioms to Python code requires a robust numerical representation.

- **Points:** The fundamental primitive. A point $P$ is represented as a coordinate vector $\mathbf{x} \in \mathbb{R}^2$ (or $\mathbb{R}^3$).
    
    - _Representation:_ `[x, y]`
        
- **Lines:** Can be represented by the equation $ax + by + c = 0$ or by two points on the line. The two-point representation is often preferred in optimization to avoid the singularity of vertical lines (infinite slope) and to keep the variables homogeneous (all variables are point coordinates).
    
    - _Representation:_ Implicitly defined by two point indices `(idx_P1, idx_P2)`.
        
- **Circles:** Represented by a center point $C$ and a scalar radius $r$, or by three points on the circumference. The Center-Radius form is generally more stable for optimization as it separates the position and size parameters.
    
    - _Representation:_ `[cx, cy, r]`
        
- **Polygons:** Ordered lists of point indices.
    
    - _Representation:_ `[idx_P1, idx_P2,..., idx_Pn]`
        

**Precision Requirements:** Competition problems often involve distinguishing between values like $\sqrt{2} + \sqrt{3} \approx 3.146$ and $\pi \approx 3.141$. Standard 32-bit floats (7 decimal digits) are insufficient. **64-bit doubles** (15-17 decimal digits) are the minimum baseline. However, to utilize integer relation algorithms like PSLQ effectively, one often needs 50-100 digits of precision to avoid false positives. Thus, the system must support **Arbitrary Precision Arithmetic** (e.g., via `mpmath` or GMP) for the final verification and extraction stages.

## Part B: Constraint Translation

The second phase of the solver is the translation of natural language geometry into a formal system of algebraic equations suitable for optimization.

### 5. LLM as Constraint Translator

The LLM acts as a semantic parser, converting the unstructured text of a competition problem into a structured computational graph.

#### 5.1 Prompting Strategies

To achieve high fidelity, the LLM prompt must be structured to output a standardized format, such as JSON or a Python Domain Specific Language (DSL).

- **Chain-of-Thought:** The prompt should encourage the LLM to first list the geometric entities ("Triangle ABC implies points A, B, C") and then the constraints ("AB=5 implies distance constraint").
    
- **Few-Shot Prompting:** Providing examples of translation is crucial.
    
    - _Example Input:_ "Let $ABC$ be a triangle with $AB=5, BC=7, CA=8$."
        
    - _Example Output:_
        
        JSON
        
        ```
        {
          "variables":,
          "constraints":
        }
        ```
        
- **Success Rate:** Current state-of-the-art models (e.g., GPT-4, Gemini 1.5 Pro) show high success rates (>90%) for standard phrasing but struggle with implicit constraints (e.g., "tangent" implies touching at one point, but also implies a specific distance relationship) or complex constructions involving loci.
    

### 6. Constraint Types

A robust geometry solver must handle a diverse vocabulary of geometric relationships.

- **Distance:** `dist(P, Q) = d`. The most common metric constraint.
    
- **Angle:** `angle(A, B, C) = θ`. Requires handling periodicity.
    
- **Collinearity:** `collinear(A, B, C)`. Points lie on a single line.
    
- **Concurrency:** `concurrent(line1, line2, line3)`. Three lines meet at a single point.
    
- **Incidence:** `P on line L` or `P on circle C`.
    
- **Perpendicularity:** `perp(line1, line2)`. Angle is $90^\circ$.
    
- **Parallelism:** `parallel(line1, line2)`. Slopes are equal.
    
- **Tangency:** `tangent(circle, line)` or `tangent(circle1, circle2)`. Distance between centers equals sum/difference of radii.
    

### 7. Constraint Representation (Algebraic Formulation)

For the optimizer, each constraint must be encoded as a residual function $f(\mathbf{x})$ such that $f(\mathbf{x}) = 0$ implies satisfaction. We minimize the sum of squared residuals: $L = \sum f_i(\mathbf{x})^2$.

#### 7.1 Distance Constraints

Instead of using the Euclidean distance formula $\sqrt{(x_1-x_2)^2 + (y_1-y_2)^2} = d$, which has a gradient singularity at $d=0$ and is computationally expensive due to the square root, we use the **squared distance**:

$$f_{dist}(\mathbf{x}) = (x_1-x_2)^2 + (y_1-y_2)^2 - d^2$$

This formulation is a polynomial (quadratic), having smooth gradients everywhere, which significantly aids convergence.

#### 7.2 Collinearity

Collinearity of points $A, B, C$ can be expressed using the cross product of vectors $\vec{AB}$ and $\vec{AC}$ in 2D (which represents the signed area of the parallelogram spanned by them):

$$f_{coll}(\mathbf{x}) = (x_B - x_A)(y_C - y_A) - (y_B - y_A)(x_C - x_A)$$

Minimizing $f_{coll}^2$ forces the area of triangle $ABC$ to zero.

#### 7.3 Perpendicularity and Parallelism

- **Perpendicularity:** The dot product of direction vectors must be zero.
    
    $$f_{perp} = \vec{u} \cdot \vec{v} = (x_2-x_1)(x_4-x_3) + (y_2-y_1)(y_4-y_3) = 0$$
    
- **Parallelism:** The cross product (2D determinant) of direction vectors must be zero.
    
    $$f_{para} = \vec{u} \times \vec{v} = (x_2-x_1)(y_4-y_3) - (y_2-y_1)(x_4-x_3) = 0$$
    

#### 7.4 Incidence (Point on Line/Circle)

- **Point P on Line AB:** Equivalent to `collinear(A, B, P)`.
    
- **Point P on Circle (Center C, Radius r):** Equivalent to `dist(P, C) = r`.
    
    $$f_{inc} = (x_P - x_C)^2 + (y_P - y_C)^2 - r^2$$
    

#### 7.5 Angular Constraints

Angles are tricky due to periodicity ($0 = 2\pi$) and sign ambiguity.

- **Cosine Law:** Use the dot product to enforce the cosine of the angle.
    
    $$\vec{BA} \cdot \vec{BC} = |\vec{BA}| |\vec{BC}| \cos \theta$$
    
    Squared residual:
    
    $$f_{angle\_cos} = (\vec{BA} \cdot \vec{BC})^2 - (|\vec{BA}|^2 |\vec{BC}|^2 \cos^2 \theta)$$
    
    _Note:_ This does not distinguish between $\theta$ and $-\theta$ or $\theta$ and $180-\theta$ fully without careful handling.
    
- **Sine Law (Cross Product):** To fix orientation (signed angle), constrain the cross product:
    
    $$\vec{BA} \times \vec{BC} = |\vec{BA}| |\vec{BC}| \sin \theta$$
    
    Combining both ensures the correct angle and quadrant.
    

#### 7.6 Tangency

Tangency between two circles (centers $C_1, C_2$, radii $r_1, r_2$) implies the distance between centers is the sum (external) or difference (internal) of radii.

$$f_{tan} = (dist(C_1, C_2)^2 - (r_1 \pm r_2)^2)$$

The choice between sum and difference creates a discrete ambiguity (branching) in the problem, often requiring the solver to explore multiple basins of attraction.

## Part C: Numerical Optimization

The translation phase yields a cost function $F(\mathbf{X}) = \sum w_i f_i(\mathbf{X})^2$. The core task is to minimize this function to zero.

### 8. Coordinate Descent Deep Dive

Coordinate Descent (CD) is an iterative optimization algorithm that minimizes the objective function with respect to one coordinate (or block of coordinates) at a time, holding others fixed.

#### 8.1 Mechanism

For a variable vector $\mathbf{x} = (x_1, \dots, x_n)$, at step $k$, CD updates $x_i$ by solving:

$$x_i^{(k+1)} = \arg\min_{\xi} F(x_1^{(k+1)}, \dots, \xi, \dots, x_n^{(k)})$$

In the context of geometry, where constraints are typically quadratic in terms of distance, the partial derivative $\frac{\partial F}{\partial x_i}$ is often a cubic polynomial. Finding the roots of a cubic is an analytical operation (using Cardano’s formula) or a very fast numerical one. This allows for **exact line search**, meaning we find the _optimal_ new value for $x_i$ in one step, rather than just taking a gradient step.

#### 8.2 Why CD for Geometry?

- **Sparsity:** Geometric constraints are local. A point $P$ usually interacts with only a few other points. When updating $P$, we only need to recompute the few constraints involving $P$, not the entire graph. This $O(1)$ update cost makes CD extremely fast per iteration.
    
- **No Hessian:** Unlike Newton's method, CD does not require computing or inverting a Hessian matrix, which scales as $O(N^3)$.
    
- **Natural "Sketching":** CD mimics how a human might refine a drawing—tweaking one point, then another, to make things fit.
    

#### 8.3 Convergence and Local Minima

CD is guaranteed to converge to a stationary point for smooth, convex functions. However, geometric constraint functions are **non-convex**. For example, the constraint "distance from origin is 5" defines a circle, which is a non-convex set.

- **Stuck in Minima:** CD can get stuck in "Nash equilibria" where no single point can move to improve the error, even though a simultaneous move of two points could. For example, two points connected by a rigid rod stuck in a narrow channel might not be able to move individually.
    
- **Block Coordinate Descent (BCD):** To mitigate this, we update **blocks** of variables—e.g., updating both $(x, y)$ of a point simultaneously. Since the sub-problem for a single point is still low-dimensional (2D), it remains computationally cheap while avoiding many single-variable traps.
    

### 9. Alternative Optimizers

While CD is efficient, the rugged landscape of geometric loss functions often necessitates more robust or hybrid approaches.

- **Gradient Descent (GD) with Momentum (Adam):** Standard GD can be slow in "canyons" (common in geometry). Momentum methods like Adam help navigate these canyons and escape shallow local minima. They are good for the "coarse" phase of finding an approximate shape.
    
- **L-BFGS (Limited-memory BFGS):** A Quasi-Newton method that approximates the Hessian. It is generally faster and more precise than GD for smooth functions and is the industry standard for unconstrained optimization. It is excellent for the "fine-tuning" phase to reach high precision.
    
- **Nelder-Mead:** A gradient-free simplex method. Useful if the loss function is non-differentiable (e.g., using `max` or `abs` functions). However, it scales poorly with dimensionality and is generally too slow for complex geometry.
    
- **Basin-Hopping / Simulated Annealing:** Global optimization strategies that introduce randomness to jump out of local minima. Crucial for constructive problems where the valid configuration might be in a completely different part of the search space.
    
- **Particle Swarm Optimization (PSO):** Maintains a population of candidate solutions. Effective for exploring the space but computationally expensive for high-precision refinement.
    

**Recommendation:** A hybrid strategy. Use **Basin-Hopping** or **Adam** to find a candidate solution (residual $< 10^{-2}$), then switch to **L-BFGS** or **Block Coordinate Descent** for rapid convergence to machine precision (residual $< 10^{-15}$).

### 10. Initialization Strategies

The non-convexity of geometry means the final result depends heavily on the starting position.

- **Random Initialization:** Scatter points randomly in a unit box. Simple but prone to local minima.
    
- **Structured Initialization:** Place points on a unit circle or grid. Useful for generic theorems.
    
- **LLM-Guided Initialization:** The LLM can predict approximate coordinates. For "Triangle ABC...", the LLM might suggest $A=(0,0), B=(1,0), C=(0.5, 0.8)$. This "warm start" places the optimizer in the correct basin of attraction, significantly increasing success rates.
    
- **Graph Decomposition:** Algorithms based on **Laman Graphs** decompose the constraint graph into rigid sub-structures. The solver initializes and solves these small clusters independently (e.g., formed by 3 points) and then assembles them. This reduces the dimensionality of the initialization problem.
    

### 11. Constraint Satisfaction vs. Optimization

- **Hard Constraints:** Must be satisfied exactly (e.g., definitions).
    
- **Soft Constraints:** Desirable properties (e.g., "keep the drawing centered").
    
- **Objective Function:**
    
    $$\min \left( \sum w_{hard} f_{hard}^2 + \sum w_{soft} f_{soft}^2 \right)$$
    
- **Penalty Methods:** Start with low $w_{hard}$ and increase it iteratively (continuation method). This allows the solver to explore the space "loosely" before locking into a precise configuration.
    
- **Lagrangian Multipliers:** A more sophisticated approach that introduces dual variables to enforce constraints exactly, avoiding the ill-conditioning of large penalty weights.
    

## Part D: Precision and Verification

Once the optimizer returns a set of coordinates, the system must determine if it has "solved" the problem.

### 12. Numerical Precision Requirements

Competition geometry is an exact science. A length of $1.41421356$ is not "about 1.4"; it is $\sqrt{2}$.

- **IEEE 754 Double Precision (64-bit):** Provides ~15-17 significant decimal digits. This is the baseline. It is sufficient for intermediate optimization but often insufficient for distinguishing complex irrational numbers (e.g., is $x = \pi + e$ or just close?).
    
- **Extended Precision:** Libraries like `mpmath` (Python) or GMP allow arbitrary precision (e.g., 100 digits).
    
- **Strategy:** Run the optimizer in 64-bit float for speed. Once converged, switch to 100-bit precision and perform a few Newton-Raphson steps to refine the solution. This "polishing" step ensures that the error is dominated by the representation, not the optimization.
    
- **Error Accumulation:** In sequential constructions, error accumulates at each step. In global optimization (solving all constraints simultaneously), errors are distributed. However, "needle-like" triangles or near-parallel lines can lead to **catastrophic cancellation**, requiring higher precision to resolve.
    

### 13. Verification Criteria

- **Tolerance Thresholds:**
    
    - $10^{-3}$: Failed/Coarse approximation.
        
    - $10^{-9}$: Likely correct for simple problems.
        
    - $10^{-15}$ (Machine Epsilon): Valid solution for 64-bit float.
        
    - $10^{-30}$: Required for high-confidence symbolic recovery.
        
- **Degeneracy Detection:** A solution where all points collapse to $(0,0)$ satisfies `dist(A,B)=0` constraints but is trivial. The verifier must check that non-zero constraints (e.g., triangle area > 0) are maintained.
    
- **Multiple Solutions:** A distance constraint $x^2 = 4$ has solutions $x=2$ and $x=-2$. The geometric problem might imply one (e.g., "point C is on the segment AB"). The verification layer must check inequalities (chirality/orientation) to filter valid branches.
    

### 14. Answer Extraction (Symbolic Recovery)

This is the bridge from the numerical world back to the symbolic world of mathematics.

#### 14.1 Computation

- **Length:** Euclidean distance.
    
- **Area:** Shoelace formula $ \frac{1}{2} | \sum (x_i y_{i+1} - x_{i+1} y_i) | $.
    
- **Angle:** `atan2` for value, or dot product for cosine.
    

#### 14.2 Rounding and Integer Relations

We observe a float value $v = 1.41421356...$ and want to find its algebraic form.

- **Simple Rounding:** If $|v - \text{round}(v)| < \epsilon$, return the integer.
    
- **Continued Fractions:** Excellent for recovering rational numbers (fractions) from decimals.
    
- **PSLQ Algorithm:** The gold standard for integer relation detection. Given a vector of constants $\mathbf{x} = (x_1, \dots, x_n)$, PSLQ finds integer coefficients $\mathbf{a}$ such that $\sum a_i x_i = 0$.
    
    - _Usage:_ To check if $v$ is a root of a quadratic, we run PSLQ on $(1, v, v^2)$. If it returns coefficients $(2, 0, -1)$, then $-v^2 + 2 = 0 \implies v = \sqrt{2}$.
        
- **LLL Algorithm (Lattice Reduction):** Similar to PSLQ, used to find short vectors in integer lattices, effectively finding polynomial relations for algebraic numbers.
    

### 15. Confidence Scoring

Confidence is a function of:

1. **Residual:** The final value of the objective function (lower is better).
    
2. **Condition Number:** A measure of how sensitive the solution is to perturbations. High condition number implies the geometry is "wobbly" or ill-defined.
    
3. **Recovery Quality:** If PSLQ finds a relation with small integers (e.g., $1, -2$), confidence is high. If it requires integers like $192837$, it is likely fitting noise (overfitting).
    

## Part E: Special Geometric Techniques

### 16. Circle Geometry

Circles introduce quadratic constraints that can create local minima.

- **Power of a Point:** $P(A) = d^2 - r^2$. This concept linearizes many circle constraints. The locus of points with equal power to two circles is a straight line (the **radical axis**). Using radical axes in the loss function instead of direct distance constraints can simplify the optimization landscape from degree-4 surfaces to linear/quadratic ones.
    
- **Inversion:** A geometric transformation that maps circles to lines. In a numerical solver, inversion is just a coordinate transformation $f(x,y)$. The solver can attempt to solve the problem in the "inverted world" (where constraints might be simpler, e.g., parallel lines instead of tangent circles) and then map the solution back. This is a form of **preconditioning** the optimization problem.
    

## 3. Case Studies and Architecture

### 3.1 AlphaGeometry's Approach vs. Numerical Approach

Google's AlphaGeometry uses a neuro-symbolic approach where the LLM proposes auxiliary constructions (points/lines) and a symbolic engine (DD+AR) deduces the proof. It does _not_ use numerical coordinates for the proof itself.

- **Contrast:** The proposed Coordinate Descent module is a _numerical_ engine. It does not deduce; it _finds_. It complements AlphaGeometry by solving metric problems (finding "the answer is 5") which pure deduction engines struggle to simplify.
    

### 3.2 Proposed Architecture: "NumGeo"

1. **Input:** "Let ABC be a triangle..."
    
2. **Translation (LLM):** Generates JSON constraint graph.
    
3. **Initialization:**
    
    - LLM guesses rough coordinates.
        
    - Graph decomposition identifies rigid clusters.
        
4. **Optimization Loop:**
    
    - **Phase 1 (Global):** Basin-Hopping + Adam optimizer (float64) to find the approximate shape.
        
    - **Phase 2 (Local):** Block Coordinate Descent (float64) to refine.
        
    - **Phase 3 (Precision):** Newton-Raphson (mpmath 100-bit) to polish.
        
5. **Verification:** Check residuals $< 10^{-30}$.
    
6. **Extraction:** Use PSLQ to identify the symbolic answer (e.g., $\sqrt{5}$).
    
7. **Output:** Symbolic answer + Confidence score.
    

## 4. Conclusion

The "Coordinate Descent Module" represents a paradigm shift from logical derivation to computational verification. By treating geometry as a rigid body simulation, we leverage the raw power of modern floating-point units to bypass the combinatorial complexity of algebraic proofs. The combination of **LLM-based translation**, **Block Coordinate Descent**, and **PSLQ Symbolic Recovery** creates a potent solver capable of tackling the metric-heavy segment of competition geometry that eludes purely symbolic systems. While it lacks the absolute certainty of an axiomatic proof, its pragmatic ability to _find_ and _verify_ answers to 100 decimal places offers a functional equivalence that is sufficient for the vast majority of mathematical inquiry and competition applications. Future work lies in the seamless integration of this numerical intuition with symbolic engines, creating a system that can not only calculate the answer but also understand _why_ it is true.

---

**Citations:**