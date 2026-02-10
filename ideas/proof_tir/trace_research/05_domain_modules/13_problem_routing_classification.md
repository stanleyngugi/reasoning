# Deep Research: Problem Routing — Classification and Module Selection

## Research Objective

Not all problems should go through the same pipeline. Combinatorics goes to Trace-to-Lean, geometry goes to Coordinate Descent, algebra goes to Numerical Sniper. We need an intelligent router that classifies problems and directs them to the right module.

## Context

Our three main modules:
1. **Trace-to-Lean**: Combinatorics, number theory, sequence problems → Berlekamp-Massey → Lean verification
2. **Coordinate Descent**: Geometry → Constraint optimization → Numerical verification
3. **Numerical Sniper**: Algebra → High-precision numerical → PSLQ → Algebraic verification

The router must quickly and accurately classify incoming problems to maximize success rate.

## Research Questions

### Part A: Classification Fundamentals

#### 1. Problem Type Signals
What textual features indicate each problem type?
- **Combinatorics**: "how many ways", "count", "arrangements", "permutations", "combinations", "binary strings"
- **Number Theory**: "mod", "divisible", "prime", "gcd", "sequence", "find the nth term"
- **Geometry**: "triangle", "circle", "point", "line", "angle", "perpendicular", "bisector", "inscribed"
- **Algebra**: "solve", "find x", "polynomial", "roots", "equation", "inequality", "minimum", "maximum"

#### 2. Keyword-Based Classification
Simple approach:
- Extract keywords from problem statement
- Match against category dictionaries
- Score each category
- Route to highest-scoring module

Limitations? When does this fail?

#### 3. LLM-Based Classification
Better approach:
- Prompt LLM to classify the problem
- Few-shot examples for each category
- Ask for confidence score
- What prompt works best?

### Part B: Fine-Grained Subcategories

#### 4. Combinatorics Subcategories
- Counting (closed-form formula exists)
- Existence proofs (prove there exists...)
- Extremal (find maximum/minimum)
- Graph theory
- Game theory

Which are amenable to Trace-to-Lean?

#### 5. Number Theory Subcategories
- Divisibility and factorization
- Modular arithmetic
- Diophantine equations
- Sequences and recurrences
- Prime number problems

Which are amenable to Trace-to-Lean?

#### 6. Geometry Subcategories
- Triangle geometry
- Circle geometry
- Coordinate geometry (explicit)
- Transformations
- Loci
- Solid geometry

Which are amenable to Coordinate Descent?

#### 7. Algebra Subcategories
- Polynomial equations
- Systems of equations
- Inequalities
- Functional equations
- Optimization

Which are amenable to Numerical Sniper?

### Part C: Hybrid and Ambiguous Problems

#### 8. Multi-Domain Problems
Some problems span categories:
- "In triangle ABC, how many lattice points..." (geometry + combinatorics)
- "Find all primes p such that p² + 2 is prime" (number theory + counting)
- "Minimize the area given algebraic constraints" (algebra + geometry)

How do we handle these?

#### 9. Ambiguous Classification
When problem type is unclear:
- Route to most likely module first
- Have fallback if first module fails
- Parallel execution of multiple modules?

#### 10. Module Confidence
Each module should report:
- "I can handle this" (high confidence)
- "Maybe" (medium confidence)
- "Not my domain" (low confidence)

Use this for routing and fallback decisions.

### Part D: Router Architecture

#### 11. Two-Stage Router
1. **Stage 1**: Quick keyword-based pre-filter (fast)
2. **Stage 2**: LLM-based classification (more accurate)

Or:
1. LLM classification
2. Module-specific feasibility check

Which is better?

#### 12. Parallel Routing
- Send problem to all modules simultaneously
- Each module returns (answer, confidence, method)
- Select highest-confidence answer
- Trade-off: resource usage vs robustness

#### 13. Sequential Routing with Fallback
- Route to primary module
- If it fails or returns low confidence, try secondary
- Continue until success or all modules exhausted

### Part E: LLM Classification Prompts

#### 14. Zero-Shot Classification
```
Classify this math problem into one of: Combinatorics, Number Theory, Geometry, Algebra.
Problem: [problem statement]
Classification:
```
How accurate is this?

#### 15. Few-Shot Classification
Provide 2-3 examples for each category.
What examples work best?

#### 16. Chain-of-Thought Classification
```
Think step by step about what type of math this problem involves.
Problem: [problem statement]
Reasoning: ...
Classification: ...
```
Does this improve accuracy?

#### 17. Sub-Classification
After main classification, determine sub-type:
```
This is a combinatorics problem. Classify further:
- Counting (find the number of ways)
- Existence (prove there exists)
- Extremal (find maximum/minimum)
- Graph theory
- Game theory
```

### Part F: Module-Specific Feasibility

#### 18. Trace-to-Lean Feasibility Check
Signals that a problem is good for Trace-to-Lean:
- Asks for a specific numerical answer
- Can be computed for small cases
- Likely has closed-form or recurrence
- Not a proof-based problem

#### 19. Coordinate Descent Feasibility Check
Signals that a problem is good for Coordinate Descent:
- Explicit geometric setup
- Asks for length, area, angle, or coordinates
- Constraints can be expressed algebraically
- Not a synthetic proof

#### 20. Numerical Sniper Feasibility Check
Signals that a problem is good for Numerical Sniper:
- Asks for exact value of an expression
- Involves roots of polynomials
- Can be set up as equations to solve
- Answer is likely an algebraic number

### Part G: Handling Proof Problems

#### 21. Proof vs Computation
Some problems ask to "prove" something:
- "Prove that n³ + 2n is divisible by 3 for all n"
- "Show that the diagonals bisect each other"

Can we convert these to computation?
- "Prove for all n" → verify for n=1..1000 (weaker but sometimes sufficient)
- "Show property P" → verify P numerically

#### 22. Problems We Can't Handle
Some problems are fundamentally non-computational:
- "Prove there are infinitely many primes"
- "Find all functions f: ℝ → ℝ such that..."
- "Construct a configuration with property X"

How do we detect and route these?

#### 23. TIR Fallback for Proof Problems
For proof-style problems:
- Try TIR (let LLM reason through the proof)
- Lower expected success rate
- Different prompting strategy

### Part H: Competition-Specific Routing

#### 24. AIMO Answer Format
AIMO answers are integers 0-99999.
This tells us:
- Not asking for a proof
- Not asking for "yes/no"
- Numerical computation is expected

Adjust routing accordingly.

#### 25. Problem Difficulty Signals
Hard problems might need:
- More sophisticated module
- More time budget
- Multiple approaches in parallel

Can we estimate difficulty from problem text?

#### 26. Time-Aware Routing
If we're running low on time:
- Route to fastest module
- Skip expensive verification
- Accept lower confidence

### Part I: Implementation

#### 27. Router Class Design
```python
class ProblemRouter:
    def classify(self, problem: str) -> Classification:
        ...
    
    def route(self, problem: str) -> Module:
        classification = self.classify(problem)
        return self.select_module(classification)
    
    def select_module(self, classification: Classification) -> Module:
        ...
```

#### 28. Classification Data Structure
```python
@dataclass
class Classification:
    primary_type: str  # "combinatorics", "geometry", etc.
    sub_type: str  # "counting", "triangle", etc.
    confidence: float  # 0.0 to 1.0
    is_proof_based: bool
    estimated_difficulty: str  # "easy", "medium", "hard"
```

#### 29. Routing Decision Logic
```python
def select_module(classification):
    if classification.primary_type == "combinatorics":
        if classification.sub_type == "counting":
            return TraceToLeanModule()
        elif classification.sub_type == "existence":
            return TIRFallback()
    elif classification.primary_type == "geometry":
        return CoordinateDescentModule()
    ...
```

### Part J: Evaluation

#### 30. Classification Accuracy Benchmark
- Labeled dataset of competition problems with ground truth category
- Measure: accuracy, precision, recall per category
- Compare keyword vs LLM approaches

#### 31. Routing Effectiveness
- Track which module was routed to
- Track whether that module succeeded
- Identify routing errors (wrong module selected)
- Optimize routing based on outcomes

#### 32. End-to-End Impact
- Compare: intelligent routing vs always-same-module
- Quantify: how much does routing improve overall score?

## Desired Output Format

1. **Category Taxonomy**: Complete problem classification hierarchy
2. **Signal Dictionaries**: Keywords/patterns for each category
3. **LLM Prompts**: Optimized classification prompts
4. **Feasibility Checkers**: Criteria for each module
5. **Router Algorithm**: Complete decision logic
6. **Fallback Strategy**: What to do when primary fails
7. **Evaluation Framework**: How to measure routing quality
8. **Implementation Code**: Python router implementation

























# Comprehensive Research Plan for a Neuro-Symbolic Problem Routing System in Mathematical Olympiads

## Executive Summary

The domain of automated mathematical reasoning has recently transitioned from a purely symbolic pursuit to a hybrid discipline, increasingly dominated by neuro-symbolic architectures that fuse the generative capabilities of Large Language Models (LLMs) with the rigorous exactitude of classical algorithms. This research report articulates a detailed and exhaustive plan for the development of a "Problem Routing" system, a meta-cognitive framework designed to classify high-level mathematical competition problems and direct them to specialized solver modules: "Trace-to-Lean" for Combinatorics and Number Theory, "Coordinate Descent" for Geometry, and "Numerical Sniper" for Algebra.

Current state-of-the-art systems, such as Google DeepMind's AlphaGeometry and the AIMO-winning NuminaMath, have demonstrated that the path to Artificial General Intelligence (AGI) in mathematics lies not in monolithic models, but in compound systems that effectively orchestrate tool use. This report proposes a novel architecture that acts as an intelligent switchboard, leveraging semantic understanding to dispatch problems to solvers grounded in specific mathematical theories—namely, the Berlekamp-Massey algorithm for linear recurrences, non-convex optimization for geometric constraints, and the PSLQ integer relation algorithm for algebraic constant recognition.

The analysis herein spans the entire lifecycle of the routing process: from the fundamentals of classification using hybrid keyword-semantic approaches to the fine-grained taxonomy of mathematical sub-domains. It rigorously examines the handling of hybrid and ambiguous problems through ensemble routing and iterative refinement strategies. The report further details the optimal prompt engineering techniques required to elicit accurate classification signals from LLMs and establishes a set of mathematical feasibility checks to prevent algorithm misuse. Finally, it provides a concrete blueprint for Pythonic implementation and a robust benchmarking framework based on datasets like MATH, MiniF2F, and the AIMO Progress Prize corpus.

---

## 1. Introduction

### 1.1 The Neuro-Symbolic Inflection Point in Mathematics

The pursuit of automating mathematical reasoning has long been a "grand challenge" in artificial intelligence. For decades, the field was bifurcated into two distinct camps: symbolic AI, which relied on formal logic and theorem provers (e.g., Coq, Isabelle, Lean) but suffered from brittleness and a lack of intuition; and connectionist AI, or neural networks, which excelled at pattern recognition but lacked the precision required for rigorous proofs. The advent of Large Language Models (LLMs) has bridged this divide, creating a new paradigm of "Neuro-Symbolic" AI where neural networks provide the intuition (the "conjecture") and symbolic algorithms provide the verification (the "proof").

Recent breakthroughs underscore the efficacy of this hybrid approach. DeepMind's AlphaGeometry, for instance, utilizes a neural language model to suggest auxiliary constructions in geometry problems, which are then verified by a symbolic deduction engine. Similarly, the "NuminaMath" system, which secured victory in the first AI Mathematical Olympiad (AIMO) Progress Prize, leveraged "Tool-Integrated Reasoning" (TIR), where an LLM generates Python code to perform intermediate calculations, effectively treating the Python interpreter as a cognitive extension.

However, a critical inefficiency remains in current systems: the "one-size-fits-all" approach to tool selection. Most systems default to a generic Python code interpreter for all problem types. This is suboptimal because different mathematical domains have distinct structural properties that map to specific, highly efficient algorithms. A combinatorics problem involving a sequence is best solved by recurrence analysis, not generic brute force. A geometry problem is fundamentally a constraint satisfaction problem, not a text generation task. The "Problem Routing" system proposed in this report addresses this gap by implementing a sophisticated classification layer that routes problems to the solver most mathematically aligned with their structure.

### 1.2 The Three Pillars of the Solver Ecosystem

The core hypothesis of this research is that routing competition problems to specialized algorithmic modules will significantly outperform generic approaches. The three proposed modules are:

1. **Trace-to-Lean (Combinatorics/Number Theory):** This module targets problems involving integer sequences, tiling patterns, and modular arithmetic. It leverages the **Berlekamp-Massey algorithm**, a powerful tool from coding theory capable of finding the shortest linear recurrence relation for a given sequence. The "Trace-to-Lean" moniker reflects its dual output: it not only provides the numerical answer (e.g., the $n$-th term) but also generates a formal "trace" or certificate—the recurrence relation itself—which can be used to construct a formal proof in the Lean theorem prover.
    
2. **Coordinate Descent (Geometry):** This module treats geometry problems as non-convex optimization tasks. By assigning coordinates to geometric entities (points, lines, circles) and defining an "energy function" based on the problem's constraints (e.g., distances, angles, tangencies), the system uses **Coordinate Descent** or trust-region methods to relax the system into a valid configuration. This numerical approach serves as a robust heuristic for finding solution values (lengths, areas) and verifying conjectures.
    
3. **Numerical Sniper (Algebra):** This module focuses on problems involving algebraic constants, roots of polynomials, and infinite series. It utilizes the **PSLQ (Partial Sum of Least Squares) algorithm**, an integer relation detection method that is significantly more efficient and numerically stable than its predecessor, the LLL algorithm. PSLQ allows the system to recover exact symbolic forms (e.g., $\sqrt{2} + \pi$) from high-precision floating-point approximations, effectively acting as a "Reverse Symbolic Calculator".
    

### 1.3 Research Objectives and Scope

This report aims to provide a comprehensive roadmap for building this routing system. It will address the following key research questions:

- **Classification:** How can we reliably distinguish between subtle variations in mathematical problem types using NLP and LLMs?
    
- **Taxonomy:** What are the precise sub-domains within Algebra, Geometry, and Combinatorics that map to our specific solvers?
    
- **Architecture:** How should the routing mechanism be architected—sequentially, in parallel, or hierarchically—to maximize accuracy and throughput?
    
- **Feasibility:** What are the mathematical boundaries of the underlying algorithms (e.g., precision limits of PSLQ, sequence length requirements for Berlekamp-Massey), and how can the router enforce them?
    
- **Proof Generation:** How can numerical results be translated into rigorous proofs, bridging the gap between "Computation" and "Formalization"?
    

The ultimate goal is to define a system capable of achieving gold-medal performance on benchmarks like the AIMO and the MATH dataset by leveraging the specialized power of domain-specific algorithms.

---

## 2. Classification Fundamentals: Keyword vs. LLM Approaches (A)

The efficacy of the routing system is entirely dependent on the accuracy of the initial classification step. If a geometry problem is misclassified as a number theory problem, the Coordinate Descent solver will never be invoked, and the system will fail. This section analyzes the two primary paradigms for text classification in this domain: Lexical (Keyword-Based) Analysis and Semantic (LLM-Based) Analysis.

### 2.1 Keyword-Based Classification (Lexical Analysis)

Keyword extraction represents the traditional, deterministic approach to text classification. It relies on identifying specific, domain-unique vocabulary within the problem statement.

#### 2.1.1 Mechanism and Implementation

Techniques such as **TF-IDF** (Term Frequency-Inverse Document Frequency) or **RAKE** (Rapid Automatic Keyword Extraction) are standard for identifying significant terms. In the context of math competitions, this involves curating high-precision dictionaries:

- **Geometry Signals:** "barycentric", "circumcircle", "collinear", "homothety", "cyclic quadrilateral", "tangent", "locus".
    
- **Combinatorics Signals:** "permutations", "combinations", "ways", "arrangement", "pigeonhole", "tiling", "grid", "paths".
    
- **Number Theory Signals:** "modulo", "divisible", "prime", "gcd", "remainder", "congruence", "Diophantine".
    
- **Algebra Signals:** "polynomial", "roots", "coefficient", "inequality", "logarithm", "real number", "complex number".
    

#### 2.1.2 Strengths and Limitations

The primary strength of keyword-based systems is their low latency and high precision _when unique terms are present_. A problem containing "barycentric coordinates" is almost certainly a geometry problem. However, this approach suffers significantly from **polysemy** (words with multiple meanings) and **context blindness**.

- **Ambiguity:** The term "normal" is highly ambiguous. In Geometry, it refers to a perpendicular vector. In Probability, it refers to the Gaussian distribution. In Group Theory, it refers to a normal subgroup. A keyword classifier cannot distinguish these without analyzing the surrounding syntax.
    
- **Implicit Domains:** Many competition problems are "disguised." A problem might ask to "count the number of ways to tile a $2 \times n$ board" without using the word "combinatorics" or "recurrence". A lexical classifier might miss this completely if "tiling" isn't in its dictionary, or if the problem uses a synonym like "cover."
    
- **False Positives:** A geometry problem might use an algebraic equation to define a curve (e.g., $x^2 + y^2 = 1$). A naive keyword classifier might trigger on "equation" and label it as Algebra, missing the geometric nature that makes Coordinate Descent the better tool.
    

### 2.2 LLM Semantic Classification (Contextual Analysis)

Large Language Models (LLMs) fundamentally change the classification landscape by processing the _semantic meaning_ and _mathematical intent_ of the problem, rather than just its surface-level vocabulary.

#### 2.2.1 Mechanism: Embeddings and Few-Shot Prompting

- **Vector Embeddings:** The problem text is converted into a high-dimensional vector using models like `text-embedding-3-small` or `BERT`. These vectors are then compared (via cosine similarity) to the centroids of known problem clusters (e.g., a "Geometry Cluster" defined by thousands of MATH dataset problems).
    
- **Zero-Shot/Few-Shot Inference:** The LLM is prompted directly to categorize the problem. "Classify the following math problem into one of these categories:. Explain your reasoning." This allows the model to leverage its pre-trained knowledge of mathematical concepts.
    

#### 2.2.2 Advantages in Mathematical Contexts

- **Deep Inference:** LLMs can infer the mathematical structure implied by the text. They can recognize that a problem asking for "integer solutions to $x^2 - Dy^2 = 1$" is a Pell's Equation problem (Number Theory) even if the term "Diophantine" is absent. They understand that "maximizing an area" implies optimization.
    
- **Multimodal Signal Processing:** Competition problems often include formatting signals. LLMs trained on code (like StarCoder or Codex) can parse LaTeX macros or **Asymptote (`[asy]`)** code blocks. The presence of `[asy]` is a nearly definitive signal for Geometry, which an LLM can weigh heavily in its decision.
    
- **Handling Word Problems:** Semantic classifiers excel at stripping away the "flavor text" of word problems (e.g., "Alice and Bob are trading apples...") to reveal the underlying algebraic system.
    

### 2.3 The Hybrid Classification Strategy

Research strongly suggests that a **hybrid hierarchical approach** yields the optimal balance of speed and accuracy. The proposed router should implement a multi-stage filter:

1. **Stage 1: The "Fast Path" (Deterministic/Regex):**
    
    - This layer executes identifying checks for "hard" signals.
        
    - **Rule:** If `[asy]` tags or specific geometry commands (`draw`, `pair`, `cycle`) are detected, the problem is immediately flagged as Geometry and sent to the Coordinate Descent module.
        
    - **Rule:** If the problem explicitly asks for a "recurrence relation" or "generating function," it is flagged for Trace-to-Lean.
        
2. **Stage 2: The "Semantic Path" (Embedding-Based):**
    
    - For problems that clear the fast path, the system generates embeddings and classifies them using a lightweight classifier (e.g., a logistic regression head on top of embeddings) or a Semantic Router pattern.
        
    - This handles the vast majority of standard problems where vocabulary is varied but intent is clear.
        
3. **Stage 3: The "Reasoning Path" (LLM-Based):**
    
    - For low-confidence classifications, a more powerful Reasoning LLM (e.g., GPT-4) is invoked with a Chain-of-Thought (CoT) prompt to analyze the problem's feasibility for each solver (discussed in Section F).
        

**Table 1: Comparative Analysis of Classification Approaches**

|**Feature**|**Keyword-Based (Lexical)**|**LLM Semantic (Embedding/Prompt)**|**Hybrid (Recommended)**|
|---|---|---|---|
|**Accuracy**|Moderate (High for technical jargon)|High (Context-aware)|Very High (Best of both)|
|**Latency**|Extremely Low ($<10$ms)|Moderate-High ($100$ms - $1$s)|Variable (Optimized)|
|**Context Handling**|Poor (Fails on polysemy)|Excellent|Excellent|
|**Resource Cost**|Negligible|Significant (GPU/API costs)|Balanced|
|**Failure Mode**|"Normal" $\to$ Geometry (False Positive)|Hallucinating a relationship|Mitigated by layered checks|

---

## 3. Fine-Grained Subcategories and Domain Taxonomy (B)

A high-level classification into "Algebra" or "Geometry" is insufficient for a specialized routing system. The underlying solvers—Berlekamp-Massey, Coordinate Descent, PSLQ—have specific mathematical requirements. We must define a taxonomy of **fine-grained subcategories** that map directly to the _capabilities_ and _input requirements_ of these algorithms.

### 3.1 Domain: Combinatorics & Number Theory

**Primary Solver:** Trace-to-Lean (Berlekamp-Massey)

**Secondary Solver:** Brute-Force Enumeration (Python)

The Berlekamp-Massey (BM) algorithm is designed to find the shortest linear feedback shift register (LFSR) that generates a given sequence. Mathematically, it finds the minimal polynomial of a linearly recurrent sequence. Therefore, the taxonomy must identify problems that can be reduced to **integer sequences**.

- **Linear Recurrence Relations:** Problems that explicitly define a sequence (e.g., Fibonacci, Lucas) or imply one through a recursive process ($a_n = c_1 a_{n-1} + \dots$). BM can recover the characteristic polynomial exactly from a finite set of terms.
    
- **Tiling and Covering:** Problems asking for "the number of ways to tile an $m \times n$ board." These problems almost always result in sequences that satisfy linear recurrences with constant coefficients. The router must identify "tiling," "domino," and "ways" as triggers.
    
- **Walks on Graphs:** Counting the number of paths of length $n$ between two vertices in a finite graph. The number of walks is given by the powers of the adjacency matrix, which satisfies the Cayley-Hamilton theorem and thus a linear recurrence.
    
- **Modular Sequences:** Number theory problems involving powers modulo $p$ (e.g., $x^n \pmod p$). These sequences are periodic (and thus linear recurrent) due to Fermat's Little Theorem and the Pigeonhole Principle.
    
- **Hidden Sequences:** Word problems that describe a step-by-step process where the state at step $n$ depends linearly on previous states (e.g., population dynamics, token passing games). The router must detect the _iterative structure_.
    

**Subcategory Exclusion:** Problems involving **permutations** ($n!$) or **catalan numbers** (which are not linear recurrent in the simple sense, though $C_n$ satisfies a recurrence with polynomial coefficients) may require identifying _holonomic_ sequences rather than simple linear ones. The router should distinguish "linear growth" from "factorial growth" to avoid sending unsolvable sequences to BM.

### 3.2 Domain: Geometry

**Primary Solver:** Coordinate Descent (Optimization)

**Secondary Solver:** Synthetic Deduction (AlphaGeometry-style)

The Coordinate Descent module treats geometry as a non-convex optimization problem $E(x) = 0$, where $E$ represents the deviation from constraints. This requires the problem to be expressible in terms of **Metric Geometry**.

- **Metric Geometry:** Problems involving explicit lengths, areas, perimeters, and trigonometric ratios. These are the ideal candidates for coordinate descent, as the loss function (e.g., $|AB - 5|^2$) is differentiable.
    
- **Optimization Geometry:** Problems asking to "minimize the length," "maximize the area," or find an "extremal configuration." Optimization algorithms like coordinate descent are natively designed for this, often outperforming human intuition.
    
- **Locus Problems:** Finding the set of points satisfying a condition. The solver can sample multiple points that satisfy the condition (energy $\approx 0$) and fit a curve (circle, line, ellipse) to the resulting coordinates.
    
- **Constructive Geometry:** Problems defined by a rigorous sequence of constructions (e.g., "Let $D$ be the intersection of the angle bisector of $A$ and..."). These can be modeled as a directed acyclic graph (DAG) of constraints, suitable for numerical solving.
    

**Subcategory Exclusion:** **Topology** problems (involving loops, holes, or connectivity without metric data) and **Projective Geometry** problems involving only incidence relations (without a metric) may be less suitable for standard coordinate descent unless augmented with specific projective coordinate systems (barycentric coordinates).

### 3.3 Domain: Algebra

**Primary Solver:** Numerical Sniper (PSLQ)

**Secondary Solver:** CAS (SymPy)

The PSLQ algorithm finds integer relations $a_1 x_1 + \dots + a_n x_n = 0$. It is effectively an "Inverse Symbolic Calculator" that can identify algebraic numbers from their decimal expansions.

- **Algebraic Identification:** Problems asking for "the value of $x$." If the answer is suspected to be a form like $\sqrt{2} + \sqrt{3}$, the solver calculates $x$ numerically to high precision and uses PSLQ to find the minimal polynomial it satisfies.
    
- **Transcendental Constants:** Problems involving $\pi, e, \gamma$. PSLQ can verify if an expression equals a linear combination of these constants (e.g., the BBP formula for $\pi$).
    
- **Inequalities:** Problems asking to "Prove $A \ge B$." The solver can numerically minimize the difference $A - B$. If the minimum is $0$ (within precision), it provides strong evidence and potentially the "extremal case" (values of variables) where equality holds, guiding the formal proof.
    
- **Summations and Products:** Evaluating infinite series or products. The system computes the sum numerically (e.g., to 100 digits) and checks for relations with known constants like $\pi^2, \zeta(3)$.
    

**Subcategory Exclusion:** **Abstract Algebra** (Group Theory, Ring Theory) deals with structures and axioms, not numerical values. These are "unsolvable" by PSLQ and must be routed to a theorem prover or a specialized algebraic reasoner.

---

## 4. Strategies for Hybrid and Ambiguous Problems (C)

A significant proportion of Olympiad-level problems are intentionally designed to be hybrid, blending concepts from multiple domains. A rigid routing system that forces a "1-of-N" classification will inevitably fail on these edge cases. We propose specific strategies to handle ambiguity: **Ensemble Routing**, **Iterative Decomposition**, and **Cascading**.

### 4.1 Taxonomy of Hybrid Problems

1. **Geometric Probability:** These problems combine geometry (calculating areas or volumes of feasible regions) with probability logic.
    
    - _Strategy:_ If the geometry is simple (e.g., regions in a unit square), route to **Coordinate Descent** to perform Monte Carlo integration or numerical area estimation. If the problem involves discrete structures, route to **Trace-to-Lean**.
        
2. **Lattice Points (Geometry of Numbers):** The intersection of Geometry and Number Theory (e.g., Pick's Theorem, counting points in a circle).
    
    - _Strategy:_ These are fundamentally **counting problems**. Route to **Trace-to-Lean** (Berlekamp-Massey). The system generates the count for radius $r=1, 2, 3 \dots$ and uses BM to detect the pattern of growth or recurrence.
        
3. **Complex Numbers in Geometry:** Geometry problems that are best solved using complex number algebra (rotation as multiplication).
    
    - _Strategy:_ This is ambiguous. If the goal is a numerical value (length), **Numerical Sniper** (PSLQ) is effective. If the goal is a configuration, **Coordinate Descent** is applicable by treating the complex plane as $\mathbb{R}^2$.
        

### 4.2 Handling Strategies

#### 4.2.1 Ensemble Routing (The "Shotgun" Approach)

For high-ambiguity problems where the confidence score of the classifier is split (e.g., Geometry: 0.45, Combinatorics: 0.45), the system should trigger _multiple_ solvers in parallel.

- **Mechanism:** Instantiate both the Geometry Solver and the Sequence Solver.
    
    - _Solver A (Geometry)_ attempts to model the problem constraints visually.
        
    - _Solver B (Sequence)_ attempts to generate the first few terms of the sequence (if applicable) using a brute-force Python script.
        
- **Resolution:** The router monitors the outputs. If Solver B successfully finds a linear recurrence with low discrepancy, its result is accepted. If Solver A minimizes the energy to zero, its configuration is accepted. This leverages the "Mixture of Experts" philosophy to maximize success probability.
    

#### 4.2.2 Neuro-Symbolic Cascading

This advanced strategy involves using the output of one solver as the input for another, creating a compound AI system.

- **Scenario:** A problem asks for the area of a sequence of shapes defined by a recursive geometric construction.
    
- **Cascade:**
    
    1. **Route to Coordinate Descent:** Calculate the area for the first 5 iterations ($A_1, A_2, A_3, A_4, A_5$) numerically.
        
    2. **Route to Trace-to-Lean:** Feed these numerical values (floating point) to the Sequence Solver.
        
    3. **Conversion:** Convert floats to nearest integers or rationals (using continued fractions).
        
    4. **Solve:** Apply Berlekamp-Massey to find the recurrence relation for the area sequence.
        
- This approach solves problems that neither solver could handle in isolation.
    

#### 4.2.3 Iterative Refinement

If the initial classification leads to solver failure (e.g., BM returns no recurrence), the system performs a "Refection Step."

1. **Error Analysis:** The LLM analyzes the solver's error log (e.g., "Sequence grew too fast," "Optimization stuck in local minima").
    
2. **Re-Routing:** The router uses this feedback to re-classify. "The sequence is not linear; perhaps it is quadratic or exponential. Let's try curve fitting or Python symbolic algebra".
    

---

## 5. Router Architecture (D)

The architectural design of the routing system determines its throughput, latency, and fault tolerance. We evaluate three primary architectures and propose a **Parallel Semantic Router** as the optimal solution.

### 5.1 Architecture Options

#### Option A: Sequential Router (Cascade)

The problem passes through a linear series of filters (Gateways).

1. **Gate 1:** Is it Geometry? (Check keywords/`[asy]`). If Yes $\to$ Coordinate Descent. Stop.
    
2. **Gate 2:** If No $\to$ Is it a Sequence? (LLM check). If Yes $\to$ Trace-to-Lean. Stop.
    
3. **Gate 3:** If No $\to$ Numerical Sniper.
    

- **Pros:** Simple to implement; low resource usage for "obvious" problems.
    
- **Cons:** "Pipeline blocking." If a geometry problem is missed by Gate 1, it is forced into the wrong solvers downstream. Errors accumulate.
    

#### Option B: Parallel Router (Mixture of Experts - MoE)

The problem is sent to a central "Gating Network" (Router) that assigns a probability distribution over the experts.

- **Mechanism:** A Semantic Router (embedding-based) assigns weights $w_i$. If $P(\text{Geom}) = 0.8$ and $P(\text{Alg}) = 0.2$, the system primarily queries Geometry, but might query Algebra if resources permit.
    
- **Pros:** High throughput; handles ambiguity naturally; allows for "Soft Routing" (sending to top-$k$ experts).
    
- **Cons:** Higher computational cost (running multiple solvers/LLM calls).
    

#### Option C: Two-Stage Hierarchical Router (Recommended)

This architecture balances efficiency and depth, minimizing the cost of LLM calls while maximizing accuracy.

- **Stage 1 (Fast Router):** A high-speed, rule-based filter combined with a lightweight embedding model (e.g., `distilbert` or a specialized encoder).
    
    - _Task:_ Filter out "junk" inputs and handle obvious cases (e.g., explicit `[asy]` tags immediately route to Geometry). This operates in $< 50$ms.
        
    - _Outcome:_ Hard routing for clear cases; pass-through for ambiguous ones.
        
- **Stage 2 (Deep Reasoning Router):** A Reasoning LLM (e.g., GPT-4o, Claude 3.5 Sonnet, or a fine-tuned LLaMA) performs "Thought-Integrated Routing."
    
    - _Task:_ It analyzes the _feasibility_ of applying a specific algorithm. It asks: "Is this sequence long enough for Berlekamp-Massey?" "Are the constraints sufficient for Coordinate Descent?".
        
    - _Outcome:_ Assigns the problem to the appropriate solver module(s) with specific instructions.
        
- **Execution Layer:** The selected solver(s) are invoked.
    
- **Synthesis Layer:** An LLM aggregates the outputs (numerical values, traces, code) and formulates the final answer/proof.
    

**Recommendation:** The **Two-Stage Hierarchical Router** is the most robust choice. It prevents the "bottleneck" of sequential systems while avoiding the excessive compute cost of a full parallel MoE for every trivial problem.

---

## 6. Optimal LLM Classification Prompts (E)

Prompt engineering is the interface between the raw problem text and the routing logic. To support the Semantic Router, we need high-quality prompts that leverage **Chain-of-Thought (CoT)** reasoning to extract feasibility signals.

### 6.1 The "Algorithm-Fit" Prompt Strategy

Instead of asking "What math domain is this?", the prompt should ask "Which computational tool fits this problem's structure?". This aligns the classification with the _capabilities_ of the backend.

**System Prompt Template:**

> "You are an expert Mathematical Router for a neuro-symbolic solver system. Your goal is not to solve the problem, but to determine which COMPUTATIONAL STRATEGY is most viable.
> 
> **Available Strategies:**
> 
> 1. **Sequence Analysis (Trace-to-Lean):** For problems that generate a sequence of integers/rationals (e.g., tiling, recursive steps, counting walks). Requires: Ability to generate at least 6-10 terms.
>     
> 2. **Geometric Optimization (Coordinate Descent):** For problems defining a geometric configuration with metric constraints (lengths, angles, areas). Requires: Explicit constraints amenable to energy minimization.
>     
> 3. **Algebraic Relation Detection (Numerical Sniper):** For problems requiring the identification of a specific constant value (root, sum, product). Requires: Answer is a single numerical constant.
>     
> 4. **General Reasoning:** For logic puzzles or proofs not amenable to the above.
>     
> 
> **Task:** Analyze the input problem.
> 
> 1. Identify the core mathematical object (Sequence, Shape, Constant).
>     
> 2. Evaluate if the necessary data (terms, constraints) can be generated.
>     
> 3. Output the selected strategy and a confidence score (0-1)."
>     

### 6.2 Few-Shot Examples (Contextual Learning)

Providing examples is crucial for calibration.

- **Example 1 (Combinatorics $\to$ Sequence Analysis):**
    
    - _Input:_ "How many ways can a $2 \times 10$ rectangle be tiled with $1 \times 2$ dominoes?"
        
    - _Reasoning:_ The problem asks for a count dependent on dimensions. Let $n=10$. This implies a sequence $a_n$ for $n=1, 2, \dots$. We can brute-force small cases ($n=1, 2, 3$) and use linear recurrence finding (Berlekamp-Massey).
        
    - _Output:_ `{"strategy": "Sequence Analysis", "confidence": 0.98}`
        
- **Example 2 (Geometry $\to$ Geometric Optimization):**
    
    - _Input:_ "In triangle $ABC$, $AB=13, BC=14, CA=15$. Point $P$ minimizes $PA^2+PB^2+PC^2$."
        
    - _Reasoning:_ Explicit metric constraints ($AB, BC, CA$). The objective is a minimization function of coordinates. This is a classic convex optimization task.
        
    - _Output:_ `{"strategy": "Geometric Optimization", "confidence": 0.99}`
        
- **Example 3 (Algebra $\to$ Numerical Sniper):**
    
    - _Input:_ "Find the value of $\sqrt{2 + \sqrt{5}} + \sqrt{2 - \sqrt{5}}$."
        
    - _Reasoning:_ The expression defines a specific constant. We can compute this to high precision (100+ digits) and use PSLQ to find the integer polynomial it satisfies, identifying the integer or algebraic value.
        
    - _Output:_ `{"strategy": "Algebraic Relation Detection", "confidence": 0.95}`
        

### 6.3 Chain-of-Thought vs. Few-Shot

Research indicates that **Chain-of-Thought (CoT)** performs better than simple Few-Shot for complex reasoning tasks. The router should be prompted to "Think step-by-step" about the _feasibility_ of the algorithm before outputting the classification. This reduces the rate of assigning "impossible" tasks (e.g., applying BM to a sequence that is too short).

---

## 7. Feasibility Checks and Algorithm Limitations (F)

A robust routing system must not only classify problems but also protect its solvers from inputs that violate their mathematical assumptions. The router must enforce **Feasibility Checks** before or during dispatch.

### 7.1 Trace-to-Lean (Berlekamp-Massey Constraints)

The Berlekamp-Massey algorithm finds the shortest LFSR of length $L$ for a sequence.

- **The $2L$ Constraint:** To uniquely determine a recurrence of order $L$, the algorithm mathematically requires a sequence of length **$2L$**. If the true recurrence has order 4, we need at least 8 terms.
    
- **Feasibility Check:**
    
    - **Generation Test:** Can the LLM (via code interpreter) efficiently generate at least 10-20 terms? If the code times out after 3 terms, BM is infeasible.
        
    - **Linearity Check:** Is the sequence likely linear? Factorials ($n!$) and powers ($n^n$) are _not_ linear recurrent. Their "linear complexity" grows with $n$. The router should detect "super-linear" growth patterns and abort BM to save resources.
        

### 7.2 Numerical Sniper (PSLQ Constraints)

The PSLQ algorithm detects integer relations $a \cdot x = 0$.

- **Precision Constraint:** To detect a relation of dimension $n$ with coefficients up to size $10^d$, one needs roughly $n \times d$ digits of precision. For checking if a number is a root of a degree-10 polynomial with large coefficients, ~200+ digits might be needed.
    
- **Feasibility Check:**
    
    - **Computability:** Can we compute the constant to 100+ digits? If the problem contains symbolic variables without values ("Let $x$ be a real number..."), PSLQ cannot run. The problem must be fully specified numerically.
        
    - **Dimension Limit:** PSLQ is typically used for dimensions $n < 100$. Attempting to find a relation among 1000 constants is computationally prohibitive.
        

### 7.3 Coordinate Descent Constraints

Coordinate descent minimizes an energy function $E(x)$.

- **Convexity & Local Minima:** The algorithm converges to local minima. It requires a "nice" energy landscape to find the global optimum (the true solution).
    
- **Feasibility Check:**
    
    - **Rigidity:** Is the geometric configuration rigid? If the problem has degrees of freedom (e.g., "triangle ABC can rotate"), the coordinates are not unique, though invariants (area) might be. The system must anchor the figure (e.g., "Set $A = (0,0), B = (c, 0)$").
        
    - **Differentiability:** Are the constraints differentiable? Coordinate descent struggles with discrete constraints (e.g., "points must have integer coordinates"). These require Lattice solvers, not continuous optimization.
        

---

## 8. Distinguishing Proof vs. Computation Problems (G)

Mathematical competitions distinguish between "finding an answer" and "proving a statement." The routing system must handle this duality, termed the **Computation-Proof Gap**.

### 8.1 The Classification Logic

- **Computation Mode:** Triggered by keywords "Find...", "Calculate...", "Evaluate...". The goal is a final value (e.g., "42", "$\sqrt{5}$"). The output of the solver (e.g., PSLQ) is the final answer.
    
- **Proof Mode:** Triggered by keywords "Prove...", "Show that...", "Demonstrate...". The goal is a logical argument.
    

### 8.2 Autoformalization: Bridging the Gap

In Proof Mode, the computational solvers act as **Truth Oracles** or **Hint Generators** for a formal proof engine (like Lean or a CoT LLM).

- **Trace-to-Lean:**
    
    - _Scenario:_ "Prove that $a_n = F_{2n}$."
        
    - _Action:_ BM finds the recurrence for $a_n$.
        
    - _Autoformalization:_ This recurrence is passed to the LLM/Lean. The LLM effectively says, "We conjecture the recurrence is $X$. Let us prove this by induction." The "Trace" (the recurrence certificate) guides the formal proof search, reducing the search space for the theorem prover.
        
- **Coordinate Descent:**
    
    - _Scenario:_ "Prove that lines $l, m, n$ are concurrent."
        
    - _Action:_ Coordinate Descent calculates coordinates and finds the intersection point distance is $0.00000001$.
        
    - _Autoformalization:_ This numerical fact ("The lines intersect at approx $(3.1, 4.2)$") validates the claim. The LLM then generates the proof sketch: "We construct the intersection of $l$ and $m$ and show it lies on $n$." The solver verifies the _truth_ before the LLM attempts the _derivation_, preventing hallucinations.
        

This **"Conjecture-and-Verify"** loop is the core of modern neuro-symbolic math, as seen in AlphaProof and AlphaGeometry.

---

## 9. Utilizing Competition-Specific Signals (AIMO) (H)

High-stakes competitions like the AI Mathematical Olympiad (AIMO), USAMO, and Putnam have specific structural signals that can be exploited for high-accuracy routing.

### 9.1 The AIMO/NuminaMath Context

The **NuminaMath** system (AIMO Progress Prize winner) popularized **Tool-Integrated Reasoning (TIR)**, interleaving Python code with natural language reasoning. Our system extends this by specializing the "tools" from generic Python to domain-specific algorithms.

### 9.2 Signal Extraction from Formats

- **Asymptote (`[asy]`):** The AIMO and Art of Problem Solving (AoPS) datasets frequently use `[asy]` blocks to render geometry diagrams.
    
    - _Signal:_ Presence of `[asy]` is a 99% confidence signal for **Geometry**. The code within the `[asy]` block (e.g., `draw((0,0)--(1,1));`) can be parsed to initialize the **Coordinate Descent** solver with starting coordinates.
        
- **LaTeX Formatting:**
    
    - `\boxed{}` usually indicates the final answer format.
        
    - `\pmod` or `\equiv` indicates **Number Theory**.
        
    - `\sum`, `\prod`, `\int` indicate **Algebra/Calculus**.
        
- **Problem Placement:** In competitions like the IMO, problem difficulty increases from P1 to P3 and P4 to P6. P1/P4 are often Algebra/Geometry; P3/P6 are often Combinatorics/Number Theory requiring deep insight. While not a hard rule, this metadata can inform the "Reasoning Router" about the likely complexity and the need for rigorous proof vs. computation.
    

---

## 10. Python Implementation Logic (I)

The system is implemented in Python, utilizing its rich ecosystem for LLM orchestration (`langchain`) and numerical computation (`scipy`, `mpmath`).

### 10.1 System Architecture (Pseudocode)

Python

```
import mpmath
from scipy.optimize import minimize
from langchain.chains import LLMChain

class ProblemRouter:
    def __init__(self, llm_classifier, solvers):
        self.classifier = llm_classifier
        self.solvers = solvers # Dictionary: {'Geometry': CD_Solver, 'Sequence': BM_Solver,...}

    def route(self, problem_text):
        # 1. Feature Extraction (Fast Path)
        signals = extract_signals(problem_text) # Checks for [asy], keywords
        if signals['has_asy']:
            print("Routing to Coordinate Descent (Asymptote detected)")
            return self.solvers['Geometry'].solve(problem_text, init_from_asy=True)
            
        # 2. Semantic Classification via LLM
        # Prompt returns JSON: {"strategy": "Sequence", "confidence": 0.9, "reasoning": "..."}
        classification = self.classifier.classify(problem_text)
        
        # 3. Feasibility & Dispatch
        if classification['strategy'] == 'Sequence':
            # 3a. Feasibility: Attempt to generate terms
            terms = self.solvers.generate_terms(problem_text, k=15)
            if len(terms) < 8: # Constraint: Need 2L terms
                print("Sequence too short for BM. Fallback to General LLM.")
                return self.solvers['GeneralLLM'].solve(problem_text)
            return self.solvers.berlekamp_massey(terms)
            
        elif classification['strategy'] == 'Algebraic_Relation':
            # 3b. Feasibility: Check for symbolic parameters
            if "Let x be" in problem_text and "Find x" not in problem_text:
                 return self.solvers['GeneralLLM'].solve(problem_text)
            return self.solvers['Algebra'].numerical_sniper(problem_text)
            
        else:
            return self.solvers['GeneralLLM'].solve(problem_text)
```

### 10.2 Solver Implementation Details

- **Trace-to-Lean:** Implements the `berlekamp_massey` function. It requires a helper function (using an LLM to write Python code) to "brute force" the first $k$ terms of the sequence described in the problem. If a recurrence is found, it outputs the next term or the closed form.
    
- **Numerical Sniper:** Uses `mpmath` for arbitrary-precision arithmetic. The PSLQ implementation (`mpmath.pslq`) takes a vector of high-precision floats and searches for integer coefficients.
    
- **Coordinate Descent:** Uses `scipy.optimize.minimize` (L-BFGS-B method). The LLM translates the geometry problem into a "loss function" (e.g., `loss = abs(dist(A,B) - 5) + abs(dist(B,C) - 5)`). The solver minimizes this loss to find coordinates.
    

---

## 11. Evaluation Benchmarks (J)

To validate the system, we must establish rigorous benchmarks comparing the Routing System against standard Chain-of-Thought (CoT) and Tool-Integrated Reasoning (TIR) baselines.

### 11.1 Datasets

- **MATH Dataset:** 12,500 problems from AMC/AIME/competitions. Contains labels for Algebra, Geometry, Counting & Probability, Number Theory. This is ideal for training and validating the classifier accuracy.
    
- **MiniF2F:** Benchmark for formal theorem proving. Useful for testing the "Trace-to-Lean" proof generation capabilities.
    
- **AIMO Progress Prize Dataset:** A curated set of high-difficulty problems (approx. 110 problems) targeting IMO level. This is the "gold standard" for testing the upper limits of the solvers.
    

### 11.2 Evaluation Metrics

1. **Routing Accuracy:** The percentage of problems routed to the "correct" solver (defined as the solver capable of solving it). This measures the classifier's performance.
    
2. **Pass@k:** The success rate with $k$ attempts. Solvers like Coordinate Descent are sensitive to random initialization. We should run them $k=32$ times and check for consensus (e.g., if 30/32 runs converge to the same value, it is likely correct).
    
3. **End-to-End Accuracy:** Comparison with vanilla GPT-4 or NuminaMath. We hypothesize that the Router will outperform on "Hard" Algebra and Geometry problems where pure LLM reasoning hallucinates, but numerical verification provides ground truth.
    

### 11.3 Experimental Hypothesis

The "Problem Routing" system is expected to show significant gains in **robustness**. While a general LLM might hallucinate a geometry proof, the Coordinate Descent module will physically verify it. While an LLM might guess the next term of a sequence, Berlekamp-Massey will mathematically prove the recurrence. The primary failure mode to mitigate is **classification error**—routing a problem to a solver that cannot handle it—which is why the **Feasibility Checks** (Section F) are the system's most critical safety rail.

---

## 12. Conclusion

The proposed "Problem Routing" system represents a paradigm shift in automated mathematical reasoning. By acknowledging that different mathematical domains require fundamentally different cognitive and computational strategies, we move away from monolithic models toward specialized, neuro-symbolic agents.

The integration of **Trace-to-Lean** (recurrence mining), **Coordinate Descent** (geometric optimization), and **Numerical Sniper** (algebraic relation discovery) addresses the specific structural properties of Combinatorics, Geometry, and Algebra respectively. The success of this system relies on the intelligent orchestration of these tools via a smart semantic router that understands not just the _content_ of a problem, but its _computational feasibility_.

Future work should focus on the "Autoformalization" loop—closing the circle by using the numerical insights from these solvers to generate rigorous, verifiable proofs in Lean, thereby solving the "Proof vs. Computation" gap that currently limits AI in high-level mathematics. This research lays the foundation for an AI mathematician that does not just "predict" answers, but "discovers" and "verifies" them with the rigor of a human expert.