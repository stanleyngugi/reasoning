# Deep Research: Trace Generation — LLM Prompting for Experimental Computation

## Research Objective

The first phase of our pipeline asks an LLM to write Python code that computes f(n) for small n. This is "trace generation" — we're not asking the LLM to solve the problem, just to experiment and produce data. We need to perfect the prompting strategy to maximize success rate and output quality.

## Context

The prompt pattern we're using:
```
"Write a Python script to EXPERIMENTALLY determine the answer for small N.
Output a list of integers. Do NOT attempt to solve for N=2025 directly."
```

This leverages what LLMs are good at (code generation) and avoids what they're bad at (complex mathematical reasoning). But we need to optimize every aspect of this.

## Research Questions

### Part A: Prompt Engineering Fundamentals

#### 1. Core Prompt Structure
- What makes a trace generation prompt effective?
- Should we provide examples (few-shot) or just instructions (zero-shot)?
- What's the optimal level of specificity vs generality?
- How do we prevent the LLM from trying to "solve" instead of "experiment"?

#### 2. Output Format Specification
- How should we specify the output format?
  - `print([f(1), f(2), ..., f(15)])`
  - JSON format
  - CSV
  - Key-value pairs with n and f(n)
- How many terms should we request? (15? 20? 30?)
- Should we ask for n=0 or start at n=1?

#### 3. Preventing Common Failure Modes
- How do we prevent the LLM from:
  - Generating code that times out (exponential loops)
  - Using libraries that might not be available
  - Making off-by-one errors
  - Producing floating-point instead of exact integers
  - Attempting the full problem instead of small cases

#### 4. Problem Type Adaptation
- Should we customize the prompt based on problem type?
  - Combinatorics: "Count the number of ways..."
  - Number theory: "Compute f(n) mod 10^9+7..."
  - Sequence problems: "Find the nth term..."
- How do we detect problem type for prompt selection?

### Part B: Code Quality Optimization

#### 5. Efficiency Requirements
- We need the trace code to run quickly (< 10 seconds for n=1..20)
- How do we prompt for efficient code?
- Should we specify time/memory constraints?
- What about memoization hints?

#### 6. Correctness Signals
- How do we increase probability of correct trace code?
- Should we ask for assertions or test cases?
- Should we request the LLM to verify its own code?
- Does asking for "step-by-step" approach help?

#### 7. Edge Case Handling
- How do we ensure the code handles:
  - n=0 (if applicable)
  - n=1 (base case)
  - Empty sets, zero counts
  - Modular arithmetic correctly

#### 8. Import and Library Usage
- What libraries should we allow/encourage?
  - `math` (factorial, comb, gcd)
  - `itertools` (combinations, permutations)
  - `functools` (lru_cache)
  - `numpy` (probably avoid due to floats)
  - `sympy` (heavy but powerful)
- How do we specify available libraries?

### Part C: Problem-Specific Strategies

#### 9. Combinatorics Problems
- "Count arrangements of..."
- "How many ways to..."
- "Find the number of valid configurations..."
- What prompt patterns work best for counting problems?
- Should we suggest specific approaches (recursion, DP, direct enumeration)?

#### 10. Number Theory Problems
- "Find f(n) mod p"
- "Compute the sum of divisors..."
- "What is the GCD of..."
- Prompt patterns for number-theoretic computations
- Handling large number concerns

#### 11. Sequence Problems
- "Find the nth term of the sequence defined by..."
- "Given a recurrence, compute..."
- Prompt patterns for sequence generation
- Ensuring correct base case handling

#### 12. Constraint Satisfaction Problems
- "How many solutions exist to..."
- "Count integers n < N such that..."
- Brute-force enumeration vs smart iteration
- When to suggest filtering vs generation

### Part D: Model-Specific Considerations

#### 13. Model Selection
- Which models are best at trace generation?
  - DeepSeek-Coder
  - Qwen-Coder
  - Claude
  - GPT-4
- Is there a significant difference between code-specialized and general models?

#### 14. Temperature and Sampling
- What temperature is optimal for trace code generation?
- Low temperature for reliability vs high temperature for creativity?
- Should we sample multiple traces and verify consistency?

#### 15. Context Window Usage
- How much of the problem context should we include?
- Do we need the full problem statement or just key constraints?
- Does providing similar examples help?

### Part E: Error Recovery and Iteration

#### 16. Execution Failure Handling
- If the generated code fails to execute, how do we recover?
- Pass the error message back to LLM?
- Retry with different prompt?
- Fallback to simpler approach?

#### 17. Partial Trace Handling
- What if the code produces only some terms?
- What if it produces wrong format?
- How do we detect and handle partial success?

#### 18. Trace Validation
- Before using the trace, how do we validate it?
- Sanity checks (non-negative, monotonic if expected, etc.)
- Cross-check with simpler cases we can verify

### Part F: Advanced Techniques

#### 19. Multi-Stage Trace Generation
- First ask for understanding of the problem
- Then ask for approach
- Then ask for code
- Does this staged approach improve quality?

#### 20. Ensemble Traces
- Generate multiple traces with different prompts/models
- Compare and vote on consensus trace
- How do we merge conflicting traces?

#### 21. Trace Augmentation
- Once we have [a₁, ..., a₁₅], can we prompt for more terms efficiently?
- Can we ask the LLM to extend the sequence using pattern it sees?
- Risk of hallucination in extension?

### Part G: Benchmark and Evaluation

#### 22. Success Metrics
- What percentage of problems yield correct traces?
- What are the main failure modes?
- How does success rate vary by problem type?

#### 23. Trace Quality Metrics
- Execution time
- Correctness of all terms
- Code readability (for debugging)
- Consistency across multiple generations

## Desired Output Format

1. **Optimal Prompt Templates**: Ready-to-use prompts for each problem type
2. **Few-Shot Examples**: Curated examples for in-context learning
3. **Failure Mode Taxonomy**: What goes wrong and how to prevent it
4. **Model Comparison**: Which models excel at trace generation
5. **Parameter Recommendations**: Temperature, token limits, sampling strategy
6. **Error Recovery Playbook**: What to do when trace generation fails
7. **Validation Checklist**: How to verify trace quality before mining





























# Optimization Strategies for Trace Generation in Experimental Mathematics via LLM-Driven Python Synthesis

## 1. Introduction: The Neurosymbolic Paradigm in Experimental Mathematics

The domain of experimental mathematics has historically relied on the interplay between human intuition and computational verification. Mathematicians formulate conjectures based on observable patterns in small numerical datasets—traces—and subsequently seek rigorous proofs to generalize these observations. The emergence of Large Language Models (LLMs) has introduced a transformative, albeit stochastic, agent into this workflow. Unlike traditional Computer Algebra Systems (CAS) which are deterministic and rigorous but require precise formal input, LLMs possess the unique capacity to bridge informal natural language descriptions of mathematical problems with executable syntactic structures. This report investigates the optimization of "Trace Generation," defined here as the automated synthesis of Python scripts designed to compute the sequence $f(n)$ for a discrete, finite range of $n$ (typically $1 \le n \le 20$).

The transition from manual script writing to LLM-driven synthesis addresses a critical bottleneck in modern mathematical discovery: the "implementation gap." While a mathematician can often describe a combinatorial object or a number-theoretic property in natural language sentences, translating this definition into an error-free, efficient algorithm is time-consuming and prone to subtle logic errors. LLMs offer a potential solution by functioning as "neurosymbolic translators," converting semantic intent into syntactic code. However, this capability is not inherent; it requires a sophisticated framework of prompt engineering, architectural constraints, and verification loops to mitigate the model's tendency towards hallucination and semantic drift.

The specific challenge of _trace generation_ differs fundamentally from general software engineering. In standard software development, code is often written to handle general cases with asymptotic efficiency. In experimental mathematics, particularly for generating the first few terms of a sequence, the priority shifts heavily towards _correctness_ and _verifiability_ over asymptotic speed. An algorithm that is $O(n!)$ but easy to verify is often preferable to an $O(n^2)$ algorithm that relies on a complex, potentially hallucinated formula. Furthermore, the "trace" itself—the sequence of integers—serves as the ground truth for subsequent symbolic regression or database lookups (e.g., OEIS). Consequently, the integrity of the generated Python script is paramount. This report details a comprehensive methodology for optimizing this generation process, covering prompt architectures, code quality enforcement, domain-specific adaptations, and robust error handling mechanisms.

## 2. Theoretical Frameworks for Code-Based Reasoning

To optimize trace generation, one must first understand the cognitive architectures that enable LLMs to reason mathematically. The evolution from direct answer generation to intermediate reasoning steps has culminated in frameworks that explicitly leverage code as a cognitive prosthetic.

### 2.1 The Semantic Gap and the Necessity of Code

LLMs are probabilistic engines trained to predict the next token in a sequence. When tasked with direct mathematical computation (e.g., "What is the 10th term of the Fibonacci sequence?"), they rely on the statistical correlations present in their training data. For well-known sequences, this "retrieval-augmented" hallucination often produces correct answers. However, for novel or complex experimentally defined functions, direct prompting fails catastrophically due to the "semantic gap"—the disconnect between the model's linguistic understanding of arithmetic concepts and its inability to perform reliable symbolic manipulation.

The integration of code generation transforms the problem from one of _calculation_ to one of _translation_. The model is not asked to compute $f(n)$; it is asked to write a program that computes $f(n)$. This leverages the model's high proficiency in programming syntax, acquired from vast corpora of code (e.g., GitHub), while offloading the computational burden to a deterministic Python interpreter. This distinction is the foundation of "Program of Thought" (PoT) and "Chain of Code" (CoC) methodologies.

### 2.2 Chain of Code (CoC) vs. Chain of Thought (CoT)

While Chain of Thought (CoT) prompting encourages the model to decompose a problem into intermediate natural language steps, it remains vulnerable to logic errors within the reasoning trace itself. A model might correctly describe the steps to solve a modular arithmetic problem but fail in the execution of those steps. Chain of Code (CoC) addresses this by interleaving natural language reasoning with executable code blocks.

In the context of trace generation, CoC allows the model to "think in code." For example, faced with a combinatorial counting problem, a CoT approach might attempt to derive a formula and fail. A CoC approach would prompt the model to:

1. Define the combinatorial object in a Python class or data structure.
    
2. Implement a generator function that yields valid objects.
    
3. Simulate the execution flow (the "LMulator" concept) to verify logic for trivial cases (e.g., $n=1$).
    
4. Construct the final loop to count these objects for the target range.
    

The "LMulator" aspect of Chain of Code is particularly relevant when parts of the problem are semantic (e.g., "count the number of palindromic sentences"). The LLM can simulate the execution of the semantic check `is_palindrome(sentence)` while the interpreter handles the iteration logic. For pure experimental mathematics, however, we enforce a strict "Program of Thought" where _all_ steps must be executable by the interpreter to ensure reproducibility.

### 2.3 System Prompts and Persona Engineering

The "System Prompt" serves as the foundational instruction set that defines the LLM's behavior, tone, and constraints. For experimental mathematics, the persona must be carefully calibrated to avoid the common pitfalls of conversational AI, such as helpfulness (which leads to guessing) or verbosity (which dilutes the code context).

**Optimized Persona Characteristics:**

- **The Rigorous Experimentalist:** The persona should value empirical data over theoretical elegance. The system prompt must explicitly discourage the derivation of closed-form formulas unless they are trivial, as LLMs frequently hallucinate incorrect coefficients in polynomials or recurrence relations.
    
- **The Defensive Coder:** Given the experimental nature of the tasks, the persona must prioritize defensive programming practices—type checking, bound assertions, and explicit state management—to catch errors early in the trace generation process.
    

#### Table 1: Comparison of Prompting Architectures for Math

|**Architecture**|**Mechanism**|**Strength in Math**|**Weakness in Math**|**Optimal Use Case**|
|---|---|---|---|---|
|**Direct Prompting**|Zero-shot Q&A|Speed|High hallucination rate on calculation|Definitions, Concepts|
|**Chain of Thought (CoT)**|Step-by-step text reasoning|Decomposes complex logic|"Semantic Gap" in arithmetic|Proof sketching|
|**Program of Thought (PoT)**|Code as reasoning steps|Verifiable, Deterministic|Limited by coding ability|Arithmetic, Algebra|
|**Chain of Code (CoC)**|Interleaved code/simulation|Handles hybrid semantic/logic tasks|Complex parsing of mixed output|**Experimental Trace Generation**|
|**Reflexion**|Iterative self-correction|High accuracy via debugging loop|High latency/token cost|Hard combinatorial problems|

## 3. Prompt Engineering Strategies for Python Synthesis

Effective prompt engineering for trace generation goes beyond the system prompt. It involves the structural decomposition of the problem and the specific instructions regarding algorithmic choices.

### 3.1 Decomposition and Planning

Complex mathematical problems often overwhelm the context window or reasoning capacity of a single inference pass. "Least-to-Most" prompting strategies are highly effective here. This involves breaking the query "Compute the sequence $f(n)$" into a sequence of sub-prompts:

1. **Decomposition:** "Identify the core mathematical objects and constraints involved in defining $f(n)$."
    
2. **Planning:** "Outline a step-by-step algorithm to generate these objects for a fixed $n$. Do not write code yet."
    
3. **Implementation:** "Translate the outlined algorithm into a Python script using `itertools` and `yield` statements."
    
4. **Verification:** "Write a test function to verify $f(1)$ and $f(2)$ against manual calculations."
    

Research suggests that forcing the model to articulate a "plan" in natural language or pseudocode before committing to Python syntax significantly reduces logical errors, particularly off-by-one errors in loop boundaries.

### 3.2 The "Brute Force" Imperative

One of the most counter-intuitive yet effective optimization strategies for LLM-based experimental math is the explicit instruction to use **brute force**.

- **Rationale:** Analytical solutions (e.g., generating functions, dynamic programming with state compression) require deep insight and are prone to subtle derivation errors. Brute force solutions (e.g., "generate all subsets and check condition") rely on simple, repetitive logic that LLMs generate with high fidelity.
    
- **Trace Constraint:** Since the goal is trace generation for _small_ $n$ (e.g., $n \le 15$), the computational cost of brute force is often acceptable. The generated trace can then be used to _find_ the efficient formula via symbolic regression (e.g., using the Berlekamp-Massey algorithm), effectively reversing the traditional workflow.
    
- **Prompt Instruction:** "Do not attempt to optimize for asymptotic efficiency. Prioritize algorithmic clarity and correctness. Use brute-force enumeration of the state space unless $N > 20$.".
    

### 3.3 Domain-Specific Prompt Templates

Different areas of mathematics require distinct prompting heuristics to maximize code quality.

#### 3.3.1 Combinatorics: The `itertools` Paradigm

For combinatorial problems (permutations, partitions, subsets), the prompt should mandate the use of Python's `itertools` library.

- **Why:** `itertools` functions are implemented in C and highly optimized. More importantly, they provide a standard vocabulary for combinatorial concepts. An LLM using `itertools.combinations(range(n), k)` is less likely to introduce indexing errors than an LLM writing a recursive function `def get_combinations(n, k):...`.
    
- **Prompt Template:** "Use `itertools` to generate candidate structures. Avoid manual recursion for standard combinatorial objects."
    

#### 3.3.2 Number Theory: The `SymPy` Standard

For sequences involving primes, divisors, or modular arithmetic, the prompt must enforce the use of `SymPy`.

- **Why:** Standard Python `float` precision is insufficient for many number-theoretic investigations (e.g., checking if $\sqrt{n}$ is integer). `SymPy` provides infinite-precision integers and verified number-theoretic functions (`isprime`, `factorint`, `totient`).
    
- **Prompt Template:** "Use `sympy` for all number theoretic predicates. Do not implement primality testing or factorization from scratch.".
    

#### 3.3.3 Graph Theory: Representation Constraints

Graph problems are notoriously difficult for LLMs because "graph" is an abstract concept. The prompt must fix the representation.

- **Why:** Textual descriptions of graphs are ambiguous. Forcing an adjacency matrix (using `numpy`) or an edge list (using `networkx`) grounds the problem.
    
- **Prompt Template:** "Represent all graphs using `networkx` objects or adjacency matrices. Use `networkx` algorithms for property checking (e.g., `is_connected`, `clique_number`) rather than implementing BFS/DFS manually.".
    

## 4. Code Quality and Library Optimization

The reliability of the generated trace is directly dependent on the quality of the underlying code. Optimization strategies here focus on leveraging the Python ecosystem to minimize the surface area for LLM-induced bugs.

### 4.1 Library Selection: The "Standard Library" of Experimental Math

Just as a human mathematician would not re-derive calculus to solve a physics problem, an LLM should not re-implement basic algorithms. We curate a "standard library" for trace generation prompts.

- **`itertools`:** As discussed, this is the engine of combinatorial enumeration. It supports lazy evaluation, which is critical for preventing memory overflows when $n$ grows.
    
- **`functools`:** specifically `@lru_cache`. Prompts should explicitly request: "Decorate recursive functions with `@functools.cache` or `@functools.lru_cache(None)`." This simple instruction automatically converts inefficient recursive implementations into dynamic programming solutions, often changing the complexity class of the script from exponential to polynomial without changing the logic.
    
- **`hypothesis`:** For property-based testing. Prompting the LLM to write a `hypothesis` test suite allows for the automated verification of the generated function against invariants (e.g., "the number of partitions is always positive").
    
- **`mpmath` / `fractions`:** For problems requiring non-integer precision, standard floats must be banned. `fractions.Fraction` or `mpmath` must be mandated to ensure exact arithmetic, preventing rounding errors from corrupting the integer trace.
    

### 4.2 Algorithmic Paradigms: Recursion vs. Iteration vs. Generators

The choice of control flow impacts both correctness and resource usage.

- **Generators (`yield`):** This is the optimal pattern for trace generation. A generator function `gen_f()` that yields one term at a time allows the consumer to stop execution arbitrarily (e.g., upon timeout) without losing the partial trace. It also decouples the generation logic from the storage logic.
    
    - _Prompt Strategy:_ "Implement the sequence as a Python generator. Yield each term $f(n)$ as it is computed.".
        
- **Recursion:** While elegant, Python's default recursion limit (1000) is easily hit. If the problem suggests recursion (e.g., tree traversal), the prompt should either request an iterative stack-based implementation or explicit recursion limit handling (`sys.setrecursionlimit`).
    
- **Memoization:** As noted, this is non-negotiable for recursive prompts in this domain.
    

### 4.3 Handling "Count the Number Of" Queries

A frequent task in experimental mathematics is counting substructures (e.g., "Count the number of non-isomorphic graphs on $n$ vertices").

- **Isomorphism:** This is a major trap. LLMs often conflate labeled and unlabeled counting. The prompt must explicitly ask: "Are the structures labeled or unlabeled?" If unlabeled, the script must implement canonicalization.
    
- **Canonicalization:** For graphs, `networkx.weisfeiler_lehman_graph_hash` or similar canonical labeling functions should be requested. For lists/sets, sorting is often sufficient.
    
- **Symmetry Breaking:** In grid-based or geometric problems, the prompt should instruct the model to fix the orientation or position of the first element to break symmetry, thus avoiding overcounting (e.g., counting the same polyomino rotated 90 degrees as distinct).
    

## 5. Problem Adaptation: Handling Complexity and Domain Shift

Experimental mathematics spans disjoint domains, each presenting unique challenges for LLM-based code synthesis.

### 5.1 NP-Hard Problems and Timeout Management

Many interesting functions $f(n)$ are the solutions to NP-hard problems (e.g., Ramsey numbers, Maximum Clique). For these, trace generation is computationally bounded.

- **The Timeout Pattern:** The generated script must be robust to long execution times. A script that hangs indefinitely yields no data.
    
- **Implementation:** The prompt should require a "watchdog" or timeout wrapper. "Wrap the computation of each $f(n)$ in a try-block with a timeout (e.g., using `func_timeout` or `signal`). If the computation exceeds 60 seconds, return 'TIMEOUT' and stop."
    
- **Implication:** This allows the researcher to obtain the trace up to the computational cliff (e.g., $n=12$) rather than receiving nothing because the job for $n=13$ hung the system.
    

### 5.2 Sequence Extrapolation: The Post-Trace Workflow

Once the trace is generated (e.g., `1, 2, 5, 14, 42`), the system often needs to identify the sequence.

- **Berlekamp-Massey Algorithm:** This algorithm finds the shortest linear recurrence relation that generates a given sequence. LLMs can be prompted to _append_ this analysis to their script: "After generating the trace, run the Berlekamp-Massey algorithm to check for linear recurrence.".
    
- **SymPy Sequence Tools:** `sympy.series.sequences.find_linear_recurrence` is a robust built-in alternative.
    
- **OEIS Integration:** The ultimate validation is the On-Line Encyclopedia of Integer Sequences (OEIS). Prompts should structure the output to be OEIS-compliant (comma-separated strings) to facilitate automated lookup. The report notes that LLMs trained on OEIS data might "leak" the answer if the sequence is famous; however, for _new_ conjectures, this lookup serves as a novelty check.
    

### 5.3 Large Integers and Precision Loss

A critical "silent failure" mode involves the handling of large integers.

- **The JSON limit:** Many LLM interaction layers use JSON for output parsing. Standard JSON parsers treat numbers as double-precision floats, which have a precision limit of $2^{53} - 1$ (approx. $9 \times 10^{15}$). Combinatorial sequences often exceed this quickly (e.g., $15! \approx 1.3 \times 10^{12}$, but $20! \approx 2.4 \times 10^{18}$).
    
- **Truncation Error:** If the Python script outputs a large integer as a raw number in JSON, the receiving system might truncate it, altering the trace (e.g., `...789` becomes `...700`).
    
- **Mitigation:** The prompt must rigorously enforce **string serialization** for the trace. "Return the sequence as a list of strings, e.g., `['1', '2', '1000000000000000']`. Do NOT return raw integers.".
    

## 6. Error Handling, Verification, and Reflexion

The stochastic nature of LLMs means that the first draft of a script is frequently flawed. Robust trace generation requires an "Agentic" loop where the model critiques and fixes its own code.

### 6.1 The Reflexion Framework

Reflexion is a technique where the model's output is evaluated, and the feedback (error messages, test failures) is fed back into the model for a second attempt.

- **Cycle:**
    
    1. **Draft:** LLM generates Script A.
        
    2. **Execute:** Script A throws `IndexError` at $n=0$.
        
    3. **Reflect:** The system captures the traceback and prompts the LLM: "The script failed with `IndexError`. Explain why and fix it."
        
    4. **Refine:** LLM generates Script B.
        
- **Effectiveness:** Research indicates that Reflexion can boost performance on coding benchmarks (like HumanEval) from ~60% to over 90%. For experimental math, where edge cases (like $n=0$ or $n=1$) are common points of failure, this loop is essential.
    

### 6.2 Sanity Checks and Property-Based Testing

Since the ground truth $f(n)$ is unknown, how do we verify the script? We verify _properties_.

- **Monotonicity:** "Is the sequence expected to be non-decreasing?" The prompt can ask the LLM to include an assertion: `assert f(n) >= f(n-1)`.
    
- **Integrality:** "Are the results integers?" `assert isinstance(result, int)`.
    
- **Existence Proofs:** For counting problems, the prompt can ask the model to print the _actual objects_ for small $n$ (e.g., $n=3$) alongside the count. A human or a secondary "Verifier" LLM can then inspect the list of objects to see if they match the problem description.
    
- **Hypothesis Library:** A sophisticated prompt asks the LLM to write a `hypothesis` test strategy. "Write a test using the `hypothesis` library that generates random inputs and asserts that $f(n)$ satisfies the recurrence relation you derived." This moves the burden of test case generation from the human to the agent.
    

### 6.3 Consensus and Sampling (AlphaCode Strategies)

For high-stakes trace generation, relying on a single inference is risky. We adapt the strategies from DeepMind's AlphaCode 2 system.

- **Massive Sampling:** Instead of one script, generate $K$ (e.g., 50) distinct scripts using a high temperature ($T \approx 0.7$).
    
- **Execution-Based Filtering:** Run all 50 scripts for a very small range (e.g., $n=1$ to $5$). Filter out those that crash or timeout.
    
- **Clustering:** Group the remaining scripts by their output traces. If 40 scripts produce Trace A and 2 scripts produce Trace B, Trace A has a much higher probability of being correct.
    
- **Selection:** Select the most efficient script (fastest execution time) from the largest cluster as the final generator. This "consensus" approach drastically reduces the impact of stochastic logic errors.
    

## 7. Model-Specific Considerations

The behavior of the LLM itself is a variable in the optimization equation.

### 7.1 DeepSeek-Coder vs. Claude 3.5 vs. GPT-4o

- **DeepSeek-Coder (V2/V3):** This model is highly optimized for code logic and syntax. It is less "chatty" than GPT-4 and adheres strictly to system prompts. It is particularly cost-effective for the "Massive Sampling" strategy due to lower API costs. It excels at implementing low-level algorithms in Python.
    
- **Claude 3.5 Sonnet:** This model demonstrates superior "reasoning" capabilities. It is the preferred model for the "Reflexion" step or for the initial "Planning" phase. It is better at catching subtle semantic misunderstandings in the problem statement (e.g., "does 'subsegment' imply contiguous?").
    
- **GPT-4o:** A strong all-rounder, but can be prone to "lazy" coding (using placeholders like `#... implementation here`) unless explicitly prompted otherwise.
    

### 7.2 Temperature Tuning

- **Zero-Shot Accuracy:** For a single attempt, a low temperature ($T=0.0$ to $0.2$) is optimal to select the most probable (and likely correct) syntax.
    
- **Diversity Sampling:** For the AlphaCode consensus strategy, higher temperatures ($T=0.6$ to $0.8$) are required to force the model to explore different algorithmic approaches (e.g., iterative vs. recursive vs. dynamic programming).
    

## 8. Strategic Framework for Trace Generation

Synthesizing these insights, we propose a unified prompt architecture: **TraceGen**.

### 8.1 The TraceGen System Prompt

# Role

You are an expert Research Software Engineer specializing in Experimental Mathematics.

# Objective

Write a robust, defensive Python script to compute the first N terms of the sequence f(n).

# Protocol

1. **Plan:** Analyze the problem. Is it combinatorial, number-theoretic, or geometric?
    
2. **Strategy:** Choose a brute-force algorithm. Correctness > Speed for small N.
    
3. **Libraries:** Use `itertools` (combinatorics), `sympy` (number theory), `networkx` (graphs).
    
4. **Implementation:** Write the function `compute_sequence(limit)`.
    
    - Use a Generator (`yield`).
        
    - Use `@functools.cache` for recursion.
        
    - Assert types and constraints (Defensive Coding).
        
    - Implement a Timeout mechanism (10s limit).
        
5. **Output:** The script must print the sequence as a list of STRINGS to avoid precision loss.
    

# Output Format

Return ONLY the Python code block.

### 8.2 The Verification Loop

1. **Generate:** produce 10 scripts ($T=0.7$) using DeepSeek-Coder.
    
2. **Filter:** Reject scripts with forbidden keywords (e.g., `os.system`, `numpy` for integer sequences).
    
3. **Execute:** Run scripts for $n=1 \dots 10$.
    
4. **Consensus:** Identify the majority trace.
    
5. **Reflexion:** If consensus is $<60\%$, feed the divergence back to Claude 3.5 Sonnet for analysis.
    

## 9. Case Study: Optimization in Action

Consider the problem: _"Find the number of ways to tile a $3 \times n$ grid with $2 \times 1$ dominoes."_

- **Naive Prompt:** "Write python code to count domino tilings."
    
    - _Result:_ LLM likely hallucinates the formula for $2 \times n$ (Fibonacci) or tries to write a complex DP solution with bugs.
        
- **TraceGen Prompt:** "Use brute force to count tilings for $3 \times n$. Represent the grid as a boolean matrix. Use recursion with memoization (`@cache`). Define a function `can_place(grid, r, c)`."
    
    - _Result:_ LLM writes a standard backtracking (DFS) solution.
        
    - _Library Check:_ `itertools` is not useful here; recursion is. `@cache` is critical.
        
    - _Sanity Check:_ $3 \times n$ must be even for a tiling to exist. The script should assert `if (3*n) % 2!= 0: return 0`.
        
    - _Trace:_ The script correctly outputs `0` for odd $n$ and the correct counts for even $n$.
        
    - _Extrapolation:_ The `Berlekamp-Massey` check on the output trace `1, 0, 3, 0, 11...` would successfully identify the linear recurrence, allowing the user to compute $f(100)$ without the brute force script.
        

## 10. Future Directions and Conclusion

The optimization of LLM prompting for trace generation represents a shift from "AI as Oracle" to "AI as Engineer." By treating the LLM not as a source of truth but as a source of _tools_ (code), we bypass its computational limitations. The integration of Agentic workflows—where Generator, Verifier, and Optimizer agents collaborate—promises to further automate the discovery process in experimental mathematics. Future systems will likely integrate formal verification (e.g., Lean) directly into the CoC loop, proving the correctness of the generated script rather than just testing it. For now, the combination of **Chain of Code**, **Brute-Force Heuristics**, and **Consensus Verification** provides a robust framework for extracting reliable mathematical data from probabilistic models.

---

### Data Tables

#### Table 2: Recommended Libraries for Experimental Math Scripts

|**Domain**|**Library**|**Key Functionality**|**Prompt Instruction**|
|---|---|---|---|
|**Combinatorics**|`itertools`|`permutations`, `combinations`|"Use itertools for enumeration"|
|**Number Theory**|`sympy`|`isprime`, `factorint`|"Use sympy for primality/factoring"|
|**Graph Theory**|`networkx`|`is_isomorphic`, `connected_components`|"Use networkx for graph properties"|
|**Testing**|`hypothesis`|Property-based test generation|"Verify using hypothesis strategies"|
|**General**|`functools`|`lru_cache` (Memoization)|"Memoize recursive functions"|
|**Precision**|`fractions`|`Fraction` (Exact Rational)|"Use Fraction, forbid floats"|

#### Table 3: Error Handling Strategies & Mitigations

|**Error Type**|**Cause**|**Detection**|**Mitigation Prompt**|
|---|---|---|---|
|**Hallucination**|LLM predicts plausible but wrong number|Sanity Check Failure|"Verify output against known cases"|
|**Truncation**|JSON/Floating point limits|Manual Inspection|"Output as list of strings"|
|**Timeout**|Algorithm too slow ($O(N!)$)|Watchdog Timer|"Implement 10s timeout"|
|**RecursionDepth**|Deep recursion without tail opt|Runtime Error|"Use iterative approach or memoization"|
|**Off-by-One**|Loop boundary errors|Property Testing|"Use Hypothesis to test range boundaries"|

---

**Sources:**