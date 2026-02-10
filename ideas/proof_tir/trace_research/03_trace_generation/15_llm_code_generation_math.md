# Deep Research: LLM Code Generation for Mathematical Computation

## Research Objective

Our system relies on LLMs to generate Python code that computes mathematical functions for small inputs. This is NOT solving the math problem — it's translating a math problem into executable code that produces a data trace. We need to understand the landscape of LLM code generation for math, optimize our approach, and understand failure modes.

## Context

The key insight: LLMs are much better at writing code than at mathematical reasoning. By asking for experimental code rather than solutions, we leverage LLM strengths. But we need to optimize this.

Example prompt:
```
"Write Python code to compute the number of ways to tile a 2×n board 
with 1×2 dominoes, for n=1 to 20. Output the list of counts."
```

## Research Questions

### Part A: LLM Code Generation Capabilities

#### 1. State-of-the-Art Benchmarks
- HumanEval, MBPP — what are current SOTA scores?
- Math-specific code generation benchmarks?
- How does performance vary by problem complexity?

#### 2. Model Comparison for Math Code
Which models are best for generating mathematical computation code?
- Code-specialized: DeepSeek-Coder, CodeLlama, StarCoder
- General: Claude, GPT-4, Gemini
- Math-specialized: Qwen-Math, DeepSeek-Math
- Open-source vs closed-source trade-offs

#### 3. Code vs Natural Language
- Success rate of LLM generating correct math code vs correct math reasoning
- Research comparing these?
- Why is code generation easier? (Syntax constraints, executable feedback)

### Part B: Mathematical Code Patterns

#### 4. Common Mathematical Constructs
How well do LLMs generate code for:
- Recursion with memoization (DP)
- Combinatorial enumeration
- Modular arithmetic
- Graph algorithms
- Numerical iteration
- Symbolic computation (using SymPy)

#### 5. Library Usage
Do LLMs correctly use:
- `math` module (factorial, comb, gcd)
- `itertools` (combinations, permutations, product)
- `functools` (lru_cache)
- `sympy` (symbols, solve, simplify)
- `numpy` (for numerical computation)

When do they hallucinate non-existent functions?

#### 6. Efficiency Patterns
Do LLMs generate efficient code?
- Recognize when memoization is needed
- Avoid redundant computation
- Use appropriate data structures
- Complexity awareness

### Part C: Prompt Engineering for Math Code

#### 7. Instruction Clarity
What makes an effective prompt for math code generation?
- Explicit input/output specification
- Constraint highlighting
- Example inputs/outputs
- Pseudo-code guidance

#### 8. "Experimental" Framing
Our key insight: asking to "experimentally determine" rather than "solve"
- Does this framing actually improve results?
- Does it reduce the tendency to attempt analytical solutions?
- What alternative framings work?

#### 9. Small-N Focus
Prompting to compute for "small values of n":
- Prevents attempts at direct computation of large values
- Encourages brute-force/iterative approaches
- How to specify the range (n=1..15, n=1..20)?

#### 10. Output Format Specification
Specifying output format:
- "Print a list of integers"
- "Output as JSON: [{n: 1, value: 1}, ...]"
- "Return a dictionary mapping n to f(n)"
- Which format minimizes parsing errors?

### Part D: Few-Shot Examples

#### 11. Effective Few-Shot Structure
For in-context learning:
- How many examples? (1? 2? 3?)
- What variety of problem types?
- Should examples show the problem → code → output chain?

#### 12. Example Selection
What makes a good few-shot example?
- Similar problem structure to target
- Clean, readable code
- Explicit comments
- Clear output format

#### 13. Chain-of-Thought Before Code
Does it help to have the model:
1. First explain the approach
2. Then write the code
3. Then show expected output

Or directly write code?

### Part E: Failure Modes

#### 14. Common Code Bugs in Math
LLMs frequently produce:
- Off-by-one errors (range(1, n) vs range(1, n+1))
- Wrong base cases for recursion
- Missing memoization (exponential blowup)
- Integer overflow (in languages other than Python)
- Floating-point when integer is needed
- Incorrect modular arithmetic

How common is each?

#### 15. Problem Misinterpretation
- Misreading constraints
- Wrong interpretation of "arrangements"
- Confusing "ordered" vs "unordered"
- Misunderstanding geometric setup

#### 16. Library Hallucinations
- Calling non-existent functions
- Wrong function signatures
- Importing unavailable packages

#### 17. Efficiency Failures
- Code that works for n=5 but times out for n=15
- Exponential recursion without memoization
- O(n²) when O(n) is possible

### Part F: Code Validation

#### 18. Execution-Based Validation
- Run the code, check for errors
- Runtime crashes (easy to detect)
- Timeout (time limit)
- Wrong output format (parser fails)

#### 19. Sanity Checks
- Is output a list of expected length?
- Are all values positive (if expected)?
- Are values monotonic (if expected)?
- Are values within reasonable range?

#### 20. Cross-Validation
- Generate multiple solutions (different prompts or samples)
- Compare outputs
- Majority vote on trace
- Helps detect subtle bugs

### Part G: Error Recovery

#### 21. From Error to Fix
When code fails, how to recover?
- Parse error message
- Show error to LLM with context
- Ask for fix
- What's the success rate of fix attempts?

#### 22. Iterative Refinement
- Run code → get error → fix → repeat
- How many iterations before giving up?
- Diminishing returns after iteration N?

#### 23. Alternative Approaches
If initial approach fails:
- Ask for different algorithm
- Suggest specific technique (DP, recursion, enumeration)
- Provide more hints about the problem

### Part H: Model Selection and Configuration

#### 24. Temperature Settings
- Low temperature (0.0-0.3): more deterministic, safer code
- High temperature (0.7-1.0): more creative, might find novel approaches
- What's optimal for math code generation?

#### 25. Token Limits
- How long are typical math code solutions?
- When is context length a bottleneck?
- Handling very complex problems

#### 26. Sampling Strategy
- Single sample vs multiple samples
- If multiple, how to aggregate?
- Cost-benefit of more samples

### Part I: Competition-Specific Considerations

#### 27. Kaggle Environment Constraints
- Available Python version
- Pre-installed packages
- Memory and time limits
- Offline (no pip install during inference)

#### 28. Time Budget for Code Generation
- How long per LLM inference?
- Budget for retries
- When to give up and try different approach?

#### 29. Model Deployment
- Running LLM on Kaggle GPUs
- Quantization for speed/memory
- vLLM or other inference optimization

### Part J: Benchmarking and Evaluation

#### 30. Trace Generation Benchmark
Create benchmark:
- 100 competition math problems suitable for trace generation
- Ground truth traces for each
- Measure: correct trace rate, code compilation rate, timeout rate

#### 31. Ablation Studies
- With vs without few-shot examples
- With vs without "experimental" framing
- Different models
- Different temperatures

#### 32. Learning Curve
- Does performance improve with more examples in context?
- Does fine-tuning on math code help?
- Transfer from general code generation

## Desired Output Format

1. **Model Comparison**: Which LLMs are best for math code generation
2. **Prompt Templates**: Optimized prompts for different problem types
3. **Few-Shot Library**: Curated examples for in-context learning
4. **Failure Mode Catalog**: Common bugs and how to detect/prevent
5. **Validation Protocols**: How to check generated code
6. **Error Recovery Playbook**: Strategies for fixing failed generation
7. **Performance Benchmarks**: Success rates across problem types
8. **Configuration Guide**: Temperature, sampling, token settings
























# Deep Research: LLM Code Generation for Mathematical Computation

## 1. Introduction: The Neuro-Symbolic Shift in Mathematical AI

The intersection of Large Language Models (LLMs) and mathematical reasoning has undergone a profound paradigm shift in recent years. Historically, models were treated as direct solvers, expected to output numerical answers or rigorous proofs through pure text generation. This approach, while occasionally successful, suffers fundamentally from the stochastic nature of token prediction. LLMs, by design, are probabilistic engines optimized for semantic coherence rather than arithmetic precision. Consequently, they are prone to "arithmetic hallucinations," logical inconsistencies, and a pervasive inability to handle multi-step calculations reliably.

The contemporary frontier, as evidenced by recent literature and benchmarks, treats LLMs not as calculators, but as _programmers_ of mathematical experiments. This methodology, variously termed "Program of Thoughts" (PoT), "Chain of Code" (CoC), or "Program-Aided Language Models" (PAL), leverages the model's superior capability in syntactic translation—specifically, transforming natural language problem statements into executable Python code. By offloading the rigorous computation to a deterministic interpreter, we bridge the gap between the creative, pattern-matching capabilities of neural networks and the precise, axiomatic nature of mathematics.

This report provides an exhaustive analysis of this landscape, specifically focusing on generating data traces for mathematical functions—a technique grounded in Experimental Mathematics. By generating Python code to compute values for small inputs (e.g., $n=1$ to $20$), systems can produce verifiable data sequences (traces) that serve as the foundation for identifying sequences in databases like the On-Line Encyclopedia of Integer Sequences (OEIS) or for deriving general formulas through symbolic regression. The research indicates that while LLMs struggle with multi-step arithmetic, they excel at structuring logical algorithms. However, this shift introduces new challenges: the models must now navigate complex library syntax, optimize algorithmic complexity to avoid timeouts, and adhere to strict runtime constraints.

The following sections dissect the model capabilities, prompt engineering strategies, failure modes, and deployment architectures required to optimize this neuro-symbolic approach. We will explore how models like DeepSeek-Coder-V2 and Qwen-2.5-Coder are challenging proprietary frontiers, how "reasoning" models like DeepSeek-R1 require novel interaction paradigms, and how to robustly deploy these systems in offline environments like Kaggle.

## 2. Landscape of LLM Code Generation Capabilities

The efficacy of an experimental mathematics system hinges entirely on the underlying model's ability to generate correct, efficient, and executable code. The landscape of available models has diversified significantly, with specialized open-weights models increasingly challenging the dominance of proprietary frontier models.

### 2.1. Model Architecture and Specialization

Current research highlights a tripartite division in model specialization: generalist frontier models, code-specialized models, and math-specialized models. Understanding the nuances of each category is essential for selecting the optimal "engine" for trace generation.

#### 2.1.1. Code-Specialized Models: The "Implementers"

Models specifically trained on massive corpora of code have demonstrated state-of-the-art (SOTA) performance on benchmarks like HumanEval and MBPP.

- **DeepSeek-Coder-V2:** This model represents a significant leap in open-weights coding capability. Utilizing a Mixture-of-Experts (MoE) architecture, it achieves performance comparable to GPT-4 Turbo in coding tasks while maintaining inference efficiency. The MoE architecture allows the model to activate only a subset of parameters relevant to the specific task (e.g., Python syntax, algorithmic logic), essentially simulating a much larger dense model without the computational cost. This makes it particularly adept at handling the syntactic rigidity required for mathematical programming. It excels at managing imports, utilizing correct library functions, and adhering to syntax constraints, which are the foundational requirements for translating mathematical word problems into executable scripts.
    
- **Qwen-2.5-Coder:** This family of models, particularly the 32B instruction-tuned variant, has shown exceptional instruction-following capabilities. In the context of trace generation, where prompts often specify strict output formats (e.g., "return a Python list of integers"), instruction adherence is as critical as coding logic. Qwen-2.5-Coder has demonstrated SOTA performance across more than ten code-focused benchmarks, often outperforming larger generalist models. Its training on diverse programming languages and tasks gives it a robust "world model" of programming concepts, allowing it to understand and implement complex algorithms like dynamic programming or breadth-first search (BFS) with high fidelity.
    
- **CodeLlama & StarCoder:** While pioneering, earlier models like CodeLlama and StarCoder have been largely superseded by the DeepSeek and Qwen families in terms of reasoning density and instruction following. However, they established the baseline for code-specialized training and remain relevant in ultra-low resource environments where highly quantized versions are necessary.
    

#### 2.1.2. Math-Specialized Models: The "Reasoners"

The emergence of math-specific models marks a significant advancement in the field. These models are fine-tuned on vast corpora of mathematical text, proofs, and problem-solution pairs, giving them an "intuition" for mathematical structures.

- **Qwen-2.5-Math:** This model leverages techniques like Chain-of-Thought (CoT) and Tool-Integrated Reasoning (TIR) to enhance multi-step problem solving. TIR is particularly relevant for our objective, as it explicitly trains the model to use external tools (like a Python interpreter) to verify intermediate steps. This alignment between training objective and inference task (trace generation) makes Qwen-2.5-Math a potent candidate. It is more likely to correctly identify the _type_ of mathematical problem (e.g., identifying a problem as a combinatorial enumeration rather than a geometric probability task) compared to a pure code model.
    
- **DeepSeek-Math & DeepSeek-R1:** DeepSeek-Math-V2 and its reinforcement learning variants (DeepSeek-R1) have pushed the boundaries of symbolic reasoning, achieving high scores on the MATH benchmark. DeepSeek-R1, in particular, utilizes a unique training paradigm that encourages extensive "internal thought" (visible as `<think>` traces) before outputting a solution. This allows the model to explore the solution space, plan the algorithm, and self-correct logical errors before committing to code. This "System 2" thinking capability is invaluable for complex combinatorial problems where the direct translation from text to code is non-trivial. However, these models can be "chatty" and harder to control programmatically, often requiring specific prompting strategies to suppress the internal monologue or format the output correctly.
    

#### 2.1.3. Generalist Models: The "Verifiers"

Generalist models like GPT-4o and Claude 3.5 Sonnet remain the gold standard for broad reasoning and robustness.

- **Claude 3.5 Sonnet:** This model has been lauded for its "agentic" coding capabilities—its ability to debug, iterate, and handle complex contexts. In a trace generation pipeline, Claude 3.5 Sonnet excels as a **Verifier** or **Error Recovery Agent**. If a primary open-weights model generates code that throws a runtime error, passing the error message and the code to Claude often results in a correct fix. Its superior "understanding" of intent allows it to spot subtle logical bugs (e.g., off-by-one errors in loop boundaries) that might escape more specialized but less robust models.
    
- **GPT-4o:** Continues to lead in broad math capability and instruction following. Its primary advantage is consistency and the vast knowledge base it draws upon. However, recent benchmarks suggest that for specific coding and symbolic tasks, specialized open models like Qwen-2.5-Math are closing the gap, sometimes surpassing it in specific domains like symbolic manipulation or pure coding efficiency.
    

### 2.2. Comparative Analysis: Code vs. Natural Language Reasoning

A critical insight derived from the literature is the significant performance disparity between CoT (text-based reasoning) and PoT/CoC (code-based reasoning) for mathematical tasks.

Research on "Program-Aided Language Models" (PAL) demonstrates that offloading reasoning to code significantly outperforms text-based chains, especially for arithmetic-heavy tasks. On the GSM8K benchmark, PAL approaches using Codex outperformed much larger PaLM-540B models using standard CoT by substantial margins (up to 15% absolute accuracy). This confirms the hypothesis that LLMs are better _translators_ (Natural Language $\to$ Python) than _calculators_.

The "Chain of Code" (CoC) framework extends this by handling "semantic" sub-tasks that are difficult to code directly. CoC encourages the model to write pseudocode or call hypothetical functions for semantic steps (e.g., `is_sarcastic(sentence)`), which are then simulated by the LLM (acting as an "LMulator") while the deterministic parts are executed by a Python interpreter. This hybrid approach achieves 84% on BIG-Bench Hard, a 12% gain over CoT.

**Table 1: Comparative Strengths of Reasoning Approaches**

|**Feature**|**Chain of Thought (CoT)**|**Program of Thoughts (PoT)**|**Chain of Code (CoC)**|
|---|---|---|---|
|**Mechanism**|Step-by-step natural language reasoning.|Translation of problem to executable Python code.|Hybrid: Code for math, LLM simulation for semantics.|
|**Arithmetic Accuracy**|Low (Probabilistic). Prone to calculation errors.|High (Deterministic). Python handles math.|High (Deterministic).|
|**Semantic Reasoning**|High. Can handle ambiguity and nuance.|Low. Requires logic to be fully formalizable.|High. "LMulator" handles semantic gaps.|
|**Trace Generation**|Poor. Manual calculation of sequences is error-prone.|Excellent. Loops/recursion are native to code.|Excellent.|
|**Best Use Case**|Logic puzzles, qualitative reasoning, proofs.|Math word problems, Combinatorics, Probability.|Mixed semantic-numeric tasks, data analysis.|

### 2.3. Strategic Model Selection for Trace Generation

For the specific objective of generating data traces ($n=1..20$) for mathematical problems, the analysis suggests the following hierarchical strategy:

1. **Primary Driver:** **DeepSeek-Coder-V2** or **Qwen-2.5-Coder (32B)**. These models offer the best balance of coding proficiency and mathematical understanding. Their specialized training on codebases allows them to use libraries like `itertools` and `sympy` more effectively than pure math models. They act as the "workhorses" of the pipeline.
    
2. **Reasoning Specialist:** **DeepSeek-R1** (or distilled variants). While powerful, R1 requires specific prompting strategies. It serves best as a "planner" for extremely complex problems where the algorithm is not immediately obvious. It can generate the _logic_ which is then implemented by the coding model.
    
3. **Fallback/Verifier:** **Claude 3.5 Sonnet**. Its superior debugging capability makes it an excellent choice for an error recovery agent. If the primary model's code fails to execute or produces obvious errors (e.g., empty lists), Claude can analyze the trace and fix the script.
    

## 3. Mathematical Code Patterns and Library Usage

To successfully generate traces, LLMs must translate mathematical concepts into executable Python patterns. The research identifies several key constructs and libraries that are essential for this task, as well as common pitfalls associated with them.

### 3.1. Common Mathematical Constructs and Patterns

**Combinatorial Enumeration:**

Problems involving counting arrangements, selections, or graph structures are best handled using Python's `itertools`. LLMs generally exhibit strong proficiency in using `permutations`, `combinations`, and `product`. A common pattern observed is generating all possibilities and filtering them based on constraints.

- _Pattern:_ `count = sum(1 for p in permutations(range(n)) if check(p))`
    
- _Strengths:_ This "brute force" approach is highly effective for small $N$ (experimental math) and is less prone to logic errors than deriving the exact combinatorial formula.
    
- _Weaknesses:_ It fails for larger inputs due to factorial complexity ($O(N!)$). However, for $N < 15$, it is often the most reliable method.
    

**Recursion and Dynamic Programming (DP):**

For sequences defined by recurrence relations (e.g., tiling problems, partition counts), LLMs frequently employ recursion. A critical success factor is the inclusion of memoization.

- _Pattern:_ Using `@functools.cache` or `@lru_cache` decorators.
    
- _Insight:_ Models like GPT-4 and DeepSeek-Coder often correctly apply these decorators when prompted for efficiency, transforming exponential time complexities ($O(2^n)$) into linear or polynomial time. Failure to use memoization is a common failure mode in naive prompts, leading to timeouts.
    

**Symbolic Computation:**

The `sympy` library is the de facto standard for symbolic math in Python. LLMs use it to solve equations, simplify expressions, and perform calculus operations.

- _Pattern:_ Defining symbols (`x = symbols('x')`), setting up equations (`Eq(lhs, rhs)`), and using `solve`.
    
- _Pitfall:_ A recurrent issue is the misuse of `solve`—specifically, failing to define symbols with appropriate assumptions (e.g., `real=True`, `integer=True`). This often leads to `solve` returning complex roots or general expression objects instead of the desired integers.
    

### 3.2. Library Usage: Capabilities and Hallucinations

**Correct Usage:**

- **`math`**: Reliably used for standard functions like `factorial`, `gcd`, `sqrt`, `comb`.
    
- **`itertools`**: Models effectively use it for generating search spaces.
    
- **`numpy`**: Often used for matrix operations or array manipulations, though sometimes overkill for simple integer sequences.
    

**Hallucinations and Errors:**

The analysis reveals that LLMs sometimes hallucinate non-existent functions within these libraries or misuse existing ones.

- **SymPy Hallucinations:** Models may invent functions like `sympy.is_prime` (which technically exists as `sympy.ntheory.isprime` but is often miscalled) or misuse `solve` syntax. A frequent error is attempting to solve inequalities directly with `solve` without using the correct `Reduce` or `solveset` paradigms.
    
- **Library Imports:** In constrained environments (like Kaggle offline notebooks), models may attempt to import libraries that are not installed (e.g., `z3-solver`, `scipy` extensions), leading to `ModuleNotFoundError`. This is particularly common when the model tries to use advanced solvers for logic puzzles.
    

### 3.3. The "Experimental" Framing: Simulation vs. Analysis

Prompting the model to "experimentally determine" the sequence rather than "solve" the problem analytically is a crucial optimization strategy.

- **Analytical Approach:** Asking "How many ways to tile..." often triggers the model to attempt to derive a generating function or a closed-form formula. This is a high-risk operation; LLMs are prone to subtle logical errors in derivation, leading to formulas that look plausible but are mathematically incorrect (hallucinated coefficients, wrong powers).
    
- **Experimental Approach:** The "experimental" frame encourages the generation of a _simulation_—writing code that models the problem mechanics directly (e.g., actually constructing the board and placing dominoes) and counting the results. This leverages the LLM's strength in translation (Language $\to$ Simulation Code) over its weakness in symbolic derivation.
    
- **Benefit:** The code for a simulation (e.g., "try all placements, check validity") is often simpler and more robust than the code for a complex mathematical formula.
    

## 4. Prompt Engineering for Math Code Generation

Effective prompting is the interface between the user's intent and the model's execution. The research highlights several advanced techniques to maximize code correctness and reliability.

### 4.1. The Chain of Code (CoC) & Program of Thoughts (PoT) Templates

**Program of Thoughts (PoT):**

The PoT paradigm completely separates reasoning from computation. The prompt explicitly instructs the model to write a Python program to solve the problem, rather than solving it in the text generation.

- _Template:_ "Write a Python function `solve()` that computes X. Do not compute the answer yourself. The function should return...".
    
- _Benefit:_ Disentangles logic generation from arithmetic, preventing calculation errors. It forces the model into "coding mode," where it attends to syntax and logic rather than prose.
    

**Chain of Code (CoC):**

CoC is more flexible, allowing the model to interleave text and code, or use pseudocode for semantic steps.

- _Structure:_
    
    1. **Think:** "To solve this, I need to count the number of valid permutations..."
        
    2. **Code (Pseudocode/Python):** `count = 0; for p in permutations: if is_valid(p): count += 1`
        
    3. **Refinement:** "The `is_valid` check is complex, let me define it...".
        
- _Application:_ This is ideal for problems that have a mix of semantic constraints (e.g., "palindromic") and numeric ones.
    

**System Prompts for Reasoning Models (DeepSeek R1):**

DeepSeek R1 and similar reasoning models respond poorly to standard rigid system prompts. The documentation advises against complex system instructions. Instead, the user prompt should trigger the reasoning process naturally.

- _Trigger:_ "Please reason step by step, and put your final answer within \boxed{}." or simply allow the model to output `<think>` tags.
    
- _Constraint:_ For code generation, one must explicitly guide the reasoning to conclude with code: "Reason about the algorithm, then implement it in Python."
    

### 4.2. "Small-N" Focus and Experimental Framing

Explicitly constraining the problem to "small values of $n$" (e.g., $n=1$ to $15$) serves a dual purpose:

1. **Computational Feasibility:** It permits inefficient $O(2^n)$ or $O(n!)$ algorithms (like brute-force recursion or permutation iteration) which are easier for LLMs to write correctly than optimized $O(n)$ DP solutions. Writing a brute-force checker is often trivial and error-free compared to deriving a recurrence relation.
    
2. **Error Prevention:** It reduces the likelihood of the model attempting to derive a closed-form formula, which is a high-risk operation.
    

- _Prompt Pattern:_ "Write a Python script to compute $f(n)$ for $n=1$ to $20$ by simulating the process. Do not attempt to derive a formula. Use brute force if necessary."
    

### 4.3. Few-Shot Examples (In-Context Learning)

The inclusion of few-shot examples significantly improves performance. The analysis suggests that examples should mimic the _structure_ of the desired output (trace generation) rather than just the problem type.

- **OEIS-Style Examples:** Providing examples where the input is a problem description and the output is a Python script printing a list `[1, 2, 5, 14,...]` aligns the model with the trace generation objective.
    
- **Format Specification:** Examples should strictly demonstrate the desired output format (e.g., JSON or a comma-separated string) to minimize parsing errors during validation. If the model sees 3 examples of code printing a list, it is statistically highly probable to output code that prints a list.
    

### 4.4. Prompt Template Library

Based on the research, here are optimized templates for different problem types:

**Template A: Combinatorial / Sequence Problems (PoT Style)**

# Role

You are an expert mathematician and Python programmer. Your goal is to experimentally determine the first 20 terms of a sequence based on a problem description.

# Task

Problem: {problem_description}

# Instructions

1. Write a Python script to compute the answer for n=1 to n=20.
    
2. Use a brute-force or simulation approach. Do not attempt to derive a closed-form formula.
    
3. The code must print the output as a Python list: `[term_1, term_2,..., term_20]`.
    
4. Use the `micropip` or standard libraries. Allowed: `itertools`, `math`, `numpy`, `sympy`.
    
5. Handle large numbers automatically (Python handles arbitrary precision integers).
    

# Code Structurepython

import itertools

def solve():

results =

for n in range(1, 21):

# Implementation here

pass

print(results)

solve()

**Template B: Reasoning-Heavy Problems (DeepSeek-R1 Style)**

Problem: {problem_description}

Please reason step-by-step to understand the problem structure.

Then, generate a Python script that simulates the problem to find the first 20 terms.

Enclose the code inpython... ``` blocks.

```

## 5. Failure Modes and Mitigation Strategies

Despite optimal prompting, code generation is prone to errors. Understanding these failure modes is essential for building robust systems that can recover from failures.

### 5.1. Taxonomy of Common Bugs

**1. SymPy "Solve" Complexities:**
A frequent issue involves `sympy.solve` returning complex roots or expression objects instead of integers.
*   *Cause:* Failure to specify `real=True` or `integer=True` in `Symbol` definitions.[20] The solver defaults to the complex domain.
*   *Mitigation:* Prepend assumptions in prompts: "Use `Symbol('x', integer=True)`". Alternatively, use `solveset(eq, x, domain=S.Integers)`.

**2. Floating Point Precision:**
LLMs often default to floating-point division (`/`) instead of integer division (`//`) or `sympy.Rational`. This leads to precision errors in recursive sequences where exact integer values are required.
*   *Mitigation:* Enforce integer arithmetic in prompts. "Use integer division `//` and avoid floating point numbers." Explicitly ban `float` types in the system prompt.

**3. Off-by-One Errors:**
Common in loop ranges (`range(n)` vs `range(n+1)`) and base cases for recursion.
*   *Detection:* Sanity checks (e.g., is the first term correct based on the problem statement?). Asking the model to "double check the base case n=1" in the reasoning trace can help.

**4. Library Hallucination:**
Calling non-existent functions (e.g., `math.combinations` instead of `math.comb`).
*   *Mitigation:* Self-correction loops that feed the `AttributeError` back to the model.

### 5.2. Efficiency Failures (Timeouts)

Kaggle and other competitive environments impose strict time limits (e.g., 9 hours for CPU notebooks, but much less per problem in a batch).
*   *Issue:* Models often generate naive recursive solutions ($O(2^n)$) without memoization.
*   *Detection:* A "watchdog" timer during execution.
*   *Recovery:* If a timeout occurs, prompt the LLM to "Optimize the code using dynamic programming or memoization." The error message `TimeoutError` should trigger a retry with an explicit efficiency constraint.

### 5.3. Error Recovery Playbook: Reflexion and Iterative Refinement

The "Reflexion" pattern is highly effective for code correction. It mimics a human developer's workflow: write, run, see error, fix.
1.  **Execute Code:** Capture `stdout` and `stderr`.
2.  **Check Status:** If `stderr` contains an error (SyntaxError, Timeout, Runtime), construct a new prompt.
3.  **Reflexion Prompt:** "The code failed with the following error: [Error Message]. Fix the code. Ensure you handle [Edge Case]."
4.  **Loop:** Repeat for $K$ iterations (typically 3-5 yields diminishing returns).[28, 29]

**Self-Debugging:**
Asking the model to generate print statements or unit tests within the code can help it "self-debug" during the generation phase, although this consumes tokens.[30]

## 6. Code Validation and Verification Protocols

Trusting LLM-generated code requires rigorous validation. Since we are solving problems where the answer might be unknown, we cannot simply check against a ground truth. We must verify the *code's behavior* and the *trace's structure*.

### 6.1. Execution-Based Validation

The primary validation is successful execution without throwing exceptions. This filters out syntax errors, hallucinations, and runtime crashes.
*   *Sandbox:* Execution must happen in a sandboxed environment to prevent malicious operations (though less of a concern in offline competitions).
*   *Output Parsing:* The output must be parseable into the expected format (e.g., a list of integers). If parsing fails, it counts as an execution failure.

### 6.2. Sanity Checks and Heuristics

*   **Type Checking:** The output must be a list of integers. Floats or complex numbers indicate a logic error.
*   **Monotonicity/Positivity:** Many combinatorial sequences are strictly increasing and positive. Checks for negative numbers or zeros (unless expected) can flag incorrect logic.[16]
*   **Trace Consistency:** If possible, generate code using two different methods (e.g., "brute force" and "recursive") and compare the traces. Agreement increases confidence (Self-Consistency).[31]

### 6.3. The Berlekamp-Massey Algorithm: Mathematical Verification

A powerful, under-utilized validation technique for integer sequences is the **Berlekamp-Massey algorithm**.
*   **Insight:** Many combinatorial sequences satisfy a linear recurrence relation (e.g., Fibonacci, Pell numbers).
*   **Validation:** Run Berlekamp-Massey on the generated trace. If it finds a short linear recurrence that generates the sequence, it strongly suggests the trace is not random noise and likely correct (or at least mathematically structured). This serves as a "mathematical syntax check" for the trace itself.[32, 33]
*   **Heuristic:** If the length of the recurrence $L$ is less than half the number of terms $N$ ($L < N/2$), the sequence is reliably determined by the recurrence. If the algorithm returns a recurrence of length $N/2$, it typically means no linear recurrence exists (or the data is random).

## 7. Deployment in Constrained Environments (Kaggle)

Running these pipelines in a Kaggle notebook (often offline) requires specific engineering. This is a common scenario for AI Math Olympiad competitions.

### 7.1. Offline Dependency Management

Kaggle competitions often disable internet access during inference. Libraries like `vllm`, `bitsandbytes`, or specific versions of `sympy` must be installed via pre-downloaded "wheels".
*   **Method:**
    1.  Create an online notebook.
    2.  Use `pip download` to fetch `.whl` files for all dependencies (and their sub-dependencies) into a directory.
    3.  Zip this directory and upload it as a Kaggle Dataset.
    4.  In the offline inference notebook, add this Dataset and install using: `pip install --no-index --find-links=/kaggle/input/dataset_name package_name`.[34, 35]
*   **Critical Note:** You must match the Python version and CUDA version of the Kaggle environment when downloading wheels. vLLM is particularly sensitive to this. Using pre-curated datasets (like "vllm-0.9.2-offline-installer") is often safer than building your own.[34]

### 7.2. Efficient Inference (vLLM)

For generating traces for hundreds of problems, standard HuggingFace `pipeline` is too slow.
*   **Solution:** Use **vLLM** (Virtual Large Language Model) for high-throughput batched inference. vLLM optimizes memory management (PagedAttention) and allows batching multiple prompts (e.g., 100 math problems) into a single forward pass.[34, 36]
*   **Configuration:** Set `tensor_parallel_size` to match the number of GPUs (e.g., 2x T4 on Kaggle). This enables the model to be sharded across GPUs, increasing speed and memory capacity.

### 7.3. Hardware Constraints

Kaggle provides limited GPU memory (2x T4 15GB or P100).
*   **Model Selection:** 7B to 14B parameter models (Qwen-2.5-Coder-7B, DeepSeek-Math-7B) fit comfortably.
*   **Quantization:** 32B models (like Qwen-2.5-Coder-32B) may require quantization (4-bit/8-bit via `bitsandbytes` or AWQ) to fit in memory.[37] While quantization can slightly degrade reasoning, for code generation, the impact is often minimal compared to the gain in model capability.

## 8. Benchmarking and Evaluation

To measure progress, a specific benchmark for *trace generation* is required, distinct from standard "answer finding" benchmarks like MATH.

### 8.1. OEIS Trace Generation Benchmark

Recent work proposes benchmarks based on the OEIS.
*   **Structure:** 1000 sequences categorized as "easy" (direct simulation) and "hard" (complex recurrences).
*   **Task:** Generate Python code to output the first $N$ terms.
*   **Metrics:**
    *   **Trace Accuracy:** Percentage of sequences where the generated trace matches the ground truth exactly.
    *   **Pass@1:** Success rate with a single generation attempt.
    *   **Pass@K:** Success rate given $K$ attempts (useful for measuring the efficacy of iterative refinement).
    *   **Lookup Table Detection:** Ensuring the model isn't just memorizing the sequence. The benchmark includes "cheating detection" to flag if the code simply prints a hardcoded list without computation.[27, 38]

## 9. Conclusion and Strategic Recommendations

The transition to **Chain of Code** represents a maturation of mathematical AI. By treating LLMs as semantic translators that map ambiguous problem statements to precise, executable Python code, we bypass the inherent arithmetic weaknesses of neural networks. This approach is not just a workaround; it is a fundamental shift towards neuro-symbolic AI, where the neural network handles the intuition and the symbolic engine (Python) handles the rigor.

**Strategic Recommendations:**
1.  **Model Standardization:** Adopt **Qwen-2.5-Coder-32B** or **DeepSeek-Coder-V2-Lite** as the primary code generation engine. Their SOTA performance on instruction following and library usage makes them superior to pure math models for this specific task.
2.  **Prompt Strategy:** Adopt the **"Experimental Mathematics" frame**. Explicitly ask for a *simulation* script for small $N$. Use PoT/CoC templates that separate reasoning from execution to prevent "thinking" errors from contaminating the code.
3.  **Validation Loop:** Implement a robust execution sandbox with **Reflexion** loops. Integrate the **Berlekamp-Massey** algorithm as a lightweight heuristic to validate the mathematical structure of generated traces, filtering out "hallucinated" sequences.
4.  **Deployment:** Prioritize **vLLM** for offline batch inference on Kaggle. The throughput gain is essential for enabling the iterative "generate-verify-fix" loops that are necessary for high reliability.

This neuro-symbolic approach—combining the creativity of LLMs with the rigor of Python execution—is the optimal path forward for automated mathematical discovery and trace generation. It transforms the LLM from a prone-to-error calculator into a tireless and increasingly capable research assistant.
```