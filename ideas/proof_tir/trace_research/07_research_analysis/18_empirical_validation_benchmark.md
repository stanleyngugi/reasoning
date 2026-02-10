# Deep Research: Empirical Validation — Benchmark Design and Testing Methodology

## Research Objective

Before deploying Trace-to-Lean in AIMO 3, we need rigorous empirical validation. This means designing benchmarks, running experiments, measuring success rates, and identifying failure modes. We need a scientific approach to validate our claims.

## Context

We claim:
- >99% success rate on combinatorics/number theory problems amenable to linear recurrence
- Significant improvement over TIR-only approaches due to verification
- Competitive performance on geometry and algebra with Coordinate Descent and Numerical Sniper

These claims need empirical evidence.

## Research Questions

### Part A: Benchmark Construction

#### 1. Problem Selection
Where to get test problems:
- AIME problems (2015-2025): ~300 problems
- USAMO/IMO problems: ~120 problems
- AIMO 1 & 2 problems (if available): ~100 problems
- NuminaMath training/validation set
- Synthetic problems (generated)

#### 2. Labeling Requirements
For each problem, label:
- Primary category (Combinatorics, Number Theory, Geometry, Algebra)
- Subcategory (Counting, Recurrence, Modular, etc.)
- Expected answer
- Ground truth trace (if computable)
- Ground truth formula (if known)
- Tractability for Trace-to-Lean (yes/no/partial)

#### 3. Tractability Criteria
What makes a problem "tractable" for Trace-to-Lean?
- Answer can be computed for small n
- Trace forms a recognizable pattern
- Pattern can be verified in Lean
- NOT a proof-based problem

#### 4. Benchmark Stratification
Create stratified test sets:
- Easy: Obvious recurrence, direct computation
- Medium: Requires tier 2-3 mining, some edge cases
- Hard: Requires full pipeline, edge cases, retries

### Part B: Experimental Design

#### 5. Controlled Variables
What to control:
- Same LLM model for all tests
- Same prompts
- Same mining algorithms
- Same Lean configuration
- Same time limits

#### 6. Independent Variables
What to vary:
- Problem type
- Difficulty level
- Mining tier used
- With/without Lean verification

#### 7. Dependent Variables (Metrics)
What to measure:
- **Accuracy**: % of correct final answers
- **Precision**: % of verified answers that are correct
- **Recall**: % of tractable problems solved
- **Latency**: Time per problem (mean, median, P95)
- **Verification Rate**: % of problems that reach Lean verification
- **Mining Success Rate**: % of traces that yield valid formula

### Part C: Baseline Comparisons

#### 8. TIR Baseline
Compare against pure TIR approach:
- NuminaMath-style: SC-TIR with majority voting
- Same model, same problems
- Measure: accuracy, confidence, failure modes

#### 9. Ablation Studies
Test each component's contribution:
- Full pipeline vs no Lean verification
- Full pipeline vs no mining (direct LLM formula)
- Each mining tier in isolation

#### 10. Oracle Comparisons
Upper bounds:
- Human performance on same problems
- Commercial API performance (GPT-4, Claude)
- What's the theoretical maximum?

### Part D: Per-Module Testing

#### 11. Trace Generator Testing
Test trace generation in isolation:
- Input: problem statement
- Output: correct trace? (Compare to ground truth)
- Metrics: trace accuracy, code compilation rate, timeout rate

#### 12. Invariant Miner Testing
Test mining in isolation:
- Input: ground truth trace
- Output: correct formula?
- Test each tier: B-M, Lagrange, LLM, OEIS
- Metrics: formula accuracy, tier success rates

#### 13. Lean Verifier Testing
Test verification in isolation:
- Input: (formula, trace) pairs — some correct, some wrong
- Output: correctly accepts/rejects?
- Metrics: false positive rate, false negative rate, latency

#### 14. Router Testing
Test classification in isolation:
- Input: problem statements
- Output: correct category?
- Metrics: classification accuracy per category

### Part E: End-to-End Testing

#### 15. Full Pipeline Test
Run complete pipeline on benchmark:
- Problem → Final Answer
- Record all intermediate results
- Measure end-to-end metrics

#### 16. Time Budget Simulation
Simulate competition conditions:
- 50 problems, 5 hours
- Sequential processing with time budgets
- Measure: problems solved, time utilization

#### 17. Stress Testing
Push system limits:
- Very long traces (n=1..100)
- Very complex formulas
- Adversarial problems
- Measure: failure modes, graceful degradation

### Part F: Failure Analysis

#### 18. Failure Categorization
For every failure, categorize:
- Trace generation failed
- Mining failed (no valid formula)
- Translation failed (Lean syntax error)
- Verification failed (formula wrong)
- Timeout
- Other

#### 19. Root Cause Analysis
For each failure category:
- What's the underlying cause?
- Is it fixable? How?
- Is it fundamental to the approach?

#### 20. Failure Case Studies
Deep dive on 10-20 interesting failures:
- What was the problem?
- What did each stage produce?
- Where did it go wrong?
- What would have fixed it?

### Part G: Statistical Analysis

#### 21. Confidence Intervals
For all metrics:
- Point estimate
- 95% confidence interval
- How many test problems needed for significance?

#### 22. Hypothesis Testing
Key hypotheses:
- H1: Trace-to-Lean accuracy > TIR accuracy
- H2: Verified answers have higher precision
- H3: Pipeline latency < 6 min average

Statistical tests to validate.

#### 23. Effect Size
Not just "statistically significant" but:
- How much better?
- Practical significance
- Cohen's d or equivalent

### Part H: Generalization Testing

#### 24. Cross-Competition Transfer
Train on AIME, test on USAMO:
- Does performance generalize?
- Overfitting to AIME-style problems?

#### 25. Year-Based Split
Train on 2015-2022, test on 2023-2025:
- Does performance generalize to newer problems?
- Any data contamination concerns?

#### 26. Synthetic Problem Testing
Generate novel problems programmatically:
- New instances of known problem types
- Test generalization beyond training distribution

### Part I: Performance Profiling

#### 27. Latency Breakdown
For each stage:
- Mean, median, P95, P99 latency
- Contribution to total time
- Bottleneck identification

#### 28. Resource Utilization
During test runs:
- GPU utilization over time
- CPU utilization over time
- Memory usage profile
- Disk I/O

#### 29. Scalability Analysis
How does performance change with:
- More problems (throughput)
- Longer traces (per-problem latency)
- Larger verification ranges (Lean load)

### Part J: Reporting and Documentation

#### 30. Result Tables
Standard format for reporting:
- Accuracy by category
- Accuracy by difficulty
- Comparison with baselines
- Statistical significance

#### 31. Visualization
Graphs to create:
- Accuracy vs problem type (bar chart)
- Latency distribution (histogram)
- Pipeline success funnel (sankey diagram)
- Learning curves (if applicable)

#### 32. Reproducibility
Ensure results are reproducible:
- Fixed random seeds
- Versioned code
- Documented configuration
- Public benchmark (if possible)

## Desired Output Format

1. **Benchmark Specification**: Detailed description of test problems
2. **Experimental Protocol**: Step-by-step testing procedure
3. **Metric Definitions**: Precise formulas for all metrics
4. **Baseline Specifications**: What we compare against
5. **Statistical Analysis Plan**: Tests and significance thresholds
6. **Result Templates**: Tables and figures to produce
7. **Failure Analysis Framework**: How to categorize and investigate failures
8. **Reproducibility Checklist**: What's needed to replicate


















# Empirical Validation Framework for Trace-to-Lean in AIMO 3: Benchmark Design and Testing Methodology

## 1. Executive Summary

The progression of the Artificial Intelligence Mathematical Olympiad (AIMO) toward its third iteration necessitates a fundamental shift in the evaluation paradigms for Large Reasoning Models (LRMs). While previous competitions rewarded systems proficient in "Tool-Integrated Reasoning" (TIR)—whereby Python interpreters verify numerical calculations—AIMO 3’s increasing emphasis on combinatorics, number theory, and abstract proof structures exposes the inherent limitations of stochastic code execution. Current state-of-the-art models, including the AIMO 1 winning NuminaMath solutions, exhibit "cognitive offloading," frequently deferring deep logical verification to external tools that lack the semantic capacity to validate universal truths or structural invariants.

This report articulates a comprehensive empirical validation framework for "Trace-to-Lean," a neuro-symbolic architecture designed to transcend these limitations by bridging informal reasoning traces with formal verification in the Lean 4 theorem prover. The central hypothesis posits that by converting natural language reasoning into formal code, Trace-to-Lean creates a rigorous "semantic filter" that eliminates the false positives endemic to TIR approaches. To validate this, we propose the construction of a stratified "Shadow Benchmark" derived from deeply curated subsets of AIME, USAMO, and OEIS datasets, explicitly designed to isolate reasoning depth from computational rote.

Our methodology introduces a multi-layered experimental design targeting four critical modules: the **Trace Generator**, utilizing Riemannian activation steering to maximize reasoning diversity ; the **Invariant Miner**, leveraging execution traces to extract formal constraints; the **Autoformalizer**, translating informal logic to Lean 4; and the **Router**, a meta-controller determining the optimal verification path. We define a rigorous statistical protocol utilizing unbiased Pass@k estimators and bootstrap confidence intervals to measure performance gains. Furthermore, we establish a novel failure taxonomy for autoformalization—categorizing errors into premise misalignment, goal hallucination, and tactic failure—to guide iterative refinement. This framework provides the scientific foundation necessary to certify Trace-to-Lean’s readiness for the adversarial environment of AIMO 3.

---

## 2. Research Context and Theoretical Foundations

### 2.1 The Trajectory of AIMO and the Verification Gap

The AIMO initiative, funded by XTX Markets, aims to create AI models capable of winning a gold medal in the International Mathematical Olympiad (IMO). The competition has evolved rapidly. The first progress prize was dominated by "NuminaMath," a solution based on DeepSeekMath-7B fine-tuned with "Chain of Thought" (CoT) and "Tool-Integrated Reasoning" (TIR). These models generate interleaved text and Python code, executing the code to verify intermediate steps.

While effective for algebra and calculus, TIR creates a "Verification Gap" in domains requiring abstract logic. In combinatorics and number theory, a Python script can verify a specific instance (e.g., "Check if $n=5$ works") but cannot easily prove a universal property (e.g., "Prove $n$ is never divisible by 7 for all $k$"). Consequently, TIR models often produce "false positives"—solutions that pass limited test cases but rely on flawed logic. AIMO 3, with its focus on "Olympiad-level" problems and rigorous evaluation, requires a system that can verify the _logic_ itself, not just the output.

### 2.2 The Trace-to-Lean Architecture

"Trace-to-Lean" addresses this gap by replacing the Python interpreter with the Lean 4 proof assistant. The architecture operates on the principle of **Autoformalization**: the automatic translation of natural language mathematics into formal definitions and theorems.

The core value proposition of Trace-to-Lean is the **"Rigor Bonus"**: the ability to filter out solutions that are numerically plausible but logically incoherent. If a reasoning trace relies on a non-existent lemma or a flawed deduction, the Lean kernel will fail to compile the corresponding proof, acting as a hard filter against hallucination. This contrasts with TIR, where such errors often propagate undetected if they do not trigger a runtime exception.

### 2.3 Research Objectives

This validation framework is designed to answer four primary research questions:

1. **Benchmark Fidelity:** How can we construct a validation set that accurately predicts AIMO 3 performance while specifically penalizing "cognitive offloading" and rewarding formal rigor?
    
2. **Comparative Efficacy:** Does Trace-to-Lean demonstrate a statistically significant improvement over an optimized TIR baseline (DeepSeek-Math + AutoTIR) in the target domains of combinatorics and number theory?
    
3. **Module Robustness:** Can we isolate the contributions of specific components, such as the "Coordinate Descent" steering for trace diversity or the "Invariant Miner" for premise extraction?
    
4. **Failure Topology:** What are the dominant failure modes of the autoformalization process, and do they correlate with problem complexity or specific mathematical sub-domains?
    

---

## 3. Benchmark Construction: The "Shadow AIMO" Protocol

Reliable validation requires a benchmark that mirrors the difficulty and distribution of the target competition while remaining strictly decontaminated from the model's pre-training corpus. We reject the use of standard benchmarks like GSM8K or MATH, which are saturated and lack the requisite reasoning depth. Instead, we propose the **Shadow AIMO Benchmark**.

### 3.1 Data Sourcing and Stratification

The benchmark comprises 1,200 problems stratified by domain and difficulty, sourced from three primary reservoirs:

#### 3.1.1 The AIME/USAMO Continuum (Core & Ceiling)

The American Invitational Mathematics Examination (AIME) and the US Mathematical Olympiad (USAMO) serve as the foundation.

- **AIME (Core Validation):** We select 500 problems from AIME (2010–2024). These problems require integer answers (000–999), making them compatible with standard evaluation metrics, yet they demand multi-step reasoning often exceeding 10 logical hops.
    
- **USAMO (Reasoning Ceiling):** We select 100 problems from USAMO (2015–2024). These serve as "ceiling tests" for the Autoformalizer. While the final output of AIMO 3 may be an answer, the _internal_ process for these problems requires full proof capability.
    
- **Stratification Target:** To align with AIMO trends, we enforce a distribution of **30% Algebra**, **25% Geometry**, **25% Combinatorics**, and **20% Number Theory**.
    

#### 3.1.2 The OEIS Integer Sequence Challenge (Pattern Mining)

To validate the **Invariant Miner** module, we incorporate the **UTMath** benchmark derived from the On-Line Encyclopedia of Integer Sequences (OEIS).

- **Rationale:** Combinatorial problems often reduce to identifying a recurrence relation. OEIS provides ground truth for these relations.
    
- **Selection:** We utilize a subset of 200 "hard" sequences where the term index $n$ is large ($n > 10^{18}$), preventing naive recursive computation and forcing the derivation of a closed-form solution or matrix exponentiation invariant.
    

#### 3.1.3 The "Lean Workbook" Corpus (Translation Fidelity)

For direct unit testing of the Autoformalizer, we utilize the **Lean Workbook**, a dataset of 57,000 formalized problems. This provides the "Rosetta Stone"—paired informal and formal statements—necessary to measure translation accuracy independent of problem-solving ability.

### 3.2 Problem Labeling and Taxonomy

Each problem in the Shadow Benchmark is manually annotated with metadata to enable granular failure analysis.

|**Metadata Field**|**Description**|**Taxonomy Values**|
|---|---|---|
|**Domain**|The primary mathematical field.|Algebra, Geometry, Combinatorics, Number Theory|
|**Reasoning Depth**|Complexity of the solution path.|**D1 (Procedural):** Direct formula application.<br><br>  <br><br>**D2 (Structural):** Requires auxiliary construction or invariant.<br><br>  <br><br>**D3 (Creative):** Requires novel mapping/isomorphism.|
|**Verification Type**|The most appropriate verification method.|**Calc:** Verifiable via Python calculation.<br><br>  <br><br>**Search:** Verifiable via Python brute-force ($<10^6$ states).<br><br>  <br><br>**Proof:** Requires formal logic/induction (Lean).|
|**Trap Type**|Specific cognitive traps embedded.|**Boundary:** Edge cases ($n=0, 1$).<br><br>  <br><br>**Over-counting:** Symmetry/distinctness confusion.<br><br>  <br><br>**Fake Pattern:** Pattern holds for $n<5$ then breaks.|

### 3.3 Decontamination and Adversarial Perturbation

Given the risk of training data contamination , we implement a rigorous "Sanitization Pipeline":

1. **13-gram Filtering:** We compute 13-gram overlaps between the benchmark and the OpenWebText/CommonCrawl corpora. Any problem with >50% overlap is discarded or flagged for rewriting.
    
2. **Adversarial Rewriting:** For 20% of the dataset, we employ a "Paraphraser Agent" to alter surface forms (e.g., changing "Alice tosses coins" to "A particle spins up/down") while preserving mathematical isomorphism. This ensures the model recognizes the _structure_, not the _text_.
    
3. **Parameter Injection:** We systematically vary numerical constants in the problem statements (e.g., changing "Find the sum to 100" to "Find the sum to 2026") to verify that the model is solving the general case rather than recalling a specific answer.
    

---

## 4. Experimental Design: Variables, Baselines, and Metrics

The experimental design is structured to prove superiority over the TIR baseline while characterizing the specific contribution of the Trace-to-Lean components.

### 4.1 Independent Variables

We manipulate three core variables to observe their effect on success rates:

1. **Inference Budget ($k$):** The number of reasoning paths generated per problem. We test at $k \in \{1, 16, 64, 128\}$. This measures the "Search Efficiency" of the system.
    
2. **Steering Strategy:** We compare **Vanilla Sampling** (Temperature $T=0.7$) against **Riemannian Activation Steering** (Coordinate Descent). This isolates the value of active exploration in the latent space.
    
3. **Verification Mode:**
    
    - _Mode A (TIR):_ Execution-based verification (Python).
        
    - _Mode B (Lean):_ Formal verification (Trace-to-Lean).
        
    - _Mode C (Hybrid/Router):_ Dynamic selection between Python and Lean.
        

### 4.2 The Baseline: Optimized TIR (DeepSeek-Math + AutoTIR)

To ensure the comparison is fair, the baseline must be the current state-of-the-art, not a strawman.

- **Model:** **DeepSeek-Math-7B-RL** fine-tuned on NuminaMath-TIR.
    
- **Mechanism:** **AutoTIR** , a framework where the model autonomously decides when to invoke the Python tool.
    
- **Decoding:** **Self-Consistency (SC-TIR)** with 64 paths, selecting the majority answer.
    
- **Justification:** This represents the "winning formula" of AIMO 1. Any improvement over this baseline validates the Trace-to-Lean hypothesis.
    

### 4.3 Dependent Variables (Metrics)

#### 4.3.1 Unbiased Pass@k Estimator

We report **Pass@k** using the unbiased estimator to account for finite sample variance. For a total of $n$ samples with $c$ correct:

$$\text{pass}@k = 1 - \frac{\binom{n-c}{k}}{\binom{n}{k}}$$

Standard deviation is calculated via bootstrapping (10,000 iterations).

#### 4.3.2 The "Formalization Tax" vs. "Rigor Bonus"

We introduce two custom metrics to quantify the trade-off of formalization:

- **Formalization Tax ($L_{form}$):** The percentage of problems where the model finds the correct answer via informal trace but fails to compile a valid proof. This represents the "cost" of using Lean.
    
    $$L_{form} = \frac{|Problems_{CorrectTrace} \cap Problems_{InvalidProof}|}{|Problems_{CorrectTrace}|}$$
    
- **Rigor Bonus ($G_{rigor}$):** The percentage of problems where TIR yields an incorrect answer (false positive) while Trace-to-Lean correctly rejects the path or finds the true solution. This represents the "gain" of formal verification.
    
    $$G_{rigor} = \frac{|Problems_{TIR\_Fail} \cap Problems_{Lean\_Success}|}{Total Problems}$$
    

---

## 5. Per-Module Validation Methodologies

Before end-to-end testing, each module undergoes rigorous unit testing to establish its performance envelope.

### 5.1 Module 1: Trace Generator & Activation Steering

**Objective:** Validate that Riemannian Block-Coordinate Descent (RBCD) steering increases the semantic diversity of generated traces.

- **Hypothesis:** Standard sampling collapses to the mode; RBCD forces exploration of orthogonal reasoning strategies.
    
- **Test Protocol:**
    
    1. Select 50 USAMO geometry problems (known for having multiple solution paths: synthetic, complex, coordinate).
        
    2. Generate 100 traces using Vanilla Sampling ($T=0.7$) and 100 using RBCD Steering.
        
    3. **Metric:** _Semantic Volume_. Embed traces using a math-specialized BERT model and calculate the volume of the convex hull of the embeddings.
        
    4. **Metric:** _Cluster Count_. Use DBSCAN to cluster traces. A higher number of clusters indicates discovery of distinct reasoning topologies.
        

### 5.2 Module 2: Invariant Miner

**Objective:** Validate the extraction of formal constraints from execution traces.

- **Context:** Identifying invariants (e.g., "energy is conserved," "parity flips each step") is crucial for proofs.
    
- **Test Protocol:**
    
    1. Input: The **UTMath (OEIS)** dataset.
        
    2. Task: The Miner must observe the first 10 terms of a sequence and output the recurrence relation in Lean syntax.
        
    3. **Metric:** _Invariant Precision_. The percentage of generated invariants that hold for terms $n=11 \dots 100$.
        
    4. **Metric:** _Constraint Strength_. We measure if the extracted invariant is "strong" (uniquely defines the sequence) or "weak" (e.g., "all terms are positive").
        

### 5.3 Module 3: Autoformalizer

**Objective:** Measure the fidelity of translating natural language to Lean 4.

- **Test Protocol:**
    
    1. Input: 500 ground-truth informal proofs from the **Lean Workbook**.
        
    2. Task: Generate the corresponding formal proof.
        
    3. **Metric:** _Compilation Rate_. The percentage of generated code that compiles.
        
    4. **Metric:** _Proof Adherence_. We use a "back-translation" technique: translate the generated Lean back to English and measure semantic overlap (BERTScore) with the original text. This detects "goal drift" (proving a different, easier theorem).
        

### 5.4 Module 4: The Router (Meta-Controller)

**Objective:** Validate the decision logic for switching between Python (TIR) and Lean.

- **Logic:** The Router analyzes the problem text to classify it as "Computation-Heavy" or "Logic-Heavy".
    
- **Test Protocol:**
    
    1. Input: A mixed bag of 200 computational AIME problems and 200 abstract USAMO problems.
        
    2. **Metric:** _Classification Accuracy_. Does the Router correctly assign USAMO problems to the Lean pipeline and AIME calc problems to the Python pipeline?
        
    3. **Metric:** _Routing Overhead_. The latency introduced by the Router inference step.
        

---

## 6. End-to-End Evaluation and Analysis

### 6.1 The "Battle of Baselines"

We execute a full-scale comparison on the Shadow AIMO Benchmark (1,000 problems).

|**System Configuration**|**Description**|**Verification Mechanism**|
|---|---|---|
|**Baseline (TIR)**|DeepSeek-Math + AutoTIR|Python Execution + Majority Vote|
|**System A (Trace-to-Lean)**|Trace (RBCD) $\rightarrow$ Autoformalize $\rightarrow$ Lean|Lean Compilation|
|**System B (Hybrid)**|Router $\rightarrow$ {TIR $\mid$ Trace-to-Lean}|Context-Dependent|

### 6.2 Statistical Analysis of Success Rates

We anticipate the results to follow a "crossover" pattern.

- **Low Difficulty (D1/D2):** TIR is expected to outperform Trace-to-Lean due to the _Formalization Tax_. Writing a proof for $2+2=4$ is harder in Lean than Python.
    
- **High Difficulty (D3/D4):** Trace-to-Lean is expected to outperform TIR due to the _Rigor Bonus_. In combinatorics, where Python simulation often misses edge cases (e.g., $n=0$), Lean's requirement for a universal quantifier proof forces correctness.
    
- **Significance Testing:** We will use the **paired t-test** on the bootstrap samples to determine if the crossover point is statistically significant ($p < 0.05$).
    

### 6.3 Performance Profiling

We must address the "Competitive Performance" requirement for AIMO 3.

- **Metric:** _Wall-Clock Latency_. Formal verification is computationally expensive.
    
- **Analysis:** We will profile the **"Time-to-First-Proof"**. If Trace-to-Lean takes >30 minutes per problem, it violates the competition constraints (typically 4 hours for 30 problems implies ~8 mins/problem).
    
- **Optimization:** We investigate the trade-off of _parallelization_. Can running 64 Lean verifications in parallel on a cluster bring the effective latency down to competitive levels?
    

---

## 7. Failure Analysis: A Taxonomy of Formalization Errors

To drive iterative improvement, we implement a granular "Failure Mode and Effects Analysis" (FMEA).

### 7.1 Taxonomy of Reasoning Errors

|**Error Category**|**Sub-Type**|**Description**|**Remediation Strategy**|
|---|---|---|---|
|**Translation Error**|_Syntax Invalidity_|Generated code violates Lean 4 syntax (e.g., wrong brackets).|Finetune on syntactically valid Lean Workbook data.|
||_Hallucinated Lemma_|Calls a `Mathlib` function that does not exist.|Retrieval-Augmented Generation (RAG) over Mathlib index.|
|**Semantic Error**|_Goal Drift_|Proves a theorem different from the requested one.|Implement "Bi-Directional Equivalence Check" (BEq).|
||_Premise Omission_|Fails to declare necessary constraints (e.g., $n > 0$).|Use Invariant Miner to explicitly extract constraints.|
|**Logical Error**|_Tactic Failure_|A specific tactic (e.g., `simp`, `rewrite`) fails to close the goal.|Use "Re-prompting" with the compiler error message.|
|**Resource Error**|_Timeout_|The Lean kernel hangs during elaboration.|Optimize `maxHeartbeats` and tactic complexity.|

### 7.2 Domain-Specific Failure Maps

We hypothesize that failure modes are domain-correlated:

- **Algebra:** High rate of _Syntax Invalidity_ due to complex equation formatting.
    
- **Combinatorics:** High rate of _Premise Omission_ (forgetting to define set finiteness).
    
- **Geometry:** High rate of _Hallucinated Lemma_ (inventing non-existent geometric theorems).
    

---

## 8. Generalization and Robustness

### 8.1 Out-of-Distribution (OOD) Testing

We test generalization using the **PutnamBench** (undergraduate level).

- **Goal:** Determine if the system can generalize to higher mathematics.
    
- **Metric:** _Transfer Ratio_. Ratio of success on Putnam vs. AIME. A high ratio indicates deep reasoning; a low ratio indicates overfitting to high school patterns.
    

### 8.2 Robustness to Noise

We employ the "Distraction Test":

- Inject irrelevant premises into the problem statement (e.g., "The weather is sunny. Find $x$ such that...").
    
- **TIR Baseline:** Often distracted, attempts to use the irrelevant data in Python code.
    
- **Trace-to-Lean:** Should ideally ignore the noise, as the formal proof construction will not find a use for the irrelevant premise in the logical chain.
    

---

## 9. Conclusion

This research report outlines a scientifically rigorous methodology for validating the Trace-to-Lean architecture. By replacing the stochastic "guessing" of TIR with the formal "proving" of Lean, we aim to elevate AI mathematical reasoning from imitation to understanding. The "Shadow AIMO" benchmark, combined with the unbiased statistical protocols and granular failure taxonomy, provides the necessary apparatus to measure this transition.

The successful execution of this validation plan will not only confirm the readiness of Trace-to-Lean for AIMO 3 but also establish a new standard for neuro-symbolic evaluation. If the data confirms that the _Rigor Bonus_ in combinatorics outweighs the _Formalization Tax_ in algebra, Trace-to-Lean will represent a decisive step forward in the quest for artificial mathematical intelligence. The path to a Gold Medal lies not in computing more numbers, but in proving more truths.