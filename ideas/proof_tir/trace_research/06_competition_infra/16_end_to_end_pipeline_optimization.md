# Deep Research: End-to-End Pipeline Optimization — Latency, Throughput, and Resource Management

## Research Objective

The complete Trace-to-Lean pipeline involves multiple stages: problem routing, trace generation, pattern mining, formula translation, Lean verification, and error recovery. We need to optimize the end-to-end performance for competition constraints: 50 problems in 5 hours = 6 minutes per problem average.

## Context

Pipeline stages:
1. **Route**: Classify problem → select module (0.5s LLM call)
2. **Generate Trace**: LLM writes Python → execute (5-30s)
3. **Mine Pattern**: Berlekamp-Massey / Lagrange / etc. (0.01-1s)
4. **Translate**: Python formula → Lean syntax (0.1-5s)
5. **Verify**: Lean native_decide (0.1-10s)
6. **Retry/Fallback**: If verification fails (variable)

Total budget: 360 seconds per problem, but with variance.

## Research Questions

### Part A: Latency Analysis

#### 1. Stage-by-Stage Timing
Measure actual latency for each stage:
- LLM inference for classification
- LLM inference for trace code generation
- Python execution of trace code
- Berlekamp-Massey on typical sequences
- Lagrange interpolation
- LLM inference for formula translation (if needed)
- Lean compilation and native_decide
- Error parsing and retry coordination

#### 2. Critical Path Analysis
Which stages are on the critical path?
- LLM inference is typically slowest
- Can anything be parallelized?
- What's the minimum possible latency?

#### 3. Variance Analysis
Latency variance by problem type:
- Simple counting: fast trace, fast mine, fast verify
- Complex recurrence: slower trace, normal mine, normal verify
- Edge cases: may require multiple retries

### Part B: Resource Utilization

#### 4. GPU Utilization
- LLM inference uses GPU
- Lean verification uses CPU
- Can we run LLM and Lean in parallel on different resources?

#### 5. Memory Management
- LLM model size (e.g., 14B model = ~28GB with 16-bit)
- Lean memory usage during verification
- Peak memory and how to avoid OOM

#### 6. CPU vs GPU Split
- LLM: GPU-bound
- Mining algorithms: CPU-bound
- Lean: CPU-bound
- Python execution: CPU-bound

Optimal resource allocation?

### Part C: Parallelization Strategies

#### 7. Pipeline Parallelism
While processing problem N:
- Lean verifying formula for problem N-1
- Mining pattern for problem N
- Generating trace for problem N+1

Can we overlap stages?

#### 8. Problem-Level Parallelism
Process multiple problems simultaneously?
- If GPU can handle multiple LLM inferences
- Lean can run multiple verification processes
- Trade-off: latency vs throughput

#### 9. Retry Parallelism
When verification fails:
- Try multiple alternative formulas in parallel
- Run different mining tiers simultaneously
- Which approach is faster?

### Part D: Caching and Reuse

#### 10. LLM Response Caching
- If same problem type seen before, reuse insights?
- Few-shot examples cached in memory
- Model weights already in GPU memory

#### 11. Lean Environment Caching
- Keep Lean process running between problems
- Pre-loaded imports and definitions
- Avoid restart overhead

#### 12. Compilation Caching
- Lean caches compiled .olean files
- Do our verification files benefit from this?
- Or is each problem unique?

### Part E: Batching Strategies

#### 13. LLM Batching
- Batch multiple LLM calls together
- Classification for several problems at once?
- Trade-off: latency (wait for batch) vs throughput

#### 14. Verification Batching
- Verify multiple formulas in one Lean file?
- Multiple theorems, each as a quick check
- Or separate files for isolation?

### Part F: Time Budget Management

#### 15. Dynamic Time Allocation
Not all problems are equal:
- Easy problems: solve in 2 minutes, bank time
- Hard problems: use banked time
- How to detect and adapt?

#### 16. Early Termination
When to give up on a problem:
- Time limit per problem (hard cutoff)
- Diminishing returns detection
- Confidence threshold

#### 17. Priority Ordering
If problems come in batches:
- Solve easy problems first (build time budget)
- Or tackle hard problems while fresh (before time pressure)
- Optimal ordering strategy?

### Part G: Failure Recovery Overhead

#### 18. Retry Cost
Each retry adds latency:
- Re-mine pattern: fast (sub-second)
- Re-translate formula: fast (LLM call)
- Re-verify in Lean: medium (1-10s)
- Regenerate trace: slow (LLM + execution)

How to minimize retry cost?

#### 19. Fallback Timing
When to trigger TIR fallback:
- After N failed verifications?
- After T seconds elapsed?
- Based on confidence scores?

#### 20. Graceful Degradation
Under time pressure:
- Skip Lean verification, use TIR directly?
- Lower verification range (n=1..50 instead of 100)?
- Accept lower confidence answers?

### Part H: Infrastructure Optimization

#### 21. LLM Inference Optimization
- vLLM for efficient batching
- Continuous batching
- Speculative decoding
- Quantization (FP8, INT4)

#### 22. Python Execution Sandboxing
- Secure execution environment
- Timeout enforcement
- Resource limits

#### 23. Lean Process Management
- Persistent Lean REPL vs spawn-per-problem
- IPC overhead (stdin/stdout vs files)
- Error stream handling

### Part I: Monitoring and Observability

#### 24. Real-Time Metrics
Track during competition:
- Problems solved / attempted
- Average time per problem
- Success rate by method
- Remaining time budget

#### 25. Logging Strategy
What to log:
- Every LLM call (input/output)
- Every execution result
- Every Lean verification (code/result)
- Timing for each stage

#### 26. Post-Competition Analysis
Use logs to:
- Identify bottlenecks
- Find systematic failures
- Improve for next competition

### Part J: Stress Testing

#### 27. Load Testing
Simulate competition conditions:
- 50 problems, 5 hours
- Measure actual throughput
- Identify breaking points

#### 28. Edge Case Testing
- Very complex problems (long traces)
- Very simple problems (immediate solve)
- Adversarial problems (designed to break pipeline)

#### 29. Recovery Testing
- Simulate LLM failures
- Simulate Lean crashes
- Simulate timeout scenarios
- Verify graceful handling

### Part K: Configuration Tuning

#### 30. Hyperparameter Selection
Key parameters to tune:
- Trace length (n=15? 20? 30?)
- Verification range (n=1..50? 100? 200?)
- Retry limits per phase
- Time allocation per stage
- Confidence thresholds

#### 31. Problem-Adaptive Configuration
Different settings for different problem types:
- Combinatorics: longer trace, B-M focus
- Number theory: shorter trace, modular focus
- Fallback: aggressive TIR

#### 32. A/B Testing
Compare configurations:
- Run same problems with different settings
- Measure: accuracy, time, confidence
- Select optimal configuration

## Desired Output Format

1. **Latency Breakdown**: Detailed timing for each stage
2. **Resource Utilization Model**: CPU/GPU/Memory usage
3. **Parallelization Strategy**: What can run in parallel
4. **Time Budget Framework**: Allocation and management
5. **Caching Strategy**: What to cache and how
6. **Monitoring Dashboard**: Key metrics to track
7. **Configuration Recommendations**: Optimal hyperparameters
8. **Stress Test Report**: Behavior under load











# Deep Research: End-to-End Pipeline Optimization — Latency, Throughput, and Resource Management

## Executive Summary

The optimization of neuro-symbolic pipelines, specifically the "Trace-to-Lean" architecture, represents a critical challenge in modern artificial intelligence, particularly within the constraints of high-stakes mathematical reasoning competitions like the AI Mathematical Olympiad (AIMO). The objective is to solve 50 complex mathematical problems within a 5-hour window, yielding a strict average time budget of 360 seconds (6 minutes) per problem. This constraint demands a rigorous analysis of latency, throughput, and resource allocation across a heterogeneous computing environment involving GPU-bound Large Language Models (LLMs) and CPU-bound symbolic verification engines (Lean 4, Python).

This report provides an exhaustive technical analysis of the end-to-end pipeline, decomposing the Trace-to-Lean workflow into its constituent atomic operations. We examine the latency characteristics of state-of-the-art inference engines (vLLM), the compilation and execution overheads of the Lean 4 theorem prover, and the algorithmic complexities of pattern mining techniques like Berlekamp-Massey. Furthermore, we propose advanced architectural strategies, including speculative decoding, persistent REPL servers with LRU caching, and dynamic time banking algorithms, to maximize throughput without compromising verification integrity. The analysis draws upon recent benchmarks from the NuminaMath project, Kimina Lean Server architecture, and vLLM performance studies to formulate a reference architecture capable of meeting the 6-minute/problem target.

The central thesis of this optimization strategy relies on shifting from a serial, problem-by-problem processing model to a highly asynchronous, resource-saturated pipeline. By decoupling the high-latency generation phase (dominated by GPU memory bandwidth) from the high-variance verification phase (dominated by CPU logic), we can achieve a throughput that effectively masks the individual latency of difficult problems. This requires a sophisticated orchestration layer capable of managing "Time Banks," assessing "Problem Difficulty" in real-time, and dynamically reallocating compute resources between speculative exploration (trace generation) and rigorous confirmation (Lean verification). The following sections detail the theoretical underpinnings, empirical benchmarks, and architectural specifications required to build this system.

## 1. Latency Analysis: The Physics of Neuro-Symbolic Compute

The Trace-to-Lean pipeline operates as a directed acyclic graph (DAG) of computational tasks, each with distinct hardware affinities and latency characteristics. To optimize the aggregate throughput, one must first dissect the atomic latency of each stage, identifying the physical and logical bottlenecks that constrain performance.

### 1.1 Micro-Latency of Large Language Model Inference

The first and most resource-intensive stage of the pipeline involves the Large Language Model (LLM), which is responsible for routing, trace generation, and formal translation. Understanding the latency here requires analyzing the mechanics of Transformer inference. LLM generation is inherently auto-regressive; the model must generate tokens sequentially, with each new token dependent on the entire history of previous tokens. This creates a linear latency dependency on the output length ($O(N)$), which is the primary contributor to the "Generate Trace" latency of 5-30 seconds.

The latency of a single decoding step is governed largely by memory bandwidth rather than compute capacity. In the context of a 14B parameter model (e.g., Qwen2.5-14B or DeepSeek-R1-Distill) running on high-end hardware like an NVIDIA A100 or H100, the GPU must load the entire model weights from High Bandwidth Memory (HBM) into the Static Random Access Memory (SRAM) of the streaming multiprocessors for every token generated, unless the batch size is sufficiently large to amortize this cost.

Recent benchmarks using the vLLM inference engine reveal that while single-stream latency is bounded by the memory wall, aggregate throughput can be significantly scaled. On an NVIDIA A40 GPU, vLLM achieves a decoding speed of approximately 47 tokens per second for a single request on a 14B model. However, the time-to-first-token (TTFT)—the latency of the prefill phase where the prompt is processed—remains a critical bottleneck for the "Routing" and "Translation" stages, which involve short generations but potentially long contexts. The prefill phase is compute-bound (quadratic attention complexity), whereas the decoding phase is memory-bound. This dichotomy necessitates different optimization strategies for the "Route" stage (latency-sensitive, short output) versus the "Generate Trace" stage (throughput-sensitive, long output).

The use of quantization further complicates this landscape. While 4-bit quantization (e.g., AWQ) significantly reduces the memory footprint—allowing a 14B model to reside in approximately 8-10 GB of VRAM—it shifts the bottleneck slightly back toward compute, as the weights must be dequantized on the fly. However, empirical data suggests that the reduction in memory traffic outweighs the dequantization overhead, resulting in a net speedup of 2.78x for 4-bit models compared to unquantized baselines on consumer hardware. This suggests that for the latency-critical generation phase, heavily quantized models are not just a capacity enabler but a latency optimization.

### 1.2 The Python Execution Sandbox: Latency vs. Security

Following trace generation, the pipeline executes the generated Python code to produce numerical artifacts for pattern mining. This stage introduces a classic trade-off between security and latency. The standard approach of using Docker containers for isolation introduces a startup penalty of 500ms to over 1 second per container. In a pipeline that might execute dozens of candidate traces per problem, a 1-second overhead is prohibitive, potentially consuming 20-30% of the allocated budget for simple problems.

Alternative sandboxing technologies like `nsjail` (namespaces jail) or `gVisor` offer a far superior latency profile. `nsjail` leverages Linux kernel namespaces directly to isolate processes without the heavy filesystem layering of Docker, achieving startup times in the range of 10-50 milliseconds. This two-order-of-magnitude improvement transforms the execution stage from a potential bottleneck into a negligible micro-task.

However, the "execution" latency is not just about startup time; it is also about runtime behavior. LLM-generated code frequently contains infinite loops or inefficient algorithms (e.g., recursive Fibonacci without memoization). A robust pipeline must enforce strict timeouts. The operating system's scheduler resolution and signal handling introduce a "jitter" in timeout enforcement. If we set a 1-second timeout, the actual termination might occur at 1.05s or 1.1s depending on system load. While seemingly minor, this variance accumulates across thousands of executions in a competition. Therefore, the execution sandbox must utilize hard resource limits (cgroups for CPU time and memory) rather than soft application-level timers to ensure deterministic latency bounds.

### 1.3 The Mathematics of Pattern Mining

Once a trace is executed, the pipeline attempts to "mine" a mathematical pattern—typically a sequence or a polynomial relation—that can be formalized. The latency of this stage is purely algorithmic and runs on the CPU. The two primary algorithms employed are the Berlekamp-Massey algorithm for linear recurrences and Lagrange interpolation for polynomials.

The Berlekamp-Massey algorithm has a time complexity of $O(N^2)$, where $N$ is the length of the sequence. For the scale of problems in AIMO (typically asking for the answer modulo 1000 or for a specific integer), the sequence length $N$ rarely exceeds 100 terms. Executing an $O(N^2)$ algorithm on $N=100$ involves roughly 10,000 operations, which modern CPUs can process in microseconds. Consequently, the latency of the mining stage is effectively zero in the broader context of the pipeline, provided the data transfer overhead between the Python execution environment and the mining logic is minimized.

A potential latency trap exists, however, in "blind mining." If the pipeline attempts to mine complex non-linear patterns or utilizes symbolic regression (which can be an expensive search process), this stage can balloon to seconds. To maintain the 6-minute cadence, the mining stage must be restricted to deterministic, polynomial-time algorithms. If Berlekamp-Massey fails to find a generator within a sub-second threshold, the pipeline should treat the sequence as "complex" and revert to LLM-based reasoning rather than burning CPU cycles on expensive symbolic regression.

### 1.4 Lean 4 Compilation and Verification Overhead

The Verification stage is the most rigorous and theoretically complex component of the pipeline. Lean 4 is not an interpreted language in the traditional sense; it is a compiled language with a sophisticated elaboration process. The distinction between `native_decide` and kernel reduction is pivotal for latency analysis.

Kernel reduction involves the Lean kernel symbolically reducing a term to a boolean `true`. This is the "gold standard" of verification but is computationally excruciating for large numbers. Calculating `1000! % 997` via kernel reduction involves expanding the factorial definition step-by-step, which creates an explosion of terms in memory. This often leads to timeouts or memory exhaustion, effectively infinite latency for practical purposes.

`native_decide`, introduced to mitigate this, compiles the Lean definition into C code, invokes a C compiler (like Clang), links it, and executes the binary. While this makes the _execution_ of the computation fast, it introduces a massive _compilation_ latency. Benchmarks indicate that the overhead of spawning the compiler, linking standard libraries, and initiating the process can take 5-10 seconds per invocation. For a pipeline aiming to check a hypothesis in sub-second time, a 10-second compilation penalty is a showstopper.

Furthermore, the initialization of the Lean environment itself—loading `Mathlib` and other dependencies—is a heavy operation. A cold start of a Lean process that imports `Mathlib` can take 10-20 seconds to resolve imports and rebuild the environment state, even with pre-compiled `.olean` files. This "import latency" is the single largest non-LLM bottleneck in the pipeline. It necessitates an architectural shift from "one process per problem" to a "persistent server" model, where the environment is loaded once and reused, reducing the marginal latency of verification to just the elaboration and checking time (often <500ms).

### 1.5 Critical Path Synthesis

Synthesizing these observations, the critical path for a single problem iteration is defined by the sequence: **LLM Generation $\rightarrow$ Lean Verification**. While Python execution and Pattern Mining are technically sequential steps, their latencies (milliseconds) are negligible compared to the LLM (seconds) and Lean (seconds).

The minimum theoretical latency for a single pass—assuming a 500-token trace generated at 50 tokens/s (10s) and a persistent Lean server verification (0.5s)—is approximately **10.5 seconds**. Given the 360-second budget, this theoretical floor allows for approximately 34 serial iterations. However, variance in generation length and the probabilistic nature of LLM correctness means we essentially play a game of probability. Optimizing the critical path therefore requires two parallel efforts: maximizing LLM throughput to generate more candidates per second, and minimizing Lean latency to verify those candidates faster than they are generated.

## 2. Resource Utilization Model

Optimizing the pipeline requires a detailed mapping of software tasks to hardware resources. The workload is heterogeneous, splitting strictly between GPU-bound tasks (inference) and CPU-bound tasks (execution, mining, verification).

### 2.1 GPU Memory Hierarchies and Utilization

The GPU is the most scarce and expensive resource. Its utilization is dominated by the LLM's memory footprint. For a 14B parameter model in FP16, the model weights alone consume ~28GB of VRAM. With the KV cache growing linearly with batch size and sequence length, a 48GB card (like an A40) or an 80GB card (A100) can quickly become saturated.

The memory hierarchy of the GPU dictates performance. The weights reside in High Bandwidth Memory (HBM). During inference, the bottleneck is moving these weights to the compute units. To maximize utilization, we must maximize the **arithmetic intensity**—the ratio of compute operations to memory access. This is achieved by increasing the batch size. If we process 1 problem, we load 28GB of weights to generate 1 token. If we process 50 problems, we load 28GB of weights to generate 50 tokens, effectively increasing efficiency by 50x.

However, the KV cache competes for HBM. A context window of 4096 tokens for a 14B model can consume gigabytes of memory per request. vLLM's PagedAttention mitigates fragmentation, but the hard limit remains. If the KV cache grows too large (e.g., deeply exploring 50 problems simultaneously), the system must swap blocks to CPU RAM, causing a catastrophic drop in performance. Therefore, the resource model must cap the number of concurrent "active" generations based on available VRAM minus the fixed weight buffer.

### 2.2 CPU Threading and Lean Processes

The CPU's role is primarily to host the Lean verification workers. Lean 4's elaboration is generally single-threaded per file. To utilize a 64-core CPU effectively, we must run multiple independent Lean workers.

However, memory bandwidth on the CPU side is also a constraint. Each Lean worker, when loaded with `Mathlib`, has a resident set size (RSS) of 2-4 GB. Launching 64 workers would require 256GB of system RAM. If the physical machine has only 128GB, the OS will begin swapping to disk, and verification latency will skyrocket due to page faults. The resource model for the CPU must therefore be **memory-constrained** rather than core-constrained. We must allocate the maximum number of workers $W$ such that $W \times \text{Memory}_{worker} < \text{Total RAM} - \text{System Reserve}$.

### 2.3 Optimal Resource Allocation Strategy

Given these constraints, the optimal allocation is a split architecture:

- **GPU:** Dedicated entirely to the LLM. No other logic should touch the GPU. The vLLM scheduler should be configured to use ~90% of VRAM, leaving a safety buffer.
    
- **CPU - High Priority:** A small pool of cores (4-8) reserved for the OS, the vLLM scheduler (which needs CPU for orchestrating the GPU), and the Pipeline Controller (routing logic).
    
- **CPU - Low Priority:** The remaining cores and RAM dedicated to the Lean Worker Pool. These workers should be pinned to specific cores to maximize L3 cache hits, but the total count must be throttled by available RAM.
    

This separation ensures that a heavy verification load (CPU spike) does not starve the GPU scheduler, which would cause the expensive GPU to idle.

## 3. Parallelization Strategies

To achieve the throughput of 50 problems in 5 hours, serial execution is insufficient. We must implement parallelism at the pipeline level, the problem level, and the verification level.

### 3.1 Pipeline Parallelism: Asynchronous Decoupling

The pipeline stages (Generate $\rightarrow$ Execute $\rightarrow$ Mine $\rightarrow$ Verify) form a classic producer-consumer model. The LLM is the **Producer**, generating candidate traces and formulas. The Lean workers are the **Consumers**, verifying them.

These two distinct resource pools (GPU vs. CPU) allow for perfect pipeline parallelism. While the GPU is generating the trace for Problem $N+1$, the CPU can be verifying the proof for Problem $N$. Implementing this requires an asynchronous control plane (e.g., using Python's `asyncio` or Rust's `tokio`). The controller submits a prompt to the vLLM API and immediately yields control. When the response returns (a "future" completes), it schedules the Python execution task. This non-blocking architecture ensures that the GPU never waits for the CPU, and the CPU never waits for the GPU, maximizing the utilization of both.

### 3.2 Problem-Level Parallelism: Batching vs. Streaming

There are two competing approaches to handling the 50 problems:

1. **Macro-Batching:** Load all 50 problems into a single vLLM batch.
    
    - _Pros:_ Maximize GPU throughput.
        
    - _Cons:_ High latency for individual problems. One "long-tail" problem that requires 2000 tokens will hold up the completion of the batch for 49 other problems that only needed 500 tokens. This creates "head-of-line blocking."
        
2. **Streaming (Continuous Batching):** vLLM's native mode. Problems are added to the batch as they arrive. When one finishes, it is evicted, and a new request takes its place.
    

The optimal strategy for AIMO is a **Continuous Batching** approach initiated with a "Micro-batch." We start by submitting perhaps 10-20 problems. As the "Easy" ones finish and exit the GPU, we perform immediate analysis. If they are solved, we are done. If not, we re-submit them (Retry) or submit new fresh problems from the unstarted pool. This keeps the GPU batch size in the efficient range (high arithmetic intensity) while avoiding the latency penalties of massive static batches.

### 3.3 Speculative Parallelism: The "Shotgun" Approach

In reasoning tasks, the LLM's first guess is not always correct. The "Shotgun" strategy involves generating $K$ independent traces for the same problem in parallel.

- **Generation:** Request 4 parallel completions from vLLM (using `n=4`). This increases the compute load but barely impacts memory bandwidth since the prompt is shared (Prefix Caching).
    
- **Verification:** All 4 traces are processed. If they yield different formulas, we have multiple hypotheses.
    
- **Race to Verify:** We submit all unique formulas to the Lean verification queue. The first one to return `True` wins. The others are cancelled.
    

This trades compute resources (processing 4x traces) for latency reduction (higher probability of finding the solution in the first pass). Given the "6 minutes average" budget, spending 4x compute for 1 minute to save 10 minutes of serial retries is a highly favorable exchange.

## 4. Time Budget Framework

A static budget of 360 seconds per problem is naive because problem difficulty is heavy-tailed. A rigid limit would cause the system to fail "Hard" problems just as it was close to a solution, while wasting time sitting idle after solving "Easy" problems quickly. We require a **Dynamic Time Allocation (DTA)** framework.

### 4.1 Statistical Modelling of Difficulty

We can categorize problems into three difficulty tiers based on early signals:

1. **Tier 1 (Trivial):** Solved by a single Python trace or standard library call (e.g., `scipy.optimize`). Expected time: 30s.
    
2. **Tier 2 (Standard):** Requires iterative reasoning and pattern mining. Expected time: 180s.
    
3. **Tier 3 (Complex):** Requires novel theorem proving or deep search. Expected time: >600s.
    

The "Route" stage LLM can predict this tier. Alternatively, we can infer it dynamically: if the first 3 traces fail to execute, the problem is likely Tier 2 or 3.

### 4.2 The "Time Bank" Algorithm

The central mechanic of the DTA framework is the **Time Bank**.

- **Initialization:** The bank starts at 0 seconds.
    
- **Allocation:** Each problem gets a _Base Budget_ ($T_{base} = 120s$).
    
- **Banking:** If a Tier 1 problem is solved in 30s, the remaining $90s$ is deposited into the Time Bank.
    
- **Withdrawal:** When a Tier 3 problem exceeds its Base Budget, it requests an "Overdraft" from the Time Bank.
    
- **Priority:** The system should aggressively prioritize solving Tier 1 problems first to capitalize the bank. This suggests a sorting strategy where the first 10 minutes of the competition are spent scanning _all_ problems and executing the easiest ones.
    

### 4.3 Early Termination Mathematics

To prevent "Zombie" processes from draining the bank, we need a mathematical termination criterion. We can model the probability of solving a problem at time $t$ given that it hasn't been solved yet as a hazard function $h(t)$. For LLM reasoning, this function often decreases over time—if you haven't solved it in 10 attempts, the 11th attempt is unlikely to succeed unless the strategy changes.

We define a **Cutoff Threshold** $C$. If $T_{elapsed} > T_{base} + \alpha \times T_{bank}$, terminate. The parameter $\alpha$ (e.g., 0.5) ensures that no single hard problem can drain the entire bank, preserving resources for other potential Tier 2 problems. Furthermore, if the "Self-Consistency" score (agreement among generated answers) remains 0 (all answers different) after 5 minutes, the problem is likely hallucinated, and we should terminate early to save the bank.

## 5. Caching Strategy

Caching is the most effective way to trade memory for latency. In this pipeline, caching must be implemented at three distinct levels: the LLM, the Lean Server, and the Compiler.

### 5.1 LLM Response and Prefix Caching

vLLM supports **Automatic Prefix Caching (APC)**. In a competition, the system prompt (e.g., "You are an expert mathematician...") and the few-shot examples (demonstrating how to write Lean traces) are identical for every problem.

- **Mechanism:** vLLM caches the KV cache blocks associated with this common prefix in GPU memory.
    
- **Impact:** When a new problem prompt is appended to this prefix, the GPU does not need to recompute the attention matrices for the prefix. This reduces the Time-To-First-Token (TTFT) significantly, potentially by 20-40% depending on the ratio of prefix length to problem length.
    
- **Implementation:** Ensure that the "System Prompt" + "Few Shot Examples" block is immutable and strictly identical across all requests to maximize cache hits.
    

### 5.2 Lean Environment Caching (The Kimina Architecture)

The **Kimina Lean Server** relies on the immutability of imports.

- **The Header:** A Lean file starts with `import Mathlib...`. This defines the environment state.
    
- **LRU Strategy:** The server maintains a dictionary mapping `hash(header)` to a `LiveProcess`. When a request arrives, we hash the imports. If a process exists, we use it.
    
- **Optimization:** The standard `Mathlib` import is massive. We can optimize by creating a custom `Competition.lean` prelude that imports only the most frequently used tactics and theorems (e.g., `Data.Nat.Prime`, `Tactic.Linarith`), rather than the entire library. This creates a "lighter" cached state. However, given the unpredictability of problems, full `Mathlib` support is usually safer. The LRU cache ensures that once `Mathlib` is loaded (taking ~15s), it stays loaded for the duration of the competition.
    

### 5.3 Compilation Artifact Caching

Lean's build system (`lake`) caches compiled modules (`.olean` files) and their C-compiled counterparts (`.o` files).

- **Pre-Competition Setup:** It is imperative to run `lake build` or `lake exe cache get` _before_ the timer starts. This ensures that the standard library logic is already compiled to machine code.
    
- **Runtime Caching:** When `native_decide` compiles a user-generated function, it creates a temporary binary. If the LLM generates the _same_ helper function for multiple traces (e.g., a primality test), standard Lean does _not_ cache this dynamic binary. A sophisticated optimization is to memoize these binary artifacts at the pipeline controller level: if the hash of the function definition matches a previous one, reuse the boolean result from the previous verification instead of asking Lean to recompile.
    

## 6. Monitoring and Observability Dashboard

Running a 50-problem parallel pipeline blindly is a recipe for failure. Real-time observability is required to make tactical decisions (e.g., "Stop attempting Problem 12, it's stuck").

### 6.1 Key Metrics and Logging Strategy

The monitoring system should track the following high-priority metrics:

|**Metric**|**Definition**|**Actionable Insight**|
|---|---|---|
|**GPU Utilization**|% of Compute/Memory used|If <80%, increase batch size. If >98%, throttle new requests.|
|**Queue Depth**|# of pending Verification tasks|If high, CPU is the bottleneck. Stop generating traces.|
|**Cache Hit Rate**|% of Lean requests finding a warm worker|If low, prompts are too diverse. Standardize imports.|
|**TTFT / TPOT**|Time to First Token / Per Output Token|Diagnoses vLLM health. High TTFT = Prefill bottleneck.|
|**Success Rate**|% of traces that verify as True|If <5%, the LLM is hallucinating or logic is too complex.|
|**Time Bank**|Available surplus seconds|Drives the dynamic timeout logic for hard problems.|

### 6.2 Visualization

A centralized dashboard (Grafana or a rich TUI) should display a "status grid" of 50 cells, representing the 50 problems.

- **Color Coding:** Grey (Pending), Yellow (Generating), Blue (Verifying), Green (Solved), Red (Failed/Given Up).
    
- **Progress Bars:** Show the time elapsed vs. budget for active problems.
    
- **Alerts:** Flash warnings if the Lean Worker Pool crashes or if GPU OOM errors occur.
    

### 6.3 Post-Competition Analysis

Logs must be structured (JSON-L) to enable replay.

- **Trace Analysis:** Save every generated trace. This data is invaluable for fine-tuning the model later (Reinforcement Learning from verifiable feedback).
    
- **Failure Taxonomy:** Categorize failures: "Syntax Error", "Timeout", "Lean Tactic Fail", "Wrong Answer". This helps identify if the bottleneck is the model's coding ability or the verifier's strictness.
    

## 7. Configuration Recommendations

Based on the analysis, the following configuration represents the optimal starting point for the AIMO challenge.

### 7.1 Hyperparameters

- **Trace Samples ($n$):** Start with $n=4$ (Shotgun). If the Time Bank is low, reduce to $n=1$.
    
- **Temperature:** $0.7$ for the first attempt to encourage diversity. If retrying, increase to $0.9$.
    
- **Verification Timeout:** Strict 10s limit for `native_decide` compilation. If it takes longer, the proof is likely inefficient.
    
- **Max Retries:** 3 per problem.
    
- **Time Budget:** $T_{base}=120s$, $\alpha=0.5$.
    

### 7.2 Problem-Adaptive Settings

- **Type: Combinatorics/Number Theory:** These often require mining sequences.
    
    - _Config:_ Enable "Trace Execution" and "Mining". Set trace length higher (allow more python steps).
        
- **Type: Geometry:** Often requires symbolic manipulation.
    
    - _Config:_ Prioritize Lean tactic generation (`linarith`, `ring`) over Python execution. Use a library like `LeanGeo` if available.
        
- **Type: Algebra:**
    
    - _Config:_ High priority. These are often the "Easy" problems. Use aggressive caching.
        

## 8. Stress Test Report: Behavior Under Load

To validate the architecture, a stress test protocol is essential. This involves simulating the full 5-hour load in a compressed timeframe or with synthetic heavy problems.

### 8.1 Load Testing Methodology

- **Synthetic Workload:** Create a dataset of 50 problems where we control the difficulty (e.g., "Count to $N$").
    
- **Saturation Test:** Submit all 50 problems simultaneously.
    
    - _Observation:_ Observe the vLLM batch scheduler. Does it gracefully handle the queue?
        
    - _Observation:_ Monitor system RAM. Do the 50 Lean workers cause swapping?
        
- **Failure Injection:** Randomly kill Lean workers. Randomly disconnect the GPU.
    
    - _Goal:_ Verify that the pipeline controller detects the failure, restarts the worker, and re-queues the task without crashing the entire run.
        

### 8.2 Expected Breaking Points

- **Memory Wall:** The most likely failure mode is System RAM exhaustion due to too many concurrent Lean environments. _Mitigation:_ strict semaphore limiting the number of Lean workers.
    
- **Context Window Overflow:** A problem that generates a massive infinite loop trace might fill the GPU KV cache. _Mitigation:_ Strict token limits (max_new_tokens=1024) in the vLLM request.
    

## Conclusion

Optimizing the Trace-to-Lean pipeline for a 6-minute-per-problem constraint requires a holistic engineering approach. It is not enough to simply "use a faster model." We must optimize the **physics** of the pipeline: minimizing memory movement on the GPU via vLLM and quantization, minimizing startup latency on the CPU via the persistent Kimina server, and managing the probabilistic nature of the workload via a Dynamic Time Banking scheduler. By implementing these strategies—specifically the asynchronous producer-consumer loop and the tiered difficulty management—the system can achieve the necessary throughput to solve 50 problems in 5 hours, transforming a disjointed set of tools into a cohesive, high-performance reasoning engine.