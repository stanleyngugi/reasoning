# Deep Research Prompt: Mathematical Reasoning with RL

## Context

> You are conducting research to inform a curriculum on Reinforcement Learning for LLM Reasoning. This curriculum emphasizes **depth over breadth**, **technical verification over theoretical claims**, and **what actually works in practice**.

This prompt focuses on **mathematical reasoning**—the application of RL to improve LLMs' ability to solve math problems. This is the most developed domain for reasoning RL, with competition results providing hard evidence.

---

## Your Research Goal

> Develop a deep understanding of what works for mathematical reasoning. The goal is to answer: **"What techniques actually improve math reasoning in LLMs, and what evidence do we have?"**

---

## Core Questions to Investigate

### 1. The State of Math Reasoning

*   What's the current state of LLM math reasoning?
*   What problems can frontier models solve? (IMO, Putnam, AIME, etc.)
*   What's the gap between current capabilities and human experts?
*   What types of math problems are hardest? (Geometry, combinatorics, etc.)

### 2. Competition Results as Evidence

Competitions provide hard ground truth. Investigate:

*   **AIMO (AI Math Olympiad)**: What approaches won? What techniques were used?
*   **IMO (International Mathematical Olympiad)**: What has AlphaProof/AlphaGeometry achieved?
*   **Putnam**: Any LLM results?
*   What do winning solutions tell us about what works?

### 3. Math-Specific Training Approaches

*   How did **DeepSeek R1** achieve its math performance?
*   What is **DeepSeekMath** and how was it trained?
*   What role does RL play specifically for math?
*   What reward signals work for math? (Correctness, step validity, etc.)
*   How important is high-quality math training data?

### 4. Formal Verification Integration

*   What is the role of formal proof assistants (Lean 4, Coq)?
*   How does **AlphaProof** integrate with Lean?
*   Can formal verification improve training (verified reasoning traces)?
*   What's the gap between informal and formal mathematical reasoning?

### 5. Tool-Integrated Mathematical Reasoning (TIR)

*   What is TIR and how does it work for math?
*   How do LLMs use Python/SymPy for calculation?
*   When is tool use necessary vs optional for math?
*   How do you train models to use tools effectively?

### 6. Test-Time Techniques for Math

*   What inference strategies work for math?
*   Best-of-N with verification?
*   Solution search with backtracking?
*   How much compute helps on hard problems?

### 7. What's Unique About Math

*   How does math reasoning differ from general reasoning?
*   Why is math a good testbed for reasoning RL?
*   What transfers from math to other domains?
*   What doesn't transfer?

---

## Evidence Standards

Math has the advantage of hard evaluation:

**Strong evidence (prioritize):**

1.  Competition results (AIMO, IMO, Putnam)
2.  Benchmark scores with proper contamination controls
3.  What frontier labs actually deploy
4.  Ablation studies on math-specific techniques

**Weak evidence (note with skepticism):**

*   "Improves on GSM8K" without contamination analysis
*   Results on easy problems only
*   Claims without code/reproduction

---

## What to Avoid

*   Don't focused only on easy benchmarks (GSM8K is somewhat saturated)
*   Don't ignore the role of data quality
*   Don't conflate arithmetic with mathematical reasoning

---

## Room for Discovery

While investigating the above questions, remain open to:

*   Novel techniques specific to mathematical reasoning
*   Surprising findings about what doesn't transfer from general RL
*   Emerging approaches (neurosymbolic, formal-informal bridges)
*   Fundamental limits of current approaches
*   What competitions reveal about model weaknesses

---

## Output Structure

Organize your findings as follows:

1.  **Executive Summary**: Key takeaways for practitioners
2.  **State of the Art**: Current capabilities and remaining gaps
3.  **Competition Analysis**: What winning approaches used
4.  **Training Techniques**: Math-specific RL and fine-tuning
5.  **Formal Verification**: Integration with proof assistants
6.  **Tool Use**: Calculators, SymPy, code execution
7.  **Test-Time Strategies**: Inference techniques for math
8.  **What's Unique About Math**: Domain-specific insights
9.  **Novel Findings**: Things you discovered beyond the core questions
10. **Open Questions**: What remains unclear?
11. **Sources**: Key papers, competition reports, and resources

---

## Final Note

> Mathematical reasoning is the most developed domain for reasoning RL, with competition results providing unusually clean evidence. Focus on what those competitions reveal about what actually works. This is arguably the best place to learn what RL for reasoning can achieve.

---

# Mathematical Reasoning with Reinforcement Learning: A Comprehensive Technical Analysis

## 1. Executive Summary

The pursuit of artificial general intelligence (AGI) has found its most rigorous proving ground in the domain of mathematical reasoning. Unlike natural language tasks, where evaluation often relies on subjective proxies for quality, mathematics offers a binary and verifiable ground truth. This characteristic—**verifiability**—has catalyzed a paradigm shift in how Large Language Models (LLMs) are trained. We are witnessing a transition from the era of **Supervised Fine-Tuning (SFT)**, which relies on imitating human data, to the era of **Reinforcement Learning (RL)**, where models generate their own training signals through exploration and self-correction.

This report synthesizes findings from the frontier of AI research to inform a curriculum on RL for mathematical reasoning. The analysis is grounded in "hard" evidence from high-stakes competitions—specifically the **International Mathematical Olympiad (IMO)**, the **AI Math Olympiad (AIMO)**, and the **William Lowell Putnam Mathematical Competition**. The results from 2024 and 2025 indicate that frontier models have successfully breached the expert threshold, with systems like **DeepSeekMath-V2** achieving near-perfect scores on undergraduate-level exams 1 and **AlphaProof** reaching silver-medal standards at the IMO. 2

Three distinct architectural paradigms have emerged as the drivers of this progress:

1.  **Pure Reinforcement Learning (DeepSeek-R1):** Utilizing large-scale RL on base models to induce emergent reasoning behaviors, such as self-verification and backtracking, without relying on massive supervised datasets.
2.  **Neuro-Symbolic Formal Verification (AlphaProof/AlphaGeometry):** Integrating LLMs with formal proof assistants (like Lean) to ensure absolute logical correctness, leveraging the "infinite" synthetic data available through formal proof search.
3.  **Tool-Integrated Reasoning (NuminaMath):** Augmenting natural language reasoning with executable code (Python) to offload calculation and algorithmic verification, a technique that dominated the AIMO.

The central insight for practitioners is that **depth of thought**—the ability to allocate test-time compute to search and verification—is now more critical than breadth of knowledge. The "scaling laws" of inference suggest that a smaller model with an effective search strategy (like Monte Carlo Tree Search or Best-of-N verification) can outperform significantly larger models. However, this progress is not without caveats; the reliance on synthetic data and the "translation tax" of converting informal math to formal code remain significant bottlenecks.

---

## 2. State of the Art: Capabilities and Gaps

The evaluation of mathematical reasoning has shifted dramatically in the past 24 months. Traditional benchmarks that once served as the gold standard are now considered saturated, necessitating a move towards competition-level problems that test novel reasoning rather than pattern matching.

### 2.1 The Saturation of Elementary Benchmarks

For years, the **GSM8K** (Grade School Math) dataset was the primary barometer for reasoning. Today, it is largely obsolete for frontier research. Models like **GPT-4o**, **Claude 3.5 Sonnet**, and **DeepSeek-V3** achieve scores exceeding **95%** on GSM8K. 3

*   **Contamination Risks:** Detailed analysis reveals that high scores on GSM8K often reflect dataset contamination rather than true reasoning. Studies using "contamination-free" variants like **GSM1k** show performance drops of up to **13%** for some model families, indicating massive overfitting. 5
*   **The MATH Benchmark:** The **MATH** dataset (high school competition level) is following a similar trajectory. **DeepSeekMath-V2** and comparable models now achieve >90% on MATH-500 subsets 4, suggesting that even this benchmark is losing its discriminative power.

### 2.2 The New Frontier: Olympiad-Level Competitions

The true state of the art is now defined by performance on the **AIME** (American Invitational Mathematics Examination), **IMO**, and **Putnam** competitions. These exams present problems that require multi-step strategic planning, distinct from the template-following required for GSM8K.

#### 2.2.1 AIME Performance
The AIME serves as the qualifying threshold for the US Math Olympiad team.
*   **Current Capability:** Frontier models like **OpenAI o1** and **DeepSeek-R1** have achieved pass rates between **70% and 80%** on AIME 2024. 3 This places AI performance within the top tier of high school mathematicians in the United States.
*   **Significance:** A score of ~12/15 on the AIME is typically sufficient to qualify for the **USA Mathematical Olympiad (USAMO)**. This indicates that LLMs have effectively "graduated" from high school math.

#### 2.2.2 The Putnam Anomaly
The Putnam Competition is notoriously difficult, with a median score often being 0 or 1 out of 120 for human undergraduates.
*   **DeepSeekMath-V2:** In a technical report released in late 2024, DeepSeek claimed a score of **118/120** on the 2024 Putnam exam. 1
*   **Analysis:** This score is staggering. The highest human score in 2024 was roughly 90. If verified, this result implies that specialized AI systems have surpassed the most elite human undergraduates in specific, closed-domain problem solving. However, this result relies on scaled **test-time compute**, meaning the model may have generated thousands of candidate solutions before converging on the correct proofs. 1

### 2.3 The Hardness Hierarchy

Deep research into these results reveals a hierarchy of difficulty across mathematical domains:

*   **Euclidean Geometry (Solved):** The combination of visual intuition and rigid axiomatic rules makes geometry a prime candidate for neuro-symbolic approaches. **AlphaGeometry 2** solves 83% of historical IMO geometry problems, surpassing the average gold medalist. 2
*   **Algebra & Number Theory (Solvable):** These domains benefit from formalization (Lean) and tool use (Python). **AlphaProof** demonstrated competency here by solving two algebra problems and one number theory problem at IMO 2024. 10
*   **Combinatorics (The Frontier):** Combinatorics remains the most challenging domain. It often involves ad-hoc construction tasks or "counting" arguments that are difficult to formalize and resist standard algorithmic tools. The search space in combinatorics is often explosive, and "intuition" plays a larger role than formal deduction. 11

### 2.4 Gap Analysis

Despite these triumphs, a "fragility gap" remains.
*   **Robustness:** Models that solve IMO problems can fail on trivial variations of the same problem if the phrasing is altered (as seen in GSM-Symbolic studies). 12
*   **Semantic Understanding:** The reliance on large-scale search (generating 10,000 candidates) suggests that models are not "solving" problems in the human sense of unified understanding but are rather navigating a probability tree to find a verify-able leaf node.

---

## 3. Competition Analysis: Hard Evidence

Competitions provide an adversarial environment that filters out theoretical claims in favor of what actually works.

### 3.1 The AI Math Olympiad (AIMO): The Triumph of Tool Use

The AIMO Progress Prize was a watershed moment for open-source mathematical reasoning.

*   **Winner:** The **NuminaMath** model (7B parameters). 13
*   **Winning Strategy:** The core innovation was **Tool-Integrated Reasoning (TIR)**. Instead of relying on the LLM to perform arithmetic or algebraic expansion mentally, the model was trained to write and execute Python code.
*   **Technical Details:**
    *   **Architecture:** Fine-tuned DeepSeekMath-Base 7B.
    *   **Inference:** Used **Self-Consistency with Tool-Integrated Reasoning (SC-TIR)**. The model generated $N=64$ solution paths. For each path, Python code blocks were executed, and the output was fed back into the context. The final answers were aggregated via majority voting. 14
*   **Key Insight:** For competition math, the ability to verify a conjecture using code (e.g., "Write a loop to check the first 100 cases") is more valuable than a larger model parameter count.

### 3.2 International Mathematical Olympiad (IMO) 2024: The Silver Standard

Google DeepMind’s dual-system entry achieved a total score of 28/42, equivalent to a **Silver Medal**. 2

#### 3.2.1 AlphaProof
*   **Role:** Solved algebra and number theory problems.
*   **Mechanism:** AlphaProof creates a bridge between informal natural language and formal **Lean** code. It uses a Gemini-based **Autoformalizer** to translate the problem into Lean. Once formalized, an AlphaZero-based **Solver** searches for a proof tree. 2
*   **Training Data:** The system was trained on millions of synthetic problems generated by formalizing informal math statements and then "proving" them within the RL loop. 15

#### 3.2.2 AlphaGeometry 2
*   **Role:** Solved the complex geometry problem (Problem 4) in 19 seconds. 2
*   **Mechanism:** A neuro-symbolic hybrid. A language model predicts useful auxiliary constructions (e.g., "Add a point $X$ at the midpoint of $AB$"). A symbolic engine (based on Wu's method or similar algebraic deduction rules) then attempts to derive the proof from the augmented diagram. 9
*   **Key Insight:** The LLM provides "intuition" (the creative step), while the symbolic engine provides "rigor" (the logical step). This decomposition bypasses the LLM's weakness in long-chain strict logic.

### 3.3 The Putnam Shock

The reported 118/120 score by **DeepSeekMath-V2** on the 2024 Putnam competition challenges previous assumptions about the "formal vs. informal" gap. 1

*   **Significance:** Unlike AlphaProof, which relies on translation to Lean, DeepSeekMath-V2 operates primarily in natural language (augmented with tool use).
*   **Methodology:** The high score is attributed to **Reinforcement Learning with Verifiable Rewards (RLVR)** and massive test-time compute. The model serves as both a generator and a verifier, iteratively refining its proofs. 17 This suggests that with sufficient RL, natural language models can approximate formal rigor without the explicit overhead of a proof assistant.

---

## 4. Math-Specific Training Approaches

The training of reasoning models has evolved into a sophisticated pipeline that prioritizes process over outcome.

### 4.1 The DeepSeek R1 Pipeline

**DeepSeek-R1** represents the state-of-the-art in "Pure RL" for reasoning. The training recipe is a departure from standard RLHF (Reinforcement Learning from Human Feedback). 3

#### 4.1.1 Phase 1: Cold Start
Before RL begins, the model must be primed. Pure RL on a raw base model can lead to unstable outputs (e.g., mixing languages or unreadable formatting).
*   **Data:** A small, high-quality dataset of "Reasoning" examples is collected. These examples feature long, detailed Chains of Thought (CoT) that are readable and structured. 19
*   **Goal:** To establish a prior distribution where the model outputs its reasoning in a specific format (e.g., `<think>...</think>`).

#### 4.1.2 Phase 2: Reasoning RL (The "Aha" Moment)
The model is then subjected to large-scale RL using the **Group Relative Policy Optimization (GRPO)** algorithm. 20
*   **Reward Signal:** The reward is purely rule-based.
    *   **Accuracy:** Did the final answer match the ground truth?
    *   **Format:** Did the model use the proper thinking tags?
*   **No Human Feedback:** Notably, there is no human preference modeling in this phase. The model learns entirely by trial and error.
*   **Emergent Behavior:** During this phase, DeepSeek observed the emergence of the "Aha moment"—traces where the model spontaneously re-evaluates its previous steps (e.g., "Wait, I made a mistake here, let me recalculate"). 3 This self-correction was not taught via SFT but discovered as a strategy to maximize the accuracy reward.

#### 4.1.3 Group Relative Policy Optimization (GRPO)
GRPO is a crucial optimization for math RL.
*   **Math:** Unlike PPO, which requires a separate Value function (Critic) network ($V_\phi$) to estimate advantages, GRPO estimates the baseline from the group mean.
    *   For a prompt $q$, sample $G$ outputs $\{o_1, \dots, o_G\}$.
    *   Calculate rewards $\{r_1, \dots, r_G\}$.
    *   The advantage for output $i$ is $A_i = \frac{r_i - \text{mean}(r)}{\text{std}(r)}$.
*   **Efficiency:** This eliminates the memory overhead of the Critic model, allowing training to scale to 671B parameters. 22
*   **Objective:**
    $$J(\theta) = \mathbb{E} [ \dots ]$$

### 4.2 DeepSeekMath and Data Quality

The performance of DeepSeekMath (the predecessor to R1) underscores the importance of pre-training data.
*   **Data Selection:** DeepSeekMath trained a FastText classifier on **OpenWebMath** to filter Common Crawl. They found that training on **120B tokens** of high-quality math data was superior to training on trillions of generic tokens. 24
*   **Lesson:** For math, data quality follows a power law. A small amount of "textbook quality" data is worth orders of magnitude more than generic web text.

### 4.3 Reward Signals

*   **Outcome Rewards (ORM):** Binary signals (Correct/Incorrect). Easy to scale but sparse. DeepSeek-R1 relies primarily on this. 26
*   **Process Rewards (PRM):** Dense signals given at each step of reasoning. While theoretically superior for guiding search, PRMs are expensive to train and prone to "reward hacking" (where the model games the verifier). Recent research suggests that a strong ORM with MCTS or Best-of-N often outperforms PRM-based approaches due to the noise in PRM labels. 27

---

## 5. Formal Verification Integration

Formal verification solves the hallucination problem by enforcing strict logical consistency.

### 5.1 The Role of Lean 4

Lean is an interactive theorem prover that serves as the "compiler" for mathematics. If a proof is accepted by Lean, it is correct.
*   **Integration:** Models like **AlphaProof** treat Lean as an environment. The LLM acts as the "Policy," generating tactics (commands like `induction n`, `simp`, `apply lemma_1`). Lean acts as the "Environment," returning the new state of the proof or an error message. 29

### 5.2 AlphaProof's Architecture

AlphaProof's success relies on solving the data scarcity problem in formal math (there are very few human-written Lean proofs compared to English proofs).
*   **Autoformalization:** A fine-tuned Gemini model translates **informal math** (from the IMO training set) into **formal Lean statements**. 15
*   **Synthetic Data Generation:** The solver attempts to prove these formal statements. When it succeeds, the (Statement, Proof) pair is added to the training set.
*   **The Loop:** This creates a self-reinforcing loop. As the solver gets better, it can prove harder statements, which creates better training data. 15

### 5.3 The Translation Tax

The major bottleneck is the "informal-to-formal" gap.
*   **Ambiguity:** Human math often leaves details implicit ("it is obvious that..."). Formal math requires every detail to be explicit.
*   **Error Propagation:** If the autoformalizer translates the problem incorrectly, the solver might prove a trivial theorem instead of the actual problem. This "alignment tax" limits the scalability of formal approaches to domains where autoformalization is robust (like Algebra). 30

---

## 6. Tool-Integrated Mathematical Reasoning (TIR)

For many practical applications, "Tool-Integrated Reasoning" is a more viable path than formal verification.

### 6.1 How TIR Works

TIR changes the action space of the LLM. Instead of just emitting text, the LLM can emit a code block (e.g., in Python).
*   **Workflow:**
    *   **Thought:** "I need to calculate the discriminant of this quadratic."
    *   **Action:** `print(b**2 - 4*a*c)`
    *   **Observation:** The Python interpreter executes the code and returns `144`.
    *   **Reasoning:** "The discriminant is positive, so there are two real roots."
*   **Benefits:** This completely eliminates arithmetic errors, which are a major source of failure for pure LLMs. It also allows for "experimental math"—writing scripts to test hypotheses. 13

### 6.2 Training for TIR

The **NuminaMath** dataset is the gold standard for TIR training. 32
*   **Construction:** Since most math datasets (like MATH) are text-only, NuminaMath used GPT-4 to rewrite existing solutions into the TIR format.
*   **Filtering:** The generated code solutions were executed. Only those that produced the correct final answer were kept. This created a massive supervised dataset of (Question, Code-Augmented Solution) pairs. 33
*   **Curriculum:** Training starts with text-only reasoning to build logic, then introduces tool-use tokens to teach the model when to call the tool. 14

---

## 7. Test-Time Techniques and Inference Strategies

Once a model is trained, the "heavy lifting" shifts to inference. The concept of **Inference Scaling Laws** suggests that computing power spent at test time is as valuable as computing power spent during training.

### 7.1 Inference Scaling Laws

Recent empirical studies 34 demonstrate a **log-linear relationship** between test-time compute and accuracy.
*   **Formula:** Accuracy $A \approx \alpha \cdot \log(C) + \beta$, where $C$ is the compute budget (number of samples or search steps).
*   **Implication:** A smaller model (e.g., 7B) with a budget of $N=1000$ samples can often outperform a larger model (e.g., 70B) with greedy decoding. This democratizes high-performance reasoning, as inference on small models is cheap.

### 7.2 Search Algorithms

To exploit this scaling law, we need effective search algorithms.

| Algorithm | Description | Pros | Cons |
| :--- | :--- | :--- | :--- |
| **Best-of-N (Majority Voting)** | Sample $N$ solutions and vote on the final answer. | Robust, easy to implement. Works well with TIR. | Computationally expensive (linear cost). |
| **Weighted Best-of-N** | Sample $N$ solutions and score them with a Verifier/PRM. | More sample-efficient than simple voting. | Requires a high-quality Verifier (hard to train). |
| **Monte Carlo Tree Search (MCTS)** | Build a search tree of reasoning steps. Explore promising branches. | Theoretical optimal for lookahead. | Very slow. Highly sensitive to the quality of the value function. |

**Evidence:** Recent findings 27 indicate that **Best-of-N with a Verifier** often outperforms MCTS in practice. The reason is that MCTS relies on intermediate value estimates ($V(s)$), which are often noisy in LLMs. If the value function is imperfect, MCTS can be misled into deep, incorrect rabbit holes. In contrast, Best-of-N evaluates the *completed* reasoning chain, which is often easier to judge.

### 7.3 Compute-Optimal Inference

The "Reasoning" curriculum should emphasize dynamic compute allocation. Easy problems should be solved with greedy decoding (System 1). Hard problems should trigger a search process (System 2).
*   **Heuristics:** Models can be trained to predict the difficulty of a problem and output a "computational budget" token, effectively learning to ask for more time when needed. 34

---

## 8. What's Unique About Math

Mathematics occupies a unique position in the landscape of AI reasoning.

### 8.1 Verifiability as the Engine

The defining feature of math is **Verifiability**. This enables **Reinforcement Learning with Verifiable Rewards (RLVR)**.
*   **Mechanism:** In domains like creative writing, the reward signal is subjective and noisy (requires a human or a strong LLM judge). In math, the reward signal is **deterministic** (the code compiles, the answer matches).
*   **Impact:** This allows for **infinite self-play**. A model can generate 10,000 problems, attempt to solve them, verify its own answers, and update its weights. This is the same mechanic that allowed AlphaZero to master Go. 36

### 8.2 The Transfer Debate

Does learning math make an LLM smarter at everything else?
*   **Evidence For:** DeepSeek-R1, trained purely on math/code RL, shows improvements in general logic and coding tasks. The "Reasoning Curriculum" suggests that the habits of mind (backtracking, verification) transfer. 38
*   **Evidence Against:** "Overfitting" phenomena. A model might learn the specific logic templates for AIME problems but fail on real-world business logic. This suggests that the transfer is limited to domains that share the underlying structural logic of the training data. 40

---

## 9. Novel Findings and Emerging Trends

### 9.1 Emergent Self-Correction ("Aha Moments")

The most striking finding from the DeepSeek-R1 research is that **self-correction is an emergent property of RL**.
*   **Observation:** Without being explicitly prompted to "check your work," models trained with RL naturally begin to generate traces like "Let me double check that calculation."
*   **Reasoning:** The RL policy learns that generating these "verification tokens" increases the probability of getting the final reward. The model effectively learns to "buy time" to think. 3

### 9.2 The "Copy-Paste" Augmentation

A surprisingly effective technique from NuminaMath is simply **repeating the problem statement** in the prompt or augmenting the data with "Copy-Paste" variations.
*   **Mechanism:** By forcing the model to process the problem statement multiple times (or in slightly different forms), the attention mechanism is forced to attend to all constraints, reducing the "skimming" errors common in LLMs. 14

### 9.3 Limits of Synthetic Data

While AlphaProof generates millions of proofs, researchers are hitting a "triviality ceiling." A synthetic generator can create infinite easy problems ($x + 1 = 2$) or infinite nonsense problems. Generating **structurally novel and challenging problems** (like those in the IMO) remains an unsolved challenge for AI. The best training data still comes from human creativity (competitions), which makes the limited supply of such data a critical bottleneck. 15

---

## 10. Open Questions

*   **Can Autoformalization Scale?** Will we reach a point where AI can automatically translate the entire arXiv into Lean, or will the "translation tax" always limit formal methods to niche domains?
*   **Is MCTS Worth It?** Given the success of simple Best-of-N and the difficulty of training stable Value functions, will complex tree search algorithms ever become the standard for text-based reasoning?
*   **The "Generalization" of Reasoning:** Can the RL techniques that solve math (RLVR) be applied to domains with "soft" verification, like law or medicine, perhaps via rigorous simulation environments?

---

## 11. Technical Roadmap for Practitioners

Based on this analysis, a curriculum for Mathematical Reasoning with RL should follow this roadmap:

1.  **Data Preparation:**
    *   Curate a high-quality "Cold Start" dataset of readable CoT traces (using distillation from frontier models).
    *   Build a TIR Dataset by rewriting existing math problems into Python-executable formats.
2.  **Base Model Training:**
    *   Start with a strong code-instructed model (e.g., **DeepSeek-Coder-V2** or **Qwen-Math**).
    *   Perform SFT on the Cold Start data to stabilize the output format.
3.  **RL Implementation:**
    *   Implement **GRPO** to scale RL without critic memory overhead.
    *   Use a **Hybrid Reward Function**: Binary Correctness (via tool execution) + Format Constraints (XML tags).
    *   Avoid complex PRMs initially; focus on rigorous Outcome Rewards.
4.  **Inference:**
    *   Deploy with **SC-TIR** (Best-of-N with code execution).
    *   Scale $N$ based on problem difficulty (Adaptive Inference).

This approach leverages the "what works" evidence: **Tool use for reliability, RL for reasoning depth, and inference scaling for hard problems.**