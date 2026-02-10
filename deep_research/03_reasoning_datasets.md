# Deep Research Prompt: Reasoning Datasets Landscape

## Context

> You are conducting research to inform a curriculum on Reinforcement Learning for LLM Reasoning. This curriculum emphasizes **depth over breadth**, **technical verification over theoretical claims**, and **what actually works in practice**.

This prompt focuses on **datasets**—the data used to train reasoning models, how to assess quality, and the emerging practice of synthetic data generation.

---

## Your Research Goal

> Develop a comprehensive understanding of the data landscape for training reasoning models. The goal is to answer:
>
> **"What data should I use to train a reasoning model, how do I know if it's good, and how do I create more?"**

---

## Core Questions to Investigate

### 1. The Dataset Landscape

*   **What are the major datasets used for training reasoning models?**
    *   **Mathematical reasoning:** GSM8K, MATH, NumGLUE, etc.
    *   **Code reasoning:** HumanEval, MBPP, CodeContests, etc.
    *   **General reasoning:** ARC, HellaSwag, etc.
*   **What are their sizes, formats, and difficulty distributions?**
*   **Which datasets are used for training vs evaluation?**
*   **What datasets did DeepSeek R1, OpenAI o1, and other frontier models use?**

### 2. What Makes a Good Reasoning Dataset

*   **What properties correlate with downstream model quality?**
*   **Is more data always better, or is there a quality-quantity tradeoff?**
*   **What's the right difficulty distribution?** (Easy examples needed for learning? Or focus on hard?)
*   **How important are solution traces vs just question-answer pairs?**
*   **Does diversity of problem types matter?**

### 3. Data Quality Assessment

*   **How do you assess if a reasoning dataset is "good"?**
*   **What are signs of contamination** (train-test overlap with benchmarks)?
*   **How do you detect labeling errors or ambiguous questions?**
*   **What automated quality checks can you run?**

### 4. Synthetic Data Generation

*   **How are frontier labs generating synthetic reasoning data?**
*   **What role do stronger models play in generating data for weaker models?**
*   **How do you ensure synthetic data is correct?** (Verification, filtering, etc.)
*   **What are the known failure modes of synthetic data?**
*   **How much does synthetic data help compared to human-curated data?**

### 5. Data for RL Training Specifically

*   **How does data for RL (GRPO, PPO) differ from SFT data?**
*   **What's needed for reward model training?**
*   **How do you create preference pairs for DPO-style training?**
*   **What's the role of negative examples (wrong solutions)?**

### 6. Practical Data Curation

*   **Where do you actually get reasoning data?**
*   **What licensing issues exist?**
*   **How much curation effort is needed?**
*   **What tools exist for data preparation?**

---

## Evidence Standards

Prioritize findings with **strong evidence**:

1.  **What frontier models were actually trained on** (DeepSeek, Qwen, etc. often publish this)
2.  **Ablation studies on data choices**
3.  **Competition-winning approaches** (AIMO—what data did winners use?)
4.  **Replicated findings across multiple teams**

**Be skeptical of:**

*   Claims that a new dataset is "better" without controlled comparisons
*   Dataset papers without downstream evaluation
*   Synthetic data claims without verification methodology

---

## What to Avoid

*   Don't just enumerate every dataset ever created
*   Don't focus only on what exists—also cover how to create/curate
*   Don't ignore quality—size isn't everything

---

## Room for Discovery

While investigating the above questions, remain open to:

*   Novel datasets you encounter that deserve attention
*   Surprising findings about what data properties matter most
*   Emerging synthetic generation techniques
*   Critiques of popular datasets
*   Connections between data and training paradigm (what data works for RL vs SFT?)

---

## Output Structure

Organize your findings as follows:

1.  **Executive Summary:** Key takeaways for practitioners
2.  **Dataset Landscape:** Major datasets organized by domain, with key stats
3.  **Quality Indicators:** What makes data good for reasoning training
4.  **Synthetic Data:** How to generate it, verify it, and when it works
5.  **RL-Specific Considerations:** Data needs for GRPO, PPO, DPO, etc.
6.  **Practical Guide:** Where to get data, how to curate, tools available
7.  **Novel Findings:** Things you discovered beyond the core questions
8.  **Open Questions:** What remains unclear?
9.  **Sources:** Key papers, datasets, and resources

---

## Final Note

> Data is often the most important factor in model quality, yet it's under-discussed compared to algorithms. Focus on practical, actionable insights. What should someone actually do when building a reasoning training dataset?

---

# Deep Research Output

# The Reasoning Datasets Landscape: A Comprehensive Curriculum Guide for Reinforcement Learning in Large Language Models

## 1. Introduction

The trajectory of Large Language Model (LLM) development has undergone a distinct phase shift.

The initial era, defined by the scaling laws of pre-training on massive unstructured corpora, has yielded to a new paradigm focused on post-training for reasoning. This shift, exemplified by frontier models such as **OpenAI's o1** and **DeepSeek-R1**, prioritizes **"System 2" thinking**—deliberate, multi-step, and verifiable cognition—over the rapid "System 1" pattern matching that characterized earlier GPT iterations.

> For researchers and curriculum designers specializing in Reinforcement Learning (RL) for LLM Reasoning, this transition necessitates a fundamental re-evaluation of the data ecosystems that fuel model training.

Understanding the dataset landscape is no longer merely about volume; it is about the **architecture of reasoning trajectories**, the **rigorousness of process verification**, and the **orchestration of synthetic data pipelines**.

This report serves as a foundational text for a curriculum on RL for LLM Reasoning. It moves beyond the simplistic view of datasets as static collections of question-answer pairs to explore the complex infrastructure required to incentivize robust reasoning behaviors.

The distinction between data for **Supervised Fine-Tuning (SFT)** and data for **Reinforcement Learning (RL)** has become technically distinct yet operationally intertwined. While SFT provides the "cold start" priors that stabilize model behavior and enforce output formatting, RL—specifically through algorithms like **Group Relative Policy Optimization (GRPO)** and **Proximal Policy Optimization (PPO)**—requires environments where correctness is objectively verifiable.

Consequently, the domain of reasoning data is currently dominated by **mathematics**, **coding**, and **scientific logic**, domains where ground truth is computable and reward signals are deterministic.

We will rigorously analyze the current dataset landscape, dissecting the methodology of synthetic data generation that powers models like DeepSeek-R1, and detailing the specific formatting requirements for modern RL pipelines. This document is designed to provide the evidence base required to construct a curriculum that is not only theoretically sound but practically applicable to the cutting edge of AI research.

## 2. The Reasoning Dataset Landscape

The landscape of reasoning datasets is bifurcated into **established benchmarks** that serve as historical baselines and **emerging frontier datasets** designed to push the limits of current cognitive architectures.

A robust curriculum must cover both to provide context on the evolution of the field, understanding that "reasoning" in LLMs is currently operationalized primarily through math and code due to the availability of automated verifiers.

### 2.1 Mathematical Reasoning Datasets

Mathematics remains the "fruit fly" of reasoning research. Its objective truth conditions, hierarchical complexity, and amenability to automated verification make it the ideal testbed for developing RL algorithms.

#### 2.1.1 Foundational Benchmarks: The Baseline

**GSM8K (Grade School Math 8K)** remains the ubiquitous entry point for training and evaluation in reasoning curriculums. Comprising approximately 8.5K high-quality, linguistically diverse grade school math word problems, it requires multi-step reasoning to solve.

*   **Composition:** The dataset is split into 7.5K training examples and 1K test examples.
*   **Pedagogical Significance:** While considered "solved" by frontier models (which routinely achieve >90% accuracy), GSM8K remains the standard for validating whether a model has acquired basic arithmetic reasoning and step-by-step coherence. It serves as the "Hello World" of reasoning RL, allowing researchers to debug RL pipelines like GRPO without the massive compute overhead associated with more complex datasets.
*   **Limitations:** Its simplicity means that strong models can often solve it via pattern matching rather than deep reasoning, necessitating harder benchmarks for advanced training.

**MATH (Mathematics Aptitude Test of Heuristics)** represents the next significant leap in difficulty. Unlike GSM8K, MATH consists of 12,500 problems drawn from prestigious mathematics competitions such as the AMC 10, AMC 12, and AIME.

*   **Granularity:** Problems are categorized by difficulty (levels 1–5) and subject (algebra, geometry, number theory, counting & probability, pre-algebra, pre-calculus, and intermediate algebra).
*   **Format:** Solutions are provided in LaTeX, often accompanied by rigorous proofs.
*   **Curriculum Utility:** This dataset defines the "high school to olympiad" capability gap. Models that excel on GSM8K often fail on MATH (specifically level 5 problems), making it the primary target for advanced reasoning optimization. The gap between a model's performance on GSM8K and MATH is often used as a proxy for its ability to generalize reasoning beyond simple templates.

#### 2.1.2 Frontier and Synthetic Math Datasets: The New Standard

The need for scale in RL training has led to the creation of massive synthetic or semi-synthetic datasets that dwarf traditional benchmarks.

**NuminaMath (CoT and TIR):** Developed by Project Numina, winners of the first **AI Mathematical Olympiad (AIMO) Progress Prize**, this dataset is critical for any modern reasoning curriculum.

*   **Scale:** It contains over 860,000 problem-solution pairs, significantly larger than MATH or GSM8K.
*   **Dual Modalities:** Crucially, it offers both **Chain-of-Thought (CoT)** data (natural language reasoning) and **Tool-Integrated Reasoning (TIR)** data. In TIR, the model learns to interleave Python code execution to solve math problems, effectively using a REPL (Read-Eval-Print Loop) as a cognitive prosthesis.
*   **Source:** The data is derived from Chinese high school exercises, US olympiads, and online forums, processed to filter for quality and diversity.
*   **Impact:** NuminaMath demonstrates the shift towards "Code-as-Reasoning," where models use computational tools to verify their own logic step-by-step, a technique that significantly reduces calculation errors.

**OpenMathReasoning:** Released by NVIDIA, this dataset represents the industrial scale of reasoning data.

*   **Structure:** It is a massive collection comprising 540,000 unique problems expanded into millions of samples, including 3.2 million CoT solutions and 1.7 million TIR solutions.
*   **Augmentation Strategy:** It utilizes "**GenSelect**," a method where models generate multiple solutions, and the most promising ones (verified by correct answers) are selected. This creates a dense training signal for RL, providing multiple valid paths to a single solution.
*   **Sources:** Heavily sourced from the Art of Problem Solving (AoPS) forums and synthetic expansions, it targets the competition-level math domain.

### 2.2 Code Reasoning and Programming Datasets

Code generation is isomorphic to logical reasoning. The execution of code provides a definitive reward signal—it either compiles and passes tests, or it does not. This objective feedback loop makes code data ideal for RL.

#### 2.2.1 Core Datasets for Evaluation and Tuning

**HumanEval:** Released by OpenAI, this dataset consists of 164 hand-written programming problems with unit tests.

*   **Curriculum Note:** It is notoriously contaminated in web-crawled pre-training data (e.g., The Pile, The Stack). Studies show significant overlap between HumanEval solutions and training corpora, meaning models often memorize these solutions rather than reason through them. A curriculum must address this by introducing decontamination strategies or using it strictly for zero-shot evaluation on check-pointed models.

**MBPP (Mostly Basic Python Problems):** Similar to HumanEval but larger (approx. 1,000 problems), focused on entry-level programming concepts. Like HumanEval, it suffers from significant contamination and is best used as a baseline rather than a primary training source for advanced reasoners.

**LiveCodeBench:** This is a dynamic benchmark that addresses the contamination crisis by collecting problems from coding contests (LeetCode, AtCoder, Codeforces) released after a model's training cutoff date.

*   **Relevance:** It tests generalization—the ability to solve novel problems—rather than recall. For a reasoning curriculum, LiveCodeBench is the superior metric for assessing true "System 2" capabilities in coding.

#### 2.2.2 Advanced Reasoning Code Data

**Apps & CodeContests:** These datasets aggregate difficult competitive programming problems. They differ from basic code generation by requiring complex algorithmic logic, often necessitating a "reasoning trace" (understanding the problem constraints, selecting the algorithm) before writing the code. They are used to train models that can handle edge cases and efficiency constraints, not just syntax.

### 2.3 General and Commonsense Reasoning

While math and code offer verifiable rewards, general reasoning requires datasets that capture logic, deduction, and world knowledge without a compiler to check the output.

**ARC-AGI (Abstraction and Reasoning Corpus):** Created by François Chollet, this benchmark measures "fluid intelligence"—the ability to learn new skills from very few examples.

*   **Format:** Visual grid transformation tasks where the model must deduce the transformation rule from a few demonstration pairs and apply it to a test input.
*   **Status:** It remains largely an unsolved benchmark for LLMs, with state-of-the-art models only recently crossing 50% accuracy. It represents the frontier of abstract reasoning that cannot be easily memorized, serving as a litmus test for general intelligence.

**HellaSwag:** A dataset for commonsense Natural Language Inference (NLI). It uses "Adversarial Filtering" to create wrong answers that are plausible to models but nonsensical to humans.

*   **Utility:** Useful for training models to reject hallucinated or illogical continuations, refining the model's internal discriminator.

**OpenThoughts:** A recently released collection (e.g., OpenThoughts-114k) that covers math, science, code, and puzzles.

*   **Construction:** It uses distillation from DeepSeek-R1 and other strong models to generate high-quality reasoning traces for open-source questions.
*   **Curriculum Relevance:** This represents the "democratization" of reasoning data, allowing researchers and students to train R1-like models on consumer hardware using high-quality open data.

**NaturalReasoning:** A massive dataset released by Meta, containing 2.8 million reasoning questions back-translated and decontaminated from web corpora.

*   **Significance:** It demonstrates the scale required for general reasoning. Unlike math datasets which are often smaller but denser, general reasoning benefits from massive variety.

### 2.4 Data Usage by Frontier Models: DeepSeek-R1 and OpenAI o1

The analysis of DeepSeek-R1 and OpenAI o1 reveals a bifurcated data strategy that shapes modern RL pipelines:

*   **Cold Start Data (SFT):** Before RL, models are fine-tuned on a small, high-quality set of reasoning problems with detailed Chains of Thought. DeepSeek-R1 uses a "cold start" dataset to prevent the "collapse" of readability. Without this, RL optimization can lead the model to develop an internal shorthand or "language mixing" that is unintelligible to humans. This data acts as a formatting prior.
*   **RL Data (PPO/GRPO):** This data often consists of prompts only (the question) and a verifier (the ground truth answer or a test case). The model generates the reasoning path (the data) during training, and the RL algorithm reinforces paths that lead to the correct verifier output. This decoupling of prompt and response is a key innovation: the "training data" is effectively the prompt distribution, while the "labels" are dynamically generated by the environment (the verifier).

**Table 1: Comparative Analysis of Key Reasoning Datasets**

| Dataset | Domain | Size | Key Feature | Primary Use in Curriculum |
| :--- | :--- | :--- | :--- | :--- |
| **GSM8K** | Math | 8.5K | Grade-school level, high verbal diversity | Intro to CoT & RLHF pipelines |
| **MATH** | Math | 12.5K | Competition level, difficulty labeled | Advanced reasoning & generalization |
| **NuminaMath** | Math/Code | 860K+ | CoT + Tool-Integrated Reasoning (TIR) | Scaling laws & Tool use |
| **OpenMathReasoning** | Math | 540K+ | Augmented solutions (GenSelect) | Data augmentation strategies |
| **HumanEval** | Code | 164 | Function synthesis w/ unit tests | Evaluation (beware contamination) |
| **ARC-AGI** | Abstract | 400-800 | Few-shot visual pattern matching | General intelligence & generalization |
| **HellaSwag** | Commonsense | 70K | Adversarially filtered endings | Logical consistency & NLI |
| **NaturalReasoning** | General | 2.8M | Backtranslated web reasoning | General domain reasoning |

Export to Sheets

## 3. What Makes a Good Reasoning Dataset?

A dataset suitable for training reasoning models differs fundamentally from one used for general conversational fluency. The curriculum must emphasize four critical properties: **Process Visibility**, **Difficulty Distribution**, **Diversity**, and **Verifiability**.

### 3.1 Process Visibility: The Chain of Thought

The core innovation in modern reasoning models is the explicit generation of a "thought process" before the final answer. Good reasoning datasets must include these traces.

*   **Trace Density:** Ideally, the dataset should not just provide the answer ($y$) but the reasoning trajectory ($r$) such that the training tuple is $(x, r, y)$. DeepSeek-R1's performance relies heavily on thousands of tokens of reasoning data, often filtered from larger sets to ensure quality.
*   **Step-wise Delimitation:** For **Process Reward Models (PRMs)**, the reasoning trace must be broken down into discrete steps (e.g., line-by-line in math proofs). Datasets like PRM800K and Math-Shepherd provide labels for these intermediate steps, not just the final outcome. This allows the RL agent to receive dense feedback ("step 3 was wrong") rather than sparse feedback ("the final answer is wrong").

### 3.2 Difficulty and "The Frontier of Capability"

Reasoning capabilities emerge when models are trained on data that sits on the frontier of their current ability.

*   **Curriculum Learning:** A dataset should ideally be stratified. Training on simple GSM8K problems alone does not generalize to MATH. Conversely, starting with MATH level 5 can lead to hallucinations if the model lacks basic arithmetic grounding. The data must provide a gradient of difficulty.
*   **Evolutionary Complexity:** Techniques like **Evol-Instruct** are used to synthetically increase the difficulty of existing datasets. This involves prompting a teacher model to rewrite a problem by adding constraints, increasing the number of reasoning steps, or complicating the input parameters. This ensures that the model is constantly challenged.

### 3.3 Diversity and De-correlation

A common failure mode in RL for reasoning is "reward hacking," where the model learns a heuristic to get the right answer without understanding (e.g., "the answer is usually the largest number in the text").

*   **Topic Diversity:** Datasets must span algebra, geometry, coding, and logic puzzles to prevent the model from overfitting to a specific template.
*   **Solution Diversity:** High-quality datasets like **OpenMathReasoning** include multiple valid reasoning paths for a single problem. This teaches the model that reasoning is a search process through a solution space, not a linear retrieval task.

### 3.4 Verifiability

For RL training, the data must allow for automated verification.

*   **Determinism:** Math and Code are preferred because the reward function is binary and objective (True/False).
*   **Grounding:** Datasets grounded in formal systems (like Lean for theorem proving or Python for calculation) are superior to purely text-based reasoning because they minimize the "hallucination gap." If a model outputs a Python script that executes and returns the correct answer, the reasoning is implicitly verified.

## 4. Data Quality Assessment

The integrity of reasoning models is inextricably linked to the purity of their training data. A curriculum on this subject must aggressively address **Data Contamination** and **Quality Filtering**, as training on test data invalidates all subsequent evaluations.

### 4.1 The Contamination Crisis

Data contamination occurs when the test set of a benchmark (like HumanEval or GSM8K) leaks into the pre-training corpus. This leads to inflated performance metrics where the model is "reciting" rather than "reasoning."

*   **Evidence:** Studies have shown that up to 20% of benchmarks like MBPP and HumanEval exist verbatim or in paraphrased forms within common corpora like The Stack or The Pile.
*   **Impact on RL:** If an RL model is trained on contaminated data, the reward signal reinforces memorization of specific strings rather than the execution of generalizable logic. This results in "fragile" models that fail when the problem phrasing is slightly altered.

### 4.2 Detecting Contamination

The curriculum should cover methods for detecting leakage:

*   **N-gram Overlap:** Checking for exact string matches (e.g., 13-grams) between training data and test benchmarks.
*   **Embedding Similarity:** Using vector databases to find semantically identical questions even if phrased differently.
*   **Model-Based Probing:** Generating completions for benchmark questions. If a model can complete the question perfectly with zero-shot prompting, it is likely contaminated.
*   **Tools:** The report identifies specific tools like **Lilac**, which automates the detection of duplicates and PII (Personal Identifiable Information), and **Cleanlab**, which identifies label errors. **LeakageDetector** is a specific tool for identifying leakage in Jupyter notebooks, a common source of code data.

### 4.3 Automated Quality Checks

Beyond contamination, reasoning data must be syntactically and logically sound.

*   **Heuristic Filtering:** Removing samples that are too short, contain "I'm sorry, but..." (refusal strings), or have malformed code blocks.
*   **Execution-Based Filtering:** For code datasets, running the solution against unit tests is the gold standard. If the code doesn't run, it is removed. This "execution feedback" loop is central to datasets like **NuminaMath-TIR**.
*   **Decontamination Pipelines:** Tools like **LLM-Decontaminator** quantify a dataset's rephrased samples relative to a benchmark and remove them, ensuring the training set is pristine.

## 5. Synthetic Data Generation

Given the scarcity of high-quality human reasoning traces, synthetic data generation has become the primary engine for training reasoning models. This section outlines the methodologies that constitute the state-of-the-art.

### 5.1 The Distillation Pipeline

The most common approach is distilling reasoning capabilities from a stronger model (Teacher) to a weaker model (Student).

*   **Method:** A model like GPT-4o, Claude 3.5, or DeepSeek-R1 is prompted to solve a problem with a detailed step-by-step explanation.
*   **Verification:** The generated solution is checked against the ground truth. Only correct solutions are kept.
*   **Example:** The **OpenThoughts** pipeline uses DeepSeek-R1 to generate reasoning traces for math and science problems, creating a fully open-source training set. This effectively transfers the "reasoning patterns" of the teacher to the student.

### 5.2 Self-Correction and Bootstrapping

Self-correction algorithms allow a model to improve its own data quality without a superior teacher, mimicking a form of metacognition.

*   **STaR (Self-Taught Reasoner):** This algorithm iterates through a loop:
    1.  The model attempts to solve problems.
    2.  Correct solutions are added to the training set.
    3.  Incorrect solutions are discarded (or rationalized with the correct answer).
    4.  The model is fine-tuned on the new set. This creates a "virtuous cycle" where the model iteratively improves its own training data.
*   **Quiet-STaR:** An evolution of STaR where the model learns to generate internal "thought tokens" (rationales) for every token in a text, not just for QA pairs. This generalizes reasoning to arbitrary text generation, effectively embedding a "pause to think" mechanism within the generation process.
*   **SCoRe (Self-Correction via RL):** A multi-turn RL approach where the model is trained to improve its answer in a second attempt. Unlike standard SFT, SCoRe explicitly trains the model to recognize errors and correct them using entirely self-generated data. This is critical because simply training on "correct" traces does not teach a model how to recover from failure.

### 5.3 Evolutionary Strategies

Evol-Instruct methodologies take a seed dataset and iteratively complexify it.

*   **In-Depth Evolution:** Adding constraints, increasing reasoning steps, or requiring specialized knowledge.
*   **In-Breadth Evolution:** Generating new, distinct topics based on the seed data.
*   **Outcome:** This prevents the model from plateauing on simple logic and forces it to generalize to more complex structural requirements.

### 5.4 Generative Adversarial Approaches

Newer methods like **GAR (Generative Adversarial Reasoner)** employ a generator (Solver) and a discriminator (Critic). The Generator creates reasoning traces, and the Discriminator learns to identify errors within those traces. The two improve in tandem, with the Discriminator providing a dynamic reward signal that evolves as the Generator improves. This adversarial dynamic prevents the Generator from finding "easy" paths that fool a static verifier.

## 6. Data for Reinforcement Learning (RL) Training

Training a reasoning model via RL requires specific data formats and structures that differ from standard SFT. This section details the technical specifications for implementing algorithms like PPO and GRPO.

### 6.1 The Shift to GRPO (Group Relative Policy Optimization)

DeepSeek-R1 and similar models have popularized GRPO over PPO for reasoning.

*   **Mechanism:** PPO typically requires a separate "value model" (critic) to estimate the expected reward, which doubles the memory footprint. GRPO eliminates the critic. Instead, it samples a group of outputs for a single prompt and normalizes the rewards within that group. The baseline is simply the average reward of the group.
*   **Data Requirement:** The dataset consists of Prompts (Questions) and Ground Truths (Verifiers). The reasoning traces are not inputs; they are generated by the model during the exploration phase. This makes GRPO highly data-efficient regarding annotation—you only need the question and the answer, not the steps.
*   **Format Example:**

```json
{
  "prompt": "Solve for x: 2x + 5 = 15",
  "answer": "5"
}
```

The RL loop generates multiple solutions ($o_1, o_2, \dots, o_G$), checks which ones equate to "5", and upweights those trajectories.

### 6.2 Reward Model Training Data

If using PPO or training a verifier/discriminator (Reward Model), specific "Preference Pairs" are needed.

*   **Structure:** The data is structured as triplets $(x, y_w, y_l)$ where $x$ is the prompt, $y_w$ is the winning (better) response, and $y_l$ is the losing response.
*   **Source:** These pairs can be generated by sampling N solutions and ranking them based on correctness (e.g., correct answer vs. incorrect answer) or efficiency (fewer steps to correct answer).
*   **Process Reward Data:** For Process Reward Models (PRMs), the data must be labeled at the step level.

```json
{
  "problem": "Calculate the derivative...",
  "steps": [
    {"step": "2x = 10", "label": "correct"},
    {"step": "x = 4", "label": "incorrect"}
  ]
}
```

Datasets like **PRM800K** follow this rigorous format to train models that can catch errors mid-thought, which is essential for training models to self-correct rather than just hoping for the right final answer.

### 6.3 Input Formatting for Reasoning Models

Frontier models utilize special tokens to demarcate reasoning, effectively creating a reserved "latent space" for thought.

*   **DeepSeek Formatting:** Uses `<think>` and `</think>` tags. The SFT data ("Cold Start") must be pre-formatted with these tags wrapping the CoT to teach the model to enter "reasoning mode" before outputting the final answer.
*   **System Prompts:** The data must often include system instructions that enforce a specific tone or structure (e.g., "You are a helpful assistant that reasons step-by-step..."). This conditions the model to expect a reasoning task.

## 7. Practical Data Curation: Sources and Tools

For a practical curriculum, students need access to tools and repositories to build their own pipelines. This section bridges the gap between theory and implementation.

### 7.1 Primary Data Sources

**Hugging Face Hub:** The central repository for open datasets. Key collections include:

*   **open-thoughts/OpenThoughts-114k:** High-quality reasoning traces suitable for SFT.
*   **AI-MO/NuminaMath-CoT:** The AIMO winning dataset, excellent for both CoT and TIR training.
*   **nvidia/OpenMathReasoning:** Massive scale synthetic math data for large-scale pre-training or RL.
*   **facebook/natural_reasoning:** A massive general reasoning dataset for domain generalization.
*   **Project Numina:** A specialized source for math and code reasoning data, often linked to formal proofs (Lean), ideal for advanced research projects.

### 7.2 Licensing Considerations

*   **Permissive:** MIT / Apache 2.0 (e.g., GSM8K, OpenMathReasoning). These are safe for commercial use and modification, making them ideal for university curriculums and startups.
*   **Restrictive/Research Only:** CC-BY-NC (Non-Commercial) or model-specific licenses. NaturalReasoning, for example, is CC-BY-NC.
*   **Distillation Clauses:** A critical nuance is that data generated by proprietary models (like GPT-4) often has terms of service restricting its use to compete with the source model. However, DeepSeek's R1 license explicitly permits distillation, making its generated data highly valuable for open research and removing legal ambiguity for curriculum use.

### 7.3 Curation Tools

*   **Distilabel (Argilla):** A framework specifically designed for generating and filtering synthetic data. It integrates with various LLM providers to build reasoning datasets via pipelines (e.g., generation -> critique -> refinement). It supports creating "preference" datasets for DPO/RLHF.
*   **TRL (Transformer Reinforcement Learning):** A Hugging Face library that standardizes the data formats for SFT, DPO, PPO, and GRPO. It simplifies the ingest process for RL training, allowing students to focus on algorithms rather than data parsing.
*   **Lilac:** A tool for exploring and clustering data to find patterns of contamination or low quality (e.g., "garbage" text, duplicates). It visually maps data manifolds, helping identify clusters of low-quality or repetitive samples.
*   **NeMo Curator:** NVIDIA's tool for large-scale data curation, including scalable deduplication and filtering, useful for processing datasets like OpenMathReasoning.

## 8. Conclusion

The transition to reasoning-focused LLMs necessitates a paradigm shift in data curation. We are moving away from the passive ingestion of web text toward the active cultivation of reasoning processes.

A robust curriculum on RL for Reasoning must center on three pillars:

1.  **Synthetic Autonomy:** Leveraging algorithms like STaR, Quiet-STaR, and self-correction (SCoRe) to generate data that outpaces human annotation capacity.
2.  **Verifiable Rigorousness:** Prioritizing domains (Math/Code) where reward signals are objective, enabling stable RL training (GRPO) without the noise of human preference.
3.  **Process Supervision:** Training models not just to answer, but to think, using datasets labeled at the step level (PRMs) to provide dense, actionable feedback.

> The future of AI reasoning lies not in bigger datasets, but in better data—data that captures the invisible steps of logic, deduction, and verification that constitute intelligence. By mastering the landscape of datasets from GSM8K to NuminaMath, and the tools to curate them, students will be equipped to build the next generation of "System 2" models.

## Appendix: Key Dataset Specifications Table

| Dataset Name | Domain | Est. Size | Format | Primary License | Source |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **GSM8K** | Math (Basic) | 8.5K | Q&A (CoT) | MIT | OpenAI |
| **MATH** | Math (Adv) | 12.5K | Q&A (LaTeX) | MIT | Hendrycks/UC Berkeley |
| **NuminaMath** | Math/Code | 860K | CoT & TIR | Apache 2.0 | Project Numina (DeepSeek) |
| **OpenMathReasoning** | Math | 540K+ | CoT & TIR | CC-BY 4.0 | NVIDIA |
| **PRM800K** | Math | 800K steps | Step-labeled | MIT | OpenAI |
| **HumanEval** | Code | 164 | Function+Tests | MIT | OpenAI |
| **OpenThoughts** | General/Math | 114K | CoT | Apache 2.0 | OpenThoughts Team |
| **ARC-AGI** | Abstract | 800 tasks | Grid JSON | Apache 2.0 | François Chollet |
| **NaturalReasoning** | General | 2.8M | Q&A | CC-BY-NC 4.0 | Meta (Facebook) |

Export to Sheets

## Appendix: RL Data Formats (JSONL Examples)

### 1. GRPO / PPO (Prompt Only with Ground Truth)

```json
{
  "prompt": "Calculate the derivative of x^2.",
  "answer": "2x"
}
```

Used for online generation where the model produces the trace and the reward function verifies against "answer".

### 2. SFT "Cold Start" (Reasoning Trace)

```json
{
  "messages":
}
```

Used to fine-tune the model to adopt the reasoning format before RL.

### 3. Preference Data (DPO)

```json
{
  "prompt": "Calculate the derivative of x^2.",
  "chosen": "The derivative is 2x because of the power rule...",
  "rejected": "The derivative is x because..."
}
```

Used for offline preference optimization.