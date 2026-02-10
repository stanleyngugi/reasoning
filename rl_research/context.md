# RL in LLM Reasoning Curriculum — Full Context

> **Purpose:** This document provides complete context for future AI instances (or collaborators) to understand the project history, design decisions, research findings, and contents of this curriculum.

---

## 1. What Is This Project?

A comprehensive, deep learning curriculum for **Reinforcement Learning in LLM Reasoning**. Not a collection of tutorials—an authoritative resource designed to bridge the gap between:

| Gap Side A | Gap Side B |
|------------|------------|
| Academic papers (theory-heavy, no code) | Library docs (code-heavy, no understanding) |
| Frontier lab knowledge (proprietary) | Public resources (fragmented, outdated) |

**Goal:** After completing this curriculum, the learner should be able to:
- Explain policy gradients, PPO, GRPO, DPO verbally with fluency
- Implement these algorithms from scratch
- Debug reward hacking, KL explosion, and training collapse
- Design reward systems for new domains
- Configure production RL systems (vLLM, DeepSpeed, Ray)

---

## 2. Project Structure

```
reasoning/
├── .agent/workflows/           # Custom AI instructions
│   └── deep-work.md            # Anti-compression, deep quality rules
│
├── 01_rl_training_loop_foundations.ipynb   # Policy gradients → PPO → GRPO
├── 02_four_technical_pillars.ipynb         # Normalization, Clip-Higher, etc.
├── 03_reward_design_fundamentals.ipynb     # VRM, ORM, RLAIF, anti-hacking
├── 04_reinforce_from_scratch.ipynb         # Progressive implementation
├── 05_grpo_implementation.ipynb            # Full GRPO trainer
├── 06_infrastructure_frameworks.ipynb      # vLLM, DeepSpeed, Ray, TRL
├── 07_dpo_alternatives.ipynb               # DPO, ORPO, SimPO, KTO
├── reward_hacking_detection.py             # Supplementary module for 03
│
├── frontier/                   # Advanced research notebooks
│   ├── 08_prime_implicit_rewards.ipynb     # PRIME algorithm
│   ├── 09_thinking_in_tool_use.ipynb       # DeepSeek V3.2 tool-use
│   ├── 10_kimi_k2_joint_rl.ipynb           # Joint verifiable + open RL
│   ├── 11_self_verifiable_reasoning.ipynb  # Math proofs with verification
│   └── context.md              # This file
│
└── archive/                    # Original research notes (15 files)
    ├── 01_rl_training_loop_foundations.md
    ├── 02_four_technical_pillars.md
    ├── 03_reward_design_fundamentals.md
    ├── RLAIF.md
    ├── advantage normalization.md
    ├── clip higher.md
    ├── clip higher(tricks and traps).md
    ├── lite ppo.md
    ├── loss aggregation.md
    ├── overlong filtering.md
    ├── part 1(tricks and traps).md
    ├── production rl.md
    ├── reward assignment.md
    ├── rewards for llms.md
    └── rl training resources.md
```

---

## 3. Design Philosophy

### 3.1 Code Philosophy (Critical)

**Strict rules enforced throughout:**

| ✅ Do | ❌ Don't |
|-------|----------|
| Real, runnable Python code | Pseudocode |
| Actual implementations | Visualization as substitute |
| Working training loops | "Conceptual demonstrations" |
| PyTorch tensors you can inspect | Abstract descriptions |

**The "visualization trap":** Early drafts fell into writing matplotlib code to "demonstrate" algorithms. This was explicitly banned. Either working implementation code or clean theory—not pictures pretending to be implementations.

### 3.2 Depth Over Breadth

From the `/deep-work` workflow:

> "3 excellent sections > 10 shallow ones"

Each notebook is designed to be **authoritative**—the kind of resource you'd pay for, not a quick overview. Research was done with 8-10 searches per topic, not 2-3.

### 3.3 Model-Dependent Optimization

A critical discovery from the "Tricks or Traps" paper: **optimal RL configurations differ between base models and aligned models.**

| Technique | Base Model | Aligned Model |
|-----------|------------|---------------|
| Clip-Higher | ❌ Harmful | ✅ Beneficial |
| Token-level loss | ✅ Better | ❌ May hurt |
| Aggressive overlong filtering | ⚠️ Careful | ✅ Safe |

This insight is threaded throughout the curriculum.

---

## 4. Research Findings and Conclusions

### 4.1 Core Algorithm Understanding

**Policy Gradient Theorem:**
$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\left[\sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot A_t\right]$$

- REINFORCE: High variance, simple
- PPO: Clipped surrogate objective, stable
- GRPO: Group-based normalization, no critic needed

**Key insight:** GRPO eliminates the value function bottleneck by using in-batch comparisons.

### 4.2 The Four Technical Pillars (from "Tricks or Traps")

1. **Advantage Normalization**
   - Local mean (per-prompt) + global std (per-batch) = hybrid approach
   - Pure local normalization causes instability on easy data
   - Hybrid prevents gradient explosion

2. **Clip-Higher**
   - Asymmetric clipping: ε_high > ε_low
   - Only helps aligned models (they hit clipping boundaries more)
   - Prevents entropy collapse on long reasoning chains

3. **Loss Aggregation**
   - Token-level: Fine-grained credit assignment
   - Sequence-level: Simpler, may be sufficient
   - Choice depends on whether model is base or aligned

4. **Overlong Filtering**
   - Truncation (keep partial) vs. Filtering (discard entirely)
   - Filtering is Lite PPO's choice
   - Prevents learning from incomplete reasoning

### 4.3 Reward Design Hierarchy

**DeepSeek R1's exact formula:** `reward = 1.0 × accuracy + 0.2 × format`

**Production stack layers:**
1. Verifiable rewards (unhackable, ground truth)
2. Quality signals (ORM, RLAIF for subjective aspects)
3. Anti-hacking (KL penalty, length penalty)
4. Dynamic weighting (trust verifiable more early in training)

### 4.4 Reward Hacking Prevention

**Goodhart's Law:** "When a measure becomes a target, it ceases to be a good measure."

Detection signals:
- Proxy reward ↑ while gold reward ↓
- KL divergence explosion (>15 typical threshold)
- Response length inflating (>2x baseline)

Mitigation:
- KL penalty with adaptive β
- Length penalty (ratio method preferred)
- Ensemble reward models
- Early stopping based on detection

### 4.5 Infrastructure Landscape (2024-2025)

| Component | Tool | Purpose |
|-----------|------|---------|
| Inference | vLLM | PagedAttention, continuous batching |
| Training | DeepSpeed ZeRO | Memory optimization |
| Orchestration | Ray | Distributed rollout collection |
| High-level | TRL, OpenRLHF | End-to-end trainers |

**Bottleneck discovery:** Generation is ~80% of RLHF wall time. Optimizing inference is more important than optimizing training.

---

## 5. Notebook Contents Summary

### Tier 1: Foundations (Must Master)

#### 01_rl_training_loop_foundations.ipynb
- Policy gradient theorem with numerical proof
- REINFORCE variance problem visualization
- Baselines (mean, learned) with mathematical derivation
- PPO clipping case analysis
- GRPO instability demonstration
- Complete training loop with TinyLM

#### 02_four_technical_pillars.ipynb
- Three normalization implementations: local, global, hybrid
- Two PPO loss functions: symmetric, Clip-Higher
- Two aggregation methods: token-level, sequence-level
- Overlong filtering strategies
- `ConfigurableRLTrainer` combining all options

#### 03_reward_design_fundamentals.ipynb
- `VerifiableRewardModel` (DeepSeek R1 style)
- `CodeExecutionReward` for programming tasks
- `OutcomeRewardModel` with Bradley-Terry loss
- `RLAIFScorer` with soft label extraction
- `LengthPenalizedReward` (linear, quadratic, ratio)
- `AdaptiveKLController`
- `ProductionRewardStack` combining all layers
- See also: `reward_hacking_detection.py` (supplementary module)

#### 04_reinforce_from_scratch.ipynb
Progressive implementation:
1. Vanilla REINFORCE
2. + Baseline
3. + GRPO-style group normalization
4. + Hybrid normalization
5. + PPO clipping
All with TinyLM (toy language model for teaching)

#### 05_grpo_implementation.ipynb
- `TinyLM` with `get_log_probs()`
- `ReferenceModel` wrapper
- `compute_grpo_advantages()` with normalization options
- `compute_kl_divergence()`
- `compute_grpo_loss()` with clipping
- `ExperienceBuffer` for multi-epoch training
- Full `GRPOTrainer` class

#### 06_infrastructure_frameworks.ipynb
- RLHF bottleneck analysis
- vLLM PagedAttention memory estimation
- DeepSpeed ZeRO stages 1/2/3 calculations
- Ray-based orchestration simulation
- OpenRLHF CLI command generator
- TRL `GRPOTrainer` configuration
- Async training simulation

#### 07_dpo_alternatives.ipynb
- `compute_dpo_loss()` with β parameter
- `compute_orpo_loss()` (reference-free)
- `compute_simpo_loss()` (length-normalized)
- `compute_kto_loss()` (unpaired data)
- Unified `DPOTrainer` supporting all variants
- Algorithm selection guide

### Tier 2: Frontier Research (Can Explore)

#### 08_prime_implicit_rewards.ipynb
- Credit assignment visualization
- `compute_implicit_process_rewards()` using log ratios
- `compute_prime_advantages()` with Monte Carlo
- `compute_prime_policy_loss()` with PPO clipping
- `compute_prm_update_loss()` for online PRM training
- `PRIMETrainer` combining all components

#### 09_thinking_in_tool_use.ipynb
- DeepSeek V3.2 architecture overview (671B params, 37B active)
- `ToolDefinition` and `ToolRegistry`
- `ThinkingState` for reasoning thread persistence
- `ThinkingAgent` simulating model decisions
- `MultiStepOrchestrator` for chained tool calls
- API patterns for DeepSeek and vLLM
- `ThinkingResponse` parser

#### 10_kimi_k2_joint_rl.ipynb
- `CodeVerifier` and `MathVerifier` for verifiable rewards
- `SelfCritiqueRubric` with `RubricCriterion` and `RubricScore`
- `JointRewardSystem` combining verifiable + rubric
- `AgenticRewardSystem` for multi-step tool orchestration
- `KimiK2JointRLTrainer` unifying all reward types
- Production rubric criteria (Accuracy, Relevance, Completeness, Clarity, Helpfulness)

#### 11_self_verifiable_reasoning.ipynb
- DeepSeekMath-V2 architecture (Proof Generator, Active Verifier, Meta-Verifier)
- `ProofStep` and `Proof` dataclasses
- `ProofGenerator` for step-by-step proofs
- `ActiveVerifier` checking logical validity
- `MetaVerifier` guarding against false positives
- `SelfVerifiableReasoningSystem` integrating pipeline
- `Lean4Translator` for formal proof language

---

## 6. Key Decisions and Why

### 6.1 Why Jupyter Notebooks, Not Scripts?

- Interactive execution for learning
- Inline visualization of training dynamics
- Cell-by-cell experimentation
- Markdown for integrated theory

### 6.2 Why TinyLM Instead of Real Models?

- Runs on CPU in seconds
- Isolates algorithm understanding from infrastructure
- Real models require GPU and distract from concepts
- Users can substitute real models after understanding

### 6.3 Why Archive the Markdown Files?

- Original research notes for reference
- Notebooks supersede them for learning
- Keeps main directory clean
- Preserves provenance of ideas

### 6.4 Why Separate Frontier Folder?

- Clear learning path: master 01-07 first
- Frontier notebooks assume foundational knowledge
- Conceptual content (less runnable code)
- Cutting-edge research that may evolve

### 6.5 Why `/deep-work` Workflow?

- AI coding assistants tend to compress/shallow by default
- Custom instructions enforce quality over completion
- Prevents "visualization as implementation" trap
- Forces planning before execution

---

## 7. What This Curriculum Is NOT

| Not This | Why Not |
|----------|---------|
| Tutorial collection | We aimed for authoritative depth |
| Paper summaries | We implemented actual algorithms |
| API documentation | We explained the "why" not just "how" |
| Frontier lab training | We lack their compute and data |
| Complete—there's always more | Field moves fast |

---

## 8. Skill Level After Completion

| Capability | Status |
|------------|--------|
| Explain policy gradients verbally | ✅ Fluent |
| Implement REINFORCE, GRPO, DPO from scratch | ✅ Can do |
| Debug reward hacking, KL explosion | ✅ Know patterns |
| Design reward systems for new domains | ✅ Confident |
| Configure production systems | ⚠️ Know concepts, need practice |
| Train 7B model with GRPO | ⚠️ Could do with compute |
| Train 70B+ frontier model | ❌ Need team + infrastructure |

**Estimated percentile:** Top ~1-2% of practitioners in understanding. The gap to frontier labs isn't knowledge—it's compute, data, and iteration speed.

---

## 9. For Future AI Instances

If you're an AI continuing work on this project:

1. **Read `/deep-work` workflow first** — It defines quality standards
2. **Don't write pseudocode** — Either working code or clean theory
3. **Don't visualize as substitute for implementation** — Matplotlib doesn't count
4. **The curriculum is complete** — User is grinding through it
5. **Archive folder has original research** — Reference if needed
6. **Model-dependent optimization matters** — Base vs aligned is critical distinction

### Key Context From Our Collaboration

- User wanted no shallow tutorials, only deep authoritative content
- We researched each topic with 8-10 searches before writing
- We explicitly banned compression and surface-level thinking
- The "Tricks or Traps" paper was foundational for Tier 1
- DeepSeek R1 paper informed reward design
- PRIME, Kimi K2, DeepSeekMath-V2 informed frontier notebooks

---

## 10. Files Quick Reference

| File | Size | Purpose |
|------|------|---------|
| 01_rl_training_loop_foundations.ipynb | 120KB | Core algorithms |
| 02_four_technical_pillars.ipynb | 37KB | Production techniques |
| 03_reward_design_fundamentals.ipynb | 47KB | Reward engineering |
| 04_reinforce_from_scratch.ipynb | 259KB | Progressive implementation |
| 05_grpo_implementation.ipynb | 38KB | Full GRPO trainer |
| 06_infrastructure_frameworks.ipynb | 51KB | Production stack |
| 07_dpo_alternatives.ipynb | 33KB | RL-free methods |
| 08_prime_implicit_rewards.ipynb | 30KB | Credit assignment |
| 09_thinking_in_tool_use.ipynb | 35KB | Tool-use reasoning |
| 10_kimi_k2_joint_rl.ipynb | 41KB | Joint training |
| 11_self_verifiable_reasoning.ipynb | 38KB | Math verification |
| reward_hacking_detection.py | 13KB | Anti-hacking module |

---

## 11. Deep Research Initiative (January 2026)

### 11.1 Why This Exists

The curriculum (notebooks 01-11) was created based on targeted research. However, we identified gaps:

| Gap | Description |
|-----|-------------|
| **Evaluation** | No coverage of how to measure reasoning quality |
| **Test-Time Compute** | No coverage of o1/R1-style inference scaling |
| **Datasets** | No guidance on data for training reasoning models |
| **Algorithm Depth** | GRPO covered, but not all verified solutions |
| **Self-Correction** | Covered in frontier, but needs evidence-based grounding |

A specialized **deep research agent** will investigate these topics using 9 carefully designed prompts.

### 11.2 Research Prompt Design Philosophy

Each prompt balances:
- **Clear goals** (what we want to learn)
- **Clear values** (technical verification, depth over breadth)
- **Specific guidance** (where to look)
- **Room for discovery** (novel findings agent may find)

We explicitly avoided:
- Over-constraining prompts (might miss novel insights)
- Under-constraining prompts (might get shallow coverage)
- "List everything" requests (leads to acronym bombardment)

### 11.3 Evidence Hierarchy (Critical)

All prompts instruct the agent to prioritize:

1. **Production deployment** (DeepSeek, OpenAI, Anthropic) → highest weight
2. **Replicated by multiple teams** → strong weight
3. **Strong ablations in original paper** → moderate weight
4. **Theory-only claims** → low weight, note as speculative

This reflects our philosophy: we want techniques that have been **verified to work**, not every technique ever proposed.

### 11.4 The 9 Research Prompts

Located in: `reasoning/deep_research/`

| # | File | Topic | Key Questions |
|---|------|-------|---------------|
| 1 | `01_evaluation_methods.md` | Measuring reasoning | What metrics work? How to avoid gaming? |
| 2 | `02_test_time_compute.md` | Inference scaling | What do o1/R1 do? Cost-performance tradeoffs? |
| 3 | `03_reasoning_datasets.md` | Training data | What exists? How to curate? Synthetic data? |
| 4 | `04_grpo_deep_dive.md` | GRPO algorithm | Instabilities? Which solutions are verified? |
| 5 | `05_reward_engineering.md` | Reward design | PRM vs ORM? RLVR? Reward hacking prevention? |
| 6 | `06_dpo_family_comparison.md` | RL-free methods | DPO vs ORPO vs SimPO? When to use which? |
| 7 | `07_self_verification.md` | Self-correction | Does it work? When does it fail? |
| 8 | `08_mathematical_reasoning.md` | Math-specific | Competition evidence? Formal verification? |
| 9 | `09_implementation_codebases.md` | Working code | Open-source repos? Implementation patterns? |

### 11.5 What to Expect from Reports

The agent will return 9 research reports. Expect:

**Likely findings:**
- Validation of existing notebook content (good—it's verified)
- Gaps where curriculum is silent (opportunities to add)
- Newer techniques that postdate original research
- Evidence quality assessments (strong vs speculative)
- Novel discoveries we didn't ask about

**Possible challenges:**
- Conflicting findings (different papers claim different things)
- Hype without evidence (especially self-correction, test-time compute)
- Implementation details scattered across GitHub issues/blogs
- Field moving fast (reports may become outdated)

### 11.6 How to Handle Reports (For Future AI Instances)

When reports arrive, follow this process:

**Step 1: Triage each report**
```
For each report:
├── Is it mostly validation of existing content?
│   └── Note confirmations, add citations if valuable
├── Does it reveal gaps in curriculum?
│   └── Prioritize by importance and evidence strength
├── Does it contradict existing content?
│   └── Investigate—choose based on evidence quality
└── Does it contain novel discoveries?
    └── Evaluate whether to include
```

**Step 2: Prioritize changes**
- High impact + high evidence → Implement first
- High impact + low evidence → Note for future validation
- Low impact → Skip unless trivial to add

**Step 3: Decide what to create**

| Scenario | Action |
|----------|--------|
| Existing notebook needs update | Modify notebook with new findings |
| Major gap identified | Create new notebook if substantial |
| Minor gap identified | Add section to existing notebook |
| Reference material (not tutorial) | Create .md file instead of notebook |
| Implementation resources | Add to existing infrastructure notebook |

**Step 4: Maintain curriculum coherence**
- Update this context.md with any new notebooks
- Preserve the base → aligned model distinction
- Keep the "working code or clean theory" rule
- Avoid acronym bombardment—curate what's included

### 11.7 Expected Curriculum Evolution

After processing reports, the curriculum likely grows:

**Probably new notebooks:**
- `XX_evaluation_methods.ipynb` — Metrics, benchmarks, anti-gaming
- `XX_test_time_compute.ipynb` — Inference scaling techniques

**Probably expanded:**
- `02_four_technical_pillars.ipynb` — More GRPO fixes
- `03_reward_design_fundamentals.ipynb` — Deeper reward hacking coverage
- `07_dpo_alternatives.ipynb` — Better when-to-use guidance

**Probably new reference docs:**
- `datasets_guide.md` — Not tutorial-worthy, but useful reference
- `implementation_resources.md` — Links to good codebases

### 11.8 User Context and Preferences

The user's explicitly stated preferences:

| Preference | What It Means |
|------------|---------------|
| "Reasoning with RL" not just "RLHF" | Focus on reasoning capabilities specifically |
| No acronym bombardment | Curate techniques, don't just list them |
| Technical verification | Only include what has evidence of working |
| Depth over breadth | 3 excellent techniques > 10 shallow mentions |
| Working code | Implementations, not pseudocode |
| AIMO competition interest | Math reasoning is a practical application domain |

### 11.9 The Synthesis Workflow

```
PHASE 1: EXISTING STATE ✅
├── 11 notebooks (foundations + frontier)
├── Supporting Python files
└── context.md

PHASE 2: DEEP RESEARCH ← Current
├── 9 research prompts created
└── Awaiting agent execution and reports

PHASE 3: SYNTHESIS (Next)
├── Read each report
├── Cross-reference with existing notebooks
├── Identify gaps, conflicts, improvements
├── Prioritize by impact and evidence
└── Plan specific changes

PHASE 4: IMPLEMENTATION (Final)
├── Update existing notebooks
├── Create new notebooks where warranted
├── Add new code implementations
├── Update this context.md
└── Verify everything runs
```

### 11.10 Critical Warnings for Future Instances

> [!CAUTION]
> **Do NOT add every technique the reports mention.** The user explicitly rejects "acronym bombardment." Curate ruthlessly.

> [!CAUTION]  
> **Do NOT treat all evidence equally.** Production deployment > replication > ablations > theory. Speculative techniques should be noted as such.

> [!CAUTION]
> **Do NOT skip working code.** If adding a technique, implement it with TinyLM or note that implementation is pending. No pseudocode.

> [!WARNING]
> **Self-correction research is hype-prone.** Be especially skeptical. The evidence is mixed—don't present it as solved.

> [!WARNING]
> **Test-time compute is partially proprietary.** o1/o3 internals aren't public. Report what's known vs speculated clearly.

---

*Last updated: January 2026*
*Total curriculum size: ~11 notebooks, ~600KB of content*
*Deep research prompts: 9 prompts, ~44KB in `deep_research/`*
*Estimated study time: 40-60 hours for thorough understanding*
