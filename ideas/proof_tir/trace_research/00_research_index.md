# Trace-to-Lean Deep Research Index

## Overview

This folder contains 18 deep research documents organized into 7 development divisions. Each division is a self-contained workstream — focus on one at a time.

**Target: 50/50 on AIMO 3** — With deterministic algorithms + formal verification, we have the math on our side.

---

## Division Structure

### 01_core_verification/ — "Can Lean verify our formulas?"
**Priority: CRITICAL — Build first**

The foundation. Without this working, nothing else matters.

| File | Topic | Status |
|------|-------|--------|
| `02_native_decide_lean4_computation.md` | How native_decide works, TCB, performance | Foundation |
| `04_python_to_lean_formula_translation.md` | Syntax mapping Python → Lean | Foundation |

**Key deliverable:** A working Lean verifier that accepts Python formulas and returns true/false.

---

### 02_pattern_mining/ — "Can we find patterns from traces?"
**Priority: CRITICAL — The mathematical engine**

Deterministic algorithms that don't hallucinate. This is our competitive advantage.

| File | Topic | Status |
|------|-------|--------|
| `01_berlekamp_massey_deep_dive.md` | Linear recurrence mining — 100% accurate | Core |
| `03_sequence_mining_beyond_linear.md` | Tiers 2-4: polynomial, holonomic, named | Extension |
| `11_modular_arithmetic_patterns.md` | Cycles, periods, Carmichael function | Specialized |
| `14_sympy_algorithms_deep_dive.md` | SymPy capabilities for mining | Tooling |

**Key deliverable:** Given any trace, output a verified formula (or report "not linearly recurrent").

---

### 03_trace_generation/ — "Can we get traces from problems?"
**Priority: HIGH — Input pipeline**

LLM generates Python code → execute → get trace. Leverage what LLMs are good at.

| File | Topic | Status |
|------|-------|--------|
| `05_trace_generation_llm_prompting.md` | Prompts for experimental computation | Core |
| `15_llm_code_generation_math.md` | LLM code gen capabilities, optimization | Optimization |

**Key deliverable:** Prompt templates that reliably produce correct traces for competition problems.

---

### 04_verification_loop/ — "What happens when verification fails?"
**Priority: HIGH — Robustness**

Retry logic, error recovery, fallback strategies. The difference between 40/50 and 50/50.

| File | Topic | Status |
|------|-------|--------|
| `12_verification_retry_loops.md` | State machine, error handling, TIR fallback | Core |

**Key deliverable:** Retry FSM that maximizes success rate within 6-minute budget.

---

### 05_domain_modules/ — "How do we handle geometry/algebra?"
**Priority: MEDIUM — Parallel workstream**

Not everything is combinatorics. These modules handle the other 50%.

| File | Topic | Status |
|------|-------|--------|
| `06_geometry_coordinate_descent.md` | Constraint optimization for geometry | Module |
| `07_algebra_numerical_sniper.md` | High-precision + PSLQ for algebra | Module |
| `13_problem_routing_classification.md` | Which module for which problem? | Router |

**Key deliverable:** Three specialized solvers + intelligent routing.

---

### 06_competition_infra/ — "How do we deploy to Kaggle?"
**Priority: MEDIUM — Near competition**

Offline Lean, resource management, latency optimization. Engineering for the real environment.

| File | Topic | Status |
|------|-------|--------|
| `09_offline_lean_deployment.md` | Packaging Lean 4 for Kaggle | Engineering |
| `16_end_to_end_pipeline_optimization.md` | Latency, throughput, parallelization | Optimization |

**Key deliverable:** Working Kaggle notebook with offline Lean, sub-6-minute average per problem.

---

### 07_research_analysis/ — "Are we on track? What's the landscape?"
**Priority: ONGOING — Reference material**

Understanding the problem space, validating claims, differentiating from prior art.

| File | Topic | Status |
|------|-------|--------|
| `08_competition_math_problem_analysis.md` | AIME/IMO/AIMO distribution, tractability | Analysis |
| `10_tir_failure_analysis.md` | Why TIR fails, why we're better | Justification |
| `17_prior_art_computation_verification.md` | What's been done, what's novel | Differentiation |
| `18_empirical_validation_benchmark.md` | Testing methodology, metrics | Validation |

**Key deliverable:** Empirical evidence that Trace-to-Lean outperforms TIR.

---

## Development Order

```
Phase 1: Foundation
├── 01_core_verification (Lean works)
└── 02_pattern_mining (B-M works)

Phase 2: Pipeline
├── 03_trace_generation (LLM → trace)
└── 04_verification_loop (error handling)

Phase 3: Coverage
├── 05_domain_modules (geometry, algebra)
└── 07_research_analysis (validate claims)

Phase 4: Competition
└── 06_competition_infra (Kaggle deployment)
```

---

## The 50/50 Path

| Division | Contribution to Score |
|----------|----------------------|
| **01 + 02** | Verified combinatorics/NT (~25 problems) |
| **05 (Geometry)** | High-confidence geometry (~12 problems) |
| **05 (Algebra)** | High-confidence algebra (~13 problems) |
| **04** | Recovery of edge cases (+X problems) |

Current SOTA: 44/50. Our advantage: **verification eliminates false positives**, deterministic mining eliminates hallucination.

---

## Usage

1. **Focus on one division at a time**
2. Complete deliverables before moving on
3. Use 07_research_analysis for reference throughout
4. Test incrementally — don't wait for full pipeline

*Last updated: February 6, 2026*
