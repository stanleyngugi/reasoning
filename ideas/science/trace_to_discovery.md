# The Inverted Arrow: Constraint-Based Scientific Discovery

> **The Inverse Trick**: Instead of humans proposing hypotheses for machines to verify, we invert the arrow—humans provide constraints, machines generate discoveries that satisfy them.
>
> **The Trace Trick**: This inversion becomes practical when experimental data serves as "traces" that implicitly define constraints, ground the search in reality, and enable self-correcting verification loops.

---

## Executive Summary

This document proposes a paradigm shift in scientific discovery. The core insight is **the inverse trick**—flipping the direction of the scientific method from "generate hypothesis → verify" to "specify constraints → synthesize discovery."

The inverse trick has been proven in multiple domains: program synthesis (CEGIS), quantum circuits (ZX-calculus), protein structure (AlphaFold), materials discovery (GNoME). But applying it to general science has seemed impossible because "you can't formalize 120 million papers."

**The trace trick solves this.** Experimental data serves as implicit constraints. LLMs extract approximate formalizations. Solvers verify consistency. The loop self-corrects. You never needed to formalize all of science—you needed to recognize it's a trace problem.

**The synthesis**: Scientific discovery becomes compilation. Specify what must be true (physics, chemistry). Provide what is observed (experimental traces). Let solvers generate the theory that satisfies both.

---

## Table of Contents

1. [The Inverse Trick: The Paradigm Shift](#1-the-inverse-trick-the-paradigm-shift)
2. [The Formalization Wall: Why It Seemed Impossible](#2-the-formalization-wall-why-it-seemed-impossible)
3. [The Trace Trick: Breaking Through](#3-the-trace-trick-breaking-through)
4. [The Architecture: Trace-to-Discovery](#4-the-architecture-trace-to-discovery)
5. [The Hacks: Practical Techniques](#5-the-hacks-practical-techniques)
6. [Existence Proofs: The Inverse Trick in Action](#6-existence-proofs-the-inverse-trick-in-action)
7. [The Infrastructure: What Exists Today](#7-the-infrastructure-what-exists-today)
8. [The Roadmap: From Insight to Impact](#8-the-roadmap-from-insight-to-impact)
9. [The Hard Questions](#9-the-hard-questions)
10. [The Vision: Science as Compilation](#10-the-vision-science-as-compilation)

---

## 1. The Inverse Trick: The Paradigm Shift

### The Traditional Arrow

For 400 years, science has followed this pattern:

```
Human Insight → Hypothesis → Prediction → Experiment → Verification
       ↓
  (THIS IS HARD)
```

The bottleneck is **hypothesis generation**. A human must have the insight. The machine merely verifies. Science progresses at the speed of human creativity.

### The Inverted Arrow

What if we flip it?

```
Constraints + Observations → Solver → Theory
                ↓
          (THIS IS TRACTABLE)
```

Instead of generating hypotheses and checking them:
1. **Specify constraints** (what must be true: conservation laws, symmetries, dimensional consistency)
2. **Provide observations** (experimental data the theory must explain)
3. **Let a solver find** the theory that satisfies all constraints and fits all observations

### Why the Inversion Works: The Generation-Verification Gap

**Core insight**: Checking if a solution satisfies constraints is computationally easier than generating the solution.

This is the P ≠ NP intuition applied to science:
- **Hard**: Propose a theory that explains all observations
- **Easy**: Check if a proposed theory is consistent with observations

If verification is fast, use it to guide generation. Let machines propose, let solvers dispose.

### Where the Inverse Trick Is Already Proven

| Domain | Forward (Traditional) | Inverted | Result |
|--------|----------------------|----------|--------|
| **Program Synthesis** | Human writes code | Spec → solver → code (CEGIS) | 10-100x productivity |
| **Quantum Circuits** | Human designs gates | Constraints → ZX-calculus → circuit | 50% gate reduction |
| **Protein Structure** | Simulate folding | Physics + evolution → structure (AlphaFold) | Solved the problem |
| **Materials** | Synthesize and test | Stability constraints → candidates (GNoME) | 2.2M new materials |
| **Drug Design** | Medicinal chemistry | Binding + ADMET → molecules | Accelerating |
| **Error Correction** | Design codes by hand | Knill-Laflamme → RL discovers codes | Novel codes found |

**The pattern is general.** Whenever you can verify more easily than generate, invert the arrow.

### The Implication for Science

Science is the ultimate generation problem. Generating correct theories about nature is what humanity has struggled with for millennia.

But **verifying** that a theory matches observations? That's tractable. Run the experiment. Compare predictions. Check consistency.

**The inverse trick for science**: Specify the constraints (laws the theory must obey) and observations (data it must explain). Let computation synthesize the theory.

---

## 2. The Formalization Wall: Why It Seemed Impossible

### The Obvious Objection

> "To use solvers, you need formal constraints. You can't formalize 120 million scientific papers."

This is why nobody has done it. The formalization wall stops everyone.

The apparent requirements:
- Formalize all of physics, chemistry, biology in first-order logic
- Convert all papers to SMT-Lib constraints
- Build a complete ontology of scientific concepts
- Handle inconsistent, evolving, fuzzy knowledge

This looks like a decade-long, billion-dollar project. Maybe impossible.

### The Specification Problem

There's a deeper issue. In software verification:
> "Writing the specification is half the work."

In science:
> "Writing the specification IS the discovery."

If we knew exactly what constraints the correct theory must satisfy, we'd already have the theory. The constraints ARE the knowledge.

### Why Everyone Got Stuck

The traditional approach:
1. Formalize scientific knowledge (impossible scale)
2. Apply formal methods (now tractable)
3. Generate discoveries

Everyone stalls at step 1. The wall is too high.

**But this framing is wrong.** The inverse trick doesn't require perfect upfront formalization. It requires a different approach entirely.

---

## 3. The Trace Trick: Breaking Through

### The Breakthrough

Here's what makes the inverse trick practical for science:

**You don't need to formalize science upfront. You need experimental data as ground truth, and you let the verification loop do the formalization incrementally.**

This is exactly how self-verifying code agents work. The pattern transfers directly.

### What Are Traces?

In software verification:
- **Trace** = execution record of what code actually does on specific inputs
- **Invariant** = property inferred from traces that should hold for ALL inputs
- **Verification** = formal proof that invariant generalizes beyond observed traces

In scientific discovery:
- **Trace** = experimental measurement, database record, observed phenomenon
- **Constraint** = scientific law inferred from traces that should hold universally
- **Verification** = formal check that constraint is consistent with ALL data

**Experimental data IS the trace. The trace grounds the system in reality.**

### The Mapping

| Software (T2V) | Science (Trace-to-Discovery) |
|----------------|------------------------------|
| Source code | Scientific theory |
| Test inputs | Experimental conditions |
| Execution trace | Experimental measurements |
| Invariant | Scientific law |
| SMT solver | Consistency checker |
| Counterexample | Falsifying observation |
| CEGIS loop | Hypothesis refinement |
| Verified program | Validated theory |

### What Traces Provide

1. **Implicit Constraints**: Every measurement is a constraint. "Under X, we observed Y." Any theory must explain this.

2. **Grounding for LLMs**: LLMs can infer constraints from concrete data, not abstract specifications. This is what they're good at.

3. **Cheap Filtering**: Before expensive verification, check if proposed constraints match traces. Discard obvious failures immediately.

4. **Counterexamples**: When verification fails, traces show WHERE. This guides refinement.

5. **Self-Correction**: The loop converges because traces are ground truth. You can't drift into hallucination.

### The Self-Correcting Formalization Loop

```python
while not consistent:
    constraints = llm.extract(papers)           # ~80% accurate
    result = solver.check(constraints, traces)  # 100% rigorous
    if result == UNSAT:
        core = solver.get_unsat_core()          # What failed?
        constraints = llm.refine(constraints, core)
    elif result == SAT:
        return solver.get_model()               # Discovery!
```

**Even with 80% extraction accuracy, the loop converges.** The solver catches errors. The traces provide ground truth. Formalization happens as a byproduct.

### The Key Insight

Everyone was thinking:
> "How do we formalize scientific knowledge?"

The insight is:
> "You don't formalize upfront. You let traces ground the process and verification formalize incrementally."

**You never had to formalize all of science. You had to recognize it's a trace problem.**

---

## 4. The Architecture: Trace-to-Discovery

### The Full Loop

```
┌─────────────────────────────────────────────────────────────────┐
│                      THE INVERTED ARROW                          │
│           Constraints + Traces → Solver → Discovery              │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ 1. TRACE COLLECTION (Ground Truth)                              │
│                                                                  │
│    - Experimental databases (ChEMBL, PDB, Materials Project)    │
│    - Robotic lab experiments (Coscientist-style)                │
│    - Extracted data from papers (GROBID + LLM)                  │
│                                                                  │
│    Traces ARE implicit constraints.                              │
│    "Under X, we observed Y" constrains any valid theory.        │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. CONSTRAINT EXTRACTION (Approximate Formalization)            │
│                                                                  │
│    LLM reads papers → Extracts claims → Structures as logic     │
│                                                                  │
│    "Compound X inhibits receptor Y with IC50 < 100nM"           │
│    → inhibits(X, Y) ∧ IC50(X, Y) < 100                          │
│                                                                  │
│    Accuracy: ~80-99% (Elicit achieves 99.4%)                    │
│    Errors: Caught by verification loop                          │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. TRACE-BASED FILTERING (Cheap Pruning)                        │
│                                                                  │
│    Check proposed constraints against trace database             │
│    Discard anything that contradicts observed data               │
│    O(n) checking, not O(2^n) solving                            │
│                                                                  │
│    Traces prune search space before solvers touch it.            │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│ 4. FORMAL VERIFICATION (The Inverse Trick in Action)            │
│                                                                  │
│    SMT/MaxSAT Solver checks:                                     │
│    - Are all constraints mutually consistent?                    │
│    - Do they explain all traces?                                 │
│    - What's the minimal theory that satisfies everything?       │
│                                                                  │
│    SAT → Found consistent model                                  │
│    UNSAT → Found CONTRADICTION → Discovery site!                │
└─────────────────────────────────────────────────────────────────┘
                                │
                    ┌───────────┴───────────┐
                    ▼                       ▼
            ┌──────────────┐        ┌──────────────────┐
            │   CONSISTENT │        │  CONTRADICTION   │
            │              │        │                  │
            │  Best-fit    │        │  UNSAT core      │
            │  model found │        │  identifies the  │
            │              │        │  failing axiom   │
            └──────────────┘        └──────────────────┘
                    │                       │
                    ▼                       ▼
┌─────────────────────────────────────────────────────────────────┐
│ 5. DISCOVERY / REFINEMENT                                        │
│                                                                  │
│    Consistent: Model IS the discovery                            │
│                                                                  │
│    Contradiction:                                                │
│    - UNSAT core identifies WHICH constraints conflict            │
│    - "Paper A claims X, Paper B claims Y, Trace shows Z"        │
│    - PINPOINTS where current science is wrong                   │
│    - LLM proposes: revised theory, new experiment, resolution   │
│                                                                  │
│    THE PARADIGM SHIFT HACK:                                      │
│    Solver doesn't invent new physics, but it identifies         │
│    exactly WHERE current physics breaks.                        │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│ 6. ROBOTIC EXPERIMENT EXECUTION (New Trace Generation)          │
│                                                                  │
│    When contradiction cannot be resolved from existing data:     │
│    - Solver identifies what experiment would resolve it          │
│    - Robotic lab executes (Coscientist-style)                   │
│    - New trace added to database                                 │
│    - Loop continues                                              │
│                                                                  │
│    CLOSED LOOP: Discovery → Experiment → Trace → Verify         │
└─────────────────────────────────────────────────────────────────┘
```

### Why Contradictions ARE Discoveries

In traditional science, contradictions are problems.

In the inverted architecture, **contradictions are the discoveries**.

When the solver returns UNSAT:
- Current constraints cannot all be true
- UNSAT core identifies the MINIMAL conflicting set
- This pinpoints exactly which assumption is wrong

A human saying "that's weird" is intuition. A solver saying "Constraint 47 from Paper A contradicts Constraint 892 from Paper B given Trace 17" is **precision**.

---

## 5. The Hacks: Practical Techniques

### Hack 1: Inconsistency Mining

**The Idea**: Treat papers as data points, not truths. Build a constraint graph. Find logical friction.

**Implementation**:
```
For each paper:
    Extract claims as constraints
    Add to constraint graph
    
Run MaxSAT:
    - Find maximum consistent subset (weighted by citations)
    - Find minimal inconsistent cores
    
Each inconsistent core = candidate discovery site
```

**Why It Works**: ContraCrow finds ~2.34 contradictions per paper. The literature is full of discovery sites.

### Hack 2: Anomaly Synthesis (The Paradigm Shift Hack)

**The Idea**: When theory says IMPOSSIBLE (UNSAT), but we observe it, the solver has isolated the flaw.

**Implementation**:
```
Formalize current paradigm as constraints
Check against experimental traces
When UNSAT with observed trace:
    Extract UNSAT core
    Core = the broken assumptions
```

**Why It Works**: Solver doesn't invent new physics. But it tells you exactly where to look.

### Hack 3: Axiomatic Density Strategy

**The Idea**: Some fields are "dense" (few rules, much coverage). Others are "sparse."

| Dense (Formalize First) | Sparse (Derive Later) |
|-------------------------|----------------------|
| Quantum mechanics | Psychology |
| Thermodynamics | Sociology |
| Organic chemistry | Economics |

**Strategy**: Formalize the dense core. Compile sparse fields as derivations.

### Hack 4: The 80/100 Split

**The Idea**: LLMs extract at ~80% accuracy. Solvers verify with 100% rigor.

```
LLM extracts → Solver rejects → LLM refines → Solver accepts
```

The feedback loop handles errors. Perfect extraction not required.

### Hack 5: Null Space Mapping

**The Idea**: Map what current theories DON'T predict.

- Where discoveries are possible
- What experiments would be maximally informative
- The "unknown unknowns" made visible

### Hack 6: Conservation Law Discovery

**The Idea**: Every conserved quantity implies a symmetry (Noether).

```
Given time-series traces:
    Search for constant quantities
    Each implies a symmetry
    Symmetries constrain dynamics
```

### Hack 7: Structural Isomorphism

**The Idea**: Many discoveries recognize two domains share structure.

```
Heat flow ↔ Diffusion ↔ Random walks ↔ Option pricing
```

Formalize as categories. Find isomorphisms. Transfer solutions.

---

## 6. Existence Proofs: The Inverse Trick in Action

### 6.1 Program Synthesis (CEGIS)

**Inversion**: Spec → solver → code (instead of human writes code)

**Result**: Sketch, Rosette achieve 10-100x productivity gains.

### 6.2 AlphaFold

**Inversion**: Physics + evolution constraints → structure (instead of simulate folding)

**Result**: Solved protein structure. Nobel Prize 2024. 200M predictions.

### 6.3 GNoME

**Inversion**: Stability constraints → materials (instead of synthesize and test)

**Result**: 2.2M new stable crystals. 736 independently synthesized. 800 years equivalent.

### 6.4 Robot Scientist Adam (Science 2009)

**Inversion**: Metabolic constraints + experiment loop → discovery

**Result**: Discovered genes encoding orphan enzymes in yeast. Genuinely novel.

### 6.5 Robot Scientist Eve

**Inversion**: Drug target constraints + automated screening → discovery

**Result**: Found TNP-470 as antimalarial with 1000x selectivity.

### 6.6 Coscientist (Nature 2023)

**Inversion**: Chemical constraints + robotic execution → synthesis

**Result**: Closed loop. GPT-4 → Opentrons → GC-MS. Outperformed Bayesian optimization.

### 6.7 Symbolic Regression: Dark Matter (NeurIPS 2020)

**Inversion**: Dimensional constraints + data → equation

**Result**: Novel formula for dark matter halo concentration. Not previously known.

### 6.8 FutureHouse Robin (2025)

**Inversion**: Literature + experimental constraints → drug candidate

**Result**: First AI-generated discovery (ripasudil for dry AMD). 2.5 months.

### 6.9 ZX-Calculus for Quantum Circuits

**Inversion**: Completeness constraints → optimized circuit

**Result**: Up to 50% T-count reduction. RL discovers novel QEC codes.

---

## 7. The Infrastructure: What Exists Today

### 7.1 Trace Databases

| Domain | Database | Scale | Coverage |
|--------|----------|-------|----------|
| Structural Biology | PDB + AlphaFold | 249K + 200M | ~90% |
| Chemistry | ChEMBL | 2.4M compounds, 20M bioactivities | ~20-30% |
| Genomics | GenBank | 265B+ bases | ~80% |
| Materials | Materials Project | 150K+ | <10% |
| General | PDFs | 200M papers | **80-90% trapped** |

**Gap**: Most data is in unstructured PDFs.

### 7.2 Knowledge Extraction

| System | Accuracy | Formal Logic? |
|--------|----------|---------------|
| Elicit | 99.4% | No |
| PaperQA2 | Superhuman | No |
| ContraCrow | ~2.34 contradictions/paper | No |

**Gap**: Nobody extracts SMT-Lib constraints.

### 7.3 Robotic Labs

| Platform | Status |
|----------|--------|
| Emerald Cloud Lab | Production |
| Coscientist | Demonstrated |
| Berkeley A-Lab | Research |

**Status**: Closed loop demonstrated. Integration is engineering.

### 7.4 What's Missing

1. **NLP → SMT transpiler**: Paper to formal constraints
2. **Contradiction graph**: Unified inconsistency map
3. **Closed-loop integration**: Components exist separately
4. **Trace infrastructure**: Most science lacks structured data

---

## 8. The Roadmap: From Insight to Impact

### Phase 1: Narrow Domain Proof (6-12 months)

- Pick axiomatically dense domain (chemistry, materials)
- Build extraction pipeline
- Find genuine contradictions
- Human review validates

### Phase 2: Contradiction Mining (12-18 months)

- Scale to 100K+ papers
- Build contradiction graph
- MaxSAT identifies discovery sites
- Integrate robotic validation

### Phase 3: First Major Discovery (18-24 months)

- AI-discovered result in top journal
- Closed loop demonstrated
- Real scientific impact

### Phase 4: Platform (2-4 years)

- Verified constraint database
- API for consistency queries
- "Verification score" for papers
- External researchers using platform

---

## 9. The Hard Questions

### Can this discover genuinely new science?

**Yes, via contradictions.** When solver identifies inconsistency between claims and traces, that IS new knowledge. Something was wrong that we didn't know.

The system doesn't invent new concepts. But it identifies exactly where current concepts fail. That's 90% of insight.

### What about paradigm shifts?

**Partially.** System can identify WHERE paradigm fails (UNSAT core). Can't invent the new paradigm.

But ~95% of science is "normal science" within paradigms. That's tractable.

### Doesn't this require perfect formalization?

**No—that's the whole point.** The loop self-corrects. 80% accuracy + solver feedback = convergence.


---

## 10. The Vision: Science as Compilation

### The Paradigm Shift

| Era | Human Provides | Machine Does |
|-----|----------------|--------------|
| Science 1.0 | Hypotheses | Verification |
| Science 2.0 | Data | Prediction |
| **Science 3.0** | **Constraints + Traces** | **Discovery** |

### The Inversion Complete

```
OLD: Human proposes → Machine verifies
NEW: Human constrains → Machine synthesizes
```

### The Core Thesis

The inverse trick works when verification is easier than generation. For science:
- Generating theories is hard (millennia of human effort)
- Checking consistency with data is tractable (run the experiment)

The trace trick makes it practical:
- Traces provide ground truth
- LLMs extract approximate constraints
- Solvers verify and catch errors
- The loop converges

### The Bottom Line

The inverse trick is proven (AlphaFold, GNoME, CEGIS, Robot Scientists).

The trace trick makes it practical for general science.

The pieces exist. The integration is the opportunity.

**The universe has a source code. We have the decompiler (LLMs) and the CPU (solvers). The trace is the debugger that keeps us honest.**

---

## Appendix A: The One-Sentence Summary

**The inverse trick for science**: Instead of humans proposing hypotheses for machines to verify, specify constraints and observations, let solvers synthesize theories—using experimental traces as ground truth to make formalization tractable through self-correcting verification loops.

---

## Appendix B: Key References

### The Inverse Trick
1. Solar-Lezama (2008) - CEGIS
2. Kambhampati (2024) - LLM-Modulo Framework

### Existence Proofs
3. Jumper et al. (2021) - AlphaFold
4. Merchant et al. (2023) - GNoME
5. King et al. (2009) - Robot Scientist Adam
6. Boiko et al. (2023) - Coscientist
7. Cranmer et al. (2020) - Symbolic Distillation

### Formal Methods
8. de Moura & Bjørner (2008) - Z3
9. Kissinger & van de Wetering (2020) - ZX-Calculus

---

## Appendix C: Complete Mapping

| Inverse Trick Component | Software | Science |
|------------------------|----------|---------|
| **The Inversion** | Spec → Code | Constraints → Theory |
| **Ground Truth** | Execution traces | Experimental data |
| **Proposer** | LLM generates code | LLM extracts constraints |
| **Verifier** | SMT solver | Consistency checker |
| **Failure Signal** | Counterexample | Contradiction |
| **Refinement** | Fix code | Revise theory |
| **Success** | Verified program | Validated discovery |
| **The Trick** | Traces ground LLM | Experiments ground LLM |

---

*This document synthesizes the inverse trick (constraint-based synthesis) with the trace trick (experimental grounding) into a framework for automated scientific discovery. The core insight: you don't need to formalize all of science—you need to recognize it's a trace problem.*

*The inverse trick is the destination. The trace trick is the vehicle. Together, they make scientific discovery compilable.*
