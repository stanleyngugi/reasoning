# Trace-to-Lean V2 Execution Specification

> Scope: End-to-end execution contract for combinatorics, number theory, algebra, and geometry.
>
> Status: Operational source of truth. This file defines stage gates, acceptance criteria, confidence policy, and fallback behavior.
>
> Companion rationale: `ideas/proof_tir/V2_RESEARCH_BASELINE.md`

---

## 0) Design Philosophy

The system is a deterministic controller around stochastic proposal modules.

- LLM role: propose structured artifacts (traces, constraints, decompositions, solve-order hints).
- Deterministic role: audit, solve, certify, verify, and reject.
- Lean role: checker of bounded computational and polynomial-certificate claims.

Core inversion:
- Do not ask the LLM to solve difficult math directly.
- Ask the LLM to produce auditable intermediate representations.

---

## 1) Hard Invariants (Non-Negotiable)

1. Never accept a final answer that fails original-constraint residual checks.
2. Never accept a branch chosen only by solver order (`sol[0]` anti-pattern).
3. Never promote heuristic output to formal tier without a machine-checkable certificate.
4. Never claim universal correctness from finite-prefix checks alone.
5. Never execute LLM-generated Lean attributes that expand trust boundaries (`@[implemented_by]`, `@[extern]`).
6. Never bypass audit-stage failures to save time.

---

## 2) Stage Graph

Stages are strict and ordered:

- `S0` Route
- `S1` Oracle Generation
- `S1.5` Lean/Artifact Sanitization
- `S2` Deterministic Audit
- `S3` Deterministic Solve
- `S4` Certificate Synthesis
- `S5` Checker
- `S6` Decision
- `S7` Fallback
- `S8` Logging + Calibration Update

No stage may be skipped for Tier A decisions.

---

## 3) Stage Contracts

## S0 Route

Input:
- problem text

Output:
- `domain` in `{combo, nt, algebra, geometry, mixed}`
- `difficulty` in `{easy, medium, hard}`
- `route_confidence` in `[0,1]`

Pass:
- valid domain label + confidence score

Policy:
- if confidence below threshold, set `domain=mixed`.

## S1 Oracle Generation

Input:
- problem text
- route metadata

Output (domain-dependent):
- combo/nt: trace programs, candidate formulas/relations
- algebra/geometry: dual formalizations `F_A`, `F_B` in schema
- inequalities: decomposition candidates (SOS/Schur/AM-GM forms)

Pass:
- artifacts parse against schema
- required fields present

Retry:
- up to `N_oracle_retries`

## S1.5 Sanitization (Mandatory)

Purpose:
- neutralize trust-boundary and parser risks before checker interaction.

Required actions:
1. Strip Lean attributes:
- `@[implemented_by]`
- `@[extern]`

2. Reject unsafe Lean constructs:
- non-terminating or IO-driven code paths in checker templates
- unauthorized imports outside allowed set

3. Canonicalize artifacts:
- symbol names
- index conventions
- numeric types (`Int`/`Nat` expectations)

4. Enforce schema version:
- reject unknown fields in strict mode.

Pass:
- sanitized artifact bundle + no critical sanitizer flags

Fail:
- return to `S1`

## S2 Deterministic Audit

Input:
- sanitized artifacts

Output:
- `audit_report` with pass/fail + typed findings

Audit checks:
1. Schema consistency (symbols, arities, target expression).
2. Degree-of-freedom sanity (especially geometry).
3. Domain/range validity (positivity, integer requirements, denominator nonzero conditions).
4. Structural consistency:
- distance-square positivity
- triangle inequality checks where applicable
- duplicate/contradictory constraints
5. Dual-formalization agreement:
- `F_A` and `F_B` equivalence after normalization OR
- explicit reconciliation map with no unresolved conflicts.

Pass:
- no critical findings

Fail:
- reject and return to `S1`

## S3 Deterministic Solve

Input:
- audited bundle

Output:
- candidate set `C = {c1..ck}`
- `solve_trace` including method path and branch history

Required solver order (default):
1. exact symbolic path:
- Sage/Singular if available
- otherwise hardened SymPy exact path
2. elimination path:
- resultant or triangular decomposition
3. numerical reconstruction fallback:
- high-precision numerical solve + PSLQ / integer snap

Branch policy:
- maintain candidate set throughout solve
- no early single-branch acceptance

Pass:
- at least one candidate with low residual on original constraints

Fail:
- `S7` fallback

## S4 Certificate Synthesis

Input:
- candidate set + solve trace

Output:
- certificate pack per candidate

Certificate classes:
- `combo_nt_relation_cert`:
  recurrence/closed-form consistency + holdout validation
- `alg_geom_root_cert`:
  elimination polynomial + domain constraints + extension witness checks
- `ineq_decomp_cert`:
  decomposition identity `P = sum(q_i^2)` or equivalent Schur-form identity
- `over_verify_cert`:
  independent consequences not used in main solve

Pass:
- machine-checkable certificate for each surviving candidate

Fail:
- downgrade candidate tier or prune

## S5 Checker

Input:
- candidate + certificate pack

Output:
- checker verdict in `{pass, fail, inconclusive}`

Checker engines:
1. deterministic Python exact residual checks
2. Lean `native_decide` bounded computation checks
3. Lean `grind` polynomial identity checks

Pass criteria:
- all mandatory checks for target tier pass

Fail behavior:
- prune candidate; if none remain, `S7`

## S6 Decision

Input:
- checked candidates
- confidence features

Output:
- final answer
- assurance tier
- evidence bundle

Decision rules:
1. prefer Tier A over Tier B over Tier C
2. on tie:
- stronger certificate class wins
- then independent-method agreement
- then lower residual / stricter domain compliance
3. if no acceptable candidate:
- `S7`

## S7 Fallback

Output:
- non-formal answer path (high-precision numerical or TIR)
- explicit downgrade label

Rule:
- fallback output cannot be labeled Tier A.

## S8 Logging and Calibration

Always log:
- stage latency and retries
- failure taxonomy tags
- final tier and evidence summary
- calibration features (Section 8)

---

## 4) Solver Decision Boundary Protocol

SymPy is scout-tier. Sage/Singular is heavy-lifter tier.

Immediate switch from SymPy to Sage/Singular when any trigger fires:
1. variables >= 3 and max degree >= 2
2. variables >= 2 and max degree >= 4
3. SymPy timeout
4. SymPy incomplete-solution signature
5. `UnsolvableFactorError` or equivalent algebraic radical barrier
6. requirement to count all roots/branches

SymPy hardening requirements:
- exact domains only (`QQ`/`ZZ` where relevant)
- rationalize inputs before solve
- strict timeout wrappers
- `check=False` when checksol hangs are known failure mode

### Research Rationale

- **SymPy Ceiling:** SymPy has documented reliability and performance ceilings on harder multivariate polynomial systems; treat as scout/fast path only.
- **Sage/Singular Role:** Robust heavy solve behavior depends on stronger algebraic kernels; Sage/Singular is default heavy path for high-complexity instances.

---

## 5) Safe Acceptance Protocol for Elimination Workflows

Elimination roots are necessary, not sufficient.

For each elimination candidate `alpha`:
1. Domain sieve:
- enforce integer/positivity/range constraints
2. Back-substitution extension:
- substitute into original system
- find valid extension for remaining variables
3. Residual proof:
- verify all original constraints exactly
4. Branch disambiguation:
- enforce domain and uniqueness constraints
5. Competition bounded-answer pass (if applicable):
- verify candidate against bounded answer policy

Tier A requires full completion of all five steps.

### Research Rationale

- **Necessary vs Sufficient:** Elimination polynomial roots are necessary conditions only; acceptance from elimination root alone is unsafe.
- **Safe Acceptance Protocol:** Tier A in algebra/geometry depends on domain sieve + back-substitution + residual proof; confirmed by systematic failure analysis.

---

## 6) Domain Minimum Acceptance Rules

## Combinatorics / Number Theory

- BM acceptance requires `k >= 2L`.
- candidate must pass holdout and adversarial-index checks.
- formula must pass pre-verify + Lean bounded verification.

## Algebra / Geometry

- dual formalization (or equivalent audit strength) required for Tier A.
- full branch tracking required.
- elimination/root certificate + back-substitution required for Tier A.
- over-verification checks required for geometry Tier A.

## Inequalities

- decomposition identity certificate required.
- deterministic nonnegativity path required (core lemmas + checker).
- decomposition prompting should follow structured templates (degree, asymptotic, symmetry, known-form mapping, then explicit decomposition proposal).
- if identity cannot be certified, downgrade tier or fallback.

### Research Rationale

- **Decomposition-First Viability:** Decomposition-guided workflows (SOS/Schur-family) materially outperform naive prompting; LLM proposes candidates, deterministic checker validates.
- **Structured Prompting:** Decomposition quality depends heavily on enforcing a 5-step reasoning protocol (degree, asymptotic, symmetry, known-form mapping, explicit proposal).
- **Known Failure Classes:** Non-SOS nonnegative forms, high-degree coefficient-sensitive decompositions, and non-symmetric/conditional variants require explicit fallback ladder.

---

## 7) Confidence Tiering

- Tier A:
  deterministic certificate + checker pass + full residual pass.
- Tier B:
  substantial deterministic evidence, partial formal certificate.
- Tier C:
  heuristic/numerical only.

Submission policy:
- always choose highest surviving tier.
- never map Tier C to high-assurance label.

### Research Rationale

- **Checker-Centric Architecture:** Checker-centric architecture beats raw generation for reliability; confirmed as stable enough for policy encoding.
- **Coverage is Difficulty-Dependent:** Coverage varies strongly by difficulty and domain; high at easier levels for polynomial-reducible classes, lower for insight-heavy olympiad strata.

---

## 8) Correlation-Aware Confidence Calibration

Naive independence multiplication is disallowed.

Required confidence features:
1. answer consistency across semantically different prompts/methods
2. confidence stability under steering/persona perturbations
3. checker pass depth and certificate class
4. residual magnitude and branch uniqueness margin

Default aggregation (provisional):
- `score = mean_conf * ans_consistency * conf_stability`

Where:
- `ans_consistency`: majority frequency under steering/method variants
- `conf_stability`: inverse variance penalty

Tier thresholds are provisional until calibrated benchmark run.

### SteerConf Protocol (Concrete Formulation)

Probe the model with 2K+1 semantically diverse personas (cautious → vanilla → confident). Collect answers y_i and confidence scores c_i.

Calibrated confidence:
- `c(x) = μ_c · κ_ans · κ_conf`

Where:
- `μ_c`: mean verbalized confidence across steered personas
- `κ_ans` (Answer Consistency): frequency of majority answer = max_y (count(y) / (2K+1))
- `κ_conf` (Confidence Stability): `1 / (1 + σ_c / μ_c)` — penalizes high variance

If σ_c is high (model uncertain when pressed), κ_conf drops, lowering the final score.

### Research Rationale

- **Correlation Reality:** Agreement among LLM agents/method variants is not independent evidence; naive independence-product confidence is unsafe.
- **Steering/Consistency Signals:** Confidence stability under semantic steering is a useful reliability feature; variance penalties required.
- **Thresholds Provisional:** Thresholds only become binding after benchmark calibration; exact values need local benchmark confirmation.

---

## 9) Lean Checker Policy

Allowed core tactics/checkers:
- `native_decide`
- `grind`

Checker safety requirements:
1. sanitize banned attributes (`@[implemented_by]`, `@[extern]`)
2. restrict imports to allowlist
3. use template-generated checker files, not free-form Lean scripts
4. isolate and timeout checker subprocesses

Performance policy:
- maintain persistent Lean worker when possible.
- avoid repeated cold starts.

### Research Rationale

- **Native Computation Path:** `native_decide` gives practical checker speedups needed for competition throughput; expanded TCB is acceptable for competition but not for theorem-library claims.
- **Sanitization Requirement:** Unsafe Lean attributes can undermine checker trust; mandatory sanitization stage is non-negotiable.
- **Grind Positivity Gap:** Positivity-style automation gap in core path may require explicit lemmas; keep inequality flows decomposition-first.

---

## 10) Deployment Contract (Offline Competition)

Core deployment:
- prepack Lean toolchain and required `.olean`s
- use persistent workers for Lean and symbolic engine

Symbolic deployment:
- if heavy algebra/geometry coverage required, package Sage/Singular
- avoid per-query engine startup

Resource policy:
- maintain explicit disk and RAM budgets for each dependency bundle
- run startup self-tests before main solve loop

### 10.1 Docker Build Recipe (Lean Offline)

```dockerfile
FROM ubuntu:22.04
RUN apt-get update && apt-get install -y \
    curl git tar zstd build-essential python3 python3-pip \
    libgmp-dev libffi-dev
RUN useradd -m -s /bin/bash kaggle_user
USER kaggle_user
RUN curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh -s -- -y --default-toolchain none
ENV PATH="/home/kaggle_user/.elan/bin:${PATH}"
ENV LEAN_VERSION="leanprover/lean4:v4.15.0"
RUN elan toolchain install ${LEAN_VERSION}
```

### 10.2 Persistent REPL Integration

```python
class LeanREPL:
    def __init__(self, repl_path, env, cwd):
        self.proc = subprocess.Popen(
            [repl_path],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=env, cwd=cwd, text=True, bufsize=1
        )
    def verify(self, theorem_code):
        command = {"cmd": theorem_code, "env": 0}
        self.proc.stdin.write(json.dumps(command) + "\n")
        self.proc.stdin.flush()
        response = self.proc.stdout.readline()
        data = json.loads(response)
        errors = [m for m in data.get("messages", []) if m["severity"] == "error"]
        return (len(errors) == 0), errors if errors else "Verified"
```

### 10.3 Size Estimates

| Component | Size |
|---|---|
| Lean toolchain | ~400-500 MB |
| Mathlib .olean cache | ~2.5-5 GB |
| REPL cold start | 2-5s |
| REPL per-command latency | <50ms |
| Total disk (Kaggle) | ~6 GB / 20 GB budget |

### Research Rationale

- **Offline Packaging:** Sideloading prebuilt toolchains is required for strict offline runs; runtime toolchain fetch assumptions are unsafe.
- **Cold Start vs Persistent Workers:** Repeated process cold starts degrade throughput materially; one-shot per-query subprocess strategy is non-competitive.

---

## 11) Failure Taxonomy

Every failed attempt must include one primary tag:
- `formalization_error`
- `sanitizer_error`
- `audit_error`
- `solver_timeout`
- `solver_incomplete`
- `branch_error`
- `certificate_error`
- `lean_check_fail`
- `fallback_used`

Secondary tags may be attached for fine-grained diagnosis.

---

## 12) Benchmark and Go/No-Go Gate

Before competition freeze, run benchmark and require:
1. Tier A precision at or above safety threshold.
2. Positive verification-generation gap.
3. Acceptable fallback frequency.
4. Throughput within runtime budget.

If any fails:
- downgrade deployment ambition
- tighten acceptance thresholds
- increase fallback share

### Research Rationale

- **Go/No-Go Mindset:** No deployment freeze without safety and throughput gates met; if gates miss, reduce ambition and rely more on fallback.
- **Required Metrics:** Minimum benchmark must include Tier A precision, fallback frequency, verification-generation gap, attempt-scaling slope, per-stage failure taxonomy rates, and latency/throughput.

---

## 13) Daily Session Protocol (Anti-Drift)

1. Choose session focus:
- combo/nt: `pattern_mining.md`
- algebra/geometry: `ALGEBRA_GEOMETRY_STRATEGY (1).md`

2. Treat this file as binding contract:
- any gate/tier/fallback change must update this file in same session.

3. End-of-session consistency check:
- focus-doc strategy aligns with this contract
- `TRACE_TO_LEAN.md` narrative remains consistent

4. Change classification:
- algorithm detail change -> focus doc
- control-flow/acceptance change -> this file first
- narrative positioning -> `TRACE_TO_LEAN.md`

---

## 14) Relation to Other Docs

- `TRACE_TO_LEAN.md`:
  architecture and positioning narrative.
- `pattern_mining.md`:
  combo/nt algorithms and tier mechanics.
- `ALGEBRA_GEOMETRY_STRATEGY (1).md`:
  algebra/geometry mechanisms and failure analyses.
- `V2_RESEARCH_BASELINE.md`:
  evidence-backed rationale for decisions in this contract.

This file is the implementation contract. If conflict exists, this file wins for runtime behavior.

---

## 15) Provisional Numeric Defaults (Pre-Calibration)

These defaults are starting points, not immutable truths.
They must be recalibrated with benchmark data before freeze.

## 15.1 Confidence Thresholds (Provisional)

- Tier A: `score >= 0.85`
- Tier B: `0.60 <= score < 0.85`
- Tier C: `score < 0.60`

Where `score` is correlation-aware and includes:
- answer consistency
- confidence stability
- certificate/checker depth

## 15.2 Attempt Budgets (Provisional)

- Standard budget: `16` structured attempts
- Escalated budget: `128` attempts

Escalation rule:
- only escalate if early attempts show promising partial structure
- do not escalate zero-signal trajectories

## 15.3 Stage Time Targets (Provisional)

- S0-S2: `10-20s`
- S3: `10-60s` (domain/difficulty dependent)
- S4-S5: `5-30s`
- fallback reserve: `60-120s`

Per-problem target:
- easy: `<=45s`
- medium: `<=120s`
- hard: `<=240s`

## 15.4 Safety Gates (Provisional)

Pre-freeze acceptance targets:
- Tier A precision: `>= 99%`
- fallback frequency under target envelope
- positive verification-generation gap
- acceptable throughput under competition budget

If any gate misses:
- tighten thresholds
- reduce Tier A admissions
- increase fallback usage

---

## 16) Difficulty-Aware Routing Defaults

## 16.1 Policy Bands (Provisional)

- Easy:
  symbolic-first with low retry budget
- Medium:
  hybrid pipeline with standard budget
- Hard:
  deeper decomposition/search, strict admission control, higher fallback expectation

## 16.2 Coverage Framing Rule

Coverage must be reported as ranges by:
- domain
- difficulty
- assurance tier

Never report a single global coverage number without confidence interval and fallback split.

### Research Rationale

- **Coverage Findings:** Polynomial/certificate reducibility varies strongly by difficulty and domain; expected fallback share increases at harder problem strata.
- **Reporting Rule:** Coverage claims must include uncertainty ranges; geometry/algebra coverage claims are stage-dependent and benchmark-validated.

---

## 17) Deployment Defaults

## 17.1 Lean Runtime Defaults

- prepack toolchain and required artifacts
- persistent worker by default
- strict sanitizer on generated Lean before checker pass

## 17.2 Symbolic Runtime Defaults

- SymPy scout path with strict timeouts
- Sage/Singular heavy path for triggered cases
- persistent symbolic worker in long run loops

## 17.3 Cold-Start Mitigation Rule

No per-query process relaunch in production scoring loops unless unavoidable.
Any architecture that repeatedly pays cold-start costs must be treated as non-production.

---

## 18) Benchmark-Calibration Procedure (Required Before Freeze)

## 18.1 Minimum Benchmark Components

Run benchmark with:
- uncontaminated recent competition set
- stress subset for hard cases
- formal-certificate-aware scoring (not answer-only scoring)

## 18.2 Required Dashboard Outputs

- Tier A/B/C precision
- fallback rate
- stage-failure taxonomy rates
- latency/throughput
- verification-generation gap
- attempt-scaling slope

### 18.2.1 Metric Definitions

- **VG Gap (Verification-Generation Gap):** `Verify(Acc) - Generate(Acc)`. Must be positive. A non-positive VG gap means scaling compute (more attempts) will not yield improvements — the verifier cannot distinguish good from bad solutions.
- **Extrapolation Slope:** `(Pass@32 - Pass@1) / log(32)`. Measures marginal return on compute. Target: > 0.05. Below 0.01 indicates reasoning stagnation — the model is not finding new solution paths with more attempts.

## 18.3 Threshold Update Rule

Do not modify tier thresholds on tiny batches.
Update thresholds only after statistically meaningful sample volume and post-hoc calibration diagnostics.

## 18.4 Quality Metrics

### Formalization Tax (L_form)

Percentage of problems where the system finds the correct answer via informal computation but fails to produce a valid Lean verification:

```
L_form = |Correct_Trace ∩ Invalid_Verification| / |Correct_Trace|
```

Measures the cost of requiring formal verification. Track per domain and difficulty band.

### Rigor Bonus (G_rigor)

Percentage of problems where TIR yields an incorrect answer (false positive) while Trace-to-Lean correctly rejects or finds the true solution:

```
G_rigor = |TIR_Fail ∩ Lean_Success| / |Total_Problems|
```

Measures the gain from formal verification. The system's value is proportional to problem difficulty.

### Crossover Hypothesis

- Easy problems (AIME 1-10): TIR outperforms due to Formalization Tax.
- Hard problems (AIME 11-15, IMO): Trace-to-Lean outperforms due to Rigor Bonus.
- Benchmark must report both metrics to validate crossover point.

---

## 19) Pipeline Parallelism and Resource Architecture

### 19.1 GPU/CPU Split

| Resource | Role | Notes |
|---|---|---|
| GPU | LLM inference only | vLLM with continuous batching |
| CPU (high priority, 4-8 cores) | OS, vLLM scheduler, pipeline controller | Reserved; not shared with workers |
| CPU (remaining cores) | Lean worker pool | Memory-constrained: 2-4 GB per worker |

### 19.2 Time Bank Algorithm

- Each problem receives base budget (`120s`).
- Fast problems deposit surplus into shared bank.
- Hard problems withdraw from bank, capped at `α=0.5` of current bank balance.
- Solve easy problems first to capitalize the bank.

### 19.3 Speculative Parallelism ("Shotgun")

- Generate `K=4` independent traces per problem in parallel (shared prefix → cheap via vLLM).
- Mine/verify all unique formulas simultaneously.
- First verified answer wins; cancel remaining.
- Trades `4x` compute for latency reduction — favorable exchange within 6-min budget.
