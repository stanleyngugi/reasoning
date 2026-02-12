# Trace-to-Lean V2 Research Baseline

> Purpose: durable capture of high-impact research findings that drive V2 execution policy.
>
> Note: This is not the runtime contract. Runtime behavior is defined in `V2_EXECUTION_SPEC.md`.

---

## 1) Why This File Exists

`DEEP_RESEARCH_AGENT_PROMPTS.md` contained long-form investigation outputs.
This baseline preserves the essential intelligence in a stable, implementation-oriented format.

This file captures:
- what we believe to be true enough to encode as policy,
- what is still uncertain,
- what must be benchmark-calibrated before hard locking thresholds.

---

## 2) Research-Constrained Reality (Executive View)

1. Core Lean path is viable but not free:
- `native_decide` + `grind` can support strong checker workflows.
- trust-boundary hygiene is mandatory (attribute sanitization).

2. Symbolic engine split is structural:
- SymPy is useful for light exact tasks and as a scout.
- Sage/Singular is required for robust multivariate heavy cases.

3. Elimination workflows need strict pruning:
- elimination roots are necessary conditions only.
- safe acceptance requires domain filtering + back-substitution + residual proof.

4. Confidence requires correlation-aware calibration:
- naive independence multiplication is unsafe.
- consistency/stability-based confidence is required.

5. Coverage is difficulty-dependent:
- high at easier levels for polynomial-reducible classes,
- lower for insight-heavy/hard olympiad strata.

---

## 3) Lean Checker Findings and Decisions

## 3.1 Native Computation Path

Working assumption:
- `native_decide` gives practical checker speedups needed for competition throughput.

Risk:
- expanded trusted code base versus kernel-only proofs.

Policy decision:
- acceptable for competition checker role.
- do not market as kernel-minimal formal proof in the strict theorem-library sense.

## 3.2 Sanitization Requirement

Critical failure mode:
- unsafe Lean attributes (e.g., implementation overrides) can undermine checker trust.

Policy decision:
- mandatory sanitization stage before checker invocation.
- strip/ban trust-expanding attributes in generated code.

## 3.3 Grind and Positivity Gap

Working assumption:
- `grind` is the core identity-check workhorse in no-Mathlib path.

Constraint:
- positivity-style automation gap in core path may require explicit lemmas.

Policy decision:
- keep inequality checker flows decomposition-first.
- use small deterministic lemmas for nonnegativity where needed.

---

## 4) Deployment Findings and Decisions

## 4.1 Offline Packaging

Working assumption:
- sideloading prebuilt toolchains/environments is required for strict offline runs.

Policy decision:
- prepack Lean toolchain and artifacts.
- prepack heavy CAS stack when targeting robust geometry/algebra coverage.

## 4.2 Cold Start vs Persistent Workers

Finding:
- repeated process cold starts degrade throughput materially.

Policy decision:
- persistent Lean worker and persistent symbolic worker are required for serious deployment.
- one-shot per-query subprocess strategy is non-competitive.

---

## 5) Symbolic Engine Findings and Decisions

## 5.1 SymPy Ceiling

Finding:
- SymPy has reliability and performance ceilings on harder multivariate polynomial systems.

Policy decision:
- treat SymPy as scout/fast path only.
- enforce strict switch triggers to heavy solver path.

## 5.2 Sage/Singular Role

Finding:
- robust heavy solve behavior depends on stronger algebraic kernels.

Policy decision:
- Sage/Singular is default heavy path for high-complexity algebra/geometry instances.

## 5.3 Decision Boundary (Operational)

Switch triggers (default):
- variable-degree complexity threshold crossed,
- timeout/incomplete signatures,
- need for complete root accounting.

These triggers are codified in `V2_EXECUTION_SPEC.md`.

---

## 6) Elimination Workflow Findings and Decisions

## 6.1 Necessary vs Sufficient

Finding:
- elimination polynomial roots are not sufficient by themselves.

Policy decision:
- no acceptance from elimination root alone.
- must pass safe acceptance protocol.

## 6.2 Safe Acceptance Protocol

Required checks:
1. domain sieve
2. back-substitution extension
3. exact residual checks on original constraints
4. branch/domain disambiguation
5. bounded-answer policy check (if contest format requires)

Tier A in algebra/geometry depends on this protocol.

---

## 7) Inequality Pipeline Findings and Decisions

## 7.1 Decomposition-First is Viable

Finding:
- decomposition-guided workflows (SOS/Schur-family) can materially outperform naive prompting.

Policy decision:
- LLM proposes decomposition candidates only.
- deterministic checker validates identity and nonnegativity path.

## 7.1.1 Structured Prompting Matters

Practical finding:
- decomposition quality depends heavily on enforcing a structured reasoning protocol.

Recommended 5-step decomposition prompt skeleton:
1. degree parity and leading-term sign checks
2. asymptotic nonnegativity sanity
3. symmetry/cyclic-class detection
4. known-form mapping (SOS/Schur/AM-GM class)
5. explicit decomposition proposal + identity residual check

Policy decision:
- decomposition prompting should be template-driven, not free-form "solve this inequality" prompting.

## 7.2 Known Failure Classes

High-risk classes:
- non-SOS nonnegative forms,
- high-degree coefficient-sensitive decompositions,
- non-symmetric/conditional variants.

Policy decision:
- explicit fallback ladder required.
- no forced certification when decomposition fails checker.

---

## 8) Confidence and Error Calibration Findings

## 8.1 Correlation Reality

Finding:
- agreement among LLM agents/method variants is not independent evidence by default.

Policy decision:
- ban independence-product confidence assumptions.
- use correlation-aware aggregation.

## 8.2 Steering/Consistency Signals

Finding:
- confidence stability under semantic steering is a useful reliability feature.

Policy decision:
- incorporate consistency and variance penalties in confidence model.
- use this for Tier A/B/C assignment, not raw model self-confidence alone.

## 8.3 Threshold Status

Current thresholds:
- provisional.

Policy:
- thresholds only become binding after benchmark calibration.

---

## 9) Coverage Findings and Planning Implications

Finding:
- polynomial/certificate reducibility varies strongly by difficulty and domain.

Planning implication:
- router must be difficulty-aware.
- expected fallback share increases at harder problem strata.

Policy decision:
- report coverage with uncertainty ranges, not single headline numbers.
- treat geometry/algebra coverage claims as stage-dependent and benchmark-validated.

---

## 10) Benchmark Findings and Decisions

## 10.1 Required Metrics

At minimum:
- Tier A precision
- fallback frequency
- verification-generation gap
- attempt-scaling slope
- per-stage failure taxonomy rates
- latency/throughput

## 10.2 Go/No-Go Mindset

Policy:
- no deployment freeze without safety and throughput gates met.
- if gates miss, reduce ambition: tighten acceptance and rely more on fallback.

---

## 11) Offline Deployment Recipes (Practical Transfer)

This section preserves practical deployment intelligence from deep investigation.

## 11.1 Lean Offline Packaging

Operational recipe:
1. prebuild and pin Lean toolchain version
2. precompile required project artifacts
3. package toolchain + artifacts as offline bundle
4. verify checker startup in sandbox before scoring loop

Risk control:
- avoid runtime toolchain fetch assumptions
- run startup self-test for checker integrity

## 11.2 Sage/Singular Packaging

Operational recipe:
1. build symbolic environment offline
2. package relocatable environment (for example, `conda-pack`-style bundle)
3. load once per run and keep persistent worker

Risk control:
- enforce disk budget checks
- enforce cold-start budget checks
- fallback to scout-only mode if heavy stack cannot be loaded

---

## 12) What is Assumed vs What is Verified

This section is intentionally explicit.

Assumed stable enough for policy encoding:
- checker-centric architecture beats raw generation for reliability.
- safe acceptance protocol is mandatory for elimination workflows.
- correlation-aware confidence is necessary.

Needs local benchmark confirmation before hard lock:
- exact tier thresholds,
- exact coverage ranges per difficulty tier,
- exact attempt budgets for each domain.

---

## 13) Migration Notes from Deep Report

The following high-impact intelligence has been moved into durable policy:
- sanitizer hard gate for Lean trust-boundary safety
- solver decision boundary (SymPy scout, Sage/Singular heavy path)
- elimination safe acceptance protocol
- decomposition-first inequality workflow + fallback ladder
- correlation-aware confidence aggregation
- benchmark/go-no-go gate framing

If `DEEP_RESEARCH_AGENT_PROMPTS.md` is deleted, this baseline plus `V2_EXECUTION_SPEC.md` retain the required operational intelligence.
