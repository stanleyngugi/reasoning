# Constrained Generation for Formal Verification: Complete Blueprint

## 1. The Core Thesis

**Formal verification fails primarily because LLMs waste compute on syntactically invalid output.**

Unconstrained generation in formal languages (Lean, Coq, SMT-LIB, Dafny) produces massive amounts of unparseable garbage. The verification step—which provides perfect, deterministic feedback—never runs because the output doesn't compile.

**The fix:** Move correctness earlier in the pipeline. Constrain generation with grammar enforcement, then use formal verification as a deterministic judge. This shifts probability mass toward valid outputs and transforms verification errors into actionable feedback.

**Key insight:** Constraints are inference-time structure. They impose what the model lacks in its weights without fine-tuning.

---

## 2. The Fundamental Principle

### Constrain the Result, Not the Reasoning

This is the most important architectural decision. Research (CRANE, ICLR 2025) proves that:

- **Strict constraints kill reasoning.** When you force an LLM to output only syntactically valid tokens from the start, you amputate its ability to "think." The intermediate tokens ARE the reasoning. Constraining them collapses the reasoning distribution.

- **Augmented grammars preserve reasoning.** The solution is to allow free-form "thought blocks" followed by constrained formal output:

```
G_augmented ::= ThoughtBlock FormalOutput
G_strict    ::= FormalOutput  // This kills reasoning
```

- **10% accuracy improvement** over both unconstrained and naively-constrained baselines on symbolic reasoning benchmarks.

**Implementation:** Use delimiter-based switching. Let the model reason freely, then trigger constraints only when entering formal code blocks:

```
[Model thinks freely in natural language...]
I need to prove n + 0 = n. I'll use induction on n.
For the base case, 0 + 0 = 0 by definition.
For the successor case, I need to use the inductive hypothesis.

<formal>
theorem add_zero (n : Nat) : n + 0 = n := by
  induction n with
  | zero => rfl
  | succ n ih => simp [Nat.add_succ, ih]
</formal>
```

The grammar constraint activates only inside `<formal>...</formal>` blocks.

---

## 3. The Architecture

### 3.1 System Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         GENERATION PIPELINE                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                 │
│  │     LLM     │───▶│   Grammar   │───▶│   Formal    │                 │
│  │  Reasoner   │    │   Engine    │    │  Verifier   │                 │
│  └─────────────┘    └─────────────┘    └─────────────┘                 │
│        │                  │                   │                         │
│        │ Unconstrained    │ Constrained       │ Deterministic           │
│        │ (CoT, planning)  │ (formal blocks)   │ (pass/fail + errors)    │
│        │                  │                   │                         │
│        ▼                  ▼                   ▼                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                      FEEDBACK LOOP                               │   │
│  │  • Structured error messages → context for retry                 │   │
│  │  • Verified proofs → positive training data                      │   │
│  │  • Failed attempts → negative training data                      │   │
│  │  • Step-level labels → process supervision                       │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Component Layers

| Layer | Component | Technology | Function |
|-------|-----------|------------|----------|
| **Input** | Intent Parser | Prompt engineering | Convert user intent to formal claim skeleton |
| **Reasoning** | LLM Core | Any capable LLM | Free-form reasoning, proof planning |
| **Switching** | TagDispatch | AC automaton | Detect `<formal>` tags, switch grammar modes |
| **Constraint** | Grammar Engine | XGrammar/llguidance | Enforce syntax validity on formal blocks |
| **Interface** | Kernel Bridge | Pantograph/PISA | Execute tactics, manage proof state |
| **Verification** | Formal Kernel | Lean/Z3/Isabelle | Check logical correctness |
| **Feedback** | Error Parser | PDA-style | Extract structured signals from errors |
| **Training** | Data Flywheel | Standard fine-tuning | Improve model on verified outputs |

---

## 4. The Grammar Engine

### 4.1 TagDispatch: Dynamic Constraint Switching

The key mechanism for interleaved reasoning and formal output:

**How it works:**

1. **Dispatching Mode:** Engine allows free-form text but runs an Aho-Corasick automaton on the output stream, scanning for trigger tags (`<formal>`, `<proof>`, `<code>`).

2. **Trigger:** When a tag is detected, engine switches to Dispatched Mode and loads the associated grammar (Lean EBNF, SMT-LIB CFG, etc.).

3. **Constrained Generation:** Tokens are masked to enforce grammar validity. Only syntactically valid continuations are allowed.

4. **Exit:** Upon detecting closing tag (`</formal>`) or grammar completion, engine returns to Dispatching Mode.

**Why Aho-Corasick:** AC automata match multiple patterns simultaneously in O(n) time. You can have many trigger tags without performance penalty.

### 4.2 The Token Misalignment Problem

LLMs use subword tokens (BPE). Grammars are defined over characters or terminals. These don't align:

- Grammar terminal `function` might become tokens `fun` + `ction`
- Single token might span grammar boundary

**Solution: Token Spanner Tables**

Pre-compute mappings from grammar terminal sequences to valid token sequences. At runtime, look up which tokens can validly continue the current grammar state. This makes constraint checking O(1) per token.

### 4.3 Just-In-Time Mask Compilation

For dynamic scenarios (new lemmas discovered, proof state changes):

1. **Cache Pool:** Maintain pool of precomputed token masks for common grammar states
2. **On-Demand:** For cache misses, compute mask at runtime
3. **Partial JIT:** During prompt processing (prefill), precompute masks for expensive states

### 4.4 Cross-Grammar Caching

Formal languages share substructures. The definition of "identifier" is similar across Lean, Python, SMT-LIB.

**FSM Hashing:** Convert grammar rules to minimized FSMs, compute 64-bit canonical hash. Cache masks by `(fsm_hash, lookahead)`. Reuse across grammars.

Result: 100x reduction in compilation time and memory for multi-grammar systems.

### 4.5 Parser Backend

Use **Earley parsing** for context-free grammars:
- Handles all CFGs including ambiguous ones
- O(n³) worst case, linear for practical grammars
- No stack explosion unlike naive pushdown automata
- Maintains "Earley sets" of possible parse states

---

## 5. The Verification Interface

### 5.1 Pantograph (Lean 4)

The gold standard for LLM-to-Lean interaction.

**Core capabilities:**

- **State Representation:** Exposes kernel-level metavariables and goal states as JSON
- **Incremental Execution:** Execute tactics one at a time, get immediate feedback
- **State Branching:** Snapshot states, try alternatives, backtrack without re-execution
- **Metavariable Coupling:** Tracks dependencies between goals (solving one may constrain others)
- **Sorry Support:** Use `sorry` to draft proofs with holes, fill later

**Protocol:**
```python
# Start proof
state = server.goal_start("∀ n : Nat, n + 0 = n")

# Execute tactic
result = server.goal_tactic(state, "intro n")
# result.goals = [{"type": "n + 0 = n", "hypotheses": ["n : Nat"]}]

# Branch for search
snapshot = state
result_a = server.goal_tactic(state, "induction n")
result_b = server.goal_tactic(snapshot, "simp")  # Try alternative
```

### 5.2 Minilang (Isabelle)

For Isabelle, the full Isar language is too complex. Minilang reduces it to 10 operations:

| Operation | Purpose |
|-----------|---------|
| `HAVE` | Introduce intermediate fact |
| `SHOW` | State goal to prove |
| `FIX` | Introduce fixed variable |
| `ASSUME` | Introduce assumption |
| `OBTAIN` | Existential elimination |
| `APPLY` | Apply tactic/theorem |
| `USING` | Provide premises |
| `BY` | Proof method |
| `DONE` | Complete proof |
| `SORRY` | Skip (for drafting) |

**Key insight:** LLM generates high-level structure, **Sledgehammer** fills in low-level details. Division of labor.

### 5.3 SMT Solver Interface (Z3/CVC5)

For SMT-backed verification:

```python
# Generate SMT-LIB under grammar constraint
query = constrained_generate(prompt, smt_lib_grammar)

# Check satisfiability
result = z3.check(query)

if result == SAT:
    # Verification FAILED - get counterexample
    model = z3.get_model()
    # Feed back: "Failed for x=0, y=-5. Fix the spec."
elif result == UNSAT:
    # Verification SUCCEEDED - property holds
    proof = z3.get_proof()
```

**Counterexample injection:** When verification fails, feed the specific failing values back to the LLM. "Your invariant failed for x=0. Strengthen it."

---

## 6. The Feedback Loop

### 6.1 Process-Driven Autoformalization (PDA)

Move beyond outcome supervision (did it work?) to process supervision (which step was correct?).

**The First Error Location Principle:**

When compilation fails at step `t`:
- Steps 0 to t-1: Labeled **POSITIVE** (valid)
- Step t and after: Labeled **NEGATIVE** (invalid or unknown)

This generates step-level training signal from every failure.

**Implementation:**
```python
def extract_labels(proof_script, compiler_output):
    error_line = parse_error_location(compiler_output)
    labels = []
    for i, step in enumerate(proof_script.steps):
        if i < error_line:
            labels.append((step, 1))  # Positive
        else:
            labels.append((step, 0))  # Negative
    return labels
```

### 6.2 Structured Error Parsing

Don't just use "it failed." Extract actionable information:

| Error Type | Information Extracted | Feedback Format |
|------------|----------------------|-----------------|
| Parse error | Line, expected token | "Expected `;` at line 5" |
| Type mismatch | Expected type, actual type | "Expected Nat, got Int at arg 2" |
| Unknown identifier | Name, similar names | "Unknown `simp`. Did you mean `simp_all`?" |
| Tactic failure | Goal state, tactic tried | "Tactic `rfl` failed on goal `x + 1 = 1 + x`" |
| Unsolved goals | Remaining goals | "1 goal remaining: `∀ y, P y`" |

### 6.3 Hierarchical Fault Localization

For repository-scale verification, locate errors hierarchically:

1. **File level:** Which files are affected?
2. **Function level:** Which definitions/theorems broke?
3. **Line level:** Exactly which line caused the failure?

This bounds the context window needed for repair.

### 6.4 The Training Loop

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA FLYWHEEL                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Verified Proofs ──────────────▶ Positive Training Data        │
│   (correct reasoning chains)                                    │
│                                                                 │
│   Failed Attempts + Error ──────▶ Negative Training Data        │
│   (what NOT to do)                                              │
│                                                                 │
│   (State, Step, Label) Tuples ──▶ Process Supervision           │
│   (step-level correctness)                                      │
│                                                                 │
│   Successful Trajectories ──────▶ Preference Pairs (DPO)        │
│   (which approach worked)                                       │
│                                                                 │
│   Error → Recovery Pairs ───────▶ Error Correction Training     │
│   (how to fix mistakes)                                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Critical insight:** Training on verified proofs doesn't just improve formal language generation. It improves **reasoning**. Verified proofs are proofs where the logic is correct. The model learns what "actually follows" vs "sounds plausible."

---

## 7. Optimizations

### 7.1 Speculative Decoding with Constraints (CDSL)

Accelerate constrained generation without sacrificing validity:

1. **Draft:** Small, fast model generates K tokens (weakly constrained or unconstrained)
2. **Verify (parallel):**
   - Target model checks probability distribution
   - Grammar engine checks syntactic validity
3. **Accept/Reject:** Accept longest valid prefix, regenerate from there
4. **Lookahead Pruning:** Reject tokens that are valid now but lead to dead ends within K steps

**Benefit:** Near-unconstrained throughput with full constraint guarantees.

### 7.2 Drafting with Sorry

Generate proof skeleton first, fill holes later:

```lean
theorem complex_theorem : P ∧ Q ∧ R := by
  constructor
  · sorry  -- Hole 1: prove P
  constructor
  · sorry  -- Hole 2: prove Q
  · sorry  -- Hole 3: prove R
```

**Benefits:**
- High-level structure is often easier than low-level details
- Holes can be dispatched to automation (Sledgehammer, aesop)
- Parallelizable: work on multiple holes simultaneously
- Graceful degradation: partial proofs are still useful

### 7.3 Automation Dispatch

For low-level proof steps, dispatch to symbolic solvers:

| Proof Assistant | Automation Tool | Handles |
|-----------------|-----------------|---------|
| Lean 4 | `aesop`, `omega`, `decide` | Routine goals, arithmetic, decidable props |
| Isabelle | Sledgehammer | Most first-order goals |
| Dafny | Z3 (automatic) | All VCs by default |
| Coq | `auto`, `lia`, `hammer` | Simple goals, linear arithmetic |

**Division of labor:** LLM provides strategy and structure, automation provides tactics and details.

---

## 8. The Verification Landscape

### 8.1 Grammar Complexity Spectrum

```
Trivial                                                          Very Hard
   │                                                                  │
   ▼                                                                  ▼
┌──────┐   ┌──────┐   ┌──────┐   ┌──────┐   ┌──────┐   ┌──────┐   ┌──────┐
│ SMT- │   │      │   │      │   │      │   │ Isa- │   │ Lean │   │ Coq/ │
│ LIB  │   │Alloy │   │Dafny │   │Verus │   │belle │   │  4   │   │ Ltac │
└──────┘   └──────┘   └──────┘   └──────┘   └──────┘   └──────┘   └──────┘
 S-expr     Relat.     C#-like    Rust      Isar       Extens.    Complex
 ~50 rules  minimal    annot.     macros    struct.    macros     tactics
```

### 8.2 System Comparison

| System | Grammar | Automation | LLM Research | Best For |
|--------|---------|------------|--------------|----------|
| **SMT-LIB** | Trivial (S-expr) | Full (decidable) | Medium | Proving concept |
| **Dafny** | Easy (C#-like) | High (Z3) | High | Software verification |
| **Verus** | Easy (Rust) | High (Z3) | High | Rust verification |
| **Isabelle** | Medium (Isar) | High (Sledgehammer) | High | Mathematical proofs |
| **Lean 4** | Hard (extensible) | Medium | Highest | Mathematical proofs |
| **Coq** | Very Hard | Low | Medium | Foundational proofs |

### 8.3 Where to Start

**Recommendation:** Start where grammar is simple to prove the concept, then move up.

| Phase | Target | Why |
|-------|--------|-----|
| **Phase 1** | Dafny / SMT-LIB | Grammar is trivial, automation is complete, feedback is instant |
| **Phase 2** | Isabelle + Minilang | 10-operation subset, Sledgehammer handles details |
| **Phase 3** | Lean 4 | Full power, but hardest grammar, largest community |

---

## 9. Implementation Plan

### 9.1 Phase 1: Prove the Thesis (Dafny)

**Goal:** Demonstrate constrained generation beats rejection sampling.

**Timeline:** 4-6 weeks

**Stack:**
- Model: Llama-3-8B or DeepSeek-Coder-7B
- Grammar: Dafny annotation EBNF (requires, ensures, invariant, decreases)
- Constraint Engine: XGrammar or Outlines
- Verifier: Dafny → Boogie → Z3
- Inference: vLLM or SGLang

**Deliverables:**
1. EBNF grammar for Dafny annotations
2. Integration with inference engine
3. Verification wrapper with error parsing
4. Comparison: constrained vs unconstrained at equal sample budget

**Success Metric:** Higher verified rate at Pass@K for constrained generation.

### 9.2 Phase 2: Build the Training Loop

**Goal:** Show that training on verified outputs improves reasoning.

**Timeline:** 4-6 weeks

**Process:**
1. Generate proofs (constrained) for problem set
2. Verify all outputs, collect (problem, proof, success) tuples
3. Fine-tune on successful proofs
4. Measure improvement on held-out set
5. Iterate

**Deliverables:**
1. Data collection pipeline
2. Fine-tuning scripts
3. Before/after comparison on reasoning benchmarks (not just formal tasks)

**Success Metric:** Model improves on both formal verification AND general reasoning tasks.

### 9.3 Phase 3: Scale to Lean

**Goal:** Apply architecture to hardest target.

**Timeline:** 8-12 weeks

**Challenges:**
- Lean's extensible grammar (macros, notations)
- Dynamic tactic applicability
- Larger proof state context

**Approach:**
1. Start with static tactic whitelist (10-20 tactics)
2. Integrate Pantograph for state management
3. Implement TagDispatch for reasoning/formal switching
4. Add dynamic constraint updates as stretch goal

**Deliverables:**
1. Lean 4 grammar subset (EBNF)
2. Pantograph integration
3. Full feedback loop
4. Results on MiniF2F subset

---

## 10. Technical Implementation

### 10.1 Grammar Definition (Dafny Example)

```ebnf
annotation      ::= requires_clause | ensures_clause | invariant_clause | decreases_clause

requires_clause ::= "requires" expression
ensures_clause  ::= "ensures" expression
invariant_clause::= "invariant" expression
decreases_clause::= "decreases" expression_list

expression      ::= or_expr
or_expr         ::= and_expr ("||" and_expr)*
and_expr        ::= cmp_expr ("&&" cmp_expr)*
cmp_expr        ::= add_expr (cmp_op add_expr)?
cmp_op          ::= "==" | "!=" | "<" | "<=" | ">" | ">="
add_expr        ::= mul_expr (("+"|"-") mul_expr)*
mul_expr        ::= unary_expr (("*"|"/"|"%") unary_expr)*
unary_expr      ::= ("!"|"-")? primary_expr
primary_expr    ::= identifier | number | "true" | "false" | "(" expression ")"
                  | "old" "(" expression ")" | "forall" quantifier | "exists" quantifier
quantifier      ::= identifier "::" expression

identifier      ::= [a-zA-Z_][a-zA-Z0-9_]*
number          ::= [0-9]+
expression_list ::= expression ("," expression)*
```

### 10.2 XGrammar Integration

```python
from xgrammar import GrammarMatcher, CompiledGrammar
from transformers import AutoTokenizer

# Load grammar
with open("dafny_annotations.ebnf") as f:
    grammar_text = f.read()

# Compile for specific tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3-8B")
compiled = CompiledGrammar(grammar_text, tokenizer)
matcher = GrammarMatcher(compiled)

def generate_with_constraints(prompt, model, max_tokens=256):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    generated = input_ids.clone()
    
    matcher.reset()
    
    for _ in range(max_tokens):
        # Get logits from model
        outputs = model(generated)
        logits = outputs.logits[0, -1, :]
        
        # Get valid token mask from grammar
        mask = matcher.get_next_token_bitmask()
        
        # Apply mask (set invalid tokens to -inf)
        logits[~mask] = float('-inf')
        
        # Sample from valid tokens
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, 1)
        
        # Update grammar state
        matcher.accept_token(next_token.item())
        
        # Append to sequence
        generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
        
        if matcher.is_terminated():
            break
    
    return tokenizer.decode(generated[0], skip_special_tokens=True)
```

### 10.3 TagDispatch Implementation

```python
import re
from enum import Enum

class Mode(Enum):
    UNCONSTRAINED = 1
    CONSTRAINED = 2

class TagDispatcher:
    def __init__(self, grammar_matcher, open_tag="<formal>", close_tag="</formal>"):
        self.matcher = grammar_matcher
        self.open_tag = open_tag
        self.close_tag = close_tag
        self.mode = Mode.UNCONSTRAINED
        self.buffer = ""
    
    def get_mask(self, logits, tokenizer):
        if self.mode == Mode.UNCONSTRAINED:
            # Check if we're about to generate open tag
            # (simplified - real impl uses AC automaton)
            return torch.ones_like(logits, dtype=torch.bool)
        else:
            # Constrained mode - use grammar mask
            return self.matcher.get_next_token_bitmask()
    
    def accept_token(self, token_id, tokenizer):
        token_str = tokenizer.decode([token_id])
        self.buffer += token_str
        
        if self.mode == Mode.UNCONSTRAINED:
            if self.buffer.endswith(self.open_tag):
                self.mode = Mode.CONSTRAINED
                self.matcher.reset()
                self.buffer = ""
        else:
            self.matcher.accept_token(token_id)
            if self.buffer.endswith(self.close_tag):
                self.mode = Mode.UNCONSTRAINED
                self.buffer = ""
```

### 10.4 Pantograph Integration

```python
import subprocess
import json

class PantographClient:
    def __init__(self):
        self.process = subprocess.Popen(
            ["lake", "env", "pantograph"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True
        )
    
    def send(self, cmd):
        self.process.stdin.write(json.dumps(cmd) + "\n")
        self.process.stdin.flush()
        response = self.process.stdout.readline()
        return json.loads(response)
    
    def goal_start(self, expr):
        return self.send({"cmd": "goal.start", "expr": expr})
    
    def goal_tactic(self, state_id, tactic):
        return self.send({
            "cmd": "goal.tactic",
            "stateId": state_id,
            "tactic": tactic
        })
    
    def goal_delete(self, state_id):
        return self.send({"cmd": "goal.delete", "stateId": state_id})

# Usage
client = PantographClient()
state = client.goal_start("∀ n : Nat, n + 0 = n")

# state = {"stateId": 0, "goals": [{"type": "∀ n : Nat, n + 0 = n", ...}]}

result = client.goal_tactic(state["stateId"], "intro n")
# result = {"stateId": 1, "goals": [{"type": "n + 0 = n", "hypotheses": [...]}]}
```

### 10.5 Feedback Loop

```python
def prove_with_feedback(theorem: str, max_attempts: int = 5):
    context = f"Prove the following theorem:\n{theorem}\n\n"
    
    for attempt in range(max_attempts):
        # Generate with constraints (reasoning free, formal constrained)
        response = generate_interleaved(
            prompt=context,
            model=llm,
            dispatcher=TagDispatcher(lean_grammar)
        )
        
        # Extract formal blocks
        proof = extract_formal_blocks(response)
        
        # Verify with Lean
        result = pantograph.verify(theorem, proof)
        
        if result.success:
            log_success(theorem, proof, attempt)
            return proof
        
        # Parse error and add to context for retry
        error_info = parse_lean_error(result.error)
        context += f"""
Attempt {attempt + 1} failed.
Your proof:
{proof}

Error at line {error_info.line}: {error_info.message}
Goal state: {error_info.goal_state}

Please try again with a corrected proof.
"""
    
    log_failure(theorem, attempts)
    return None

def parse_lean_error(error_text):
    """Extract structured info from Lean error message."""
    # Line number
    line_match = re.search(r':(\d+):\d+:', error_text)
    line = int(line_match.group(1)) if line_match else None
    
    # Error type and message
    if "type mismatch" in error_text:
        error_type = "type_mismatch"
        # Extract expected and actual types
        expected = re.search(r'expected\s+(.+?)(?:\n|$)', error_text)
        actual = re.search(r'actual\s+(.+?)(?:\n|$)', error_text)
    elif "unknown identifier" in error_text:
        error_type = "unknown_identifier"
    elif "unsolved goals" in error_text:
        error_type = "unsolved_goals"
    else:
        error_type = "other"
    
    return ErrorInfo(line=line, type=error_type, message=error_text)
```

---

## 11. Metrics

### 11.1 Primary Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **Parse Rate** | % of outputs that parse | 100% (with constraints) |
| **Verified Rate @ K** | % of problems solved in K attempts | Higher than baseline |
| **Samples to Solution** | Average attempts for successful proof | Lower than baseline |
| **Syntax → Semantic Ratio** | Of parseable outputs, % that verify | Higher with constraints |

### 11.2 Efficiency Metrics

| Metric | Description |
|--------|-------------|
| **Constraint Overhead** | Additional latency from grammar enforcement |
| **Verifier Calls Saved** | Reduction in wasted verification attempts |
| **Tokens per Proof** | Generation efficiency |
| **Time to Solution** | Wall-clock time for successful proof |

### 11.3 The Key Experiment

```
Constrained Generation @ Pass@K  vs  Unconstrained @ Pass@K

Variables:
- K = {1, 4, 16, 64, 256}
- Model = {7B, 13B, 70B}
- Problem set = {Dafny-110, MiniF2F subset}

Hypothesis: Constrained wins at all K values, advantage grows with difficulty.
```

---

## 12. Scalability Properties

### 12.1 What This Architecture Scales With

| Dimension | Scaling Behavior |
|-----------|-----------------|
| **Model size** | Better base model → better reasoning → better proofs |
| **Model capability** | Smarter models need less retry, win larger |
| **Inference speed** | Faster inference → more attempts per second |
| **Training data** | Flywheel generates more data → better model → more data |
| **Compute** | More attempts → higher success rate (but sublinear) |

### 12.2 What This Architecture Doesn't Require

| Not Required | Why |
|--------------|-----|
| Custom architecture | Uses standard LLM |
| MCTS infrastructure | LLM plans in CoT, no tree search needed |
| Test-time training | Works at pure inference time |
| Massive compute | Production-ready latency |
| Human labels | Verification is the label |

### 12.3 Comparison with Alternative Approaches

| Approach | Compute | Latency | Scalability |
|----------|---------|---------|-------------|
| **AlphaProof** | 100s TPU-days | Days | Research-only |
| **MCTS + LLM** | High (rollouts) | Minutes-hours | Expensive |
| **Rejection Sampling** | Wastes ~90% | Variable | Poor efficiency |
| **This Approach** | Standard inference | Seconds | Production-ready |

---

## 13. Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Grammar too restrictive | Can't express valid proofs | Start small, expand based on failure analysis |
| Constraint overhead | Slower than expected | Use CDSL, JIT compilation, batch at tactic boundaries |
| Dynamic gating latency | Pantograph queries too slow | Cache applicable tactics, precompute common states |
| Token alignment | Subwords break grammar | Use libraries with built-in spanner tables |
| Lean extensibility | Static grammar incomplete | Accept partial coverage, focus on common tactics |
| Training instability | Flywheel doesn't compound | Careful data balancing, curriculum learning |

---

## 14. The Broader Vision

### 14.1 Universal Architecture

The loop applies across all formal systems:

| Domain | Verifier | Grammar | Application |
|--------|----------|---------|-------------|
| Mathematics | Lean kernel | Tactic subset | Theorem proving |
| Software | Z3/CVC5 | SMT-LIB | Verification conditions |
| Protocols | SPIN | Promela | Concurrent systems |
| Systems | TLA+ | TLA+ spec | Distributed systems |
| Hardware | nuXmv | SMV | Circuit verification |
| Rust | Verus | Annotation macros | Memory safety |

### 14.2 The Endgame

**Short term:** LLMs that generate verified code and proofs reliably.

**Medium term:** Self-improving systems that generate training data through verification.

**Long term:** Autonomous mathematical discovery—generating and proving new theorems.

The key insight: formal verification provides **perfect supervision**. No human labels needed. No reward hacking. The kernel is truth. Train on truth, learn to reason correctly.

---

## 15. Summary

**The thesis:** Constrained generation + formal verification beats brute-force rejection sampling.

**The architecture:** LLM reasons freely → grammar constrains formal output → verifier judges → errors become feedback → model improves.

**The principle:** Constrain the result, not the reasoning.

**The advantage:** Simple, scalable, production-ready. Rides the wave of LLM improvements. No custom infrastructure.

**The path:** Start with Dafny (easy grammar), prove the concept, build the training loop, scale to Lean.

**The flywheel:** Every verified proof becomes training data. The model improves. The loop compounds.

This is an engineering problem. The theory is sound. The tools exist. Build it.
