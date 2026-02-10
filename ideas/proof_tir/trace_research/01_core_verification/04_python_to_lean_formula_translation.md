# Deep Research: Python-to-Lean Formula Translation

## Research Objective

Understand the complete landscape of translating mathematical formulas from Python syntax to Lean 4 syntax. This is NOT autoformalization (natural language to proof). This is syntax mapping between two formal languages. We need to know: what's trivial, what's tricky, what's impossible, and how to handle each case.

## Context

Our system produces formulas in Python-like syntax from the mining phase:
- `n * (n + 1) // 2`
- `pow(2, n, mod) * factorial(n) % mod`
- `sum(comb(n, k) for k in range(n+1))`
- Recurrences like `f(n) = 2*f(n-1) + f(n-2)`

We need to translate these to Lean 4 for verification via `native_decide`. The claim is that this is "trivial syntax mapping." We need to verify this claim and understand edge cases.

## Research Questions

### Part A: Syntax Mapping Fundamentals

#### 1. Basic Arithmetic Operations
Map these Python constructs to Lean 4:
- `+`, `-`, `*` — straightforward?
- `/` (division) — Python float division vs Lean's behavior
- `//` (integer division) — Lean equivalent? `Nat.div`?
- `%` (modulo) — Lean's `Nat.mod` semantics? Negative numbers?
- `**` (power) — Lean's `Nat.pow`? What about large exponents?
- `pow(a, b, m)` (modular exponentiation) — need custom implementation?

#### 2. Comparison and Boolean
- `==`, `!=`, `<`, `>`, `<=`, `>=` — how do these map?
- `and`, `or`, `not` — Lean's `&&`, `||`, `!`?
- Chained comparisons `a < b < c` — need expansion?

#### 3. Common Mathematical Functions
How do we translate:
- `abs(x)` — `Int.natAbs`?
- `factorial(n)` — `Nat.factorial`?
- `comb(n, k)` — `Nat.choose`?
- `gcd(a, b)` — `Nat.gcd`?
- `lcm(a, b)` — `Nat.lcm`?
- `floor(x)`, `ceil(x)` — for rational/real to integer
- `sqrt(n)` — integer square root? `Nat.sqrt`?
- `log(x)`, `log2(x)` — do these exist for Nat?

#### 4. List/Sequence Operations
- `sum(...)` — `List.sum`?
- `prod(...)` — `List.prod`?
- `range(n)` — `List.range n`?
- List comprehensions `[f(x) for x in range(n)]` — `List.map`?
- `all(...)`, `any(...)` — `List.all`, `List.any`?

### Part B: Tricky Translations

#### 5. Division Semantics Mismatch
- Python `//` truncates toward negative infinity
- What does Lean's `Nat.div` do? (Truncates toward zero for Nat, but Nat is non-negative)
- What about `Int.div`? Is it Euclidean division or truncating?
- How do we handle expressions like `(n * (n + 1)) // 2` where we need EXACT integer result?

#### 6. Type Coercion
- Python is dynamically typed; Lean is statically typed
- When do we need explicit `Nat` vs `Int` vs `Fin`?
- How do we handle expressions that mix types?
- What about expressions that are "obviously integers" but Lean doesn't know it?
  - Example: `n * (n + 1) / 2` — division by 2 of product of consecutive integers

#### 7. Recursion and Iteration
- Python `for` loops → Lean recursion or fold?
- How do we translate recursive functions with guaranteed termination?
- What about memoization patterns?
- Example: translating a DP-style computation

#### 8. Conditional Expressions
- Python `a if condition else b` → Lean `if condition then a else b`
- Piecewise functions with multiple conditions
- Guard expressions

### Part C: Complex Formula Patterns

#### 9. Nested Summations
```python
sum(sum(f(i, j) for j in range(i)) for i in range(n))
```
How does this translate to Lean?

#### 10. Binomial Sums
```python
sum(comb(n, k) * (-1)**k for k in range(n+1))
```
Lean translation with proper types?

#### 11. Recurrence Definitions
Given `f(n) = 2*f(n-1) + f(n-2)` with base cases:
- How do we define this in Lean?
- Termination proof requirements
- Efficient computation for large n

#### 12. Modular Arithmetic Chains
```python
((a * b) % m + c) % m
```
Does order of operations match? Associativity concerns?

### Part D: Template-Based Translation

#### 13. Fixed Templates
Design templates for common formula types:
- Closed-form polynomial: `f(n) = an³ + bn² + cn + d`
- Closed-form with factorial: `f(n) = n! / (k! * (n-k)!)`
- Linear recurrence: `f(n) = c₁f(n-1) + c₂f(n-2) + ...`
- Modular exponentiation: `f(n) = a^n mod m`

#### 14. LLM-Assisted Translation
- When do we need LLM help for translation?
- What few-shot examples maximize translation accuracy?
- How do we validate that the translation is correct?

#### 15. Error Recovery
- If Lean rejects a translation (syntax error), what do we do?
- Can we use the error message to fix the translation?
- What are common translation bugs and their fixes?

### Part E: Lean 4 Specifics

#### 16. Decidability Requirements
- For `native_decide` to work, the comparison must be `Decidable`
- What types have decidable equality by default?
- How do we ensure our formulas use decidable operations?

#### 17. Performance Considerations
- Are some Lean representations faster than others for `native_decide`?
- Should we use `Array` vs `List`?
- Tail recursion vs non-tail recursion?

#### 18. Mathlib vs Core Lean
- What mathematical functions are in core Lean 4 vs Mathlib?
- For competition (offline), what do we need to include?
- Minimal imports for maximum functionality?

### Part F: Validation and Testing

#### 19. Translation Correctness
- How do we verify that Lean formula computes the same as Python formula?
- Run both on test inputs and compare?
- Formal equivalence proof? (Probably overkill)

#### 20. Edge Cases to Test
- n = 0 (empty products, base cases)
- n = 1 (boundary)
- Large n (10^6, 10^9)
- Negative integers (if applicable)
- Modular arithmetic edge cases

## Desired Output Format

1. **Syntax Mapping Table**: Python construct → Lean 4 equivalent
2. **Edge Case Analysis**: What's tricky and why
3. **Template Library**: Ready-to-use Lean templates for common patterns
4. **Translation Examples**: 20+ worked examples from simple to complex
5. **Error Pattern Guide**: Common translation bugs and fixes
6. **Type Handling Guide**: When and how to handle types
7. **Implementation Recommendations**: How to build the translator


















# Comprehensive Analysis of Python-to-Lean 4 Formula Translation for Formal Verification

## Executive Summary

This report presents a rigorous architectural and semantic analysis of translating mathematical formulas from Python syntax into the Lean 4 proof assistant language. The primary objective is to enable the automated verification of "mined" mathematical conjectures—formulas generated by heuristic search or machine learning—using Lean 4's `native_decide` tactic.

The central finding of this research is that the common assumption of a "trivial syntax mapping" between Python and Lean 4 is semantically hazardous. While the surface syntaxes share commonalities in arithmetic notation, the underlying operational semantics diverge critically in three areas: integer division behavior, variable-precision integer typing, and the handling of recursion. Python’s dynamic, arbitrary-precision signed integers and floor-based division contrast sharply with Lean 4’s strict separation of Natural Numbers (`Nat`) and Integers (`Int`), as well as its default truncation-based division.

Failure to account for these divergences results in "verification gaps," where a translated formula is proven true in Lean 4 but represents a different mathematical statement than the Python original. This report establishes a comprehensive translation methodology, prescribing specific type coercions (`Int` by default), operator mappings (Euclidean division), and structural transformations (fuel-based recursion) required to bridge these semantic gaps. It culminates in a set of translation templates and implementation recommendations designed to ensure that the rigorous guarantees of `native_decide` apply faithfully to the source conjectures.

---

## 1. Introduction: The Translation Landscape

The intersection of experimental mathematics and formal verification has created a demand for automated pipelines that can promote heuristic conjectures into machine-checked theorems. Systems like the Ramanujan Machine or symbolic regression models often output candidate formulas in Python, the lingua franca of scientific computing. To verify these candidates without manual intervention, we require a transpiler that converts Python Abstract Syntax Trees (AST) into Lean 4 terms.

### 1.1 The Theoretical Gap

The translation challenge is not merely syntactic but type-theoretical. Python operates under a latently typed, imperative paradigm where `x = x + 1` is a state mutation and `5 / 2` dynamically promotes to a float. Lean 4 is a pure functional language based on the Calculus of Inductive Constructions (CIC). It enforces strict static typing and immutability.

Consequently, translating a formula for `native_decide`—which executes the Lean code as a compiled binary to prove a proposition—requires the translator to perform type inference and semantic anchoring that Python’s interpreter usually handles at runtime. The translator must "freeze" the dynamic behavior of Python into a static, decidable Lean definition.

### 1.2 Defining the Success Criteria

A successful translation must satisfy three properties:

1. **Syntactic Validity:** The output must compile in Lean 4.
    
2. **Semantic Fidelity:** The Lean function must return the same values as the Python function for the same inputs, respecting Python’s handling of negative numbers, overflow (which Python avoids), and rounding.
    
3. **Verifiability:** The resulting Lean definition must be computable and terminating, allowing `native_decide` to reduce the proposition `lhs = rhs` to `true`.
    

---

## 2. Part A: Syntax Mapping Fundamentals

The first stratum of translation involves mapping primitive literals, operators, and variable declarations. This section provides the definitive mapping table and analyzes the immediate type constraints.

### 2.1 Primitive Types and Literals

In Python, numerical literals are polymorphic. In Lean, they are overloaded via the `OfNat` type class, but their concrete type is determined by the surrounding context.

|**Python Construct**|**Lean 4 Translation**|**Type Context**|**Analysis**|
|---|---|---|---|
|`123` (int)|`123`|`Int` (Recommended)|Python integers are signed and arbitrary precision. Lean's `Nat` is unsigned. To preserve subtraction semantics (See 2.2), `Int` is the mandatory default target.|
|`3.14` (float)|`3.14`|`Float`|**Warning:** Floating point equality is generally undecidable or ill-defined in formal logic. Formulas relying on Float equality should be rejected or mapped to `Rat` (Rationals).|
|`True` / `False`|`true` / `false`|`Bool`|Note the lowercase convention in Lean.|
|`"string"`|`"string"`|`String`|Rare in math formulas but identical syntax.|
|``|`#`|`Array Int`|**Critical optimization:** Python lists are arrays. Lean `List` is a linked list ($O(n)$ access). Lean `Array` ($O(1)$ access) is required for performance in `native_decide`.|

### 2.2 Arithmetic Operators

The "Trivial Mapping" hypothesis holds for addition and multiplication but collapses for subtraction and division.

**The Subtraction Problem:**

In Python, `3 - 5` evaluates to `-2`. In Lean 4, if the inferred type is `Nat`, the operation uses _saturating subtraction_ (monus), defined as $a \dot{-} b = \max(0, a - b)$. Thus, `(3 : Nat) - 5` evaluates to `0`.

- **Implication:** A formula like `(n - k)` in a summation, which might transiently dip below zero or be used in a signed context, will silently produce incorrect values if translated to `Nat`.
    
- **Rule:** All subtraction `a - b` must be translated in the `Int` context unless `a >= b` is provable statically.
    

**The Division Problem:**

Python’s `/` operator always returns a `float` in Python 3. Lean’s `/` operator on `Int` or `Nat` performs integer division.

- **Rule:** If the Python formula strictly implies integer arithmetic (e.g., combinatorics), occurrences of `/` must be inspected. If they represent exact division, translate to `Int.div`. If they represent floating point operations, the formula is likely unsuitable for exact verification via `native_decide`.
    

### 2.3 Boolean and Comparison Operators

Lean distinguishes between _computable booleans_ (`Bool`) and _logical propositions_ (`Prop`). `native_decide` requires the former.

|**Python**|**Lean 4 (Bool context)**|**Lean 4 (Prop context)**|**Notes**|
|---|---|---|---|
|`a == b`|`a == b`|`a = b`|`==` uses `BEq` instance; `=` uses `Eq`. `native_decide` reduces `BEq` to `true`.|
|`a!= b`|`a!= b`|`a \neq b`||
|`a < b`|`a < b`|`a < b`|`Nat`/`Int` instances are decidable.|
|`x and y`|`x && y`|`x \and y`|Python `and` returns the _value_ of the operand (truthy/falsy). Lean `&&` returns strict `Bool`.|
|`x or y`|`x \| y`|`x \or y`|Similarly, Lean requires boolean operands.|
|`not x`|`!x`|`\not x`||

**The "Truthy" Trap:**

Python allows `if 5:...` (integers are truthy). Lean requires strict Booleans `if 5!= 0 then...`. The translator must inject explicit non-zero checks when non-booleans appear in conditional slots.

---

## 3. Part B: Tricky Translations and Edge Cases

This section details the areas where direct translation fails due to deep semantic mismatches. These are the primary sources of "silent errors" in verification.

### 3.1 Division Semantics Mismatch

There is a profound divergence in how Python and Lean handle integer division and modulo for negative numbers.

#### 3.1.1 Floor vs. Truncation

- **Python (`//`):** Performs **floor division**.
    
    $$-5 // 2 = \lfloor -2.5 \rfloor = -3$$
    
    The result is rounded towards $-\infty$.
    
- **Lean 4 (`Int.div`):** Performs **truncated division**.
    
    $$(-5 : Int) / 2 = \text{trunc}(-2.5) = -2$$
    
    The result is rounded towards $0$.
    

**Translation Strategy:**

To translate Python's `//` correctly, one cannot use Lean's `/`. Instead, one must use **Euclidean division** or explicitly defined floor division. Lean's `Int.ediv` behaves like floor division for positive divisors but maintains the property that the remainder is always non-negative.

- _Correct Mapping:_ `a // b` $\rightarrow$ `Int.ediv a b` (Caveat: Verify behavior for negative `b`).
    
- _Verification:_
    
    - Python: `(-5) // 2` $\rightarrow$ `-3`.
        
    - Lean: `Int.ediv (-5) 2` $\rightarrow$ `-3`. (Matches).
        
    - Python: `5 // (-2)` $\rightarrow$ `-3`.
        
    - Lean: `Int.ediv 5 (-2)` $\rightarrow$ `-2`. (**Mismatch**).
        

**Conclusion:** If the divisor `b` can be negative, a custom `python_div` helper function is required in Lean to replicate Python's specific flooring logic exactly.

#### 3.1.2 Modulo Sign

- **Python (`%`):** Result takes the sign of the **divisor**.
    
    - `-5 % 2` $\rightarrow$ `1` (positive divisor $\to$ positive result).
        
- **Lean 4 (`%` / `Int.mod`):** Result takes the sign of the **dividend**.
    
    - `(-5) % 2` $\rightarrow$ `-1` (negative dividend $\to$ negative result).
        
- **Lean 4 (`Int.emod`):** Euclidean modulo; result is always non-negative.
    
    - `(-5).emod 2` $\rightarrow$ `1`.
        

**Translation Strategy:** For the vast majority of mined formulas where the modulus $n > 0$, Lean's `Int.emod` is the correct semantic equivalent to Python's `%`. Standard `Int.mod` (`%`) should be avoided unless the Python code specifically implies C-style truncation modulo.

### 3.2 Type Coercion (Dynamic vs Static)

Python implicitly promotes types. `math.sqrt(4)` returns `2.0` (float), even though the result is integral. Lean's `Float.sqrt` returns a `Float`, which is opaque to the kernel's equality checker (decidability of Float equality is problematic).

**Edge Case Analysis:**

- **Scenario:** `comb(n, k) * 0.5`.
    
- **Python:** Result is float.
    
- **Lean:** `Nat.choose` returns `Nat`. Multiplication by `0.5` requires coercion to `Float` or `Rat`.
    
- **Recommendation:** If a formula involves fractional scalars, translate the entire expression to the Field of Rationals (`Rat`). `native_decide` handles `Rat` arithmetic efficiently and exactly. Avoid `Float` entirely for verification purposes.
    

### 3.3 Conditional Expressions

Python's `x if c else y` maps to `if c then x else y`. However, Lean requires a proof that `c` is `Decidable`.

- **Tricky Case:** `x if x in large_list else y`.
    
- **Lean:** `if large_list.contains x then...`.
    
    - This requires `large_list` to be an `Array` or `List` with a `BEq` instance for the element type.
        
    - If `large_list` is a `Set` (logical set), it is undecidable. The translation must ensure data structures remain computational (collections) rather than logical (sets).
        

---

## 4. Part C: Complex Formula Patterns

Mined formulas often feature higher-order constructs like summations (`sum`), products (`prod`), and sequences.

### 4.1 Nested Summations

**Python Syntax:**

Python

```
sum(f(i, j) for i in range(n) for j in range(i))
```

This is a generator expression flattening two loops.

**Lean 4 Translation Patterns:**

There are two approaches: Logical (`Finset`) vs Computational (`fold`).

**1. Logical Approach (Mathlib):**

Lean

```
∑ i in Finset.range n, ∑ j in Finset.range i, f i j
```

- _Pros:_ Elegant, standard mathematical notation.
    
- _Cons:_ `Finset.range` builds a structure that ensures no duplicates. While correct, it adds overhead. For `native_decide` on small $n$, this is acceptable.
    

**2. Computational Approach (Arrays):**

Lean

```
(Array.range n).foldl (init := 0) fun acc i =>
  acc + (Array.range i).foldl (init := 0) fun inner_acc j =>
    inner_acc + f i j
```

- _Pros:_ Extremely fast. Maps directly to the imperative execution model of Python.
    
- _Recommendation:_ For the "mining" context where $N$ might be 100 or 1000, the Computational Approach using `Array` operations is strictly preferred to avoid timeout in the reduction engine.
    

### 4.2 Binomial Sums and Alternating Series

**Python:**

Python

```
sum((-1)**k * comb(n, k) for k in range(n+1))
```

**Lean 4 Translation:**

Lean

```
(Array.range (n+1)).foldl (init := 0) fun acc k =>
  acc + ((-1 : Int) ^ k) * (Nat.choose n k : Int)
```

**Key Insights:**

1. **Type Lifting:** The term `(-1)**k` forces the entire accumulator to be `Int`.
    
2. **Explicit Casting:** `Nat.choose` returns `Nat`. It must be explicitly cast `(... : Int)` to multiply with the negative integer. Failure to cast results in a type mismatch error ("expected Int, got Nat").
    
3. **Power Operator:** Lean's `^` on `Int` requires a `Nat` exponent. Python allows `k` to be negative (returning float). The translator must assert `k : Nat`.
    

### 4.3 Recurrence Definitions

Recurrences defined in Python are often naive recursive functions.

**Python:**

Python

```
def a(n):
    if n == 0: return 1
    return a(n-1) + a(n-2)
```

**Lean 4 Translation:**

Direct translation works but is inefficient ($O(2^n)$).

Lean

```
def a : Nat → Nat

| 0 => 1
| 1 => 1
| n + 2 => a (n + 1) + a n
```

- _Termination:_ Lean automatically proves termination here because the argument `n` strictly decreases.
    
- _Optimization:_ For `native_decide` to verify $a(100)$, this naive translation will time out or overflow the stack. The translator should detect linear recurrences and generate **tail-recursive** or **memoized** implementations (see Section 6.3).
    

### 4.4 Modular Arithmetic Chains

**Python:** `pow(base, exp, mod)`

**Lean:** `Int.powMod` is not in the core `Init`. It exists in `Mathlib.Data.Int.Basic`.

- _Issue:_ Standard `^` followed by `%` is inefficient for large numbers.
    
- _Recommendation:_ The translation template library must include a verified binary exponentiation function `pow_mod (b e m : Nat) : Nat` to ensure that verifications involving cryptography-scale numbers do not hang the compiler.
    

---

## 5. Part D: Template-Based Translation System

To achieve high throughput and reliability, the translation should be template-driven rather than purely generative. This section defines the standard templates.

### 5.1 The Template Library

#### Template A: The Closed-Form Evaluator

Used for formulas that depend only on the index `n` and standard constants.

Lean

```
def formula_A (n : Int) : Int :=
  --
```

#### Template B: The Fuel-Guarded Recursion

Used for `while` loops or complex recursions where termination is not structurally obvious (e.g., $n \to n/2$).

Lean

```
def formula_B (n : Nat) (fuel : Nat := 1000000) : Option Nat :=
  match fuel with

| 0 => none -- Failure to terminate within bound
| fuel' + 1 =>
      --
```

_Why Fuel?_ Proving termination for arbitrary Python loops is undecidable (Halting Problem). Fuel allows the definition to be accepted by Lean immediately as structurally terminating (on the `fuel` argument), satisfying the kernel while allowing `native_decide` to execute the logic for finite steps.

#### Template C: The Array Accumulator (Dynamic Programming)

Used for sequences like Fibonacci, Catalan, etc., to ensure $O(n)$ verification.

Lean

```
def formula_C (n : Nat) : Int :=
  let init_arr := mkArray (n + 1) (0 : Int)
  --
  let res_arr := (Array.range n).foldl (init := init_arr) fun memo i =>
     -- [Update memo at i]
  res_arr.get! n
```

### 5.2 LLM-Assisted Translation

Large Language Models (LLMs) can bridge the gap between "messy" Python and strict Lean.

- **Role:** The LLM does not write the proof. It parses the Python intent and selects the correct Template (A, B, or C).
    
- **Prompting Strategy:** "Translate this Python function to Lean 4 using `Int` types. If it is a recurrence, use an `Array` accumulator. Do not use floats."
    
- **Error Recovery:** If the generated code fails to compile (e.g., type mismatch), the error message from Lean is fed back to the LLM for a "repair" pass. This "compiler-in-the-loop" approach resolves most implicit coercion issues.
    

---

## 6. Part E: Lean 4 Specifics & `native_decide`

This section analyzes the mechanics of the verification target.

### 6.1 Decidability Requirements

`native_decide` works by synthesizing an instance of `Decidable P`.

- **Constraint:** The proposition $P$ must be decidable.
    
    - `∀ n, f n = g n` is **undecidable** over infinite `Nat`.
        
    - `∀ n ∈ [0..100], f n = g n` **is decidable**.
        
- **Translation Requirement:** The translator must wrap the formula in a bounded quantifier.
    
    Lean
    
    ```
    def check_range (limit : Nat) : Bool :=
      (List.range limit).all fun n => python_trans n == lean_trans n
    ```
    

### 6.2 Performance Considerations: Arrays vs. Lists

In Python, `list` is a dynamic array. In Lean, `List` is a linked list.

- **Access Cost:** Python `L[i]` is $O(1)$. Lean `L.get! i` is $O(i)$.
    
- **Impact:** A nested loop over a Lean `List` behaves as $O(n^2)$ access overhead, turning an $O(n^2)$ algorithm into $O(n^3)$.
    
- **Mandate:** The translator must map Python lists to Lean **`Array`**.
    
    - Python `L.append(x)` $\to$ Lean `A.push x`.
        
    - Python `L[i] = x` $\to$ Lean `A.set! i x`.
        
    - _Note:_ Lean's `Array` uses functional limits but is optimized in C++ (via the `native_decide` backend) to use destructive updates when the reference count is 1 (Linear Types optimization). This makes it nearly as fast as C++ arrays.
        

### 6.3 Decidable Equality

For custom data structures (e.g., Python classes mapped to Lean structures), one must derive `DecidableEq`.

Lean

```
structure Point where
  x : Int
  y : Int
deriving DecidableEq, Repr
```

Without `deriving DecidableEq`, `native_decide` cannot compare two `Point` objects.

---

## 7. Part F: Validation and Testing

How do we trust the translation?

### 7.1 Translation Correctness

We employ a "Rosetta Stone" testing strategy.

1. **Input Generation:** Generate random inputs (integers, lists).
    
2. **Dual Execution:** Run the Python code and the compiled Lean code (`#eval`).
    
3. **Assertion:** `assert python_out == lean_out`.
    
    This must be part of the CI pipeline for the translation tool.
    

### 7.2 Edge Cases to Test

- **Negative Modulo:** Inputs like `-10`, `divisor=3`. Validate `Int.emod` behavior.
    
- **Recursion Limits:** Inputs that trigger recursion depth > 1000. Verify `fuel` parameters are sufficient.
    
- **Integer Overflow:** While both support arbitrary precision, `native_decide` relies on GMP (GNU Multiple Precision library). Test with numbers $> 2^{64}$.
    
- **Empty Sequences:** `sum()`. Python returns 0. Lean `foldl` with init 0 returns 0.
    

---

## 8. Type Handling Guide

A centralized reference for type decisions.

|**Concept**|**Python**|**Lean 4**|**Translation Action**|
|---|---|---|---|
|**Integer**|`int`|`Int`|Always map to `Int` to preserve subtraction semantics.|
|**Index**|`int`|`Nat`|Arrays require `Nat` indices. Cast `i.toNat`.|
|**Float**|`float`|`Float`|Avoid. If unavoidable, use `Float`, but equality checks will likely fail verification.|
|**Rational**|`fractions.Fraction`|`Rat`|Preferred over Float. Requires `import Mathlib.Data.Rat`.|
|**Boolean**|`bool`|`Bool`|Distinct from `Prop`.|
|**Void/Null**|`None`|`Option`|Python `Optional[int]` $\to$ Lean `Option Int`.|

---

## 9. Implementation Recommendations

To build this system:

1. **Build a "Lean Prelude":** Create a `Translation.lean` file containing helpers that mimic Python behavior.
    
    - `def py_div (a b : Int) : Int :=...` (Flooring division)
        
    - `def py_pow (a b : Int) : Int :=...` (Safe power)
        
    - `def py_list_get (L : Array α) (i : Int) : α :=...` (Handles negative indexing `-1`)
        
2. **Two-Pass Translation:**
    
    - _Pass 1:_ AST-to-AST conversion. Rewrite Python `for` loops into tail-recursive functions with accumulators.
        
    - _Pass 2:_ Type Inference. Propagate `Int` constraints. If a variable is used as an index, insert `.toNat`.
        
3. **Native Decide Harness:**
    
    Generate the verification theorem automatically:
    
    Lean
    
    ```
    theorem verify_formula : check_range 100 = true := by native_decide
    ```
    
4. **Fuel Injection:** Always use fuel-based recursion for generated code to avoid the halting problem during the Lean elaboration phase.
    

## Conclusion

Translating Python to Lean 4 for verification is a rigorous exercise in semantic alignment. The "trivial" view overlooks the profound differences in integer arithmetic and memory models. By adopting `Int` as the default numeric type, `Array` as the default sequence, and Euclidean division as the standard operator, we can construct a high-fidelity translation layer. This enables `native_decide` to serve as a powerful, automated verifier for the growing body of mined mathematical knowledge.

---

## Appendix: Syntax Mapping Table

|**Feature**|**Python**|**Lean 4**|
|---|---|---|
|**Function Def**|`def f(x): return x+1`|`def f (x : Int) : Int := x + 1`|
|**Lambda**|`lambda x: x+1`|`fun x => x + 1`|
|**List Comp**|`[x*2 for x in L]`|`L.map (fun x => x * 2)`|
|**Filter**|`[x for x in L if x > 0]`|`L.filter (fun x => x > 0)`|
|**Range**|`range(n)`|`Array.range n`|
|**Slice**|`L[a:b]`|`L.extract a b`|
|**Update**|`L[i] = v` (mutation)|`L.set! i v` (functional update)|
|**Print**|`print(x)`|`IO.println x`|
|**Assert**|`assert x == y`|`if x!= y then panic! "error"`|

_(End of Report)_