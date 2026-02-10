# Image-to-Math: A Deterministic Paradigm for Visual Generation

## 1. Executive Summary: The Visual Compiler

This report delineates a comprehensive research trajectory toward a transformative paradigm in generative computer vision: **Image-to-Math**. This paradigm proposes a fundamental architectural shift, moving the field beyond the current stochastic dominance of pixel-based probabilistic diffusion toward a deterministic, semantically grounded framework where visual media is generated, represented, and manipulated as compact mathematical formulas.

The central thesis of this report is that the current reliance on latent diffusion models (LDMs), while phenomenologically impressive, represents a local maximum in the optimization landscape of visual intelligence. Diffusion models operate on the premise that images are statistical distributions of raster data. We argue, conversely, that **images are fundamentally mathematical objects**. Every digital photograph, vector illustration, or procedural texture can be rigorously defined as a function $f: \mathbb{R}^n \rightarrow \mathbb{R}^3$, mapping spatial and temporal coordinates to color values. The challenge for the next generation of AI is not to refine the statistical approximation of pixels, but to extract the underlying generator functions—the "source code"—of the visual world.

The goal of the Image-to-Math paradigm is to construct a **Visual Compiler**: a neurosymbolic pipeline that accepts raw visual data (images) and decompiles it into editable, verifiable, and resolution-independent mathematical code. This code may take the form of Scalable Vector Graphics (SVG) for geometric art, TikZ scripts for scientific diagrams, constructive solid geometry (CSG) for 3D CAD, or procedural shaders (GLSL) for photorealistic textures.

**Strategic Imperative:**

This is not merely a proposal for "interpretable AI" targeting niche vector graphic workflows. It is positioned as a potential **new standard** for general-purpose image generation. By treating images as code, this paradigm addresses the intractable structural limitations of diffusion models—specifically their inability to guarantee geometric consistency, their infinite-cost upscaling, and their lack of native editability.

**Technological Feasibility:**

This report finds that the technological prerequisites for this paradigm have matured significantly within the 2024-2025 research cycle, moving the concept from theoretical possibility to engineering reality.

- **Rendering is Solved:** The historical bottleneck of differentiable rendering speed has been shattered. The introduction of **Bézier Splatting (NeurIPS 2025)** delivers a 30x to 150x acceleration over previous state-of-the-art methods like DiffVG, enabling the real-time optimization of vector parameters within a training loop.
    
- **Neurosymbolic Integration:** New architectures such as **NESYDM (Neurosymbolic Diffusion Models)** and **DeTikZify** demonstrate that neural networks can successfully guide discrete symbolic search processes, using computation as a ground-truth oracle.
    
- **Data Availability:** The release of massive paired datasets like **StarVector** (2.1M SVGs) and **Shaders21k** provides the necessary signal to train models that map visual percepts to code.
    

This report is structured to provide an exhaustive analysis of the theoretical foundations, the limitations of the incumbent diffusion paradigm, the emerging technical stack, and a concrete roadmap for implementation. It serves as a blueprint for engineering a system where the "image" is transient, but the "math" is eternal.

---

## 2. The Theoretical Foundation: Images as Functions

To understand the necessity of the Image-to-Math paradigm, we must first rigorously define the ontological status of an image. In the current era of deep learning, images are predominantly treated as tensors—static grids of discrete values (pixels). This tensor-centric view creates a dependency on resolution and couples the information content of the image to its storage size.

### 2.1 The Functional Definition

We posit that any image $I$ is a discretization of a continuous signal defined by a function $f$.

$$f(x, y, t) \rightarrow (r, g, b, \alpha)$$

Where $(x, y)$ are spatial coordinates, $t$ is time (for video), and the output is a tuple of color and opacity.

- **Raster Image:** A discrete sampling of $f$ at integer grid points. This is a lossy compression of the function.
    
- **Vector Image (SVG):** A parametric description of $f$ using geometric primitives (Bézier curves, lines, polygons). This preserves the function's scalability.
    
- **Implicit Representation (INR):** A neural network approximating $f$ continuously.
    
- **Procedural Shader:** An algorithmic description of $f$ (e.g., fractal noise, raymarching).
    

The Image-to-Math paradigm seeks to reverse the discretization process: given the sampled raster $I$, recover the optimal function $f$ that generated it. This is an inverse problem that has historically been ill-posed due to the ambiguity of interpretation. However, with the advent of large Vision-Language Models (VLMs), we now possess the "semantic priors" necessary to resolve this ambiguity—distinguishing a "circle" from a "polygon with 100 sides."

### 2.2 Minimum Description Length (MDL) and Kolmogorov Complexity

The theoretical objective of Image-to-Math is closely aligned with minimizing the Kolmogorov Complexity of visual data. The Kolmogorov complexity $K(I)$ of an image $I$ is the length of the shortest program $P$ that outputs $I$.

$$K(I) = \min \{|P| : U(P) = I\}$$

Where $U$ is a universal Turing machine (or in our case, a renderer).

Current bitmap formats (JPEG, PNG) compress statistical redundancy (entropy coding). They do not compress _structural_ redundancy. A 4K image of a Mandelbrot set requires megabytes to store as a PNG, yet the generating formula is only a few bytes of code: $z_{n+1} = z_n^2 + c$. The PNG representation is inefficient because it ignores the generator function.

The Image-to-Math paradigm optimizes for MDL. By finding the shortest mathematical description (code) that reproduces the image, we achieve:

1. **Extreme Compression:** Representing complex structures with minimal bits.
    
2. **Semantic Understanding:** The "shortest program" typically corresponds to the true causal structure of the object (e.g., describing a wheel as a "circle" rather than a list of 1000 edge points).
    
3. **Generalization:** A function derived from low-resolution data can theoretically reconstruct the object at infinite resolution.
    

### 2.3 The Ground Truth Principle

A critical philosophical distinction in this work is the definition of "Ground Truth." In Reinforcement Learning from Human Feedback (RLHF), the ground truth is subjective human preference—a "noisy," unstable signal. In Image-to-Math, **the ground truth is the image itself.**

This creates a self-contained, objective verification loop:

1. **Hypothesis:** The model proposes a formula $f$.
    
2. **Experiment:** The system renders $I_{pred} = \text{Render}(f)$.
    
3. **Verification:** The system compares $I_{pred}$ to the original $I_{target}$.
    
4. **Feedback:** The error $|I_{pred} - I_{target}|$ provides a deterministic gradient signal.
    

This mechanism removes the need for expensive human labeling or messy "aesthetic" scores. If the rendered pixels match the target pixels, the formula is objectively correct (within a tolerance). This allows for unsupervised learning at a massive scale, limited only by compute.

---

## 3. The Meta-Pattern: From "Trace-to-Lean" to "Image-to-Math"

This research identifies a powerful "meta-pattern" emerging across high-reliability AI domains, specifically in mathematical reasoning, which serves as the architectural template for our proposal.

### 3.1 The "Trace-to-Lean" Analogy

In the domain of automated theorem proving, the "Trace-to-Lean" system represents a breakthrough in reliability. It addresses the hallucination problem of LLMs by fundamentally changing the neural network's role.

**The Problem:** Asking an LLM to "prove a theorem" directly is risky. The LLM generates text that _resembles_ a proof, but often contains subtle logical fallacies or invented lemmas.

**The Solution:**

1. **Neural Role:** The LLM acts as a **proposer**. It suggests a tactic or a code snippet.
    
2. **Deterministic Core:** A formal verification system (the Lean Theorem Prover) compiles the code.
    
3. **Verification:** The system runs `native_decide`. This is a binary, computable check. The proof is either valid or invalid. There is no gray area.
    

### 3.2 Translating the Pattern to Vision

The Image-to-Math paradigm applies this exact architecture to visual generation, replacing the logical verifier with a visual verifier (the renderer).

|**Feature**|**Math Reasoning (Trace-to-Lean)**|**Visual Generation (Image-to-Math)**|
|---|---|---|
|**Input**|Mathematical Statement|Target Raster Image|
|**Neural Role**|Propose Proof Strategy / Code|Propose Mathematical Formula / Primitives|
|**Deterministic Core**|Theorem Prover (Lean)|Differentiable Renderer / Compiler|
|**Verification**|`native_decide` (Logical Consistency)|Render $\to$ Pixel Loss (Visual Consistency)|
|**Ground Truth**|Mathematical Axioms|The Input Image Pixels|
|**Objective**|Valid Proof|Visually Identical Reconstruction|

**Key Insight:** Both systems avoid asking neural networks to do the "hard part" (rigorous logic or pixel-perfect rendering). Instead, they ask the neural network to perform **perception and intuition** (proposing a solution), while delegating **verification and execution** to a deterministic computational engine.

This separation of concerns is the only viable path to **Hallucination-Free Generation**. Just as a compiled code snippet either runs or errors, a rendered mathematical formula is structurally rigid—it cannot have the "dream-like" incoherence of a diffusion artifact.

---

## 4. The Crisis of the Diffusion Paradigm

To justify the investment in a new paradigm, we must rigorously analyze the fundamental limitations of the incumbent: Latent Diffusion Models (LDMs). While models like Stable Diffusion and Midjourney have achieved cultural dominance, they face asymptotic limits rooted in their architecture.

### 4.1 The Raster Dependence (The "Pixel Trap")

Diffusion models are fundamentally denoisers of raster grids. They learn the statistical distribution of pixel values.

- **Fixed Resolution:** A diffusion model trained on $512 \times 512$ images treats that grid size as a canonical law. Generating at $1024 \times 1024$ usually requires "hacks" (like Multi-Diffusion or tiling) or a separate upscaler model.
    
- **Upscaling is Hallucination:** When an upscaler increases resolution, it invents new details based on probability. It does not reveal "true" hidden detail because there is no underlying function.
    
- **The "Wobbly Line" Problem:** To a diffusion model, a straight line is not the geometric concept $y=mx+b$. It is a sequence of pixels $(x_1, y_1), (x_2, y_2)...$ that statistically tend to be collinear. Consequently, diffusion models struggle to generate perfectly straight lines, concentric circles, or parallel text, often producing "wobbly" or topologically broken structures.
    

### 4.2 The Stochasticity & Editability Gap

- **Non-Deterministic Output:** Running the same prompt with a different random seed produces a completely different image. This stochasticity is fatal for engineering and design workflows. If an architect wants to "widen the door by 10%," a diffusion model might rewrite the entire building façade in response to the prompt change.
    
- **Black Box Representation:** The "knowledge" of the image is distributed across billions of weights. There is no specific neuron representing "radius" or "color #FF0000." Therefore, precise parametric editing is mathematically impossible in a standard diffusion architecture. One cannot "reach in" and adjust a variable.
    

### 4.3 Inference Inefficiency and Economic Cost

- **Iterative Denoising:** Diffusion requires tens or hundreds of forward passes (steps) to resolve an image. Each step involves processing the entire latent tensor.
    
- **Content-Agnostic Compute:** Generating a simple white square requires the same massive compute load as generating a complex baroque painting. The model does not understand that the square has a simpler description complexity.
    
- **Storage:** A high-resolution raster image requires significant storage. A vector description of the same image (if geometric) requires bytes.
    

**Conclusion:** Diffusion models are excellent **texture synthesizers** but poor **geometry engines**. They simulate the _appearance_ of structure without understanding the _rules_ of structure. Image-to-Math inverts this: it guarantees structure first, with texture applied as a procedural or learned attribute.

---

## 5. Technical Landscape: The Enabling Technologies

The feasibility of Image-to-Math rests on three converging technological breakthroughs that have matured in the 2024-2025 cycle: **Differentiable Rendering**, **Neurosymbolic AI**, and **Implicit Neural Representations**.

### 5.1 Differentiable Rendering: The Verification Engine

For a neural network to learn to output math, it needs gradient feedback. Differentiable rendering allows gradients to flow from the pixel loss of a rendered image back to the parameters of the vector primitives (control points, stroke widths, colors).

#### 5.1.1 The Legacy: DiffVG

**DiffVG** (Differentiable Vector Graphics) was the pioneering library (SIGGRAPH Asia 2020) that enabled differentiation through SVG rendering.

- **Mechanism:** It computes gradients for vector shape parameters by analyzing how pixel coverage changes as shape boundaries move.
    
- **Limitation:** It is computationally expensive. Optimization of complex images can take minutes or hours. It relies on expensive supersampling for anti-aliasing gradients. This slowness prevented it from being used in the inner loop of large-scale training.
    

#### 5.1.2 The Breakthrough: Bézier Splatting (NeurIPS 2025)

A critical development identified in the 2025 research literature is **Bézier Splatting**. This technology bridges the gap between vector graphics and the ultra-fast Gaussian Splatting techniques used in 3D.

- **Core Concept:** Instead of traditional rasterization (checking if pixels are inside curves), this method samples Bézier curves into **2D Gaussian primitives**.
    
- **Mechanism:**
    
    1. **Curve Discretization:** The Bézier curve is adaptively sampled into a set of points.
        
    2. **Gaussian Association:** Each point is treated as the center of a 2D anisotropic Gaussian. The parameters of the Gaussian (position, rotation, scale) are directly inherited from the curve's control points and tangents.
        
    3. **Splatting:** These Gaussians are projected ("splatted") onto the canvas using highly optimized tile-based rasterizers.
        
- **Performance Metrics:**
    
    - **Forward Pass:** 30x faster than DiffVG.
        
    - **Backward Pass:** 150x faster than DiffVG.
        
- **Implication:** This order-of-magnitude speedup makes it feasible, for the first time, to use differentiable vector rendering as a **loss function** inside the training loop of a large generative model. It supports the conversion to standard XML-based SVG, ensuring interoperability with existing tools like Adobe Illustrator.
    

### 5.2 Neurosymbolic Architectures: The Decoder

The decoder must translate the visual encoding into discrete symbols (code). This requires handling the discrete nature of code (tokens) and the continuous nature of visual signals.

#### 5.2.1 Neurosymbolic Diffusion Models (NESYDM)

**NESYDM** (2025) represents a crucial architectural advance for handling discrete structural data.

- **The Challenge:** Standard diffusion generates continuous variables (Gaussian noise). Code is discrete. Existing "discrete diffusion" methods often struggle with the rigid syntax of programming languages.
    
- **The NESYDM Solution:**
    
    - **Masked Diffusion:** It uses **Masked Diffusion Models (MDMs)** for concept extraction. Instead of adding noise, it masks out parts of the program/concept vector.
        
    - **Dependency Modeling:** It treats program symbols as latent variables. Starting with a fully masked vector, it iteratively "unmasks" concepts (symbols/parameters) conditioned on the input image.
        
    - **Joint Distribution:** Crucially, it models the **dependencies** between symbols (e.g., an "opening parenthesis" dictates a future "closing parenthesis") better than autoregressive models by modeling the joint distribution.
        
- **Relevance:** This architecture is ideal for generating complex hierarchical formulas where parameters are interdependent (e.g., a nested loop in TikZ code).
    

#### 5.2.2 DeTikZify and MCTS Refinement

**DeTikZify** (NeurIPS 2024 Spotlight) provides the blueprint for the inference strategy.

- **Task:** Generating TikZ (LaTeX graphics code) from sketches.
    
- **MCTS Refinement Loop:** It introduces a **Monte Carlo Tree Search (MCTS)** inference algorithm to boost performance without retraining.
    
    1. **Selection:** The model proposes multiple code continuations (branches).
        
    2. **Rollout:** Each continuation is simulated (rendered into an image).
        
    3. **Evaluation:** A reward model (vision encoder) scores the visual similarity between the rendered code and the input image.
        
    4. **Backpropagation (MCTS):** High-reward paths are reinforced.
        
- **Self-Correction:** This proves that **computation can verify generation**. The model "thinks" by simulating the code execution and checking the visual result before finalizing the output. This is the "System 2" thinking equivalent for image generation.
    

### 5.3 Implicit Neural Representations (INRs): The Photorealism Bridge

While Vectors (SVG) handle "flat" graphics (logos, diagrams), **Implicit Neural Representations** (INRs) handle gradients, textures, and photorealism. They act as the "mathematical formula" for continuous tone images.

#### 5.3.1 SIREN (Sinusoidal Representation Networks)

- **Spectral Bias Problem:** Standard ReLU networks fail to learn high-frequency details (textures, sharp edges) because they are theoretically biased toward low-frequency functions. A ReLU network fitting an image will produce a blurry blob.
    
- **The SIREN Solution:** Using periodic **Sine** activation functions allows the network to fit high-frequency signals efficiently.
    
- **Image-to-Math Connection:** A SIREN is essentially a complex Fourier series. It represents an image as a superposition of sine waves. This _is_ a mathematical formula.
    
- **Compression:** Recent benchmarks (COIN, COIN++) show that INRs can compress images better than JPEG at low bitrates by storing the _weights_ of the function rather than the pixels.
    
- **High-Frequency Optimization:** New work on **HF-SIREN** and **FINER** (2025) further improves the ability to capture fine textures by modulating the frequency spectrum during training.
    

---

## 6. Proposed Architecture: The "Math-Gen" Pipeline

Based on the existing components, we propose a unified architecture for the Image-to-Math paradigm. This system functions as a **Visual Compiler**, translating pixel data into executable code.

### 6.1 System Overview

The pipeline consists of four distinct stages: Perception, Decoding, Verification, and Abstraction.

### Stage 1: The Semantic Perceiver (VLM-Encoder)

- **Input:** Raster Image $I$ (e.g., a PNG of a logo or a diagram).
    
- **Model:** A Vision-Language Model (like LLaVA, GPT-4V fine-tune, or Qwen-VL) acting as the encoder.
    
- **Function:** The VLM does not generate pixels. It generates a **Semantic Embedding** ($Z_{sem}$) and a **Structural Skeleton**.
    
    - It identifies high-level concepts: "This is a circle next to a rectangle," "This is a chaotic texture (requires noise function)," or "This is a recurring fractal pattern."
        
    - It outputs a "Program Sketch" or a set of constraints (e.g., "Symmetry: Radial").
        
- **Rationale:** We use the VLM for what it is best at: semantic understanding and identifying relationships that are not purely local (e.g., recognizing that a set of lines forms a "grid").
    

### Stage 2: The Neurosymbolic Decoder (NESYDM)

- **Input:** Semantic Embedding $Z_{sem}$.
    
- **Model:** A Neurosymbolic Diffusion Model (NESYDM).
    
- **Process:**
    
    - The model performs discrete diffusion in the **Program Latent Space**.
        
    - It iteratively unmasks code tokens. This could be SVG paths, TikZ commands, or Python Shader code depending on the "mode" identified by the perceiver.
        
    - It respects syntactic constraints (ensuring valid code) by masking invalid tokens at each step (grammar-constrained decoding).
        
- **Output:** A candidate program $P_0$.
    

### Stage 3: The Differentiable Validation Loop (The "Trace-to-Lean" Core)

- **Input:** Candidate Program $P_0$.
    
- **Renderer:**
    
    - For Vector/Shapes: **Bézier Splatting**.
        
    - For Texture/Photorealism: **Differentiable Shader Interpreter** (e.g., executing GLSL graph via Taichi or JAX).
        
- **Verification:**
    
    1. Render $I_{pred} = \text{Render}(P_0)$.
        
    2. Compute Loss $\mathcal{L} = \mathcal{L}_{pixel}(I_{pred}, I_{target}) + \mathcal{L}_{perceptual}(I_{pred}, I_{target}) + \lambda \cdot \text{CodeLength}(P_0)$.
        
    3. **Optimization:** Backpropagate gradients _through the renderer_ to update the numerical parameters of $P_0$ (e.g., control point coordinates, color values).
        
- **Refinement (MCTS):** If the parameter optimization fails to reduce loss below a threshold (indicating the wrong _structure_), the MCTS module (DeTikZify style) prunes the branch and proposes a different structural primitive (e.g., switch from "Circle" to "Polygon").
    

### Stage 4: Abstraction & Canonicalization (DreamCoder)

- **Problem:** The raw output from Stage 3 might be "spaghetti code" (e.g., drawing a grid by defining 100 individual lines manually). This is mathematically correct but not semantically useful.
    
- **Solution:** An abstraction layer inspired by **DreamCoder** / **LILO** (Learning Interpretable Libraries).
    
    - **Library Learning:** The system scans the generated program for repeating sub-patterns.
        
    - **Refactoring:** It compresses these patterns into new functions. `line(0,0)... line(0,10)` becomes `for i in range(10): line(0, i)`.
        
    - **AutoDoc:** It assigns readable names to these functions (e.g., `draw_grid`).
        
- **Result:** The final output is clean, editable, human-readable code that captures the _logic_ of the image, not just the _appearance_.
    

Code snippet

```
graph TD
    A[Input Image] --> B[VLM Perceiver]
    B --> C{Strategy Select}
    C -->|Vector/Logo| D
    C -->|Texture/Photo| E
    D --> F
    E --> G
    F --> H
    G --> H
    H --> I[Comparator (Loss)]
    I -->|Gradient Feedback| D
    I -->|Gradient Feedback| E
    I -->|Refinement Signal| B
    H --> J{Matches Ground Truth?}
    J -->|No| K
    K --> D
    J -->|Yes| L
    L --> M[Final Output: Editable Math/Code]
```

---

## 7. Data Infrastructure: The Fuel for the Paradigm

Training this pipeline requires a shift from "Image-Text" datasets (LAION-5B) to "Image-Code" datasets. Fortunately, the 2024-2025 period has seen an explosion in such resources.

### 7.1 The StarVector Ecosystem

The **StarVector** project has provided the foundational dataset for vector learning.

- **Volume:** **2.1 Million** SVG-Image pairs.
    
- **Content:** Contains `text2svg-stack`, which pairs SVGs with textual descriptions generated by multimodal models (BLIP2, CogVLM).
    
- **Complexity:** It moves beyond simple icons to include complex illustrations, charts, and user interface elements.
    
- **Significance:** This dataset is large enough to train the "Visual Perceiver" (Stage 1) to map visual features to SVG primitives with high fidelity.
    

### 7.2 Shaders21k and the Procedural Domain

For photorealistic and textural content, we rely on **Shaders21k**.

- **Content:** 21,000+ GLSL procedural shaders scraped from Shadertoy.
    
- **Nature:** These are small programs that generate infinite-resolution textures (clouds, water, fire, fractals) using math (SDFs, noise functions).
    
- **Role:** These serve as the training ground for the system to learn "Procedural Logic." The model learns that a "cloudy sky" is best represented not by a jpeg, but by `fbm(noise(x,y))`.
    

### 7.3 The "Desmos Gap" and Construction Strategy

A critical missing piece is a dataset of "pure" mathematical graphs (parametric curves, inequalities). **Desmos** is the gold standard platform for this art form. A specific recommendation of this report is the construction of a **Desmos1M** dataset.

- **Construction Method:**
    
    - **Source:** Desmos Art Contests, Reddit r/desmos (communities that create art using only math formulas).
        
    - **Scraping:** Use `Calc.getState()` from the Desmos API to retrieve the full JSON state (expressions, variables). Use `Calc.screenshot()` to get the ground truth image.
        
    - **Value:** This dataset teaches the model the direct link between mathematical syntax (e.g., $r = \cos(k\theta)$) and visual geometry (a rose curve). This is the purest form of Image-to-Math data.
        

### 7.4 VLMaterial

**VLMaterial** (ICLR 2025) provides 550,000 procedural material graphs (Blender node graphs). This connects visual surface properties (roughness, metallicity) to graph-based functional representations, essential for 3D workflows.

---

## 8. Strategic Comparison: Determinism vs. Diffusion

This section provides a rigorous comparative analysis for decision-makers evaluating this technology stack against the current industry standard.

|**Feature**|**Diffusion Model (Legacy)**|**Image-to-Math (New Paradigm)**|
|---|---|---|
|**Fundamental Unit**|Pixel / Latent Patch|Primitive / Function|
|**Resolution**|Fixed (Raster Dependent)|Infinite (Vector / Math)|
|**Reproducibility**|Stochastic (Random Seed)|Deterministic (Exact Formula)|
|**Editability**|Inpainting (Probabilistic Guesswork)|Parametric (e.g., `radius = 5`)|
|**Representation**|Black Box Weights|Human-Readable Code|
|**Training Data**|Unlabeled Images (Billions)|Paired Code/Image (Millions)|
|**Inference Speed**|Slow (Iterative Denoising, 20-50 steps)|Fast (Single Pass + Optimization)|
|**Compression**|Poor (Latent Representation is large)|Extreme (Code is bytes/kilobytes)|
|**Verification**|Subjective (Human/CLIP)|Objective (Computational/Pixel Loss)|
|**Upscaling**|Hallucination-based (AI upscaler)|Mathematical Re-rendering|

### 8.1 The Interpretability Advantage

In typical AI safety discussions, interpretability is a "nice to have" for auditing. In Image-to-Math, interpretability is a **core utility feature**.

- **Source Code as Deliverable:** The user gets the "source code" of the image. They can copy the SVG into Illustrator, copy the TikZ into Overleaf, or copy the Python into Blender. The AI becomes a tool for **authoring**, not just **rendering**.
    
- **Semantic Editing:** A user can ask, "Make the grid lines thinner." In diffusion, this is a prompt engineering nightmare that often changes the whole image. In Math, it is a deterministic variable change: `line_width = 0.5`.
    

---

## 9. Implementation Roadmap & Challenges

This report outlines a phased execution plan to realize the Visual Compiler.

### 9.1 Phase 1: The Vector Engine (Months 1-6)

- **Goal:** Replicate and integrate **Bézier Splatting** into a generative pipeline.
    
- **Action:** Train a NESYDM decoder on the **StarVector** dataset (2.1M images).
    
- **Objective:** Achieve real-time conversion of raster logos, icons, and illustrations to editable SVG.
    
- **Metric:** Compare reconstruction error (LPIPS) and topology accuracy against DiffVG benchmarks.
    
- **Risk:** Handling complex topologies (e.g., text conversion). _Mitigation:_ Use specialized OCR-to-Font modules.
    

### 9.2 Phase 2: The Math Engine (Months 6-12)

- **Goal:** Incorporate implicit functions and formulas for continuous tones.
    
- **Action:** Scrape/Build **Desmos1M** and process **Shaders21k**. Train models to output Python/GLSL code for textures (clouds, fire, noise).
    
- **Objective:** Train the model to decompose an image into "Structure" (Vector) + "Texture" (Shader).
    
- **Integration:** Combine Vector (for shapes) and Shaders (for textures) into a hybrid representation.
    

### 9.3 Phase 3: The Abstraction Layer (Year 2)

- **Goal:** Human-level code quality.
    
- **Action:** Implement **DreamCoder/LILO** loops.
    
- **Objective:** The model must self-refactor. It should not output 1000 lines of code if a `for` loop can do it in 3 lines. This requires "sleeping" phases where the model analyzes its own outputs to build a library of reusable functions.
    
- **Metric:** Minimum Description Length (MDL). The shorter the code (for the same visual output), the higher the "intelligence" of the model.
    

### 9.4 Key Risks and Mitigations

- **Risk:** **Spectral Bias.** Neural networks struggle with high-frequency textures.
    
    - _Mitigation:_ Use **SIREN** layers and Fourier Feature mappings which are mathematically proven to resolve this.
        
- **Risk:** **The "Photo" Barrier.** Some images (e.g., a photograph of a chaotic crowd) are too entropically complex for concise formulas.
    
    - _Mitigation:_ **Hybrid Fallback.** The system should degrade gracefully. It can use Math for the "structure" (buildings, horizon) and a standard Texture Map (or localized diffusion patch) for the "chaos" (crowd faces). The goal is to maximize the mathematical portion, not enforce it dogmatically where it fails.
        

---

## 10. Conclusion

The **Image-to-Math** paradigm is the logical, necessary next step in the evolution of generative computer vision. It represents a maturation from the "brute force" statistical correlation of pixels (Diffusion) to the "intellectual" understanding of visual structure (Math).

The convergence of **Bézier Splatting** for speed, **NESYDM** for neurosymbolic reasoning, and **Large Scale Datasets** (StarVector) creates the perfect window of opportunity to build this system now. The result will not just be another image generator, but a **Visual Compiler**: a system that reads the visual world and writes the code that describes it, unlocking infinite resolution, perfect reproducibility, and deep editability. This is the path to a truly rigorous visual intelligence.

---

**Citations used in this report:** .