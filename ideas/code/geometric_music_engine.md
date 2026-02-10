# The Geometric Music Engine
## A Neuro-Symbolic System for Structure-Guaranteed Music Generation

---

# Executive Summary

This document presents the **Geometric Music Engine (GME)** — a neuro-symbolic architecture that solves the fundamental limitation of current AI music generation: the **Structure Gap**. While statistical models (Suno, Udio, MusicLM) excel at generating locally coherent audio, they fail at maintaining long-range harmonic structure, formal coherence, and teleological direction. They are "Markovian drunks" — locally plausible, globally aimless.

The GME inverts the paradigm. Instead of deriving structure from statistical correlations in audio data, it **imposes structure via geometric constraints** on pitch-class lattices (Tonnetz, voice-leading orbifolds, Zeitnetze) and leverages neural networks solely for acoustic rendering. Music becomes a **pathfinding problem** rather than a prediction problem.

**The Core Insight**: This is the **AlphaGo of music**. Symbolic systems for music (OpenMusic, music theory) have existed for decades. Neural audio generation has existed for a decade. Neither alone produces coherent long-form music. The synthesis — geometry for structure, neural nets for sound — is the breakthrough.

**Strategic Positioning**: Not a consumer product. An exclusive licensing tool for major record labels. The KLAY Vision precedent (November 2025: licensing deals with Universal, Sony, and Warner) validates that labels want controllable, explainable, legally clean AI. The GME is architected for exactly this.

---

# Part I: The Problem — Why Statistical Models Fail

## 1.1 The Structure Gap

Academic research documents persistent failures in long-term coherence across all major AI music systems:

| Model | Architecture | Documented Failures | Root Cause |
|-------|--------------|---------------------|------------|
| **Suno v3/v4** | Transformer + Diffusion | Genre drift, "jarring jumps" between sections, redundant chorus repetition | Implicit harmony learning; no explicit state tracking |
| **Udio** | Diffusion Spectrogram | High fidelity but weak formal arcs; "hallucinates" key changes | Acoustic texture prioritized over symbolic logic |
| **MusicLM** | Hierarchical Transformer | "Struggles to capture precise alignments," fails on long-range dependencies | Semantic tokens capture mood, not functional harmony |
| **MusicGen** | Autoregressive Transformer | "Forgets primers immediately," lacks global planning | Next-token prediction is locally greedy |

**The Quote**: "Autoregressive models seem to 'forget' about the primer almost immediately... the lack of long-term structure is apparent." — Google Magenta, Music Transformer

**Root Cause**: All current systems learn harmony *implicitly* from audio tokens. They capture surface patterns but fail on "higher-level features like tonal functions" (Toiviainen). No commercial system uses explicit music theory.

## 1.2 Why This Matters

Music is not random. Human composers plan:
- **Teleological arcs**: Tension builds toward climax, then resolves
- **Formal structures**: Verse-chorus-bridge, sonata-allegro, 12-bar blues
- **Harmonic grammar**: $V^7 \to I$ is not just statistically probable — it's functionally required

Statistical models approximate this grammar through correlation but do not *comprehend* it. They cannot be directed to "build harmonic tension using chromatic mediants for 16 bars." They behave like savants — capable of mimicking style but incapable of following explicit structural direction.

## 1.3 The Scarcity Paradox

As the marginal cost of generating high-fidelity *sound* approaches zero, the value of *structure* — coherent, meaningful, emotionally resonant arrangement — approaches infinity. When "slop" is infinite, curated structure becomes the premium asset.

---

# Part II: Mathematical Foundations

## 2.1 Pitch-Class Space ($\mathbb{Z}_{12}$)

Western 12-tone equal temperament forms the cyclic group:

$$\mathbb{Z}_{12} = \{0, 1, 2, \ldots, 11\}$$

| Pitch Class | Integer | Role in C Major |
|-------------|---------|-----------------|
| C | 0 | Tonic |
| D | 2 | Supertonic |
| E | 4 | Mediant |
| F | 5 | Subdominant |
| G | 7 | Dominant |
| A | 9 | Submediant |
| B | 11 | Leading Tone |

**Chords as Sets**: A C Major triad is $\{0, 4, 7\}$ — a geometric object (triangle) in $\mathbb{Z}_{12}$.

## 2.2 The T/I Group ($D_{12}$)

The operations of **Transposition** ($T_n$) and **Inversion** ($I_n$) form the **Dihedral Group $D_{12}$** (order 24):

$$T_n(x) = x + n \pmod{12}$$
$$I_n(x) = -x + n \pmod{12}$$

**Properties**:
- Isomorphic to the symmetries of a regular 12-gon
- $T_1$ = rotation by 30°; $I_0$ = reflection across C/F♯ axis
- Non-abelian: $T_n \circ I_m \neq I_m \circ T_n$

**Musical Meaning**: Major and minor triads are *chiral reflections* — geometric mirror images under inversion.

**⚠️ Common Error**: The T/I group is $D_{12}$ (subscript = polygon order), NOT $D_{24}$ (which would have order 48).

## 2.3 Neo-Riemannian Theory: The PLR Group

The **PLR Group** models smooth voice-leading between the 24 major/minor triads:

| Operation | Name | Action | Example | Voice Movement |
|-----------|------|--------|---------|----------------|
| **P** | Parallel | Major ↔ Minor (same root) | C → Cm | Third moves by semitone (E → E♭) |
| **L** | Leading-tone | Root moves by semitone | C → Em | Root moves (C → B) |
| **R** | Relative | Fifth moves by whole tone | C → Am | Fifth moves (G → A) |

**Group Structure**: PLR ≅ $D_{12}$ (order 24), acting simply transitively on the 24 triads.

**Algebraic Relation**: $P = R(LR)^3$ — the group is generated by L and R alone.

**Subgroups**:
- $\langle L, P \rangle \cong S_3$ (order 6) — hexatonic systems
- $\langle P, R \rangle$ (order 8) — octatonic systems

**⚠️ Common Error**: PLR is NOT $S_4 \times \mathbb{Z}_2$ (order 48). It is strictly $D_{12}$ (order 24).

## 2.4 The Tonnetz

The **Tonnetz** (Tone Network) is a simplicial complex where:
- **Vertices**: Pitch classes
- **Edges**: Consonant intervals (P5 horizontal, M3 diagonal, m3 counter-diagonal)
- **Faces**: Triangles representing major and minor triads

**PLR as Geometric Flips**:
- P = flip across the P5 edge (root–fifth)
- L = flip across the m3 edge (third–fifth)
- R = flip across the M3 edge (root–third)

**Topology**: The Tonnetz wraps into a **torus** ($T^2$) via octave/enharmonic equivalence.

**Computational Advantage**: A chord progression becomes a path on the dual graph ("Chicken-wire Torus"). Modulation from C Major to F♯ Major — classically "distant" — is a specific geodesic path with quantifiable cost.

## 2.5 Voice-Leading Orbifolds (Tymoczko)

For extended chords (7ths, 9ths, clusters), the Tonnetz is insufficient. Tymoczko's voice-leading geometries model chords as points in **orbifolds** — quotient spaces $\mathbb{T}^n / S_n$:

- 2-note chords inhabit a **Möbius strip**
- 3-note chords inhabit a **twisted triangular prism** with mirror boundaries

**Voice Leading = Line Segments**: The "work" of a progression is the Manhattan distance:

$$d(A, B) = \sum_{i=1}^{n} |a_i - b_i|$$

| Transformation | Voice Movement | Manhattan Distance |
|----------------|---------------|-------------------|
| P (C → Cm) | E → E♭ | 1 |
| L (C → Em) | C → B | 1 |
| R (C → Am) | G → A | 2 |
| Slide (C → C♯m) | C → C♯, G → G♯ | 2 |

## 2.6 The Rhythm Gap: Zeitnetze

**Critical Limitation**: The Tonnetz handles pitch, not time.

**Solution**: **Zeitnetze** (Time Networks) apply the same group-theoretic principles to rhythm:
- **Beat-Class Sets**: Rhythms modeled as subsets of $\mathbb{Z}_n$ (e.g., $\mathbb{Z}_{16}$ for 16th notes)
- **DFT for Rhythm**: Emmanuel Amiot's application of the Discrete Fourier Transform quantifies "groove," "evenness," and "syncopation"

**Implementation**: The Architect layer includes a parallel Rhythmic Graph:
- **Nodes**: Rhythmic archetypes (clave, swing, four-on-the-floor, breakbeat)
- **Edges**: Transformations that add/subtract onsets or shift phases
- **Coupling**: Tension curves dictate both harmonic dissonance AND rhythmic density/syncopation

**Status**: Theoretical framework exists (Amiot); no complete implementation. This is original research territory.

## 2.7 Limitations of Geometric Models

**What geometry does NOT capture**:
1. **Timbre**: Sound color is orthogonal to pitch geometry
2. **Cultural context**: Geometric proximity ≠ perceptual proximity
3. **Melodic contour**: Voice-leading geometry doesn't distinguish resolutions from other stepwise motion
4. **Non-12-TET systems**: Microtonal, just intonation require different lattices
5. **The "soul" problem**: Technical correctness ≠ emotional resonance

---

# Part III: The Geodesic Audio Stack

```
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 1: THE DIRECTOR (Intent Interface)                      │
│  ──────────────────────────────────────────                     │
│  • Drawable tension curves T(t) ∈ [0,1]                         │
│  • Entropy/complexity curves E(t) ∈ [0,1]                       │
│  • Formal markers (cadences, key changes, climax points)        │
│  • Style/genre conditioning                                     │
│                                                                 │
│  OUTPUT → Geometric constraints for the Architect               │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 2: THE ARCHITECT (Geometric Brain)                       │
│  ──────────────────────────────────────────                     │
│  • Tonnetz graph (24 triads, PLR edges)                         │
│  • A*/Dijkstra pathfinding under constraints                    │
│  • GNN-learned edge weights for style-specific navigation       │
│  • Coupled rhythmic Zeitnetz for joint optimization             │
│                                                                 │
│  TECH: networkx, music21, custom CSP solver, ChordGNN           │
│                                                                 │
│  OUTPUT → Symbolic skeleton (MIDI/MusicXML)                     │
│           Mathematically guaranteed coherent progression        │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 3: THE PERFORMER (Neural Renderer)                       │
│  ──────────────────────────────────────────                     │
│  • MIDI-DDSP / RenderBox for expressive synthesis               │
│  • Humanization: 1/f noise in timing, dynamics, articulation    │
│  • Latent Diffusion + ControlNet for full-band rendering        │
│  • Strategic imperfection injection                             │
│                                                                 │
│  OUTPUT → High-fidelity audio                                   │
└─────────────────────────────────────────────────────────────────┘
```

## 3.1 Layer 1: The Director

**The Key Insight**: Most AI music tools ask "what notes?" or "what genre?" We ask **"what emotional arc?"**

**Interface Primitives**:
- **Tension Curve $T(t)$**: Normalized intensity over time
- **Entropy Curve $E(t)$**: Predictability vs. surprise
- **Harmonic Rhythm**: Chord change frequency
- **Formal Markers**: "Climax at 75%," "Authentic cadence at bar 32"

**Constraint Translation**:
```
T(0.6) = 0.8  →  "Maximize d_L1(chord, tonic) at bar 24"
E(0.9) = 0.2  →  "Use only diatonic chords in final section"
```

**This is the primary moat.** The tension-curve-to-geometric-constraint translation is novel and non-obvious.

## 3.2 Layer 2: The Architect

**Core Algorithm**: Constrained shortest-path on weighted Tonnetz graph.

**Graph Construction**:
- **Nodes**: 24 triads (basic), 96+ for 7ths (jazz), arbitrary pitch-class sets (contemporary)
- **Edges**: PLR + extensions (S, N, H, tritone sub, etc.)
- **Weights**: Style-dependent
  - *Classical*: Parallel fifths = ∞ weight; PLR = low weight
  - *Jazz*: Tritone subs = low weight; extensions encouraged

**Constraint Types**:
1. **Boundary**: Start/end chords
2. **Waypoint**: "Pass through vi at bar 8"
3. **Distance**: "Peak tension (max distance from I) at bar 16"
4. **Exclusion**: "No diminished chords" (for pop)
5. **Cadential**: "V-I at phrase boundaries"

**A* Heuristic**:
```python
def tension_heuristic(current_node, target_node, current_bar, tension_curve):
    desired_tension = tension_curve[current_bar]
    distance_from_tonic = shortest_path_length(current_node, tonic)
    tension_error = abs(distance_from_tonic - (desired_tension * MAX_DISTANCE))
    return tension_error
```

**GNN Integration**: Use ChordGNN/MusGConv trained on specific corpora (Beatles, Bach) to *learn* edge weights. A* then navigates this "learned manifold" — composing "like Bach" by navigating harmonic space using Bach's specific "gravity."

## 3.3 Layer 3: The Performer

**Current State of MIDI-to-Audio (2025)**:

| Technology | Maturity | Capabilities | Limitations |
|------------|----------|--------------|-------------|
| **MIDI-DDSP** | Research/Demo | Realistic monophonic instruments | Monophonic only |
| **RenderBox** | Emerging | Expressive performance from MIDI + text | Early stage |
| **Music ControlNet** | Early Research | Melody/rhythm conditioning | Coarse control |
| **Stable Audio + ControlNet** | Prototype | 44.1kHz stereo | CQT-based, not precise MIDI |

**Critical Requirement — Strategic Imperfection**:

Research shows geometric perfection can feel sterile. The Performer must inject:
- **Micro-timing deviations** (rubato, groove)
- **Dynamic phrasing** (crescendo, accent)
- **1/f noise** in velocity and articulation

**Pragmatic Approach**: Phase 1-2 outputs MIDI. Users render in DAW. Neural rendering is Phase 3.

---

# Part IV: Empirical Research Findings

## 4.1 Does Geometric Music Sound Good?

Research reveals a nuanced picture:

| Finding | Source | Implication |
|---------|--------|-------------|
| **Listeners attend to 5-10 notes**, not 100-bar forms | Cortical Melodic Predictions (PMC 2023) | Long-range structure matters less than local coherence |
| **Peak pleasure = low uncertainty + high surprise** | Cheung et al. (Current Biology 2019) | Optimal *violations* of rules create engagement |
| **Technical perfection feels sterile** | Avdeeff 2019; Stammer 2025 | Must inject controlled imperfection |
| **Familiarity > complexity** | Repeated Listening Study (PMC 2017) | Repeated exposure matters more than structural sophistication |
| **Voice leading facilitates processing** but doesn't maximize pleasure | Wall et al. (Nature Sci Rep 2020) | VL is necessary but not sufficient |

**Critical Insight**: Geometric validity is *necessary but not sufficient*. The system must introduce strategic rule-breaking.

## 4.2 Existing Neuro-Symbolic Music Systems

| System | Status | Approach | Results |
|--------|--------|----------|---------|
| **MorpheuS** | Working | VNS + tension profiles | Live orchestra performances; proves tension curves work |
| **NotaGen** (2025) | Working | GPT-2 + CLaMP-DPO | Outperformed baselines in A/B tests; pianist performed output |
| **RL-Tuner + CP** | Research | RL + constraint programming | 4% improvement in constraint satisfaction |
| **Parley** | Research | Rule-based + neural listening | Generates MIDI + scores |

**Gap Identified**: No system combines **Tonnetz pathfinding + GNN-learned weights + tension curves + neural rendering**. The niche is open.

## 4.3 What Labels Want

From KLAY Vision deals and WMG's Robert Kyncl (Nov 2025):

| Requirement | Description |
|-------------|-------------|
| **Licensed training data** | No scraping; legal clarity |
| **Artist consent** | Opt-in for voice/likeness |
| **Revenue sharing** | Artists paid for training + outputs |
| **Attribution** | Proper credit systems |
| **Explainability** | Must be able to audit why a progression was chosen |
| **Control** | Inference-time adjustment without regeneration |

**The GME satisfies all of these by design.** Geometric paths are explainable. Tension curves provide control. The symbolic layer is not trained on copyrighted audio.

---

# Part V: Competitive Landscape

## 5.1 Major Labs — Paradigm Blindness

| Lab | Approach | Geometry/Theory? |
|-----|----------|------------------|
| **DeepMind (Lyria)** | End-to-end diffusion/transformer | ❌ |
| **Meta (MusicGen)** | Autoregressive transformer | ❌ |
| **OpenAI** | No active music product | — |
| **Suno/Udio** | Transformer + diffusion | ❌ |

**No major lab is pursuing structure-first geometry.** They are trapped in the "scale solves everything" paradigm. The inversion (structure first, then render) is not obvious to people trained to let models learn everything.

## 5.2 Academic Research

| Research | Status | Threat Level |
|----------|--------|--------------|
| **ProGress (2025)** | Schenkerian analysis + graph diffusion | Medium — different approach (Schenkerian, not Tonnetz) |
| **IRCAM PhD** | Geometric deep learning for audio | Low — not generation-focused |
| **ChordGNN / MusGConv** | GNN for music analysis | Low — analysis, not generation |

**The intersection of Tonnetz + GNN + generation is unoccupied.**

## 5.3 Symbolic Tools

| Tool | Language | Learning Curve | Status |
|------|----------|----------------|--------|
| **OpenMusic** | Lisp | Very High | Academic niche |
| **OM#** | Lisp | High | Active development |
| **music21** | Python | Medium | Analysis-focused |

**OpenMusic has the right math but catastrophic UX.** This is the cautionary tale — and the opportunity.

---

# Part VI: Strategic Analysis

## 6.1 Market Reality

**Music Industry**: ~$30B annually
- Labels take ~70%
- Streaming ~20%
- Artists ~10%

**Comparison**: Enterprise AI = $500B+; Healthcare AI = $100B+

**Implication**: This is not a scale opportunity. It is an **influence opportunity**.

## 6.2 The Influence Frame

Music is infrastructure for culture. Whoever controls music generation controls:
- The emotional texture of advertising, film, games
- TikTok sounds (which drive trends)
- Background music for everything
- What "now" sounds like

**The value is positional, not transactional.**

## 6.3 Power Dynamics

If a frontier AI lab builds hit-quality music generation:

| Party | Position |
|-------|----------|
| **AI Lab** | Has the tech. Can go direct. Doesn't need labels. |
| **Labels** | Can't build it. Can't compete. Must negotiate. |
| **Artists** | Fragmented. No collective leverage. |

Labels become distribution partners, not gatekeepers. **You set terms.**

## 6.4 The Math-Only Advantage

The GME does NOT train on copyrighted audio for composition. It uses:
- Group theory ($D_{12}$, PLR) — public math
- Graph algorithms (A*, Dijkstra) — commodity
- Music theory (Tonnetz) — 19th century

**Zero copyright exposure** on the symbolic layer. Only the neural renderer touches audio, and that can use synthetic/licensed data.

This sidesteps the entire Suno/Udio lawsuit exposure.

## 6.5 Moat Analysis

**Thin Moat (Replicable)**:
- Tonnetz math (public)
- A* algorithm (commodity)
- music21/networkx (open source)

**Defensible Moat (Hard to Replicate)**:

| Moat | Mechanism |
|------|-----------|
| **First-mover synthesis** | No one else is building the full stack |
| **Label contracts** | Exclusive deals lock out competitors |
| **Proprietary training data** | Multitrack masters from label deals |
| **Taste prediction models** | Predicting hits is harder than generating music |
| **Constraint language IP** | Tension-curve → geometric-constraint translation |
| **Non-publication** | If you don't publish, competitors must independently rediscover |

**The moat is not the Tonnetz. It's execution + relationships + being first.**

## 6.6 The KLAY Vision Precedent (Verified)

**November 20, 2025**: KLAY Vision signed licensing deals with all three major labels (UMG, Sony, Warner) and their publishing arms.

**Leadership**:
- CEO: Ary Attie (musician)
- CCO: Thomas Hesse (former President, Sony Music Global Digital)
- Chief AI Officer: Björn Winckler (former DeepMind music lead)
- CTO: Brian Whitman (former Spotify/Echo Nest founder)

**Model**: "Large Music Model" trained on licensed data. Positions as "force multiplier for artistry," not replacement.

**Implication**: Labels will work with AI companies that offer control, attribution, and legal clarity. The GME is architected for exactly this.

---

# Part VII: Probability Assessment

| Outcome | Probability | Reasoning |
|---------|-------------|-----------|
| Produces better structure than statistical models | **80%** | MorpheuS already proves tension curves work |
| Produces emotionally preferred music | **50%** | Requires surprise injection + imperfection; not guaranteed |
| Labels adopt it | **60%** | KLAY precedent validates market; GME fits requirements |
| Becomes dominant paradigm | **30%** | Requires execution + timing + competitors staying blind |

---

# Part VIII: Execution Roadmap

## Phase 1: Proof of Concept (3-6 months)

**Goal**: Validate that geometric constraints produce perceptibly better structure.

**Deliverables**:
1. Python Tonnetz navigator (networkx + music21)
2. Constraint solver for harmonic paths
3. Tension-curve input → MIDI output
4. A/B listener study: constrained vs. unconstrained

**Success Criteria**: Statistically significant listener preference for constrained outputs.

**Output**: Internal demo. Do NOT publish.

## Phase 2: Rhythmic Extension (6-12 months)

**Goal**: Solve the rhythm gap.

**Research Questions**:
- Can metric hierarchies be modeled as a lattice?
- How do harmonic rhythm and metric structure interact?
- Can GNNs learn style-specific rhythmic patterns?

**Deliverables**:
1. Zeitnetz formalism
2. Joint harmonic-rhythmic pathfinding
3. Style transfer for rhythmic patterns

**Status**: Original research. No established solution.

## Phase 3: Neural Integration (12-24 months)

**Goal**: End-to-end audio generation.

**Dependencies**:
- MIDI-DDSP polyphonic extension
- ControlNet for audio maturation
- Label partnership for training data

**Alternative**: Partner with existing renderer rather than building from scratch.

## Phase 4: Label Partnership (18-36 months)

**Goal**: Commercial deployment.

**Prerequisites**:
- Working demo generating radio-quality output
- Legal framework for AI-generated music rights
- Relationship-building with label innovation teams

**Structure**: Exclusive pilot with one major label. Prove value. Expand.

**Target**: Innovation arms of UMG or Warner (teams that negotiated Udio/KLAY deals).

---

# Part IX: Technical Implementation

## 9.1 Tonnetz Graph Construction

```python
import networkx as nx
from music21 import chord

class Tonnetz:
    def __init__(self):
        self.graph = nx.Graph()
        self.triads = self._generate_triads()
        self._build_graph()

    def _generate_triads(self):
        # Generate all 24 Major and Minor triads
        # Represented as frozensets: frozenset({0, 4, 7}) for C Major
        triads = []
        for root in range(12):
            major = frozenset({root, (root + 4) % 12, (root + 7) % 12})
            minor = frozenset({root, (root + 3) % 12, (root + 7) % 12})
            triads.extend([major, minor])
        return triads

    def _apply_P(self, triad):
        # Parallel: Major ↔ Minor (third moves by semitone)
        root = min(triad)
        if (root + 4) % 12 in triad:  # Major
            return frozenset({root, (root + 3) % 12, (root + 7) % 12})
        else:  # Minor
            return frozenset({root, (root + 4) % 12, (root + 7) % 12})

    def _apply_L(self, triad):
        # Leading-tone exchange: root moves by semitone
        # C Major → E Minor; E Minor → C Major
        ...

    def _apply_R(self, triad):
        # Relative: fifth moves by whole tone
        # C Major → A Minor; A Minor → C Major
        ...

    def _build_graph(self):
        for triad in self.triads:
            self.graph.add_node(triad)
            self.graph.add_edge(triad, self._apply_P(triad), weight=1.0, transform='P')
            self.graph.add_edge(triad, self._apply_L(triad), weight=1.0, transform='L')
            self.graph.add_edge(triad, self._apply_R(triad), weight=2.0, transform='R')
```

## 9.2 A* with Tension Heuristic

```python
def a_star_with_tension(graph, start, goal, tension_curve, tonic):
    """
    A* search where cost includes tension curve matching.
    
    Args:
        graph: Tonnetz graph
        start: Starting triad
        goal: Target triad
        tension_curve: Dict[bar_number -> desired_tension (0-1)]
        tonic: The tonic triad for distance calculations
    """
    import heapq
    
    open_set = [(0, start, [start])]
    visited = set()
    
    while open_set:
        f_score, current, path = heapq.heappop(open_set)
        
        if current == goal:
            return path
        
        if current in visited:
            continue
        visited.add(current)
        
        current_bar = len(path)
        
        for neighbor in graph.neighbors(current):
            if neighbor in visited:
                continue
            
            # g(n): cumulative voice-leading work
            edge_weight = graph[current][neighbor]['weight']
            g_score = len(path) * edge_weight
            
            # h(n): tension matching + distance to goal
            desired_tension = tension_curve.get(current_bar, 0.5)
            distance_from_tonic = nx.shortest_path_length(graph, neighbor, tonic)
            max_distance = 6  # Maximum PLR distance on Tonnetz
            
            tension_error = abs(distance_from_tonic / max_distance - desired_tension)
            goal_distance = nx.shortest_path_length(graph, neighbor, goal)
            
            h_score = tension_error + goal_distance * 0.5
            
            f_score = g_score + h_score
            heapq.heappush(open_set, (f_score, neighbor, path + [neighbor]))
    
    return None  # No path found
```

---

# Part X: Open Research Questions

1. **Rhythmic Formalism**: What is the geometric structure of meter and groove? How do we build a working Zeitnetz pathfinder?

2. **Surprise Injection**: How do we systematically introduce optimal rule violations? What's the math for "just the right amount of unexpectedness"?

3. **Taste Prediction**: Can we model cultural resonance? What features predict hits vs. flops? This may be the deepest moat.

4. **Evaluation Metrics**: How do we rigorously measure "coherence" and "emotional resonance"? Listener studies are expensive and subjective.

5. **Polyphonic Rendering**: When will MIDI-DDSP or equivalent support full-band polyphonic synthesis?

6. **Non-Western Systems**: Can the framework extend to maqam, raga, or other non-12-TET traditions?

---

# Part XI: Key References

## Academic Papers
- Tymoczko, D. (2006). "The Geometry of Musical Chords." *Science*.
- Cohn, R. (1998). "An Introduction to Neo-Riemannian Theory." *Journal of Music Theory*.
- Thickstun et al. (2024). "Anticipatory Music Transformer." *TMLR*.
- Wu et al. (2022). "MIDI-DDSP." *ICLR*.
- Amiot, E. (2016). "Music Through Fourier Space." Springer.
- Cheung et al. (2019). "Uncertainty and Surprise Jointly Predict Musical Pleasure." *Current Biology*.

## Software
- music21: https://web.mit.edu/music21/
- networkx: https://networkx.org/
- HexaChord: https://louisbigo.com/hexachord
- TonnetzViz: https://cifkao.github.io/tonnetz-viz/

## Industry Sources
- KLAY Vision Announcements (Nov 2025): UMG, Sony, Warner press releases
- WMG Robert Kyncl blog (Nov 2025): AI licensing principles
- PRS for Music AI Survey (2026): Creator sentiment

---

# Conclusion

The **Geometric Music Engine** represents the necessary evolution of AI music from "stochastic mimicry" to "structural composition." The current generation of statistical models has hit a hard ceiling of coherence that more data cannot solve. By explicitly modeling music as a geometric pathfinding problem, the GME introduces the teleology and grammar that these models lack.

**Why this could be big**:
1. **The problem is real**: Structure Gap is documented and unsolved
2. **The solution is sound**: Geometry + neural is the AlphaGo pattern
3. **The competition is blind**: Major labs are chasing scale, not structure
4. **The market is ready**: KLAY proves labels want controllable, explainable AI
5. **The moat is execution**: First to synthesize and license wins

**Why it might not work**:
1. **Geometric validity ≠ emotional resonance**: The "soul" problem is real
2. **Rhythm is unsolved**: No working Zeitnetz implementation exists
3. **Neural rendering is immature**: MIDI-to-audio is early stage
4. **Listener perception is complex**: Short-range attention may dominate

**The Test**: Can Phase 1 produce output that makes a listener *feel* something — not just "is it grammatically correct" but "does it move me"?

If yes → the opportunity is real.

**Execute.**

---

*"Structure eats scale — but only if the structure has soul."*
