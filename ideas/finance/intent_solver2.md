# Technical Architecture of CowSwap/UniswapX Solvers and Neuro-Symbolic Optimization for Batch Auctions

## 1. Introduction: The Transition to Solver-Centric Execution

The evolution of decentralized finance (DeFi) execution architectures has reached a critical inflection point, transitioning from the deterministic, on-chain routing logic that characterized the first generation of Automated Market Makers (AMMs) to a probabilistic, off-chain optimization paradigm driven by **solvers**. This shift is not merely an incremental improvement in trade execution but a fundamental restructuring of the market's topology, moving from a "dumb pipe" model of smart contract interaction to an "intent-centric" architecture where complex agents compete to satisfy user desires.

In the nascent stages of DeFi, protocols like Uniswap v1 and v2 operated on a principle of rigid, sequential execution. A user wishing to swap Token A for Token B would interact directly with a smart contract, the logic of which was immutable and entirely visible on-chain. The path of execution was predetermined by the state of the liquidity pools at the exact moment the transaction was mined. While transparent, this model introduced significant inefficiencies, primarily in the form of slippage, high gas costs for failed transactions, and, most perniciously, susceptibility to Maximal Extractable Value (MEV). Sophisticated searchers could monitor the public memory pool (mempool), observe pending user transactions, and algorithmically insert their own transactions before and after the victim's trade (sandwich attacks) to extract value.

The industry's response to these structural inefficiencies has been the development of **Order Flow Auctions (OFAs)** and **Batch Auctions**, exemplified by protocols such as **CowSwap** and **UniswapX**. In these systems, users do not submit transactions; they broadcast **intents**—signed messages cryptographically authorizing a state transition (e.g., "I am willing to sell 1 ETH if I receive at least 2000 USDC"). The responsibility for executing these intents—finding the liquidity, calculating the route, paying the gas, and managing the slippage risk—is delegated to a new class of network participants: **solvers** (in the context of CowSwap) or **fillers** (in the context of UniswapX).

This report provides an exhaustive technical analysis of the architectures underpinning these solver-based systems. It explores the profound computational challenges inherent in **Multi-Token Batch Clearing**, a problem that maps to the NP-hard domain of combinatorial optimization. To navigate this complexity within the strict latency constraints of Ethereum block times (12 seconds), modern solver architectures are increasingly adopting **Neuro-Symbolic Optimization**—a hybrid artificial intelligence approach that fuses the pattern-recognition capabilities of **Graph Neural Networks (GNNs)** with the rigorous logical guarantees of **Mixed Integer Linear Programming (MILP)** solvers.

Furthermore, this analysis dissects the specific software infrastructure that enables these high-frequency optimization loops, identifying the **Rust** programming language ecosystem—specifically libraries such as `barter-rs` for data ingestion and `revm` for execution simulation—as the de facto standard for solver engineering. We will also examine the emerging security and privacy frontiers, including the integration of **Trusted Execution Environments (TEEs)** like Intel SGX via Flashbots SUAVE to create verifiable, private solving environments , and the potential vulnerabilities of financial GNNs to adversarial attacks and oracle manipulation.

---

## 2. Theoretical Foundations: The Mathematics of Batch Auctions

### 2.1 The Combinatorial Optimization Problem

The core distinction between a traditional AMM router and a CowSwap solver lies in the scope of optimization. A router optimizes a single trade path (User A $\to$ Token B) against a snapshot of liquidity. A batch auction solver, however, must optimize a set of $N$ orders simultaneously against a set of $M$ liquidity sources, while also considering the possibility of Peer-to-Peer (P2P) matches between users. This transforms the routing problem into a **global surplus maximization problem**.

In the context of CowSwap, the "Coincidence of Wants" (CoW) phenomenon allows orders to be matched directly against each other. If User A sells ETH for USDC, and User B sells USDC for ETH, the protocol matches them directly at the oracle price, bypassing the AMM fee and slippage entirely. When expanded to multi-hop ring trades (e.g., A sells ETH for DAI $\to$ B sells DAI for USDC $\to$ C sells USDC for ETH), finding the optimal configuration of matches becomes a complex graph theory problem.

The problem is formally modeled as a **Mixed Integer Linear Program (MILP)**. The objective function $J(x)$ is typically defined to maximize the total economic surplus of the batch:

$$\text{Maximize } \sum_{i \in \text{Orders}} (u_i(x_i) - c_i(x_i)) - \sum_{j \in \text{TXs}} \text{GasCost}_j$$

Where:

- $u_i(x_i)$ represents the utility derived by order $i$ (e.g., the amount of buy-token received).
    
- $c_i(x_i)$ represents the cost incurred (e.g., the amount of sell-token sent).
    
- The gas cost term penalizes complex execution paths, ensuring that the solver only selects intricate routes (like 4-hop ring trades) if the economic surplus generated outweighs the additional computation and storage costs on the blockchain.
    

### 2.2 Constraints and Uniform Clearing Prices (UCP)

The optimization is subject to a rigorous set of constraints that define the validity of a solution.

**1. Conservation of Flow (Kirchhoff's Law):**

For every token in the batch and every node (user or pool) in the graph, the total inflow must equal the total outflow (plus any net change in balance allowed by the order).

$$\sum_{\text{incoming edges}} \text{amount}_{\text{in}} = \sum_{\text{outgoing edges}} \text{amount}_{\text{out}}$$

This ensures that the solver cannot create or destroy tokens out of thin air.

**2. Limit Price Constraints:**

For every user order $i$, the realized exchange rate must be strictly better than or equal to the signed limit price.

$$\frac{\text{BuyAmount}_i}{\text{SellAmount}_i} \ge \text{LimitPrice}_i$$

**3. Uniform Clearing Prices (UCP):**

This is the most mathematically restrictive constraint in the CowSwap protocol. UCP dictates that within a single batch, all orders trading the same token pair must clear at the same price. If User A and User B are both selling ETH for USDC in the same block, they must receive the exact same exchange rate, regardless of their order size (assuming sufficient liquidity). This prevents "ordering" attacks and ensures fairness.

$$\forall i, j \in \text{Orders}(A \to B): P_i = P_j = P_{\text{clearing}}$$

This constraint linearizes the price space but complicates the solver's task, as it removes the degree of freedom to price discriminate based on order size.

**4. AMM Constant Function Invariants:**

For liquidity sourced from AMMs (e.g., Uniswap v2), the solver must respect the invariant $x \cdot y = k$. Since this is a non-linear constraint (convex curve), it cannot be directly input into a Linear Programming (LP) solver.

- **Linearization:** Solvers approximate the curve using piecewise linear segments. A specific arc of the curve is approximated by a set of chords.
    
- **Binary Variables:** Integer variables (binary flags) are used to select which linear segment is "active" for a given trade. The precision of the solver depends on the number of segments used; more segments yield better pricing but increase the number of binary variables, exponentially increasing the solve time.
    

### 2.3 Complexity Class and the Need for Heuristics

The resulting MILP, with its combination of continuous variables (amounts) and integer variables (segment selection, order inclusion flags), falls into the **NP-hard** complexity class. Finding the provably optimal solution for a large batch (e.g., 50 orders, 20 tokens) can theoretically take hours or days using standard Branch-and-Bound algorithms.

However, the Ethereum blockchain produces a new block every 12 seconds. Solvers have a strict deadline: they must submit their solution transaction before the block is proposed. This creates a hard real-time constraint. A solver that finds the "perfect" solution in 13 seconds is worthless; a solver that finds a "good enough" solution in 10 seconds wins the reward. This tension between **Optimality** and **Latency** is the driving force behind the adoption of **Neuro-Symbolic** architectures.

---

## 3. Neuro-Symbolic Architecture: Bridging Intuition and Logic

To overcome the NP-hard bottleneck, modern solver architectures are hybridizing classic operations research techniques with deep learning. This approach, termed **Neuro-Symbolic AI**, uses neural networks to guide the symbolic search process, effectively giving the solver an "intuition" about where the optimal solution lies.

### 3.1 Graph Neural Networks (GNNs) for MILP Representation

The first challenge in applying Deep Learning to this domain is data representation. A batch auction is not an image (grid data) or a sentence (sequence data); it is an unordered set of relations. The number of orders and tokens varies from block to block. **Graph Neural Networks (GNNs)** are the natural architecture for this problem because they are permutation-invariant and can handle variable-sized graphs.

The MILP is represented as a **Bipartite Graph** (specifically, a variable-constraint graph):

- **Variable Nodes ($V$):** Represent the decision variables (e.g., "Amount of Order $i$ to route to Pool $j$").
    
- **Constraint Nodes ($C$):** Represent the constraints (e.g., "Balance of User $X$ $\ge$ 0").
    
- **Edges ($E$):** Connect a variable node $v$ to a constraint node $c$ if variable $v$ appears in constraint $c$. The edge weight corresponds to the coefficient in the constraint matrix.
    

This graph structure completely captures the mathematical essence of the optimization problem. The GNN processes this graph through several layers of **Message Passing**, where nodes aggregate information from their neighbors.

- A Variable Node learns about the tightness of the constraints it is involved in.
    
- A Constraint Node learns about the state of the variables that satisfy it.
    

After $k$ layers of message passing, the network produces an embedding vector for each variable, encoding its "importance" to the global solution.

### 3.2 "Learning to Branch": Accelerating the Search

Standard solvers like **SCIP** or **Gurobi** solve MILPs using **Branch-and-Bound (B&B)**. They relax the integer constraints (allowing fractional values), solve the easy linear relaxation, and then "branch" by forcing an integer variable to be either $\le \lfloor x \rfloor$ or $\ge \lceil x \rceil$, creating two child problems.

The efficiency of B&B is entirely dependent on the **Branching Rule**: which variable do we split on first? A good choice (e.g., branching on whether to include a massive "whale" order) splits the search space effectively. A bad choice leads to a massive, unbalanced tree.

Traditional heuristics (like Strong Branching) are accurate but computationally expensive, requiring trial solves at every node.

**Neuro-Symbolic solvers replace the branching heuristic with a policy network learned by the GNN.**

- **Mechanism:** At each node of the search tree, the GNN predicts which variable will lead to the smallest tree size.
    
- **Speed:** Inference (a forward pass of the GNN) takes milliseconds, whereas Strong Branching might take seconds.
    
- **Result:** The solver makes "expert" branching decisions at "heuristic" speeds, often reducing the tree size by orders of magnitude.
    

### 3.3 "Neural Diving": Primal Heuristics for Feasibility

In a batch auction, finding _any_ feasible solution with a positive surplus is better than finding nothing. **Neural Diving** is a technique used to find high-quality feasible solutions (primal bounds) very early in the search.

1. **Generative Model:** A GNN is trained to estimate the marginal probability that a binary variable is equal to 1 in the optimal solution. (e.g., "Probability that Order $A$ matches with Order $B$ is 99%").
    
2. **Partial Assignment:** Based on these probabilities, the solver fixes the high-confidence variables (setting them to 0 or 1 hard).
    
3. **Sub-MIP Solve:** This leaves a much smaller "Sub-MIP" containing only the uncertain/difficult decisions (e.g., exact split percentages).
    
4. **Completion:** The symbolic solver solves this reduced problem quickly.
    

This allows CowSwap solvers to instantly identify the obvious "CoWs" (the easy matches) and spend the remaining block time optimizing the complex, marginal routing decisions.

### 3.4 Training Data Generation: Imitation Learning

Training these models requires a massive dataset of "Problem -> Optimal Solution" pairs. Solvers generate this data through **Imitation Learning**.

- **Offline Generation:** The solver team downloads millions of historical CowSwap batches. They run a powerful solver (e.g., Gurobi running for 1 hour per batch) to find the absolute ground-truth optimal solution.
    
- **Supervised Training:** The GNN is trained to minimize the difference between its predicted branching policy and the decisions made by the expert Gurobi run.
    
- **Synthetic Augmentation:** To make the model robust to rare events (e.g., extreme volatility), teams generate synthetic MILP instances using "Generative Adversarial Networks" (GANs) tailored for graph structures, creating hypothetical batches that mimic the statistical properties of DeFi flows (power-law distribution of trade sizes).
    

---

## 4. Engineering the Solver: The Rust Ecosystem

The implementation of these theoretical concepts requires a systems programming language capable of ensuring memory safety without garbage collection pauses, which could be fatal in a 12-second execution window. The DeFi solver community has converged on **Rust** as the standard.

### 4.1 Data Ingestion: `barter-rs`

The nervous system of a solver is its data ingestion pipeline. Solvers must maintain a real-time view of the global liquidity state, including both on-chain AMM reserves and off-chain CEX order books (for hedging).

**`barter-rs`** is a high-performance Rust framework designed specifically for this purpose.

- **Architecture:** It uses a modular trait-based system (`MarketStream`) to normalize WebSocket connections from diverse exchanges (Binance, Kraken, OKX) into a unified data structure.
    
- **Async/Await:** Leveraging the `tokio` runtime, `barter-rs` handles thousands of concurrent WebSocket streams. When a price update arrives (e.g., "Binance ETH/USDC ask moves to 2005"), it is processed via a zero-copy deserialization path (`serde_json` with `RawValue`) to minimize latency.
    
- **Private Inventory Hedging:** For solvers acting as Market Makers (PMMs), `barter-rs` feeds directly into the risk engine. If the solver fills a CowSwap user's sell order, it must instantly sell that exposure on a CEX. `barter-rs` provides the trigger mechanism for these atomic hedges.
    

### 4.2 Simulation Engine: `revm` and `ethers-rs`

Once the Neuro-Symbolic engine proposes a batch (a set of trades), the solver must verify its validity. Will it revert? How much gas will it burn?

**`revm` (Rust EVM)** is the industry-standard library for this simulation.

- **Lightweight Execution:** Unlike a full Geth node, `revm` is stripped down to the bare essentials of the Ethereum Virtual Machine. It does not manage a blockchain; it only executes bytecode against a state.
    
- **State Forking:** Solvers use a "Fork DB" pattern. They pull the Merkle proofs of the necessary state (account balances, pool storage slots) from an RPC node (like Alchemy) and cache them in memory.
    
- **Shadow Simulation:** The solver runs the proposed batch against this in-memory state. `revm` executes the transaction trace in microseconds.
    
    - **Honeypot Detection:** Solvers utilize `revm` to simulate a "Buy -> Approve -> Sell" cycle for every token in the batch. If the simulation fails (e.g., due to a malicious transfer restriction in the token contract), the token is flagged as a honeypot and excluded from the batch to prevent the solver's bond from being slashed.
        
- **Gas Estimation:** `revm` provides precise gas metering. The solver uses this to calculate the exact transaction cost, which is a term in the MILP objective function ($-\sum \text{GasCost}$).
    

**`alloy` and `ethers-rs`:**

Once validated, the batch is encoded into a transaction using `alloy` (the successor to `ethers-rs`). `alloy` provides low-level, zero-allocation encoding of Solidity ABI types. The transaction is then signed and broadcast to the CowSwap driver or the Flashbots bundle endpoint.

### 4.3 Yearn SeaSolver and Enso Shortcuts: Composable Primitives

The solver ecosystem is evolving beyond simple token swaps into complex "DeFi Actions."

**Yearn SeaSolver:**

This component allows Yearn strategies to act as "internal" solvers. The architecture leverages the CoW Protocol to settle trades _within_ the Yearn treasury.

- **Mechanism:** If Strategy A needs to rebalance (Sell ETH, Buy USDC) and Strategy B needs to deposit (Sell USDC, Buy ETH), SeaSolver identifies this overlap.
    
- **Implementation:** It constructs a "Cow" settlement that swaps the assets between the two strategy contracts directly.
    
- **Benefit:** This avoids all external market interaction, saving gas and LP fees. It effectively turns the Yearn Treasury into a private "Dark Pool" solved by the CoW logic.
    

**Enso Shortcuts:** Enso abstracts complex, multi-step DeFi interactions (e.g., "Zap into a Curve Pool and Stake the LP token on Convex") into **Shortcuts**.

- **Solver Integration:** A solver can treat a Shortcut as a single "atomic unit" or a "macro-instruction."
    
- **Programmable Intents:** Instead of just optimizing "Token In -> Token Out," the solver optimizes "Token In -> Yield Position Out." The Neuro-Symbolic model learns to route not just through pools, but through these logic macros, expanding the search space to include yield-bearing opportunities.
    

---

## 5. Comparative Analysis: CowSwap vs. UniswapX vs. 1inch Fusion

While all three systems use off-chain agents, their auction mechanisms impose different architectural requirements on the solver.

### 5.1 Mechanism Design Differences

|**Feature**|**CowSwap (CoW Protocol)**|**UniswapX**|**1inch Fusion**|
|---|---|---|---|
|**Auction Type**|**Frequent Batch Auction (FBA)**|**Dutch Auction (RFQ)**|**Dutch Auction (RFQ)**|
|**Pricing**|**Uniform Clearing Price (UCP)**|**Discriminatory (Pay-as-Bid)**|**Discriminatory (Pay-as-Bid)**|
|**Winner Selection**|Maximize **Global Batch Surplus**|First to fill (Latency / Gas War)|Slot-based / Whitelisted Resolvers|
|**Solver Strategy**|**Combinatorial Optimization (MILP)**|**Latency Arb / Inventory Mgmt**|**Inventory Mgmt / Pathfinding**|
|**Gas Liability**|Protocol Reimburses Solver (if winner)|Filler Pays (Risk of Revert)|Resolver Pays|

### 5.2 UniswapX and the JIT Liquidity Paradox

UniswapX employs a Dutch Auction where the order price decays over time. This creates a game of "Chicken" for fillers.

- **The Filler's Dilemma:** If a filler executes the order immediately, they pay a high price (low profit). If they wait for the price to decay, another filler might beat them to it.
    
- **JIT Liquidity:** A sophisticated strategy employed by UniswapX fillers is **Just-In-Time (JIT) Liquidity**.
    
    - The filler observes a large swap intent.
        
    - In the _same transaction_ as the fill, the filler adds massive liquidity to the Uniswap v3 pool at the exact tick range of the swap.
        
    - They execute the swap against their own liquidity (capturing the fee).
        
    - They withdraw the liquidity immediately.
        
- **Implication:** This requires solvers to be integrated deeply with LP strategy engines. While it improves execution price for the specific user, research suggests it can degrade the long-term health of passive LPs, who are "robbed" of the fee volume from informed flow.
    

### 5.3 1inch Fusion and "Pathfinder"

1inch's resolver architecture relies on the **Pathfinder** algorithm. Unlike CowSwap's global MILP, Pathfinder is fundamentally a routing algorithm (likely a variant of Dijkstra's algorithm or A* search adapted for liquidity density).

- **Optimization:** It splits a single large order into hundreds of "micro-chunks," routing them dynamically across different depths of various pools.
    
- **Rust Migration:** Recent updates indicate 1inch is moving critical Pathfinder logic to Rust to leverage parallel processing, allowing them to explore more "hops" (longer paths) within the quote generation window.
    

---

## 6. Advanced Topics: Security, Privacy, and Adversarial Dynamics

As solvers effectively become the "mempool" for intent-based systems, the centralization of this role poses significant risks. The industry is moving toward hardware-enforced security to mitigate this.

### 6.1 Trusted Execution Environments (TEEs) and Flashbots SUAVE

**Flashbots SUAVE (Single Unifying Auction for Value Expression)** envisions a future where the solver logic runs inside **Intel SGX (Software Guard Extensions)** enclaves.

- **The Privacy Problem:** Currently, a user must trust the solver not to front-run their intent or leak their strategy.
    
- **The TEE Solution:**
    
    - **Encrypted Intents:** The user encrypts their intent with the public key of the TEE.
        
    - **Secure Enclave:** The solver code runs inside the CPU's encrypted memory region (enclave). It decrypts the intent, calculates the route, and constructs the transaction.
        
    - **Inaccessibility:** The host operating system (and the human operator of the solver) cannot inspect the memory of the enclave. They never see the raw intent, only the final transaction.
        
- **Remote Attestation:** The TEE produces a cryptographic proof (quote) signed by the hardware manufacturer (Intel) verifying that the code running inside matches a specific hash. This guarantees that the solver is running the honest, open-source algorithm and not a modified "sandwiching" version.
    
- **Performance Overhead:** Running complex logic like Geth or GNN inference inside SGX incurs overhead due to memory encryption and context switching (paging). Flashbots has demonstrated "Geth inside SGX" , but optimizing heavy MILP solvers for TEEs remains an active research frontier.
    

### 6.2 Adversarial Attacks on Financial GNNs

The reliance on Neuro-Symbolic AI introduces a new attack surface: **Adversarial Machine Learning**.

- **Graph Poisoning:** An attacker can manipulate the input graph seen by the solver's GNN.
    
    - _Attack:_ The adversary generates thousands of "dust" orders or modifies liquidity positions in specific pools to create a specific topological pattern (e.g., a fake ring structure).
        
    - _Effect:_ The GNN, trained on normal market data, misinterprets this pattern. It might predict high probabilities for branches that are actually dead ends.
        
    - _Result:_ The solver's heuristic leads it down a "rabbit hole," wasting its limited computation time. The solver fails to find the optimal solution for the _real_ orders in the batch, or times out completely (Denial of Service).
        
- **Defense - Sparse K-NN Graphs:** Research indicates that "sparsifying" the graph (only connecting an asset to its $K$ most correlated peers, rather than all possible pools) increases robustness. By pruning the graph, the solver removes the "noise" introduced by the adversary, making the GNN harder to fool.
    

### 6.3 Oracle Manipulation

Solvers often use on-chain oracles (like Chainlink or Uniswap TWAP) to calculate the "reference price" for surplus estimation or to set the initial constraints for the MILP.

- **Vulnerability:** If an attacker manipulates the oracle price (e.g., via a flash loan attack on a thin pool), the solver might miscalculate the value of a trade.
    
- **Mitigation:** Robust solvers implement "Sanity Check" modules—simple statistical models or secondary oracles—that reject prices deviating significantly from the CEX mid-price (sourced via `barter-rs`). "DeFiGuard" systems use GNNs to detect transaction patterns characteristic of oracle manipulation attacks before the solver commits to a batch.
    

---

## 7. Case Studies and Market Dynamics

### 7.1 Solver Consolidation: Barter Acquires Copium

The recent acquisition of **Copium Capital’s** solver codebase by **Barter** highlights the "winner-takes-all" economics of the solver market.

- **Specialization:** Copium specialized in **RFQ** strategies (sourcing liquidity from private market makers). Barter specialized in **AMM** routing (optimizing Uniswap/Curve paths).
    
- **Synergy:** By merging, the new entity can solve batches that require _both_ deep CEX liquidity (for majors like ETH) and complex AMM routing (for long-tail tokens).
    
- **Risk:** This consolidation reduces the diversity of algorithms competing in the auction. If one dominant solver code wins 80% of blocks, the protocol essentially reverts to a centralized sequencer model, albeit one that is kept honest by the threat of competition.
    

---

## 8. Conclusion: The Future of Algorithmic Settlement

The technical architecture of DeFi settlement has evolved into a sophisticated discipline at the intersection of **High-Performance Computing**, **Operations Research**, and **Artificial Intelligence**. The "Solver" is no longer a simple script; it is a Neuro-Symbolic engine running on bare-metal Rust infrastructure, solving NP-hard problems in real-time while navigating a hostile, adversarial environment.

**CowSwap's** batch auction model, with its global optimization objective, provides the most fertile ground for these advanced techniques. The integration of **GNNs** to guide **MILP** solvers represents a breakthrough in handling the combinatorial explosion of multi-token clearing. Meanwhile, **UniswapX** pushes the frontier of latency and inventory management, driving innovations in JIT liquidity and Dutch Auction game theory.

Looking forward, the integration of **TEEs** will likely become the standard for solver integrity, enabling "Private Mempools" where execution is verifiable yet confidential. However, the cat-and-mouse game between solvers and adversarial actors (graph poisoners, oracle manipulators) will continue to drive the complexity of these systems. The victors in the "Solver Wars" will be those who can best synthesize the mathematical rigor of optimization theory with the adaptive intuition of deep learning, all built upon a fault-tolerant, high-velocity software stack.

---

**Table 1: Summary of Solver Infrastructure Stack**

|**Component**|**Technology**|**Primary Function**|**Key Library/Tool**|
|---|---|---|---|
|**Data Ingestion**|Rust / WebSockets|Normalize CEX/DEX market data|`barter-rs`, `tokio-tungstenite`|
|**Optimization (AI)**|Python / PyTorch|Learn branching policies (GNN)|`PyTorch Geometric`, `NetworkX`|
|**Optimization (Symbolic)**|C++ / Rust FFI|Solve MILP / Exact constraints|`SCIP`, `Gurobi`, `HiGHS`|
|**Simulation**|Rust|Verify transactions & Estimate Gas|`revm`, `foundry-evm`|
|**Transaction Mgmt**|Rust|Encode calldata & Broadcast|`alloy`, `ethers-rs`|
|**Privacy / Security**|C++ / Rust|Secure Enclave Execution|`Intel SGX SDK`, `Gramine`|