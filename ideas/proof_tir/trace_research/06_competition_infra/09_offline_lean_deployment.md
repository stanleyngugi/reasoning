# Deep Research: Offline Lean 4 Deployment — Kaggle Competition Environment

## Research Objective

Deploy a fully functional Lean 4 environment in Kaggle's offline competition setting. The standard Lean installation requires internet access (elan fetches toolchains, lake fetches dependencies). We need to pre-package everything and make it work with zero network access.

## Context

AIMO competition constraints:
- No internet access during inference
- 5-hour GPU runtime (H100s for AIMO 3)
- 20GB disk, 30GB RAM (approximate)
- Must work on first run — no debugging opportunity

Our system needs Lean 4 for `native_decide` verification. This is the critical engineering challenge.

## Research Questions

### Part A: Lean 4 Installation Anatomy

#### 1. Standard Installation Process
- What does `elan` do during installation?
- What toolchain files are downloaded?
- Where are they stored? (`~/.elan/toolchains/`)
- What's the total size of a Lean 4 toolchain?

#### 2. Toolchain Components
- `lean` binary — the compiler/type checker
- `lake` — the build system
- `leanc` — C++ compiler wrapper
- Standard library (Init, Lean, Std)
- What else?

#### 3. Mathlib4 Dependencies
- What is Mathlib4? (Community math library)
- Do we need it for our use case? (Maybe just core library suffices?)
- If we need it, how large is it? (~5GB compiled `.olean` files)
- Mathlib dependencies: Batteries, other packages

### Part B: Creating Offline Packages

#### 4. Freezing the Toolchain
- How do we download a specific Lean 4 version for offline use?
- `elan toolchain install leanprover/lean4:v4.X.X` — where are files stored?
- Can we simply tarball `~/.elan/toolchains/leanprover--lean4---v4.X.X/`?
- What about platform-specific binaries (Linux x86_64)?

#### 5. Pre-building .olean Files
- `.olean` files are compiled Lean files (like `.pyc` for Python)
- Compiling Mathlib from source takes hours — we need pre-built cache
- `lake exe cache get` downloads pre-built `.olean` files
- Where are they stored? How do we package them?

#### 6. Directory Structure
Create the exact directory layout needed:
```
/kaggle/input/lean-offline/
├── elan/
│   └── toolchains/
│       └── leanprover--lean4---v4.7.0/
├── project/
│   ├── lakefile.lean
│   ├── lean-toolchain
│   └── .lake/
│       ├── build/
│       └── packages/
```

#### 7. Environment Variables
- `ELAN_HOME` — where elan looks for toolchains
- `LEAN_PATH` — where Lean looks for imports
- `LEAN_SRC_PATH` — where Lean looks for source files
- `PATH` — must include Lean binaries
- What else needs to be set?

### Part C: Portability Challenges

#### 8. Absolute Path Issues
- Lean 4 `.olean` files may contain embedded absolute paths
- Moving from build machine to Kaggle breaks these paths
- How do we work around this?
- Is there a `--prefix` or relocation mechanism?

#### 9. Platform Dependencies
- Kaggle runs Linux x86_64
- Building on macOS or Windows won't work
- Must build on compatible Linux environment
- Docker for reproducible builds?

#### 10. Endianness and Memory Mapping
- `.olean` files use memory mapping tricks
- Are there endianness issues?
- Any CPU-specific optimizations that could break?

### Part D: Minimal Lean Setup

#### 11. Do We Need Mathlib?
For our use case (`native_decide` on basic Nat/Int operations):
- Core Lean 4 should be sufficient
- What exactly is in core vs Mathlib?
- Can we avoid Mathlib entirely and reduce package size dramatically?

#### 12. Minimal lakefile.lean
```lean
import Lake
open Lake DSL

package myProject

@[default_target]
lean_lib Verify
```
What's the minimal project configuration?

#### 13. Single-File Verification
Can we just run `lean --run file.lean` without Lake?
- Pros: simpler, faster startup
- Cons: no caching, no dependencies
- When does this work?

### Part E: Runtime Integration

#### 14. Python-Lean Communication
- How does our Python code invoke Lean?
- `subprocess.run(['lean', '--run', 'verify.lean'])`?
- Lean REPL for persistent process?
- JSON communication protocol?

#### 15. LeanInteract / Pantograph
- What are these tools?
- Do they help with Python-Lean integration?
- Do they add complexity or simplify?
- Are they needed for our use case?

#### 16. File I/O Pattern
- Generate `.lean` file from Python
- Invoke Lean compiler
- Parse output (success/failure/errors)
- What's the latency per invocation?

### Part F: Performance Optimization

#### 17. Startup Time
- How long does `lean --run file.lean` take to start?
- Is there a "warm-up" cost we can amortize?
- Persistent Lean process vs spawn-per-problem?

#### 18. Compilation Caching
- If we verify 50 problems with similar structure, can we cache compilation?
- Pre-compile template code, just swap the formula?
- Lake's incremental compilation?

#### 19. Memory Usage
- How much RAM does Lean use for `native_decide`?
- Does it scale with problem size?
- Memory limits on Kaggle — how close are we to the edge?

### Part G: Testing and Validation

#### 20. Local Kaggle Simulation
- Can we simulate Kaggle's offline environment locally?
- Docker container with no network access
- Test the full setup before submission

#### 21. Failure Modes
- What happens if Lean binary is corrupted?
- What if PATH is misconfigured?
- What if .olean files are incompatible?
- Error messages and debugging strategies

#### 22. Fallback Strategies
- If Lean fails to initialize, what's plan B?
- Can we detect Lean failure early and skip verification?
- Graceful degradation to TIR-only?

### Part H: Step-by-Step Build Guide

#### 23. Build Environment
- Docker image specification
- Lean version selection (which 4.X.X?)
- Build commands sequence

#### 24. Packaging Commands
```bash
# Step 1: Install Lean
# Step 2: Create project
# Step 3: Build/cache dependencies
# Step 4: Package for Kaggle
```

#### 25. Kaggle Setup Script
```python
# Runtime initialization
# Environment variable setup
# Verification that Lean works
```

### Part I: Alternative Approaches

#### 26. Lean 4 on Kaggle — Has Anyone Done This?
- Search Kaggle notebooks for Lean 4 usage
- Any published solutions using Lean?
- Learn from prior attempts (successes and failures)

#### 27. Other Verifiers
If Lean proves too difficult:
- Could we use Z3 for simpler checks?
- Isabelle? Coq?
- What's the trade-off?

#### 28. Custom Verification
Could we write our own lightweight verifier?
- Just for formula checking, not full theorem proving
- Lose Lean's guarantees, gain simplicity
- Is this worth considering?

## Desired Output Format

1. **Complete Build Guide**: Step-by-step from scratch to packaged
2. **Directory Structure Template**: Exact layout needed
3. **Environment Variable Reference**: What to set and why
4. **Troubleshooting Guide**: Common issues and fixes
5. **Size Estimates**: How big is each component
6. **Performance Benchmarks**: Startup time, per-problem latency
7. **Kaggle-Specific Notes**: Platform quirks and workarounds
8. **Sample Code**: Working Python wrapper for Lean invocation











# Comprehensive Research Report: Engineering Offline Lean 4 Deployments for High-Performance Inference in Kaggle Environments

## Executive Summary

The integration of formal verification systems into competitive artificial intelligence environments represents a paradigm shift in the validation of machine-generated reasoning. The AI Mathematical Olympiad (AIMO) and similar computational challenges have introduced a rigorous constraint set—strict offline inference, finite GPU runtimes, and ephemeral storage architectures—that fundamentally conflicts with the "always-connected" design philosophy of modern proof assistants. Standard Lean 4 workflows rely on a dynamic, network-dependent ecosystem where the `elan` version manager and `lake` build system continuously synchronize toolchains and dependencies from remote repositories. To deploy Lean 4 effectively within a Kaggle notebook for tasks such as `native_decide` verification or automated proof checking, one must engineer a bespoke, air-gapped runtime environment.

This report presents an exhaustive technical methodology for decoupling Lean 4 from its network dependencies and deploying it as a frozen, deterministic artifact. The analysis dissects the anatomy of the Lean toolchain, the binary structure of compilation artifacts (`.olean` files), and the Linux filesystem hierarchy distinct to Kaggle’s containerized runners. It proposes a "push-and-freeze" architecture that inverts the standard dependency resolution model, moving all build-time complexity to a pre-deployment Docker phase.

Key findings indicate that successful deployment hinges on three critical engineering pillars:

1. **Hermetic Toolchain Encapsulation**: Bypassing `elan`'s dynamic resolution by manually constructing a rigid directory structure and overriding system paths via environment variables (`LEAN_SYSROOT`, `LEAN_PATH`).
    
2. **Static Dependency Resolution**: "Tricking" the `lake` build system into accepting a pre-populated, local dependency graph through precise manipulation of `lake-manifest.json` and directory relocations, ensuring `mathlib` artifacts are loaded from read-only mounts without recompilation.
    
3. **Persistent Runtime Interaction**: Utilizing the `leanprover-community/repl` over standard CLI invocation to amortize startup costs (3-8 seconds per process) across thousands of inference calls, enabling high-throughput verification via a persistent JSON-RPC channel.
    

This document serves as a complete implementation guide, offering step-by-step build instructions, directory layouts, and Python orchestration scripts designed to achieve zero-network compliance while maximizing inference throughput on NVIDIA H100 hardware.

---

## 1. The Lean 4 Runtime Architecture: Anatomy of a Proof Assistant

To engineer a robust offline deployment, one must first deconstruct the standard online workflow. Lean 4 is not merely a compiler; it is a complex suite of interacting components designed to facilitate interactive theorem proving, batch compilation, and metaprogramming.

### 1.1 The Role of Elan: The Gatekeeper

In a standard development environment, `elan` acts as the master orchestrator. Modeled after Rust’s `rustup`, `elan` virtualizes the Lean installation. When a user invokes `lean`, they are not calling the compiler directly. Instead, they are invoking a shim binary—a proxy executable—that inspects the current directory for a `lean-toolchain` file.

This file contains a version string, such as `leanprover/lean4:v4.15.0`. The shim then queries `elan`'s internal registry to locate the corresponding toolchain. If the toolchain is missing or if the registry metadata is stale, `elan` immediately attempts to contact GitHub or Azure blob storage to download the required assets. In a Kaggle environment with the network interface disabled, this behavior is catastrophic. The shim will either hang indefinitely waiting for a connection or terminate with a resolution error.

Therefore, the first requirement of an offline strategy is to bypass `elan` entirely or configuring it in a way that strictly forbids network attempts. The most robust approach is to bundle the specific target toolchain binary and invoke it directly, effectively manually performing the resolution that `elan` would otherwise automate.

A resolved toolchain installation on Linux typically consumes between 400MB and 600MB and consists of a standard UNIX-like hierarchy:

- **`bin/`**: Contains the critical executables:
    
    - `lean`: The core compiler, type checker, and elaborator.
        
    - `lake`: The build system and package manager.
        
    - `leanc`: A wrapper around the C compiler (typically `clang` or `gcc`) used to compile Lean's intermediate C output into native objects.
        
- **`lib/lean/`**: Stores the pre-compiled artifacts of the core language libraries (`Init`, `Lean`, `Std`). These `.olean` files are the binary representation of the language's axioms and primitive definitions.
    
- **`include/`**: Contains C++ header files (`lean.h`) required for the Foreign Function Interface (FFI) and for compiling code generated by `native_decide`.
    

### 1.2 Lake: The Dependency Graph Manager

`lake` (Lean Make) is the build system that manages project configuration and transitive dependencies. A Lean project is defined by a `lakefile.lean` or `lakefile.toml`, which specifies dependencies (e.g., `mathlib`, `aesop`, `batteries`).

In an online context, `lake` is highly dynamic. It resolves dependencies by:

1. Reading `lake-manifest.json` to determine exact git commit hashes.
    
2. Cloning repositories into `.lake/packages/`.
    
3. Computing build hashes to determine if recompilation is necessary.
    
4. Downloading pre-built `.olean` files from a cloud cache (via `lake exe cache get`) to avoid the prohibitive cost of compiling Mathlib from source.
    

For offline deployment, we must present `lake` with a fait accompli: a `.lake` directory that is already fully populated, compiled, and consistent. Crucially, `lake` uses file timestamps and hashes to determine validity. If the "thawing" process on Kaggle modifies timestamps carelessly, `lake` might decide that the pre-compiled artifacts are stale and attempt a rebuild. Since `mathlib` takes hours to compile on a standard machine, triggering a rebuild during a 5-hour Kaggle competition is effectively a failure condition.

### 1.3 The Compilation Artifacts:.olean and.ilean

The efficiency of Lean relies on its compilation artifacts. Understanding these is vital for preserving relocation integrity.

- **`.olean` (Object Lean)**: These are binary files containing the serialized state of a compiled module. They include the definitions, theorems, and proofs. Lean uses `mmap` (memory mapping) to load these files directly into memory. This means the binary layout on disk must perfectly match the in-memory layout expected by the Lean executable. This introduces a strict requirement: the architecture and OS of the build machine (e.g., Docker container) must match the deployment machine (Kaggle runner). Both are x86_64 Linux, which simplifies matters, but cross-compiling from macOS (ARM64) is strictly impossible for these artifacts.
    
- **`.ilean` (Interface Lean)**: These files contain information used by the Lean language server for autocompletion and "go-to-definition". While critical for an IDE experience, they are generally not required for batch verification tasks like `native_decide` or `lake build`, allowing us to exclude them to save space if necessary.
    

### 1.4 Native Decide and Dynamic Linking

The specific use case of `native_decide` introduces an additional layer of complexity. `native_decide` works by taking a decidable proposition, compiling the decision procedure into C code, compiling that C code into a shared object (`.so`) using the system's C compiler, dynamically loading that shared object into the running Lean process, and executing it.

This means the offline environment must not only support Lean but also provide a working C toolchain (`gcc` or `clang`) and the necessary headers. The `leanc` wrapper manages this, but it relies on environment variables like `LEAN_CC` to find the compiler. If the toolchain is moved from its default location, `leanc` might fail to locate the `lean.h` header file, causing `native_decide` to crash.

---

## 2. The Offline Constraint Analysis

The Kaggle environment imposes a set of rigid boundaries that dictate the engineering solution. It is not merely a Linux server; it is a restricted container with specific filesystem behaviors.

### 2.1 The Kaggle Filesystem Hierarchy

A Kaggle notebook interacts with three primary directory zones, each with distinct properties that influence our deployment strategy :

1. **`/kaggle/input/`**:
    
    - **Nature**: Read-only.
        
    - **Content**: This is where datasets uploaded to the competition are mounted. Our pre-packaged Lean artifact will reside here (e.g., `/kaggle/input/lean-offline-bundle/`).
        
    - **Constraint**: We cannot execute `chmod` or write lockfiles here. `lake` often attempts to write a `.lake/build/lake.lock` file or update log files. Running `lake` directly against a project in `/kaggle/input/` will fail with "Permission denied".
        
    - **Implication**: While the heavy toolchain binaries can remain here, the active project directory likely needs to be copied to a writable location.
        
2. **`/kaggle/working/`**:
    
    - **Nature**: Read-write, Ephemeral.
        
    - **Content**: The current working directory for the notebook.
        
    - **Capacity**: Nominally 20GB, though often shared with other system processes. This is sufficient for a standard Lean project plus Mathlib artifacts (approx. 5-7GB), but efficient space management is prudent.
        
    - **Implication**: This is the target destination for the "thawing" process.
        
3. **`/tmp/`**:
    
    - **Nature**: Read-write, very ephemeral.
        
    - **Content**: System temporary files.
        
    - **Implication**: Useful for temporary C files generated by `native_decide`, but typically small.
        

### 2.2 Network Isolation & Package Management

During a submission run, the container's network interface is disabled or severely restricted (no outbound internet access).

- **Pip**: `pip install` will fail. We cannot install Python wrappers like `lean-interact` or `pathlib` at runtime unless they are pre-downloaded as wheels.
    
- **Git**: `lake` uses `git` to verify repository integrity. If `lake` attempts to fetch a remote to validate a hash, it will fail. We must ensure `lake-manifest.json` and the `.lake/packages` directory are in a state that requires no network verification.
    

### 2.3 Hardware & Resource Limits

The AIMO 3 environment typically provides NVIDIA H100 GPUs and roughly 30GB of system RAM.

- **RAM Constraint**: Loading the full `Mathlib` environment into a Lean process can consume 2-4GB of RAM. If we spawn multiple concurrent Lean processes (e.g., using Python's `multiprocessing`), we could hit the 30GB ceiling rapidly. A single persistent server process is the preferred architecture.
    
- **CPU Constraint**: While the GPU is powerful, Lean is largely single-threaded per task (though `lake` parallelizes builds). `native_decide` compilation is CPU-bound. The 4 vCPUs usually available on Kaggle nodes are the bottleneck for compilation, not verification. This reinforces the need to pre-compile absolutely everything.
    

---

## 3. Methodology: Creating the "Frozen" Artifact

The core strategy is to create a "fat" archive—a single tarball containing the specific Lean toolchain, the fully built project, all transitive dependencies (Mathlib), and the necessary Python interaction scripts. This process _must_ be performed in a Docker container that replicates the Kaggle OS to ensure binary compatibility (GLIBC versions, etc.).

### 3.1 Step 1: The Hermetic Build Environment

We use a Docker container to act as the "Build Server". This prevents contamination from the host system (e.g., user-specific configurations in `~/.elan`) and ensures we are building for the correct architecture (Linux x86_64).

**Rationale**: Kaggle Docker images are based on Ubuntu (typically 20.04 or 22.04). Using an Alpine Linux image would be a mistake due to `musl` vs `glibc` incompatibilities. We select Ubuntu 22.04 as the baseline.

Dockerfile

```
# Dockerfile.lean_build
FROM ubuntu:22.04

# Install system dependencies
# build-essential is CRITICAL for 'leanc' (C compiler) and compiling the REPL.
# python3-pip is needed to download python wheels.
# git and curl are needed for elan/lake.
RUN apt-get update && apt-get install -y \
    curl git tar zstd build-essential python3 python3-pip \
    libgmp-dev libffi-dev

# Setup a non-root user to mimic Kaggle's permission model
RUN useradd -m -s /bin/bash kaggle_user
USER kaggle_user
WORKDIR /home/kaggle_user

# Install Elan (Lean Version Manager) non-interactively
# We perform this install to get the 'elan' binary, which we then use
# to fetch the specific toolchain we want to freeze.
RUN curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh -s -- -y --default-toolchain none
ENV PATH="/home/kaggle_user/.elan/bin:${PATH}"
```

### 3.2 Step 2: Toolchain Acquisition and Freezing

In a standard install, `elan` places toolchains in `~/.elan/toolchains/`. The directory names are often mangled (e.g., `leanprover--lean4---v4.7.0`). For our offline bundle, we want a deterministic path.

We use `elan` to install the specific version required by the project's `lean-toolchain` file. Then, we identify exactly where it was installed so we can copy it out later.

Bash

```
# Inside the container
# This installs the toolchain specified by the string (e.g., v4.15.0)
elan toolchain install leanprover/lean4:v4.15.0

# Verify it works
lean +leanprover/lean4:v4.15.0 --version
```

### 3.3 Step 3: Project Initialization and Mathlib Compilation

This is the most time-consuming step. We must build the project and, crucially, download the pre-compiled Mathlib artifacts.

1. **Initialize**: Create the project directory.
    
2. **Configure `lakefile.lean`**: Ensure it requires `mathlib` (and `repl`, if building from source).
    
3. **Update Manifest**: Run `lake update`. This resolves the dependency graph and generates `lake-manifest.json`. This file locks the exact commit hashes of every dependency.
    
4. **Cache Retrieval**: Run `lake exe cache get`. This connects to the remote cache (usually hosted on Azure) and downloads the `.olean` files for the specific Mathlib revision. **Without this step, the Kaggle runner would attempt to build Mathlib from scratch, which takes 4+ hours and would timeout the competition submission**.
    
5. **Full Build**: Run `lake build`. This ensures that all local files and interface files are generated and linked.
    

**The "Repl" Wrapper**: For robust Python-Lean communication, we use the `leanprover-community/repl` tool. This is a small Lean program that imports the target environment and listens on stdin for JSON commands. It is vastly superior to parsing `lean` CLI output because it maintains state (the "Environment") between commands, avoiding the overhead of re-parsing imports.

- We clone the `repl` repo.
    
- We configure its `lakefile.toml` or `lakefile.lean` to include the _same_ dependencies as our main project (or add the main project as a dependency of the REPL).
    
- We run `lake build` to produce the `repl` binary.
    

### 3.4 Step 4: Packaging and Relocation Prep

We now organize the artifacts into the structure Kaggle expects. We create a staging directory `lean-offline-bundle`.

- **Toolchain**: We copy the specific toolchain directory (e.g., `~/.elan/toolchains/leanprover--lean4---v4.15.0`) to `lean-offline-bundle/toolchain`.
    
- **Project**: We copy the entire project directory, including the massive `.lake` folder (which contains the `build` and `packages` subdirectories), to `lean-offline-bundle/project`.
    
- **Python Wheels**: We download any necessary Python libraries (e.g., `lean-interact` if we use it, though direct `subprocess` is often safer) into `lean-offline-bundle/wheels` using `pip download`.
    

Finally, we compress this folder into `lean-offline-bundle.tar.gz`.

---

## 4. Deployment: The "Thawing" Process on Kaggle

The deployment phase runs inside the Kaggle notebook. We assume the `lean-offline-bundle.tar.gz` has been uploaded as a dataset and is available at `/kaggle/input/lean-offline-bundle/`.

### 4.1 The Read-Only Constraints & Workarounds

As established, `/kaggle/input/` is read-only.

- **The Toolchain**: Can remain in `/kaggle/input`. The executables (`lean`, `leanc`) do not need to write to their own binary directories during execution. We can add `/kaggle/input/.../toolchain/bin` to the system `PATH`.
    
- **The Project**: Must be moved. `lake` and `lean` (when running `native_decide`) may attempt to create lock files or write temporary build artifacts. We copy the project folder from `/kaggle/input/.../project` to `/kaggle/working/project`. This operation takes a few seconds but is essential for permissions.
    

### 4.2 Environment Variable Configuration

This is the single most critical step in the offline deployment. Standard `elan` usage hides these variables from the user, but we must set them manually to "wire up" the disjointed components.

1. **`LEAN_SYSROOT`**: This variable tells the `lean` binary where its own installation root is. We set this to `/kaggle/input/lean-offline-bundle/toolchain`. If this is unset, `lean` might try to find its libraries relative to its binary location, which usually works, but setting it explicitly is safer.
    
2. **`LEAN_PATH`**: This is the search path for `.olean` files. It acts like `PYTHONPATH`. We must strictly construct this path to include:
    
    - The toolchain's core libraries: `$LEAN_SYSROOT/lib/lean`
        
    - The project's build directory: `/kaggle/working/project/.lake/build/lib`
        
    - The dependencies' build directories (Mathlib): `/kaggle/working/project/.lake/packages/mathlib/.lake/build/lib` (and similarly for `std`, `aesop`, etc.).
        
    - _Note_: Ideally, `lake` manages this for us if we invoke via `lake env`. However, relying on `lake env` adds overhead. Hardcoding `LEAN_PATH` for the final `repl` binary is a powerful optimization.
        
3. **`LEAN_SRC_PATH`**: Points to the source `.lean` files. While less critical for binary execution, it is needed for error reporting and `native_decide` compilation which often references source locations.
    
4. **`PATH`**: We prepend `$LEAN_SYSROOT/bin` to the system `PATH` so that `lean` and `leanc` are found.
    
5. **`LEAN_CC`**: Specifies the C compiler. The bundled toolchain usually expects `clang`. On Kaggle, we might want to force it to use the system `/usr/bin/gcc` or `/usr/bin/clang` to ensure compatibility with system headers. Setting `LEAN_CC=clang` is often necessary.
    

### 4.3 Setup Script Orchestration

A robust Python script handles this initialization. It detects the environment, performs the copy, sets the variables, and verifies the binary is executable.

Python

```
import os
import shutil
import subprocess
import sys

def setup_offline_lean(dataset_path="/kaggle/input/lean-offline-bundle"):
    """
    Bootstraps the Lean 4 environment from a read-only Kaggle dataset.
    """
    print(">>> Initializing Lean 4 Offline Environment...")
    
    # 1. Define Paths
    toolchain_dir = os.path.join(dataset_path, "toolchain")
    project_src = os.path.join(dataset_path, "project")
    working_dir = "/kaggle/working/lean_project"
    
    # 2. Copy Project to Writable Space
    # We remove any existing copy to ensure a clean state on rerun
    if os.path.exists(working_dir):
        print(f"    Cleaning existing directory: {working_dir}")
        shutil.rmtree(working_dir)
        
    print(f"    Copying project from {project_src} to {working_dir}...")
    # shutil.copytree is efficient
    shutil.copytree(project_src, working_dir)
    
    # 3. Configure Environment Variables
    # We construct a new environment dict to pass to subprocesses
    env = os.environ.copy()
    
    # Update PATH to include the toolchain binaries
    lean_bin = os.path.join(toolchain_dir, "bin")
    env = f"{lean_bin}:{env}"
    
    # Set LEAN_SYSROOT
    env = toolchain_dir
    
    # Set LEAN_CC explicitly to system clang to avoid header issues
    env["LEAN_CC"] = "clang"
    
    # 4. Verify Installation
    try:
        # We run 'lean --version' to check if the binary is executable
        # and if shared libraries (like libgmp) are found.
        result = subprocess.run(
            ["lean", "--version"], 
            env=env, 
            capture_output=True, 
            text=True, 
            check=True
        )
        print(f"    Success! Detected: {result.stdout.strip()}")
    except subprocess.CalledProcessError as e:
        print(f"    FATAL: Lean binary failed to execute.\n{e.stderr}")
        raise
        
    print(">>> Environment Ready.")
    return env, working_dir
```

---

## 5. Runtime Integration Strategies

Once Lean is executable, the challenge moves to efficient interaction. The AIMO competition often requires testing thousands of candidates. The latency of starting the Lean process becomes the dominant factor.

### 5.1 The Latency Problem of CLI Invocation

The naive approach—using `subprocess.run()` for every problem—is prohibitively slow.

- **Startup Overhead**: When `lean` starts, it must initialize the runtime, load the core library, and then sequentially load every imported module. For Mathlib, this involves loading hundreds of megabytes of `.olean` data into memory.
    
- **Quantified Cost**: Loading a basic file might take 0.5s. Loading `Mathlib.Data.Real.Basic` can take 3-5 seconds depending on disk I/O and caching.
    
- **Impact**: If a solution requires checking 100 candidate proofs for 50 problems (5000 checks), and each takes 3 seconds, the total time is ~4 hours. This consumes nearly the entire competition runtime.
    

### 5.2 The Persistent REPL Solution

The superior architecture uses `leanprover-community/repl`. This tool starts a Lean environment _once_, loads the necessary imports (e.g., Mathlib), and then listens for commands via stdin/stdout. This essentially creates a "Lean Server".

**Mechanism:**

1. **Start**: Launch the `repl` subprocess. It incurs the 3-5 second startup cost _once_.
    
2. **Command**: Python sends a JSON command to stdin.
    
    JSON
    
    ```
    { "cmd": "def check : Bool := native_decide... #eval check" }
    ```
    
3. **Execution**: The `repl` parses the command in the _existing_ memory-resident environment.
    
4. **Response**: `repl` returns a JSON object with the result.
    
    JSON
    
    ```
    { "messages":, "env": 1 }
    ```
    
5. **Loop**: The process stays alive for the next command.
    

**Performance Gain**: Subsequent commands typically execute in milliseconds (excluding the actual proof computation time), as the environment is already hot.

### 5.3 Optimizing `native_decide`

`native_decide` is a tactic that uses Lean's compiler to verify decidable propositions. It is significantly faster than kernel reduction (`rfl`) for large computational problems.

**How it works**:

1. It compiles the proposition into a C function.
    
2. It invokes `leanc` to compile that C function into a shared library.
    
3. It uses `dlopen` to load the library.
    
4. It executes the function.
    

**Offline Risks**:

- **Soundness**: `native_decide` relies on the correctness of the compiler and the C compiler. It is generally trusted but has had edge cases (e.g., `reduceBool` bugs).
    
- **Configuration**: In modern Lean versions (v4.15+), `native_decide` is robust but requires the `+revert` option or careful context management to ensure variables are properly captured.
    
- **Headers**: As mentioned, `leanc` must find `lean.h`. Our `LEAN_SYSROOT` variable helps `leanc` locate `include/`, but forcing `LEAN_CC` is a necessary fallback if the bundled paths are misaligned.
    

---

## 6. Performance Optimization & Resource Management

### 6.1 Memory Mapping and Architecture

Lean's `.olean` files are memory-mapped (`mmap`). This allows the OS to page in only the necessary parts of Mathlib, rather than loading the entire 5GB binary blob into RAM at once.

- **Implication**: This is generally safe on 30GB RAM nodes. However, if you run parallel inference (e.g., 4 workers), and each worker maps Mathlib, the OS handles shared pages efficiently _if_ they are reading the exact same files. If you copy the project to 4 different temporary directories, the OS might not deduplicate the pages, leading to OOM (Out Of Memory) kills.
    
- **Recommendation**: Use a single shared `project` directory in `/kaggle/working/` for reading imports. If workers need to write temporary files, they should write unique filenames to `/tmp/` rather than duplicating the whole project structure.
    

### 6.2 Stripping Mathlib

If the competition solutions rely only on basic arithmetic (e.g., `Nat`, `Int`, `List`), Mathlib is unnecessary overhead.

- **Minimalist Build**: Create a project that imports only `Init` and `Std`.
    
- **Size Reduction**: Mathlib compiled is ~5GB. Core Lean is ~200MB. This drastically reduces "thawing" time (copying files) and startup time.
    
- **Decision Matrix**:
    
    - Use **Core Only** if the AI generates simple computational checks (`123 * 456 = 56088`).
        
    - Use **Mathlib** if the AI generates proofs using high-level theorems (`Data.Real.Basic`, `Topology`, etc.).
        

---

## 7. Troubleshooting & Validation

When deployment fails on Kaggle, debugging is blind. Common failure modes include:

- **Error: "unknown package 'Mathlib'"**:
    
    - _Cause_: `LEAN_PATH` is incorrectly set or `lake-manifest.json` does not match the directory structure.
        
    - _Fix_: Log `os.environ` in the setup script. Ensure the path points to `build/lib` subdirectories, not just the package root.
        
- **Error: "file '...' not found" during `native_decide`**:
    
    - _Cause_: `leanc` cannot find standard headers (`stdio.h`) or the compiler binary.
        
    - _Fix_: Verify `LEAN_CC` is set to `clang`. Ensure `PATH` includes `/usr/bin`.
        
- **Error: Segmentation Fault**:
    
    - _Cause_: Architecture mismatch (e.g., building on M1 Mac, running on Kaggle x86_64).
        
    - _Fix_: Strictly rebuild the "fat" archive using a Docker container targeting `linux/amd64`.
        

---

# Part B: Technical Implementation Guide

This section provides the specific commands, scripts, and configurations to implement the methodology.

## 1. Directory Structure Template

The final dataset uploaded to Kaggle must adhere to this exact structure.

/kaggle/input/lean-offline-bundle/

├── toolchain/ # The expanded Elan toolchain

│ ├── bin/ # lean, lake, leanc

│ ├── lib/

│ │ ├── lean/ # Core.olean files (Init, Std)

│ └── include/ # C headers for native_decide

├── project/ # Your Lean Project

│ ├── lakefile.lean

│ ├── lean-toolchain # Must match../toolchain version

│ ├── lake-manifest.json # LOCKED versions

│ ├──.lake/ # PRE-COMPILED ARTIFACTS

│ │ ├── build/

│ │ │ ├── lib/ # Project.olean files

│ │ └── packages/ # Dependencies (Mathlib, Aesop)

│ │ ├── mathlib/

│ │ │ ├──.lake/build/lib/ # Mathlib.olean files

│ │ └──...

│ └── repl/ # Compiled REPL binary directory

└── scripts/

└── setup_lean.py

## 2. Step-by-Step Build Guide (Local Machine)

### Phase 1: Docker Build Container

Create a `Dockerfile` to mimic Kaggle's environment.

Dockerfile

```
# filename: Dockerfile.lean_build
FROM ubuntu:22.04

# Install system dependencies
# build-essential is CRITICAL for 'leanc' and compiling the REPL.
RUN apt-get update && apt-get install -y \
    curl git tar zstd build-essential python3 python3-pip \
    libgmp-dev libffi-dev

# Setup user to avoid root permission issues later
RUN useradd -m -s /bin/bash kaggle_user
USER kaggle_user
WORKDIR /home/kaggle_user

# Install Elan (Lean Version Manager)
RUN curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh -s -- -y --default-toolchain none
ENV PATH="/home/kaggle_user/.elan/bin:${PATH}"

# Install the specific toolchain
ENV LEAN_VERSION="leanprover/lean4:v4.15.0"
RUN elan toolchain install ${LEAN_VERSION}
```

Build the image:

Bash

```
docker build -t lean-offline-builder -f Dockerfile.lean_build.
```

### Phase 2: Compile Project & Mathlib

Run the container to build the artifacts. Mount a local directory to extract the results.

Bash

```
docker run -it --rm -v $(pwd)/output:/output lean-offline-builder bash
```

Inside the container:

Bash

```
# 1. Create/Clone Project
mkdir my_project && cd my_project
lake init my_project math  # Creates a project with Mathlib dependency

# 2. Pin Dependencies (Critical for reproducibility)
lake update

# 3. Fetch Pre-built Mathlib Cache
lake exe cache get

# 4. Build the Project and REPL
# Clone REPL inside the project
git clone https://github.com/leanprover-community/repl.git
# Add REPL to lakefile if necessary or build it standalone
cd repl
lake build
cd..
# Build main project
lake build

# 5. Prepare the "Fat" Bundle
mkdir -p /output/bundle/toolchain
mkdir -p /output/bundle/project

# Copy the specific toolchain
TOOLCHAIN_PATH=$(elan which lean | xargs dirname | xargs dirname)
cp -rL $TOOLCHAIN_PATH/* /output/bundle/toolchain/

# Copy the fully built project
cp -r. /output/bundle/project/

# 6. Create Archive
cd /output/bundle
tar -czf lean_offline_v1.tar.gz *
```

## 3. Kaggle Setup Script (Python)

Use this script in your inference notebook to initialize the environment.

Python

```
import os
import sys
import shutil
import subprocess
import json

# --- Configuration ---
DATASET_ROOT = "/kaggle/input/lean-offline-v1"
TOOLCHAIN_ROOT = os.path.join(DATASET_ROOT, "toolchain")
PROJECT_SOURCE = os.path.join(DATASET_ROOT, "project")
WORKING_DIR = "/kaggle/working/lean_project"

def setup_lean_environment():
    print(">>> Setting up Lean 4 Offline Environment...")
    
    # 1. Copy Project to Writable Space
    if os.path.exists(WORKING_DIR):
        shutil.rmtree(WORKING_DIR)
    shutil.copytree(PROJECT_SOURCE, WORKING_DIR)
    
    # 2. Construct Environment Variables
    lean_bin = os.path.join(TOOLCHAIN_ROOT, "bin")
    lean_lib = os.path.join(TOOLCHAIN_ROOT, "lib", "lean")
    
    # Path to Mathlib oleans
    mathlib_lib = os.path.join(WORKING_DIR, ".lake", "packages", "mathlib", ".lake", "build", "lib")
    project_lib = os.path.join(WORKING_DIR, ".lake", "build", "lib")
    
    new_env = os.environ.copy()
    new_env = TOOLCHAIN_ROOT
    new_env = f"{lean_bin}:{new_env}"
    # Force system clang for native_decide
    new_env["LEAN_CC"] = "clang" 
    
    # Constructing LEAN_PATH
    lean_path_entries =
    new_env = ":".join(lean_path_entries)
    
    return new_env, WORKING_DIR

lean_env, project_root = setup_lean_environment()
```

## 4. Runtime Integration: The Python Wrapper

Python

```
class LeanREPL:
    def __init__(self, repl_path, env, cwd):
        self.proc = subprocess.Popen(
            [repl_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            cwd=cwd,
            text=True,
            bufsize=1 # Line buffered
        )
        
    def verify(self, theorem_code):
        # Wraps code in a command that the REPL understands
        # Note: Protocol specifics depend on the exact REPL version
        # This example assumes the standard JSON-RPC style
        
        # We construct a full file content or command
        command = {
            "cmd": theorem_code,
            "env": 0 # Default environment
        }
        
        try:
            self.proc.stdin.write(json.dumps(command) + "\n")
            self.proc.stdin.flush()
            
            response = self.proc.stdout.readline()
            if not response:
                return False, "REPL crashed"
                
            data = json.loads(response)
            # Check for errors in "messages" list
            errors = [m for m in data.get("messages",) if m["severity"] == "error"]
            
            if errors:
                return False, errors
            return True, "Verified"
            
        except BrokenPipeError:
            return False, "REPL Pipe Broken"

# Usage
repl_bin = os.path.join(project_root, "repl", ".lake", "build", "bin", "repl")
verifier = LeanREPL(repl_bin, lean_env, project_root)

code = "example : 2 + 2 = 4 := by native_decide"
is_valid, msg = verifier.verify(code)
print(f"Result: {is_valid}, Msg: {msg}")
```

## 5. Size Estimates & Benchmarks

- **Lean Toolchain**: ~400 MB
    
- **Mathlib Binaries**: ~2.5 GB (stripped debug) - 5 GB (full)
    
- **Startup Time (CLI)**: 2.0s - 5.0s
    
- **REPL Latency**: < 50ms per command
    
- **Kaggle Disk Usage**: ~6GB / 20GB
    

By strictly following this "freeze-and-thaw" architecture, you create a deterministic, high-performance verification engine that respects all AIMO competition constraints.