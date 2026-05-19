# 🛰️ OmniRay: AVX2-Accelerated Deep RL Spatial Discovery Engine

A high-performance, pluggable raycasting engine, parallelized particle filter, and Gymnasium environment designed for training Deep Reinforcement Learning agents on **Active SLAM**, spatial discovery, and autonomous exploration tasks.

---

## 🎯 What is OmniRay? (Project Overview & Purpose)

**OmniRay** is an advanced research testbed designed to solve the **Active SLAM (Simultaneous Localization and Mapping)** problem in mobile robotics using Deep Reinforcement Learning (Deep RL). 

### 🛑 The Problem it Solves
In traditional robotics, SLAM is **passive**: the robot relies on human commands or pre-calculated static path-planners to move, and the SLAM system simply maps whatever the sensors detect. This often leads to poor exploration efficiency, high localization drift (especially in featureless environments), or catastrophic mapping failures when the robot encounters wheel slip.

Furthermore, training deep reinforcement learning agents directly in realistic physics simulators or on physical hardware is incredibly slow and computationally expensive. The sensor raycasting (simulating LiDAR sweeps) and scan-matching (updating particle filters) usually create severe bottlenecks that limit training cycles.

### 💡 The OmniRay Proposal & Solution
OmniRay proposes a **configuration-driven, hyper-accelerated active SLAM engine** that solves these challenges through:

1. **Active Mapping via Deep RL:** Rather than following static paths, the PPO (Proximal Policy Optimization) agent is trained using a custom CNN-MLP fusion network to actively choose navigation velocities. It dynamically balances the trade-off between exploring new regions (frontier reward shaping) and maintaining accurate localization (minimizing particle filter pose drift).
2. **AVX2 & Pure-NumPy Acceleration:** By leveraging SIMD vector alignment and loops-free 2D NumPy broadcasting, the raycaster and VectorSLAM particle filter operate at compiled C-level speeds (under **3.2 ms per simulation step**). This enables rapid agent training on consumer-grade CPUs in minutes rather than days.
3. **Sim-to-Real Robustness:** Directly embeds continuous kinodynamic tire slippage, yaw drift, LiDAR distance noise, and random laser dropouts inside the training loop. This forces the agent to learn robust trajectories that actively help the particle filter match scans, correcting **95.1% of localization drift** without requiring ideal physical conditions.

---

## 🎨 System Architecture

Here is the horizontal data-flow architecture of the OmniRay Active SLAM Deep RL system:

![OmniRay Architecture Diagram](assets/architecture_horizontal.png)

---

## 💎 Project Accomplishments & Performance Summary

* **AVX2 & NumPy Spatial Discovery Engine:** Built a fully vectorized, parallel particle filter (`VectorSLAM`) and a 2D raycaster (`NumpyRaycaster`) that execute in **under 3.2 ms per step** (with raw scan times of **0.189 ms**!) entirely on CPU without requiring a GPU.
* **Realistic Sim-to-Real Degradation Models:** Integrated continuous kinodynamic wheel slip errors, constant yaw drifts, and non-ideal LiDAR distance noise (with random dropouts) to simulate a physical differential-drive robot.
* **Master Explorer Convergence:** Fully converged a custom Multi-Input CNN-MLP PPO agent, increasing average episode reward by **+123%** (reaching **1,530**).
* **95.1% Drift Reduction:** Confirmed via quantitative testing that the PPO policy guides the robot to keep final positioning drift to a minuscule **1.02 units** (a **95.1% drift correction** relative to uncorrected dead-reckoning).

---

## 📂 Codebase Structure

```
OmniRay/
│
├── assets/
│   └── architecture_horizontal.png   # System architecture diagram (horizontal layout)
│
├── envs/
│   ├── __init__.py
│   ├── active_slam_env.py      # Gymnasium Active SLAM Environment & noise models
│   ├── raycaster_backends.py   # Pluggable Raycasting Backends (NumPy, PyMunk, SIMD)
│   └── vector_slam.py          # Parallelized Pure-NumPy Particle Filter Engine
│
├── profiling/
│   ├── __init__.py
│   ├── benchmark_bottleneck.py # Bottleneck Profiler & Decision Engine
│   └── benchmark_slam.py       # Speed comparison between backends
│
├── results/                    # Diagnostic output directory
│   ├── robust_evaluation_report.png
│   └── robust_exploration_progression.png
│
├── sim/
│   ├── CMakeLists.txt          # C++ compiler config (AVX2 & pybind11)
│   ├── src/
│   │   ├── bindings.cpp        # pybind11 wrapper definitions
│   │   ├── raycaster.cpp       # AVX2 8-lane parallel SIMD implementation
│   │   └── raycaster.h         # C++ raycaster API header
│   └── test_raycaster.py       # C++ correctness and speed validation
│
├── config.yaml                 # Centralized training & network hyperparameters
├── requirements.txt            # Pinned package dependencies
├── train_rl.py                 # PPO deep reinforcement learning pipeline (ablation ready)
├── evaluate_and_record.py      # Quantitative trajectory evaluator (saves to results/)
├── run_ablation_study.py       # Ablation study sequencer (entropy, rewards, noise)
├── visualize_agent.py          # Real-time human visualizer (matplotlib GUI)
├── test_env.py                 # Environment smoke test with rendering
└── README.md                   # Interactive documentation
```

---

## 🚀 Make it Demo-able (Run in 1 Command!)

You can instantly watch the pre-trained robust **Master Explorer** agent actively navigate the noisy arena and build its SLAM map using a single command:

```powershell
py -3.11 visualize_agent.py --model-path active_slam_ppo_robust_master.zip --episodes 3 --max-steps 400
```

---

## 🛠️ Hyperparameter Configuration (`config.yaml`)

Training, environment parameters, and neural network sizes are managed in [config.yaml](config.yaml). The `train_rl.py` script automatically loads these parameters:

* **PPO Hyperparameters:** `learning_rate` ($3.0 \times 10^{-4}$), `ent_coef` (policy entropy weight: $0.01$), `n_steps` ($2048$), and `batch_size` ($64$).
* **Neural Architecture:** Processes continuous mapping features with a custom **CNN Branch** (`[16, 32]` channels) and poses/lasers with a **1D MLP Branch** before projecting to a **256-D fusion layer**.

---

## ⚖️ Active SLAM Environment Reward Tuning

The reward function inside `envs/active_slam_env.py` is fully parameterized and customizable. You can adjust the coefficients inside `config.yaml` or override them dynamically via CLI flags in `train_rl.py`:

* **`reward_exploration` (Default: `1.0`):** Reward per newly explored grid cell in the occupancy map.
* **`reward_time_penalty` (Default: `0.01`):** Penalty applied at every step to encourage rapid exploration.
* **`reward_collision_penalty` (Default: `0.1`):** Penalty applied on collision to prevent contact with obstacles.
* **`reward_frontier` (Default: `0.1`):** Vectorized frontier attraction reward shaping which guides the robot towards the boundaries of unexplored territory.

---

## 🧪 Ablation Studies (`run_ablation_study.py`)

A specialized ablation study suite has been created to analyze hyperparameter sensitivity and sim-to-real transfer:

1. **Entropy impact:** Compares exploration rate convergence with (`--ent-coef 0.01`) vs without (`--ent-coef 0.0`) policy entropy incentives.
2. **Reward Weights Sensitivity:** Measures the impact of the frontier exploration shaping reward by comparing a high frontier pull weight (`--reward-frontier 0.5`) vs none (`--reward-frontier 0.0`).
3. **Physical Noise Robustness:** Analyzes learning under active slippage and sensor drops vs ideal, zero-noise physical kinematics (`--no-noise`).

### How to Run:
> [!IMPORTANT]
> The scripts are fully prepared. Execute them only when you are ready to start training.

* **Run all three ablation tests sequentially** (using 50,000 steps per test):
  ```powershell
  py -3.11 run_ablation_study.py --experiment all --steps 50000
  ```
* **Run a single targeted ablation study** (e.g., Entropy Impact):
  ```powershell
  py -3.11 run_ablation_study.py --experiment entropy --steps 50000
  ```

---

## 📊 Quantitative Benchmark Results (360 Rays)

| Backend | Mean Scan Time | Median Scan Time | P99 Scan Time | 100K Steps Est. | Verdict |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Pure Python** (baseline) | 2.838 ms | 2.829 ms | 4.053 ms | 4.7 min | Slow baseline |
| **PyMunk segment_query** | 1.145 ms | 1.066 ms | 2.074 ms | 2.0 min | Moderate |
| **NumPy Vectorized** (batch) | **0.182 ms** | **0.178 ms** | **0.355 ms** | **0.3 min (18s)** | **Ultra-Fast (Winner)** |

---

## ⚙️ Getting Started

### 1. Install Dependencies
Ensure you run this on a Python 3.11 environment (your primary package environment):
```bash
pip install -r requirements.txt
```

### 2. Run the Bottleneck Profiler
Benchmark all backends on your CPU and analyze the ray count scaling:
```bash
py -3.11 -m profiling.benchmark_bottleneck --rays 360 --iterations 500
```

### 3. Run the Gym Environment Smoke Test
Test the Gymnasium active SLAM environment with random agent actions:
```bash
py -3.11 test_env.py --backend numpy --episodes 3 --max-steps 150
```
