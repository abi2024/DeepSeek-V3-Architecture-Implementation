# DeepSeek-V3 Architecture & MoE-Microscope 🔬

[![PyTorch](https://img.shields.io/badge/PyTorch-2.4%2B-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Architecture](https://img.shields.io/badge/Architecture-DeepSeek_V3-purple)](https://github.com/deepseek-ai)
[![License](https://img.shields.io/badge/License-MIT-green)]()

A high-fidelity implementation of the **DeepSeek-V3** architecture, converted from a Smollm2 baseline. This project features **Multi-Head Latent Attention (MLHA)** and **DeepSeekMoE** with a **Loss-less Load Balancing** strategy.

Crucially, it includes the **MoE-Microscope**: an MLOps dashboard for real-time monitoring of expert routing dynamics, designed to detect and diagnose "Router Collapse" before it destroys training runs.

---

## 🏗️ Architecture: DeepSeek-V3 Implementation

We have restructured the standard Llama/Smollm2 transformer block into the DeepSeek specification:

### 1. Multi-Head Latent Attention (MLHA)
Standard Multi-Head Attention (MHA) suffers from massive Key-Value (KV) cache memory usage during inference. MLHA solves this via **Low-Rank Compression**:
*   **Mechanism:** Projects KV heads into a compressed latent vector space before projecting back to heads.
*   **Compression Ratio:** `8:1` (Reduces KV memory footprint by 87.5%).
*   **Benefit:** Allows significantly larger batch sizes and context lengths during inference without sacrificing generation quality.

### 2. DeepSeekMoE (Mixture-of-Experts)
Unlike standard Top-K MoE (e.g., Mixtral), this architecture separates knowledge responsibilities:
*   **Shared Expert:** A dedicated expert that processes **every** token. This captures "common knowledge" and syntactic structures ensuring stability.
*   **Routed Experts:** 8 specialized experts where tokens are routed dynamically via Top-2 selection.
*   **Total Active Parameters:** 1 Shared + 2 Routed = 3 Active Experts per token.

### 3. Loss-Less Load Balancing
Standard MoE models use an **Auxiliary Loss** (added to the total loss) to force load balancing. This gradient conflict often degrades primary model performance.
*   **Our Solution:** **Bias-Based Balancing**.
*   We maintain a dynamic `bias` term for each expert.
*   If an expert is overloaded (receives too many tokens), its bias is decreased in the router logits, naturally steering tokens to underutilized experts.
*   **Result:** Perfect load balancing without polluting the gradient with auxiliary loss terms.

---

## 🔬 MoE-Microscope Dashboard

Mixture-of-Experts training is notoriously unstable. The **MoE-Microscope** is a real-time observability suite built with **Streamlit** and **Plotly** to visualize the internal state of the router.

### Why It Matters?
Standard loss curves cannot detect **Router Collapse**—a failure state where the router stops learning and dumps all tokens into a single expert, effectively turning a massive MoE into a tiny dense model.

### Key Metrics Monitored
| Metric | Visualization | Diagnosis |
| :--- | :--- | :--- |
| **Expert Load Heatmap** | Matrix (Layer × Expert) | **Vertical lines** indicate healthy specialization. **Horizontal lines** indicate collapse (one expert dominating a layer). |
| **Router Entropy** | Line Chart | Measures routing confidence. **~2.0** = Random/Unlearned. **~0.0** = Collapse. **0.5-1.5** = Healthy "Opinionated" Routing. |
| **Token Dropping Rate** | Area Chart | Percentage of tokens discarded due to expert buffer overflow. Spikes indicate aggressive load imbalance. |

---

## 📂 Repository Structure

```text
DeepSeek-MoE/
├── checkpoints/          # Automatic state saving (Best & Latest)
├── data/                 # Dataset storage
├── logs/                 # JSON logs for the Microscope dashboard
├── src/
│   ├── config.py         # Hyperparameters & Architecture Config
│   ├── dataset.py        # Tokenization & Batching
│   ├── model.py          # Core DeepSeek-V3 PyTorch Implementation
│   ├── monitor.py        # MLOps Metrics Backend
│   ├── utils.py          # Logging & Checkpointing utilities
│   ├── router_entropy.png
│   └── expert_heatmap.png
├── dashboard.py          # Streamlit Visualization Frontend
├── train.py              # Main Training Loop (AMP enabled)
├── ANALYSIS.md           # Deep dive into training dynamics
├── ITERATION_LOG.md      # Changelog of experiments
└── requirements.txt      # Dependencies
```

---

## 🚀 Quick Start

### 1. Installation
Clone the repo and install dependencies. Note: We use specific CUDA-enabled PyTorch builds.
```bash
# Create venv
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install (Ensure you have CUDA 12.x drivers if using GPU)
pip install -r requirements.txt
```

### 2. Training
Start the training loop. This will automatically download the tokenizer, process the dataset, and begin training with Mixed Precision (AMP).
```bash
python train.py
```
*   **Console Output:** Shows real-time Loss, VRAM usage, and Entropy metrics.
*   **Artifacts:** Checkpoints saved to `checkpoints/`; Logs saved to `logs/`.

### 3. Launching the Microscope
While training is running (or after), launch the dashboard in a separate terminal:
```bash
streamlit run dashboard.py
```
This will open `http://localhost:8501` in your browser.

---

## 📊 Performance & Results

**Training Configuration:**
*   **Dataset:** TinyShakespeare (Character Level)
*   **Context Window:** 1024 tokens
*   **Hidden Size:** 768
*   **Layers:** 10 (Scaled for RTX 3060 VRAM)
*   **Experts:** 8 Routed + 1 Shared

**Final Metrics (10,000 Steps):**
*   **Final Loss:** ~4.06 (Converged)
*   **Router Entropy:** ~0.09 (High Confidence)
*   **Token Drop Rate:** < 1.0% (Effective balancing)

## 🕵️ Analysis: Detecting "The Rich Get Richer"
While our loss curve looked perfect (dropping to 4.06), the **MoE-Microscope** detected a critical hidden issue: **Layer-Specific Router Collapse**.

While Layers 0 and 1 remained dynamic, Layers 2-9 effectively collapsed into single experts, turning the deep MoE into a Dense model.

👉 **[Read the Full Post-Mortem Analysis Here](ANALYSIS.md)** to see the visualizations and root cause diagnosis.

### Generated Output Sample
```text
Now!” pointing and I not to try any straining
Jaggers, but. exercise to
I said, not my magnificent granted in that, employed one by the man and
his left the old to no sign for such answer...
```

---

## 🛠️ Tech Stack
*   **Core:** PyTorch, NumPy
*   **Tokenizer:** HuggingFace Transformers (Smollm2)
*   **Observability:** Streamlit, Plotly, Pandas
*   **Optimization:** CUDA AMP (Automatic Mixed Precision)