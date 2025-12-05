# 🛠️ Engineering Changelog: DeepSeek MoE Iteration 2

This document details the architectural and hyperparameter changes moving from Iteration 1 to Iteration 2, based on the **MoE-Microscope** forensic analysis.

---

## 🚨 Analysis of Iteration 1 (The Baseline)
**Outcome:** Model trained to convergence (Loss ~4.0) but exhibited **instability (`nan` loss)** at the end and **Router Collapse** in deep layers.

### Diagnosed Issues
1.  **Router Collapse:** Layers 2-9 ignored 7 out of 8 experts. The router became "over-confident" in specific experts early in training.
2.  **Weak Balancing:** The "Loss-less Balancing" mechanism (Bias Update) was too slow (`0.001`) to counteract the "Rich-Get-Richer" loop.
3.  **Determinism:** Without noise, the router had no incentive to explore underutilized experts once a pattern was established.
4.  **Instability:** The final gradient explosion (`nan`) suggests the learning rate was slightly too aggressive for the sparse architecture.

---

## 🚀 Iteration 2: "The Explorer"
**Goal:** Force the router to explore all experts and stabilize training.

### 1. Code Changes
| Component | Change | Reason |
| :--- | :--- | :--- |
| **Router** | Added **Gaussian Jitter** | Injecting random noise into router logits during training forces the model to occasionally pick "second-best" experts, preventing early collapse. |
| **Config** | Parametrized Bias Rate | Moved `bias_update_rate` from hardcoded value to `config.py` for rapid tuning. |

### 2. Hyperparameter Tuning
| Parameter | Iteration 1 | Iteration 2 | Why? |
| :--- | :--- | :--- | :--- |
| `expert_bias_update_rate` | `0.001` (Hardcoded) | **`0.05`** | **50x increase.** We need to aggressively punish greedy experts immediately when they start dominating. |
| `router_jitter_noise` | `0.0` (None) | **`0.02`** | Adds uncertainty to routing, effectively "shaking" the router out of local minima (collapse). |
| `learning_rate` | `3e-4` | **`2e-4`** | Slightly reduced to prevent the `nan` gradient explosion seen at the end of Iteration 1. |

---

## 🔮 Hypothesis
By increasing the penalty for greediness and adding random noise:
1.  The **Heatmaps** for Layers 2-9 should show much more distributed blue patterns (less white horizontal lines).
2.  **Router Entropy** should stay higher (above 0.5) for longer.
3.  **Final Loss** might be slightly *higher* initially (due to noise), but the model will be more robust and actually utilize its full parameter count.