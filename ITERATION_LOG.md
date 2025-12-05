## Iteration 1: The Baseline MoE
**Date:** December 5, 2025
**Goal:** Implement DeepSeek-V3 Architecture (MLHA + MoE) and validate with the Microscope Dashboard.

### Configuration
*   **Dataset:** TinyShakespeare (Char-level)
*   **Layers:** 10
*   **Hidden Size:** 768
*   **Experts:** 8 Routed (Top-2) + 1 Shared
*   **Attention:** MLHA (Compression Ratio 8)
*   **Balancing:** Loss-less Bias Update (Rate: `0.001`)
*   **Training:** 10,000 Steps, Batch Size 4

### Results
*   **Final Loss:** 4.0677 (Excellent Convergence)
*   **Generation:** Coherent Shakespearean text produced.

### 🔍 Microscope Analysis
**Status:** ⚠️ Partial Collapse
*   **Healthy:** Layers 0 and 1 maintained dynamic routing.
*   **Collapsed:** Layers 2 through 9 collapsed into single-expert Dense layers.
*   **Diagnosis:** The `bias_update_rate` of 0.001 was too weak to counteract the "rich get richer" dynamic of the experts in deeper layers.

### Next Steps (for Iteration 2)
1.  Increase `bias_update_rate` to `0.05`.
2.  Add Gaussian Jitter to router logits to encourage exploration.
3.  Verify if Layer 2-9 collapse is resolved.
