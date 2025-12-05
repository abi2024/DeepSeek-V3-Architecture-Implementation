# 📜 Generation Analysis Log

This document tracks the qualitative performance of the model across iterations. We test the `best_model.pt` checkpoint from each run with standard prompts to evaluate coherence, structure, and knowledge retrieval.

---

## 🟢 Iteration 1: The "Collapsed" Run
**Date:** December 5, 2025
**Checkpoint:** 10,000 Steps
**State:** Partial Router Collapse (Layers 2-9)
**Final Loss:** `nan` (Instability detected at final step, effective loss ~4.1)

### 🧪 Test Setup
*   **Prompt:** "The King said"
*   **Temperature:** 0.8
*   **Max Tokens:** 150

### 📝 Generated Outputs

#### Response 1
```text
The King said, and all in
GOWBURY.
Go to England shall be;
KINGHAM.
[_Exeunt._]
Ay,
[_Exit._]

[_Exit La soldier we shall be so he will not be his Grace.

His good manhood,
KING.
[_Exit._]
```

#### Response 2
```text
The King said
KING.
KING.
...
With him and my life.
The King, my lord, good my heart.
KING.
My lord, that was by his sovereign.
```

#### Response 3
```text
K health.
GENT.
KATHARINE.
KING.
Madam? Your wife, in faith,
Go show.
KING HENRY.
If so.
KING HENRY.
Or I think we make your majesty upon these person:
My lord.
```

### 🧐 Analysis of Output

#### 1. Structure Learning (High Success)
The model has perfectly internalized the **structure** of the training dataset (TinyShakespeare).
*   It correctly formats character names in uppercase (`KING HENRY`, `BUCKINGHAM`).
*   It correctly uses stage directions (`[_Exit._]`, `[_Exeunt._]`).
*   It maintains the dialog format (Name -> Line).

#### 2. Syntax vs. Semantics
*   **Syntax (Grammar):** Mostly functional ("Go to England shall be", "His good manhood"). This suggests the **Shared Expert** (which processes all tokens) successfully captured basic English and Shakespearean grammar rules.
*   **Semantics (Meaning):** Low coherence. The dialogue loops (`KING. KING. KING.`) and lacks a continuous narrative thread.

#### 3. The "Collapse" Signature
The repetitive nature and lack of deep reasoning align with our **MoE-Microscope** findings. Because Layers 2-9 collapsed into single experts, the model effectively operated as a shallow, dense network. It lacks the "capacity" to maintain long-range context or complex storytelling, resulting in "surface-level" Shakespeare mimicry.

#### 4. The NaN Warning
The loader reported `Best Loss: nan`. This correlates with the instability seen in the final training steps. The model likely experienced a gradient explosion due to the fighting between the Router (trying to pick one expert) and the Bias Loss (trying to force diversity), which resulted in the weights becoming volatile.

