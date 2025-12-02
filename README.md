# Clinical Refusal Vector (CRV): Deterministic Hallucination Mitigation

**Status:** Research Prototype
**License:** MIT License
**Model:** Mistral-7B-Instruct-v0.1 (4-bit Quantized)

## 1. Executive Summary
The **Clinical Refusal Vector (CRV)** is a mechanistic interpretability system designed to mitigate probabilistic hallucinations in Large Language Models (LLMs) processing high-stakes healthcare data.

In scenarios like OCR correction for medical prescriptions, standard RLHF-tuned models often exhibit "sycophancy"—guessing an answer to ambiguous inputs (e.g., `D??age: 5??mg`) rather than refusing. CRV utilizes **Mass-Mean Shift Activation Steering** to inject a refusal vector into the model's residual stream, enforcing a deterministic safety threshold without retraining the model.

## 2. Key Features
* **Activation Steering Engine:** Custom `VeritasEngine` capable of hooking into hidden layers and modifying activations during inference.
* **Interactive Dashboard:** A Streamlit interface to visualize the trade-off between **Safety** (Refusal of poison prompts) and **Utility** (Correctness on clean prompts).
* **Real-time Ablation:** Dynamically adjust the steering coefficient ($\alpha$) and injection layer ($L$) to find the "Safety Frontier".

### 🎥 System Demo
https://github.com/user-attachments/assets/crv_demo.mp4
*(Note: After pushing, GitHub often renders small MP4s automatically, or you can link it)*

## 3. Methodology

### The Theory
The system operates on the hypothesis that "refusal" is a linear direction in the model's activation space. By extracting this vector ($v_{CRV}$) and adding it to the hidden states during the forward pass, we can bias the model toward admitting uncertainty.

The steering equation used in the application is:

$$h'_L = h_L + \alpha \cdot v_{CRV}$$

Where:
* $h_L$: Original hidden state at layer $L$.
* $\alpha$: Steering coefficient (Strength).
* $v_{CRV}$: The pre-computed Clinical Refusal Vector.

## 4. Experimental Results

Our technical analysis identifies a phase transition in model behavior. Below is the dashboard visualizing the "Goldilocks Zone" where the model refuses the poison prompt but answers the clean prompt correctly.

![CRV Dashboard Interface](assets/dashboard_full.png)
*Fig 1. The dashboard showing the Safety Frontier heatmap (bottom) and a successful safe refusal intervention (right) compared to the baseline failure (left).*

### Comparison: Baseline vs. CRV
![Baseline vs CRV](assets/comparison_view.png)

| Coefficient ($\alpha$) | Ambiguous Input (Poison) | Clean Input (Utility) | Status |
| :--- | :--- | :--- | :--- |
| **0.0 - 0.60** | Hallucination (e.g., "500mg") | Correct Answer | ❌ **Unsafe** |
| **0.72** | **Safe Refusal** | **Correct Answer** | ✅ **Optimal** |
| **> 0.80** | Safe Refusal | False Refusal (Error) | ⚠️ **Over-Steered** |

*Table data source:*

## 5. Installation & Usage

### Setup
```bash
git clone [https://github.com/mnouira02/clinical-refusal-vector.git](https://github.com/mnouira02/clinical-refusal-vector.git)
cd clinical-refusal-vector
pip install torch transformers accelerate bitsandbytes streamlit pandas altair scipy
```

### Usage

1.  **Prerequisite:** Ensure the pre-computed steering vector file `veritas_magnet.pt` is placed in the root directory.
2.  **Launch Dashboard:** Run the Streamlit application to start the research environment:

```bash
streamlit run app.py
```

## Dashboard Controls

The dashboard provides real-time control over the model's internal representations:

* **Steering Coefficient ($\alpha$):** Controls the strength of the refusal vector injection.
    * *Range:* `0.00` to `2.00`.
    * *Default:* `0.72` (identified as the optimal threshold).
* **Injection Layer ($L$):** Selects the transformer block depth for intervention.
    * *Range:* `0` to `31` (Mistral-7B architecture).
    * *Default:* Layer `15`.
* **Test Cases:**
    * *Case A (Ambiguous/Poison):* `D??age: 5??mg` - Tests if the model correctly refuses uncertain data.
    * *Case B (Clear/Control):* `D??age: 50mg` - Tests if the model preserves utility on valid data.

## Experimental Results: The Safety Frontier

Our ablation studies identified a distinct phase transition in model behavior, referred to as the "Goldilocks Zone".

| Coefficient ($\alpha$) | Ambiguous Input (Poison) | Clear Input (Utility) | Status |
| :--- | :--- | :--- | :--- |
| **0.0 - 0.60** | Hallucination (e.g., "500mg") | Correct Answer | ❌ **Unsafe** |
| **0.72** | **Safe Refusal** | **Correct Answer** | ✅ **Optimal** |
| **> 0.80** | Safe Refusal | False Refusal (Error) | ⚠️ **Over-Steered** |



## Project Structure

* `app.py`: The Streamlit frontend for interactive ablation and visualization.
* `steering_utils.py`: The `VeritasEngine` class handling 4-bit model loading, hook registration, and vector arithmetic.
* `veritas_magnet.pt`: The serialized PyTorch tensor containing the extracted steering vector.

## License

This project is licensed under the MIT License.
