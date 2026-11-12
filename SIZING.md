# How the network sizes were chosen (width and depth)

The general, architecture-independent version of this procedure is `SIZING_METHOD.md`.

Record of the sizing process for the three model families, 2026-10-30. All numbers are RMS relative error over (t_d, E), in %, mean over 5 seeds, unless stated. Weights include biases. Training set: 2800 points × 2 outputs = 5600 target values.

**Final result:**

| Family | Layers (input counted) | Hidden neurons | Weights | Train / val RMS |
|---|---|---|---|---|
| ReLU MLP | 3: input, 1 hidden, output | 50 | 352 | 0.39 / 0.42 % |
| KBNN | 4: input, 1 hidden (2 blocks), linear, knowledge | 25 per block | 381 | 0.38 / 0.44 % |
| tanh MLP (control) | 3 | 12 | 86 | 0.45 / 0.46 % |

**Common to every step:**
- Base training set (2800 points). Early stopping on the validation set: each network keeps its best-validation weights.
- Full-batch Adam with cosine learning-rate decay, 5 random seeds per size.
- Only training and validation errors were used. The test and extrapolation sets were evaluated once, after the final choice (`final_eval.py`).

## Step 1: exploratory sweep (exp_01, 30k epochs, one hidden layer)

Purpose: see how error scales with size and compare the variants before choosing anything.

| Hidden neurons | 5 | 10 | 20 | 40 | 80 | 160 |
|---|---|---|---|---|---|---|
| ReLU MLP val | 3.58 | 1.62 | 0.89 | 0.49 | 0.34 | 0.28 |
| tanh MLP val | 1.33 | 0.60 | 0.35 | 0.30 | 0.27 | 0.26 |

| Hidden neurons per KBNN block | 3 | 5 | 10 | 20 | 40 | 80 |
|---|---|---|---|---|---|---|
| KBNN val | 2.48 | 1.56 | 0.79 | 0.50 | 0.46 | 0.39 |

- The KBNN's linear correction input beat √ at every size, so √ was dropped.
- The tanh MLP levels off near 0.26 % with one hidden layer. The ReLU networks keep improving as they grow.

## Step 2: first width rule (exp_04, 60k epochs, one hidden layer), superseded

Rule: the lowest validation error, unless a smaller size is within 5 % of it; then the smallest such.

| Family | Widths tried | Val RMS | Picked |
|---|---|---|---|
| ReLU MLP | 80 / 160 / 320 / 640 | 0.33 / 0.26 / 0.19 / 0.15 | 640 (4482 weights) |
| tanh MLP | 80 / 160 / 320 | 0.22 / 0.22 / 0.20 | 320 |
| KBNN | 40 / 80 / 160 / 320 per block | 0.41 / 0.36 / 0.33 / 0.28 | 320 (4806 weights) |

## Step 3: depth check (exp_05, two hidden layers, 60k epochs), superseded

Same rule, applied jointly over one and two hidden layers. With two hidden layers the KBNN has two layers in each block (5 layers in all).

| Family | Two hidden layers × width: val RMS | Picked |
|---|---|---|
| ReLU MLP | 2×20 0.45, 2×40 0.42, 2×80 0.42 | one layer × 640 |
| tanh MLP | 2×20 0.096, 2×40 0.063, 2×80 0.051 | 2×80 (7042 weights) |
| KBNN | 2×10 0.49, 2×20 0.47, 2×40 0.56 | one layer × 320 |

- A second layer made both ReLU families worse. It made the tanh MLP 3–4× more accurate: two tanh layers can form the products between inputs (the slope τ depends on VCM, VDD and T) that one layer cannot.
- These picks were frozen and evaluated once (kept in `results/final_exp05.pt`, `results/final_eval_exp05.csv`).

## Why steps 2–3 were replaced

**The networks were overlearning.** The course's training procedure (Zhang et al.) compares training error E_tr with validation error E_v. When E_tr ≪ E_v, the network is memorizing its training points and should be smaller. For the step 2 networks:

| ReLU MLP width | 20 | 40 | 80 | 160 | 320 | 640 |
|---|---|---|---|---|---|---|
| Weights | 142 | 282 | 562 | 1122 | 2242 | 4482 |
| Train RMS | 0.83 | 0.46 | 0.27 | 0.17 | 0.08 | 0.05 |
| Val RMS | 0.89 | 0.49 | 0.33 | 0.26 | 0.19 | 0.15 |
| Val / train | 1.07 | 1.07 | 1.22 | 1.50 | 2.47 | **3.07** |

(Widths 20–40 from exp_01, 80–640 from exp_04.) The KBNN at 320 per block had the same ratio, 3.2. The tanh networks stayed at 1.25 or below.

- **640 neurons were 4482 weights for 5600 training values,** almost one weight per value.
- **The rule had no "good enough".** The data comes from a simulator and has no noise floor, so validation error never levels off. For the ReLU MLP it falls only about as H^−0.38, 20–25 % per doubling: one ReLU neuron adds one flat fold, and a smooth curved surface needs many folds. A 5 % tolerance is never met, so the rule always picks the largest width tried.
- **The accuracy was not needed.** The application needs a delay error well below τ ≈ 10 ps, i.e. about 1 ps: an error δt changes the predicted metastability rate by about e^(δt/τ). The 40-neuron ReLU MLP already averaged 0.55 ps on validation.

## Step 4: the course's procedure (exp_06, 60k epochs), final

1. **Target accuracy E_d = 0.5 %** RMS. That is about 0.5 ps on a 100 ps delay, half the 1 ps budget.
2. Grow the hidden layer in small steps. At each size, classify the mean over 5 seeds:
   - underlearning: E_tr > E_d
   - overlearning: E_tr ≤ E_d < E_v
   - good learning: E_tr ≤ E_d and E_v ≤ E_d
3. **Final width:** the smallest size with good learning.
4. **Depth:** one hidden layer. A second layer is added only if one layer cannot reach E_d; one layer reached it for every family.

| ReLU MLP width | 10 | 15 | 20 | 25 | 30 | 35 | 40 | **50** | 60 | 80 |
|---|---|---|---|---|---|---|---|---|---|---|
| Weights | 72 | 107 | 142 | 177 | 212 | 247 | 282 | **352** | 422 | 562 |
| Train | 1.57 | 1.06 | 0.83 | 0.70 | 0.61 | 0.51 | 0.50 | **0.39** | 0.35 | 0.31 |
| Val | 1.70 | 1.10 | 0.88 | 0.75 | 0.66 | 0.55 | 0.53 | **0.42** | 0.40 | 0.36 |
| State | under | under | under | under | under | under | over* | **good** | good | good |

| tanh MLP width | 5 | 8 | 10 | **12** | 15 | 20 | 25 | 30 | 40 |
|---|---|---|---|---|---|---|---|---|---|
| Weights | 37 | 58 | 72 | **86** | 107 | 142 | 177 | 212 | 282 |
| Train | 1.22 | 0.69 | 0.53 | **0.45** | 0.37 | 0.27 | 0.25 | 0.25 | 0.23 |
| Val | 1.18 | 0.70 | 0.54 | **0.46** | 0.38 | 0.28 | 0.26 | 0.26 | 0.23 |
| State | under | under | under | **good** | good | good | good | good | good |

| KBNN width per block | 5 | 8 | 10 | 12 | 15 | 20 | **25** | 30 | 40 |
|---|---|---|---|---|---|---|---|---|---|
| Weights | 81 | 126 | 156 | 186 | 231 | 306 | **381** | 456 | 606 |
| Train | 1.54 | 0.99 | 0.79 | 0.66 | 0.55 | 0.49 | **0.38** | 0.36 | 0.32 |
| Val | 1.54 | 1.02 | 0.85 | 0.68 | 0.60 | 0.54 | **0.44** | 0.44 | 0.42 |
| State | under | under | under | under | under | over* | **good** | good | good |

\* Overlearning only by the definition. The training error is just under E_d and the validation error just over it, with val/train 1.05 (ReLU 40) and 1.11 (KBNN 20). These networks sit on the threshold; they are not memorizing.

- At the chosen sizes val/train is 1.02–1.18, so none of the final networks overlearns.
- **Weights needed to reach E_d:** tanh MLP 86, ReLU MLP 352, KBNN 381. The built-in knowledge does not make in-range fitting cheaper; its sub-networks are ReLU too. What it buys is extrapolation (`README.md`, frozen models).

## How to describe it in the report

- State E_d and why: the 1 ps budget from the metastability application.
- Show the step 4 tables (or a plot of train and val error vs weights) as the answer to "how many hidden neurons".
- Mention the superseded rule in one or two sentences, as the overlearning example: validation-only selection on noise-free data grows the network without bound.
- Depth: one hidden layer meets the target; two layers helped only the tanh MLP (exp_05).

## Files

`exp_01_sizes.py`, `exp_04_select.py`, `exp_05_depth.py`, `exp_06_target.py`; logs and per-network CSVs in `results/` (`exp_06_target.csv` has train_rms and val_rms per network). The final networks are in `results/final.pt`.
