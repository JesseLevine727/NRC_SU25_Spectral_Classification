# Sizing a neural network: a general procedure

This is how width and depth were chosen in this project, written so it applies to any architecture: MLP, CNN, RNN/LSTM/GRU, transformer or KBNN. The project's own record, with every table, is `SIZING.md`.

The core is the course's training procedure (Zhang, Gupta and Devabhaktuni, IEEE T-MTT 2003): set a target accuracy, grow the network, and judge each size by its training and validation errors together.

## The procedure in brief

```
1. fix the data splits, the error metric and the training recipe
2. set the target error E_d from the application
3. choose the family from the data's structure; name its width and depth knobs
4. depth = the smallest that can represent the structure
5. for each width, smallest first:
       train k seeds; E_tr, E_v = mean over the seeds
       E_tr > E_d             underlearning   -> next width
       E_tr <= E_d < E_v      overlearning    -> step 6
       E_tr <= E_d, E_v <= E_d  good learning -> stop: this is the size
6. if E_tr stops falling above E_d: add one depth unit and redo step 5
7. freeze; evaluate the test (and extrapolation) sets once
```

## Step 1: fix everything except size

Sizes can only be compared if nothing else changes between them.

- **Splits: train / validation / test**, plus an extrapolation set if the model will be used outside its training range.
  - Split by the independent unit:
    - one sample point for tabular data (our Sobol points)
    - one image or waveform for a CNN
    - one whole sequence, run or circuit for an RNN
  - Putting neighbouring time steps of the same waveform into both train and validation leaks: the two are nearly identical, so the validation error looks good while the network memorizes.
- **Metric: one number per network**, in terms the application cares about. Define it before training. Ours is the RMS relative error over both outputs.
- **Training recipe: the same at every size.** That means the optimizer, learning-rate schedule, epochs, loss and early stopping.
  - The recipe must be enough for the largest size: training error should fall as the network grows.
  - If a bigger network has a *higher* training error, the optimizer is failing, not the size. Fix the recipe and rerun before going on.
- **Seeds: 3–5 per size.** Decide on the mean and report the spread. At small sizes the seed-to-seed spread can exceed the difference between neighbouring sizes.
- **Test set: untouched until the end.** It measures the final model. Using it to pick the size turns it into a second validation set.

## Step 2: set the target error E_d from the application

- **Work backwards from what the model is used for.**
  - Ours: the metastability rate goes as e^(−t/τ), so a delay error δt multiplies the rate by e^(δt/τ). With τ ≈ 10 ps, 1 ps is about a 10 % rate error.
  - E_d = 0.5 % is about 0.5 ps on a 100 ps delay, half that budget.
- **Noisy data** (measurements, Monte Carlo with mismatch): E_d cannot be below the noise floor. Estimate the floor from repeated points and set E_d above it.
- **Without a target there is no stopping point.** On noise-free data (any simulator), validation error keeps falling with size. Picking the lowest validation error therefore picks the largest network tried: our first rule chose 640 neurons (exp_04).

## Step 3: choose the family from the data's structure, then name its knobs

The family is chosen for the structure it builds in, not for its size. Each family then has its own meaning of width and depth.

| Family | Use when the input is | Width knob | Depth knob | Fixed by the problem, not tuned |
|---|---|---|---|---|
| MLP | a few scalar variables, smooth response (our latch) | neurons per hidden layer | hidden layers | inputs, outputs, activation |
| CNN (1-D or 2-D) | a grid with local, shift-invariant patterns: waveforms, spectra, images, layouts | channels per layer; usually one base width, doubled each stage | convolution blocks | kernel size (3 is typical), input length, required receptive field |
| RNN, LSTM, GRU | a sequence whose output depends on its history: transient responses, time series | hidden-state size | stacked recurrent layers | time step, training window length |
| Transformer | long sequences or sets with long-range dependencies | model dimension (feed-forward width and heads scale with it) | blocks | context length, tokenization |
| KBNN, physics-informed | a known equation or empirical law | neurons per sub-network | layers per sub-network | the knowledge formula; which inputs each block sees |
| GNN | a graph: netlist, mesh | node feature size | message-passing rounds | the graph |

**One capacity measure for every family: trainable weights**, including biases. Compare it with the number of independent training values:

- **Tabular:** samples × outputs. Ours is 2800 × 2 = 5600 values. The chosen ReLU MLP has 352 weights; the rejected 640-neuron one had 4482, almost one weight per value.
- **Sequences:** time steps within one run are strongly correlated. The real data size is closer to the number of runs or operating conditions than to the number of time steps.
- **Shared weights:** a CNN reuses its weights at every position and an RNN at every time step. Their weight count does not grow with input length, so a long input is not a large data set.

When the weights approach the number of independent values, expect overlearning.

## Step 4: start at the smallest depth that can represent the structure

Choose depth first, from the structure. Width then provides the accuracy.

- **MLP:** one hidden layer. It can approximate any smooth function given enough neurons.
- **CNN:** enough layers that the receptive field covers the largest feature that decides the output.
  - For kernel k and strides s: RF = 1 + Σ_l (k_l − 1) · Π_{i<l} s_i.
  - With k = 3 and stride 1 that is 1 + 2L, so a 400-sample waveform needs 200 layers. With stride 2 in every layer it is 2^(L+1) − 1, so 8 layers suffice.
  - Alternatively, end with global pooling.
- **RNN:** one recurrent layer, since the recurrence already carries the history. The training window (truncated back-propagation) must be longer than the longest dependency.
- **Transformer:** one or two blocks.
- **KBNN:** one hidden layer per sub-network.

## Step 5: grow the width and classify each size

1. **Coarse sweep, doubling** (5, 10, 20, 40, 80 …) to find roughly where E_tr crosses E_d (our exp_01).
2. **Fine sweep, linear steps** around the crossing (our exp_06: 10, 15, 20 … 80).
3. **Classify each size**, using the mean over the seeds:

   | State | Condition | Meaning |
   |---|---|---|
   | underlearning | E_tr > E_d | too small |
   | overlearning | E_tr ≤ E_d < E_v | fits its training points but not new ones |
   | good learning | E_tr ≤ E_d and E_v ≤ E_d | candidate |

4. **Choose the smallest size with good learning.**

**Second diagnostic: E_v / E_tr.**
- About 1.0–1.2 is healthy. Our final networks are at 1.02–1.18; the rejected 640-neuron network was at 3.07.
- A size labelled "overlearning" with a ratio near 1.05 is only on the threshold, not memorizing; the next size up will be good. This happened to our ReLU MLP at 40 and KBNN at 20.

**Families with several width knobs** (CNN channels per stage, a transformer's dimension, heads and feed-forward width): scale them all by one multiplier and keep their ratios fixed. For example, channels 16-32-64 × 1.5 → 24-48-96. One knob gives one curve and one decision.

## Step 6: what to do in each state

| What you see | Meaning | Action |
|---|---|---|
| E_tr > E_d, still falling with width | too small | next width |
| E_tr > E_d, flat over two doublings | width is not the bottleneck | add one depth unit (layer or block) and redo the width sweep from small; or improve the input features; check the recipe |
| E_tr ≤ E_d < E_v | overlearning | in order: more training data; a smaller network; regularization (weight decay, dropout, augmentation for a CNN) |
| a bigger network has higher E_tr | optimization failure, not capacity | fix the recipe and rerun |
| both ≤ E_d | good learning | stop: the smallest such size |
| nothing reaches E_d, even one layer deeper | limit of the family or the inputs, or E_d below the noise | revisit E_d, the inputs or the family |

- **Treat a deeper network as a different function class**, tested by the same rule; it is not an automatic improvement. Keep the extra depth only if it reaches E_d when one layer cannot, or reaches it with fewer weights.
  - In our exp_05, a second layer made both ReLU families worse.
  - It made the tanh MLP 3–4× more accurate, because two tanh layers can form products of inputs that one layer cannot.
  - One layer met E_d for every family, so one layer was kept.
- **The cost of more data depends on the source.** With a simulator it is the cheapest fix for overlearning: one more Spectre batch. With measurements it is the most expensive.

## Step 7: freeze, then test once

- Keep the chosen size. Report its mean ± sd over the seeds, and use one network downstream: ours is the seed with the best validation error.
- **Evaluate the test set once.** If test error is well above validation error, the validation set was unrepresentative or overused. Do not resize based on the test set.
- **The extrapolation set measures mostly the family's structure, not its size.**
  - Mean t_d error from 1 to 100 µV: KBNN (381 weights) 0.6 %, ReLU MLP (352) 4.2 %, tanh MLP (86) 3.6 %.
  - Size helps the MLPs only a little. The superseded 4482-weight ReLU MLP and 7042-weight tanh MLP reached 2.8 % and 1.2 % (worst 12.5 % and 13 %), still behind the small KBNN (worst 2.9 %).
  - Building in the right structure helps far more than adding weights.

## Pitfalls

1. **Lowest-validation selection on noise-free data** picks the largest network tried (exp_04: 640 neurons, val/train 3.07).
2. **Comparing sizes trained with different budgets.** exp_01 used 30k epochs and exp_04 60k, so sizes were compared only within one experiment.
3. **Deciding on one seed.**
4. **Choosing or tuning on the test set.**
5. **Correlated samples on both sides of the train/validation split:** neighbouring time steps, neighbouring pixels, the same die or circuit.
6. **Adding depth by default.**

**Limits.** The procedure targets small engineering models, where the smallest adequate network is preferred because it is cheap to evaluate and behaves predictably.

Very large image or language networks are often trained far past the weight-to-data ratio where this procedure says "overlearning", and rely on regularization and huge data sets ("double descent"). The step 5 classification still tells you whether validation meets the target, but "smallest good size" is no longer the usual goal there.

## Worked example: this project

| Step | What was done |
|---|---|
| 1 | Train: 2800 scrambled Sobol points. Validation and test: 600 independent random points each, split by point. Extrapolation: 500 points at 1–100 µV. Full-batch Adam, 60k epochs, best-validation snapshot, 5 seeds. |
| 2 | E_d = 0.5 % RMS relative error, from the 1 ps delay budget |
| 3 | MLP (4 scalar inputs, smooth response) and KBNN (log law known). Width = hidden neurons, per block for the KBNN; depth = hidden layers. |
| 4 | One hidden layer |
| 5 | Widths 10–80 → ReLU MLP 50 (352 weights), tanh MLP 12 (86), KBNN 25 per block (381) |
| 6 | Two hidden layers tried (exp_05); not needed |
| 7 | `final_eval.py` run once on test and extrapolation |

## The same procedure for a CNN or an RNN (sketches)

These are hypothetical, built on the same latch, to show what changes.

**1-D CNN predicting t_d and E from the X and Y waveforms** (2 channels, 400 samples at 1 ps):
- **Data size and split:** the number of waveforms, not samples. Split by waveform.
- **Depth:** stride-2 blocks until the receptive field covers 400 samples, so 8 layers; or fewer layers ending in global pooling.
- **Width:** base channels c ∈ {4, 8, 16, 32} with stages c, 2c, 4c; then a fine sweep around the crossing.
- **Classification:** E_tr and E_v against the same 0.5 % E_d.

**GRU modelling the latch transient** (inputs per time step: clk, INP, INN, VDD, T; outputs: X, Y, OUTP):
- **Data size and split:** the number of transient runs. Split by run, so validation uses input levels and conditions never seen in training.
- **Metric:** follows the application. Take the t_d extracted from the predicted waveform, in ps, rather than a per-sample waveform error.
- **Fixed by the problem:** the training window must exceed the longest metastable hold. At 1 nV that is about 250 ps, more than 250 steps at 1 ps.
- **Width:** hidden-state size 8, 16, 32 …
- **Depth:** one layer first, a second only if E_tr stalls.
