# What the leakage calculation establishes

**Preliminary.** The argument below has not been independently checked. The measured quality and the arithmetic can be checked separately from the claims about all protocols in the stated class.

## Setting and assumptions

A client wants next-token distributions for its private text; a server owns the model. Embedding lookup, attention projections, MLP projections and the output head are optical products. Other forward-pass operations are digital on the client. Biases and normalisation parameters are public and excluded from the weight count. A tied embedding is counted once.

Write matrix sizes as $`N_l=m_l n_l`$, mean squares as $`s_l`$, and $`N=\sum_l N_l`$. Weight leakage is the Holevo information between weights and the client's entire final state, divided by $N$, maximised over client strategies and weight priors with the prescribed matrix mean squares. Strategies may act coherently across all light and all texts, and replace the fibre by any quantum channel. Data leakage is the analogous worst-case information about the client's text available to the server, divided by its token count. Only one party is dishonest at a time.

The claimed floor is restricted by these assumptions:

- **Uniform encoding:** for each text, the honest products use $`W_l+Z_l`$, with i.i.d. entries of $`Z_l`$ and one variance $`v_l`$ per matrix. Noise is fresh and independent across texts.
- **Optical roles:** the server contributes weights only through its light; the client contributes data only through its own optics and detection. Return-light protocols and protocols that send the client's data as light are outside the claimed scope.
- **Weight-blindness:** conditional on activations, the honest error law does not depend on the weights. Without this condition, a construction that transmits a fixed model only when the weights equal that model has at most one bit of information per network under this definition, yet gives that model exactly.
- **Transformations:** output-preserving transformations drawn afresh per text from a weight-independent law are allowed. The matching argument below has additional rank limitations.
- **Convex quality:** expected next-token negative log likelihood is convex in the variance vector over the feasible region. The numerical checks do not prove this.
- **A1, for non-Gaussian laws:** among i.i.d. laws of a given variance, Gaussian noise is quality-optimal. Least information at fixed covariance alone does not imply best quality. A1 has not been tested on these language models.

## P0 and an attack that approaches it

For every weight in each encoded matrix view and text, send $`|\alpha_l(w+z)\rangle`$, with $`z\sim\mathcal N(0,v_l)`$ and $`\alpha_l^2\geq10^4/v_l`$. The client uses passive optics to route the span of its activations into measured modes, homodynes those modes, and forms the products digitally. No input-dependent information returns, so the server's text leakage is exactly zero.

With $`\Gamma_l=s_l/v_l`$, the bright-limit leakage after $K$ texts is

```math
L(K)=\sum_l\frac{N_l}{2N}\log_2(1+K\Gamma_l).
```

For tied embedding and output-head views, add their SNRs before taking the logarithm, then count the matrix once. The bound has no abort condition or failure probability. It maximises over all client strategies, rather than only measurements of separate pulses.

To see the optical bound, use quadrature $`x=a+a^\dagger`$ with vacuum variance one. Define

```math
g(\nu)=\frac{\nu+1}{2}\log_2\frac{\nu+1}{2}-\frac{\nu-1}{2}\log_2\frac{\nu-1}{2}.
```

A passive transformation combines the independent copies of a weight into a single signal mode. Its Holevo information is bounded by

```math
g\!\left(\sqrt{1+4\alpha^2v+4K\alpha^2s}\right)-g\!\left(\sqrt{1+4\alpha^2v}\right).
```

Gaussian average states maximise entropy at fixed covariance; an isotropic Gaussian weight prior attains the stated matrix bound. The finite-brightness expression is no greater than the bright bound parameterised by $s/v$. The actual finite-brightness homodyne SNR is $`s/(v+1/(4\alpha^2))`$, a different quantity. The code distinguishes them.

An explicit client homodynes every mode and averages the copies. Its weight estimate has variance $`v_l/K`$ in the bright limit, or relative RMS error $`1/\sqrt{K\Gamma_l}`$. The attack simulation fits a linear estimator on its own samples, without using the capacity value as its answer. Its near-saturation supports the interpretation as an exposed noisy model. It is a Monte Carlo calculation under the worst-case Gaussian prior, not a measurement of information under an empirical distribution of trained models.

The motion visual uses this unbiased averaged copy, normalising its expected error by the matrix's weight RMS. It does not divide by each individual weight or show a realised reconstruction. Its symmetric bands mark one RMS error either side of zero estimation error, not uncertainty about a measured mean. For independent tied lookup and head observations, inverse-variance combination gives an effective variance $`1/[K(1/v_{\rm embed}+1/v_{\rm head})]`$; since their weights and mean square are shared, their $`\Gamma`$ values add. The movie includes every integer $K$ from 1 to 489 with fixed axes and no weight samples. This yields an error reduction of $`\sqrt{489}`$, about 22 times, while information grows logarithmically. The 489-text value is the upper count for a million tokens, whereas the floor uses 488 complete texts.

## From honest outputs to conditional floors

**Gaussian comparison, Lemma A.** For Gaussian weights and independent additive noise of fixed covariance, conditional entropy is bounded by the entropy of the linear-estimation error. This gives the Gaussian-channel information as a lower bound; Gaussian noise attains equality. No quality assumption is needed for this information comparison.

**Chain rule, Lemma B.** Decompose an isotropic Gaussian weight prior into independent orthogonal subspaces. The Holevo chain rule lets each conditional term be bounded using its own measurement on the client's state, even when those measurements cannot be performed together.

**Zero-budget switching, Theorem 2.** If the server's state, including its weight register, is independent of the text, Uhlmann's theorem gives a client-side transformation between purified executions on different texts. Applying the chain argument to spanning activation directions gives at least a full noisy weight copy per text. Fresh randomness permits repetition. This step presupposes the required spanning inputs and the stated error model.

**Approximate switching, Theorem 3.** At budget $b$, let $`\epsilon=2048b`$ and $`\tau=\sqrt{1-2^{-\epsilon}}`$. A coupling argument bounds the loss from imperfect switching. For a zero-budget floor $`C_0`$ in total bits and $J$ chain steps, the implementation uses

```math
C_b\geq\max\{0,(1-\tau)C_0-\sqrt{2\log_2(e)\tau(1-\tau)JC_0}-J[h_2(\tau)+\tau\log_2(1/\tau)]\}.
```

The Gaussian information-density variance bound supplies the square-root penalty. This is useful at $b=10^{-4}$. At larger measured budgets the honest-output floor below is stronger. For repeated texts, keeping subsets of slots gives additional switching bounds; an upper estimate for these is below the honest floor in the reported cases, which the CPU command checks.

**Honest information, Lemma C.** Every protocol in the class reveals the products available to its honest client. For an $m$ by $n$ matrix whose activations span rank $r$, concentrating the allowed prior power in that span gives

```math
H_1(v)=\frac{mr}{2}\log_2(1+ns/(rv)).
```

For repeated texts with span bases $`Q_a`$, let $`\mu_i`$ be the nonzero eigenvalues of $`\sum_aQ_aQ_a^T`$, with rank $r$. Repeating $q$ texts gives the expression used in the stored calculation,

```math
H_K(v)=\frac m2\sum_i\log_2(1+(K/q)(n/r)(s/v)\mu_i).
```

Embedding lookup instead counts exposed vocabulary rows. Numerical ranks use a relative threshold of $10^{-6}$; spectral compression takes lower endpoints. **The run files store the $`H_K`$ floor in `test.floor_honest_K`, but its activation spectra were not retained.** It used eight random-token texts. The CPU command imports that measurement and checks the downstream arithmetic; it cannot independently recompute the floor.

**Fresh transformations, Theorem 5.** Where each text exposes full-rank views, rows can be matched between views to undo fresh permutations. The proposed treatment of scalings and residual rotations also relies on matching. The matching argument quantifies a penalty only for permutations; it does not provide a quantitative penalty for every continuous transformation. Rank-deficient views, especially inputs wider than a text, limit this reasoning. The tables must not be read as an independently established theorem for arbitrary masks.

**Single-round reduction, Theorem 6.** Within the stated class, if later server responses depend only on received systems and fresh randomness, with no later weight-dependent information or delayed secret, the client can simulate those responses locally. The chain argument then uses the zero-budget floor at any data budget. This conditional observation does not extend the claimed scope to the published return-light protocol.

## Minimising over quality-compatible variances

A floor at P0's own variance is insufficient: a different allocation might preserve quality with less information. Under convex quality, a feasible allocation obeys the tangent cut

```math
g\cdot(v-v^*)\leq\delta,
```

where $g$ is the expected NLL gradient at the measured point and $`\delta`$ is the slack to the quality target. Gradients use an antithetic Stein estimator with 256 probes and seed 9000. Bonferroni normal intervals give simultaneous nominal 95% coverage of the gradient components. The slack uses a separate lower normal interval for mean NLL over the quality draws. This is not a single exact 95% coverage guarantee for every result.

For each coordinate use the lower gradient endpoint above $`v^*`$ and the upper endpoint below it. The implementation clips gradients at zero, so the interpretation also requires nonnegative true derivatives. Weak duality then gives a lower bound on the minimum information under that relaxed cut. On each variance-grid interval, use decreasing information at the upper endpoint and increasing cut cost at the lower endpoint, including the intervals next to zero and infinity. Every nonnegative multiplier gives a valid bound for these gradient intervals; numerical maximisation improves it without requiring an exact optimum. Convexity is required to justify the cut, not this final grid step.

Tied views use the largest bound from convex mixtures of their separate information functions. Each reported budget entry is the largest floor proved at that budget or any larger budget, since an admissible protocol at a smaller budget is also admissible at a larger one.

## Evidence limits

The convexity diagnostic uses log-variance step 0.3, 32 test windows, four draws and three directions: all matrices, attention matrices and MLP matrices. None is significantly concave at the one-sided 95% threshold. Almost all directions at the 2% and 10% quality levels are significantly convex in this diagnostic; most at 0.5% are inconclusive. All three estimates for 0.6B at 0.5% are slightly negative within one standard error. Curvature in log variance does not establish convexity in variance, even along the sampled directions, and no other directions were tested.

At the 0.5% level, the fraction in matrices whose gradient lower endpoint reaches zero is roughly 17% for 4B and 33% for 8B when weighting encoded matrix views. The 4B diagnostic counts the tied matrix twice in its denominator; with unique weights as the denominator it is 18.35%. Their noise is left unconstrained by that endpoint, weakening the floor. Both denominators are explicit in [the diagnostics](results/diagnostics.csv). Wider activation inputs also weaken the one-text honest bound, particularly for 8B at budgets 0.01 and 1. The reported floors use 256 probes throughout; there is no completed higher-probe measurement.

Small-network context only: on MNIST, A1 held within error at SNR 3 but non-Gaussian laws were up to 0.4 percentage points better at SNR 1 (1.5 standard errors); shaped noise beat uniform P0 without a valid shaped-protocol floor, and an estimate for client-encoded optical data at budget 1 suggested an improvement for one image but was not measured end to end.

The [published return-light protocol](https://doi.org/10.1103/k8wg-qmbh) is outside this bound. So are per-weight shaped noise and client-encoded optical data. No universal optimum over optical inference protocols is claimed.
