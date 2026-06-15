# Theory Note: Why Large-Context Entropic OT-CFM Works

This note records the theoretical story behind the Flash global OT-CFM
experiments.  The goal is not to claim that entropic OT is always better than
exact OT.  The claim is more precise:

> CFM only needs a valid endpoint coupling.  Entropic OT is a valid coupling
> with controlled transport-cost bias, and FlashSinkhorn makes the large-context
> regime accessible.  In that regime, global entropic OT can dominate exact
> local minibatch OT because local OT is feasible-set restricted.

The setup matches the CIFAR-10 and ImageNet-64 experiments in this repository:
training happens in image or latent space, while the coupling cost may be
computed in a fixed feature space

```math
C_{ij} = {1 \over 2 d_h} \|h(x_0^i)-h(x_1^j)\|_2^2.
```

All baselines must use the same feature map \(h\) when this cost is used.

## 1. Compatibility: CFM Works With Any Coupling

Let \(\mu_0,\mu_1\) be source and target distributions on \(\mathbb R^d\), and
let \(\pi \in \Pi(\mu_0,\mu_1)\) be any coupling with finite second moment.
This includes independent coupling, exact OT, entropic OT, local minibatch OT,
or a global context coupling.

Sample \((X_0,X_1)\sim \pi\) and define the deterministic linear conditional
path

```math
X_t = (1-t)X_0 + tX_1,\qquad U = X_1-X_0.
```

Let \(\rho_t = (X_t)_\# \pi\).  Define the Eulerian velocity

```math
v_t^\pi(x) = \mathbb E[U \mid X_t=x].
```

### Theorem 1: Any Endpoint Coupling Defines a Valid CFM Path

For any \(\pi \in \Pi(\mu_0,\mu_1)\), the pair \((\rho_t,v_t^\pi)\) solves the
continuity equation in weak form:

```math
\partial_t \rho_t + \nabla \cdot (\rho_t v_t^\pi)=0,\qquad
\rho_0=\mu_0,\quad \rho_1=\mu_1.
```

Moreover, the CFM objective

```math
{\mathcal L}_{\mathrm{CFM}}(\theta;\pi)
=
\mathbb E_{t\sim U[0,1],(X_0,X_1)\sim\pi}
\|v_\theta(t,X_t)-U\|_2^2
```

has the same minimizer as regression onto the marginal velocity \(v_t^\pi\):

```math
{\mathcal L}_{\mathrm{CFM}}(\theta;\pi)
=
\mathbb E_{t,X_t}\|v_\theta(t,X_t)-v_t^\pi(X_t)\|_2^2
+
\mathbb E_{t,X_t}\operatorname{Var}(U\mid X_t).
```

The second term is independent of \(\theta\).

### Proof Sketch

For any smooth compactly supported test function \(\varphi\),

```math
{d\over dt}\mathbb E[\varphi(X_t)]
=
\mathbb E[\nabla\varphi(X_t)^\top U]
=
\mathbb E[\nabla\varphi(X_t)^\top v_t^\pi(X_t)],
```

which is the weak continuity equation.  The loss decomposition is the standard
squared-loss orthogonal projection identity for conditional expectation.

### Consequence

Entropic OT is not an approximation to the *validity* of CFM.  It only changes
which valid path \(\rho_t\) and velocity \(v_t^\pi\) are learned.

## 2. Entropic OT Is Cost-Near-Optimal

For empirical marginals \(a,b\), let

```math
U(a,b)=\{\pi\ge 0:\pi\mathbf 1=b,\ \pi^\top\mathbf 1=a\}
```

with the convention that \(\pi_{ij}\) couples source \(i\) to target \(j\).  Let
\(\pi^0\) solve exact OT:

```math
\pi^0 \in \arg\min_{\pi\in U(a,b)} \langle C,\pi\rangle.
```

Let \(\pi^\varepsilon\) solve entropic OT:

```math
\pi^\varepsilon
=
\arg\min_{\pi\in U(a,b)}
\langle C,\pi\rangle
+
\varepsilon \mathrm{KL}(\pi\,\|\,a\otimes b).
```

### Theorem 2: Explicit Entropic Bias Bound

For any nonnegative cost matrix \(C\),

```math
\langle C,\pi^0\rangle
\le
\langle C,\pi^\varepsilon\rangle
\le
\langle C,\pi^0\rangle
+
\varepsilon \min\{H(a),H(b)\}.
```

For uniform \(n\times m\) empirical marginals,

```math
\langle C,\pi^\varepsilon\rangle
\le
\langle C,\pi^0\rangle
+
\varepsilon \log \min\{n,m\}.
```

### Proof

The lower bound holds because \(\pi^0\) minimizes \(\langle C,\pi\rangle\).  For
the upper bound, optimality of \(\pi^\varepsilon\) gives

```math
\langle C,\pi^\varepsilon\rangle
+
\varepsilon \mathrm{KL}(\pi^\varepsilon\|a\otimes b)
\le
\langle C,\pi^0\rangle
+
\varepsilon \mathrm{KL}(\pi^0\|a\otimes b).
```

Drop the nonnegative KL term on the left.  The remaining KL is the mutual
information of the discrete source-target pair under \(\pi^0\), hence it is at
most \(\min\{H(a),H(b)\}\).  For uniform marginals this is
\(\log\min\{n,m\}\).

### Consequence

Entropic OT has a simple, explicit cost bias.  This lets us compare it against
local exact OT without hand-waving about approximation.

## 3. Why Global Entropic Can Beat Local Exact

Consider a global empirical batch split into \(B\) equal local blocks, as in
DDP.  Let \(W_{\mathrm{global}}\) be the exact OT cost over the full global
batch.  Let \(W_{\mathrm{block}}\) be the cost of exact OT solved separately
inside each local block and assembled into a block-diagonal plan.

Because block-local plans form a strict subset of all global feasible plans,

```math
W_{\mathrm{global}} \le W_{\mathrm{block}}.
```

Let \(\pi^\varepsilon_{\mathrm{global}}\) be the entropic OT plan over the
global context.  Theorem 2 gives

```math
C(\pi^\varepsilon_{\mathrm{global}})
\le
W_{\mathrm{global}} + \varepsilon \log n
```

for an \(n\times n\) uniform context.

### Corollary 3: Sufficient Condition for Global Entropic to Beat Local Exact

If

```math
W_{\mathrm{block}} - W_{\mathrm{global}} > \varepsilon \log n,
```

then

```math
C(\pi^\varepsilon_{\mathrm{global}}) < W_{\mathrm{block}}.
```

Thus the regularized global plan has lower cost than exact local POT.

### Interpretation

This is the core theory for Flash global OT-CFM:

- local POT is exact but context-starved;
- global entropic OT is approximate but searches a richer feasible set;
- FlashSinkhorn makes the global entropic solve cheap enough to use during
  training.

The ImageNet-64 Phase 1 diagnostic validates the first inequality source:
global exact OT has lower cost than block-local exact OT, and roughly
\(87\%-89\%\) of global assignments cross local blocks.

## 4. Row-Conditional Sampling Is Unbiased for CFM

In training we do not materialize all pairs from \(\pi^\varepsilon\).  For each
source row \(i\), we sample

```math
J \sim q_\varepsilon(j\mid i) = {\pi^\varepsilon_{ij}\over a_i}.
```

If \(I\sim a\) and \(J\mid I\sim q_\varepsilon(\cdot\mid I)\), then

```math
\Pr[I=i,J=j] = a_i{\pi^\varepsilon_{ij}\over a_i}
=\pi^\varepsilon_{ij}.
```

Therefore the sampled endpoint pair has exactly the desired joint law.  In a
finite minibatch where each row is used once, the target marginal is correct in
expectation:

```math
\mathbb E\left[{1\over n}\sum_{i=1}^n \mathbf 1\{J_i=j\}\right] = b_j.
```

This explains why duplicate target samples do not invalidate CFM.  They do
change finite-batch variance, so duplicate fraction must be reported.

## 5. Path Quality and Low-NFE Sampling

For deterministic linear CFM paths, the target velocity is \(U=X_1-X_0\).
Thus the conditional kinetic energy under a coupling \(\pi\) is

```math
\mathbb E_\pi \|U\|_2^2
=
\mathbb E_\pi \|X_1-X_0\|_2^2.
```

If the OT cost is computed in model space, this is exactly the OT objective up
to scaling.  If the OT cost is computed in feature space \(h\), then the same
statement holds in feature geometry:

```math
\mathbb E_\pi \|h(X_1)-h(X_0)\|_2^2.
```

For a finite training context, if \(h\) is a \((1\pm\delta)\) distance-preserving
embedding on the source-target set, then feature-space cost improvements
transfer to raw-space pair distances up to the JL distortion factor:

```math
(1-\delta)\|x-y\|_2^2
\le
\|h(x)-h(y)\|_2^2
\le
(1+\delta)\|x-y\|_2^2.
```

Lower coupling cost should reduce target velocity norm and can reduce gradient
variance.  Empirically this should show up most clearly in low-NFE FID, because
straighter/easier paths require fewer ODE solver steps.

## 6. Statistical Stability Support

Entropic OT is smoother than unregularized OT.  Existing Sinkhorn divergence
theory shows that entropic regularization interpolates between OT and MMD and
can enjoy \(1/\sqrt n\)-type sample behavior at fixed \(\varepsilon\), with
constants depending on \(\varepsilon\).  We should cite this as prior support
rather than present it as a new theorem unless we include a complete proof.

The paper-level message is:

> Entropic regularization is not only computationally convenient.  In the
> minibatch/online setting, it also stabilizes the empirical coupling problem.

## 7. Experimental Checks Required by the Theory

Coupling diagnostics:

- report \(W_{\mathrm{block}}\), \(W_{\mathrm{global}}\), entropic sampled
  cost, and \(\varepsilon\log n\);
- report whether \(W_{\mathrm{block}}-W_{\mathrm{global}}>\varepsilon\log n\);
- report cross-block fraction;
- report duplicate fraction for row-conditional sampling.

Training diagnostics:

- compare independent, local exact POT, local entropic OT, and Flash global
  entropic OT;
- sweep context size and \(\varepsilon\);
- log CFM loss, path/coupling cost, OT time, step time, and images/sec.

Quality metrics:

- report FID/KID versus NFE;
- report same-step and same-wall-clock comparisons;
- use CIFAR-10 repeated seeds and ImageNet-64 as the main scaling benchmark.

## 8. Reviewer Pressure Test

**"OT-CFM uses exact OT; entropic OT is not OT-CFM."**

CFM only requires a coupling with correct marginals.  Entropic OT is such a
coupling.  Exact OT is one useful choice, not a validity requirement.

**"This is just a Schrödinger bridge."**

No.  We use static entropic endpoint coupling with deterministic linear CFM
paths.  We are not adding Brownian bridge noise or matching the full stochastic
Schrödinger bridge dynamics.

**"Entropic OT has worse cost than exact OT."**

True for the same context.  The relevant comparison is not global exact versus
global entropic; it is local exact versus global entropic.  Corollary 3 states
when global entropic wins.

**"Row-conditional sampling duplicates targets."**

Yes.  The target marginal is correct in expectation and the sampled pair law is
the entropic plan when the source row is sampled from \(a\).  Duplicate rate is
a finite-batch variance diagnostic, not a validity failure.

**"Projected-feature OT changes the objective."**

Yes.  We should explicitly define the coupling geometry by \(h\).  CFM still
trains the velocity in pixel or latent space, and every OT baseline must use the
same \(h\).

## 9. Literature Anchors

- Flow Matching shows CFM is an equivalent regression objective for marginal FM
  under conditional probability paths and highlights OT paths as efficient
  straight conditional paths.
- OT-CFM uses OT endpoint couplings to simplify flows and approximate dynamic
  OT when the true OT plan is available.
- Multisample Flow Matching permits nontrivial minibatch couplings with correct
  marginals and links them to lower variance and straighter learned flows.
- Simulation-Free Schrödinger Bridge work uses static entropy-regularized OT in
  simulation-free flow/score matching.
- Cuturi's Sinkhorn work motivates entropic OT as a maximum-entropy smoothing
  of OT that is fast and GPU-friendly.
- Sinkhorn divergence/sample-complexity work supports the statistical-stability
  role of fixed-\(\varepsilon\) regularization.
