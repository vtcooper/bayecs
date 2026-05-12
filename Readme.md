# Stan implementation of Sherwood Webb et al 2020

# 1. Introduction

This is a Stan-based approach for calculating the posterior distributions of ECS from [Sherwood, Webb et al. 2020](https://doi.org/10.1029/2019RG000678) (hereafter SW20). The original code is slow, taking $\mathcal{O}\!\left(10\;\text{hrs}\right)$ to calculate a full posterior. The Stan implementation brings it down to $\mathcal{O}\!\left(10\;\text{secs}\right)$.

By refactoring the posterior calculation in Stan we hope to achieve two goals: 
1. increase the speed by two orders of magnitude taking advantage of Stan's Hamiltonian Monte Carlo (HMC) sampler
2. take advantage of Stan's expressive natural language syntax to make the Bayesian framework more user friendly.

### Environment: 
 python environment for running the code can be found in `stan.yml` and can be installed and activated with conda. The most important library is the `cmdstanpy` library which is a python wrapper on stan. 
```
conda env create -f stan.yml
conda activate stan
```
### Difference from SW20:
There is only one (intentional) difference from SW20's model. Our model does not account for the correlation between historical forcing ($F_{hist}$) and the forcing associatd with a doubling of CO$_2$  $(F_{2\times\text{CO}_2})$. I tried implementing this correlation and it resulted in very small differences in the posterior  of <0.05K at all percentiles (consistent with SW20), but sacrificed code simplicity and ease of reading. 

### Colab:
A Google colab version of the code can be found here. The colab version may be several commits behind the repository, and is primarily intended as a frictionless demo that does not require installing a new python environment. 
  
### This document   
The rest of this document outlines the overall statistical model and focuses on
a few issues related to the numerical estimation of the posterior.
In order to understand the underlying Bayesian framework, readers
should familiarize themselves with [Sherwood, Webb, et al. 2020](https://doi.org/10.1029/2019RG000678), and
[Marvel and Webb 2025](https://doi.org/10.5194/esd-16-317-2025).


# 2. Stan syntax primer

Stan requires us to define the parameters (i.e. the primary, unconstrained
parameters that will be Monte Carlo sampled), any transformed parameters
(i.e. parameters deterministically constrained by the primary parameters),
and the model formulation. To understand how to formulate a model
in Stan, let's take as an example a posterior with two independent
parameters $\theta_{1,2}$ and two independent pieces of observational
evidence $y_{1}$, $y_{2}$, which would make the posterior:

$$
p(\theta\mid y)=p(y_{1},y_{2}\mid\theta_{1},\theta_{2})\,p(\theta_{1},\theta_{2})=p(y_{1}\mid\theta_{1},\theta_{2})\,p(y_{2}\mid\theta_{1},\theta_{2})\,p(\theta_{1})\,p(\theta_{2})
$$

If, say, the probabilities of $y_{1},y_{2},\theta_{1},\theta_{2}$
are Gaussian, we could also write this Bayesian model as

$$
y_{1}\mid\theta_{1},\theta_{2}\sim\mathcal{N}\!\left(\mu_{y_{1}}(\theta_{1},\theta_{2}),\;\sigma_{y_{1}}\!\left(\theta_{1},\theta_{2}\right)\right)
$$

$$
y_{2}\mid\theta_{1},\theta_{2}\sim\mathcal{N}\!\left(\mu_{y_{2}}(\theta_{1},\theta_{2}),\;\sigma_{y_{2}}\!\left(\theta_{1},\theta_{2}\right)\right)
$$

$$
\theta_{1}\sim\mathcal{N}\!\left(\mu_{\theta_{1}},\sigma_{\theta_{1}}\right)
$$

$$
\theta_{2}\sim\mathcal{N}\!\left(\mu_{\theta_{2}},\sigma_{\theta_{2}}\right).
$$

Stan uses two possible formulations. The simplest form of the syntax
is based on the second, distributional formulation, where we define
the model by simply writing out the four distributions above.

In the back-end, each distributional statement adds to the log-likelihood.
For example, since $\theta_{1}$ has a Gaussian distribution,

$$
p_{\theta_{1}}(\theta_{1})=\left(2\pi\sigma_{\theta_{1}}^{2}\right)^{-1/2}\exp\!\left(-\frac{1}{2}\frac{\left(\theta_{1}-\mu_{\theta_{1}}\right)^{2}}{\sigma_{\theta_{1}}^{2}}\right)
$$
$$
\log p_{\theta_{1}}(\theta_{1})=-\frac{1}{2}\log\!\left(2\pi\right)-\log\sigma_{\theta_{1}}-\frac{1}{2}\frac{\left(\theta_{1}-\mu_{\theta_{1}}\right)^{2}}{\sigma_{\theta_{1}}^{2}}
$$

The distributional statement
$$
\theta_{1}\sim\mathcal{N}\!\left(\mu_{\theta_{1}},\sigma_{\theta_{1}}\right)
$$
is equivalent to adding $\log p_{\theta_{1}}(\theta_{1})$ to the
log-posterior (the `target`, in Stan language),
$$
\log\mathcal{L}=\log\mathcal{L}+\log p_{\theta_{1}}(\theta_{1}).
$$
We can also directly add to log-posterior, and we will need to do that
to account for a change of variables if we want to go from uniform
$S$ to uniform $\lambda$ priors without having to change the model structure. 

# 3. The Sherwood-Webb et al. model

### 3.1 Lines of evidence

SW20 uses four lines of evidence (sec. 7 of SW20): process, historical,
cold past climates (LGM) and warm past climates (Pliocene). Under
the independence ansatz (eq. 11), the posterior for climate sensitivity,
$S$, is:

$$
p(S\mid E_{\text{proc}},E_{\text{hist}},E_{\text{paleo}})\propto p\!\left(E_{\text{proc}}\mid S\right)\,p\!\left(E_{\text{hist}}\mid S\right)\,p\!\left(E_{\text{LGM}}\mid S\right)\,p\!\left(E_{\text{plio}}\mid S\right)\,p_{S}(S)
$$

$$
p(S\mid E_{\text{proc}},E_{\text{hist}},E_{\text{paleo}})\propto\mathcal{L}_{\text{proc}}\!\left(S\right)\mathcal{L}_{\text{hist}}\!\left(S\right)\mathcal{L}_{\text{LGM}}\!\left(S\right)\mathcal{L}_{\text{plio}}\!\left(S\right)p_{S}(S)
$$

The parameters and their coupling equations are as follows:

- **Shared parameters:** $S$, $F_{2\times\mathrm{CO}_{2}}$, $\zeta$, with the feedback
  parameter defined as $\lambda=-F_{2\times\mathrm{CO}_{2}}/S$.
- **Process** (sec. 3): $\lambda_{j}$, 11 feedback parameters, each
  with a Gaussian distribution.
- **Historical** (sec. 4): $T_{\text{hist}}, F_{\text{hist}}, N_{\text{hist}}, \Delta\lambda$, coupled
  through equation 6,

  $$
  T_{\text{hist}}=\frac{-\!\left(F_{\text{hist}}-N_{\text{hist}}\right)}{\lambda-\Delta\lambda}.
  $$

- **Cold past climates** (sec. 5.2.2): $F_{\text{LGM}}, T_{\text{LGM}}, \alpha, \zeta$,
  coupled through equation 22,

  $$
  F_{\text{LGM}}=0.57\cdot F_{2\times\mathrm{CO}_{2}}-T_{\text{LGM}}\!\left(\frac{\lambda}{1+\zeta}+\frac{\alpha}{2}T_{\text{LGM}}\right).
  $$

- **Warm past climates** (sec. 5.2.3): $F_{\text{plio}}, T_{\text{plio}}, f_{\text{CH}_4}, f_{\text{ESS}}$,
  coupled through equation 23,

  $$
  T_{\text{plio}}=\frac{-F_{\text{plio}}\cdot\!\left(1+f_{\text{CH}_4}\right)\!\cdot\!\left(1+f_{\text{ESS}}\right)}{\lambda/(1+\zeta)}.
  $$

### 3.2 Nuisance parameters, primary parameters, and constrained parameters

A Bayesian calculation computes the probability of parameters, $\theta$,
given observations $y$:

$$
p(\theta\mid y)\propto p\!\left(y\mid\theta\right)p(\theta).
$$

The SW20 framework is unusual in that it does not provide data points
$y_{j}$ that can be evaluated by plugging them into a parametric likelihood.
Rather, it provides distributions of the likelihoods for different
lines of evidence, $p\!\left(E_{i}\mid S\right)$, that are written in terms
of the distributions of certain variables (e.g. $T_{\text{hist}}, F_{\text{hist}}, N_{\text{hist}}, \Delta\lambda$
for the historical evidence). These distributions need to be interpreted
as likelihoods rather than frequentist sample distributions (see also
Marvel and Webb 2025). Let's take the example of the historical data.
For a Bayesian calculation we need to write a likelihood as a probability
of the evidence conditional on the parameter $S$ we want to estimate:

$$
\mathcal{L}_{\text{hist}}\!\left(S\right)=p\!\left(E_{\text{hist}}\mid S\right).
$$

However, SW20 does not provide data points, but rather distributions
for $\left(T_{\text{hist}},N_{\text{hist}},F_{\text{hist}},\Delta\lambda\right)$. The
likelihood of the historical evidence can be written as the likelihood
of a given combination of variables:

$$
\mathcal{L}_{\text{hist}}\!\left(S\right)=p\!\left(E_{\text{hist}}\mid S\right)=p(T_{\text{hist}},N_{\text{hist}},F_{\text{hist}},\Delta\lambda).
$$

Thus $\left(T_{\text{hist}},N_{\text{hist}},F_{\text{hist}},\Delta\lambda\right)$ are
not data points but should be viewed as parameters — *nuisance parameters*,
since we will integrate them out to get a marginal likelihood or posterior
for the parameter of interest $S$.

Ostensibly the likelihood written above is not a function of $S$
(or $\lambda$). However, $S$ links $F,T,N,\Delta\lambda$ such that
*for a given value of $S$*, the four values are not independent.
Thus, if we introduce $S$ as a parameter, only three of the other
four variables are independent, and one has to become dependent on
the others. For example, if we choose temperature as the dependent
variable, we can write

$$
T(F,N,\Delta\lambda,S)=\frac{-\!\left(F-N\right)}{\lambda-\Delta\lambda}=\frac{-\!\left(F-N\right)}{-F_{2\times}/S-\Delta\lambda}.
$$

So we can write temperature as a function of $\lambda$ or of $S, F_{2\times}$
and write the likelihood as:

$$
p\!\left(T,N,F,\Delta\lambda\mid S,F_{2\times}\right)=p(T\mid S,F_{2\times})\,p(N)\,p(F)\,p(\Delta\lambda)
$$

$$
\mathcal{L}=p\!\left(T,N,F,\Delta\lambda\mid S,F_{2\times}\right)\propto\exp\!\left[-\frac{\left(\frac{-\left(F-N\right)}{-F_{2\times}/S-\Delta\lambda}-\mu_{T}\right)^{2}}{2\sigma_{T}^{2}}\right]\exp\!\left[-\frac{\left(N-\mu_{N}\right)^{2}}{2\sigma_{N}^{2}}\right]\exp\!\left[-\frac{\left(\Delta\lambda-\mu_{\Delta\lambda}\right)^{2}}{2\sigma_{\Delta\lambda}^{2}}\right]\exp\!\left[-\frac{\left(F-\mu_{F}\right)^{2}}{2\sigma_{F}^{2}}\right].
$$

Thus temperature is now a *constrained* parameter that is a deterministic
function of other parameters. In practice we will sample the primary
(unconstrained) parameters $\left(N_{\text{hist}},F_{\text{hist}},\Delta\lambda,F_{2\times},S\right)$
via a Monte Carlo technique, and we'll compute the constrained parameter
$T_{\text{hist}}$ (transformed parameter in Stan's language) from the primary
parameters, before plugging it into the likelihood distribution for
$T_{\text{hist}}$. Note that we do not necessarily need to make $T_{\text{hist}}$
the constrained parameter, and we could make $N_{\text{hist}}$, $F_{\text{hist}}$,
or even $\Delta\lambda$ instead.

### 3.3 The full model:


$$
\begin{aligned}
p(S\mid E) & \propto\mathcal{L}_{\text{proc}}\!\left(S\right)\mathcal{L}_{\text{hist}}\!\left(S\right)\mathcal{L}_{\text{LGM}}\!\left(S\right)\mathcal{L}_{\text{plio}}\!\left(S\right)\,p(\zeta)\,p(F_{2\times})\,p_{S}(S)\\
\mathcal{L}_{\text{proc}} & =p(\lambda\mid S,F_{2\times})\\
\mathcal{L}_{\text{hist}} & =p\!\left(T_{\text{hist}}\mid S,N_{\text{hist}},\Delta\lambda,F_{\text{hist}}\right)\,p\!\left(N_{\text{hist}}\right)\,p\!\left(F_{\text{hist}}\right)\,p(\Delta\lambda)\\
\mathcal{L}_{\text{LGM}} & =p\!\left(F_{\text{LGM}}\mid S,T_{\text{LGM}},\zeta,\alpha\right)\,p\!\left(T_{\text{LGM}}\right)\,p(\alpha)\\
\mathcal{L}_{\text{plio}} & =p\!\left(T_{\text{plio}}\mid S,F_{\text{plio}},f_{\text{ESS}},f_{\text{CH}_4}\right)\,p(F_{\text{plio}})\,p\!\left(f_{\text{ESS}}\right)\,p(f_{\text{CH}_4})
\end{aligned}
$$

# 4. Implementation Issues

## 4.1 Choices of constrained parameters

The coupling equations link $S$ with the nuisance parameters used
in the likelihood formulation. However, it is up to us to choose which
of the parameters becomes the constrained parameter. For example, the
following two formulations for the historical likelihood are equivalent:

$$
\mathcal{L}_{\text{hist}}\!\left(S\right)=p(T\mid S,F_{2\times})\,p(N)\,p(F)\,p(\Delta\lambda); \quad\text{with}\quad T=\frac{-\left(F-N\right)}{\lambda-\Delta\lambda}=\frac{-\left(F-N\right)}{-F_{2\times}/S-\Delta\lambda}
$$

$$
\mathcal{L}_{\text{hist}}\!\left(S\right)=p(N\mid S,F_{2\times})\,p(T)\,p(F)\,p(\Delta\lambda); \quad\text{with}\quad N=F-T\!\left(-F_{2\times}/S-\Delta\lambda\right).
$$

However, there are a couple of considerations that can help us decide
which parameter to choose as the constrained parameter:

**1. Pick the formulation that doesn't require solving a non-trivial
inverse.** The LGM equation (eq. 22) is quadratic in $T_{\text{LGM}}$
because of the state-dependence term $\tfrac{\alpha}{2}(\Delta T)^{2}$:

$$
\frac{\alpha}{2}T_{\text{LGM}}^{\,2}+\frac{\lambda}{1+\zeta}T_{\text{LGM}}-\bigl[0.57\,F_{2\times}-F_{\text{LGM}}\bigr]=0.
$$

Treating $T_{\text{LGM}}$ as the S-coupling would require solving
this for $T_{\text{LGM}}$ via the quadratic formula, which has (i)
two roots and a branch-picking problem, (ii) a $1/\alpha$ singularity
that is hit constantly because $\alpha\sim\mathcal{N}(0.1,0.1)$ passes
through zero, and (iii) a discriminant that can go negative near the
boundary of physical configurations, producing NaN in autodiff. Going
the other way — sampling $T_{\text{LGM}}$ as primary and computing
$F_{\text{LGM}}$ forward — is a single evaluation of a smooth
polynomial, with no branches and no singularities. So $F_{\text{LGM}}$ is
forced to be the S-coupling for LGM.

**2. Don't put a kinked / discontinuous likelihood on a transformed
parameter.** $F_{\text{hist}}$ has Sherwood's asymmetric Gaussian observation
(different $\sigma$ above and below $\mu_{F}$); the log-density
jumps by $\log(\sigma_{\text{lo}}/\sigma_{\text{hi}})\approx0.747$
at $F_{\text{hist}}=\mu_{F}$. In the Stan Hamiltonian Monte Carlo
(HMC) sampler, if $F_{\text{hist}}$ were the historical S-coupling
(transformed parameter receiving the asymmetric likelihood), that
discontinuity would propagate through the Hamiltonian energy-balance
equation into the joint log-density in primary-parameter space, biasing
HMC. So we do not want to use $F_{\text{hist}}$. We could choose
either $N_{\text{hist}}$ or $\Delta\lambda$ instead.

The primary unconstrained parameters define Stan's sampling space,
and they are declared in the `parameters` block. We will use the
following unconstrained parameters:

$$
\theta=\bigl[S,\,F_{2\times},\,\zeta,\,F_{\text{hist}},\,N_{\text{hist}},\,\Delta\lambda,\,T_{\text{LGM}},\,\alpha,\,\mathrm{CO}_{2,\text{plio}},\,f_{\mathrm{CH}_4},\,f_{\text{ESS}}\bigr].
$$

Stan allows us to define any number of constrained parameters, which
are declared in the `transformed parameters` block. These will consist
not only of the three parameters defined by the three coupling equations
needed to write the likelihood, but also any other constrained
parameter that might make the model formulation easier, or that we
want to output. For example, we may want to define $\lambda=-F_{2\times}/S$
or $F_{\text{plio}}=\log_2\!\left(\mathrm{CO}_{2,\text{plio}}/284\right)\cdot F_{2\times}$.
In this model formulation, I've chosen $T_{\text{hist}}$, $F_{\text{LGM}}$, and $T_{\text{plio}}$.
Any other formulation that does not violate rules 1 and 2 should work.

## 4.2 Avoiding biases from the historical aerosol forcing formulation
**tl;dr:** $F_{hist}$ has an asymmetric Gaussian likelihood.  Sampling $F_{hist}$ directly through Stan's
creates a finite log-density jump at $F=\mu_F$ that biases HMC.  We sample an unconstrained z-score F_z ~ N(0,1) and define  F_hist = mu_F + F_z * sigma(sign(F_z)) as a transformed parameter; this  induces exactly the same asymmetric distribution but is smooth in sampling space. 

SW20- encode the 5/50/95 percentiles of $F_{\text{hist}}$ via
two half-Gaussians glued at $\mu_{F}$,

$$
p_{F}(F)\;=\;\begin{cases}
\dfrac{1}{\sigma_{\text{lo}}\sqrt{2\pi}}\exp\!\Big(-\tfrac{1}{2}\big(\tfrac{F-\mu_{F}}{\sigma_{\text{lo}}}\big)^{2}\Big), & F<\mu_{F},\\[8pt]
\dfrac{1}{\sigma_{\text{hi}}\sqrt{2\pi}}\exp\!\Big(-\tfrac{1}{2}\big(\tfrac{F-\mu_{F}}{\sigma_{\text{hi}}}\big)^{2}\Big), & F\ge\mu_{F},
\end{cases}\qquad\sigma_{\text{lo}}=1.131,\;\sigma_{\text{hi}}=0.535.
$$

Each half integrates to $\tfrac{1}{2}$, so $p_{F}$ is a proper
PDF, but it is discontinuous at $F=\mu_F$, where the log-density jumps
by

$$
\Delta\log p_{F}\;=\;\log\!\bigl(\sigma_{\text{lo}}/\sigma_{\text{hi}}\bigr)\;\approx\;0.747.
$$

HMC's leapfrog integrator assumes a continuously differentiable target,
and the jump in (log-)density leads to a slight bias in the
posterior.

**Fix.** Introduce $z\sim\mathcal{N}(0,1)$ and define $F$ deterministically:

$$
F(z)\;=\;\mu_{F}\;+\;\begin{cases}
z\,\sigma_{\text{lo}}, & z<0,\\
z\,\sigma_{\text{hi}}, & z\ge0.
\end{cases}
$$

Under this map, $F$ has exactly the asymmetric Gaussian density of
Section 4.2: for $F<\mu_{F}$, we have $z=(F-\mu_{F})/\sigma_{\text{lo}}$
and $\lvert\mathrm{d}F/\mathrm{d}z\rvert=\sigma_{\text{lo}}$, so

$$
p(F)\;=\;\frac{p_{z}(z)}{\lvert\mathrm{d}F/\mathrm{d}z\rvert}\;=\;\frac{1}{\sigma_{\text{lo}}\sqrt{2\pi}}\exp\!\Bigl(-\tfrac{1}{2}\bigl(\tfrac{F-\mu_{F}}{\sigma_{\text{lo}}}\bigr)^{2}\Bigr)
$$

and analogously on the other side. Critically, in the *sampling
space* ($z$-space) the prior is $\mathcal{N}(0,1)$, which is everywhere
smooth. The map $F(z)$ is continuous (both branches give $F=\mu_{F}$
at $z=0$); only its derivative has a kink at $z=0$, which HMC handles
correctly. The discontinuous jump in log-density that confused leapfrog
in $F$-space has been moved to a kink in the *transformation*,
where it is benign.

## 4.3 Switching between uniform-$\lambda$ and uniform-$S$ priors

SW20 presents two choices of priors: uniform in $\lambda$ and uniform
in $S$. If we write the model in $S$-space, with $S$ as a primary
parameter, declaring `S` with `<lower=A, upper=B>` is equivalent
to a prior on $S$ that is uniform on $(A,B)$.

Setting $\lambda=-F_{2\times}/S$ at fixed $F_{2\times}$, the
change of variable gives

$$
\frac{\partial S}{\partial\lambda}\bigg|_{F_{2\times}}=\frac{F_{2\times}}{\lambda^{2}}=\frac{S^{2}}{F_{2\times}},\qquad\frac{\partial\lambda}{\partial S}\bigg|_{F_{2\times}}=\frac{F_{2\times}}{S^{2}}.
$$

The implicit prior on $(\lambda,F_{2\times})$ is then

$$
p(\lambda,F_{2\times})=p_{S}\!\left(S(\lambda,F_{2\times})\right)\,\frac{\partial S}{\partial\lambda}\bigg|_{F_{2\times}}\,p_{F_{2\times}}(F_{2\times})\;=\;p_{S}\!\left(S(\lambda,F_{2\times})\right)\cdot\frac{F_{2\times}}{\lambda^{2}}\cdot p_{F_{2\times}}(F_{2\times})\;\propto\;\frac{F_{2\times}}{\lambda^{2}}\,p_{F_{2\times}}(F_{2\times}),
$$

i.e. *non-uniform* in $\lambda$ at fixed $F_{2\times}$. This
is the "$U_{S}$" prior of Sherwood (sec. 7.2): uniform on $S$.

To recover the paper's $U_{\lambda}$ prior (uniform on each component
feedback $\lambda_{i}$, which by convolution is approximately uniform
on $\lambda$) we can either rewrite the model to use $\lambda$ as
a primary parameter and give it a uniform prior, or make the change
of variables by adding

```stan
target += log(F_2xCO2) - 2 * log(S);
```

This multiplies the joint by $S^{2}/F_{2\times}$, giving an implicit
prior on $(\lambda,F_{2\times})$ proportional to $p_{F_{2\times}}(F_{2\times})$
— i.e. uniform in $\lambda$ at every $F_{2\times}$ slice, with $F_{2\times}$
retaining its Gaussian prior. This matches Sherwood's $U_{\lambda}$
exactly (uniform-$\lambda$ $\times$ Gaussian-$F_{2\times}$).
