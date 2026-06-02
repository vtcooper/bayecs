/*  Sherwood et al. 2020 Bayesian model.

    Three lines of evidence (sec. 7 of Sherwood et al.):

      L_proc    process understanding (sec. 3): aggregate Gaussian on the
                total feedback parameter lambda = -F_2xCO2/S, derived as the
                sum of 11 component-feedback Gaussians (Table 1).
      L_hist    historical warming and TOA imbalance with pattern effect
                (sec. 4, eq. 6):
                N = F + T*(lambda - dlambda).
      L_LGM     Last Glacial Maximum (sec. 5.2.2, eq. 22):
                T_LGM = (F_other_LGM - 0.57*F_2xCO2)
                        / (dlambda_LGM - lambda/(1+zeta)).
      L_plio    mid-Pliocene Warm Period (sec. 5.2.3, eq. 23):
                T_plio = (-F_plio_CO2*(1+fCH4) - F_plio_nonGHG)
                         / (lambda/(1+zeta) - dlambda_plio),
                where F_plio_CO2 = log2(CO2_plio/284) * F_2xCO2.

    Shared parameters: S, F_2xCO2, zeta.

    --- prior choice (sec. 7.2) -----------------------------------------------

    We want the framework to be able to handle both uniform lambda (UL) and Uniform (US)
    The transformation between them is: S=-F_2xCO2/lambda, with both F_2xCO2 and lambda as parameters. 
    The change of variables formula means that p(lambda)=p(S)*|dS/dlambda|.

    *Uniform S*
    With S as a directly-sampled parameter on uniform bounds, the implicit
    prior on (S, F_2xCO2) is uniform x N(F_2xCO2; mu, sig).  Mapped to
    (lambda, F_2xCO2) via lambda = -F_2xCO2/S we get |dS/dlambda| = F_2xCO2/lambda^2,
    which induces a 1/lambda^2 prior on lambda. 

    *Uniform lambda*
    To match the paper's "UL" baseline we can rescale the joint prior of p(S,F_2xCO2) by S^2/F_2xCO2. 
    This way, the change of variable formula will result in an implicit 
    joint prior p(lambda,F_2xCO2) that is uniform in lambda x the gaussian in F_2xCO2. 
    
    In practice, the easiest way to do this is to add the scaling to the unnormalized joint posterior:
        target += log(F_2xCO2) - 2*log(S);

    --- note: F_hist / F_2xCO2 correlation, explored and not implemented -----

    Sherwood sec. 4.1.2 (p. 43) decomposes F_hist into a CO2 component (which
    is proportional to F_2xCO2) and a non-CO2 component (independent), to
    preserve the F_hist / F_2xCO2 correlation.  
    
    I implemented this decomposition and found that it changed the 
    marginal posterior of S by <0.05K at every perecentile for both UL and US 
    cases. This is consistent with SW20's remark on page 13:
    "In practice, Delta_F_2xCO2 contributes very little to the uncertainty in historical
    or paleo forcings and therefore plays a weak role in those likelihoods."
    
    Impementing the modificaitons came with significantly increased code compexity
    and reduced ease of reading the code. An editorial decision was made to revert to 
    a version without the correlation. 
*/


data {
    // Prior choice: 0 = US, 1 = UL
    int<lower=0, upper=1> use_uniform_lambda_prior; 

    // ---- Process likelihood (Sherwood Table 1, aggregate) ----
    // The sum of 11 individual feedback Gaussians (convolves to N(-1.30, 0.44)).
    real          mu_lambda;
    real<lower=0> sig_lambda;

    // ---- Forcing (Sherwood sec. 3.2.1) ----
    real          mu_F2xCO2;
    real<lower=0> sig_F2xCO2;

    // ---- State-dependence (Sherwood Table 7/8) ----
    real          mu_zeta;
    real<lower=0> sig_zeta;

    // ---- Historical (Sherwood Table 5, baseline row) ----
    real          mu_T_hist;
    real<lower=0> sig_T_hist;

    real          mu_N_hist;
    real<lower=0> sig_N_hist;

    real          mu_F_hist;
    real<lower=0> sig_F_hist;

    real          mu_dlambda;
    real<lower=0> sig_dlambda;

    // ---- LGM (Sherwood Table 7) ----
    real          mu_T_LGM;
    real<lower=0> sig_T_LGM;

    real          mu_F_other_LGM;
    real<lower=0> sig_F_other_LGM;

    real          mu_dlambda_LGM;
    real<lower=0> sig_dlambda_LGM;

    // real          mu_alpha;
    // real<lower=0> sig_alpha;

    // ---- Pliocene (Sherwood Table 8) ----
    real          mu_T_plio;
    real<lower=0> sig_T_plio;

    real          mu_CO2_plio;
    real<lower=0> sig_CO2_plio;

    real          mu_F_plio_nonGHG;
    real<lower=0> sig_F_plio_nonGHG;

    real          mu_dlambda_plio;
    real<lower=0> sig_dlambda_plio;

    real          mu_fCH4;
    real<lower=0> sig_fCH4;

    // real          mu_fESS;
    // real<lower=0> sig_fESS;
}
parameters {
    // The sampling space: the independent parameters that are Monte Carlo sampled
    real <lower=0.1, upper=20> S;
    real F_2xCO2;
    real zeta;

    // historical nuisance parameters
    real F_hist;
    real T_hist;
    real dlambda;

    // LGM nuisance
    // real <upper=0> T_LGM;
    real F_other_LGM;
    real dlambda_LGM;
    // real alpha;

    // Pliocene nuisance
    real <lower=0> CO2_plio;
    real F_plio_nonGHG;
    real dlambda_plio;
    real fCH4;
    // real fESS;
}
transformed parameters{

    // These are dependent parameters that are a function of the independent parameters
    real l;        // lambda
    real N_hist;
    real T_LGM;
    real F_plio_CO2;
    real T_plio;

    // parameter formulas
    
    // feedback 
    l       = -F_2xCO2 / S;

    // coupling equations
    N_hist  = F_hist + T_hist * (l-dlambda);

    // F_other_LGM   = 0.57*F_2xCO2 - T_LGM*(l/(1+zeta) + alpha/2*T_LGM); OLD VERSION
    // F_other_LGM   = 0.57*F_2xCO2 - T_LGM*(l/(1+zeta) - dlambda_LGM);
    T_LGM = (F_other_LGM - 0.57*F_2xCO2) / (dlambda_LGM - l/(1 + zeta));

    F_plio_CO2  = log(CO2_plio/284) / log(2) * F_2xCO2;
    // T_plio  = (-F_plio_CO2*(1+fCH4)*(1+fESS)) / (l/(1+zeta)); OLD VERSION
    T_plio  = (-F_plio_CO2*(1+fCH4) - F_plio_nonGHG) / (l/(1+zeta) - dlambda_plio);
}
model {
    // Shared
    F_2xCO2 ~ normal(mu_F2xCO2, sig_F2xCO2);
    zeta    ~ normal(mu_zeta, sig_zeta);

    // Process likelihood
    l ~ normal(mu_lambda, sig_lambda);

    // Historical
    F_hist  ~ normal(mu_F_hist, sig_F_hist);
    T_hist  ~ normal(mu_T_hist , sig_T_hist);
    N_hist  ~ normal(mu_N_hist , sig_N_hist);
    dlambda ~ normal(mu_dlambda, sig_dlambda);

    // LGM
    T_LGM       ~ normal(mu_T_LGM, sig_T_LGM);
    F_other_LGM ~ normal(mu_F_other_LGM, sig_F_other_LGM);
    dlambda_LGM ~ normal(mu_dlambda_LGM, sig_dlambda_LGM);

    // Pliocene
    T_plio        ~ normal(mu_T_plio, sig_T_plio);
    CO2_plio      ~ normal(mu_CO2_plio, sig_CO2_plio);
    dlambda_plio  ~ normal(mu_dlambda_plio, sig_dlambda_plio);
    fCH4          ~ normal(mu_fCH4, sig_fCH4);
    F_plio_nonGHG ~ normal(mu_F_plio_nonGHG, sig_F_plio_nonGHG);
    // fESS       ~ normal(mu_fESS, sig_fESS);

    // UL prior: convert from default (uniform-S) to uniform-lambda 
    // by multiplying by the Jacobian (or its inverse)
    if (use_uniform_lambda_prior == 1)
        target += log(F_2xCO2) - 2 * log(S);
}
