/*  Sherwood et al. 2020 Bayesian model.

    Three lines of evidence (sec. 7 of Sherwood et al.):

      L_proc    process understanding (sec. 3): aggregate Gaussian on the
                total feedback parameter lambda = -F_2xCO2/S, derived as the
                sum of 11 component-feedback Gaussians (Table 1).
      L_hist    historical warming and TOA imbalance with pattern effect
                (sec. 4, eq. 6):
                N = F + T*(lambda - dlambda).
                Historical forcing is decomposed as
                F = F_CO2 + F_anthro_aerosol + F_other, with
                F_CO2 proportional to the shared F_2xCO2 parameter.
      L_LGM     Last Glacial Maximum budget residual:
                N_LGM = F_other_LGM + f_CO2_LGM*F_2xCO2
                        + T_LGM*(lambda/(1+zeta) - dlambda_LGM).
                The signed dimensionless multiplier f_CO2_LGM is sampled,
                so the LGM CO2 forcing inherits uncertainty from both the
                shared F_2xCO2 and forcing state dependence.
      L_plio    mid-Pliocene Warm Period budget residual:
                N_plio = F_plio_CO2*(1+fCH4) + F_plio_nonGHG
                         + T_plio*(lambda/(1+zeta) - dlambda_plio),
                where F_plio_CO2 is the Meinshausen et al. (2020) CO2
                concentration-forcing ratio times the shared F_2xCO2.

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

    --- F_hist / F_2xCO2 correlation -----------------------------------------

    Sherwood sec. 4.1.2 (p. 43) decomposes F_hist into a CO2 component (which
    is proportional to F_2xCO2) and a non-CO2 component (independent), to
    preserve the F_hist / F_2xCO2 correlation. Here the non-CO2 term is split
    further into anthropogenic aerosol (ARI + ACI) and residual other forcing
    so that aerosol forcing can be diagnosed and varied explicitly.
*/

functions {
    // Stratospheric-adjusted CO2 forcing from Meinshausen et al. (2020),
    // evaluated at preindustrial N2O = 273 ppb. A common SARF-to-ERF factor
    // cancels when this function is used as a ratio to its own CO2 doubling.
    real meinshausen_co2_sarf(real co2_ppm) {
        real a1 = -2.4785e-7;
        real b1 = 7.5906e-4;
        real c1 = -2.1492e-3;
        real d1 = 5.2488;
        real co2_ref = 277.15;
        real n2o_pi = 273.0;
        real co2_alpha_max = co2_ref - b1 / (2 * a1);
        real alpha_prime;

        if (co2_ppm <= co2_ref)
            alpha_prime = d1;
        else if (co2_ppm < co2_alpha_max)
            alpha_prime = d1 + a1 * square(co2_ppm - co2_ref)
                             + b1 * (co2_ppm - co2_ref);
        else
            alpha_prime = d1 - square(b1) / (4 * a1);

        return (alpha_prime + c1 * sqrt(n2o_pi)) * log(co2_ppm / co2_ref);
    }
}


data {
    // Prior choice: 0 = US, 1 = UL
    int<lower=0, upper=1> use_uniform_lambda_prior; 

    // ---- Process likelihood (Sherwood Table 1, aggregate) ----
    // The sum of 11 individual feedback Gaussians (convolves to N(-1.30, 0.44)).
    real          mu_lambda;
    real<lower=0> sig_lambda;

    // ---- Forcing (Sherwood sec. 3.2.1) ----
    real<lower=0> erf_2x;
    real<lower=0> sig_F2xCO2;

    // ---- State-dependence (Sherwood Table 7/8) ----
    real          mu_zeta;
    real<lower=0> sig_zeta;

    // ---- Historical (Sherwood Table 5, baseline row) ----
    real          mu_T_hist;
    real<lower=0> sig_T_hist;

    real          mu_N_hist;
    real<lower=0> sig_N_hist;

    // Central historical CO2 forcing evaluated at erf_2x. Its uncertainty
    // from radiative efficiency is inherited from the shared F_2xCO2 draw.
    real          mu_F_CO2_hist;

    real          mu_F_anthro_aerosol_hist;
    real<lower=0> sig_F_anthro_aerosol_hist;

    real          mu_F_other_hist;
    real<lower=0> sig_F_other_hist;

    // Historical pattern-effect prior: skew-normal(location, scale, shape).
    real          loc_dlambda;
    real<lower=0> scale_dlambda;
    real          shape_dlambda;

    // ---- LGM (Sherwood Table 7) ----
    real          mu_T_LGM;
    real<lower=0> sig_T_LGM;

    real          mu_F_other_LGM;
    real<lower=0> sig_F_other_LGM;

    // Signed LGM CO2 forcing multiplier relative to F_2xCO2.
    real          mu_f_CO2_LGM;
    real<lower=0> sig_f_CO2_LGM;

    real          mu_N_LGM;
    real<lower=0> sig_N_LGM;

    real          mu_dlambda_LGM;
    real<lower=0> sig_dlambda_LGM;

    // real          mu_alpha;
    // real<lower=0> sig_alpha;

    // ---- Pliocene (Sherwood Table 8) ----
    real          mu_T_plio;
    real<lower=0> sig_T_plio;

    real          mu_CO2_plio;
    real<lower=0> sig_CO2_plio;
    real<lower=0> CO2_PI;

    real          mu_F_plio_nonGHG;
    real<lower=0> sig_F_plio_nonGHG;

    real          mu_N_plio;
    real<lower=0> sig_N_plio;

    real          mu_dlambda_plio;
    real<lower=0> sig_dlambda_plio;

    real<lower=-1, upper=1> rho_dlambda_LGM_plio;

    real          mu_fCH4;
    real<lower=0> sig_fCH4;

    // real          mu_fESS;
    // real<lower=0> sig_fESS;
}
parameters {
    // The sampling space: the independent parameters that are Monte Carlo sampled
    real <lower=0.1, upper=20> S;
    real<lower=0> F_2xCO2;
    real zeta;

    // historical nuisance parameters
    real F_anthro_aerosol_hist;
    real F_other_hist;
    real T_hist;
    real dlambda;

    // LGM nuisance
    real T_LGM;
    real F_other_LGM;
    real f_CO2_LGM;
    real dlambda_LGM;
    // real alpha;

    // Pliocene nuisance
    real T_plio;
    real <lower=0> CO2_plio;
    real F_plio_nonGHG;
    real dlambda_plio;
    real fCH4;
    // real fESS;
}
transformed parameters{

    // These are dependent parameters that are a function of the independent parameters
    real l;        // lambda
    real F_CO2_hist;
    real F_hist;
    real N_hist;
    real F_CO2_LGM;
    real N_LGM;
    real f_CO2_plio;
    real F_plio_CO2;
    real N_plio;

    // parameter formulas
    
    // feedback 
    l       = -F_2xCO2 / S;

    // Historical CO2 forcing shares the same radiative-efficiency uncertainty
    // as F_2xCO2. At F_2xCO2 = erf_2x its central value is mu_F_CO2_hist.
    F_CO2_hist = mu_F_CO2_hist * F_2xCO2 / erf_2x;
    F_hist = F_CO2_hist + F_anthro_aerosol_hist + F_other_hist;

    // coupling equations
    N_hist  = F_hist + T_hist * (l-dlambda);

    F_CO2_LGM = f_CO2_LGM * F_2xCO2;
    N_LGM = F_other_LGM + F_CO2_LGM
            - T_LGM * (dlambda_LGM - l / (1 + zeta));

    f_CO2_plio = (meinshausen_co2_sarf(CO2_plio)
                  - meinshausen_co2_sarf(CO2_PI))
                 / (meinshausen_co2_sarf(2 * CO2_PI)
                    - meinshausen_co2_sarf(CO2_PI));
    F_plio_CO2 = f_CO2_plio * F_2xCO2;
    // T_plio = (-F_plio_CO2*(1+fCH4) - F_plio_nonGHG) / (l/(1+zeta) - dlambda_plio); OLD VERSION
    N_plio = F_plio_CO2*(1+fCH4) + F_plio_nonGHG + T_plio*(l/(1+zeta) - dlambda_plio);
}
model {
    // Shared
    F_2xCO2 ~ normal(erf_2x, sig_F2xCO2);
    zeta    ~ normal(mu_zeta, sig_zeta);

    // Process likelihood
    l ~ normal(mu_lambda, sig_lambda);

    // Historical
    F_anthro_aerosol_hist ~ normal(mu_F_anthro_aerosol_hist,
                                   sig_F_anthro_aerosol_hist);
    F_other_hist          ~ normal(mu_F_other_hist, sig_F_other_hist);
    T_hist  ~ normal(mu_T_hist , sig_T_hist);
    N_hist  ~ normal(mu_N_hist , sig_N_hist);
    dlambda ~ skew_normal(loc_dlambda, scale_dlambda, shape_dlambda);

    // LGM
    T_LGM       ~ normal(mu_T_LGM, sig_T_LGM);
    F_other_LGM ~ normal(mu_F_other_LGM, sig_F_other_LGM);
    f_CO2_LGM   ~ normal(mu_f_CO2_LGM, sig_f_CO2_LGM);
    N_LGM       ~ normal(mu_N_LGM, sig_N_LGM);

    // Pliocene
    T_plio        ~ normal(mu_T_plio, sig_T_plio);
    CO2_plio      ~ normal(mu_CO2_plio, sig_CO2_plio);
    fCH4          ~ normal(mu_fCH4, sig_fCH4);
    F_plio_nonGHG ~ normal(mu_F_plio_nonGHG, sig_F_plio_nonGHG);
    N_plio        ~ normal(mu_N_plio, sig_N_plio);
    // fESS       ~ normal(mu_fESS, sig_fESS);

    // Correlated LGM/Pliocene pattern-effect uncertainty
    {
        vector[2] dlambda_pair;
        vector[2] mu_dlambda_pair;
        matrix[2, 2] cov_dlambda_pair;

        dlambda_pair[1] = dlambda_LGM;
        dlambda_pair[2] = dlambda_plio;

        mu_dlambda_pair[1] = mu_dlambda_LGM;
        mu_dlambda_pair[2] = mu_dlambda_plio;

        cov_dlambda_pair[1, 1] = square(sig_dlambda_LGM);
        cov_dlambda_pair[2, 2] = square(sig_dlambda_plio);
        cov_dlambda_pair[1, 2] = rho_dlambda_LGM_plio * sig_dlambda_LGM * sig_dlambda_plio;
        cov_dlambda_pair[2, 1] = cov_dlambda_pair[1, 2];

        dlambda_pair ~ multi_normal(mu_dlambda_pair, cov_dlambda_pair);
    }

    // UL prior: convert from default (uniform-S) to uniform-lambda 
    // by multiplying by the Jacobian (or its inverse)
    if (use_uniform_lambda_prior == 1)
        target += log(F_2xCO2) - 2 * log(S);
}
