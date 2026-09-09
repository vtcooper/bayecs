/*  Community ECS assessment model with data-selectable sensitivity options.

    Core lines of evidence:

      L_proc    process understanding: aggregate Gaussian on the
                total 2xCO2 feedback, lambda = -F_2xCO2/S, formed as the
                sum of component feedbacks.

      L_hist    historical warming and TOA imbalance with pattern effect:
                N = F + (lambda - dlambda)*T.
                Historical forcing is decomposed as
                F = F_CO2 + F_anthro_aerosol + F_other, with
                F_CO2 proportional to the shared F_2xCO2 parameter.

      L_trend   2006--2024 trends in temperature, effective radiative forcing,
                and TOA imbalance:
                N_trend = F_trend + (lambda - dlambda_trend)*T_trend.
                Trend forcing is decomposed as
                F_trend = F_CO2_trend + F_anthro_aerosol_trend
                          + F_other_trend, with F_CO2_trend proportional to
                          the shared F_2xCO2 parameter.
                Matching historical and Trend aerosol components have an
                assessed cross-period covariance, as do the matching
                residual-other forcing components. The Trend T and N
                observations retain their assessed within-period covariance.
                This evidence is included in the current community baseline.
                It is not fully independent of L_hist: shared forcing errors
                are modeled below, while the remaining cross-period error
                covariances are currently treated as negligible.

      L_LGM     Last Glacial Maximum budget residual:
                N_LGM = F_other_LGM + f_CO2_LGM*F_2xCO2
                        + T_LGM*(lambda - dlambda_LGM).
                The signed dimensionless multiplier f_CO2_LGM is sampled,
                so the LGM CO2 forcing inherits uncertainty from both the
                shared F_2xCO2 and forcing state dependence.

      L_plio    Pliocene (mid-Piacenzian Warm Period) budget residual:
                N_plio = F_plio_CO2*(1+fCH4) + F_plio_nonGHG
                         + T_plio*(lambda - dlambda_plio),
                where F_plio_CO2 is the Meinshausen et al. (2020) CO2
                concentration-forcing ratio times the shared F_2xCO2.

    Sampled shared parameters: lambda and F_2xCO2. ECS is the
    transformed output S = -F_2xCO2/lambda.

    Model settings (all are fixed during a fit):
      lambda_prior_type        = 0: uniform-S prior (US)
                                 1: uniform-lambda prior (UL)
                                 2: reflected-lognormal lambda prior, defined
                                    on the positive damping r = -lambda

      include_process          = 0: omit the Process feedback likelihood
                                 1: include it
      use_lambda_F2x_correlation = 0: independent Gaussian assessments of
                                       lambda and F_2xCO2
                                    1: correlated bivariate Gaussian;
                                       requires include_process = 1
      include_historical       = 0: omit the historical energy-budget likelihood
                                 1: include it
      include_trend            = 0: omit the recent-trend likelihood
                                 1: include it (community baseline)
      include_lgm              = 0: omit the LGM energy-budget likelihood
                                 1: include it
      include_pliocene         = 0: omit the Pliocene energy-budget likelihood
                                 1: include it
      use_score_T              = 0: new default, evaluate model N=F+lambda*T
                                 1: rearrange every energy budget and
                                    evaluate model T=(N-F)/lambda.
      use_pattern_effect_copula = 0: skew-normal historical pattern effect,
                                     correlated Gaussian LGM/Pliocene effects
                                  1: three-way Gaussian copula preserving those
                                     three marginal distributions

    When a line is excluded, its budget/observation coupling is removed but
    its nuisance variables retain normalized, proper distributions. This
    keeps one fixed parameter interface for every switch combination. After
    marginalization, those disconnected nuisance variables contribute only a
    constant and cannot change the posterior of S, lambda, or F_2xCO2.

    CAUTION: the recent Trend interval overlaps the historical record. The
    model includes the assessed cross-period forcing covariance, but no
    historical-Trend covariance for temperature, TOA imbalance, pattern
    effects, or other errors. That is an explicit sensitivity assumption, not
    a claim that the two lines of evidence are otherwise independent.


    --- score-N / score-T choice ---------------------------------------------

    For a generic budget N = F + lambda*T, score N samples T and evaluates the
    budget-implied N against its assessment. Score T samples N, rearranges the
    same equation to T = (N-F)/lambda, and evaluates that temperature against
    its assessment. No Jacobian is added: these are deliberately different
    response-variable likelihood conventions, not a reparameterization of one
    common probability model. Score T is singular at lambda = 0, so a numerically
    negligible interval around zero is assigned zero probability.

    Note for paleo: assuming N is exactly zero and evaluating F = -lambda*T
    produces essentially the same result as assuming N is Gaussian with mean
    zero and small uncertainty.


    --- prior choice (sec. 7.2) -----------------------------------------------

    The model samples lambda directly and reports S = -F_2xCO2/lambda as a
    transformed parameter. The lambda bounds depend on F_2xCO2 so that they
    preserve the 0.05 <= S <= 20 K support.

    *Uniform lambda*
    Direct sampling supplies the uniform-lambda measure without an additional
    target adjustment to specify a non-uniform prior.
    
    *Uniform S*
    The change of variables has
        |dS/dlambda| = F_2xCO2/lambda^2.
    Therefore the uniform-S option adds
        target += log(F_2xCO2) - 2*log(-lambda)
    to the density expressed in the sampled (lambda, F_2xCO2) coordinates.

    *Reflected lognormal lambda*
    A conventional lognormal distribution cannot be defined on negative
    lambda. This option therefore places the lognormal density on the positive
    damping magnitude r = -lambda. Its inputs are the median of r and the
    standard deviation of log(r). Since |dr/dlambda| = 1, this reflection does
    not require an additional Jacobian term.


    --- historical/Trend forcing correlation with F_2xCO2 --------------------

    We decompose F_hist into a CO2 component (which is proportional to
    F_2xCO2) and non-CO2 components independent of F_2xCO2, to
    preserve the F_hist / F_2xCO2 correlation. Here the non-CO2 term is split
    further into anthropogenic aerosol (ERFari + ERFaci) and residual other forcing
    so that aerosol forcing can be diagnosed and varied explicitly.

    The recent-Trend forcing uses the same construction: its CO2 component is
    proportional to F_2xCO2, while its anthropogenic aerosol (ARI + ACI) and
    residual-other components are jointly distributed with the matching
    historical components using their assessed covariances. The native CO2
    ensemble spread is not added separately, which would double count the
    shared radiative-efficiency uncertainty.
*/

functions {
    int value_is_finite(real x) {
        return !(is_nan(x) || is_inf(x));
    }

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
    // Prior choice: 0 = US, 1 = UL, 2 = lognormal(-lambda). This is
    // independent of all line-of-evidence switches.
    int<lower=0, upper=2> lambda_prior_type;

    // Reflected-lognormal prior parameters for r = -lambda. The first is the
    // median of r; the second is the standard deviation of log(r).
    real<lower=0> lognormal_r_median;
    real<lower=0> lognormal_r_log_sigma;

    // Line-of-evidence switches.
    int<lower=0, upper=1> include_process;
    int<lower=0, upper=1> include_historical;
    int<lower=0, upper=1> include_trend;
    int<lower=0, upper=1> include_lgm;
    int<lower=0, upper=1> include_pliocene;

    // Global energy-budget response convention: 0 = score N, 1 = score T.
    int<lower=0, upper=1> use_score_T;

    // Optional dependence structures.
    int<lower=0, upper=1> use_lambda_F2x_correlation;
    int<lower=0, upper=1> use_pattern_effect_copula;

    // ---- Process likelihood (aggregate) ----
    // The sum of Gaussian component feedbacks.
    real          mu_lambda;
    real<lower=0> sig_lambda;

    // ---- Forcing assessment ----
    real<lower=0> erf_2x;
    real<lower=0> sig_F2xCO2;

    // Correlation of the Gaussian Process lambda and F_2xCO2 assessments.
    // Negative means larger forcing accompanies more-negative lambda.
    real<lower=-1, upper=1> rho_lambda_F2x;

    // ---- Historical evidence ----
    real          mu_T_hist;
    real<lower=0> sig_T_hist;

    real          mu_N_hist;
    real<lower=0> sig_N_hist;

    // Central historical CO2 forcing evaluated at erf_2x. Its uncertainty
    // is inherited from the shared F_2xCO2 draw.
    real          mu_F_CO2_hist;

    real          mu_F_anthro_aerosol_hist;
    real<lower=0> sig_F_anthro_aerosol_hist;

    real          mu_F_other_hist;
    real<lower=0> sig_F_other_hist;

    // Historical pattern-effect prior: skew-normal(location, scale, shape).
    // Note: shape=0 gives normal(mean=loc, sd=scale), so var=scale^2.
    real          loc_dlambda;
    real<lower=0> scale_dlambda;
    real          shape_dlambda;

    // ---- Recent Trend (2006--2024; all trends are per decade) ----
    real          mu_T_trend;
    real<lower=0> sig_T_trend;

    real          mu_N_trend;
    real<lower=0> sig_N_trend;

    // Covariance of the observed temperature and TOA-imbalance trend errors.
    real          cov_TN_trend;

    // Central CO2 forcing trend evaluated at erf_2x. Its uncertainty from
    // is inherited from the shared F_2xCO2 draw.
    real          mu_F_CO2_trend;

    real          mu_F_anthro_aerosol_trend;
    real<lower=0> sig_F_anthro_aerosol_trend;

    real          mu_F_other_trend;
    real<lower=0> sig_F_other_trend;

    // Final assessed covariance between the historical change and recent
    // trend for each non-CO2 forcing component. Its construction, including
    // structural-error sensitivity choices, is handled before data are passed
    // to Stan. Units are
    // (W m^-2)(W m^-2 decade^-1).
    real cov_F_anthro_aerosol_hist_trend;
    real cov_F_other_hist_trend;

    real          mu_dlambda_trend;
    real<lower=0> sig_dlambda_trend;

    // ---- LGM ----
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

    // ---- Pliocene ----
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

    // Pattern-effect correlations. The historical correlations are used only
    // by the copula; rho_dlambda_LGM_plio is used by both pattern-prior modes.
    real<lower=-1, upper=1> rho_dlambda_hist_LGM;
    real<lower=-1, upper=1> rho_dlambda_hist_plio;
    real<lower=-1, upper=1> rho_dlambda_LGM_plio;

    real          mu_fCH4;
    real<lower=0> sig_fCH4;

    // real          mu_fESS;
    // real<lower=0> sig_fESS;
}
transformed data {
    real score_T_min_abs_feedback;
    matrix[2, 2] cov_lambda_F2x;
    matrix[2, 2] L_lambda_F2x;
    matrix[2, 2] cov_TN_trend_matrix;
    matrix[2, 2] L_TN_trend;
    matrix[2, 2] cov_F_anthro_aerosol_hist_trend_matrix;
    matrix[2, 2] L_F_anthro_aerosol_hist_trend;
    matrix[2, 2] cov_F_other_hist_trend_matrix;
    matrix[2, 2] L_F_other_hist_trend;
    matrix[3, 3] R_dlambda_copula;
    matrix[3, 3] L_dlambda_copula;

    // The score-T equations are undefined at zero effective feedback. This
    // guard is many orders of magnitude below scientifically relevant values.
    score_T_min_abs_feedback = 1e-10;

    if (lambda_prior_type == 2) {
        if (lognormal_r_median <= 0)
            reject("lognormal_r_median must be positive for the lognormal lambda prior.");
        if (lognormal_r_log_sigma <= 0)
            reject("lognormal_r_log_sigma must be positive for the lognormal lambda prior.");
    }

    if (use_lambda_F2x_correlation == 1) {
        if (include_process == 0)
            reject("lambda/F_2xCO2 correlation requires include_process = 1.");
        if (abs(rho_lambda_F2x) >= 1)
            reject("rho_lambda_F2x must be strictly between -1 and 1 when enabled.");

        cov_lambda_F2x[1, 1] = square(sig_F2xCO2);
        cov_lambda_F2x[2, 2] = square(sig_lambda);
        cov_lambda_F2x[1, 2] = rho_lambda_F2x * sig_F2xCO2 * sig_lambda;
        cov_lambda_F2x[2, 1] = cov_lambda_F2x[1, 2];
        L_lambda_F2x = cholesky_decompose(cov_lambda_F2x);
    } else {
        // Assigned but unused when the two assessments are independent.
        cov_lambda_F2x = diag_matrix(rep_vector(1.0, 2));
        L_lambda_F2x = diag_matrix(rep_vector(1.0, 2));
    }

    if (include_trend == 1) {
        if (abs(cov_TN_trend) >= sig_T_trend * sig_N_trend)
            reject("Trend T/N covariance matrix must be positive definite: ",
                   "abs(cov_TN_trend) must be less than ",
                   "sig_T_trend * sig_N_trend.");

        cov_TN_trend_matrix[1, 1] = square(sig_T_trend);
        cov_TN_trend_matrix[2, 2] = square(sig_N_trend);
        cov_TN_trend_matrix[1, 2] = cov_TN_trend;
        cov_TN_trend_matrix[2, 1] = cov_TN_trend;
        L_TN_trend = cholesky_decompose(cov_TN_trend_matrix);
    } else {
        // Assigned but unused; disabled options should not reject on an
        // irrelevant covariance input.
        cov_TN_trend_matrix = diag_matrix(rep_vector(1.0, 2));
        L_TN_trend = diag_matrix(rep_vector(1.0, 2));
    }

    if (include_historical == 1 && include_trend == 1) {
        if (abs(cov_F_anthro_aerosol_hist_trend)
            >= sig_F_anthro_aerosol_hist * sig_F_anthro_aerosol_trend)
            reject("Historical/Trend aerosol forcing covariance matrix must be ",
                   "positive definite.");
        if (abs(cov_F_other_hist_trend)
            >= sig_F_other_hist * sig_F_other_trend)
            reject("Historical/Trend residual-other forcing covariance matrix ",
                   "must be positive definite.");

        cov_F_anthro_aerosol_hist_trend_matrix[1, 1]
            = square(sig_F_anthro_aerosol_hist);
        cov_F_anthro_aerosol_hist_trend_matrix[2, 2]
            = square(sig_F_anthro_aerosol_trend);
        cov_F_anthro_aerosol_hist_trend_matrix[1, 2]
            = cov_F_anthro_aerosol_hist_trend;
        cov_F_anthro_aerosol_hist_trend_matrix[2, 1]
            = cov_F_anthro_aerosol_hist_trend;
        L_F_anthro_aerosol_hist_trend = cholesky_decompose(
            cov_F_anthro_aerosol_hist_trend_matrix
        );

        cov_F_other_hist_trend_matrix[1, 1] = square(sig_F_other_hist);
        cov_F_other_hist_trend_matrix[2, 2] = square(sig_F_other_trend);
        cov_F_other_hist_trend_matrix[1, 2] = cov_F_other_hist_trend;
        cov_F_other_hist_trend_matrix[2, 1] = cov_F_other_hist_trend;
        L_F_other_hist_trend = cholesky_decompose(
            cov_F_other_hist_trend_matrix
        );
    } else {
        // The covariance does not affect either marginal when only one line is
        // enabled, so use independent normalized priors for better sampling.
        cov_F_anthro_aerosol_hist_trend_matrix
            = diag_matrix(rep_vector(1.0, 2));
        L_F_anthro_aerosol_hist_trend = diag_matrix(rep_vector(1.0, 2));
        cov_F_other_hist_trend_matrix = diag_matrix(rep_vector(1.0, 2));
        L_F_other_hist_trend = diag_matrix(rep_vector(1.0, 2));
    }

    R_dlambda_copula = diag_matrix(rep_vector(1.0, 3));
    R_dlambda_copula[1, 2] = rho_dlambda_hist_LGM;
    R_dlambda_copula[2, 1] = rho_dlambda_hist_LGM;
    R_dlambda_copula[1, 3] = rho_dlambda_hist_plio;
    R_dlambda_copula[3, 1] = rho_dlambda_hist_plio;
    R_dlambda_copula[2, 3] = rho_dlambda_LGM_plio;
    R_dlambda_copula[3, 2] = rho_dlambda_LGM_plio;

    if (use_pattern_effect_copula == 1) {
        if (determinant(R_dlambda_copula) <= 0)
            reject("Historical/LGM/Pliocene pattern-effect correlation matrix ",
                   "must be positive definite.");
        L_dlambda_copula = cholesky_decompose(R_dlambda_copula);
    } else {
        if (abs(rho_dlambda_LGM_plio) >= 1)
            reject("LGM/Pliocene pattern-effect correlation must be strictly ",
                   "between -1 and 1 in baseline mode.");
        // Assigned but unused in the baseline pattern-prior mode.
        L_dlambda_copula = diag_matrix(rep_vector(1.0, 3));
    }
}
parameters {
    // The sampling space: the independent parameters that are Monte Carlo sampled
    // A tiny positive numerical floor prevents an underflowed warmup proposal
    // from collapsing the two F-dependent bounds on lamb to the same value.
    real<lower=1e-6> F_2xCO2;
    // These F-dependent bounds preserve 0.05 <= S <= 20 under
    // S = -F_2xCO2/lamb and enforce the physical lamb < 0 domain.
    real<lower=-F_2xCO2 / 0.05, upper=-F_2xCO2 / 20> lamb;

    // historical nuisance parameters
    real F_anthro_aerosol_hist;
    real F_other_hist;
    real T_hist;
    real N_hist_scoreT;
    real dlambda;

    // LGM nuisance
    real T_LGM;
    real N_LGM_scoreT;
    real F_other_LGM;
    real f_CO2_LGM;
    real dlambda_LGM;
    // Pliocene nuisance
    real T_plio;
    real N_plio_scoreT;
    real <lower=0> CO2_plio;
    real F_plio_nonGHG;
    real dlambda_plio;
    real fCH4;
    // real fESS;

    // recent-Trend nuisance parameters
    real F_anthro_aerosol_trend;
    real F_other_trend;
    real T_trend;
    real N_trend_scoreT;
    real dlambda_trend;
}
transformed parameters{

    // These are dependent parameters that are a function of the independent parameters
    real S;        // equilibrium climate sensitivity
    real F_CO2_hist;
    real F_hist;
    real N_hist;
    real T_hist_scoreT;
    real F_CO2_trend;
    real F_trend;
    real N_trend;
    real T_trend_scoreT;
    real F_CO2_LGM;
    real N_LGM;
    real T_LGM_scoreT;
    real f_CO2_plio;
    real F_plio_CO2;
    real N_plio;
    real T_plio_scoreT;

    // parameter formulas
    
    // ECS derived from the directly sampled feedback parameter.
    S = -F_2xCO2 / lamb;

    // Historical CO2 forcing shares the same radiative-efficiency uncertainty
    // as F_2xCO2. At F_2xCO2 = erf_2x its central value is mu_F_CO2_hist.
    F_CO2_hist = mu_F_CO2_hist * F_2xCO2 / erf_2x;
    F_hist = F_CO2_hist + F_anthro_aerosol_hist + F_other_hist;

    // coupling equations
    N_hist  = F_hist + T_hist * (lamb - dlambda);
    if (abs(lamb - dlambda) > score_T_min_abs_feedback)
        T_hist_scoreT = (N_hist_scoreT - F_hist) / (lamb - dlambda);
    else
        T_hist_scoreT = 0;

    // The Trend CO2 forcing shares the same uncertainty as F_2xCO2, which
    // automatically supplies its covariance with historical CO2 forcing.
    // Aerosol and residual-other forcing are separate nuisance pairs.
    F_CO2_trend = mu_F_CO2_trend * F_2xCO2 / erf_2x;
    F_trend = F_CO2_trend + F_anthro_aerosol_trend + F_other_trend;

    // 2006--2024 trend energy budget.
    N_trend = F_trend + T_trend * (lamb - dlambda_trend);
    if (abs(lamb - dlambda_trend) > score_T_min_abs_feedback)
        T_trend_scoreT = (N_trend_scoreT - F_trend)
                         / (lamb - dlambda_trend);
    else
        T_trend_scoreT = 0;

    F_CO2_LGM = f_CO2_LGM * F_2xCO2;
    N_LGM = F_other_LGM + F_CO2_LGM
            + T_LGM * (lamb - dlambda_LGM);
    if (abs(lamb - dlambda_LGM) > score_T_min_abs_feedback)
        T_LGM_scoreT = (N_LGM_scoreT - F_other_LGM - F_CO2_LGM)
                       / (lamb - dlambda_LGM);
    else
        T_LGM_scoreT = 0;

    f_CO2_plio = (meinshausen_co2_sarf(CO2_plio)
                  - meinshausen_co2_sarf(CO2_PI))
                 / (meinshausen_co2_sarf(2 * CO2_PI)
                    - meinshausen_co2_sarf(CO2_PI));
    F_plio_CO2 = f_CO2_plio * F_2xCO2;
    N_plio = F_plio_CO2*(1+fCH4) + F_plio_nonGHG
             + T_plio*(lamb - dlambda_plio);
    if (abs(lamb - dlambda_plio) > score_T_min_abs_feedback)
        T_plio_scoreT = (N_plio_scoreT - F_plio_CO2 * (1 + fCH4)
                         - F_plio_nonGHG)
                        / (lamb - dlambda_plio);
    else
        T_plio_scoreT = 0;
}
model {
    // Shared forcing and optional Process likelihood. The correlated branch
    // preserves both Gaussian marginals and changes only their dependence.
    if (include_process == 1) {
        if (use_lambda_F2x_correlation == 1) {
            vector[2] lambda_F2x_pair;
            vector[2] mu_lambda_F2x_pair;

            lambda_F2x_pair[1] = F_2xCO2;
            lambda_F2x_pair[2] = lamb;
            mu_lambda_F2x_pair[1] = erf_2x;
            mu_lambda_F2x_pair[2] = mu_lambda;
            lambda_F2x_pair ~ multi_normal_cholesky(
                mu_lambda_F2x_pair, L_lambda_F2x
            );
        } else {
            F_2xCO2 ~ normal(erf_2x, sig_F2xCO2);
            lamb ~ normal(mu_lambda, sig_lambda);
        }
    } else {
        // F_2xCO2 retains its assessment when Process evidence is omitted.
        F_2xCO2 ~ normal(erf_2x, sig_F2xCO2);
    }

    // Historical/Trend component-forcing priors. When both lines are enabled,
    // preserve the supplied covariance for each matching component.
    // Cross-component covariance remains negligible and is not included.
    if (include_historical == 1 && include_trend == 1) {
        vector[2] F_anthro_aerosol_hist_trend;
        vector[2] mu_F_anthro_aerosol_hist_trend;
        vector[2] F_other_hist_trend;
        vector[2] mu_F_other_hist_trend;

        F_anthro_aerosol_hist_trend[1] = F_anthro_aerosol_hist;
        F_anthro_aerosol_hist_trend[2] = F_anthro_aerosol_trend;
        mu_F_anthro_aerosol_hist_trend[1] = mu_F_anthro_aerosol_hist;
        mu_F_anthro_aerosol_hist_trend[2] = mu_F_anthro_aerosol_trend;
        F_anthro_aerosol_hist_trend ~ multi_normal_cholesky(
            mu_F_anthro_aerosol_hist_trend,
            L_F_anthro_aerosol_hist_trend
        );

        F_other_hist_trend[1] = F_other_hist;
        F_other_hist_trend[2] = F_other_trend;
        mu_F_other_hist_trend[1] = mu_F_other_hist;
        mu_F_other_hist_trend[2] = mu_F_other_trend;
        F_other_hist_trend ~ multi_normal_cholesky(
            mu_F_other_hist_trend, L_F_other_hist_trend
        );
    } else {
        F_anthro_aerosol_hist ~ normal(
            mu_F_anthro_aerosol_hist, sig_F_anthro_aerosol_hist
        );
        F_anthro_aerosol_trend ~ normal(
            mu_F_anthro_aerosol_trend, sig_F_anthro_aerosol_trend
        );
        F_other_hist ~ normal(mu_F_other_hist, sig_F_other_hist);
        F_other_trend ~ normal(mu_F_other_trend, sig_F_other_trend);
    }

    // Historical
    T_hist  ~ normal(mu_T_hist , sig_T_hist);
    N_hist_scoreT ~ normal(mu_N_hist, sig_N_hist);
    if (include_historical == 1) {
        if (use_score_T == 1) {
            if (abs(lamb - dlambda) > score_T_min_abs_feedback
                && value_is_finite(T_hist_scoreT))
                T_hist_scoreT ~ normal(mu_T_hist, sig_T_hist);
            else
                target += negative_infinity();
        } else {
            if (value_is_finite(N_hist))
                N_hist ~ normal(mu_N_hist, sig_N_hist);
            else
                target += negative_infinity();
        }
    }

    // Recent-Trend nuisance priors are always proper. When Trend is disabled,
    // they remain independent of all shared ECS parameters and therefore do
    // not alter their posterior; the corresponding output columns are inert.
    dlambda_trend ~ normal(mu_dlambda_trend, sig_dlambda_trend);
    if (include_trend == 1) {
        // Score the observed T/N pair in the selected orientation. The same
        // covariance matrix is used for both response-variable conventions.
        vector[2] trend_pair;
        vector[2] mu_trend_pair;

        mu_trend_pair[1] = mu_T_trend;
        mu_trend_pair[2] = mu_N_trend;

        if (use_score_T == 1) {
            // T_trend is inactive in this branch but retains a proper prior.
            T_trend ~ normal(mu_T_trend, sig_T_trend);
            trend_pair[1] = T_trend_scoreT;
            trend_pair[2] = N_trend_scoreT;
            if (abs(lamb - dlambda_trend) > score_T_min_abs_feedback
                && value_is_finite(T_trend_scoreT)
                && value_is_finite(N_trend_scoreT))
                trend_pair ~ multi_normal_cholesky(
                    mu_trend_pair, L_TN_trend
                );
            else
                target += negative_infinity();
        } else {
            // N_trend_scoreT is inactive in this branch but remains proper.
            N_trend_scoreT ~ normal(mu_N_trend, sig_N_trend);
            trend_pair[1] = T_trend;
            trend_pair[2] = N_trend;
            if (value_is_finite(T_trend) && value_is_finite(N_trend))
                trend_pair ~ multi_normal_cholesky(
                    mu_trend_pair, L_TN_trend
                );
            else
                target += negative_infinity();
        }
    } else {
        T_trend ~ normal(mu_T_trend, sig_T_trend);
        N_trend_scoreT ~ normal(mu_N_trend, sig_N_trend);
    }

    // LGM
    T_LGM       ~ normal(mu_T_LGM, sig_T_LGM);
    N_LGM_scoreT ~ normal(mu_N_LGM, sig_N_LGM);
    F_other_LGM ~ normal(mu_F_other_LGM, sig_F_other_LGM);
    f_CO2_LGM   ~ normal(mu_f_CO2_LGM, sig_f_CO2_LGM);
    if (include_lgm == 1) {
        if (use_score_T == 1) {
            if (abs(lamb - dlambda_LGM) > score_T_min_abs_feedback
                && value_is_finite(T_LGM_scoreT))
                T_LGM_scoreT ~ normal(mu_T_LGM, sig_T_LGM);
            else
                target += negative_infinity();
        } else {
            if (value_is_finite(N_LGM))
                N_LGM ~ normal(mu_N_LGM, sig_N_LGM);
            else
                target += negative_infinity();
        }
    }

    // Pliocene
    T_plio        ~ normal(mu_T_plio, sig_T_plio);
    N_plio_scoreT ~ normal(mu_N_plio, sig_N_plio);
    CO2_plio      ~ normal(mu_CO2_plio, sig_CO2_plio);
    fCH4          ~ normal(mu_fCH4, sig_fCH4);
    F_plio_nonGHG ~ normal(mu_F_plio_nonGHG, sig_F_plio_nonGHG);
    if (include_pliocene == 1) {
        if (use_score_T == 1) {
            if (abs(lamb - dlambda_plio) > score_T_min_abs_feedback
                && value_is_finite(T_plio_scoreT))
                T_plio_scoreT ~ normal(mu_T_plio, sig_T_plio);
            else
                target += negative_infinity();
        } else {
            if (value_is_finite(N_plio))
                N_plio ~ normal(mu_N_plio, sig_N_plio);
            else
                target += negative_infinity();
        }
    }
    // fESS       ~ normal(mu_fESS, sig_fESS);

    // Select exactly one joint prior for the three pattern effects.
    if (use_pattern_effect_copula == 1) {
        // The marginal-density terms preserve the requested skew-normal and
        // Gaussian priors; the density ratio adds the Gaussian copula.
        vector[3] z_dlambda;
        real p_hist;

        p_hist = fmin(
            1 - 1e-12,
            fmax(
                1e-12,
                skew_normal_cdf(
                    dlambda | loc_dlambda, scale_dlambda, shape_dlambda
                )
            )
        );

        z_dlambda[1] = inv_Phi(p_hist);
        z_dlambda[2] = (dlambda_LGM - mu_dlambda_LGM) / sig_dlambda_LGM;
        z_dlambda[3] = (dlambda_plio - mu_dlambda_plio) / sig_dlambda_plio;

        target += skew_normal_lpdf(
            dlambda | loc_dlambda, scale_dlambda, shape_dlambda
        );
        target += normal_lpdf(
            dlambda_LGM | mu_dlambda_LGM, sig_dlambda_LGM
        );
        target += normal_lpdf(
            dlambda_plio | mu_dlambda_plio, sig_dlambda_plio
        );
        target += multi_normal_cholesky_lpdf(
            z_dlambda | rep_vector(0.0, 3), L_dlambda_copula
        );
        target += -std_normal_lpdf(z_dlambda);
    } else {
        vector[2] dlambda_pair;
        vector[2] mu_dlambda_pair;
        matrix[2, 2] cov_dlambda_pair;

        dlambda ~ skew_normal(loc_dlambda, scale_dlambda, shape_dlambda);

        dlambda_pair[1] = dlambda_LGM;
        dlambda_pair[2] = dlambda_plio;
        mu_dlambda_pair[1] = mu_dlambda_LGM;
        mu_dlambda_pair[2] = mu_dlambda_plio;
        cov_dlambda_pair[1, 1] = square(sig_dlambda_LGM);
        cov_dlambda_pair[2, 2] = square(sig_dlambda_plio);
        cov_dlambda_pair[1, 2] = rho_dlambda_LGM_plio
                                 * sig_dlambda_LGM * sig_dlambda_plio;
        cov_dlambda_pair[2, 1] = cov_dlambda_pair[1, 2];

        dlambda_pair ~ multi_normal(mu_dlambda_pair, cov_dlambda_pair);
    }

    // Direct sampling gives the uniform-lambda measure by default. The other
    // choices add their density in the sampled lambda coordinates. This remains
    // selectable whether or not the Process likelihood is included.
    if (lambda_prior_type == 0) {
        target += log(F_2xCO2) - 2 * log(-lamb);
    } else if (lambda_prior_type == 2) {
        target += lognormal_lpdf(
            -lamb | log(lognormal_r_median), lognormal_r_log_sigma
        );
    }
}
