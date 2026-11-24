# Model Card & Reproducibility Report — `upliftlab`

## 1. System Overview

- **System**: `upliftlab` — Causal uplift modeling and multi-arm email targeting policy framework comparing `Response`, `SLearner`, `TLearner`, `XLearner`, `DRLearner`, and `TransformedOutcome` meta-learners on shared LightGBM base estimators.
- **Intended Use**: Incremental targeting under campaign budget constraints, decile uplift calibration, and unbiased off-policy evaluation via Horvitz-Thompson Inverse Propensity Weighting (IPW).

## 2. Dataset & Experimental Protocol

- **Dataset**: Hillstrom MineThatData E-Mail Analytics Challenge (64,000 customers randomly assigned to `No E-Mail`, `Mens E-Mail`, or `Womens E-Mail`; verified via SHA-256 `0e5893329d8b93cefecc571777672028290ab69865718020c78c7284f291aece`).
- **Split**: Arm-stratified 50/50 split (`31,999` train / `32,001` test, `seed=0`).
- **Features**: 11 strictly pre-treatment features (`recency`, `history`, `mens`, `womens`, `newbie`, one-hot `zip_code` and `channel`; `history_segment` dropped).
- **Model Selection**: 5-fold treatment-stratified CV Qini area on the training split only; test split is evaluated once.
- **Uncertainty**: 500-sample arm-stratified bootstrap 95% percentile intervals for Qini area, paired Qini differences, and IPW policy values.

## 3. Observed Reproducible Benchmarks

From [`results/summary.json`](../results/summary.json) on the 32,001-customer held-out test split:

| Pair / Outcome | Test ATE | CV-Selected | Selected Qini (95% CI) | Response Qini (95% CI) | Random Qini (95% CI) |
| :--- | :---: | :--- | :---: | :---: | :---: |
| `womens_vs_none/visit` | +0.04467 | `s_learner` | **0.00811** `[0.00550, 0.01090]` | 0.00435 `[0.00194, 0.00699]` | -0.00099 `[-0.00338, 0.00127]` |
| `mens_vs_none/visit` | +0.07517 | `s_learner` | 0.00098 `[-0.00164, 0.00346]` | 0.00319 `[0.00059, 0.00586]` | 0.00091 `[-0.00166, 0.00370]` |
| `mens_vs_none/conversion` | +0.00488 | `s_learner` | -0.00007 `[-0.00083, 0.00069]` | -0.00002 `[-0.00073, 0.00069]` | -0.00006 `[-0.00088, 0.00068]` |

- On `womens_vs_none/visit`, `s_learner` beats `response` by **+0.00376** Qini (`95% CI [0.00133, 0.00605]`) and captures **84.7%** of the full campaign lift while emailing only **50%** of customers (`37.81` vs `22.33` extra visits per 1,000 customers under random targeting).

## 4. Reproducibility Instructions

```bash
# 1. Run simulated-RCT, edge-case, and serving tests
pytest -v

# 2. Reproduce full Hillstrom benchmark (downloads & verifies CSV automatically)
upliftlab-run --data data/hillstrom.csv --out results

# 3. Build & run serving container
docker build -t upliftlab:latest .
docker run -p 8000:8000 upliftlab:latest
```

## 5. Limitations & Causal Assumptions

- **Randomization Requirement**: Unbiased Qini and IPW estimates rely on known, strictly positive treatment propensities from a randomized experiment; observational logs require propensity estimation and overlap diagnostics.
- **Rare-Outcome Variance**: On low-rate outcomes such as `conversion` (<1.1%), CATE variance is high and bootstrap intervals span zero — targeting models should only be deployed when paired intervals exclude zero.
- **SUTVA / Interference**: Assumes no cross-customer spillover between treatment arms during the 2-week measurement window.
