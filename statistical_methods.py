#!/usr/bin/env python3
"""Statistical utility methods for categorical feature evaluation and binary outcomes.

The script provides seven reusable methods:
1. conditional error rate given a feature value
2. risk ratio from two feature values
3. logistic regression for binary correct/incorrect outcomes
4. average treatment effect with propensity-adjusted estimation
5. McNemar's test for paired binary samples
6. Wilcoxon signed-rank test for paired samples
7. Fisher's exact test on 2x2 contingency tables

Dependencies: pandas, numpy, scipy, scikit-learn
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from scipy.stats import chi2, fisher_exact, wilcoxon

from sklearn.linear_model import LogisticRegression


def _as_binary_outcome(series: pd.Series, outcome_col: str = "correct") -> pd.Series:
    """Normalize outcomes to 0/1 where 1 means correct and 0 means incorrect."""
    y = pd.to_numeric(series, errors="coerce")
    if y.isna().any():
        raise ValueError(f"Outcome column '{outcome_col}' contains missing or non-numeric values.")
    # Accept booleans, ints, floats, strings like 'correct'/'incorrect'
    if y.dtype.kind in "biuf":
        y = y.astype(int)
        if set(y.unique()).issubset({0, 1}):
            return y
    # Strings as labels
    y2 = series.astype(str).str.strip().str.lower()
    mapping = {
        "1": 1,
        "correct": 1,
        "true": 1,
        "yes": 1,
        "success": 1,
        "0": 0,
        "incorrect": 0,
        "false": 0,
        "no": 0,
        "failure": 0,
    }
    y3 = y2.map(mapping)
    if y3.isna().any():
        raise ValueError(f"Outcome column '{outcome_col}' must be binary (0/1, True/False, or correct/incorrect labels).")
    return y3.astype(int)


def compute_conditional_error_rate(
    df: pd.DataFrame,
    feature_col: str,
    feature_value,
    outcome_col: str = "correct",
) -> float:
    """Return P(error | feature == feature_value) from a DataFrame.

    Parameters
    ----------
    df:
        DataFrame containing the categorical feature and binary correctness vector.
    feature_col:
        Categorical feature column to condition on.
    feature_value:
        Specific value of feature_col to condition on.
    outcome_col:
        Binary correctness column. Use 1 for correct and 0 for incorrect.

    Returns
    -------
    float
        Conditional error rate, i.e. the empirical fraction of rows with outcome == 0
        among rows where the feature equals the requested value.
    """
    if feature_col not in df.columns:
        raise KeyError(f"Column '{feature_col}' not found in DataFrame.")
    if outcome_col not in df.columns:
        raise KeyError(f"Column '{outcome_col}' not found in DataFrame.")

    sub = df[df[feature_col] == feature_value]
    if sub.empty:
        raise ValueError(f"No rows found for feature '{feature_col}' == {feature_value!r}.")

    y = _as_binary_outcome(sub[outcome_col], outcome_col)
    # Interpret error as outcome == 0 (incorrect)
    return float((y == 0).mean())


def risk_ratio(
    df: pd.DataFrame,
    feature_col: str,
    feature_value_a,
    feature_value_b,
    outcome_col: str = "correct",
) -> float:
    """Return the risk ratio of error rates for value_a versus value_b.

    Risk ratio is defined as err(feature=a) / err(feature=b).
    """
    ra = compute_conditional_error_rate(df, feature_col, feature_value_a, outcome_col)
    rb = compute_conditional_error_rate(df, feature_col, feature_value_b, outcome_col)
    if rb == 0:
        return np.inf
    return float(ra / rb)


def logistic_regression(
    df: pd.DataFrame,
    feature_cols: list[str],
    outcome_col: str = "correct",
    random_state: int = 0,
) -> dict:
    """Fit a binary logistic regression using numeric and encoded categorical columns.

    Returns a dictionary with the fitted LogisticRegression object, feature names,
    coefficients, intercept, odds ratios, predicted probabilities, and model score.
    """
    if outcome_col not in df.columns:
        raise KeyError(f"Column '{outcome_col}' not found in DataFrame.")

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing feature columns: {missing}")

    y = _as_binary_outcome(df[outcome_col], outcome_col)
    X = df[feature_cols].copy()

    # Build design matrix allowing categorical variables one-hot encoded.
    X_design = []
    for col in feature_cols:
        s = X[col]
        if pd.api.types.is_numeric_dtype(s):
            X_design.append(pd.DataFrame({col: s}))
        else:
            dummies = pd.get_dummies(s, prefix=col, drop_first=False)
            X_design.append(dummies)

    X_design_df = pd.concat(X_design, axis=1)
    X_design_df = X_design_df.replace({True: 1, False: 0})

    # Drop rows with missing all rows; very light cleaning.
    keep = X_design_df.dropna(axis=1, how="all")
    keep = keep.dropna(axis=0, how="any")
    y_clean = y.loc[keep.index]

    model = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=random_state)
    model.fit(keep, y_clean)

    coef = pd.Series(model.coef_[0], index=keep.columns)
    odds_ratio = pd.Series(np.exp(model.coef_[0]), index=keep.columns)

    return {
        "model": model,
        "feature_names": list(keep.columns),
        "coefficients": coef,
        "intercept": float(model.intercept_[0]),
        "odds_ratios": odds_ratio,
        "score": float(model.score(keep, y_clean)),
        "predicted_probabilities": model.predict_proba(keep)[:, 1],
    }


def average_treatment_effect(
    df: pd.DataFrame,
    treatment_col: str,
    outcome_col: str = "correct",
    covariate_cols: list[str] | None = None,
    treated_value=1,
    control_value=0,
    threshold: float | None = None,
) -> dict:
    """Estimate an ATE for binary outcome using propensity-score IPW adjustment.

    This supports a binary treatment representation (requires treated/control values,
    or a numeric threshold to split a continuous treatment into treated and control).
    If covariates are supplied, they are used to estimate a propensity model for
    treatment assignment and therefore adjust the effect estimate.

    Returns a dictionary with estimated ATE, treated/control weighted means, propensity
    model, and treatment rows used. All categorical covariates are one-hot encoded.
    """
    if treatment_col not in df.columns:
        raise KeyError(f"Treatment column '{treatment_col}' not found.")
    if outcome_col not in df.columns:
        raise KeyError(f"Outcome column '{outcome_col}' not found.")

    outcome = _as_binary_outcome(df[outcome_col], outcome_col)

    # Construct binary treatment assignment.
    if threshold is not None and pd.api.types.is_numeric_dtype(df[treatment_col]):
        treat = df[treatment_col] >= threshold
    elif df[treatment_col].nunique() == 2:
        # Automatically map the two unique values into treated/control if values are exactly 0/1.
        map_values = list(df[treatment_col].dropna().unique())
        if treated_value in map_values and control_value in map_values:
            treat = df[treatment_col].eq(treated_value)
        else:
            # If not given, use the first two sorted unique categories as control/treated
            treat = df[treatment_col].eq(map_values[0])
    else:
        # If the treatment column is categorical with many categories, pick the requested value.
        if treated_value is None:
            raise ValueError("For a categorical treatment, provide treated_value and control_value.")
        treat = df[treatment_col].eq(treated_value)

    y = outcome.astype(int)
    t = treat.astype(int)

    if covariate_cols is None:
        covariate_cols = [c for c in df.columns if c not in {treatment_col, outcome_col}]

    cov = [c for c in covariate_cols if c in df.columns]
    if len(cov) == 0:
        # No covariates: no adjustment, simple difference in weighted means.
        cov = []

    # Build design matrix for propensity model (categorical covariates one-hot encoded).
    if cov:
        prop_X = []
        for c in cov:
            series = df[c]
            if pd.api.types.is_numeric_dtype(series):
                prop_X.append(pd.DataFrame({c: series}))
            else:
                prop_X.append(pd.get_dummies(series, prefix=c, drop_first=False))
        prop_design = pd.concat(prop_X, axis=1)
        prop_design = prop_design.replace({True: 1, False: 0})
        # Drop columns with no variation.
        prop_design = prop_design.dropna(axis=1, how="all")
        prop_model = LogisticRegression(max_iter=1000, solver="lbfgs")
        prop_model.fit(prop_design, t)
        ps = prop_model.predict_proba(prop_design)[:, 1]
    else:
        ps = np.full(len(df), float(t.mean()))

    # Weighted average potential means by inverse probability weighting.
    ps = np.clip(ps, 1e-6, 1 - 1e-6)
    w_treat = t / ps
    w_control = (1 - t) / (1 - ps)

    n_t = t.sum()
    n_c = (1 - t).sum()
    if n_t == 0 or n_c == 0:
        raise ValueError("Treatment column needs at least one row in each treatment group.")

    mu1 = np.sum(y * t / ps) / np.sum(t / ps)
    mu0 = np.sum(y * (1 - t) / (1 - ps)) / np.sum((1 - t) / (1 - ps))
    ate = float(mu1 - mu0)

    return {
        "ATE": ate,
        "mean_treated_outcome": float(mu1),
        "mean_control_outcome": float(mu0),
        "propensity_model": prop_model if cov else None,
        "propensity_scores": ps,
        "treated_count": int(n_t),
        "control_count": int(n_c),
    }


def mcnemar_test(
    df: pd.DataFrame,
    label_a_col: str,
    label_b_col: str,
) -> dict:
    """Return McNemar's paired-sample test for two binary classifiers/models.

    The disagreement table is built from the paired predictions of label_a_col and
    label_b_col, where 1 denotes correct and 0 denotes incorrect.
    """
    if label_a_col not in df.columns or label_b_col not in df.columns:
        raise KeyError("Both label columns must be present in the DataFrame.")

    a = _as_binary_outcome(df[label_a_col], label_a_col)
    b = _as_binary_outcome(df[label_b_col], label_b_col)

    # Create paired table: [a,b,c,d] = [both correct, a correct/b incorrect,
    # a incorrect/b correct, both incorrect]
    n11 = int(((a == 1) & (b == 1)).sum())
    n10 = int(((a == 1) & (b == 0)).sum())
    n01 = int(((a == 0) & (b == 1)).sum())
    n00 = int(((a == 0) & (b == 0)).sum())

    # McNemar statistic using the discordant pairs only.
    total_discordant = n10 + n01
    if total_discordant == 0:
        return {"statistic": 0.0, "p_value": 1.0, "table": [[n11, n10], [n01, n00]], "decision": "No discordant pairs"}

    stat = ((abs(n10 - n01) - 1) ** 2) / (n10 + n01)
    # Conservative without Yates correction; chi-square 1 df is used by convention.
    pvalue = float(chi2.sf(stat, df=1))

    return {
        "statistic": float(stat),
        "p_value": pvalue,
        "table": [[n11, n10], [n01, n00]],
        "decision": "reject H0" if pvalue < 0.05 else "fail to reject H0",
    }


def wilcoxon_signed_rank_test(
    df: pd.DataFrame,
    metric_a_col: str,
    metric_b_col: str,
    alternative: str = "two-sided",
) -> dict:
    """Return the Wilcoxon signed-rank test for paired samples.

    Parameters
    ----------
    df:
        DataFrame with a metric measured twice on each sample.
    metric_a_col:
        First paired metric column.
    metric_b_col:
        Second paired metric column.
    alternative:
        'two-sided', 'greater', or 'less'.
    """
    if metric_a_col not in df.columns or metric_b_col not in df.columns:
        raise KeyError("Metric columns must exist in the DataFrame.")

    a = pd.to_numeric(df[metric_a_col], errors="coerce")
    b = pd.to_numeric(df[metric_b_col], errors="coerce")
    if a.isna().any() or b.isna().any():
        raise ValueError("Metric columns contain non-numeric or missing values.")

    # Drop pairs where difference is exactly 0.
    diff = (a - b).dropna()
    res = wilcoxon(diff, alternative=alternative)

    return {
        "statistic": float(res.statistic),
        "p_value": float(res.pvalue),
        "alternative": alternative,
        "n_pairs": int(len(df)),
    }


def fisher_exact_test(
    df: pd.DataFrame,
    row_col: str,
    col_col: str,
    alternative: str = "two-sided",
) -> dict:
    """Return Fisher's exact test from a 2x2 contingency table built from df.

    Parameters
    ----------
    df:
        DataFrame with two categorical columns.
    row_col:
        First categorical column used to form rows.
    col_col:
        Second categorical column used to form columns.
    alternative:
        'two-sided', 'greater', or 'less'.
    """
    if row_col not in df.columns or col_col not in df.columns:
        raise KeyError("Input categorical columns must be present in the DataFrame.")

    table = pd.crosstab(df[row_col], df[col_col])
    if table.shape != (2, 2):
        # Collapse to the first two rows/columns if needed.
        if table.shape[0] == 2 and table.shape[1] == 2:
            pass
        else:
            raise ValueError("Fisher's exact test requires a 2x2 contingency table.")

    # Convert to numpy array for SciPy
    arr = table.to_numpy(dtype=int)
    odds_ratio, p_value = fisher_exact(arr, alternative=alternative)
    return {
        "odds_ratio": float(odds_ratio),
        "p_value": float(p_value),
        "table": table,
        "alternative": alternative,
    }


if __name__ == "__main__":
    # Tiny smoke test and usage example.
    df = pd.DataFrame(
        {
            "feature": ["A", "A", "B", "B", "A", "B"],
            "correct": [1, 0, 1, 0, 1, 0],
            "score_a": [1, 2, 2, 1, 3, 2],
            "score_b": [2, 1, 1, 2, 4, 3],
            "treat": [0, 0, 1, 1, 0, 1],
        }
    )

    print("Conditional error rate:", compute_conditional_error_rate(df, "feature", "A"))
    print("Risk ratio:", risk_ratio(df, "feature", "A", "B"))
    print("Logistic regression:", logistic_regression(df, ["feature", "score_a"], "correct").get("score"))
    print("ATE:", average_treatment_effect(df, "treat", "correct", covariate_cols=["score_a"])['ATE'])
    print("McNemar:", mcnemar_test(df.assign(pred_a=df.correct, pred_b=(df.correct == 1).astype(int)), "pred_a", "pred_b"))
    print("Wilcoxon:", wilcoxon_signed_rank_test(df, "score_a", "score_b"))
    print("Fisher:", fisher_exact_test(df.assign(group=np.where(df.feature == "A", "A", "B")), "group", "feature"))
