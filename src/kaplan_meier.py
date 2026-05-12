"""Kaplan-Meier survival curves by credit score band."""

import matplotlib.pyplot as plt

from lifelines import KaplanMeierFitter
from lifelines.utils import median_survival_times


def credit_band(score):
    if score < 580:
        return "score_<580"
    elif score < 670:
        return "score_580-669"
    elif score < 740:
        return "score_670-739"
    else:
        return "score_740plus"


def fit_km(data, outcome_col="time_end", event_col="event_default"):
    """Fit KM curves per credit score band and plot."""
    data = data.copy()
    data["band"] = data["credit_score"].apply(credit_band)
    bands = ["score_<580", "score_580-669", "score_670-739", "score_740plus"]
    labels = ["<580", "580-669", "670-739", "740+"]

    fig, ax = plt.subplots(figsize=(9, 5))

    results = {}
    for band, label in zip(bands, labels):
        df_band = data[data["band"] == band]
        kmf = KaplanMeierFitter()
        kmf.fit(df_band[outcome_col], df_band[event_col], label=label)

        median_t = median_survival_times(kmf.survival_function_)
        n = len(df_band)
        events = df_band[event_col].sum()
        results[band] = {
            "n": n,
            "events": int(events),
            "median_survival_months": round(float(median_t), 2),
            "survival_function": kmf.survival_function_.to_dict(),
        }

        kmf.plot_survival_function(ax=ax, ci_show=True)

    # 12- and 24-month survival probabilities
    for band, label in zip(bands, labels):
        df_band = data[data["band"] == band]
        kmf = KaplanMeierFitter()
        kmf.fit(df_band[outcome_col], df_band[event_col])
        s12 = float(kmf.survival_function_at_times(12).iloc[0])
        s24 = float(kmf.survival_function_at_times(24).iloc[0])
        results[band]["survival_12m"] = round(s12, 4)
        results[band]["survival_24m"] = round(s24, 4)

    ax.set_title("Kaplan-Meier Survival Curves by Credit Score Band")
    ax.set_xlabel("Months")
    ax.set_ylabel("Survival Probability")
    ax.set_ylim(0, 1.05)
    ax.legend(title="Credit Score")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("reports/km_survival_curves.png", dpi=150)
    plt.close()
    print("Saved reports/km_survival_curves.png")

    return results