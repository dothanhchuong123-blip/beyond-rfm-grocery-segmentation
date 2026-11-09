"""Business-persona lens: RFM K-means with k=4.

This is a complementary, interpretive view.  It does NOT replace the k=2
selection of the main study (src/analysis.py), which is the statistically
preferred solution.  Run from the repo root:  .venv/bin/python src/personas.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "results"  # frozen household table written by src/analysis.py
OUT, FIG = ROOT / "results", ROOT / "figures"
RFM = ["recency", "frequency", "monetary"]
SEED, K = 42, 4

hh = pd.read_csv(SRC / "households_rfm.csv")

# Same transformation as src/analysis.py: log1p, then standardize.
X = StandardScaler().fit_transform(np.log1p(hh[RFM]))


def fit(data, k, seed=SEED):
    return KMeans(n_clusters=k, n_init=20, random_state=seed).fit_predict(data)


raw4, raw2 = fit(X, K), fit(X, 2)
sil4, sil2 = silhouette_score(X, raw4), silhouette_score(X, raw2)

# Deterministic ordering: by mean monetary value, descending.
hh["raw"] = raw4
order = hh.groupby("raw").monetary.mean().sort_values(ascending=False).index
rank = {c: i for i, c in enumerate(order)}
hh["rank"] = hh.raw.map(rank)

# Persona names follow the profile.  Rank 0 = highest spend; rank 3 = lowest
# spend.  The checks below make the names fail loudly if the profile changes.
names = {0: "Loyal high-value", 1: "Steady regulars",
         2: "Occasional shoppers", 3: "Lapsed / at-risk"}
rec = hh.groupby("rank").recency.mean()
frq = hh.groupby("rank").frequency.mean()
assert rec.idxmax() == 3, "lowest-spend persona is no longer the most lapsed"
assert frq.is_monotonic_decreasing and rec.idxmin() == 0, "profile ordering changed"
hh["persona"] = hh["rank"].map(names)

# Profiles: sizes, mean/median R/F/M, holdout (day 501-711) outcomes.
rows = []
for r, name in names.items():
    g = hh[hh["rank"] == r]
    row = {"persona": name, "households": len(g), "share_pct": 100 * len(g) / len(hh)}
    for c in RFM:
        row[f"{c}_mean"], row[f"{c}_median"] = g[c].mean(), g[c].median()
    row.update(future_spend_mean=g.future_spend.mean(),
               future_spend_median=g.future_spend.median(),
               future_baskets_mean=g.future_baskets.mean(),
               future_active_rate=g.future_active.mean())
    rows.append(row)
prof = pd.DataFrame(rows)
tot = hh.future_spend.sum()
prof["holdout_spend_share_pct"] = [
    100 * hh.loc[hh["rank"] == r, "future_spend"].sum() / tot for r in names]
prof["silhouette_k4"], prof["silhouette_k2"] = sil4, sil2

# Stability: ARI of each 80% subsample fit against the full-sample labels,
# evaluated on the subsample rows (20 draws, seed 42).
rng = np.random.default_rng(SEED)
ari = []
for _ in range(20):
    idx = rng.choice(len(X), int(0.8 * len(X)), replace=False)
    ari.append(adjusted_rand_score(raw4[idx], fit(X[idx], K)))
prof["ari_mean"], prof["ari_min"], prof["ari_max"] = np.mean(ari), np.min(ari), np.max(ari)
prof.round(4).to_csv(OUT / "rfm_personas_profiles.csv", index=False)
hh[["household_key", "persona"]].to_csv(OUT / "rfm_personas_assignments.csv", index=False)

# Figure, in the style of the existing figures (teal accent, no top/right spines).
plt.rcParams.update({"figure.dpi": 140, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})
cols = ["#0f766e", "#4f6fa5", "#c87733", "#9a3b3b"]
fig, ax = plt.subplots(1, 3, figsize=(13, 4.2))
for r, name in names.items():
    g = hh[hh["rank"] == r]
    ax[0].scatter(np.log1p(g.frequency), np.log1p(g.monetary), s=7, alpha=0.45,
                  color=cols[r], label=f"{name} (n={len(g):,})")
ax[0].set(xlabel="log(1 + frequency)", ylabel="log(1 + monetary)",
          title="Personas in calibration space")
ax[0].legend(fontsize=7.5, frameon=False, loc="upper left", markerscale=2)
labels = [n.replace(" ", "\n", 1) for n in names.values()]
zs = pd.DataFrame(X, columns=RFM).groupby(hh["rank"].values).mean()
w = 0.26
for i, (c, lab) in enumerate(zip(RFM, ["Recency", "Frequency", "Monetary"])):
    ax[1].bar(np.arange(4) + (i - 1) * w, zs[c], w, label=lab,
              color=["#c87733", "#4f6fa5", "#0f766e"][i])
ax[1].axhline(0, color="black", linewidth=0.8)
ax[1].set_xticks(range(4), labels, fontsize=8)
ax[1].set(ylabel="Mean standardized value", title="RFM profile")
ax[1].legend(fontsize=8, frameon=False)
ax[2].bar(range(4), prof.future_spend_mean, color=cols)
for i, (v, a) in enumerate(zip(prof.future_spend_mean, prof.future_active_rate)):
    ax[2].text(i, v, f"${v:,.0f}\n{100*a:.0f}% active", ha="center", va="bottom", fontsize=8)
ax[2].set_xticks(range(4), labels, fontsize=8)
ax[2].set(ylabel="Mean holdout spend (day 501-711)", title="Holdout outcome")
ax[2].set_ylim(0, prof.future_spend_mean.max() * 1.2)
fig.tight_layout()
fig.savefig(FIG / "13_rfm_personas.png", bbox_inches="tight")

# LaTeX subsection: every number below is read from `prof`/`hh`.
f0 = lambda v: f"{v:,.0f}"
tab = "\n".join(
    f"{p.persona} & {f0(p.households)} & {p.share_pct:.1f} & {p.recency_median:.0f} & "
    f"{p.frequency_median:.0f} & {f0(p.monetary_median)} & {f0(p.future_spend_mean)} & "
    f"{100*p.future_active_rate:.1f} & {p.holdout_spend_share_pct:.1f}\\\\"
    for p in prof.itertuples())
P = {n: prof[prof.persona == n].iloc[0] for n in names.values()}
H, S, O, L = (P[n] for n in names.values())
tex = rf"""\subsection{{Business personas: a four-group lens on RFM}}
The statistical selection above chooses $k=2$ for every feature space, which separates high from low activity but gives marketing little to act on. As a complementary \emph{{interpretive}} view we refit K-means on the same standardized log RFM coordinates with $k={K}$ (20 initializations, seed {SEED}). This does not replace the $k=2$ result; it re-slices the same activity gradient into groups that are easier to name and target. Personas are ordered by mean monetary value.

\begin{{table}}[htbp]\centering\small
\caption{{RFM personas ($k={K}$). Recency, frequency and monetary value are calibration medians (days $\leq500$); the last three columns use the later period (days 501--711), which was not used for clustering.}}\label{{tab:personas}}
\setlength{{\tabcolsep}}{{4pt}}\begin{{tabular}}{{lrrrrrrrr}}\toprule
Persona & $n$ & \% & Rec. & Freq. & Mon. (\$) & Later spend (\$) & Active (\%) & Share (\%)\\\midrule
{tab}\bottomrule
\end{{tabular}}\end{{table}}
\begin{{figure}}[htbp]\centering
\includegraphics[width=\textwidth]{{../figures/13_rfm_personas.png}}
\caption{{Four RFM personas. Left: households in log frequency--monetary space. Middle: mean standardized R, F and M per persona. Right: mean holdout spend with the share of households that purchased again.}}\label{{fig:personas}}
\end{{figure}}
\paragraph{{Interpretation.}} \emph{{Loyal high-value}} households ({f0(H.households)}, {H.share_pct:.1f}\%) shop about {H.frequency_median:.0f} times with median spend \${f0(H.monetary_median)}, were last seen a median {H.recency_median:.0f} days before day 500, and account for {H.holdout_spend_share_pct:.1f}\% of later spend. \emph{{Steady regulars}} ({f0(S.households)}, {S.share_pct:.1f}\%) are recent, mid-frequency shoppers contributing {S.holdout_spend_share_pct:.1f}\% of later spend. \emph{{Occasional shoppers}} ({f0(O.households)}, {O.share_pct:.1f}\%) buy roughly {O.frequency_median:.0f} times with median recency {O.recency_median:.0f} days. \emph{{Lapsed / at-risk}} households ({f0(L.households)}, {L.share_pct:.1f}\%) have the longest gap since their last purchase (median {L.recency_median:.0f} days), and only {100*L.future_active_rate:.1f}\% purchased in the later period against at least {100*min(H.future_active_rate, S.future_active_rate, O.future_active_rate):.1f}\% elsewhere.

\paragraph{{Implications.}} \emph{{Loyal high-value}}: protect rather than discount; prioritize service quality, early access and loyalty rewards, since losing one costs the most. \emph{{Steady regulars}}: the main growth pool; use cross-sell and basket-building offers to move them toward the top group. \emph{{Occasional shoppers}}: raise visit frequency with reminders and targeted coupons for categories they already buy. \emph{{Lapsed / at-risk}}: low-cost win-back messages with a firm budget cap, as expected payoff is small (mean later spend \${f0(L.future_spend_mean)}).

\paragraph{{Caveat.}} Four groups are weaker than two statistically: over all 2,498 households the silhouette is {sil4:.3f} for $k={K}$ against {sil2:.3f} for $k=2$ (Table~\ref{{tab:selected}} reports 0.456 for $k=2$ on the fixed 1,500-household evaluation sample). Subsample stability is high on average (mean ARI {np.mean(ari):.2f}) but uneven (range {np.min(ari):.2f}--{np.max(ari):.2f} over 20 independent 80\% samples), so individual boundary households can move between neighbouring personas when the gradient is cut at slightly different places. The personas are a descriptive lens for communication and targeting, not evidence of four natural household types, and the labels are our interpretation of the numbers.
"""
(ROOT / "report" / "personas.tex").write_text(tex)
print(prof.round(2).T.to_string())
