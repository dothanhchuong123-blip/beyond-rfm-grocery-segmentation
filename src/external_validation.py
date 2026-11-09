"""External validation of the frozen K-means (k=2) household clusters.

Labels are read from results/households_rfm.csv and results/households_product.csv
and are NOT refit. Demographics and coupon/campaign tables were never clustering
inputs, so they serve as outside evidence. All associations are observational.

Run from the repository root:  python src/external_validation.py
Outputs: results/demographic_profiles.csv, results/demographic_tests.csv,
         results/coupon_response.csv, figures/14_demographics.png,
         figures/15_coupon_response.png
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

ROOT = Path(__file__).resolve().parents[1]
RAW, FIG = ROOT / "data" / "raw", ROOT / "figures"
OUT = ROOT / "results"
import os
SNAP = Path(os.environ["LABELS_DIR"]) if os.environ.get("LABELS_DIR") else None  # optional: read frozen labels from another dir
CUTOFF = 500          # calibration DAY<=500; holdout DAY>500
SEED, NBOOT = 42, 2000
COLORS = {"Higher": "#1f4e79", "Lower": "#e08a1e"}

plt.rcParams.update({"figure.dpi": 140, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})

# ---- frozen labels: label the higher-activity cluster by mean calibration spend ----
def load(space):
    d = pd.read_csv((SNAP or OUT) / f"households_{space}.csv")
    hi = d.groupby("cluster").monetary.mean().idxmax()
    d["group"] = np.where(d.cluster == hi, "Higher", "Lower")
    return d[["household_key", "group", "monetary"]]

labels = {"RFM": load("rfm"), "Product": load("product")}

def wilson(k, n, z=1.959964):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    den = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
    return c - h, c + h

def cramers_v(table):
    chi2 = chi2_contingency(table, correction=False)[0]
    return np.sqrt(chi2 / (table.to_numpy().sum() * (min(table.shape) - 1)))

# ================= Part 1: demographics =================
demo = pd.read_csv(RAW / "hh_demographic.csv")
ORDER = {
    "AGE_DESC": ["19-24", "25-34", "35-44", "45-54", "55-64", "65+"],
    "INCOME_DESC": ["Under 15K", "15-24K", "25-34K", "35-49K", "50-74K", "75-99K",
                    "100-124K", "125-149K", "150-174K", "175-199K", "200-249K", "250K+"],
    "HOUSEHOLD_SIZE_DESC": ["1", "2", "3", "4", "5+"],
    "KID_CATEGORY_DESC": ["None/Unknown", "1", "2", "3+"],
    "HOMEOWNER_DESC": ["Homeowner", "Renter", "Probable Owner", "Probable Renter", "Unknown"],
}
# Merge sparse top income bands (<30 households each) into broader bands for testing.
INCOME_MAP = {"Under 15K": "<25K", "15-24K": "<25K", "25-34K": "25-49K", "35-49K": "25-49K",
              "50-74K": "50-74K", "75-99K": "75-99K", "100-124K": "100-149K",
              "125-149K": "100-149K", "150-174K": "150K+", "175-199K": "150K+",
              "200-249K": "150K+", "250K+": "150K+"}
INCOME_ORDER = ["<25K", "25-49K", "50-74K", "75-99K", "100-149K", "150K+"]

prof_rows, test_rows, cover_rows = [], [], []
for space, lab in labels.items():
    m = lab.merge(demo, on="household_key", how="left", indicator=True)
    m["has_demo"] = m["_merge"] == "both"
    # Coverage by cluster (selection-bias check)
    ct = pd.crosstab(m.group, m.has_demo)
    chi_p = chi2_contingency(ct, correction=False)[1]
    for g in ["Higher", "Lower"]:
        s = m[m.group == g]
        lo, hi = wilson(s.has_demo.sum(), len(s))
        cover_rows.append(dict(space=space, group=g, n=len(s), with_demographics=int(s.has_demo.sum()),
                               coverage=s.has_demo.mean(), ci_low=lo, ci_high=hi,
                               coverage_diff_chi2_p=chi_p))
    d = m[m.has_demo].copy()
    d["INCOME_DESC"] = d.INCOME_DESC.map(INCOME_MAP)
    order = dict(ORDER, INCOME_DESC=INCOME_ORDER)
    for var, cats in order.items():
        d[var] = pd.Categorical(d[var].astype(str), categories=cats)
        tab = pd.crosstab(d.group, d[var]).reindex(index=["Higher", "Lower"], columns=cats)
        for g in tab.index:
            for c in cats:
                prof_rows.append(dict(space=space, variable=var, group=g, category=c,
                                      n=int(tab.loc[g, c]), share=tab.loc[g, c] / tab.loc[g].sum()))
        t = tab.loc[:, tab.sum() > 0]
        chi2, p, dof, exp = chi2_contingency(t, correction=False)
        test_rows.append(dict(space=space, variable=var, n=int(t.to_numpy().sum()), dof=dof,
                              chi2=chi2, p_value=p, cramers_v=cramers_v(t),
                              min_expected=exp.min(),
                              cells_expected_lt5=int((exp < 5).sum())))
cover = pd.DataFrame(cover_rows)
profiles = pd.DataFrame(prof_rows)
tests = pd.DataFrame(test_rows)
# Holm correction within each space across the 5 demographic tests
tests["p_holm"] = np.nan
for sp, idx in tests.groupby("space").groups.items():
    p = tests.loc[idx, "p_value"].to_numpy()
    o = np.argsort(p)
    adj = np.minimum(1, np.maximum.accumulate((len(p) - np.arange(len(p))) * p[o]))
    out = np.empty_like(adj); out[o] = adj
    tests.loc[idx, "p_holm"] = out
cov_prof = cover.rename(columns={"with_demographics": "n_with"}).assign(variable="COVERAGE")
cov_prof = pd.concat([cov_prof.assign(category="has_demographics", n=cov_prof.n_with, share=cov_prof.coverage)[["space", "variable", "group", "category", "n", "share"]],
                      cov_prof.assign(category="no_demographics", n=cov_prof.n - cov_prof.n_with, share=1 - cov_prof.coverage)[["space", "variable", "group", "category", "n", "share"]]])
pd.concat([profiles, cov_prof], ignore_index=True).to_csv(OUT / "demographic_profiles.csv", index=False)
# Coverage rows are appended to the tests file as variable='COVERAGE' for compactness
cov_tests = (cover.groupby("space").agg(n=("n", "sum"), p_value=("coverage_diff_chi2_p", "first"))
             .reset_index().assign(variable="COVERAGE (has demographics)"))
pd.concat([tests, cov_tests], ignore_index=True).to_csv(OUT / "demographic_tests.csv", index=False)

# ================= Part 2: campaign / coupon response =================
camp = pd.read_csv(RAW / "campaign_table.csv")
desc = pd.read_csv(RAW / "campaign_desc.csv")
red = pd.read_csv(RAW / "coupon_redempt.csv")
# Out-of-sample campaigns: those that START after the calibration cutoff.
post = desc[desc.START_DAY > CUTOFF].CAMPAIGN
tgt = camp[camp.CAMPAIGN.isin(post)][["household_key", "CAMPAIGN"]].drop_duplicates()
# A targeted (household,campaign) pair "responds" if the household redeemed a coupon of
# that campaign on a DAY inside the campaign window (and necessarily DAY>500).
rr = red.merge(desc, on="CAMPAIGN")
rr = rr[(rr.DAY >= rr.START_DAY) & (rr.DAY <= rr.END_DAY) & (rr.DAY > CUTOFF)]
resp_pairs = rr[["household_key", "CAMPAIGN"]].drop_duplicates().assign(resp=1)
pairs = tgt.merge(resp_pairs, how="left", on=["household_key", "CAMPAIGN"]).fillna({"resp": 0})
hh_t = pairs.groupby("household_key").agg(n_campaigns=("CAMPAIGN", "size"), n_resp=("resp", "sum"))
hh_t["responded"] = hh_t.n_resp > 0
# Secondary: any holdout redemption regardless of targeting data (all households)
any_red = set(red[red.DAY > CUTOFF].household_key)

def two_prop_p(k1, n1, k2, n2):
    return chi2_contingency([[k1, n1 - k1], [k2, n2 - k2]], correction=False)[1]

rng = np.random.default_rng(SEED)
resp_rows, hh_tab = [], {}
for space, lab in labels.items():
    h = lab.set_index("household_key").join(hh_t)
    h["targeted"] = h.n_campaigns.notna()
    h["responded"] = h.responded.astype("boolean").fillna(False).astype(bool)
    h["any_holdout_redemption"] = h.index.isin(any_red)
    hh_tab[space] = h
    for g in ["Higher", "Lower"]:
        s = h[h.group == g]; st = s[s.targeted]
        metrics = [
            ("targeted_share", s.targeted.sum(), len(s), "all clustered households"),
            ("responder_rate_among_targeted", st.responded.sum(), len(st), "targeted households"),
            ("pair_response_rate", st.n_resp.sum(), st.n_campaigns.sum(), "targeted household-campaign pairs"),
            ("any_holdout_redemption_all", s.any_holdout_redemption.sum(), len(s), "all clustered households"),
        ]
        for name, k, n, den in metrics:
            lo, hi = wilson(k, n)
            resp_rows.append(dict(space=space, group=g, metric=name, denominator=den,
                                  successes=int(k), n=int(n), rate=k / n, ci_low=lo, ci_high=hi))
resp = pd.DataFrame(resp_rows)

# Difference tests: Higher minus Lower, and bootstrap CI (households resampled)
def gap(h, col, cond=None):
    x = h if cond is None else h[cond(h)]
    a, b = x[x.group == "Higher"][col].mean(), x[x.group == "Lower"][col].mean()
    return a - b

diff_rows = []
boot = {}
for space, h in hh_tab.items():
    hh_idx = np.arange(len(h))
    arr = {c: h[c].to_numpy(float) for c in ["targeted", "responded", "any_holdout_redemption"]}
    grp = (h.group == "Higher").to_numpy()
    def stats(ix):
        g, t, r, a = grp[ix], arr["targeted"][ix] > 0, arr["responded"][ix], arr["any_holdout_redemption"][ix]
        def d(v, mask):
            m1, m0 = mask & g, mask & ~g
            return v[m1].mean() - v[m0].mean() if m1.any() and m0.any() else np.nan
        allm = np.ones_like(g, bool)
        return np.array([d(t.astype(float), allm), d(r.astype(float), t), d(a, allm)])
    bs = np.array([stats(rng.integers(0, len(h), len(h))) for _ in range(NBOOT)])
    boot[space] = bs
    point = stats(hh_idx)
    names = ["targeted_share", "responder_rate_among_targeted", "any_holdout_redemption_all"]
    for i, nm in enumerate(names):
        a = resp[(resp.space == space) & (resp.metric == nm)].set_index("group")
        p = two_prop_p(a.loc["Higher", "successes"], a.loc["Higher", "n"],
                       a.loc["Lower", "successes"], a.loc["Lower", "n"])
        lo, hi = np.nanpercentile(bs[:, i], [2.5, 97.5])
        diff_rows.append(dict(space=space, group="Higher minus Lower", metric=nm,
                              denominator="difference", successes=np.nan, n=np.nan,
                              rate=point[i], ci_low=lo, ci_high=hi, p_value=p))
# Cross-space contrast: is the Higher-Lower gap in responder rate different for Product vs RFM?
# (paired bootstrap on the same resampled households; index 1 = responder gap among targeted)
ix_common = hh_tab["RFM"].index.intersection(hh_tab["Product"].index)
Hr, Hp = hh_tab["RFM"].loc[ix_common], hh_tab["Product"].loc[ix_common]
def gap_c(h, ix):
    s = h.iloc[ix]; s = s[s.targeted]
    return s[s.group == "Higher"].responded.mean() - s[s.group == "Lower"].responded.mean()
cont = []
for _ in range(NBOOT):
    ix = rng.integers(0, len(ix_common), len(ix_common))
    cont.append(gap_c(Hp, ix) - gap_c(Hr, ix))
pt = gap_c(Hp, np.arange(len(ix_common))) - gap_c(Hr, np.arange(len(ix_common)))
lo, hi = np.nanpercentile(cont, [2.5, 97.5])
diff_rows.append(dict(space="Product minus RFM", group="Higher-Lower gap", metric="responder_rate_among_targeted",
                      denominator="difference of gaps", rate=pt, ci_low=lo, ci_high=hi, p_value=np.nan))
resp = pd.concat([resp, pd.DataFrame(diff_rows)], ignore_index=True)
resp.to_csv(OUT / "coupon_response.csv", index=False)

# ================= Figures =================
# Figure 14: demographic distributions per cluster (Product labels top, RFM below)
vars_ = ["AGE_DESC", "INCOME_DESC", "HOUSEHOLD_SIZE_DESC", "KID_CATEGORY_DESC", "HOMEOWNER_DESC"]
titles = {"AGE_DESC": "Age", "INCOME_DESC": "Income (USD)", "HOUSEHOLD_SIZE_DESC": "Household size",
          "KID_CATEGORY_DESC": "Kids", "HOMEOWNER_DESC": "Homeowner status"}
fig, axes = plt.subplots(2, 5, figsize=(16, 6.2), gridspec_kw={"width_ratios": [6, 6, 5, 4, 5]})
for r, space in enumerate(["RFM", "Product"]):
    for c, v in enumerate(vars_):
        ax = axes[r, c]
        p = profiles[(profiles.space == space) & (profiles.variable == v)]
        cats = list(dict.fromkeys(p.category))
        x = np.arange(len(cats))
        for k, g in enumerate(["Higher", "Lower"]):
            q = p[p.group == g].set_index("category").reindex(cats)
            ax.bar(x + (k - .5) * .38, q.share * 100, .38, color=COLORS[g], label=f"{g}-activity")
        ax.set_xticks(x, cats, rotation=40, ha="right", fontsize=8)
        t = tests[(tests.space == space) & (tests.variable == v)].iloc[0]
        ptxt = "p<0.001" if t.p_value < 0.001 else f"p={t.p_value:.3f}"
        ax.set_title(f"{titles[v]}\nV={t.cramers_v:.2f}, {ptxt}", fontsize=9, weight="bold")
        if c == 0:
            ax.set_ylabel(f"{space} labels\n% of group")
        if r == 0 and c == 0:
            ax.legend(frameon=False, fontsize=8)
ns = cover.set_index(["space", "group"]).with_demographics
fig.suptitle("Household demographics by frozen cluster (not clustering inputs). Households with demographics: "
             f"RFM {ns['RFM','Higher']} higher / {ns['RFM','Lower']} lower; "
             f"Product {ns['Product','Higher']} higher / {ns['Product','Lower']} lower", weight="bold", fontsize=11)
fig.tight_layout()
fig.savefig(FIG / "14_demographics.png", bbox_inches="tight")
plt.close(fig)

# Figure 15: coupon response by group, RFM vs Product
panels = [("targeted_share", "Targeted by a post-day-500 campaign"),
          ("responder_rate_among_targeted", "Redeemed, among targeted households")]
fig, axs = plt.subplots(1, 2, figsize=(11, 4))
for ax, (met, ttl) in zip(axs, panels):
    for i, space in enumerate(["RFM", "Product"]):
        for k, g in enumerate(["Higher", "Lower"]):
            row = resp[(resp.space == space) & (resp.group == g) & (resp.metric == met)].iloc[0]
            xx = i + (k - .5) * .36
            ax.bar(xx, row.rate * 100, .34, color=COLORS[g], label=f"{g}-activity" if i == 0 else None)
            ax.errorbar(xx, row.rate * 100, yerr=[[ (row.rate - row.ci_low) * 100], [(row.ci_high - row.rate) * 100]],
                        color="black", capsize=3, lw=1)
            ax.text(xx, row.ci_high * 100 + 1, f"{row.rate*100:.1f}%", ha="center", fontsize=8)
    ax.set_xticks([0, 1], ["RFM labels", "Product labels"])
    ax.set_ylabel("%"); ax.set_ylim(0, 100); ax.set_title(ttl, weight="bold", fontsize=10)
axs[1].legend(frameon=False, loc="upper right")
fig.suptitle("Out-of-sample campaign targeting and coupon redemption (95% Wilson intervals)", weight="bold")
fig.tight_layout()
fig.savefig(FIG / "15_coupon_response.png", bbox_inches="tight")
plt.close(fig)

# ---- console summary ----
pd.set_option("display.width", 200)
print(cover.round(4).to_string(index=False))
print(tests.round(4).to_string(index=False))
print(resp.round(4).to_string(index=False))
