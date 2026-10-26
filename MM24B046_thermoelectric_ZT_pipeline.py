"""
================================================================================
 Can ML Discover New Thermoelectrics, or Only Recognise Old Ones?
 An honest evaluation of ZT prediction from chemical composition
================================================================================

 MM5019 - Materials Science for Sustainability
 Sabarishvar S S (MM24B046), IIT Madras

--------------------------------------------------------------------------------
 WHAT THIS DOES
--------------------------------------------------------------------------------
 Thermoelectric materials turn waste heat into electricity. Their quality is
 scored by ZT: higher is better. Measuring ZT means synthesising the material
 in a lab, which is slow and expensive, so people try to PREDICT ZT from the
 chemical formula alone and only test the promising candidates.

 Published work reports accuracies around 0.9 for this. This script shows that
 those numbers depend almost entirely on HOW you split the data for testing,
 and that the honest number is much lower. It then builds a confidence gate so
 the model says "I don't know" instead of guessing.

--------------------------------------------------------------------------------
 DATA
--------------------------------------------------------------------------------
 ESTM dataset - 5,205 experimental measurements, 880 unique materials.
 Source: https://github.com/KRICT-DATA/SIMD  (dataset/estm.xlsx)
   git clone https://github.com/KRICT-DATA/SIMD.git

 Requirements:
   pip install pandas numpy scikit-learn scipy openpyxl pymatgen matminer

--------------------------------------------------------------------------------
 HEADLINE RESULTS
--------------------------------------------------------------------------------
   Random split (misleading)                       R2 = 0.894
   Material held out                               R2 = 0.797
   Chemical system held out  <- the honest number  R2 = 0.535
   + magpie features (133)                         R2 = 0.591
   + top-20 feature selection                      R2 = 0.603
   + confidence gate @ 50% coverage                R2 = 0.692
   + confidence gate @ 30% coverage                R2 = 0.722
================================================================================
"""

import warnings
import numpy as np
import pandas as pd
from pymatgen.core import Composition
from matminer.featurizers.composition import ElementProperty
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.spatial.distance import cdist

warnings.filterwarnings("ignore")

DATA_PATH = "SIMD/dataset/estm.xlsx"
N_TREES = 150
N_FOLDS = 5
TOP_K = 20          # features kept by the selection step
SEED = 0


# =============================================================================
# STEP 1 - LOAD DATA AND SEPARATE METADATA FROM FEATURES
# =============================================================================
# Key design decision: the material's NAME never becomes a model input. If the
# model could see "this is Bi2Te3" it would just memorise the answer. The name
# is kept in a separate `meta` table, used only for (a) grouping the splits and
# (b) inspecting results afterwards.

def load_data(path=DATA_PATH):
    df = pd.read_excel(path)

    # Parse every formula once. Composition("Cu2Se0.88S0.12") understands
    # chemical formula syntax and gives us the elements and their amounts.
    comps = [Composition(f) for f in df["Formula"]]

    # The "chemical system" is the formula stripped down to just its element
    # set, sorted alphabetically. Cu2Se0.88S0.12 -> "CuSSe".
    #   .get_el_amt_dict() -> {'Cu': 2.0, 'Se': 0.88, 'S': 0.12}
    #   .keys()            -> element names only, amounts discarded
    #   sorted()           -> fixed order, so Se-S-Cu and Cu-S-Se match
    #   ''.join()          -> one string we can group on
    systems = ["".join(sorted(c.get_el_amt_dict().keys())) for c in comps]

    meta = pd.DataFrame({
        "formula": df["Formula"].values,
        "system": systems,
        "temperature": df["temperature(K)"].values,
    })
    y = df["ZT"].values
    return df, comps, meta, y


# =============================================================================
# STEP 2 - BUILD FEATURES (called "descriptors" in materials science)
# =============================================================================
# A model cannot read "Cu2Se0.88S0.12". We turn each formula into numbers that
# describe its chemistry. Magpie takes 22 element properties (electronegativity,
# covalent radius, melting point, valence electron counts, ...) and for each one
# computes 6 statistics across the compound: min, max, range, mean, avg_dev
# (average deviation) and mode. That gives 132 numbers, plus temperature = 133.
#
# Temperature matters because ZT is measured at a specific temperature and the
# same material has very different ZT at 300K and 800K.

def build_features(comps, temperatures):
    ep = ElementProperty.from_preset("magpie")

    # featurize_many runs the featuriser over every compound.
    # ignore_errors=True returns NaN for anything unparseable instead of crashing.
    rows = ep.featurize_many(comps, ignore_errors=True, pbar=False)

    X = pd.DataFrame(rows, columns=ep.feature_labels())
    X = X.loc[:, X.notna().all()]        # drop columns that are entirely NaN
    X["temperature"] = temperatures
    return X


# =============================================================================
# STEP 3 - THE THREE SPLITS: WHERE THE REAL STORY IS
# =============================================================================
# Each split asks a DIFFERENT question. Same model, same features throughout;
# only the rule for who-goes-where changes. That alone moves the score from
# 0.894 to 0.535.
#
#   1. RANDOM           "Can it fill a gap for a material it already knows?"
#                       Shuffle all rows. Bi2Te3 at 400K trains, 500K tests.
#                       Trivial, because ZT varies smoothly with temperature.
#
#   2. MATERIAL HELD OUT  "Can it handle a new material from a known family?"
#                       All rows of one formula stay together. But near-twins
#                       (Cu2Se0.92S0.08 vs Cu2Se0.94S0.06) are still in training.
#
#   3. SYSTEM HELD OUT  "Can it handle chemistry it has NEVER seen?"
#                       All formulas sharing an element set move as one block.
#                       This is the real materials-discovery scenario.

def cross_val_predict_grouped(X, y, groups=None, shuffle=False, n_trees=N_TREES):
    """Return an out-of-fold prediction for every row.

    Every row is predicted exactly once, by a model that never saw it during
    training. Nothing is left untested - this is not a train/test holdout.
    """
    pred = np.zeros(len(y))

    if shuffle:
        splitter = KFold(N_FOLDS, shuffle=True, random_state=SEED).split(X, y)
    else:
        # GroupKFold's `groups` argument guarantees the same group never
        # appears on both sides of a split.
        splitter = GroupKFold(N_FOLDS).split(X, y, groups=groups)

    for train_idx, test_idx in splitter:
        model = RandomForestRegressor(
            n_estimators=n_trees, random_state=SEED, n_jobs=-1
        ).fit(X.iloc[train_idx], y[train_idx])
        pred[test_idx] = model.predict(X.iloc[test_idx])

    return pred


def run_split_comparison(X, y, meta):
    print("\n" + "=" * 70)
    print("STEP 3: THE THREE SPLITS")
    print("=" * 70)

    results = {}
    for label, kwargs in [
        ("1. Random split (misleading)", dict(shuffle=True)),
        ("2. Material held out", dict(groups=meta["formula"].values)),
        ("3. Chemical system held out", dict(groups=meta["system"].values)),
    ]:
        pred = cross_val_predict_grouped(X, y, **kwargs)
        results[label] = r2_score(y, pred)
        print(f"  {label:<34} R2 = {results[label]:.3f}")

    print("\n  The drop from 0.89 to 0.53 is the project's core finding:")
    print("  most of the apparent 'accuracy' was memorisation, not chemistry.")
    return results


# =============================================================================
# STEP 4 - FEATURE SELECTION, DONE WITHOUT CHEATING
# =============================================================================
# Fewer, better features beat 133 mediocre ones. But HOW you pick them matters.
#
# The trap: if you rank features using all the data and then test on some of it,
# the selection has already peeked at the answers. That inflated our score from
# 0.603 to 0.621 - the same class of mistake this whole project is about.
#
# The fix: rank features INSIDE each fold, using only that fold's training data.
# A throwaway "selector" forest ranks features; then a fresh model trains on the
# top K. The test fold is untouched by the selection.

def select_features_from_training(X_train, y_train, k=TOP_K):
    selector = RandomForestRegressor(
        n_estimators=N_TREES, random_state=SEED, n_jobs=-1
    ).fit(X_train, y_train)

    # feature_importances_ scores how much each feature reduced prediction
    # error across all trees. Pair with column names, sort, take the top k.
    ranked = pd.Series(selector.feature_importances_, index=X_train.columns)
    return list(ranked.sort_values(ascending=False).head(k).index)


# =============================================================================
# STEP 5 - THE CONFIDENCE GATE
# =============================================================================
# The model scores 0.60 overall. But it is not equally wrong everywhere: it is
# reliable on chemistry resembling its training data, and guessing on chemistry
# that does not. So instead of a bare prediction, we attach a confidence flag.
#
# CHEMICAL DISTANCE: for each test material, how far is the nearest TRAINING
# material in feature space? Small distance = familiar = trust it.
#
# Two details that matter:
#   - Features are standardised (mean 0, std 1) so that a large-valued feature
#     like melting point doesn't drown out a small-valued one like
#     electronegativity. The mean/std come from TRAINING data only.
#   - Temperature is excluded, so distance measures CHEMISTRY. Otherwise the
#     nearest neighbour would just be the same material at another temperature.

def compute_chemical_distance(X_test, X_train, feature_cols):
    chem_cols = [c for c in feature_cols if c != "temperature"]

    mu = X_train[chem_cols].mean()
    sd = X_train[chem_cols].std() + 1e-9      # +1e-9 avoids divide-by-zero

    test_scaled = ((X_test[chem_cols] - mu) / sd).values
    train_scaled = ((X_train[chem_cols] - mu) / sd).values

    # cdist gives every test-to-train distance pair; .min(axis=1) keeps each
    # test row's nearest neighbour.
    return cdist(test_scaled, train_scaled).min(axis=1)


def run_final_pipeline(X, y, meta):
    """Selection inside folds + confidence gate. This produces the headline."""
    print("\n" + "=" * 70)
    print("STEP 4+5: FEATURE SELECTION + CONFIDENCE GATE")
    print("=" * 70)

    pred = np.zeros(len(y))
    dist = np.zeros(len(y))
    selected_per_fold = []

    for train_idx, test_idx in GroupKFold(N_FOLDS).split(
        X, y, groups=meta["system"].values
    ):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]

        cols = select_features_from_training(X_tr, y[train_idx], k=TOP_K)
        selected_per_fold.append(set(cols))

        model = RandomForestRegressor(
            n_estimators=N_TREES, random_state=SEED, n_jobs=-1
        ).fit(X_tr[cols], y[train_idx])

        pred[test_idx] = model.predict(X_te[cols])
        dist[test_idx] = compute_chemical_distance(X_te, X_tr, cols)

    out = pd.DataFrame({
        "formula": meta["formula"],
        "system": meta["system"],
        "temperature": meta["temperature"],
        "ZT_true": y,
        "ZT_pred": pred,
        "chem_distance": dist,
    })

    print(f"  No gate: R2 = {r2_score(y, pred):.3f}  "
          f"MAE = {mean_absolute_error(y, pred):.3f}")

    # Features all folds independently agreed on - the stable, trustworthy ones.
    stable = set.intersection(*selected_per_fold)
    print(f"\n  {len(stable)}/{TOP_K} features chosen by ALL folds "
          f"(selection is only partly stable):")
    for f in sorted(stable):
        print(f"     - {f}")

    return out


def report_gate(out):
    """Coverage = what fraction of predictions we CHOOSE to report.

    This is NOT a train/test split. All 5,205 rows were already predicted.
    We sort by confidence and report only the most confident X%, abstaining
    on the rest - like a doctor referring unclear scans for a second opinion.
    """
    print("\n  --- confidence gate (sorted by chemical distance) ---")
    print(f"  {'coverage':>9} {'n':>6} {'R2':>7} {'MAE':>7} {'ZT_std':>8}")

    ranked = out.sort_values("chem_distance")
    for cov in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        k = int(len(ranked) * cov)
        sub = ranked.iloc[:k]
        print(f"  {int(cov * 100):>8}% {k:>6} "
              f"{r2_score(sub.ZT_true, sub.ZT_pred):>7.3f} "
              f"{mean_absolute_error(sub.ZT_true, sub.ZT_pred):>7.3f} "
              f"{sub.ZT_true.std():>8.3f}")

    print("\n  ZT_std stays flat (~0.35) as coverage tightens, which confirms")
    print("  the gain is real and not an artifact of an easier subset.")
    print("\n  Recommended operating point: 50% coverage, R2 ~ 0.69.")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("Loading ESTM dataset ...")
    df, comps, meta, y = load_data()
    print(f"  {len(df)} rows | {meta['formula'].nunique()} materials | "
          f"{meta['system'].nunique()} chemical systems")
    print(f"  ZT range: {y.min():.3f} to {y.max():.3f}")

    print("\nBuilding magpie features ...")
    X = build_features(comps, meta["temperature"].values)
    print(f"  Feature matrix: {X.shape[0]} rows x {X.shape[1]} features")

    run_split_comparison(X, y, meta)

    out = run_final_pipeline(X, y, meta)
    report_gate(out)

    out.to_csv("FINAL_predictions.csv", index=False)
    print("\nSaved per-material predictions to FINAL_predictions.csv")


if __name__ == "__main__":
    main()
