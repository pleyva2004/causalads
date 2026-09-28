"""Stage: prediction — who is likely to convert? Classical supervised ML on the RCT data.

Models are tuned by Optuna on TRAIN 5-fold CV (objective PR-AUC), the imbalance and
CV-variance studies use train / train+val only, calibrators are fit on VAL, and the
test split is scored once."""
from __future__ import annotations

import time

import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from causalads import config as C
from causalads import data, io

NJ = 4
MODELS = ["lr", "rf", "xgb", "lgbm"]
TREES = ["rf", "xgb", "lgbm"]
LABEL = {"lr": "Logistic Regression", "rf": "Random Forest", "xgb": "XGBoost", "lgbm": "LightGBM"}
STRATS = ["none", "weight", "undersample"]
TRIALS = {"lr": 0, "rf": 20, "xgb": 30, "lgbm": 30}
UNDERSAMPLE_RATIO = 10  # negatives kept per positive
TOP_FRAC = 0.20  # contact rule: score the top 20% (matches a fixed-budget campaign)
N_BOOT = C.N_BOOTSTRAP


def features(df: pd.DataFrame) -> pd.DataFrame:
    X = df[data.FEATURES].astype(float).copy()
    X["mens_email"] = (df["arm"] == 1).astype(float)
    X["womens_email"] = (df["arm"] == 2).astype(float)
    return X


def make(name: str, params: dict, spw: float | None = None) -> Pipeline:
    """spw = neg/pos ratio when reweighting, else None. Preprocessing lives in the Pipeline."""
    imp = ("imp", SimpleImputer(strategy="median"))
    if name == "lr":
        est = LogisticRegression(C=params.get("C", 1.0), max_iter=3000,
                                 class_weight="balanced" if spw else None)
        return Pipeline([imp, ("sc", StandardScaler()), ("m", est)])
    if name == "rf":
        est = RandomForestClassifier(**params, class_weight="balanced_subsample" if spw else None,
                                     n_jobs=NJ, random_state=C.SEED)
    elif name == "xgb":
        est = XGBClassifier(**params, scale_pos_weight=spw or 1.0, tree_method="hist", n_jobs=NJ,
                            random_state=C.SEED, eval_metric="logloss", verbosity=0)
    else:
        est = LGBMClassifier(**params, scale_pos_weight=spw or 1.0, n_jobs=NJ,
                             random_state=C.SEED, verbose=-1)
    return Pipeline([imp, ("m", est)])


def fit_predict(name: str, params: dict, strat: str, Xtr: pd.DataFrame, ytr: np.ndarray,
                Xte: pd.DataFrame, seed: int = C.SEED, return_model: bool = False):
    spw = None
    if strat == "weight":
        spw = float((ytr == 0).sum() / max(ytr.sum(), 1))
    elif strat == "undersample":
        rng = np.random.default_rng(seed)
        neg = np.flatnonzero(ytr == 0)
        keep = np.concatenate([np.flatnonzero(ytr == 1),
                               rng.choice(neg, min(len(neg), UNDERSAMPLE_RATIO * int(ytr.sum())), replace=False)])
        Xtr, ytr = Xtr.iloc[keep], ytr[keep]
    model = make(name, params, spw).fit(Xtr, ytr)
    p = model.predict_proba(Xte)[:, 1]
    return (p, model) if return_model else p


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error with equal-mass bins (equal-width is degenerate at a 0.9% base rate)."""
    order = np.argsort(p, kind="stable")
    return float(sum(len(b) / len(y) * abs(p[b].mean() - y[b].mean()) for b in np.array_split(order, bins)))


def reliability(y: np.ndarray, p: np.ndarray, bins: int = 10) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(p, kind="stable")
    parts = np.array_split(order, bins)
    return np.array([p[b].mean() for b in parts]), np.array([y[b].mean() for b in parts])


def ci(draws: list | np.ndarray) -> tuple[float, float]:
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return float(lo), float(hi)


def tune(name: str, X: pd.DataFrame, y: np.ndarray) -> dict:
    skf = list(StratifiedKFold(5, shuffle=True, random_state=C.SEED).split(X, y))

    def cv_ap(params: dict) -> float:
        return float(np.mean([average_precision_score(y[va], fit_predict(name, params, "none", X.iloc[tr], y[tr], X.iloc[va]))
                              for tr, va in skf]))

    if name == "lr":
        grid = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
        scores = {c: cv_ap({"C": c}) for c in grid}
        best = max(scores, key=scores.get)
        return {"params": {"C": best}, "cv_pr_auc": scores[best], "n_trials": len(grid)}

    def objective(trial: optuna.Trial) -> float:
        if name == "rf":
            p = {"n_estimators": trial.suggest_int("n_estimators", 100, 250, step=50),
                 "max_depth": trial.suggest_int("max_depth", 3, 12),
                 "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 200, log=True),
                 "max_features": trial.suggest_float("max_features", 0.3, 1.0)}
        elif name == "xgb":
            p = {"n_estimators": trial.suggest_int("n_estimators", 100, 400, step=50),
                 "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                 "max_depth": trial.suggest_int("max_depth", 2, 5),
                 "min_child_weight": trial.suggest_float("min_child_weight", 1, 50, log=True),
                 "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                 "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                 "reg_lambda": trial.suggest_float("reg_lambda", 1, 30, log=True)}
        else:
            p = {"n_estimators": trial.suggest_int("n_estimators", 100, 400, step=50),
                 "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                 "num_leaves": trial.suggest_int("num_leaves", 4, 31),
                 "min_child_samples": trial.suggest_int("min_child_samples", 20, 200, log=True),
                 "subsample": trial.suggest_float("subsample", 0.5, 1.0), "subsample_freq": 1,
                 "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                 "reg_lambda": trial.suggest_float("reg_lambda", 1, 30, log=True)}
        return cv_ap(p)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=C.SEED))
    study.optimize(objective, n_trials=TRIALS[name])
    params = dict(study.best_params)
    if name == "lgbm":
        params["subsample_freq"] = 1
    return {"params": params, "cv_pr_auc": float(study.best_value), "n_trials": TRIALS[name]}


def imbalance_study(tuned: dict, X: pd.DataFrame, y: np.ndarray) -> tuple[dict, dict]:
    """Same tuned hyperparameters, three imbalance strategies, 5-fold CV on train only."""
    skf = list(StratifiedKFold(5, shuffle=True, random_state=C.SEED + 1).split(X, y))
    base = float(y.mean())
    out, oof_all = {}, {}
    for name in MODELS:
        for strat in STRATS:
            rows, oof = [], np.zeros(len(y))
            for k, (tr, va) in enumerate(skf):
                p = fit_predict(name, tuned[name]["params"], strat, X.iloc[tr], y[tr], X.iloc[va], seed=k)
                oof[va] = p
                rows.append({"roc_auc": roc_auc_score(y[va], p), "pr_auc": average_precision_score(y[va], p),
                             "brier": brier_score_loss(y[va], p), "ece": ece(y[va], p),
                             "mean_pred_over_base": p.mean() / y[va].mean()})
            df = pd.DataFrame(rows)
            out[f"{name}|{strat}"] = {m: {"mean": float(df[m].mean()), "std": float(df[m].std(ddof=1))} for m in df}
            oof_all[(name, strat)] = oof
    return {"base_rate": base, "cv": out}, oof_all


def repeated_cv(tuned: dict, X: pd.DataFrame, y: np.ndarray) -> dict:
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=C.SEED)
    out = {}
    for name in MODELS:
        roc, ap = [], []
        for tr, te in rskf.split(X, y):
            p = fit_predict(name, tuned[name]["params"], "none", X.iloc[tr], y[tr], X.iloc[te])
            roc.append(roc_auc_score(y[te], p))
            ap.append(average_precision_score(y[te], p))
        out[name] = {"roc_auc": roc, "pr_auc": ap,
                     "roc_mean": float(np.mean(roc)), "roc_std": float(np.std(roc, ddof=1)),
                     "pr_mean": float(np.mean(ap)), "pr_std": float(np.std(ap, ddof=1))}
    return out


def topk_lift(y: np.ndarray, s: np.ndarray, frac: float) -> float:
    k = int(frac * len(y))
    return float(y[np.argpartition(-s, k)[:k]].mean() / max(y.mean(), 1e-12))


def boot_metrics(y: np.ndarray, S: dict[str, np.ndarray], best: str, n_boot: int = N_BOOT) -> dict:
    """Percentile bootstrap over test rows, same resamples for every model (paired)."""
    rng = np.random.default_rng(C.SEED)
    n = len(y)
    fns = {"roc_auc": roc_auc_score, "pr_auc": average_precision_score, "brier": brier_score_loss,
           "log_loss": lambda a, b: log_loss(a, np.clip(b, 1e-6, 1 - 1e-6), labels=[0, 1]),
           "lift_top10": lambda a, b: topk_lift(a, b, 0.10), "lift_top20": lambda a, b: topk_lift(a, b, 0.20)}
    draws = {m: {k: [] for k in fns} for m in S}
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        yi = y[i]
        if yi.sum() == 0:
            continue
        for m in S:
            for k, f in fns.items():
                draws[m][k].append(f(yi, S[m][i]))
    out = {"models": {}}
    for m in S:
        out["models"][m] = {k: {"est": float(f(y, S[m])), "ci": ci(draws[m][k])} for k, f in fns.items()}
    if best != "lr":
        for k in ("roc_auc", "pr_auc"):
            d = np.array(draws[best][k]) - np.array(draws["lr"][k])
            est = out["models"][best][k]["est"] - out["models"]["lr"][k]["est"]
            out[f"diff_{k}"] = {"est": float(est), "ci": ci(d),
                                "p_two_sided": float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))}
    return out


def calibration_study(yv: np.ndarray, pv: dict, yt: np.ndarray, pt: dict, n_boot: int = 1000) -> dict:
    """Fit Platt / isotonic on VAL predictions, evaluate before/after on TEST."""
    eps = 1e-6
    lg = lambda p: np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps)))
    variants: dict[str, dict[str, np.ndarray]] = {}
    for m in pv:
        platt = LogisticRegression(C=1e6, max_iter=1000).fit(lg(pv[m])[:, None], yv)
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(pv[m], yv)
        variants[m] = {"raw": pt[m], "platt": platt.predict_proba(lg(pt[m])[:, None])[:, 1], "isotonic": iso.predict(pt[m])}
    rng = np.random.default_rng(C.SEED)
    n = len(yt)
    idx = [rng.integers(0, n, n) for _ in range(n_boot)]
    out = {}
    for m, vs in variants.items():
        out[m] = {}
        for v, p in vs.items():
            b = [brier_score_loss(yt[i], p[i]) for i in idx]
            e = [ece(yt[i], p[i]) for i in idx]
            out[m][v] = {"brier": {"est": float(brier_score_loss(yt, p)), "ci": ci(b)},
                         "ece": {"est": ece(yt, p), "ci": ci(e)}, "mean_pred": float(p.mean())}
    return {"results": out, "variants": variants, "base_rate_test": float(yt.mean())}


def decile_table(y: np.ndarray, s: np.ndarray, n_boot: int = 1000) -> list[dict]:
    def rates(yy: np.ndarray, ss: np.ndarray) -> np.ndarray:
        order = np.argsort(-ss, kind="stable")
        return np.array([yy[b].mean() for b in np.array_split(order, 10)])
    rng = np.random.default_rng(C.SEED)
    n = len(y)
    draws = np.array([rates(y[i], s[i]) for i in (rng.integers(0, n, n) for _ in range(n_boot))])
    r, base = rates(y, s), y.mean()
    order = np.argsort(-s, kind="stable")
    conv = np.array([y[b].sum() for b in np.array_split(order, 10)])
    return [{"decile": d + 1, "rate": float(r[d]), "rate_ci": ci(draws[:, d]), "lift": float(r[d] / base),
             "lift_ci": ci(draws[:, d] / base), "conversions": int(conv[d]),
             "cum_gain": float(conv[:d + 1].sum() / conv.sum())} for d in range(10)]


def subgroup_table(te: pd.DataFrame, y: np.ndarray, s: np.ndarray, n_boot: int = 500) -> list[dict]:
    rng = np.random.default_rng(C.SEED)
    keys = {"channel": te["channel"], "zip_code": te["zip_code"],
            "newbie": te["newbie"].map({0: "existing", 1: "newbie"}),
            "history_bucket": te["history_bucket"].astype(str), "arm": te["arm"].map(C.ARMS)}
    rows = []
    for var, col in keys.items():
        for g in sorted(col.unique()):
            idx = np.flatnonzero(col.to_numpy() == g)
            yg, sg = y[idx], s[idx]
            obs, pred = [], []
            auc = []
            for _ in range(n_boot):
                j = rng.integers(0, len(idx), len(idx))
                obs.append(yg[j].mean())
                pred.append(sg[j].mean())
                if 0 < yg[j].sum() < len(j):
                    auc.append(roc_auc_score(yg[j], sg[j]))
            enough = yg.sum() >= 10
            rows.append({"variable": var, "group": g, "n": int(len(idx)), "conversions": int(yg.sum()),
                         "observed": float(yg.mean()), "observed_ci": ci(obs),
                         "predicted": float(sg.mean()), "predicted_ci": ci(pred),
                         "roc_auc": float(roc_auc_score(yg, sg)) if enough else None,
                         "roc_auc_ci": ci(auc) if enough else None})
    return rows


def false_negatives(te: pd.DataFrame, y: np.ndarray, flag: np.ndarray) -> dict:
    conv = te[y == 1].copy()
    conv["caught"] = flag[y == 1]
    cols = ["recency", "history", "newbie", "recent_buyer", "zip_urban", "zip_suburban", "zip_rural",
            "ch_phone", "ch_web", "ch_multi", "mens", "womens"]
    conv["mens_email"] = (conv["arm"] == 1).astype(int)
    conv["womens_email"] = (conv["arm"] == 2).astype(int)
    conv["no_email"] = (conv["arm"] == 0).astype(int)
    cols += ["mens_email", "womens_email", "no_email"]
    tab = conv.groupby("caught")[cols].mean().T.rename(columns={True: "caught", False: "missed"})
    return {"n_caught": int(conv["caught"].sum()), "n_missed": int((~conv["caught"]).sum()),
            "means": {k: {"caught": float(r["caught"]), "missed": float(r["missed"])} for k, r in tab.iterrows()},
            "median_spend": {"caught": float(conv.loc[conv["caught"], "spend"].median()),
                             "missed": float(conv.loc[~conv["caught"], "spend"].median())}}


def run() -> dict:
    from causalads import prediction_report as rep
    io.style()
    t0 = time.time()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    tr, va, te = data.split_frames()
    Xtr, Xva, Xte = features(tr), features(va), features(te)
    trva = pd.concat([tr, va], ignore_index=True)
    Xall = features(trva)
    R: dict = {"n": {"train": len(tr), "val": len(va), "test": len(te)}, "features": list(Xtr.columns)}
    A: dict = {}

    for target in ("conversion", "visit"):
        ytr, yva, yte = (d[target].to_numpy() for d in (tr, va, te))
        yall = trva[target].to_numpy()
        T: dict = {"base_rate": {"train": float(ytr.mean()), "test": float(yte.mean()), "test_positives": int(yte.sum())}}
        tuned = {}
        for name in MODELS:
            tuned[name] = tune(name, Xtr, ytr)
            print(f"[{target}] tuned {name}: cv PR-AUC {tuned[name]['cv_pr_auc']:.4f} ({time.time() - t0:.0f}s)", flush=True)
        T["tuned"] = tuned

        # final fits on TRAIN only; selection on VAL PR-AUC, test untouched until below
        pv, pt, models = {}, {}, {}
        for name in MODELS:
            pv[name], models[name] = fit_predict(name, tuned[name]["params"], "none", Xtr, ytr, Xva, return_model=True)
            pt[name] = models[name].predict_proba(Xte)[:, 1]
        val_pr = {m: float(average_precision_score(yva, pv[m])) for m in MODELS}
        val_roc = {m: float(roc_auc_score(yva, pv[m])) for m in MODELS}
        best = max(MODELS, key=val_pr.get)
        best_tree = max(TREES, key=val_pr.get)
        T.update({"val_pr_auc": val_pr, "val_roc_auc": val_roc, "best": best, "best_tree": best_tree})
        print(f"[{target}] val PR-AUC {val_pr} -> best {best}", flush=True)

        T["test"] = boot_metrics(yte, pt, best)
        print(f"[{target}] test bootstrap done ({time.time() - t0:.0f}s)", flush=True)
        T["cv_repeated"] = repeated_cv(tuned, Xall, yall)
        print(f"[{target}] repeated CV done ({time.time() - t0:.0f}s)", flush=True)
        A[target] = {"yte": yte, "pt": pt, "pv": pv, "yva": yva}

        if target == "conversion":
            imb, oof = imbalance_study(tuned, Xtr, ytr)
            T["imbalance"] = imb
            A[target]["oof"] = oof
            A[target]["ytr"] = ytr
            print(f"[{target}] imbalance study done ({time.time() - t0:.0f}s)", flush=True)

            cal = calibration_study(yva, pv, yte, pt)
            T["calibration"] = {"results": cal["results"], "base_rate_test": cal["base_rate_test"]}
            A[target]["cal_variants"] = cal["variants"]

            # decision rule fixed a priori: contact top 20% by score; threshold = val 80th percentile
            s_best = pt[best]
            thr = float(np.quantile(pv[best], 1 - TOP_FRAC))
            flag = s_best >= thr
            tp, fp = int((flag & (yte == 1)).sum()), int((flag & (yte == 0)).sum())
            fn, tn = int((~flag & (yte == 1)).sum()), int((~flag & (yte == 0)).sum())
            rng = np.random.default_rng(C.SEED)
            prec, rec = [], []
            for _ in range(N_BOOT):
                i = rng.integers(0, len(yte), len(yte))
                f, yi = flag[i], yte[i]
                prec.append((f & (yi == 1)).sum() / max(f.sum(), 1))
                rec.append((f & (yi == 1)).sum() / max(yi.sum(), 1))
            T["confusion"] = {"rule": f"contact customers whose score is >= the VAL {int((1 - TOP_FRAC) * 100)}th percentile "
                                      f"of the best model's scores (top {int(TOP_FRAC * 100)}%)",
                              "threshold": thr, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                              "flagged_share": float(flag.mean()),
                              "precision": {"est": tp / max(tp + fp, 1), "ci": ci(prec)},
                              "recall": {"est": tp / max(tp + fn, 1), "ci": ci(rec)}}
            T["deciles"] = {"best": decile_table(yte, s_best),
                            "lr": decile_table(yte, pt["lr"])}
            T["false_negatives"] = false_negatives(te, yte, flag)
            T["subgroups"] = subgroup_table(te, yte, s_best)
            A[target]["flag"] = flag
            A[target]["best_scores"] = s_best
            print(f"[{target}] calibration/deciles/subgroups done ({time.time() - t0:.0f}s)", flush=True)

            # attribution on the best tree model (val rows: never used for fitting)
            import shap
            mdl = models[best_tree]
            Xv_t = pd.DataFrame(mdl[:-1].transform(Xva), columns=Xva.columns)
            samp = Xv_t.sample(n=min(6000, len(Xv_t)), random_state=C.SEED)
            sv = shap.TreeExplainer(mdl[-1]).shap_values(samp)
            sv = sv[1] if isinstance(sv, list) else (sv[:, :, 1] if sv.ndim == 3 else sv)
            mean_abs = np.abs(sv).mean(0)
            pi = permutation_importance(mdl, Xva, yva, scoring="average_precision", n_repeats=10,
                                        random_state=C.SEED, n_jobs=1)
            cols = list(Xva.columns)
            rho, prho = spearmanr(mean_abs, pi.importances_mean)
            T["attribution"] = {
                "model": best_tree, "n_shap_rows": len(samp),
                "shap_mean_abs": dict(zip(cols, mean_abs.tolist())),
                "perm_importance": {c: {"mean": float(pi.importances_mean[k]), "std": float(pi.importances_std[k])}
                                    for k, c in enumerate(cols)},
                "spearman_shap_vs_perm": {"rho": float(rho), "p": float(prho)},
                "top3_shap": [cols[k] for k in np.argsort(-mean_abs)[:3]],
                "top3_perm": [cols[k] for k in np.argsort(-pi.importances_mean)[:3]]}
            A[target]["shap"] = (sv, samp)
            print(f"[{target}] attribution done ({time.time() - t0:.0f}s)", flush=True)
        R[target] = T

    R["headline"] = _headline(R)
    R["runtime_seconds"] = time.time() - t0
    io.save_json(R, "prediction")
    rep.figures(R, A, te)
    (C.RESULTS / "prediction.md").write_text(rep.markdown(R))
    print(f"prediction done in {time.time() - t0:.0f}s")
    return R


def _headline(R: dict) -> dict:
    h: dict = {}
    for t in ("conversion", "visit"):
        T = R[t]
        h[t] = {"best_model_by_val_pr_auc": T["best"], "base_rate_test": T["base_rate"]["test"],
                "test": {m: {k: T["test"]["models"][m][k] for k in ("roc_auc", "pr_auc", "brier", "log_loss")} for m in MODELS},
                "cv_5x3": {m: {"roc_auc_mean": T["cv_repeated"][m]["roc_mean"], "roc_auc_std": T["cv_repeated"][m]["roc_std"],
                               "pr_auc_mean": T["cv_repeated"][m]["pr_mean"], "pr_auc_std": T["cv_repeated"][m]["pr_std"]}
                           for m in MODELS}}
        for k in ("diff_roc_auc", "diff_pr_auc"):
            if k in T["test"]:
                h[t][f"best_vs_lr_{k}"] = T["test"][k]
    c = R["conversion"]
    b = c["best"]
    h["conversion"]["calibration_test"] = {m: {v: {"brier": r["brier"], "ece": r["ece"]} for v, r in c["calibration"]["results"][m].items()}
                                           for m in MODELS}
    h["conversion"]["top_decile_lift"] = {m: c["test"]["models"][m]["lift_top10"] for m in MODELS}
    h["conversion"]["top3_shap_features"] = c["attribution"]["top3_shap"]
    h["conversion"]["contact_top20"] = {"precision": c["confusion"]["precision"], "recall": c["confusion"]["recall"]}
    return h
