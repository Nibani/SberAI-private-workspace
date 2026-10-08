"""Reproduce fixed-label interpretation and the separately registered analogue test.

Run only after the lead has saved C and allocated the single compute slot:
    python -X utf8 -m scripts.competition_interpretation --execute

This module never fits the main grouping, reads a 2025 outcome, or fetches a URL.
Mobility values are deliberately absent: only full-name candidates are audited.
"""
from __future__ import annotations

import os
for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_variable] = "1"

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.metrics import adjusted_rand_score
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from sbercluster import panel as P

OUT = ROOT / "reports/competition-enhancement"
SPEC_PATH = OUT / "mirkin/analysis-spec.json"
PREREG_PATH = OUT / "economic_cases/analogue-preregistration.json"
SPEC_SHA = "ddc35f6111a53ef8714d12596c4b4f1423e9f28e1002eb5e1ccedd0d195096b0"
PREREG_SHA = "5f9ec3bd9f1ac9fdbdcb0cfa8981919e75d511a6990fa77b0a6ca7b034971b2b"
GROUPS = tuple(range(4))
GROUP_NAMES = ("Высокие расходы и услуги", "Средние расходы", "Низкие расходы и повседневные покупки", "Низкая доля маркетплейсов")
FEATURES = ("total_rub", "health_pct", "marketplaces_pct", "horeca_pct", "food_pct", "transport_pct", "wage_total", "population_total", "market_access", "agriculture_share", "manufacturing_share", "mining_share", "public_share")
PROXIES = ("Proxy_I", "Proxy_II", "Proxy_III", "Proxy_IV", "UNKNOWN")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean_json(value.tolist())
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    return value


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean_json(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def save_csv(path: Path, rows, columns=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows, columns=columns)
    frame.to_csv(path, index=False, encoding="utf-8", float_format="%.17g", na_rep="")


def status(folder: str, state: str, details: dict) -> None:
    directory = OUT / folder
    files = {str(p.relative_to(ROOT)).replace("\\", "/"): digest(p) for p in sorted(directory.rglob("*")) if p.is_file() and p.name != "status.json"}
    save_json(directory / "status.json", {"status": state, "spec_sha256": SPEC_SHA, "details": details, "files": files})


def finite_mean(x) -> float:
    values = [float(v) for v in np.asarray(x).ravel() if math.isfinite(v)]
    return math.fsum(values) / len(values) if values else float("nan")


def ratio(numerator, denominator):
    n, d = np.asarray(numerator, float), np.asarray(denominator, float)
    result = np.full(n.shape, np.nan)
    eligible = np.isfinite(n) & (n >= 0) & np.isfinite(d) & (d > 0)
    result[eligible] = n[eligible] / d[eligible]
    return result


def deviations(group_mean, global_mean):
    a, b = np.broadcast_arrays(np.asarray(group_mean, float), np.asarray(global_mean, float))
    out = np.full(a.shape, np.nan)
    good = np.isfinite(a) & np.isfinite(b) & (np.abs(b) > 1e-12)
    out[good] = (a[good] - b[good]) / b[good]
    return out


def frozen_x23(panel, model_path):
    """Restore 2023 attributes with the recorded scales, including legacy NPZs.

    The legacy four-key file predates serialization of the monthly reference.
    run_v12 and the atlas use the full pinned panel's median log total in each
    month; this is recoverable for 2023 without opening a new outcome.
    """
    with np.load(model_path, allow_pickle=False) as model:
        required = {"centers", "structure_center", "scale", "ids"}
        if not required.issubset(model.files):
            raise ValueError("Frozen model lacks a required recorded array")
        if not np.array_equal(panel.ids, model["ids"]):
            raise ValueError("Frozen model ID order changed")
        structure_center = model["structure_center"].copy()
        scale = model["scale"].copy()
        centers = model["centers"].copy()
        if structure_center.shape != (5,) or scale.shape != (6,) or np.any(scale <= 0):
            raise ValueError("Invalid frozen structure center or scale")
        if "level_center" in model.files:
            reference = model["level_center"]
            if reference.shape != (panel.totals.shape[1],):
                raise ValueError("Stored monthly reference has the wrong shape")
            level_center = reference[:12].copy()
        else:
            level_center = np.median(np.log(panel.totals[:, :12]), axis=0)
    structure = (np.log(panel.shares[:, :12]) - structure_center) / scale[:5]
    level = (np.log(panel.totals[:, :12]) - level_center) / scale[5]
    monthly = np.concatenate([structure, level[:, :, None]], axis=2)
    return P.annual_profile(monthly, slice(0, 12)), centers


def baseline_check():
    """Check actual IDs, 2023 coordinates, centers and peer lists, no new outcomes."""
    preflight()
    from scripts.extend_atlas_v12 import network_neighbors
    from scripts.atlas_payload import read_atlas_payload
    panel = P.read_panel(ROOT / "data/v12/panel.csv.gz")
    profile, stored_centers = frozen_x23(panel, ROOT / "reports/v1.2/model.npz")
    summary = json.loads((ROOT / "reports/v1.2/summary.json").read_text(encoding="utf-8"))
    config = json.loads((ROOT / "configs/v12.json").read_text(encoding="utf-8"))
    reference_ids, reference_profile, reference_nearest = network_neighbors(summary, config)
    if not np.array_equal(panel.ids, np.array(reference_ids)):
        raise ValueError("Frozen IDs differ from the actual atlas build recipe")
    if not np.array_equal(profile, reference_profile):
        raise ValueError("Frozen X23 differs bitwise from the actual atlas build recipe")
    labels = pd.read_csv(ROOT / "reports/v1.2/labels.csv").set_index("entity_id").reindex(panel.ids)
    group = labels.type_2023.to_numpy(int)
    restored_centers = np.array([profile[group == g].mean(axis=0) for g in GROUPS])
    if not np.array_equal(stored_centers, restored_centers):
        raise ValueError("Frozen centers are not bitwise reproduced by actual group means")
    distance = cdist(profile, profile)
    np.fill_diagonal(distance, np.inf)
    nearest = np.argsort(distance, axis=1, kind="stable")[:, :15]
    if not np.array_equal(nearest, reference_nearest):
        raise ValueError("Frozen nearest15 differs from the actual atlas build recipe")
    atlas = read_atlas_payload(ROOT / "docs/index.html")
    entities = {e["id"]: e for e in atlas["entities"]}
    if set(entities) != set(panel.ids):
        raise ValueError("Actual delivered atlas IDs differ from the frozen panel")
    for i, entity in enumerate(panel.ids):
        if entities[entity]["v12_features"] != [round(float(v),10) for v in profile[i]]:
            raise ValueError(f"Delivered atlas X23 differs: {entity}")
        if entities[entity]["v12_neighbors"] != list(panel.ids[nearest[i]]):
            raise ValueError(f"Delivered atlas nearest15 differs: {entity}")
    with np.load(ROOT / "reports/v1.2/model.npz", allow_pickle=False) as model:
        keys = list(model.files)
    return {"baseline_only": True, "entities": len(panel.ids), "model_keys": keys,
            "ids_exact": True, "X23_bitwise_exact_against_build": True,
            "centers_bitwise_exact": True, "nearest15_exact_against_build": True,
            "atlas_rounded_X23_exact": True, "atlas_nearest15_exact": True,
            "new_economic_outcomes_computed": False,
            "X23_array_sha256": hashlib.sha256(profile.tobytes()).hexdigest(),
            "nearest15_array_sha256": hashlib.sha256(nearest.tobytes()).hexdigest(),
            "model_sha256": digest(ROOT/"reports/v1.2/model.npz"),
            "panel_sha256": digest(ROOT/"data/v12/panel.csv.gz"),
            "summary_sha256": digest(ROOT/"reports/v1.2/summary.json"),
            "actual_build_source_sha256": digest(ROOT/"scripts/extend_atlas_v12.py"),
            "delivered_atlas_html_sha256": digest(ROOT/"docs/index.html"),
            "checked_source_sha256": digest(Path(__file__))}


def annual_data():
    panel = P.read_panel(ROOT / "data/v12/panel.csv.gz")
    labels = pd.read_csv(ROOT / "reports/v1.2/labels.csv", dtype={"entity_id": str}).set_index("entity_id").reindex(panel.ids)
    required_labels = ["name", "region", "type_2023", "type_2024", "type_2024_absolute", "persistent_change"]
    if labels[required_labels].isna().any().any() or not np.array_equal(labels["name"].to_numpy(), panel.names):
        raise ValueError("Labels and canonical panel metadata do not agree")
    group = labels.type_2023.to_numpy(int)
    if [int(np.sum(group == g)) for g in GROUPS] != [338, 580, 971, 127]:
        raise ValueError("Current main group counts changed")
    cohort = pd.read_csv(ROOT / "reports/external-national-2026-10-03/cohort-2023.csv").set_index("entity_id").reindex(panel.ids)
    bridge = pd.read_csv(ROOT / "reports/external-v4/municipality_bridge.csv").set_index("entity_id").reindex(panel.ids)
    if cohort.index.has_duplicates or bridge.index.has_duplicates:
        raise ValueError("Duplicate external entity key")
    regions = bridge.region_code.to_numpy(int)
    if np.isnan(bridge.region_code).any() or np.any(regions != cohort.region_code.to_numpy(int)):
        raise ValueError("External registry region IDs disagree")
    features = {"total_rub": np.mean(panel.totals[:, :12], axis=1)}
    for j, category in enumerate(P.CATEGORIES):
        features[f"{category}_pct"] = np.mean(100 * panel.shares[:, :12, j], axis=1)
    for name in ("wage_total", "population_total"):
        features[name] = cohort[name].to_numpy(float)
    features["market_access"] = bridge.market_access.to_numpy(float)
    denominator = cohort.headcount_total.to_numpy(float)
    for feature, letters in (("agriculture_share", "A"), ("manufacturing_share", "C"), ("mining_share", "B"), ("public_share", "OPQ")):
        parts = cohort[[f"headcount_{c}" for c in letters]].to_numpy(float)
        observed = np.all(np.isfinite(parts) & (parts >= 0), axis=1)
        numerator = np.where(observed, np.sum(parts, axis=1), np.nan)
        features[feature] = ratio(numerator, denominator)
    for feature in FEATURES:
        x = features[feature]
        if np.any(np.isfinite(x) & (x < 0)):
            raise ValueError(f"Negative original-scale feature: {feature}")
    x23, centers = frozen_x23(panel, ROOT / "reports/v1.2/model.npz")
    if not np.allclose(np.array([x23[group == g].mean(axis=0) for g in GROUPS]), centers, atol=1e-10, rtol=1e-10):
        raise ValueError("Reconstructed frozen profiles do not reproduce group centers")
    return panel, labels, cohort, bridge, group, regions, features, x23, centers


def summarize_mirkin(values, group):
    rows = []
    for feature, x in values.items():
        global_mean = finite_mean(x)
        for g in GROUPS:
            observed = x[group == g]
            mean = finite_mean(observed)
            rows.append({"group": g, "feature": feature, "n_group": int(np.isfinite(observed).sum()), "n_global": int(np.isfinite(x).sum()), "mean_group": mean, "mean_global": global_mean, "relative_deviation": float(deviations(mean, global_mean))})
    return rows


def mirkin(panel, group, regions, features, spec):
    rows = [{"entity_id": str(entity), "group": int(group[i]), "feature": f, "value": float(features[f][i])} for i, entity in enumerate(panel.ids) for f in FEATURES]
    save_csv(OUT / "mirkin-input.csv", rows)
    summary = summarize_mirkin(features, group)
    save_csv(OUT / "mirkin-summary.csv", summary)
    extra, thresholds = [], []
    for r in summary:
        f, g = r["feature"], r["group"]
        threshold = 1 if f.endswith("_pct") else 0.01 if f.endswith("_share") else spec["mirkin"]["small_mean_flags"][f]
        observed = (group == g) & np.isfinite(features[f])
        extra.append({**r, "N_group": int(np.sum(group == g)), "N_global": len(group), "coverage_group": r["n_group"] / np.sum(group == g), "coverage_global": r["n_global"] / len(group), "regions_group": len(np.unique(regions[observed])), "absolute_difference": r["mean_group"] - r["mean_global"], "small_denominator": bool(math.isfinite(r["mean_global"]) and abs(r["mean_global"]) < threshold), "ratio_over_one_count": int(np.sum(observed & (features[f] > 1))) if f.endswith("_share") else 0, "year": 2024 if f == "market_access" else 2023, "coverage_below_half": r["n_group"] < 0.5 * np.sum(group == g)})
        for t in spec["mirkin"]["sensitivity_thresholds"]:
            thresholds.append({"group":g,"feature":f,"threshold":t,"relative_deviation":r["relative_deviation"],"display":bool(math.isfinite(r["relative_deviation"]) and abs(r["relative_deviation"]) >= t)})
    save_csv(OUT / "mirkin/coverage.csv", extra)
    save_csv(OUT / "mirkin/thresholds.csv", thresholds)
    region_ids = np.unique(regions)
    values = np.column_stack([features[f] for f in FEATURES])
    sums = np.zeros((len(region_ids), 4, len(FEATURES)))
    counts = np.zeros_like(sums)
    coverage, descriptions = [], []
    for ri, region in enumerate(region_ids):
        for g in GROUPS:
            indices = (regions == region) & (group == g)
            part = values[indices]
            sums[ri, g] = np.nansum(part, axis=0)
            counts[ri, g] = np.isfinite(part).sum(axis=0)
            for j, f in enumerate(FEATURES):
                coverage.append({"region_code":int(region),"group":g,"feature":f,"N_group_region":int(indices.sum()),"n_observed":int(counts[ri,g,j])})
                n_global = int(np.isfinite(values[regions == region, j]).sum())
                if n_global >= 30 and counts[ri,g,j] >= 10:
                    mean = sums[ri,g,j] / counts[ri,g,j]
                    global_mean = finite_mean(values[regions == region,j])
                    descriptions.append({"region_code":int(region),"group":g,"feature":f,"n_group":int(counts[ri,g,j]),"n_global":n_global,"mean_group":mean,"mean_global":global_mean,"relative_deviation":float(deviations(mean,global_mean))})
    save_csv(OUT / "mirkin/coverage-by-region.csv", coverage)
    save_csv(OUT / "mirkin/regional-descriptions.csv", descriptions)
    all_s, all_n = sums.sum(axis=0), counts.sum(axis=0)
    loo = []
    for ri, region in enumerate(region_ids):
        kept_s, kept_n = all_s-sums[ri], all_n-counts[ri]
        gm = np.divide(kept_s,kept_n,out=np.full_like(kept_s,np.nan),where=kept_n>0)
        total_n = kept_n.sum(axis=0)
        global_m = np.divide(kept_s.sum(axis=0),total_n,out=np.full(len(FEATURES),np.nan),where=total_n>0)
        ds = deviations(gm,global_m)
        for g in GROUPS:
            for j,f in enumerate(FEATURES):
                loo.append({"excluded_region_code":int(region),"group":g,"feature":f,"n_group":int(kept_n[g,j]),"n_global":int(total_n[j]),"relative_deviation":ds[g,j]})
    save_csv(OUT / "mirkin/leave-one-region-out.csv", loo)
    loo_frame = pd.DataFrame(loo)
    stability = []
    for r in summary:
        ds = loo_frame.loc[(loo_frame.group == r['group']) & (loo_frame.feature == r['feature']), 'relative_deviation'].to_numpy()
        good = ds[np.isfinite(ds)]
        d = r['relative_deviation']
        for t in spec['mirkin']['sensitivity_thresholds']:
            stability.append({'group':r['group'],'feature':r['feature'],'threshold':t,'valid_regions_excluded':len(good),'invalid_regions_excluded':len(ds)-len(good),'minimum_d':min(good) if len(good) else np.nan,'maximum_d':max(good) if len(good) else np.nan,'sign_changes':int(np.sum(np.sign(good) != np.sign(d))) if math.isfinite(d) else None,'threshold_retention_fraction':float(np.mean(np.abs(good)>=t)) if len(good) else np.nan})
    save_csv(OUT / "mirkin/stability.csv", stability)
    rng = np.random.default_rng(1729)
    weights = np.array([np.bincount(draw,minlength=len(region_ids)) for draw in rng.integers(0,len(region_ids),size=(2000,len(region_ids)))])
    bs = np.einsum('br,rgf->bgf',weights,sums)
    bn = np.einsum('br,rgf->bgf',weights,counts)
    bm = np.divide(bs,bn,out=np.full_like(bs,np.nan),where=bn>0)
    bglobal_n = bn.sum(axis=1)
    bglobal = np.divide(bs.sum(axis=1),bglobal_n,out=np.full_like(bglobal_n,np.nan),where=bglobal_n>0)
    bd = deviations(bm,bglobal[:,None,:])
    intervals, equal_regions = [], []
    region_global_n = counts.sum(axis=1)
    region_global_m = np.divide(sums.sum(axis=1),region_global_n,out=np.full_like(region_global_n,np.nan),where=region_global_n>0)
    region_group_m = np.divide(sums,counts,out=np.full_like(sums,np.nan),where=counts>0)
    for g in GROUPS:
        for j, f in enumerate(FEATURES):
            boot = bd[:,g,j]
            good = boot[np.isfinite(boot)]
            intervals.append({'group':g,'feature':f,'relative_deviation_lower95':float(np.quantile(good,0.025)) if len(good) else np.nan,'relative_deviation_upper95':float(np.quantile(good,0.975)) if len(good) else np.nan,'valid_draws':len(good),'invalid_draws':2000-len(good),'description_only':True})
            group_m = finite_mean(region_group_m[:,g,j])
            global_m = finite_mean(region_global_m[:,j])
            equal_regions.append({'group':g,'feature':f,'mean_group_equal_observed_regions':group_m,'mean_global_equal_observed_regions':global_m,'relative_deviation':float(deviations(group_m,global_m)),'group_regions':int(np.isfinite(region_group_m[:,g,j]).sum()),'global_regions':int(np.isfinite(region_global_m[:,j]).sum()),'different_weighting_than_primary':True})
    save_csv(OUT / "mirkin/bootstrap-intervals.csv", intervals)
    save_csv(OUT / "mirkin/equal-region-weighting.csv", equal_regions)
    save_json(OUT / "mirkin/source-and-units.json", {'source':'https://link.springer.com/article/10.1134/S1064562424602002','equations':'12–13','method_license':'CC BY 4.0','feature_units':{'total_rub':'arithmetic mean monthly modeled cashless spending, rubles; already per resident estimate','*_pct':'percent of modeled category / modeled total; not asserted disjoint or additive','wage_total':'2023 mean organization wage, rubles; excludes small enterprises','population_total':'2023 source population on January1; municipal total, not city population','market_access':'2024 points0–1000; retrospective','*_share':'observed organization-sector headcount / observed organization total'},'external_origin':'Rosstat BД ПМО, distributed by Если быть точным; pinned cohort and join audit documented in reports/external-national-2026-10-03','limitations':['fixed joint labels are not independent economic validation','feature-specific observed samples and regional dependence','30% is a display threshold','no other category reconstructed from five dependent intensities','same global feature denominator including examined group']})
    status('mirkin','COMPLETED',{'groups':[338,580,971,127],'features':list(FEATURES),'input_rows':len(rows),'summary_rows':len(summary),'formula':'(group arithmetic mean - overall arithmetic mean)/overall arithmetic mean','root_csv_sha256':{'mirkin-input.csv':digest(OUT/'mirkin-input.csv'),'mirkin-summary.csv':digest(OUT/'mirkin-summary.csv')}})


def within_regions(array, regions):
    x = np.asarray(array,float).copy()
    for region in np.unique(regions):
        ix = regions == region
        x[ix] -= x[ix].mean(axis=0)
    return x


def residual_ss(x,y):
    coefficients, _, rank, _ = np.linalg.lstsq(x,y,rcond=None)
    residual = y-x@coefficients
    return float(residual@residual), int(rank)


def partial_r2(reduced, full):
    if reduced <= 1e-12:
        return np.nan
    delta=(reduced-full)/reduced
    if delta < -1e-8:
        raise ValueError('Nested model increased residual sum of squares')
    return max(0.0,delta)


def conditional_description(cohort, bridge, group, regions, features, x23):
    pop=features['population_total']
    base=np.column_stack([np.log(features['total_rub']),np.where(pop>0,np.log(np.maximum(pop,1)),np.nan)])
    municipal_type=bridge.municipal_district_type.fillna('').to_numpy(str)
    dummy=pd.get_dummies(municipal_type,drop_first=True,dtype=float).to_numpy()
    group_dummy=np.column_stack([group==g for g in (1,2,3)]).astype(float)
    targets={'log_wage_total_2023':np.where(features['wage_total']>0,np.log(np.maximum(features['wage_total'],1)),np.nan),'log1p_market_access_2024':np.log1p(features['market_access']),**{f:features[f] for f in FEATURES if f.endswith('_share')}}
    scores,coverage,influence=[],[],[]
    for target,y in targets.items():
        ok=np.isfinite(y)&np.isfinite(base).all(axis=1)&np.isfinite(x23[:,:5]).all(axis=1)&(municipal_type!='')
        r=regions[ok]
        b=np.column_stack([base[ok],dummy[ok]])
        s=np.std(b,axis=0)
        b=np.divide(b-b.mean(axis=0),s,out=np.zeros_like(b),where=s>0)
        matrices=[b,np.column_stack([b,group_dummy[ok]]),np.column_stack([b,x23[ok,:5]]),np.column_stack([b,x23[ok,:5],group_dummy[ok]])]
        matrices=[within_regions(m,r) for m in matrices]
        centered_y=within_regions(y[ok],r)
        fit=[residual_ss(m,centered_y) for m in matrices]
        for j,name in enumerate(('M0','M1','M2','M3')):
            scores.append({'target':target,'model':name,'n_entities':int(ok.sum()),'n_regions':len(np.unique(r)),'residual_ss':fit[j][0],'within_region_rank':fit[j][1],'within_region_columns':matrices[j].shape[1],'partial_R2_vs_reduced':partial_r2(fit[j-1][0],fit[j][0]) if j in (1,3) else np.nan,'formal_p_value':np.nan,'descriptive_only':True})
        for region in np.unique(regions):
            for g in GROUPS:
                part=(regions==region)&(group==g)
                coverage.append({'target':target,'region_code':int(region),'group':g,'N_panel':int(part.sum()),'n_complete':int((part&ok).sum())})
        for region in np.unique(r):
            keep=r!=region
            fitted=[residual_ss(m[keep],centered_y[keep])[0] for m in matrices]
            for full,reduced in ((1,0),(3,2)):
                influence.append({'target':target,'excluded_region_code':int(region),'contrast':f'M{full}-M{reduced}','partial_R2':partial_r2(fitted[reduced],fitted[full]),'n_entities':int(keep.sum())})
    save_csv(OUT/'economic_cases/controlled-description.csv',scores)
    save_csv(OUT/'economic_cases/controlled-coverage.csv',coverage)
    save_csv(OUT/'economic_cases/controlled-leave-one-region-out.csv',influence)


def mobility(panel):
    source=pd.read_csv(OUT/'mobility/source-names-2024.csv',dtype=str)
    registry=pd.read_csv(OUT/'mobility/registry-2024.csv',dtype=str).fillna('')
    known=set(panel.ids)
    audit=[]
    for _,r in source.iterrows():
        candidates=registry[registry.municipal_district_name==r.ref_area]
        ids=sorted(set(candidates.territory_id),key=lambda v:int(v))
        decision='UNMATCHED' if not ids else 'AMBIGUOUS' if len(candidates)>1 else 'EXACT_NAME_CANDIDATE_NOT_APPROVED'
        candidate=candidates.iloc[0] if len(candidates)==1 else None
        audit.append({'source_indicator_id':r.indicator_id,'source_ref_area':r.ref_area,'source_period':r.period,'source_region':'UNVERIFIED','candidate_count':len(ids),'candidate_territory_ids':'|'.join(ids),'registry_territory_id':ids[0] if len(ids)==1 else '', 'candidate_entity_id':f'tid_{ids[0]}' if len(ids)==1 else '', 'candidate_in_panel':f'tid_{ids[0]}' in known if len(ids)==1 else False,'registry_oktmo':candidate.oktmo if candidate is not None else '', 'registry_region_code':candidate.region_code if candidate is not None else '', 'registry_sheet_rows':'|'.join(candidates.registry_sheet_row),'registry_year_from':candidate.year_from if candidate is not None else '', 'registry_year_to':candidate.year_to if candidate is not None else '', 'match_rule':'exact full name; all active2024 registry rows; no fuzzy or guessed region','evidence_url':'https://sberindex.ru/api/researches/v1/dataset-borders-and-changes-of-municipalities','decision':decision,'boundary_and_publisher_id_verified':False})
    frame=pd.DataFrame(audit)
    save_csv(OUT/'mobility/join-audit.csv',frame)
    save_csv(OUT/'mobility/ambiguous.csv',frame[frame.decision=='AMBIGUOUS'])
    save_csv(OUT/'mobility/unmatched.csv',frame[frame.decision=='UNMATCHED'])
    save_json(OUT/'mobility/acquisition.json',{'dataset_url':'https://sberindex.ru/api/dataset/v1/indeks-mobilnosti','card_url':'https://sberindex.ru/ru/dashboards/indeks-mobilnosti','http200_open_get':True,'2024_rows':297,'source_unit':'км','source_frequency':'Год','period_literal':'2024-12-30T21:00:00.000Z','calendar_and_method_status':'UNVERIFIED','license_status':'UNVERIFIED','publisher_municipal_id':'ABSENT; indicator_id is a series ID','attempts':[{'url':'https://sberindex.ru/api/dataset/v1/meta/indeks-mobilnosti','result':'403 Forbidden SOWA; no retry or bypass'},{'url':'https://sberindex.ru/ru/dashboards/indeks-mobilnosti','result':'Firecrawl HTTP200 contained only cookie notice; search excerpt said SZFO and January–September averages; not full method'},{'url':'https://sberuniversity.ru/articles/vozmozhnosti-ispolzovaniya-dannykh-sberindeksa-v-analitike-i-prinyatii-resheniy/','result':'403; no retry or bypass'}],'no_mobility_values_in_runner':True,'why_no_valid_test':'Definition and license unverified. Exact registry-name candidates do not establish publisher geographic IDs or actual boundary identity. No candidate is silently accepted.'})
    status('mobility','UNAVAILABLE',{'reason':'Не подтверждены методика и лицензия. Источник не публикует ID МО; проверены точные имена, но нет утверждённого перехода географии и границ. Статистика мобильности не выполнялась.','source_names':len(frame),'decisions':frame.decision.value_counts().to_dict(),'approved_matches':0,'outcome_models_fitted':0})


def classify_proxy(pop, municipal_type, regions, large=500000,middle=30000,expand=False,add_adygea=False,industry=None,industry_threshold=None):
    pop=np.asarray(pop,float)
    regions=np.asarray(regions,int)
    result=np.full(len(pop),'UNKNOWN',dtype=object)
    geography=np.isin(regions,[4,5,6,7,9,15,17,20]+([1] if add_adygea else []))
    types=np.asarray(municipal_type,str)
    known=np.isfinite(pop)&(pop>0)&(types!='')
    urban=np.isin(types,['городской округ','город федерального значения']+(['муниципальный округ'] if expand else []))
    result[known]='Proxy_III'
    size_ii=known&urban&(pop>=middle)&(pop<large)
    result[size_ii]='Proxy_II'
    if industry_threshold is not None:
        result[size_ii&~np.isfinite(industry)]='UNKNOWN'
        result[size_ii&np.isfinite(industry)&(industry<industry_threshold)]='Proxy_III'
    result[known&urban&(pop>=large)]='Proxy_I'
    result[geography]='Proxy_IV'
    return result


def association(group,proxy,mask):
    good=mask&(proxy!='UNKNOWN')
    n=int(good.sum())
    if not n:
        return {'n_known':0,'n_unknown':int(mask.sum()),'cramer_V':np.nan,'ARI':np.nan}
    table=np.array([[np.sum(good&(group==g)&(proxy==p)) for p in PROXIES[:-1]] for g in GROUPS],dtype=float)
    table=table[table.sum(axis=1)>0][:,table.sum(axis=0)>0]
    denominator=min(table.shape)-1
    if denominator>0:
        expected=table.sum(axis=1)[:,None]*table.sum(axis=0)[None,:]/n
        chi=float(np.sum((table-expected)**2/expected))
        v=math.sqrt(chi/n/denominator)
    else:
        v=np.nan
    return {'n_known':n,'n_unknown':int(np.sum(mask&(proxy=='UNKNOWN'))),'cramer_V':v,'ARI':float(adjusted_rand_score(group[good],proxy[good]))}


def four_russias(panel,cohort,bridge,group,regions,features):
    pop=features['population_total']
    typ=bridge.municipal_district_type.fillna('').to_numpy(str)
    intra=bridge.intracity_moscow_petersburg.astype(str).str.lower().isin(['true','1']).to_numpy()
    parts=cohort[['headcount_B','headcount_C','headcount_D','headcount_E']].to_numpy(float)
    observed=np.isfinite(parts).all(axis=1)&(parts>=0).all(axis=1)
    industry=ratio(np.where(observed,parts.sum(axis=1),np.nan),cohort.headcount_total.to_numpy(float))
    industry[(industry>1)|~np.isfinite(industry)]=np.nan
    primary=classify_proxy(pop,typ,regions)
    assignments,contingency,metrics,drop=[],[],[],[]
    variants=[('R500-30',primary,np.ones(len(pop),bool))]
    for large in (250000,500000,1000000):
        for middle in (20000,30000):
            if (large,middle)!=(500000,30000):
                variants.append((f'R{large//1000}-{middle//1000}',classify_proxy(pop,typ,regions,large,middle),np.ones(len(pop),bool)))
    variants.extend([('R500-30-expand-municipal-okrug',classify_proxy(pop,typ,regions,expand=True),np.ones(len(pop),bool)),('R500-30-add-Adygea',classify_proxy(pop,typ,regions,add_adygea=True),np.ones(len(pop),bool)),('R500-30-exclude-intracity',primary,~intra),('R500-30-exclude-capitals',primary,~np.isin(regions,[77,78])),('R500-30-exclude-IV',primary,primary!='Proxy_IV')])
    for t in (0.2,0.3,0.4):
        variants.append((f'R500-30-industrial-{int(t*100)}',classify_proxy(pop,typ,regions,industry=industry,industry_threshold=t),np.ones(len(pop),bool)))
    for variant,proxy,mask in variants:
        scores=association(group,proxy,mask)
        metrics.append({'variant':variant,**scores,'N_included':int(mask.sum()),'population_observed_n':int(np.sum(mask&np.isfinite(pop))),'population_observed_sum':float(np.nansum(pop[mask])),'description_only':True})
        for i in np.flatnonzero(mask):
            assignments.append({'variant':variant,'entity_id':str(panel.ids[i]),'region_code':int(regions[i]),'group':int(group[i]),'proxy':proxy[i],'municipal_type':typ[i],'municipal_population_2023':pop[i],'industrial_jobs_fraction':industry[i],'intracity':bool(intra[i]),'city_population_verified':False})
        for g in GROUPS:
            for p in PROXIES:
                part=mask&(group==g)&(proxy==p)
                n=int(part.sum())
                nr=int(np.sum(mask&(group==g)))
                nc=int(np.sum(mask&(proxy==p)))
                contingency.append({'variant':variant,'group':g,'proxy':p,'n':n,'row_share':n/nr if nr else np.nan,'column_share':n/nc if nc else np.nan,'population_observed_n':int(np.sum(part&np.isfinite(pop))),'population_observed_sum':float(np.nansum(pop[part])),'population_denominator':'only included observed municipalities; not Russia population'})
        for region in np.unique(regions[mask]):
            drop.append({'variant':variant,'excluded_region_code':int(region),**association(group,proxy,mask&(regions!=region))})
    save_csv(OUT/'four_russias/assignments.csv',assignments)
    save_csv(OUT/'four_russias/contingency.csv',contingency)
    save_csv(OUT/'four_russias/sensitivity.csv',metrics)
    save_csv(OUT/'four_russias/leave-one-region-out.csv',drop)
    save_json(OUT/'four_russias/source-and-limits.json',{'source':'https://www.vedomosti.ru/opinion/articles/2011/12/30/chetyre_rossii','author':'Наталья Зубаревич','original_year':2011,'primary':'R500-30 municipal population and legal municipal type proxy','original_read':True,'interpretation':'Муниципальное приближение к исторической схеме; промышленность и социальные признаки исходного текста не восстановлены по одному размеру МО.','Proxy_I':'known municipal population >=500000 in urban legal types; municipal territory can include settlements beyond city','Proxy_II':'known urban municipal population30000–499999; industrial character unverified in primary','Proxy_III':'remaining known municipality; does not establish its rural share','Proxy_IV':'specified republic geography, takes priority over size','UNKNOWN':'unobserved municipal population or municipal type outside IV; no parent-city values invented','industrial_thresholds':'20/30/40% observed B+C+D+E organization jobs; author operationalization, not Zubarevich numeric rule','city_version_status':'UNAVAILABLE: no verified2023 city-population-to-fixed-municipality crosswalk; no parent-city population imputation','historical_exceptions':'Article names large industrial exceptions; no current2023 production claim or guessed name-to-ID crosswalk','limits':['ARI and Cramer V are association, not truth or accuracy','population coverage pertains only to matched panel and excludes unknown values','intracity excluded sensitivity avoids adding unverified parent totals','no historical national percentages transferred to2023']})
    status('four_russias','COMPLETED',{'primary':'R500-30','variants':len(variants),'municipal_proxy_not_truth':True,'city_level_version':'UNAVAILABLE','primary_metrics':metrics[0]})


def numeric_id(entity):
    return int(str(entity).split('_')[1])


def pick_third(ids,labels,x23,centers):
    g=labels.type_2023.to_numpy(int)
    adjusted=labels.type_2024.to_numpy(int)
    absolute=labels.type_2024_absolute.to_numpy(int)
    persistent=labels.persistent_change.astype(str).str.lower().eq('true').to_numpy()
    eligible=(absolute!=g)&(adjusted==g)&~persistent&~np.isin(ids,['tid_196','tid_616'])
    if eligible.any():
        return min(ids[eligible],key=numeric_id),'minimum numeric ID, absolute change only, adjusted fixed and not persistent'
    ds=np.sort(cdist(x23,centers,metric='sqeuclidean'),axis=1)
    margin=ds[:,1]-ds[:,0]
    candidates=[i for i in range(len(ids)) if ids[i] not in ('tid_196','tid_616')]
    i=min(candidates,key=lambda k:(margin[k],numeric_id(ids[k])))
    return ids[i],'fallback: minimum squared-centroid margin, numeric ID tie'


def nearest_rows(distance,pool,k=15):
    pool=np.asarray(pool,int)
    order=np.argsort(distance[:,pool],axis=1,kind='stable')[:,:k]
    return pool[order]


def economic_cases(panel,labels,bridge,group,regions,features,x23,centers,distance):
    third,rule=pick_third(panel.ids,labels,x23,centers)
    cases=['tid_196','tid_616',str(third)]
    selected,neighbors,detail=[],[],[]
    typ=bridge.municipal_district_type.fillna('').to_numpy(str)
    spend=features['total_rub']
    for case in cases:
        i=int(np.flatnonzero(panel.ids==case)[0])
        label=labels.iloc[i]
        selected.append({'entity_id':case,'name':str(panel.names[i]),'region_code':int(regions[i]),'group_2023':int(group[i]),'group_name':GROUP_NAMES[group[i]],'group_2024_adjusted':int(label.type_2024),'group_2024_absolute':int(label.type_2024_absolute),'persistent_change':str(label.persistent_change).lower()=='true','selection_rule':'fixed existing case before specification' if case in cases[:2] else rule,'interpretation':'descriptive; label and source values do not establish cause or industrial/rotational economy'})
        for year,start in ((2023,0),(2024,12)):
            attrs={'total_rub':float(panel.totals[i,start:start+12].mean()),**{f'{c}_pct':float(100*panel.shares[i,start:start+12,j].mean()) for j,c in enumerate(P.CATEGORIES)}}
            if year==2023:
                attrs.update({f:float(features[f][i]) for f in FEATURES[6:] if f!='market_access'})
            else:
                attrs['market_access']=float(features['market_access'][i])
            for f,value in attrs.items():
                detail.append({'case_entity_id':case,'feature':f,'year':year,'value':value,'source':'data/v12/panel.csv.gz' if f in FEATURES[:6] else 'reports/external-v4/municipality_bridge.csv' if f=='market_access' else 'reports/external-national-2026-10-03/cohort-2023.csv','missing':not math.isfinite(value),'aggregation':'arithmetic mean twelve months' if f in FEATURES[:6] else 'published joined annual value or observed headcount ratio'})
        candidates=np.argsort(distance[i],kind='stable')
        expense_factor=np.maximum(spend/spend[i],spend[i]/spend)
        methods=[('all_X23_15',None,candidates[:15])]
        for factor in (1.1,1.25,1.5,2.0):
            allowable=(typ==typ[i])&(expense_factor<=factor)&(np.arange(len(group))!=i)
            chosen=[j for j in candidates if allowable[j]][:15]
            methods.append(('same_legal_type_and_expense_factor',factor,chosen))
        for method,factor,chosen in methods:
            for rank,j in enumerate(chosen,1):
                neighbors.append({'case_entity_id':case,'comparison':method,'expense_factor_limit':factor,'neighbor_rank':rank,'neighbor_entity_id':str(panel.ids[j]),'neighbor_name':str(panel.names[j]),'neighbor_region_code':int(regions[j]),'same_region':bool(regions[i]==regions[j]),'neighbor_group_2023':int(group[j]),'same_group':bool(group[i]==group[j]),'distance_frozen_X23':float(distance[i,j]),'symmetric_mean_expense_factor':float(expense_factor[j]),'case_legal_type':typ[i],'neighbor_legal_type':typ[j],'neighbors_returned':len(chosen)})
    save_csv(OUT/'economic_cases/selected-cases.csv',selected)
    save_csv(OUT/'economic_cases/case-values.csv',detail)
    save_csv(OUT/'economic_cases/case-neighbors.csv',neighbors)
    save_json(OUT/'economic_cases/comparison-function.json',{'distance':'Euclidean in frozen six X23 coordinates: annual medians of the original standardized monthly five log intensities and relative spending level','formula':'sqrt(sum_j (X23_query_j-X23_candidate_j)^2)','scaler_source':'reports/v1.2/model.npz; no refit','order':'stable distance sorting in canonical lexicographic panel ID order','legal_and_expense_filter':'same municipal_district_type, max(mean_expense_j/mean_expense_i, mean_expense_i/mean_expense_j)<=factor; self excluded','factors':[1.1,1.25,1.5,2.0],'no_fillers':True,'region':'reported, not filtered for this descriptive case function','source_limitation':'selection cases chosen before new outcomes; no monetary benefit, policy effect or treatment recommendation measured'})
    return selected


def analogue_identity(panel,features,distance):
    nearest=np.argsort(distance,axis=1,kind='stable')[:,:15]
    direct=nearest_rows(distance,np.arange(len(panel.ids)),15)
    if not np.array_equal(nearest,direct) or np.any(nearest==np.arange(len(panel.ids))[:,None]):
        raise ValueError('Same-X23 nearest donor identity or self exclusion failed')
    growth=100*(panel.totals[:,12:].mean(axis=1)/features['total_rub']-1)
    predicted=np.median(growth[nearest],axis=1)
    saved=pd.read_csv(ROOT/'reports/v1.2/analogue_predictions.csv')
    saved=saved[saved.predictor=='network_analogues'].set_index('entity_id').reindex(panel.ids)
    discrepancy=np.abs(saved.predicted_growth_pct.to_numpy(float)-predicted)
    if not np.all(np.isfinite(discrepancy)) or discrepancy.max()>1e-10:
        raise ValueError('Current network_analogues prediction is not reproduced by ordinary15NN')
    save_csv(OUT/'economic_cases/identity-predictions.csv',[{'entity_id':str(panel.ids[i]),'same_X23_prediction':predicted[i],'saved_network_analogues_prediction':float(saved.predicted_growth_pct.iloc[i]),'absolute_difference':float(discrepancy[i]),'neighbors':'|'.join(map(str,panel.ids[nearest[i]]))} for i in range(len(panel.ids))])
    save_json(OUT/'economic_cases/identity.json',{'same_neighbor_function':bool(np.array_equal(nearest,direct)),'n_entities':len(panel.ids),'max_absolute_prediction_difference':float(discrepancy.max()),'k':15,'metric':'Euclidean six frozen X23 attributes','type_restricted':False,'multi_hop':False,'additional_information_claim':False,'interpretation':'Current network_analogues is exactly the same donor/predictor as ordinary same-X23 15NN. Labels and graph paths are not used.'})
    return growth


def analogue_experiment(panel,bridge,group,regions,features,distance,growth,prereg):
    unique=np.unique(regions)
    if len(unique)!=73:
        raise ValueError('Registered37/36 split requires exactly73 canonical regions')
    task=prereg['task_id']
    ordered=sorted(unique,key=lambda r:(hashlib.sha256((task+'|'+str(int(r))).encode('utf-8')).hexdigest(),int(r)))
    donor_regions=np.array(ordered[:37])
    evaluation_regions=np.array(ordered[37:])
    donors=np.flatnonzero(np.isin(regions,donor_regions))
    evaluation=np.flatnonzero(np.isin(regions,evaluation_regions))
    save_csv(OUT/'economic_cases/analogue-split.csv',[{'region_code':int(r),'role':'donor' if r in donor_regions else 'evaluation','hash_sha256':hashlib.sha256((task+'|'+str(int(r))).encode()).hexdigest(),'n_entities':int(np.sum(regions==r))} for r in ordered])
    continuous=nearest_rows(distance[evaluation],donors,15)
    restricted=continuous.copy()
    fallback=np.zeros(len(evaluation),bool)
    for g in GROUPS:
        ix=group[evaluation]==g
        pool=donors[group[donors]==g]
        if len(pool)<15:
            fallback[ix]=True
        elif ix.any():
            restricted[ix]=nearest_rows(distance[evaluation[ix]],pool,15)
    pred_cont=np.median(growth[continuous],axis=1)
    pred_type=np.median(growth[restricted],axis=1)
    error_cont=np.abs(growth[evaluation]-pred_cont)
    error_type=np.abs(growth[evaluation]-pred_type)
    predrows=[]
    for j,i in enumerate(evaluation):
        predrows.append({'entity_id':str(panel.ids[i]),'region_code':int(regions[i]),'group_2023':int(group[i]),'observed_growth_pct':growth[i],'continuous15_prediction_pct':pred_cont[j],'type15_prediction_pct':pred_type[j],'continuous15_AE_pp':error_cont[j],'type15_AE_pp':error_type[j],'type_fallback_to_continuous':bool(fallback[j]),'continuous_donors':'|'.join(map(str,panel.ids[continuous[j]])),'type_donors':'|'.join(map(str,panel.ids[restricted[j]]))})
    save_csv(OUT/'economic_cases/analogue-test-predictions.csv',predrows)
    region_rows=[]
    for r in evaluation_regions:
        ix=regions[evaluation]==r
        region_rows.append({'region_code':int(r),'n_entities':int(ix.sum()),'continuous15_MAE_pp':float(error_cont[ix].mean()),'type15_MAE_pp':float(error_type[ix].mean()),'fallback_entities':int(fallback[ix].sum())})
    save_csv(OUT/'economic_cases/analogue-test-by-region.csv',region_rows)
    rc=np.array([r['continuous15_MAE_pp'] for r in region_rows])
    rt=np.array([r['type15_MAE_pp'] for r in region_rows])
    main_cont=float(rc.mean()); main_type=float(rt.mean())
    effect=(main_cont-main_type)/main_cont if main_cont>0 else np.nan
    draws=np.random.default_rng(1729).integers(0,len(region_rows),size=(9999,len(region_rows)))
    bc=rc[draws].mean(axis=1); bt=rt[draws].mean(axis=1)
    improvements=np.divide(bc-bt,bc,out=np.full_like(bc,np.nan),where=bc>0)
    valid=improvements[np.isfinite(improvements)]
    ci=np.quantile(valid,[0.025,0.975]) if len(valid) else [np.nan,np.nan]
    save_csv(OUT/'economic_cases/analogue-test-bootstrap.csv',[{'draw':i,'continuous15_region_mean_MAE_pp':bc[i],'type15_region_mean_MAE_pp':bt[i],'relative_MAE_improvement':improvements[i],'valid':bool(np.isfinite(improvements[i]))} for i in range(9999)])
    drop=[]
    for i,r in enumerate(evaluation_regions):
        c=float(np.delete(rc,i).mean()); t=float(np.delete(rt,i).mean())
        drop.append({'excluded_region_code':int(r),'continuous15_region_mean_MAE_pp':c,'type15_region_mean_MAE_pp':t,'relative_MAE_improvement':(c-t)/c if c>0 else np.nan})
    save_csv(OUT/'economic_cases/analogue-test-leave-one-region-out.csv',drop)
    low_market=float(np.quantile(features['marketplaces_pct'],0.1))
    observed_pop=features['population_total'][np.isfinite(features['population_total'])]
    low_pop=float(np.quantile(observed_pop,0.1))
    intra=bridge.intracity_moscow_petersburg.astype(str).str.lower().isin(['true','1']).to_numpy()
    subsets={'all':np.ones(len(evaluation),bool),'low_marketplace_2023_decile':features['marketplaces_pct'][evaluation]<=low_market,'low_observed_population_2023_decile':np.isfinite(features['population_total'][evaluation])&(features['population_total'][evaluation]<=low_pop),'Moscow_SPb_intracity':intra[evaluation],**{f'group_{g}':group[evaluation]==g for g in GROUPS}}
    subrows=[]
    for name,mask in subsets.items():
        subregions=np.unique(regions[evaluation][mask])
        c=finite_mean([error_cont[mask&(regions[evaluation]==r)].mean() for r in subregions])
        t=finite_mean([error_type[mask&(regions[evaluation]==r)].mean() for r in subregions])
        subrows.append({'subset':name,'n_entities':int(mask.sum()),'n_regions':len(subregions),'continuous_region_mean_MAE_pp':c,'type_region_mean_MAE_pp':t,'relative_MAE_improvement':(c-t)/c if c>0 else np.nan,'description_only':True})
    save_csv(OUT/'economic_cases/analogue-test-subsets.csv',subrows)
    result={'prereg_sha256':PREREG_SHA,'donor_regions':37,'evaluation_regions':36,'donor_entities':len(donors),'evaluation_entities':len(evaluation),'outcome':'nominal mean-total-rub growth2024, percent','primary_weight':'equal evaluation-region weight','continuous15_region_mean_MAE_pp':main_cont,'type15_region_mean_MAE_pp':main_type,'relative_MAE_improvement':effect,'relative_MAE_improvement_lower95':float(ci[0]),'relative_MAE_improvement_upper95':float(ci[1]),'operational_criterion_lower95_above_5percent':bool(math.isfinite(ci[0]) and ci[0]>0.05),'valid_bootstrap_draws':len(valid),'invalid_bootstrap_draws':9999-len(valid),'fallback_entities':int(fallback.sum()),'low_2023_marketplace_pct_cutoff':low_market,'low_observed2023_population_cutoff':low_pop,'no_formal_p_values':True,'monetary_value_measured':False,'prospective_validation':False,'limitations':prereg['limitations']}
    save_json(OUT/'economic_cases/analogue-test-summary.json',result)
    return result


def preflight():
    if digest(SPEC_PATH)!=SPEC_SHA or digest(PREREG_PATH)!=PREREG_SHA:
        raise ValueError('The before-outcome specifications changed; do not rewrite them after results')
    spec=json.loads(SPEC_PATH.read_text(encoding='utf-8'))
    prereg=json.loads(PREREG_PATH.read_text(encoding='utf-8'))
    for name,expected in spec['input_sha256'].items():
        if digest(ROOT/name)!=expected:
            raise ValueError(f'Pinned input changed: {name}')
    return spec,prereg


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--execute',action='store_true',help='Run only after saved C and an allocated compute slot')
    mode.add_argument('--baseline-check',action='store_true',help='Check pinned2023 features and existing peer lists; no new outcomes')
    args=parser.parse_args()
    try:
        import psutil
        if sys.platform=='win32':
            psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except ImportError:
        if sys.platform=='win32':
            raise RuntimeError('psutil is required to set BelowNormal on Windows')
    started=time.time()
    if args.baseline_check:
        with threadpool_limits(limits=1):
            result=baseline_check()
        print(json.dumps(clean_json(result),ensure_ascii=False,indent=2),flush=True)
        return
    spec,prereg=preflight()
    with threadpool_limits(limits=1):
        data=annual_data()
        panel,labels,cohort,bridge,group,regions,features,x23,centers=data
        print('Pinned inputs verified; current main labels and frozen X23 reconstructed.',flush=True)
        mirkin(panel,group,regions,features,spec)
        print('Mirkin arithmetic profiles and sensitivity completed.',flush=True)
        mobility(panel)
        four_russias(panel,cohort,bridge,group,regions,features)
        print('Mobility negative conclusion and Four Russias municipal proxies completed.',flush=True)
        distance=cdist(x23,x23)
        np.fill_diagonal(distance,np.inf)
        cases=economic_cases(panel,labels,bridge,group,regions,features,x23,centers,distance)
        growth=analogue_identity(panel,features,distance)
        result=analogue_experiment(panel,bridge,group,regions,features,distance,growth,prereg)
        conditional_description(cohort,bridge,group,regions,features,x23)
        save_json(OUT/'economic_cases/run-environment.json',{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'platform':platform.platform(),'threads':1,'priority':'BelowNormal on Windows','elapsed_seconds':time.time()-started,'script_sha256':digest(Path(__file__)),'spec_sha256':SPEC_SHA,'prereg_sha256':PREREG_SHA,'input_sha256':spec['input_sha256'],'excluded_outcome_years':[2025]})
        status('economic_cases','COMPLETED',{'selected_ids':[r['entity_id'] for r in cases],'network_analogues_same_15NN':True,'analogue_test':result,'controlled_description':'common cases, region and scale controlled, no formal p-values','future_claim':False})
        print(json.dumps(clean_json({'elapsed_seconds':time.time()-started,'cases':[r['entity_id'] for r in cases],'mobility':'UNAVAILABLE','analogue_test':result}),ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
