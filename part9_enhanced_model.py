"""
Part 9 — Enhanced Hedonic Model with Real Transaction Data
===========================================================
Runs standalone; imports variables from the already-executed main notebook
environment if available, or re-creates them from scratch.

Data sources used:
  - data/raw/redfin_dc_metro.csv   : Redfin DC MSA metro-level SFR stats (2016-2026)
  - data/raw/redfin_dc_cities.csv  : Redfin city-level SFR stats for Arlington, Alexandria, etc.
  - data/processed/neighborhood_quality.py : ZIP_QUALITY + CITY_PPSF lookup tables
"""

import warnings; warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pickle
import sys, os
sys.path.insert(0, 'data/processed')

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

from neighborhood_quality import ZIP_QUALITY, CITY_PPSF

np.random.seed(42)

def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# ── 9.1  Load and summarise Redfin real data ─────────────────────────────────
print("=" * 70)
print("  9.1  Real Transaction Data (Redfin)")
print("=" * 70)

metro  = pd.read_csv('data/raw/redfin_dc_metro.csv',  low_memory=False)
cities = pd.read_csv('data/raw/redfin_dc_cities.csv', low_memory=False)

for df_r in [metro, cities]:
    for col in ['MEDIAN_SALE_PRICE', 'MEDIAN_PPSF', 'HOMES_SOLD',
                'MEDIAN_DOM', 'AVG_SALE_TO_LIST', 'SOLD_ABOVE_LIST']:
        df_r[col] = pd.to_numeric(df_r[col], errors='coerce')
    df_r['PERIOD_BEGIN'] = pd.to_datetime(df_r['PERIOD_BEGIN'])
    df_r['IS_SEASONALLY_ADJUSTED'] = df_r['IS_SEASONALLY_ADJUSTED'].astype(bool)

# City-level SFR PPSF (2022-2025 median, non-SA)
sfr_city = (cities[
    (cities['PROPERTY_TYPE'] == 'Single Family Residential') &
    (cities['PERIOD_BEGIN']  >= '2022-01-01') &
    (~cities['IS_SEASONALLY_ADJUSTED'])
].groupby('CITY')[['MEDIAN_PPSF', 'MEDIAN_SALE_PRICE', 'MEDIAN_DOM',
                    'AVG_SALE_TO_LIST', 'SOLD_ABOVE_LIST']].median().round(2))

print("\nReal Redfin SFR Stats by City (2022-2025 median, non-SA):")
print(sfr_city.rename(columns={
    'MEDIAN_PPSF':        'PPSF',
    'MEDIAN_SALE_PRICE':  'Median Price',
    'MEDIAN_DOM':         'DOM',
    'AVG_SALE_TO_LIST':   'Sale/List',
    'SOLD_ABOVE_LIST':    '%AboveList',
}).to_string())

# Metro-level SFR trend
sfr_metro = (metro[
    (metro['PROPERTY_TYPE'] == 'Single Family Residential') &
    (~metro['IS_SEASONALLY_ADJUSTED'])
].sort_values('PERIOD_BEGIN').copy())

latest = sfr_metro.dropna(subset=['MEDIAN_SALE_PRICE', 'MEDIAN_PPSF']).iloc[-1]
print(f"\nDC MSA metro (latest {latest['PERIOD_BEGIN'].strftime('%Y-%m')}):")
print(f"  Median price : ${latest['MEDIAN_SALE_PRICE']:,.0f}")
print(f"  Median PPSF  : ${latest['MEDIAN_PPSF']:.0f}")
print(f"  Median DOM   : {latest['MEDIAN_DOM']:.0f}")
print(f"  Sale/List    : {latest['AVG_SALE_TO_LIST']:.3f}")

# ── 9.2  Plot real market data ───────────────────────────────────────────────
print("\n  9.2  Plotting real market data...")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Real Redfin Market Data — DC Metro SFR', fontsize=12, fontweight='bold')

ax = axes[0]
ppsf_vals = pd.Series({c: CITY_PPSF[c] for c in sorted(CITY_PPSF, key=CITY_PPSF.get)})
colors_bar = ['coral' if c == 'Arlington' else 'steelblue' for c in ppsf_vals.index]
ax.barh(ppsf_vals.index, ppsf_vals.values, color=colors_bar)
ax.axvline(sfr_metro['MEDIAN_PPSF'].median(), color='green', linestyle='--',
           linewidth=1.5, label=f"DC MSA metro median ${sfr_metro['MEDIAN_PPSF'].median():.0f}")
ax.set_xlabel('Median PPSF ($)')
ax.set_title('A. Real Redfin Median PPSF by City\n(SFR, 2022–2025, Redfin)')
ax.legend(fontsize=9)
for i, (city, v) in enumerate(ppsf_vals.items()):
    ax.text(v + 3, i, f'${v:,.0f}', va='center', fontsize=9)

ax = axes[1]
ax.plot(sfr_metro['PERIOD_BEGIN'], sfr_metro['MEDIAN_PPSF'],
        color='steelblue', linewidth=2, label='DC MSA median PPSF')
ax.fill_between(sfr_metro['PERIOD_BEGIN'], sfr_metro['MEDIAN_PPSF'],
                alpha=0.15, color='steelblue')
ax.axhline(CITY_PPSF['Arlington'], color='coral', linestyle='--', linewidth=1.5,
           label=f"Arlington SFR PPSF ${CITY_PPSF['Arlington']:.0f}")
ax.set_xlabel('Date'); ax.set_ylabel('Median PPSF ($)')
ax.set_title('B. DC MSA SFR PPSF Trend (Redfin)\nvs Arlington City Level')
ax.legend(fontsize=9)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('$%d'))
plt.tight_layout()
plt.savefig('figures/11_redfin_real_ppsf.png', dpi=130, bbox_inches='tight')
plt.show()
print("  Figure saved: figures/11_redfin_real_ppsf.png")

# ── 9.3  Rebuild synthetic data with real PPSF anchors ───────────────────────
print("\n" + "=" * 70)
print("  9.3  Rebuild Synthetic Data (real PPSF anchors + new features)")
print("=" * 70)

N   = 10_000
rng = np.random.default_rng(42)

# Property characteristics (identical seed/params → same structural sample)
sqft_living  = np.clip(rng.lognormal(7.4, 0.45, N), 700, 6000).astype(int)
sqft_lot     = np.clip(rng.lognormal(8.5, 0.55, N), 1500, 25000).astype(int)
bedrooms     = np.clip(rng.poisson(3.2, N), 1, 8)
bath_base    = 1.0 + 0.0012 * sqft_living
bathrooms    = np.clip(rng.normal(bath_base, 0.6), 1.0, 7.0).round(1)
year_built   = np.clip(rng.integers(1920, 2025, N), 1920, 2024)
age          = 2025 - year_built
grade        = np.clip(rng.normal(7.2, 1.3, N), 3, 13).round().astype(int)
condition    = np.clip(rng.normal(3.2, 0.8, N), 1, 5).round().astype(int)
garage_sqft  = np.where(rng.random(N) < 0.68,
                        np.clip(rng.normal(500, 120, N), 300, 720), 0).astype(int)
bsmt_sqft    = np.where(rng.random(N) < 0.72,
                        (sqft_living * rng.uniform(0.3, 0.9, N)).astype(int), 0)
renovated    = (rng.random(N) < 0.18).astype(int)
waterfront   = (rng.random(N) < 0.005).astype(int)

ZIPS_LIST = list(ZIP_QUALITY.keys())
zip_wts_raw = {
    '22201': 0.10, '22202': 0.07, '22203': 0.10, '22204': 0.06, '22205': 0.07,
    '22206': 0.05, '22207': 0.08, '22209': 0.04, '22301': 0.06, '22302': 0.04,
    '22304': 0.03, '22305': 0.03, '22314': 0.04, '22101': 0.05, '22102': 0.04,
    '22003': 0.04, '22030': 0.03, '22031': 0.03, '20001': 0.03, '20007': 0.03,
    '20009': 0.03, '20015': 0.03,
}
zip_wts   = np.array([zip_wts_raw[z] for z in ZIPS_LIST])
zip_wts  /= zip_wts.sum()
zip_codes = rng.choice(ZIPS_LIST, size=N, p=zip_wts)

# Neighbourhood quality arrays
city_ppsf_arr  = np.array([CITY_PPSF[ZIP_QUALITY[z]['city']]           for z in zip_codes])
metro_dist_arr = np.array([ZIP_QUALITY[z]['metro_dist_mi']              for z in zip_codes])
school_arr     = np.array([ZIP_QUALITY[z]['school_rating']              for z in zip_codes])
walk_arr       = np.array([ZIP_QUALITY[z]['walk_score']                 for z in zip_codes])
crime_arr      = np.array([ZIP_QUALITY[z]['crime_safety']               for z in zip_codes])
transit_arr    = np.array([ZIP_QUALITY[z]['transit_score']              for z in zip_codes])

def quality_multiplier(metro_mi, school, walk):
    """
    Within-city price multiplier (centred at 1.0 for average quality).
    Coefficients from DC Metro hedonic literature:
      metro_dist: -0.04 log-pts/mi  (Dewees 1976; Bajic 1983; Hess 2007)
      school:     +0.025 log-pts/pt (Black 1999; Figlio & Lucas 2004)
      walk_score: +0.003 log-pts/pt (Cortright 2009)
    """
    return np.exp(
        -0.04 * np.log1p(metro_mi)
        + 0.025 * (school - 7.0)
        + 0.003 * (walk - 70.0)
    )

qual_mult = quality_multiplier(metro_dist_arr, school_arr, walk_arr)

# Structural hedonic component (same coefficients as original Part 2)
log_price_struct = (
    5.33
    + 0.80 * np.log(sqft_living)
    + 0.03 * bedrooms
    + 0.09 * bathrooms
    + 0.10 * np.log(sqft_lot)
    - 0.004 * age + 5e-6 * age ** 2
    + 0.09 * grade
    + 0.05 * condition
    + 0.06 * (garage_sqft > 0).astype(float)
    + 1.5e-4 * bsmt_sqft
    + 0.05 * renovated
    + 0.25 * waterfront
    + rng.normal(0, 0.10, N)
)
struct_price = np.exp(log_price_struct)
struct_norm  = struct_price / struct_price.mean()   # normalise to mean=1

# Final price = city PPSF × sqft × within-city quality mult × structural norm
price_enhanced = city_ppsf_arr * sqft_living * qual_mult * struct_norm
price_enhanced = np.clip(price_enhanced, 180_000, 5_000_000)

mean_ppsf = np.mean(list(CITY_PPSF.values()))

df2 = pd.DataFrame({
    'price':         price_enhanced,
    'sqft_living':   sqft_living,
    'sqft_lot':      sqft_lot,
    'bedrooms':      bedrooms,
    'bathrooms':     bathrooms,
    'yr_built':      year_built,
    'age':           age,
    'grade':         grade,
    'condition':     condition,
    'garage_sqft':   garage_sqft,
    'bsmt_sqft':     bsmt_sqft,
    'renovated':     renovated,
    'waterfront':    waterfront,
    'zip_code':      zip_codes,
    # Real-data neighbourhood quality features
    'metro_dist_mi': metro_dist_arr,
    'school_rating': school_arr,
    'walk_score':    walk_arr,
    'crime_safety':  crime_arr,
    'transit_score': transit_arr,
    'city_ppsf':     city_ppsf_arr,
})

# Summary
arlington_zips = ['22201','22202','22203','22204','22205','22206','22207','22209']
alex_zips      = ['22301','22302','22304','22305','22314']
fx_zips        = ['22003','22030','22031']
print(f"\nEnhanced dataset: {len(df2):,} rows x {df2.shape[1]} columns")
print(f"\nCalibrated price statistics vs real Redfin data:")
print(f"  {'Metric':<35} {'Synthetic Enhanced':>20} {'Real Redfin':>15}")
print(f"  {'-'*72}")
print(f"  {'DC MSA median price':<35} ${df2['price'].median():>19,.0f}  ${latest['MEDIAN_SALE_PRICE']:>14,.0f}")
print(f"  {'DC MSA implied PPSF':<35} ${(df2['price']/df2['sqft_living']).median():>19,.0f}  ${latest['MEDIAN_PPSF']:>14.0f}")
if len(sfr_city) > 0 and 'Arlington' in sfr_city.index:
    print(f"  {'Arlington median price':<35} ${df2[df2.zip_code.isin(arlington_zips)].price.median():>19,.0f}  ${sfr_city.loc['Arlington','MEDIAN_SALE_PRICE']:>14,.0f}")
    print(f"  {'Arlington implied PPSF':<35} ${(df2[df2.zip_code.isin(arlington_zips)]['price']/df2[df2.zip_code.isin(arlington_zips)]['sqft_living']).median():>19,.0f}  ${sfr_city.loc['Arlington','MEDIAN_PPSF']:>14.0f}")
if 'Alexandria' in sfr_city.index:
    print(f"  {'Alexandria median price':<35} ${df2[df2.zip_code.isin(alex_zips)].price.median():>19,.0f}  ${sfr_city.loc['Alexandria','MEDIAN_SALE_PRICE']:>14,.0f}")
if 'Fairfax' in sfr_city.index:
    print(f"  {'Fairfax median price':<35} ${df2[df2.zip_code.isin(fx_zips)].price.median():>19,.0f}  ${sfr_city.loc['Fairfax','MEDIAN_SALE_PRICE']:>14,.0f}")

# ── 9.4  Feature engineering & model training ────────────────────────────────
print("\n" + "=" * 70)
print("  9.4  Feature Engineering & Model Training")
print("=" * 70)

df2['log_price']     = np.log(df2['price'])
df2['log_sqft']      = np.log(df2['sqft_living'])
df2['log_lot']       = np.log(df2['sqft_lot'])
df2['age_sq']        = df2['age'] ** 2
df2['bath_per_bed']  = df2['bathrooms'] / df2['bedrooms'].clip(lower=1)
df2['sqft_per_room'] = df2['sqft_living'] / (df2['bedrooms'] + df2['bathrooms']).clip(lower=1)
df2['has_garage']    = (df2['garage_sqft'] > 0).astype(int)
df2['has_basement']  = (df2['bsmt_sqft'] > 0).astype(int)
df2['log_garage']    = np.log1p(df2['garage_sqft'])
df2['log_bsmt']      = np.log1p(df2['bsmt_sqft'])
df2['grade_cond']    = df2['grade'] * df2['condition']
df2['size_grade']    = df2['log_sqft'] * df2['grade']
df2['zip_premium']   = df2['city_ppsf'] / mean_ppsf - 1.0
df2['log_metro_dist']= np.log1p(df2['metro_dist_mi'])
df2['school_x_size'] = df2['school_rating'] * df2['log_sqft']

FEAT_ENH = [
    # Original structural features
    'log_sqft', 'log_lot', 'bedrooms', 'bathrooms', 'bath_per_bed', 'sqft_per_room',
    'age', 'age_sq', 'grade', 'condition', 'has_garage', 'log_garage',
    'has_basement', 'log_bsmt', 'renovated', 'waterfront',
    # Original location proxy (now calibrated to real city PPSF)
    'zip_premium',
    # ── NEW: real-data neighbourhood quality features ──
    'log_metro_dist',    # Metro accessibility (key in DC market)
    'school_rating',     # School quality (major driver in Arlington)
    'walk_score',        # Walkability
    'crime_safety',      # Safety
    # Interactions
    'grade_cond', 'size_grade', 'school_x_size',
]

X2     = df2[FEAT_ENH].values
y2     = df2['log_price'].values
y2_raw = df2['price'].values

X2_tr, X2_te, y2_tr, y2_te, yr2_tr, yr2_te = train_test_split(
    X2, y2, y2_raw, test_size=0.20, random_state=42
)
sc2      = RobustScaler()
X2_tr_s  = sc2.fit_transform(X2_tr)
X2_te_s  = sc2.transform(X2_te)

print(f"\nTraining:  {X2_tr.shape[0]:,} samples  |  {len(FEAT_ENH)} features")
print(f"Test:      {X2_te.shape[0]:,} samples")
print(f"New features vs original: log_metro_dist, school_rating, walk_score, crime_safety, school_x_size")

MODELS_ENH = {
    'OLS Linear':        LinearRegression(),
    'Ridge (α=1)':       Ridge(alpha=1.0),
    'Lasso (α=0.001)':   Lasso(alpha=0.001, max_iter=5000),
    'ElasticNet':        ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=5000),
    'Random Forest':     RandomForestRegressor(n_estimators=300, max_depth=12,
                                               min_samples_leaf=5, n_jobs=-1, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=400, learning_rate=0.05,
                                                    max_depth=5, subsample=0.8,
                                                    min_samples_leaf=10, random_state=42),
    'XGBoost':           xgb.XGBRegressor(n_estimators=400, learning_rate=0.05,
                                           max_depth=5, subsample=0.8, colsample_bytree=0.8,
                                           reg_alpha=0.1, eval_metric='rmse',
                                           random_state=42, verbosity=0),
}

cv2 = KFold(n_splits=5, shuffle=True, random_state=42)
results2 = {}

print()
for name, model in MODELS_ENH.items():
    use_scaled = name not in ('Random Forest', 'Gradient Boosting', 'XGBoost')
    Xtr = X2_tr_s if use_scaled else X2_tr
    Xte = X2_te_s if use_scaled else X2_te
    cvs = cross_val_score(model, Xtr, y2_tr, cv=cv2,
                          scoring='neg_root_mean_squared_error', n_jobs=-1)
    model.fit(Xtr, y2_tr)
    lp  = model.predict(Xte)
    yp  = np.exp(lp)
    results2[name] = {
        'model': model, 'use_scaled': use_scaled,
        'cv_rmse_log': -cvs.mean(),
        'y_pred':      yp,
        'test_rmse':   np.sqrt(mean_squared_error(yr2_te, yp)),
        'test_mae':    mean_absolute_error(yr2_te, yp),
        'test_r2':     r2_score(np.log(yr2_te), lp),
        'test_mape':   mape(yr2_te, yp),
    }
    print(f"  {name:<25}  MAPE={results2[name]['test_mape']:.2f}%  R²={results2[name]['test_r2']:.4f}")

best2 = min(results2, key=lambda n: results2[n]['test_mape'])
print(f"\nBest enhanced model: {best2}  (MAPE={results2[best2]['test_mape']:.2f}%)")

# ── 9.5  Subject property prediction ─────────────────────────────────────────
print("\n" + "=" * 70)
print("  9.5  Enhanced Prediction — 30 N Highland St, Arlington VA 22201")
print("=" * 70)

zip_s = '22201'
zq    = ZIP_QUALITY[zip_s]
z_ppsf= CITY_PPSF[zq['city']]
z_prem= z_ppsf / mean_ppsf - 1.0

# Original model predictions (from Part 6, saved in model bundle)
try:
    with open(f"models/best_model_Ridge_(α=1).pkl", 'rb') as f_pkl:
        bundle = pickle.load(f_pkl)
    orig_model  = bundle['model']
    orig_scaler = bundle['scaler']
    orig_features = bundle.get('feature_cols', [])
    # Build original feature vector (19 features, no neighbourhood quality)
    orig_zip_meta = bundle.get('zip_meta', {})
    orig_zip_prem = orig_zip_meta.get(zip_s, {}).get('premium', 0.28)
    a_subj = 2025 - 1936
    X_orig = np.array([[
        np.log(1960), np.log(5965), 3, 4.5, 4.5/3, 1960/(3+4.5),
        a_subj, a_subj**2, 8, 4, 0, 0.0, 1, np.log1p(980),
        1, 0, orig_zip_prem,
        8*4, np.log(1960)*8,
    ]])
    X_orig_s = orig_scaler.transform(X_orig)
    orig_pred = np.exp(orig_model.predict(X_orig_s)[0])
    print(f"\n  Original model (Ridge, synthetic-only): ${orig_pred:,.0f}")
except Exception as e:
    orig_pred = 1_058_224   # known from notebook output
    print(f"  Original model (Ridge, synthetic-only): ${orig_pred:,.0f}  [cached]")

# Build enhanced feature vector (23 features)
a2 = 2025 - 1936
X_subj2 = np.array([[
    np.log(1960), np.log(5965), 3, 4.5, 4.5/3, 1960/(3+4.5),
    a2, a2**2, 8, 4, 0, 0.0, 1, np.log1p(980), 1, 0,
    z_prem,
    np.log1p(zq['metro_dist_mi']),
    zq['school_rating'],
    zq['walk_score'],
    zq['crime_safety'],
    8 * 4,
    np.log(1960) * 8,
    zq['school_rating'] * np.log(1960),
]])

print(f"\n  Neighbourhood features for 22201 (Lyon Village/Courthouse):")
print(f"    Courthouse Metro distance : {zq['metro_dist_mi']} mi  →  log={np.log1p(zq['metro_dist_mi']):.3f}")
print(f"    School rating             : {zq['school_rating']} / 10")
print(f"    Walk score                : {zq['walk_score']} / 100")
print(f"    Crime safety              : {zq['crime_safety']} / 10")
print(f"    City PPSF (Arlington)     : ${z_ppsf:,.0f} / sqft  (real Redfin)")
print(f"    Implied 1960 sqft × PPSF : ${1960*z_ppsf:,.0f}")
print()

enh_preds = {}
print(f"  {'Model':<25}  {'Original':>12}  {'Enhanced':>12}  {'Change':>10}  {'vs Assess.':>12}")
print(f"  {'-'*76}")
ext_assess = 1_345_100

for name, v in results2.items():
    Xs2      = sc2.transform(X_subj2) if v['use_scaled'] else X_subj2
    lp2      = v['model'].predict(Xs2)[0]
    enh_p    = np.exp(lp2)
    enh_preds[name] = enh_p
    chg      = (enh_p / orig_pred - 1) * 100
    vs_ref   = (enh_p / ext_assess - 1) * 100
    marker   = "  ← BEST" if name == best2 else ""
    print(f"  {name:<25}  ${orig_pred:>11,.0f}  ${enh_p:>11,.0f}  {chg:>+9.1f}%  {vs_ref:>+11.1f}%{marker}")

tree_enh = np.mean([enh_preds[n] for n in ['Random Forest', 'Gradient Boosting', 'XGBoost']])
print(f"  {'Tree Ensemble (3-model)':<25}  ${orig_pred:>11,.0f}  ${tree_enh:>11,.0f}  "
      f"{(tree_enh/orig_pred-1)*100:>+9.1f}%  {(tree_enh/ext_assess-1)*100:>+11.1f}%")

# ── 9.6  MAPE comparison ──────────────────────────────────────────────────────
# Load original model MAPE from known notebook results
orig_mapes = {
    'OLS Linear':        10.42,
    'Ridge (α=1)':        9.83,
    'Lasso (α=0.001)':   10.71,
    'ElasticNet':        11.39,
    'Random Forest':     10.24,
    'Gradient Boosting':  9.96,
    'XGBoost':           10.01,
}

print("\n" + "=" * 70)
print("  Model Performance: Original vs Enhanced")
print("=" * 70)
print(f"  {'Model':<25}  {'Orig MAPE':>10}  {'Enh MAPE':>10}  {'Improvement':>12}  {'Enh R²':>8}")
print(f"  {'-'*72}")
for name in results2:
    om = orig_mapes.get(name, np.nan)
    em = results2[name]['test_mape']
    imp = om - em
    r2  = results2[name]['test_r2']
    marker = "  ← BEST" if name == best2 else ""
    print(f"  {name:<25}  {om:>9.2f}%  {em:>9.2f}%  {imp:>+11.2f}pp  {r2:>7.4f}{marker}")

# ── 9.7  Figures ─────────────────────────────────────────────────────────────
print("\n  9.7  Generating comparison figures...")

fig, axes = plt.subplots(1, 3, figsize=(17, 6))
fig.suptitle(
    "30 N Highland St, Arlington VA — Enhanced Hedonic Model\n"
    "Real Redfin PPSF Anchors + Metro Proximity + Schools + Walkability",
    fontsize=12, fontweight='bold'
)

model_names_list = list(enh_preds.keys())
orig_vals_list   = [orig_pred] * len(model_names_list)
enh_vals_list    = [enh_preds[n] for n in model_names_list]

# Panel A: Before vs After predictions
ax = axes[0]
x = np.arange(len(model_names_list))
w = 0.35
ax.barh(x + w/2, [v/1e6 for v in orig_vals_list], w, color='steelblue', alpha=0.85, label='Original')
ax.barh(x - w/2, [v/1e6 for v in enh_vals_list],  w, color='coral',     alpha=0.85, label='Enhanced')
ax.axvline(ext_assess/1e6,    color='green',  linestyle='--', linewidth=1.5, label='Arlington Assess. $1.345M')
ax.axvline(1_452_385/1e6,     color='purple', linestyle=':',  linewidth=1.5, label='Redfin AVM $1.452M')
ax.set_yticks(x)
ax.set_yticklabels(model_names_list, fontsize=8)
ax.set_xlabel('Estimated Value ($M)')
ax.set_title('A. Before vs After Enhancement\n(all model predictions)')
ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('$%.1fM'))
ax.legend(fontsize=7.5, loc='lower right')

# Panel B: MAPE improvement bar chart
ax = axes[1]
names_list = list(results2.keys())
om_vals    = [orig_mapes.get(n, 0) for n in names_list]
em_vals    = [results2[n]['test_mape'] for n in names_list]
x2 = np.arange(len(names_list))
w2 = 0.35
ax.bar(x2 - w2/2, om_vals, w2, color='steelblue', alpha=0.85, label='Original')
ax.bar(x2 + w2/2, em_vals, w2, color='coral',     alpha=0.85, label='Enhanced')
ax.set_xticks(x2)
ax.set_xticklabels(names_list, rotation=30, ha='right', fontsize=8)
ax.set_ylabel('Test MAPE (%)')
ax.set_title('B. Accuracy Improvement\n(MAPE on held-out test set)')
ax.legend(fontsize=9)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f%%'))

# Panel C: Full valuation reconciliation (enhanced)
ax = axes[2]
best2_val = enh_preds[best2]
all_src = {
    f'Enhanced\n{best2}': best2_val,
    'Enhanced\nTree Ensemble': tree_enh,
    'Arlington Co.\nAssessment': 1_345_100,
    'Redfin AVM': 1_452_385,
    'Zillow AVM': 1_471_100,
}
colors_r = ['coral', 'darkorange', 'seagreen', 'steelblue', 'purple']
vals_r   = [v/1e6 for v in all_src.values()]
bars     = ax.barh(list(all_src.keys()), vals_r, color=colors_r, alpha=0.85)
wts      = [0.10, 0.10, 0.35, 0.25, 0.20]
blend    = sum(w*v for w, v in zip(wts, all_src.values()))
ax.axvline(blend/1e6, color='black', linestyle='--', linewidth=2.0,
           label=f'Weighted Blend: ${blend/1e6:.2f}M')
ax.set_xlabel('Estimated Value ($M)')
ax.set_title('C. Full Valuation Reconciliation\n(30 N Highland, Enhanced)')
ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('$%.2fM'))
ax.legend(fontsize=9)
for bar, val in zip(bars, vals_r):
    ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
            f'${val:.2f}M', va='center', fontsize=9)

plt.tight_layout()
plt.savefig('figures/12_enhanced_model_comparison.png', dpi=130, bbox_inches='tight')
plt.show()
print("  Figure saved: figures/12_enhanced_model_comparison.png")

# ── 9.8  Feature importance ───────────────────────────────────────────────────
tree_m2 = results2.get('XGBoost', results2.get('Gradient Boosting'))['model']
fi2 = pd.DataFrame({'feature': FEAT_ENH, 'importance': tree_m2.feature_importances_})
fi2 = fi2.sort_values('importance', ascending=False)

print("\n  Top feature importances (enhanced XGBoost model):")
print(f"  {'Feature':<22}  {'Importance':>10}")
for _, row in fi2.head(12).iterrows():
    marker = "  ← NEW" if row['feature'] in ('log_metro_dist','school_rating',
                                               'walk_score','crime_safety','school_x_size') else ""
    print(f"  {row['feature']:<22}  {row['importance']:>10.4f}{marker}")

# Feature importance chart
fig2, ax2 = plt.subplots(figsize=(9, 6))
fi2_top = fi2.head(15).sort_values('importance')
colors_fi = ['coral' if f in ('log_metro_dist', 'school_rating', 'walk_score',
                               'crime_safety', 'school_x_size')
             else 'steelblue' for f in fi2_top['feature']]
ax2.barh(fi2_top['feature'], fi2_top['importance'], color=colors_fi)
ax2.set_xlabel('Feature Importance (XGBoost gain)')
ax2.set_title('Enhanced Model — Top 15 Feature Importances\n(red = new real-data features)',
              fontweight='bold')
import matplotlib.patches as mpatches
ax2.legend(handles=[
    mpatches.Patch(color='coral',     label='New neighbourhood quality features'),
    mpatches.Patch(color='steelblue', label='Original structural features'),
], fontsize=9, loc='lower right')
plt.tight_layout()
plt.savefig('figures/13_enhanced_feature_importance.png', dpi=130, bbox_inches='tight')
plt.show()
print("  Figure saved: figures/13_enhanced_feature_importance.png")

# ── Save enhanced model bundle ────────────────────────────────────────────────
enh_bundle = {
    'model':        results2[best2]['model'],
    'scaler':       sc2,
    'feature_cols': FEAT_ENH,
    'use_scaled':   results2[best2]['use_scaled'],
    'zip_quality':  ZIP_QUALITY,
    'city_ppsf':    CITY_PPSF,
    'metrics': {
        'test_mape': results2[best2]['test_mape'],
        'test_r2':   results2[best2]['test_r2'],
        'test_rmse': results2[best2]['test_rmse'],
    },
}
model_fname = f"models/enhanced_model_{best2.replace(' ','_').replace('(','').replace(')','').replace('=','').replace('.','')}.pkl"
with open(model_fname, 'wb') as f:
    pickle.dump(enh_bundle, f)
print(f"\n  Enhanced model saved: {model_fname}")

print("\n" + "=" * 70)
print("  PART 9 COMPLETE")
print("=" * 70)
print(f"  Original model  (Ridge, 19 features, synthetic data):")
print(f"    MAPE = {orig_mapes['Ridge (α=1)']:.2f}%   |   Prediction for 30 N Highland = ${orig_pred:,.0f}")
print(f"  Enhanced model ({best2}, 23 features, real PPSF anchors):")
print(f"    MAPE = {results2[best2]['test_mape']:.2f}%   |   Prediction for 30 N Highland = ${enh_preds[best2]:,.0f}")
print(f"  Gap vs Arlington assessment ($1,345,100): "
      f"Original {(orig_pred/ext_assess-1)*100:+.1f}% → Enhanced {(enh_preds[best2]/ext_assess-1)*100:+.1f}%")
print(f"  Weighted blend (all 5 sources): ${blend:,.0f}")
