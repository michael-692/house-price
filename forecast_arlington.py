"""
House Price Forecast Model - Arlington, Virginia
=================================================
Data Source: FHFA House Price Index (HPI) — hpi_master.csv
Target:      Quarterly HPI for Arlington-Alexandria-Reston, VA-WV MSAD
Goal:        Forecast Q2 2026 house price YoY change for Arlington, VA

Approach
--------
  - FHFA traditional all-transactions quarterly HPI (1975 Q4 – Q2 2025)
  - Train/test split: hold out last 3 years (12 quarters) for evaluation
  - Model 1: SARIMA(2,1,1)(1,1,1)[4] — on HPI level
  - Model 2: Random Forest — on QoQ % change (avoids level extrapolation)
  - Model 3: Gradient Boosting — on QoQ % change
  - Final forecast: weighted ensemble (inverse-RMSE weights)
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX
import json

# ─────────────────────────────────────────────
# 1. LOAD & PREPARE DATA
# ─────────────────────────────────────────────
print("=" * 65)
print("HOUSE PRICE FORECAST MODEL — ARLINGTON, VIRGINIA")
print("=" * 65)

df_raw = pd.read_csv('hpi_master.csv')

arlington = df_raw[
    (df_raw['place_id'] == '11694') &
    (df_raw['hpi_type'] == 'traditional') &
    (df_raw['hpi_flavor'] == 'all-transactions') &
    (df_raw['frequency'] == 'quarterly')
].copy().sort_values(['yr', 'period']).reset_index(drop=True)

quarter_month = {1: 3, 2: 6, 3: 9, 4: 12}
arlington['date'] = pd.to_datetime(
    arlington['yr'].astype(str) + '-' +
    arlington['period'].map(quarter_month).astype(str) + '-01'
) + pd.offsets.MonthEnd(0)

arlington = arlington.set_index('date').rename(columns={'index_nsa': 'hpi'})
hpi = arlington['hpi'].copy().asfreq('QE-DEC')

print(f"\nData range  : {hpi.index[0].date()} → {hpi.index[-1].date()}")
print(f"Observations: {len(hpi)}")
print(f"Latest HPI  : {hpi.iloc[-1]:.2f}  ({arlington['yr'].iloc[-1]} Q{arlington['period'].iloc[-1]})")

# ─────────────────────────────────────────────
# 2. STATIONARITY CHECK
# ─────────────────────────────────────────────
from statsmodels.tsa.stattools import adfuller
adf_result = adfuller(hpi.dropna(), autolag='AIC')
print(f"\nADF p-value (HPI level): {adf_result[1]:.4f}  "
      f"({'non-stationary' if adf_result[1] > 0.05 else 'stationary'})")
adf_diff = adfuller(hpi.diff().dropna(), autolag='AIC')
print(f"ADF p-value (1st diff) : {adf_diff[1]:.4f}  "
      f"({'non-stationary' if adf_diff[1] > 0.05 else 'stationary'})")

# ─────────────────────────────────────────────
# 3. TRAIN / TEST SPLIT
# ─────────────────────────────────────────────
TEST_QUARTERS = 12   # hold out last 3 years

n_total  = len(hpi)
n_train  = n_total - TEST_QUARTERS
train_hpi = hpi.iloc[:n_train]
test_hpi  = hpi.iloc[n_train:]

print(f"\nTraining set: {train_hpi.index[0].date()} → {train_hpi.index[-1].date()}  ({len(train_hpi)} obs)")
print(f"Test set    : {test_hpi.index[0].date()}  → {test_hpi.index[-1].date()}  ({len(test_hpi)} obs)")

# ─────────────────────────────────────────────
# 4a. SARIMA — fitted on train, evaluated on test
# ─────────────────────────────────────────────
print("\n" + "─" * 55)
print("Model 1: SARIMA(2,1,1)(1,1,1)[4]")
print("─" * 55)

sarima_train_mdl = SARIMAX(
    train_hpi,
    order=(2, 1, 1),
    seasonal_order=(1, 1, 1, 4),
    enforce_stationarity=False,
    enforce_invertibility=False
)
sarima_train_fit = sarima_train_mdl.fit(disp=False)

# Rolling one-step-ahead predictions over the test window
sarima_test_preds = []
for i in range(TEST_QUARTERS):
    fc = sarima_train_fit.apply(hpi.iloc[:n_train + i]).forecast(steps=1)
    sarima_test_preds.append(float(fc.iloc[0]))
sarima_test_preds = np.array(sarima_test_preds)

sarima_mae  = mean_absolute_error(test_hpi, sarima_test_preds)
sarima_rmse = np.sqrt(mean_squared_error(test_hpi, sarima_test_preds))
sarima_r2   = r2_score(test_hpi, sarima_test_preds)
print(f"  MAE : {sarima_mae:.2f}")
print(f"  RMSE: {sarima_rmse:.2f}")
print(f"  R²  : {sarima_r2:.4f}")

# ─────────────────────────────────────────────
# 4b. ML MODELS — predict QoQ % change, reconstruct HPI
# ─────────────────────────────────────────────
def build_features(hpi_series):
    """Build feature matrix using lag/rolling features of QoQ % change."""
    s = hpi_series.copy()
    df = pd.DataFrame(index=s.index)
    df['hpi']      = s
    df['qoq']      = s.pct_change(1) * 100
    df['yoy']      = s.pct_change(4) * 100
    df['qoq_lag1'] = df['qoq'].shift(1)
    df['qoq_lag2'] = df['qoq'].shift(2)
    df['qoq_lag4'] = df['qoq'].shift(4)
    df['yoy_lag1'] = df['yoy'].shift(1)
    df['yoy_lag4'] = df['yoy'].shift(4)
    df['roll4_qoq_mean'] = df['qoq'].shift(1).rolling(4).mean()
    df['roll4_qoq_std']  = df['qoq'].shift(1).rolling(4).std()
    df['roll8_qoq_mean'] = df['qoq'].shift(1).rolling(8).mean()
    df['q1'] = (s.index.quarter == 1).astype(int)
    df['q2'] = (s.index.quarter == 2).astype(int)
    df['q3'] = (s.index.quarter == 3).astype(int)
    df['trend'] = np.arange(len(s))
    return df

FEAT_COLS = [
    'qoq_lag1', 'qoq_lag2', 'qoq_lag4',
    'yoy_lag1', 'yoy_lag4',
    'roll4_qoq_mean', 'roll4_qoq_std', 'roll8_qoq_mean',
    'q1', 'q2', 'q3', 'trend'
]

full_df = build_features(hpi)
TARGET  = 'qoq'  # predict next quarter's QoQ % change

# ML train/test uses same split but on percent changes
ml_data = full_df.dropna(subset=FEAT_COLS + [TARGET]).copy()
ml_n    = len(ml_data)

# Align with HPI test period by date
ml_train = ml_data[ml_data.index < test_hpi.index[0]]
ml_test  = ml_data[ml_data.index.isin(test_hpi.index)]

X_tr = ml_train[FEAT_COLS]; y_tr = ml_train[TARGET]
X_te = ml_test[FEAT_COLS];  y_te = ml_test[TARGET]

print(f"\n  ML train obs: {len(X_tr)}  ML test obs: {len(X_te)}")

# ── Random Forest ──────────────────────────────────
print("\n" + "─" * 55)
print("Model 2: Random Forest (target = QoQ % change)")
print("─" * 55)

scaler_rf = StandardScaler()
X_tr_sc = scaler_rf.fit_transform(X_tr)
X_te_sc = scaler_rf.transform(X_te)

rf_mdl = RandomForestRegressor(n_estimators=500, max_depth=5,
                               min_samples_leaf=5, random_state=42)
rf_mdl.fit(X_tr_sc, y_tr)
rf_qoq_pred = rf_mdl.predict(X_te_sc)

# Reconstruct HPI from predicted QoQ changes
rf_hpi_pred = []
prev = train_hpi.iloc[-1]
for qoq in rf_qoq_pred:
    nxt = prev * (1 + qoq / 100)
    rf_hpi_pred.append(nxt)
    prev = nxt
rf_hpi_pred = np.array(rf_hpi_pred)

rf_mae  = mean_absolute_error(test_hpi, rf_hpi_pred)
rf_rmse = np.sqrt(mean_squared_error(test_hpi, rf_hpi_pred))
rf_r2   = r2_score(test_hpi, rf_hpi_pred)
print(f"  MAE : {rf_mae:.2f}")
print(f"  RMSE: {rf_rmse:.2f}")
print(f"  R²  : {rf_r2:.4f}")

# ── Gradient Boosting ──────────────────────────────
print("\n" + "─" * 55)
print("Model 3: Gradient Boosting (target = QoQ % change)")
print("─" * 55)

scaler_gb = StandardScaler()
X_tr_gb = scaler_gb.fit_transform(X_tr)
X_te_gb = scaler_gb.transform(X_te)

gb_mdl = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05,
                                    max_depth=4, subsample=0.8, random_state=42)
gb_mdl.fit(X_tr_gb, y_tr)
gb_qoq_pred = gb_mdl.predict(X_te_gb)

gb_hpi_pred = []
prev = train_hpi.iloc[-1]
for qoq in gb_qoq_pred:
    nxt = prev * (1 + qoq / 100)
    gb_hpi_pred.append(nxt)
    prev = nxt
gb_hpi_pred = np.array(gb_hpi_pred)

gb_mae  = mean_absolute_error(test_hpi, gb_hpi_pred)
gb_rmse = np.sqrt(mean_squared_error(test_hpi, gb_hpi_pred))
gb_r2   = r2_score(test_hpi, gb_hpi_pred)
print(f"  MAE : {gb_mae:.2f}")
print(f"  RMSE: {gb_rmse:.2f}")
print(f"  R²  : {gb_r2:.4f}")

# ─────────────────────────────────────────────
# 5. ENSEMBLE
# ─────────────────────────────────────────────
print("\n" + "─" * 55)
print("Model 4: Weighted Ensemble (inverse-RMSE weights)")
print("─" * 55)

inv_rmse = np.array([1/sarima_rmse, 1/rf_rmse, 1/gb_rmse])
weights  = inv_rmse / inv_rmse.sum()
print(f"  SARIMA weight       : {weights[0]:.4f}")
print(f"  Random Forest weight: {weights[1]:.4f}")
print(f"  Grad. Boost weight  : {weights[2]:.4f}")

ens_pred = weights[0]*sarima_test_preds + weights[1]*rf_hpi_pred + weights[2]*gb_hpi_pred

ens_mae  = mean_absolute_error(test_hpi, ens_pred)
ens_rmse = np.sqrt(mean_squared_error(test_hpi, ens_pred))
ens_r2   = r2_score(test_hpi, ens_pred)
print(f"  MAE : {ens_mae:.2f}")
print(f"  RMSE: {ens_rmse:.2f}")
print(f"  R²  : {ens_r2:.4f}")

# ─────────────────────────────────────────────
# 6. MODEL COMPARISON
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("MODEL COMPARISON — Test Set Performance (12 quarters)")
print("=" * 60)
print(f"{'Model':<28} {'MAE':>7} {'RMSE':>7} {'R²':>8}")
print("-" * 60)
for name, mae, rmse, r2 in [
    ("SARIMA(2,1,1)(1,1,1)[4]", sarima_mae, sarima_rmse, sarima_r2),
    ("Random Forest",           rf_mae,     rf_rmse,     rf_r2),
    ("Gradient Boosting",       gb_mae,     gb_rmse,     gb_r2),
    ("Weighted Ensemble",       ens_mae,    ens_rmse,    ens_r2),
]:
    print(f"{name:<28} {mae:>7.2f} {rmse:>7.2f} {r2:>8.4f}")

# ─────────────────────────────────────────────
# 7. FORECAST Q2 2026
#    Refit on ALL data; iteratively step to Q2 2026
# ─────────────────────────────────────────────
print("\n" + "=" * 65)
print("Q2 2026 FORECAST — ARLINGTON-ALEXANDRIA-RESTON, VA")
print("=" * 65)

# Latest observed: Q2 2025
latest_hpi_val = float(hpi.iloc[-1])
latest_yr      = int(arlington['yr'].iloc[-1])
latest_q       = int(arlington['period'].iloc[-1])
print(f"\nLatest observed: {latest_yr} Q{latest_q}  HPI = {latest_hpi_val:.2f}")

# Quarters to step: Q3 2025, Q4 2025, Q1 2026, Q2 2026  → 4 steps
n_steps = 4
print(f"Steps to Q2 2026: {n_steps}")

# ── SARIMA full-data forecast ──────────────────
sarima_full_mdl = SARIMAX(
    hpi,
    order=(2, 1, 1),
    seasonal_order=(1, 1, 1, 4),
    enforce_stationarity=False,
    enforce_invertibility=False
)
sarima_full_fit = sarima_full_mdl.fit(disp=False)

# Use get_prediction with integer index (avoids date range issues)
start_idx = len(hpi)
end_idx   = len(hpi) + n_steps - 1
sarima_fc = sarima_full_fit.predict(start=start_idx, end=end_idx)
sarima_q2_2026 = float(sarima_fc.iloc[-1])

print(f"\n  SARIMA Q2 2026 HPI: {sarima_q2_2026:.2f}")

# ── ML iterative forecast (retrained on full data) ──
full_df_all = build_features(hpi)
ml_all = full_df_all.dropna(subset=FEAT_COLS + [TARGET])
X_all_sc_rf = scaler_rf.fit_transform(ml_all[FEAT_COLS])
X_all_sc_gb = scaler_gb.fit_transform(ml_all[FEAT_COLS])

rf_full = RandomForestRegressor(n_estimators=500, max_depth=5,
                                min_samples_leaf=5, random_state=42)
rf_full.fit(X_all_sc_rf, ml_all[TARGET])

gb_full = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05,
                                     max_depth=4, subsample=0.8, random_state=42)
gb_full.fit(X_all_sc_gb, ml_all[TARGET])

def next_q(yr, q):
    return (yr+1, 1) if q == 4 else (yr, q+1)

def build_one_row(hpi_series_vals, quarter, trend_idx):
    """Build a single-row feature dict from the most recent HPI values."""
    s = list(hpi_series_vals)
    n = len(s)
    qoq_vals  = [((s[i]-s[i-1])/s[i-1]*100) if i > 0 else 0 for i in range(n)]
    yoy_vals  = [((s[i]-s[i-4])/s[i-4]*100) if i >= 4 else 0 for i in range(n)]

    qoq_now   = qoq_vals[-1]
    qoq_lag1  = qoq_vals[-2] if n >= 2 else 0
    qoq_lag2  = qoq_vals[-3] if n >= 3 else 0
    qoq_lag4  = qoq_vals[-5] if n >= 5 else 0
    yoy_lag1  = yoy_vals[-2] if n >= 2 else 0
    yoy_lag4  = yoy_vals[-5] if n >= 5 else 0
    roll4_m   = np.mean(qoq_vals[-5:-1]) if n >= 5 else qoq_now
    roll4_std = np.std(qoq_vals[-5:-1])  if n >= 5 else 0
    roll8_m   = np.mean(qoq_vals[-9:-1]) if n >= 9 else roll4_m

    return {
        'qoq_lag1': qoq_lag1, 'qoq_lag2': qoq_lag2, 'qoq_lag4': qoq_lag4,
        'yoy_lag1': yoy_lag1, 'yoy_lag4': yoy_lag4,
        'roll4_qoq_mean': roll4_m, 'roll4_qoq_std': roll4_std,
        'roll8_qoq_mean': roll8_m,
        'q1': int(quarter == 1), 'q2': int(quarter == 2), 'q3': int(quarter == 3),
        'trend': trend_idx
    }

rolling_hpi = list(hpi.values)
trend_start = len(hpi)
cur_yr, cur_q = latest_yr, latest_q

future_records = []
for step in range(n_steps):
    nyr, nq = next_q(cur_yr, cur_q)
    row_dict = build_one_row(rolling_hpi, nq, trend_start + step)
    x_row = pd.DataFrame([row_dict])[FEAT_COLS]

    rf_qoq = float(rf_full.predict(scaler_rf.transform(x_row))[0])
    gb_qoq = float(gb_full.predict(scaler_gb.transform(x_row))[0])

    # Reconstruct HPI from QoQ change
    rf_hpi  = rolling_hpi[-1] * (1 + rf_qoq / 100)
    gb_hpi  = rolling_hpi[-1] * (1 + gb_qoq / 100)
    sar_hpi = float(sarima_fc.iloc[step])
    ens_hpi = weights[0]*sar_hpi + weights[1]*rf_hpi + weights[2]*gb_hpi

    future_records.append({'yr': nyr, 'q': nq,
                           'sarima': sar_hpi, 'rf': rf_hpi,
                           'gb': gb_hpi, 'ensemble': ens_hpi})
    rolling_hpi.append(ens_hpi)
    cur_yr, cur_q = nyr, nq

# Extract Q2 2026
q2_2026_rec = [r for r in future_records if r['yr'] == 2026 and r['q'] == 2][0]
q2_2026_ens = q2_2026_rec['ensemble']
q2_2026_sar = q2_2026_rec['sarima']
q2_2026_rf  = q2_2026_rec['rf']
q2_2026_gb  = q2_2026_rec['gb']

# Year-over-year: compare to Q2 2025 (latest observed)
hpi_q2_2025 = latest_hpi_val   # Q2 2025 is the last known data point

yoy_ens = (q2_2026_ens / hpi_q2_2025 - 1) * 100
yoy_sar = (q2_2026_sar / hpi_q2_2025 - 1) * 100
yoy_rf  = (q2_2026_rf  / hpi_q2_2025 - 1) * 100
yoy_gb  = (q2_2026_gb  / hpi_q2_2025 - 1) * 100

print(f"\n{'─'*60}")
print(f"  Q2 2025 HPI (observed):  {hpi_q2_2025:.2f}")
print(f"{'─'*60}")
print(f"  {'Model':<22} {'Q2 2026 HPI':>12} {'YoY Change':>12}")
print(f"  {'─'*48}")
print(f"  {'SARIMA':<22} {q2_2026_sar:>12.2f} {yoy_sar:>+11.2f}%")
print(f"  {'Random Forest':<22} {q2_2026_rf:>12.2f} {yoy_rf:>+11.2f}%")
print(f"  {'Gradient Boosting':<22} {q2_2026_gb:>12.2f} {yoy_gb:>+11.2f}%")
print(f"  {'Weighted Ensemble':<22} {q2_2026_ens:>12.2f} {yoy_ens:>+11.2f}%")
print(f"{'─'*60}")

print(f"""
╔═══════════════════════════════════════════════════════════╗
║  FINAL FORECAST — Q2 2026                                ║
║  Region: Arlington-Alexandria-Reston, VA-WV MSA          ║
╠═══════════════════════════════════════════════════════════╣
║  Forecast HPI (Q2 2026):   {q2_2026_ens:>8.2f}                    ║
║  Observed HPI (Q2 2025):   {hpi_q2_2025:>8.2f}                    ║
║  YoY Price Change:         {yoy_ens:>+8.2f}%                   ║
╚═══════════════════════════════════════════════════════════╝
""")

print(f"  Interpretation: House prices in the Arlington, VA MSA")
print(f"  are forecast to {'increase' if yoy_ens > 0 else 'decrease'} by approximately")
print(f"  {abs(yoy_ens):.1f}% year-over-year in Q2 2026.")

# ─────────────────────────────────────────────
# 8. VISUALIZATIONS
# ─────────────────────────────────────────────
print("\nGenerating charts...")

colors = {'sarima': '#1f77b4', 'rf': '#ff7f0e', 'gb': '#2ca02c',
          'ensemble': '#d62728', 'actual': 'black'}

fig = plt.figure(figsize=(16, 15))
gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.3)

# ── Plot 1: Full HPI history + test predictions ──────────────────
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(hpi.index, hpi.values, color=colors['actual'], lw=1.8,
         label='Observed HPI', zorder=5)
ax1.axvline(test_hpi.index[0], color='gray', ls='--', lw=1.2, label='Train/Test Split')
ax1.plot(test_hpi.index, test_hpi.values, 'ko', ms=5, label='Actual (test)')
ax1.plot(test_hpi.index, sarima_test_preds, color=colors['sarima'],  ls='--', lw=1.5, label='SARIMA')
ax1.plot(test_hpi.index, rf_hpi_pred,       color=colors['rf'],       ls='--', lw=1.5, label='Random Forest')
ax1.plot(test_hpi.index, gb_hpi_pred,       color=colors['gb'],       ls='--', lw=1.5, label='Gradient Boosting')
ax1.plot(test_hpi.index, ens_pred,          color=colors['ensemble'], ls='-',  lw=2.2, label='Ensemble')
ax1.set_title('FHFA HPI — Arlington-Alexandria-Reston, VA: Historical Series & Test-Period Model Predictions',
              fontsize=11, fontweight='bold')
ax1.set_ylabel('HPI (Index, base ≈ 1980)')
ax1.legend(fontsize=7.5, ncol=3, loc='upper left')
ax1.grid(alpha=0.3)

# ── Plot 2: Forecast to Q2 2026 ──────────────────────────────────
ax2 = fig.add_subplot(gs[1, :])
show_from = pd.Timestamp('2019-01-01')
hist_slice = hpi[hpi.index >= show_from]
ax2.plot(hist_slice.index, hist_slice.values, color=colors['actual'], lw=2, label='Observed HPI')

fut_dates = [pd.Timestamp(f"{r['yr']}-{quarter_month[r['q']]:02d}-01") + pd.offsets.MonthEnd(0)
             for r in future_records]
ax2.plot(fut_dates, [r['sarima']   for r in future_records], color=colors['sarima'],  ls='--', lw=1.5, label='SARIMA')
ax2.plot(fut_dates, [r['rf']       for r in future_records], color=colors['rf'],       ls='--', lw=1.5, label='Random Forest')
ax2.plot(fut_dates, [r['gb']       for r in future_records], color=colors['gb'],       ls='--', lw=1.5, label='Gradient Boosting')
ax2.plot(fut_dates, [r['ensemble'] for r in future_records], color=colors['ensemble'], ls='-',  lw=2.5, label='Ensemble')

q2_date = fut_dates[-1]
ax2.axvline(q2_date, color='purple', ls=':', lw=1.5, alpha=0.7)
ax2.scatter([q2_date], [q2_2026_ens], color='purple', zorder=10, s=100,
            label=f'Q2 2026 → {q2_2026_ens:.1f}  (YoY {yoy_ens:+.1f}%)')
ax2.set_title('HPI Forecast — Arlington, VA: Q2 2026 Projection', fontsize=11, fontweight='bold')
ax2.set_ylabel('HPI (Index)')
ax2.legend(fontsize=7.5, ncol=2)
ax2.grid(alpha=0.3)

# ── Plot 3: RMSE comparison ───────────────────────────────────────
ax3 = fig.add_subplot(gs[2, 0])
model_names = ['SARIMA', 'Rand. Forest', 'Grad. Boost', 'Ensemble']
rmse_vals   = [sarima_rmse, rf_rmse, gb_rmse, ens_rmse]
bar_colors  = [colors['sarima'], colors['rf'], colors['gb'], colors['ensemble']]
bars = ax3.bar(model_names, rmse_vals, color=bar_colors, alpha=0.85, edgecolor='black')
for bar, v in zip(bars, rmse_vals):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
             f'{v:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax3.set_title('Test-Set RMSE by Model\n(lower is better)', fontsize=10, fontweight='bold')
ax3.set_ylabel('RMSE (Index Points)')
ax3.grid(axis='y', alpha=0.3)

# ── Plot 4: Actual vs Predicted scatter (Ensemble) ───────────────
ax4 = fig.add_subplot(gs[2, 1])
ax4.scatter(test_hpi.values, ens_pred, color=colors['ensemble'],
            alpha=0.85, edgecolor='black', s=55, zorder=5)
mn = min(test_hpi.min(), ens_pred.min()) - 3
mx = max(test_hpi.max(), ens_pred.max()) + 3
ax4.plot([mn, mx], [mn, mx], 'k--', lw=1.2, label='Perfect fit')
ax4.set_xlabel('Actual HPI (Test Set)')
ax4.set_ylabel('Predicted HPI (Ensemble)')
ax4.set_title(f'Actual vs Predicted — Ensemble\n(R² = {ens_r2:.4f})', fontsize=10, fontweight='bold')
ax4.legend(fontsize=8)
ax4.grid(alpha=0.3)

plt.suptitle(
    'Arlington-Alexandria-Reston, VA — FHFA House Price Index\n'
    'Forecast Model: SARIMA + Random Forest + Gradient Boosting Ensemble',
    fontsize=13, fontweight='bold', y=0.995
)
plt.savefig('arlington_hpi_forecast.png', dpi=150, bbox_inches='tight')
print("  Saved: arlington_hpi_forecast.png")

# ─────────────────────────────────────────────
# 9. SAVE JSON RESULTS
# ─────────────────────────────────────────────
results = {
    "model_description": "House Price Forecast — Arlington, VA (FHFA HPI)",
    "data_source": "FHFA HPI Master File (hpi_master.csv)",
    "region": "Arlington-Alexandria-Reston, VA-WV MSAD (place_id 11694)",
    "hpi_series": "Traditional, All-Transactions, Quarterly",
    "data_range": f"{hpi.index[0].date()} to {hpi.index[-1].date()}",
    "train_period": f"{train_hpi.index[0].date()} to {train_hpi.index[-1].date()} ({len(train_hpi)} obs)",
    "test_period":  f"{test_hpi.index[0].date()} to {test_hpi.index[-1].date()} ({len(test_hpi)} obs)",
    "models": {
        "SARIMA": {
            "spec": "SARIMA(2,1,1)(1,1,1)[4]",
            "target": "HPI level",
            "test_MAE": round(sarima_mae, 3),
            "test_RMSE": round(sarima_rmse, 3),
            "test_R2": round(sarima_r2, 4)
        },
        "RandomForest": {
            "spec": "n_estimators=500, max_depth=5, min_samples_leaf=5",
            "target": "QoQ % change",
            "test_MAE": round(rf_mae, 3),
            "test_RMSE": round(rf_rmse, 3),
            "test_R2": round(rf_r2, 4)
        },
        "GradientBoosting": {
            "spec": "n_estimators=300, lr=0.05, max_depth=4, subsample=0.8",
            "target": "QoQ % change",
            "test_MAE": round(gb_mae, 3),
            "test_RMSE": round(gb_rmse, 3),
            "test_R2": round(gb_r2, 4)
        },
        "WeightedEnsemble": {
            "weights": {
                "SARIMA": round(float(weights[0]), 4),
                "RandomForest": round(float(weights[1]), 4),
                "GradientBoosting": round(float(weights[2]), 4)
            },
            "test_MAE": round(ens_mae, 3),
            "test_RMSE": round(ens_rmse, 3),
            "test_R2": round(ens_r2, 4)
        }
    },
    "forecast_path": [
        {"quarter": f"{r['yr']} Q{r['q']}",
         "hpi_sarima": round(r['sarima'], 2),
         "hpi_rf": round(r['rf'], 2),
         "hpi_gb": round(r['gb'], 2),
         "hpi_ensemble": round(r['ensemble'], 2)}
        for r in future_records
    ],
    "q2_2026_prediction": {
        "quarter": "2026 Q2",
        "q2_2025_hpi_observed": round(hpi_q2_2025, 2),
        "SARIMA_HPI": round(q2_2026_sar, 2),
        "RandomForest_HPI": round(q2_2026_rf, 2),
        "GradientBoosting_HPI": round(q2_2026_gb, 2),
        "Ensemble_HPI": round(q2_2026_ens, 2),
        "YoY_pct_change_SARIMA": round(yoy_sar, 2),
        "YoY_pct_change_RandomForest": round(yoy_rf, 2),
        "YoY_pct_change_GradientBoosting": round(yoy_gb, 2),
        "YoY_pct_change_Ensemble": round(yoy_ens, 2),
        "direction": "increase" if yoy_ens > 0 else "decrease"
    }
}

with open('arlington_forecast_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("  Saved: arlington_forecast_results.json")

print("\nDone.")
