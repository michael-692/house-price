"""
DC Metro Housing Price Drift Analysis
Structural break detection on FHFA HPI (1975-2025)
Data: FHFA all-transactions quarterly HPI, Washington DC MSA (traditional)
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import ruptures as rpt
from scipy import stats

# ── 0. Load & clean ────────────────────────────────────────────────────────
df  = pd.read_csv('hpi_master.csv')
dc  = df[
    (df['place_name']  == 'Washington') &
    (df['hpi_type']    == 'traditional') &        # <-- correct series
    (df['hpi_flavor']  == 'all-transactions') &
    (df['frequency']   == 'quarterly')
].copy().sort_values(['yr','period']).reset_index(drop=True)

dc['date'] = pd.to_datetime(
    dc['yr'].astype(str) + '-' +
    ((dc['period']-1)*3+1).astype(str).str.zfill(2) + '-01'
)

# ── 1. Dollar calibration ──────────────────────────────────────────────────
# Anchor: Arlington SFR median Q1 1991 ≈ $230,000
# (consistent with user's $200k in mid-1980s; NAHB/NAR historical data)
BASE_YEAR_PRICE = 230_000
base_idx  = dc[dc['date'] == '1991-01-01']['index_nsa'].values[0]
dc['price'] = dc['index_nsa'] / base_idx * BASE_YEAR_PRICE

print("═"*65)
print("  DC Metro (Washington MSA) — FHFA All-Transactions HPI")
print("  hpi_type: traditional  |  frequency: quarterly")
print("═"*65)
print(f"  Observations : {len(dc)}  ({dc['yr'].min()} Q1 → {dc['yr'].max()} Q2)")
print(f"  Base (Q1-1991 index)   : {base_idx:.2f}")
print()
print(f"  {'Year':<8} {'Index':>8}  {'Calib. Arlington SFR':>22}")
for yr in [1975,1980,1985,1990,1995,2000,2005,2010,2015,2020,2025]:
    row = dc[dc['yr']==yr]
    if row.empty: continue
    r = row.iloc[0]
    print(f"  {yr:<8} {r['index_nsa']:>8.1f}  ${r['price']:>20,.0f}")
print()

# Total appreciation
p0 = dc['price'].iloc[0]
p1 = dc['price'].iloc[-1]
print(f"  1975→2025 appreciation : {p1/p0:.1f}x  "
      f"(${p0:,.0f} → ${p1:,.0f})")

# ── 2. Growth rates ────────────────────────────────────────────────────────
dc['yoy']    = dc['price'].pct_change(4) * 100   # 4-quarter YoY
dc['roll8q'] = dc['price'].pct_change(8) / 2 * 100  # 2-yr rolling annualised

# ── 3. Structural break detection ─────────────────────────────────────────
yoy_clean = dc.dropna(subset=['yoy']).reset_index(drop=True)
sig = yoy_clean['yoy'].values.reshape(-1, 1)

# Bai-Perron (Dynp) — optimal 4-break model
algo_bp = rpt.Dynp(model='l2', min_size=8, jump=1).fit(sig)
bp4_raw   = algo_bp.predict(n_bkps=4)
bp4_dates = [yoy_clean['date'].iloc[min(i, len(yoy_clean)-1)-1] for i in bp4_raw[:-1]]

print("Structural Breaks (Bai-Perron, n=4):")
for d in bp4_dates:
    q = (d.month-1)//3 + 1
    print(f"  {d.year} Q{q}")

# ── 4. Era analysis ────────────────────────────────────────────────────────
all_bounds = [dc['date'].iloc[0]] + sorted(bp4_dates) + [dc['date'].iloc[-1]]
print()
print(f"{'Era / Period':<30} {'Avg YoY':>8}  {'Cum. Gain':>10}  {'Causal Factors'}")
print("─"*90)

ERA_CAUSES = [
    "Stagflation / Nixon shock / Volcker shock; DC federal pay raises",
    "Volcker disinflation plateau; mild S&L early signs; defence buildup begins",
    "Reagan defence buildup; Pentagon contractor boom; suburban growth",
    "Clinton-era federal expansion; dot-com contractor wave; low mortgage rates",
    "9/11 → DHS / intelligence buildout; Great Housing Bubble; GSE credit expansion; COVID surge + Amazon HQ2 + rate-suppressed demand",
]

for i, (lo, hi) in enumerate(zip(all_bounds[:-1], all_bounds[1:])):
    seg = dc[(dc['date']>=lo) & (dc['date']<hi)].dropna(subset=['yoy'])
    if len(seg) < 2: continue
    avg_yoy = seg['yoy'].mean()
    cum     = (seg['price'].iloc[-1]/seg['price'].iloc[0] - 1)*100
    label   = f"{lo.year} – {hi.year}"
    cause   = ERA_CAUSES[i] if i < len(ERA_CAUSES) else ""
    print(f"  {label:<28} {avg_yoy:>7.1f}%  {cum:>9.1f}%  {cause[:55]}")

# ── 5. Master figure ───────────────────────────────────────────────────────
EVENTS = [
    # (date, short label, panel-A ypos fraction, color)
    ('1979-01', "Volcker\nshock",           0.25, 'brown'),
    ('1981-06', "Reagan\ndefence\nbuildout", 0.42, 'steelblue'),
    ('1989-06', "S&L bubble\npeak",          0.80, 'crimson'),
    ('1993-01', "BRAC /\nbase closures",     0.55, 'darkorange'),
    ('1997-01', "Dot-com +\nfed. contractor\nboom", 0.38, 'seagreen'),
    ('2001-09', "9/11 →\nDHS / intel\nbuildout", 0.45, 'darkorange'),
    ('2004-01', "Housing\nbubble peak\n(DC)",  0.90, 'crimson'),
    ('2009-03', "ARRA +\nQE begins",          0.30, 'seagreen'),
    ('2018-11', "Amazon\nHQ2",               0.55, 'purple'),
    ('2020-04', "COVID:\nrates → 0%",         0.72, 'crimson'),
    ('2022-06', "Fed rate\nhikes",            0.48, 'saddlebrown'),
]

fig, axes = plt.subplots(3, 1, figsize=(15, 17), sharex=True)
fig.suptitle(
    'Washington DC Metro — 50-Year Single-Family Housing Price Drift\n'
    'FHFA All-Transactions HPI, Quarterly 1975 Q1 – 2025 Q2  |  Calibrated to Arlington SFR',
    fontsize=13, fontweight='bold', y=0.99
)

# ── Panel A: Price level ──────────────────────────────────────────────────
ax = axes[0]
ax.fill_between(dc['date'], dc['price']/1e6, alpha=0.18, color='steelblue')
ax.plot(dc['date'], dc['price']/1e6, color='steelblue', linewidth=2.0, label='Calibrated SFR price')

# Shade eras
era_palette = ['#dbeafe','#fef3c7','#d1fae5','#fee2e2','#ede9fe']
era_names   = ['Era 1\n1975-80', 'Era 2\n1980-87', 'Era 3\n1987-97',
               'Era 4\n1997-09', 'Era 5\n2009-25']
for (lo, hi), pal, nm in zip(era_bounds := list(zip(all_bounds[:-1], all_bounds[1:])),
                              era_palette, era_names):
    ax.axvspan(lo, hi, alpha=0.30, color=pal, zorder=0)

for bp in bp4_dates:
    ax.axvline(bp, color='red', linestyle='--', linewidth=1.4, alpha=0.75)

ax.set_ylabel('Estimated Arlington SFR Value ($M)', fontsize=10)
ax.set_title('A.  Calibrated Price Level  (Arlington SFR: Q1-1991 anchor = $230k)', fontsize=10)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('$%.1fM'))
ax.annotate('↑ $200k era\n(1985)', xy=(pd.Timestamp('1985-07-01'), 0.185),
            fontsize=8.5, color='gray', ha='center', style='italic')

# Event labels on panel A
for ev, label, yf, col in EVENTS:
    try:
        ts = pd.Timestamp(ev)
        yval = ax.get_ylim()[1] * yf if ax.get_ylim()[1] > 0 else yf
        ax.annotate(label, xy=(ts, 0),
                    xytext=(ts, dc['price'].max()/1e6 * yf),
                    fontsize=6.5, color=col, ha='center',
                    bbox=dict(boxstyle='round,pad=0.15', fc='white', ec=col, alpha=0.8))
    except Exception:
        pass

# ── Panel B: YoY growth ───────────────────────────────────────────────────
ax = axes[1]
ax.axhline(0, color='black', linewidth=0.8)
ax.fill_between(dc['date'], dc['yoy'].clip(lower=0), 0,
                alpha=0.55, color='seagreen', label='Appreciation')
ax.fill_between(dc['date'], dc['yoy'].clip(upper=0), 0,
                alpha=0.55, color='crimson', label='Depreciation')
ax.plot(dc['date'], dc['roll8q'], color='navy', linewidth=1.8,
        label='2-yr rolling avg')

for bp in bp4_dates:
    ax.axvline(bp, color='red', linestyle='--', linewidth=1.4, alpha=0.75,
               label='Structural break' if bp == bp4_dates[0] else '')

ax.set_ylabel('YoY Growth (%)', fontsize=10)
ax.set_title('B.  Annual Appreciation Rate  (YoY, 2-yr rolling avg in navy)', fontsize=10)
ax.legend(loc='upper right', fontsize=8, ncol=2)
ax.set_ylim(-15, 25)

# ── Panel C: Cumulative from 1975 ─────────────────────────────────────────
ax = axes[2]
dc['cum'] = (dc['price'] / dc['price'].iloc[0] - 1) * 100
ax.plot(dc['date'], dc['cum'], color='darkviolet', linewidth=2)
ax.fill_between(dc['date'], dc['cum'], alpha=0.15, color='darkviolet')

for bp in bp4_dates:
    ax.axvline(bp, color='red', linestyle='--', linewidth=1.4, alpha=0.75)

# Era shade + label
for (lo, hi), pal, nm in zip(era_bounds, era_palette, era_names):
    ax.axvspan(lo, hi, alpha=0.28, color=pal, zorder=0)
    mid = lo + (hi-lo)/2
    seg_cum = dc[(dc['date']>=lo) & (dc['date']<hi)]['cum']
    if seg_cum.empty: continue
    ytop = seg_cum.max()
    ax.text(mid, ytop * 0.85, nm, ha='center', va='top',
            fontsize=8, color='#333',
            bbox=dict(boxstyle='round,pad=0.2', fc=pal, ec='gray', alpha=0.9))

ax.set_ylabel('Cumulative Gain from 1975 Q1 (%)', fontsize=10)
ax.set_xlabel('Year', fontsize=10)
ax.set_title('C.  50-Year Cumulative Appreciation  (Bai-Perron breakpoints in red dashed)', fontsize=10)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f%%'))

for a in axes:
    a.grid(axis='y', alpha=0.3, linestyle=':')
    a.set_xlim(dc['date'].min(), dc['date'].max())

plt.tight_layout(rect=[0,0,1,0.97])
plt.savefig('figures/10_dc_price_drift_analysis.png', dpi=150, bbox_inches='tight')
plt.show()
print("Figure saved: figures/10_dc_price_drift_analysis.png")

# ── 6. Key finding summary ─────────────────────────────────────────────────
print()
print("═"*65)
print("  KEY FINDINGS: When did DC prices drift significantly upward?")
print("═"*65)
for i, (bp, nxt) in enumerate(zip([dc['date'].iloc[0]]+bp4_dates,
                                   bp4_dates+[dc['date'].iloc[-1]])):
    seg = dc[(dc['date']>=bp) & (dc['date']<nxt)].dropna(subset=['yoy'])
    if len(seg)<2: continue
    print(f"\n  Breakpoint {i+1}: {bp.year} Q{(bp.month-1)//3+1}  →  "
          f"{nxt.year} Q{(nxt.month-1)//3+1}")
    print(f"    Avg YoY growth : {seg['yoy'].mean():+.1f}%")
    print(f"    Cum. gain      : {(seg['price'].iloc[-1]/seg['price'].iloc[0]-1)*100:.1f}%")
    print(f"    Start price    : ${seg['price'].iloc[0]:,.0f}")
    print(f"    End price      : ${seg['price'].iloc[-1]:,.0f}")
