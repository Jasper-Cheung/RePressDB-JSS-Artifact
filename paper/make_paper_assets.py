#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'experiment_data'


def gm(values):
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a) & (a > 0)]
    return float(np.exp(np.mean(np.log(a))))


def pct(x):
    return 100.0 * x


def fmt4(x):
    return f'{x:.4f}'


def latex_escape(s: str) -> str:
    return s.replace('_', r'\_')


def write(path: Path, text: str):
    path.write_text(text, encoding='utf-8')


def bootstrap_ci(values, seed=20260719, n=50000):
    a = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n, len(a)))
    est = np.exp(np.mean(np.log(a[idx]), axis=1))
    return tuple(float(x) for x in np.quantile(est, [0.025, 0.975]))


mg = pd.read_csv(DATA / 'multiseed_global_results.csv')
ua = pd.read_csv(DATA / 'unified_ablation_summary.csv')
cf = pd.read_csv(DATA / 'crash_fuzz_summary.csv')
cr = pd.read_csv(DATA / 'crash_fuzz_records.csv')
sp = pd.read_csv(DATA / 'sampled_profile_summary.csv')
ce = pd.read_csv(DATA / 'cost_error_summary.csv')
vstats = pd.read_csv(DATA / 'v170_implementation_paired_statistics.csv')
vpol = pd.read_csv(DATA / 'v170_policy_statistics_by_implementation.csv')
vsumm = pd.read_csv(DATA / 'v170_end_to_end_summary.csv')
veq = pd.read_csv(DATA / 'v170_migration_decision_equivalence.csv')
vass = json.loads((DATA / 'v170_assessment.json').read_text(encoding='utf-8'))
cal = json.loads((DATA / 'calibration_overhead_summary.json').read_text(encoding='utf-8'))
legacy = pd.read_csv(DATA / 'legacy_diagnostic_end_to_end_summary.csv')

# -------------------------------------------------------------------------
# Frozen numerical macros
# -------------------------------------------------------------------------
piv = mg.pivot_table(index=['trace', 'chunk_size', 'trace_seed'], columns='policy', values='total_time_ns')
ratios = {}
for denom, key in [
    ('shape_ewma_be_frozen', 'ReplayRepressShape'),
    ('initial_best', 'ReplayRepressInitial'),
    ('page_hinkley_be', 'ReplayRepressPH'),
    ('afhc_d_frozen', 'ReplayRepressAFHC'),
    ('best_static', 'ReplayRepressBestStatic'),
]:
    r = piv['repressdb_frozen'] / piv[denom]
    ratios[key] = {
        'gm': gm(r),
        'low': bootstrap_ci(r, seed=20260719 + len(ratios))[0],
        'high': bootstrap_ci(r, seed=20260719 + len(ratios))[1],
        'wins': int((r < 1).sum()),
        'n': int(len(r)),
    }

agg = mg.groupby('policy').agg(
    migration_count=('migration_count', 'sum'),
    unamortized=('unamortized_migration_count', 'sum'),
    stale=('stale_migration_count', 'sum'),
    assessable=('stale_assessable_migration_count', 'sum'),
    rewrite_bytes=('rewrite_bytes', 'sum'),
)
agg['unamortized_rate'] = agg['unamortized'] / agg['migration_count']
agg['stale_rate'] = agg['stale'] / agg['assessable']

cached_service = vpol[(vpol.implementation == 'cached') & (vpol.scope == 'aggregate') & (vpol.metric == 'service_total_time_ns')]
cached_p99 = vpol[(vpol.implementation == 'cached') & (vpol.scope == 'aggregate') & (vpol.metric == 'query_p99_ns')]
impl_service = vstats[(vstats.scope == 'policy_aggregate') & (vstats.metric == 'service_total_time_ns')]
impl_dispatch = vstats[(vstats.scope == 'policy_aggregate') & (vstats.metric == 'query_catalog_dispatch_time_ns')]
impl_p99 = vstats[(vstats.scope == 'policy_aggregate') & (vstats.metric == 'query_p99_ns')]


def policy_row(df, num, den):
    return df[(df.numerator_policy == num) & (df.denominator_policy == den)].iloc[0]


def impl_row(df, policy):
    return df[df.policy == policy].iloc[0]

rs = policy_row(cached_service, 'repressdb_frozen', 'shape_ewma_be_frozen')
ri = policy_row(cached_service, 'repressdb_frozen', 'initial_best')
rsp99 = policy_row(cached_p99, 'repressdb_frozen', 'shape_ewma_be_frozen')
rip99 = policy_row(cached_p99, 'repressdb_frozen', 'initial_best')

legacy_primary = legacy[legacy.result_group == 'primary']
legacy_piv = legacy_primary.pivot(index=['trace', 'chunk_size', 'repeat'], columns='policy', values='service_total_time_ns')
legacy_rs = gm(legacy_piv['repressdb_frozen'] / legacy_piv['shape_ewma_be_frozen'])

macro = []
macro.append('% Auto-generated from frozen CSV/JSON evidence. Do not edit manually.')
macro.append(r'\newcommand{\EvidenceFreezeVOneFiveOne}{19aad963a625e14c2298bb783bc4885dff062ae17427200f0f1cfbcb7713ed39}')
macro.append(r'\newcommand{\EvidenceFreezeVOneSevenZero}{ff773467a5de296e07dcbaac904b2d26c466a815ca7b16bdcddbff527a2c4088}')
for key, value in ratios.items():
    macro.append(fr'\newcommand{{\{key}Ratio}}{{{value["gm"]:.4f}}}')
    macro.append(fr'\newcommand{{\{key}Low}}{{{value["low"]:.4f}}}')
    macro.append(fr'\newcommand{{\{key}High}}{{{value["high"]:.4f}}}')
    macro.append(fr'\newcommand{{\{key}Wins}}{{{value["wins"]}}}')
macro += [
    fr'\newcommand{{\ReplayRepressMigrations}}{{{int(agg.loc["repressdb_frozen", "migration_count"])}}}',
    fr'\newcommand{{\ReplayShapeMigrations}}{{{int(agg.loc["shape_ewma_be_frozen", "migration_count"])}}}',
    fr'\newcommand{{\ReplayRepressUnamortized}}{{{100*agg.loc["repressdb_frozen", "unamortized_rate"]:.2f}\%}}',
    fr'\newcommand{{\ReplayShapeUnamortized}}{{{100*agg.loc["shape_ewma_be_frozen", "unamortized_rate"]:.2f}\%}}',
    fr'\newcommand{{\ReplayRepressStale}}{{{100*agg.loc["repressdb_frozen", "stale_rate"]:.2f}\%}}',
    fr'\newcommand{{\ReplayShapeStale}}{{{100*agg.loc["shape_ewma_be_frozen", "stale_rate"]:.2f}\%}}',
    fr'\newcommand{{\EndToEndQueries}}{{{int(vass["total_measured_queries"]):,}}}',
    fr'\newcommand{{\EndToEndPairs}}{{{int(vass["pair_count"])}}}',
    fr'\newcommand{{\EndToEndWorkers}}{{{int(vass["worker_count"])}}}',
    fr'\newcommand{{\EndToEndRepressShapeRatio}}{{{rs.geometric_mean_ratio:.4f}}}',
    fr'\newcommand{{\EndToEndRepressShapeLow}}{{{rs.ci_low:.4f}}}',
    fr'\newcommand{{\EndToEndRepressShapeHigh}}{{{rs.ci_high:.4f}}}',
    fr'\newcommand{{\EndToEndRepressShapeMedian}}{{{rs.median_ratio:.4f}}}',
    fr'\newcommand{{\EndToEndRepressInitialRatio}}{{{ri.geometric_mean_ratio:.4f}}}',
    fr'\newcommand{{\EndToEndRepressInitialLow}}{{{ri.ci_low:.4f}}}',
    fr'\newcommand{{\EndToEndRepressInitialHigh}}{{{ri.ci_high:.4f}}}',
    fr'\newcommand{{\EndToEndRepressShapePnnRatio}}{{{rsp99.geometric_mean_ratio:.4f}}}',
    fr'\newcommand{{\EndToEndRepressShapePnnLow}}{{{rsp99.ci_low:.4f}}}',
    fr'\newcommand{{\EndToEndRepressShapePnnHigh}}{{{rsp99.ci_high:.4f}}}',
    fr'\newcommand{{\EndToEndRepressInitialPnnRatio}}{{{rip99.geometric_mean_ratio:.4f}}}',
    fr'\newcommand{{\EndToEndRepressInitialPnnLow}}{{{rip99.ci_low:.4f}}}',
    fr'\newcommand{{\EndToEndRepressInitialPnnHigh}}{{{rip99.ci_high:.4f}}}',
    fr'\newcommand{{\LegacyDiagnosticRepressShape}}{{{legacy_rs:.4f}}}',
    r'\newcommand{\CombinedRecoveryTrials}{1,100}',
    fr'\newcommand{{\CalibrationSeconds}}{{{cal["wall_elapsed_ns"]/1e9:.2f}}}',
    fr'\newcommand{{\CalibrationReadMB}}{{{cal["process_read_bytes"]/1e6:.2f}}}',
    fr'\newcommand{{\CalibrationWriteMB}}{{{cal["process_write_bytes"]/1e6:.2f}}}',
    fr'\newcommand{{\CalibrationCandidateRatio}}{{{cal["candidate_storage_over_raw_ratio"]:.2f}}}',
]
for policy, name in [('initial_best', 'Initial'), ('shape_ewma_be_frozen', 'Shape'), ('repressdb_frozen', 'Repress')]:
    s = impl_row(impl_service, policy)
    d = impl_row(impl_dispatch, policy)
    p = impl_row(impl_p99, policy)
    macro += [
        fr'\newcommand{{\Cache{name}ServiceRatio}}{{{s.geometric_mean_ratio:.4f}}}',
        fr'\newcommand{{\Cache{name}ServiceLow}}{{{s.ci_low:.4f}}}',
        fr'\newcommand{{\Cache{name}ServiceHigh}}{{{s.ci_high:.4f}}}',
        fr'\newcommand{{\Cache{name}DispatchRatio}}{{{d.geometric_mean_ratio:.4f}}}',
        fr'\newcommand{{\Cache{name}PnnRatio}}{{{p.geometric_mean_ratio:.4f}}}',
    ]
write(ROOT / 'numbers_frozen.tex', '\n'.join(macro) + '\n')

# -------------------------------------------------------------------------
# Multi-seed measured-cost table
# -------------------------------------------------------------------------
trace_labels = {
    'R1_A_gradual_periodic': 'Gradual--periodic',
    'R1_B_mean_reverting': 'Mean-reverting',
    'R1_C_noisy_bursts': 'Noisy bursts',
}
rows = []
for (trace, size), group in mg.groupby(['trace', 'chunk_size'], sort=True):
    p = group.pivot(index='trace_seed', columns='policy', values='total_time_ns')
    rs_ = gm(p['repressdb_frozen'] / p['shape_ewma_be_frozen'])
    ri_ = gm(p['repressdb_frozen'] / p['initial_best'])
    rp_ = gm(p['repressdb_frozen'] / p['page_hinkley_be'])
    rg = group[group.policy == 'repressdb_frozen']
    sg = group[group.policy == 'shape_ewma_be_frozen']
    rows.append((trace_labels[trace], int(size/1024), rs_, ri_, rp_, int(rg.migration_count.sum()), int(sg.migration_count.sum())))

text = [
    r'\begin{table*}[t]',
    r'\centering',
    r'\caption{Multi-seed measured-cost replay. Each ratio is an equal-weight geometric mean over three independently generated trace realizations fixed before policy execution.}',
    r'\label{tab:multiseed_cost}',
    r'\begin{threeparttable}',
    r'\begin{tabular}{@{}lrrrrrr@{}}',
    r'\toprule',
    r'Trace & Size & RePress/Shape & RePress/initial & RePress/PH-BE & RePress mig. & Shape mig. \\',
    r'\midrule',
]
for row in rows:
    text.append(f'{row[0]} & {row[1]}K & {row[2]:.4f} & {row[3]:.4f} & {row[4]:.4f} & {row[5]} & {row[6]} \\\\')
text += [
    r'\midrule',
    f'All cells & -- & {ratios["ReplayRepressShape"]["gm"]:.4f} & {ratios["ReplayRepressInitial"]["gm"]:.4f} & {ratios["ReplayRepressPH"]["gm"]:.4f} & {int(agg.loc["repressdb_frozen", "migration_count"])} & {int(agg.loc["shape_ewma_be_frozen", "migration_count"])} \\\\',
    r'\bottomrule',
    r'\end{tabular}',
    r'\begin{tablenotes}[flushleft]\footnotesize',
    r'\item PH-BE denotes the Page--Hinkley break-even baseline configured on PEMS04. Ratios below one favor RePressDB. File-backed service time is evaluated separately in RQ2.',
    r'\end{tablenotes}',
    r'\end{threeparttable}',
    r'\end{table*}',
]
write(ROOT / 'table_multiseed_generated.tex', '\n'.join(text) + '\n')

# Unified ablation table
variant_labels = {
    'full_repressdb': 'Full RePressDB',
    'no_break_even': 'No break-even',
    'no_confirmation': 'No confirmation',
    'short_window_only': 'Short-window only',
    'no_cooldown': 'No cooldown',
    'no_write_budget': 'No write budget',
}
text = [
    r'\begin{table*}[t]', r'\centering',
    r'\caption{Unified six-cell ablation. Cost ratios are relative to full RePressDB. The reported measures jointly characterize replay cost and migration selectivity.}',
    r'\label{tab:unified_ablation}',
    r'\begin{tabular}{@{}lrrrrr@{}}',
    r'\toprule',
    r'Variant & Cost ratio & 95\% CI & Migrations & Median unamortized & Median stale \\',
    r'\midrule',
]
for _, r in ua.iterrows():
    text.append(f'{variant_labels[r.variant]} & {r.geometric_mean_total_ratio_to_full:.4f} & [{r.ci_low:.4f}, {r.ci_high:.4f}] & {int(r.total_migrations)} & {100*r.median_unamortized_rate:.2f}\\% & {100*r.median_stale_rate:.2f}\\% \\\\')
text += [r'\bottomrule', r'\end{tabular}', r'\end{table*}']
write(ROOT / 'table_ablation_generated.tex', '\n'.join(text) + '\n')


# -------------------------------------------------------------------------
# Figures
# -------------------------------------------------------------------------
plt.rcParams.update({'font.size': 9, 'font.family': 'serif'})

# Figure 1: architecture + state machine
fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
ax = axes[0]; ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis('off')

def box(ax, x, y, w, h, text, fc='#f4f4f4', ec='#333333', lw=1.2, fs=9):
    patch = FancyBboxPatch((x,y), w,h, boxstyle='round,pad=0.03,rounding_size=0.08', facecolor=fc, edgecolor=ec, linewidth=lw)
    ax.add_patch(patch); ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=fs)
    return patch

def arrow(ax, x1,y1,x2,y2, text=None, rad=0.0):
    a=FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=10,linewidth=1.0,color='#333333',connectionstyle=f'arc3,rad={rad}')
    ax.add_patch(a)
    if text: ax.text((x1+x2)/2,(y1+y2)/2+0.2,text,ha='center',va='center',fontsize=7)

box(ax,0.3,7.8,2.2,1.0,'Query API')
box(ax,3.0,7.8,2.5,1.0,'Active-view cache\n(chunk, version)','#e8f1f8')
box(ax,6.0,7.8,3.4,1.0,'Representation executor\nLZ4 / Zstd / blocks')
arrow(ax,2.5,8.3,3.0,8.3); arrow(ax,5.5,8.3,6.0,8.3)
box(ax,0.3,4.8,2.2,1.1,'Query-shape monitor\nshort + long windows','#f8f0df')
box(ax,3.0,4.8,2.5,1.1,'Risk-aware controller\nconfirmation + BE + budget','#f8f0df')
box(ax,6.0,4.8,3.4,1.1,'Migration coordinator\nencode, verify, publish, commit','#f8f0df')
arrow(ax,1.4,7.8,1.4,5.9,'query class'); arrow(ax,2.5,5.35,3.0,5.35); arrow(ax,5.5,5.35,6.0,5.35)
box(ax,0.3,1.5,2.8,1.2,'SQLite WAL catalog\nactive + previous + state','#e8f8ec')
box(ax,3.7,1.5,2.8,1.2,'Versioned payloads\nand block metadata','#e8f8ec')
box(ax,7.1,1.5,2.3,1.2,'Recovery manager\nstartup + fsck','#e8f8ec')
arrow(ax,7.0,4.8,2.4,2.7,'catalog transaction',rad=0.08); arrow(ax,8.3,4.8,5.1,2.7,'fsync + replace',rad=-0.05)
arrow(ax,8.25,2.7,8.25,4.8,'invalidate cache',rad=0.08)
arrow(ax,7.1,2.1,6.5,2.1); arrow(ax,7.1,2.4,3.1,2.4,rad=0.08)
ax.set_title('(a) Components and execution boundaries',fontweight='bold')

ax = axes[1]; ax.set_xlim(0, 12); ax.set_ylim(0, 8); ax.axis('off')
states=[('idle',0.3),('intent',2.1),('encoded',4.0),('validated',5.9),('published',7.8),('committed\npending cleanup',9.7)]
for s,x in states:
    box(ax,x,4.9,1.5,1.0,s,'#f4f4f4')
for i in range(len(states)-1): arrow(ax,states[i][1]+1.5,5.4,states[i+1][1],5.4)
arrow(ax,11.2,4.9,1.05,4.1,'cleanup → idle',rad=0.35)
box(ax,0.5,1.1,3.1,1.5,'Pre-commit interruption\nold active remains visible;\nstaged/published target removed','#fff2e8')
box(ax,4.45,1.1,3.1,1.5,'Post-commit validation\nnew active checked while\nprevious version is retained','#fff2e8')
box(ax,8.4,1.1,3.1,1.5,'Invalid committed target\nrollback to validated previous;\notherwise finish cleanup','#fff2e8')
arrow(ax,2.1,4.9,2.1,2.6); arrow(ax,7.0,4.9,6.0,2.6); arrow(ax,10.45,4.9,9.9,2.6)
ax.text(0.4,7.2,'Invariants',fontweight='bold')
ax.text(0.4,6.7,'I1  Every catalog-visible active version has a complete, checksum-validated payload.',fontsize=8)
ax.text(0.4,6.25,'I2  At most one version is active; the previous version remains non-active until cleanup.',fontsize=8)
ax.text(0.4,5.8,'I3  Recovery terminates in idle state or reports an integrity failure; it never silently serves an invalid active.',fontsize=8)
ax.set_title('(b) Migration states and recovery obligations',fontweight='bold')
fig.tight_layout()
fig.savefig(ROOT/'Figure_1_architecture_state_machine.pdf',bbox_inches='tight')
fig.savefig(ROOT/'Figure_1_architecture_state_machine.png',dpi=220,bbox_inches='tight')
plt.close(fig)

# Figure 4 multiseed ratios
fig, ax = plt.subplots(figsize=(8.5,4.6))
traces=list(trace_labels.keys()); x=np.arange(6)
labels=[]; means=[]; lows=[]; highs=[]
for trace in traces:
    for size in [8192,16384]:
        g=mg[(mg.trace==trace)&(mg.chunk_size==size)]
        p=g.pivot(index='trace_seed',columns='policy',values='total_time_ns')
        r=(p.repressdb_frozen/p.shape_ewma_be_frozen).to_numpy()
        labels.append(f'{trace_labels[trace]}\n{size//1024}K')
        means.append(gm(r)); lows.append(np.min(r)); highs.append(np.max(r))
        xi=len(means)-1
        ax.scatter(np.repeat(xi,len(r)),r,s=22,zorder=3)
ax.errorbar(x,means,yerr=[np.array(means)-np.array(lows),np.array(highs)-np.array(means)],fmt='o',capsize=4,linewidth=1.2,label='geometric mean and seed range')
ax.axhline(1.0,color='black',linewidth=1,linestyle='--')
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel('Measured-cost ratio: RePressDB / Shape-EWMA-BE')
ax.set_ylim(0.90,1.02); ax.grid(axis='y',alpha=.25)
ax.legend(frameon=False,loc='lower right')
fig.tight_layout(); fig.savefig(ROOT/'Figure_4_multiseed_total_cost.pdf',bbox_inches='tight'); fig.savefig(ROOT/'Figure_4_multiseed_total_cost.png',dpi=220,bbox_inches='tight'); plt.close(fig)

# Figure 5 multiseed migration risks
fig, axes=plt.subplots(1,3,figsize=(10.5,3.8))
policies=['afhc_d_frozen','page_hinkley_be','shape_ewma_be_frozen','repressdb_frozen']
labels=['AF-HC-D','PH-BE','Shape','RePressDB']
axes[0].bar(labels,[agg.loc[p,'migration_count'] for p in policies]); axes[0].set_ylabel('Committed migrations'); axes[0].tick_params(axis='x',rotation=20)
axes[1].bar(labels,[100*agg.loc[p,'unamortized_rate'] for p in policies]); axes[1].set_ylabel('Unamortized migrations (%)'); axes[1].tick_params(axis='x',rotation=20)
axes[2].bar(labels,[100*agg.loc[p,'stale_rate'] for p in policies]); axes[2].set_ylabel('Stale migrations (%)'); axes[2].tick_params(axis='x',rotation=20)
for ax in axes: ax.grid(axis='y',alpha=.25)
fig.tight_layout(); fig.savefig(ROOT/'Figure_5_multiseed_migration_risk.pdf',bbox_inches='tight'); fig.savefig(ROOT/'Figure_5_multiseed_migration_risk.png',dpi=220,bbox_inches='tight'); plt.close(fig)

# Figure 9 active-view ablation and end-to-end policy comparison
fig, axes=plt.subplots(1,3,figsize=(11.5,3.9))
pl=['initial_best','shape_ewma_be_frozen','repressdb_frozen']; ll=['Initial','Shape','RePressDB']
for ax,df,title,ylabel in [
    (axes[0],impl_service,'Active view / uncached service time','Service-time ratio'),
    (axes[1],impl_dispatch,'Active view / uncached dispatch time','Dispatch-time ratio'),
]:
    vals=[]; lo=[]; hi=[]
    for p in pl:
        r=impl_row(df,p); vals.append(r.geometric_mean_ratio); lo.append(r.ci_low); hi.append(r.ci_high)
    xx=np.arange(3)
    ax.errorbar(xx,vals,yerr=[np.array(vals)-np.array(lo),np.array(hi)-np.array(vals)],fmt='o',capsize=4)
    ax.axhline(1,color='black',linestyle='--',linewidth=1); ax.set_xticks(xx); ax.set_xticklabels(ll); ax.set_ylabel(ylabel); ax.set_title(title); ax.grid(axis='y',alpha=.25)
# policy comparison using the active-view path
comparisons=[('RePress / Shape',rs),('RePress / initial',ri),('RePress / Shape p99',rsp99),('RePress / initial p99',rip99)]
vals=[r.geometric_mean_ratio for _,r in comparisons]; lo=[r.ci_low for _,r in comparisons]; hi=[r.ci_high for _,r in comparisons]
xx=np.arange(len(vals)); axes[2].errorbar(xx,vals,yerr=[np.array(vals)-np.array(lo),np.array(hi)-np.array(vals)],fmt='o',capsize=4); axes[2].axhline(1,color='black',linestyle='--',linewidth=1); axes[2].set_xticks(xx); axes[2].set_xticklabels([x[0] for x in comparisons],rotation=25,ha='right'); axes[2].set_ylabel('Paired ratio'); axes[2].set_title('Policy comparison with active-view execution'); axes[2].grid(axis='y',alpha=.25)
fig.tight_layout(); fig.savefig(ROOT/'Figure_9_active_view_end_to_end.pdf',bbox_inches='tight'); fig.savefig(ROOT/'Figure_9_active_view_end_to_end.png',dpi=220,bbox_inches='tight'); plt.close(fig)

# Figure 10 calibration robustness
fig, axes=plt.subplots(1,2,figsize=(8.4,3.7))
for policy,label in [('repressdb_frozen','RePressDB'),('shape_ewma_be_frozen','Shape')]:
    g=sp[sp.policy==policy]
    axes[0].plot(100*g.sample_fraction,100*(g.geometric_mean_total_ratio-1),marker='o',label=label)
axes[0].axhline(0,color='black',linewidth=1)
axes[0].set_xlabel('Chunks used for generic profile (%)')
axes[0].set_ylabel('True-cost change (%)')
axes[0].set_title('(a) Generic sampled profile',fontweight='bold')
axes[0].legend(frameon=False)
for policy,label in [('repressdb_frozen','RePressDB'),('shape_ewma_be_frozen','Shape')]:
    g=ce[(ce.policy==policy)&(ce.error_level>0)]
    axes[1].plot(100*g.error_level,100*(g.geometric_mean_total_ratio-1),marker='o',label=label)
axes[1].axhline(0,color='black',linewidth=1)
axes[1].set_xlabel('Independent bounded error (±%)')
axes[1].set_ylabel('True-cost change (%)')
axes[1].set_title('(b) Cost-table perturbation',fontweight='bold')
axes[1].legend(frameon=False)
for ax in axes:
    ax.grid(axis='y',alpha=.25)
fig.tight_layout()
fig.savefig(ROOT/'Figure_10_recovery_calibration.pdf',bbox_inches='tight')
fig.savefig(ROOT/'Figure_10_recovery_calibration.png',dpi=220,bbox_inches='tight')
plt.close(fig)

# Numeric audit
checks = {
    'v170_all_ten': bool(vass['acceptance_checks']['all_ten']),
    'all_54_migration_signatures_match': bool(veq.exact_migration_signature_match.all() and len(veq)==54),
    'multiseed_18_cells': bool(len(piv)==18),
    'crash_fuzz_500_pass': bool(cf.passed.sum()==500 and cf.trials.sum()==500),
    'repress_shape_replay_18_of_18': bool((piv.repressdb_frozen < piv.shape_ewma_be_frozen).all()),
}
if not all(checks.values()):
    raise RuntimeError(checks)
manifest=[]
for path in sorted(ROOT.glob('Figure_2_reconfiguration_lifecycle.*'))+sorted(ROOT.glob('Figure_8_interference_core.*'))+sorted(ROOT.glob('Figure_9_*'))+sorted(ROOT.glob('Figure_10_*'))+[ROOT/'numbers_frozen.tex',ROOT/'table_multiseed_generated.tex',ROOT/'table_ablation_generated.tex']:
    manifest.append({'file':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size})
write(ROOT/'paper_asset_manifest.json',json.dumps({'checks':checks,'files':manifest},indent=2))
print(json.dumps(checks,indent=2))
