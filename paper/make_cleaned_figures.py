from __future__ import annotations
from pathlib import Path
import math, hashlib
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'experiment_data'

PAL = {
    'ink': '#202226',
    'muted': '#68707C',
    'grid': '#D8D4CF',
    'spine': '#8A8580',
    'blue': '#5271B6',
    'blue_light': '#D3DCF0',
    'coral': '#DE837D',
    'coral_light': '#F2B7BE',
    'yellow': '#E2B83F',
    'yellow_light': '#F6E295',
    'gray': '#A6B0C0',
    'gray_light': '#EEF3FA',
    'rose': '#C58A86',
    'olive': '#8D935E',
}

mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Tinos', 'TeX Gyre Termes', 'Nimbus Roman', 'Liberation Serif', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 8.7,
    'axes.labelsize': 8.7,
    'xtick.labelsize': 7.7,
    'ytick.labelsize': 7.7,
    'legend.fontsize': 7.3,
    'axes.linewidth': 0.8,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'savefig.pad_inches': 0.04,
    'axes.unicode_minus': False,
})


def save(fig, name, dpi=360):
    fig.savefig(ROOT / f'{name}.pdf', bbox_inches='tight')
    fig.savefig(ROOT / f'{name}.png', dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def style(ax, grid='y'):
    ax.set_facecolor('white')
    for side in ['top', 'right']:
        ax.spines[side].set_visible(False)
    ax.spines['left'].set_color(PAL['spine'])
    ax.spines['bottom'].set_color(PAL['spine'])
    ax.tick_params(direction='out', color=PAL['spine'], length=3.0, width=0.8)
    if grid:
        ax.grid(axis=grid, color=PAL['grid'], lw=0.7)
        ax.set_axisbelow(True)


def panel(ax, text):
    ax.text(0.0, 1.02, text, transform=ax.transAxes, ha='left', va='bottom', fontsize=9.0, fontweight='bold', color=PAL['ink'])


def lighten(c, frac=0.72):
    rgb = np.array(to_rgb(c), float)
    return tuple(rgb + (1-rgb)*frac)


def gradient_bar(ax, xcenter, height, width, color, bottom=0.0, edgecolor=None, alpha_bottom=0.20, alpha_top=0.92, steps=30):
    if edgecolor is None:
        edgecolor = color
    left = xcenter - width/2
    if abs(height) < 1e-12:
        ax.add_patch(Rectangle((left, bottom), width, 0, fc='none', ec=edgecolor, lw=0.8))
        return
    sign = 1 if height >= 0 else -1
    step_h = abs(height)/steps
    for i in range(steps):
        alpha = alpha_bottom + (alpha_top-alpha_bottom)*(i/max(1, steps-1))
        if sign > 0:
            y0 = bottom + i*step_h
        else:
            y0 = bottom - (i+1)*step_h
            alpha = alpha_top + (alpha_bottom-alpha_top)*(i/max(1, steps-1))
        ax.add_patch(Rectangle((left, y0), width, step_h, fc=color, ec='none', alpha=alpha, zorder=2))
    ax.add_patch(Rectangle((left, min(bottom, bottom+height)), width, abs(height), fc='none', ec=edgecolor, lw=0.8, zorder=2.2))


def box(ax, x, y, w, h, text, fs=8.2, weight='normal', lw=1.1):
    r = Rectangle((x,y), w,h, facecolor='white', edgecolor=PAL['blue'], linewidth=lw)
    ax.add_patch(r)
    ax.text(x+w/2, y+h/2, text, ha='center', va='center', fontsize=fs, fontweight=weight, color=PAL['ink'], linespacing=1.15)
    return (x,y,w,h)


def arrow(ax, p1, p2, lw=1.15, color=None, mutation=10):
    if color is None:
        color = PAL['blue']
    a = FancyArrowPatch(p1, p2, arrowstyle='->', mutation_scale=mutation, linewidth=lw, color=color,
                        shrinkA=0, shrinkB=0, connectionstyle='arc3,rad=0')
    ax.add_patch(a)
    return a


def elbow_arrow(ax, pts, lw=1.15, color=None, mutation=10):
    if color is None:
        color = PAL['blue']
    for p, q in zip(pts[:-2], pts[1:-1]):
        ax.plot([p[0], q[0]], [p[1], q[1]], color=color, lw=lw, solid_capstyle='butt')
    p = pts[-2]; q = pts[-1]
    arrow(ax, p, q, lw=lw, color=color, mutation=mutation)


def gm(values):
    a = np.asarray(values, float)
    a = a[np.isfinite(a) & (a > 0)]
    return float(np.exp(np.mean(np.log(a))))


def bootstrap_ci(values, seed=20260719, n=12000):
    a = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n, len(a)))
    est = np.exp(np.mean(np.log(a[idx]), axis=1))
    return tuple(float(x) for x in np.quantile(est,[0.025,0.975]))

# data
mg = pd.read_csv(DATA/'multiseed_global_results.csv')
sp = pd.read_csv(DATA/'sampled_profile_summary.csv')
ce = pd.read_csv(DATA/'cost_error_summary.csv')
vstats = pd.read_csv(DATA/'v170_implementation_paired_statistics.csv')
vpol = pd.read_csv(DATA/'v170_policy_statistics_by_implementation.csv')
r1c = pd.read_csv(ROOT/'r1_cumulative_cost_curves.csv')
s1 = pd.read_csv(ROOT/'s1_holdout_global_comparison_summary.csv')
b3c = pd.read_csv(ROOT/'b3c_balanced_interference_summary.csv')

# ---------- Figure 1: architecture ----------
fig, ax = plt.subplots(figsize=(7.15,4.55))
fig.patch.set_facecolor('white')
ax.set_xlim(0,12); ax.set_ylim(0,7.2); ax.axis('off')
# row 1
b_monitor=box(ax,0.35,5.30,2.15,0.90,'Workload\nmonitor',fs=8.5,weight='bold')
b_dec=box(ax,3.15,5.30,2.15,0.90,'Adaptation\ndecision',fs=8.5,weight='bold')
b_reconf=box(ax,6.00,5.30,2.35,0.90,'Reconfiguration\nmanager',fs=8.5,weight='bold')
b_rec=box(ax,9.05,5.30,2.15,0.90,'Recovery\nmanager',fs=8.5,weight='bold')
# row 2
b_ev=box(ax,0.35,3.25,2.15,1.25,'Workload\nevidence\nshort / long\nwindows',fs=8.0)
b_adm=box(ax,3.15,3.05,2.15,1.65,'Admission logic\nconfirmation\nbreak-even\nwrite budget',fs=7.9)
b_exec=box(ax,6.00,3.05,2.35,1.65,'Execution protocol\nprepare\npublish\ncommit\ncleanup',fs=7.9)
b_start=box(ax,9.05,3.25,2.15,1.25,'Startup checks\nrollback /\ncleanup',fs=8.0)
# divider and managed row
ax.plot([0.25,11.55],[2.55,2.55],color=PAL['grid'],lw=0.9)
b_active=box(ax,0.70,0.55,2.75,1.25,'Active view\nversion-keyed metadata\nreader dispatch',fs=8.1)
b_store=box(ax,4.60,0.55,2.85,1.25,'Single-active store\nSQLite catalog +\nversioned payloads',fs=8.1)
b_query=box(ax,8.55,0.55,2.75,1.25,'Query / migration\nexecution path',fs=8.1)
# labels: intentionally close to their groups
ax.text(5.78,6.52,'Managing layer',ha='center',va='center',fontsize=9.7,fontweight='bold',color=PAL['ink'])
ax.text(0.45,2.35,'Managed storage',ha='left',va='center',fontsize=9.4,fontweight='bold',color=PAL['ink'])
# horizontal main paths
arrow(ax,(2.50,5.75),(3.15,5.75)); arrow(ax,(5.30,5.75),(6.00,5.75)); arrow(ax,(8.35,5.75),(9.05,5.75))
arrow(ax,(2.50,3.88),(3.15,3.88)); arrow(ax,(5.30,3.88),(6.00,3.88)); arrow(ax,(8.35,3.88),(9.05,3.88))
arrow(ax,(3.45,1.175),(4.60,1.175)); arrow(ax,(7.45,1.175),(8.55,1.175))
# vertical manager-to-component paths
arrow(ax,(1.425,5.30),(1.425,4.50)); arrow(ax,(4.225,5.30),(4.225,4.70)); arrow(ax,(7.175,5.30),(7.175,4.70)); arrow(ax,(10.125,5.30),(10.125,4.50))
# evidence -> active
elbow_arrow(ax,[(1.425,3.25),(1.425,2.15),(2.075,2.15),(2.075,1.80)])
# execution -> store (top-left of store)
elbow_arrow(ax,[(7.175,3.05),(7.175,2.05),(5.45,2.05),(5.45,1.80)])
# startup -> store (top-right of store); separate route, no shared segment
elbow_arrow(ax,[(10.125,3.25),(10.125,2.25),(6.70,2.25),(6.70,1.80)])
save(fig,'Figure_1_architecture')

# ---------- Figure 1 state / obligation view ----------
fig, ax = plt.subplots(figsize=(7.15,3.75))
fig.patch.set_facecolor('white')
ax.set_xlim(0,12); ax.set_ylim(0,5.2); ax.axis('off')
states=[('Idle',0.35),('Intent',2.15),('Encoded',3.95),('Validated',5.75),('Published',7.55),('Committed',9.35)]
for name,x in states:
    box(ax,x,3.20,1.35,0.72,name,fs=8.3,weight='bold')
for i in range(len(states)-1):
    arrow(ax,(states[i][1]+1.35,3.56),(states[i+1][1],3.56))
# visibility boundary between Validated and Published
bx=7.35
ax.plot([bx,bx],[2.86,4.42],color=PAL['blue'],lw=0.9,ls=(0,(5,3)))
ax.text(bx+0.10,4.32,'Visibility boundary',ha='left',va='bottom',fontsize=7.8,color=PAL['ink'])
# committed -> idle return routed above the state row so it cannot be confused with the obligation callouts
elbow_arrow(ax,[(10.70,3.56),(11.30,3.56),(11.30,4.62),(1.025,4.62),(1.025,3.92)],lw=1.0)
# obligation callouts: annotations, not a second state sequence
box(ax,0.40,0.55,3.05,1.10,'Old active version\nremains visible',fs=8.0,weight='bold')
box(ax,4.45,0.55,3.05,1.10,'Candidate version\nchecked before visibility',fs=8.0,weight='bold')
box(ax,8.55,0.55,3.05,1.10,'Previous version retained\nuntil cleanup',fs=8.0,weight='bold')
# Validated milestone fans out to the two pre-visibility obligations
vx=5.75+1.35/2
ax.plot([vx,vx],[3.20,2.25],color=PAL['blue'],lw=1.0)
ax.plot([1.925,5.975],[2.25,2.25],color=PAL['blue'],lw=1.0)
ax.plot([1.925,1.925],[2.25,1.65],color=PAL['blue'],lw=1.0)
ax.plot([5.975,5.975],[2.25,1.65],color=PAL['blue'],lw=1.0)
# Committed milestone maps to the post-commit retention obligation
cx=9.35+1.35/2
ax.plot([cx,cx],[3.20,1.65],color=PAL['blue'],lw=1.0)
save(fig,'Figure_1_architecture_state_machine')

# ---------- Figure 2: lifecycle ----------
fig, ax = plt.subplots(figsize=(7.15,3.65))
fig.patch.set_facecolor('white')
ax.set_xlim(0,12); ax.set_ylim(0,4.9); ax.axis('off')
steps=[('Prepare',0.25),('Encode',2.15),('Validate',4.05),('Publish',5.95),('Catalog\ncommit',7.85),('Cleanup',9.85)]
for name,x in steps:
    box(ax,x,3.08,1.45,0.76,name,fs=8.2,weight='bold')
for i in range(len(steps)-1):
    arrow(ax,(steps[i][1]+1.45,3.46),(steps[i+1][1],3.46))
bx=7.65
ax.plot([bx,bx],[2.75,4.45],color=PAL['blue'],lw=0.9,ls=(0,(5,3)))
ax.text(bx+0.12,4.33,'Catalog visibility boundary',ha='left',va='bottom',fontsize=7.8,color=PAL['ink'])
# lower explanatory boxes
box(ax,0.25,0.52,3.15,1.35,'Pre-commit interruption\nDiscard candidate and keep the\nprevious active version.',fs=7.25,weight='bold')
box(ax,4.05,0.52,3.15,1.35,'Post-publication check\nValidation runs before the\nnew version becomes visible.',fs=7.25,weight='bold')
box(ax,8.15,0.52,3.20,1.35,'Post-commit recovery\nKeep the previous version until\ncleanup succeeds.',fs=7.25,weight='bold')
# arrows touch source/target rectangles exactly
elbow_arrow(ax,[(0.975,3.08),(0.975,2.25),(1.825,2.25),(1.825,1.87)],lw=1.05)
elbow_arrow(ax,[(4.775,3.08),(4.775,2.25),(5.625,2.25),(5.625,1.87)],lw=1.05)
elbow_arrow(ax,[(10.575,3.08),(10.575,2.25),(9.75,2.25),(9.75,1.87)],lw=1.05)
save(fig,'Figure_2_reconfiguration_lifecycle')

# ---------- Figure 2 continuous traces: accepted 3x3 separated 2D ----------
def stable_seed(*parts) -> int:
    token='|'.join(str(p) for p in parts)
    return int.from_bytes(hashlib.sha256(token.encode('utf-8')).digest()[:8],'little')

def softmax(x):
    s=x-np.max(x,axis=-1,keepdims=True); e=np.exp(s); return e/e.sum(axis=-1,keepdims=True)

def category_probabilities(trace_name,n_events=640000):
    rng=np.random.default_rng(stable_seed(20260728,trace_name,'category'))
    x=np.linspace(0,1,n_events,dtype=np.float64)
    if trace_name=='R1_A_gradual_periodic':
        sigmoid=1/(1+np.exp(-(x-0.50)/0.12))
        point=np.clip(0.05+0.65*sigmoid+0.045*np.sin(2*np.pi*9*x),0.03,0.76)
        aggregate=np.clip(0.16+0.035*np.sin(2*np.pi*4*x+1.1),0.08,0.25)
        scan=np.maximum(1-point-aggregate,0.08)
        p=np.stack([point,scan,aggregate],axis=1); return p/p.sum(axis=1,keepdims=True)
    step=100; knots=math.ceil(n_events/step)+1
    logits=np.zeros((knots,3),float); logits[0]=np.asarray([-0.4,0.9,-0.2]); center=np.asarray([-0.15,0.50,-0.20])
    for i in range(1,knots):
        logits[i]=0.985*logits[i-1]+0.015*center+rng.normal(0,0.075,3)
        logits[i,0]+=0.08*math.sin(2*math.pi*i/170.0); logits[i,2]+=0.06*math.sin(2*math.pi*i/113.0+0.7)
    if trace_name=='R1_C_noisy_bursts':
        pos=np.arange(knots,dtype=float)
        for _ in range(24):
            ci=int(rng.integers(10,knots-10)); width=float(rng.uniform(2,12)); cat=int(rng.integers(0,3)); amp=float(rng.uniform(0.65,1.6))
            logits[:,cat]+=amp*np.exp(-0.5*((pos-ci)/width)**2)
    kp=np.arange(knots)*step; ep=np.arange(n_events)
    interp=np.column_stack([np.interp(ep,kp,logits[:,j]) for j in range(3)])
    return softmax(interp)

trace_order=['R1_A_gradual_periodic','R1_B_mean_reverting','R1_C_noisy_bursts']; trace_short=['R1-A','R1-B','R1-C']
classes=['Point','Scan','Aggregate']; cols=[PAL['blue'],PAL['coral'],PAL['yellow']]
fig, axes=plt.subplots(3,3,figsize=(8.7,6.4),sharex='col',sharey='row'); fig.patch.set_facecolor('white')
for col,(tr,tlabel) in enumerate(zip(trace_order,trace_short)):
    probs=category_probabilities(tr); block=1000; usable=(len(probs)//block)*block
    coarse=probs[:usable].reshape(-1,block,3).mean(axis=1)
    coarse=pd.DataFrame(coarse).rolling(9,center=True,min_periods=1).mean().to_numpy(); x=np.linspace(0,1,len(coarse))
    for row,(cn,c) in enumerate(zip(classes,cols)):
        ax=axes[row,col]; style(ax,grid=None); ax.grid(True,axis='both',color=PAL['grid'],lw=0.7); ax.set_axisbelow(True)
        ax.plot(x,coarse[:,row],color=c,lw=1.0)
        ax.set_xlim(0,1); ax.set_ylim(0,1); ax.set_xticks([0,.25,.5,.75,1]); ax.set_yticks([0,.25,.5,.75,1])
        if row==0: ax.text(.02,1.06,f'({chr(97+col)}) {tlabel}',transform=ax.transAxes,ha='left',va='bottom',fontsize=8.6)
        if col==0: ax.set_ylabel(cn)
        if row==2: ax.set_xlabel('Normalized query time')
fig.subplots_adjust(left=.08,right=.99,bottom=.10,top=.95,hspace=.20,wspace=.18)
save(fig,'Figure_2_continuous_traces')

# ---------- Figure 3: clean-stage holdout ----------
trace_keys=['shape_only','frequency_only','joint_orthogonal','brief_burst','stable_scan']; tlabels=['Shape','Frequency','Joint','Brief','Stable']
fig=plt.figure(figsize=(7.15,4.85)); gs=fig.add_gridspec(2,2,height_ratios=[1.22,1.0],hspace=.42,wspace=.28)
for pi,dataset in enumerate(['PEMS08','BDG2']):
    ax=fig.add_subplot(gs[0,pi]); x=np.arange(len(trace_keys)); width=.32; allvals=[]
    for si,(size,color) in enumerate([(8192,PAL['blue_light']),(16384,PAL['blue'])]):
        vals=[]
        for tr in trace_keys:
            row=s1[(s1.dataset==dataset)&(s1.trace==tr)&(s1.chunk_size==size)].iloc[0]
            vals.append(100*(row.geometric_mean_ratio_repressdb_over_shape_ewma-1))
        allvals.extend(vals)
        xp=x+(si-.5)*width
        for xx,v in zip(xp,vals): gradient_bar(ax,xx,v,width*.88,color,edgecolor=PAL['blue'])
    lo=min(-3.0,min(allvals)-2.0); hi=max(allvals)+4.0
    ax.set_ylim(lo,hi)
    ax.axhline(0,color=PAL['spine'],lw=.9); ax.set_xticks(x,tlabels,rotation=17); ax.set_ylabel('Cost relative to Shape-EWMA-BE (%)'); style(ax); panel(ax,f'({chr(97+pi)})')
    if pi==1:
        ax.legend(handles=[Rectangle((0,0),1,1,fc=PAL['blue_light'],ec=PAL['blue'],label='8K'),Rectangle((0,0),1,1,fc=PAL['blue'],ec=PAL['blue'],label='16K')],frameon=False,ncol=2,loc='upper left')
# lower slope panels from known holdout totals
hold=pd.DataFrame({'dataset':['PEMS08','PEMS08','BDG2','BDG2'],'policy':['Shape-EWMA-BE','RePressDB']*2,'migrations':[532,342,517,309],'rewrite_mb':[18.314880,12.580190,19.137950,11.852998]})
for j,(metric,ylabel) in enumerate([('migrations','Migration count'),('rewrite_mb','Rewritten payload (MB)')]):
    ax=fig.add_subplot(gs[1,j]); xx=np.array([0,1])
    for ds,c,m in [('PEMS08',PAL['coral'],'o'),('BDG2',PAL['blue'],'s')]:
        sub=hold[hold.dataset==ds].set_index('policy'); vals=[float(sub.loc['Shape-EWMA-BE',metric]),float(sub.loc['RePressDB',metric])]
        ax.plot(xx,vals,color=c,lw=1.15,marker=m,ms=4.8,label=ds)
        for x0,v in zip(xx,vals): ax.text(x0,v+(8 if metric=='migrations' else .28),f'{v:.0f}' if metric=='migrations' else f'{v:.1f}',ha='center',fontsize=6.7)
    ax.set_xticks(xx,['Shape-EWMA-BE','RePressDB']); ax.set_ylabel(ylabel); ax.set_xlim(-.28,1.28); style(ax); panel(ax,f'({chr(99+j)})')
    if j==0: ax.legend(frameon=False,loc='upper right')
fig.subplots_adjust(left=.09,right=.99,bottom=.12,top=.95)
save(fig,'Figure_3_strong_baseline_tradeoff')

# ---------- Figure 4 multiseed cost ----------
fig, ax=plt.subplots(figsize=(6.2,3.55)); style(ax)
centers=np.array([0,1.55,3.10]); offs={8192:-.13,16384:.13}; colors={8192:PAL['coral'],16384:PAL['blue']}
for ti,tr in enumerate(trace_order):
    for size in [8192,16384]:
        g=mg[(mg.trace==tr)&(mg.chunk_size==size)]; piv=g.pivot(index='trace_seed',columns='policy',values='total_time_ns'); r=(piv['repressdb_frozen']/piv['shape_ewma_be_frozen']).to_numpy(); x0=centers[ti]+offs[size]
        for j,v in enumerate(r): ax.scatter(x0+(j-1)*.025,v,s=20,color=lighten(colors[size],.10),edgecolor=colors[size],zorder=3)
        mean=gm(r); lo,hi=bootstrap_ci(r,20260721+ti*2+(size==16384)); ax.errorbar(x0,mean,yerr=[[mean-lo],[hi-mean]],fmt='o',color=colors[size],ecolor=colors[size],capsize=3,ms=4.6,lw=1.0)
ax.axhline(1,color=PAL['spine'],lw=.9,ls=(0,(4,2))); ax.set_xticks(centers,['R1-A','R1-B','R1-C']); ax.set_ylabel('Measured-cost ratio\nRePressDB / Shape-EWMA-BE'); ax.set_xlim(-.55,3.65); ax.set_ylim(.90,1.015)
ax.legend(handles=[Line2D([0],[0],marker='o',color='none',markerfacecolor=lighten(PAL['coral'],.10),markeredgecolor=PAL['coral'],label='8K',ms=5),Line2D([0],[0],marker='o',color='none',markerfacecolor=lighten(PAL['blue'],.10),markeredgecolor=PAL['blue'],label='16K',ms=5)],frameon=False,ncol=2,loc='lower left')
save(fig,'Figure_4_multiseed_total_cost')

# ---------- Figure 5 migration risk ----------
agg=mg.groupby('policy').agg(migration_count=('migration_count','sum'),unamortized=('unamortized_migration_count','sum'),stale=('stale_migration_count','sum'),assessable=('stale_assessable_migration_count','sum'))
agg['unamortized_rate']=100*agg['unamortized']/agg['migration_count']; agg['stale_rate']=100*agg['stale']/agg['assessable']
policies=['afhc_d_frozen','page_hinkley_be','shape_ewma_be_frozen','repressdb_frozen']; plabels=['AF-HC-D','PH-BE','Shape-EWMA-BE','RePressDB']; pcols=[PAL['yellow_light'],PAL['gray'],PAL['coral'],PAL['blue']]
fig,axes=plt.subplots(1,3,figsize=(9.0,3.05)); xpos=np.array([0,1.15,2.30,3.45]); bw=.72
for i,(ax,(metric,ylabel)) in enumerate(zip(axes,[('migration_count','Committed migrations'),('unamortized_rate','Unamortized migrations (%)'),('stale_rate','Stale migrations (%)')])):
    vals=[float(agg.loc[p,metric]) for p in policies]
    for x0,v,c in zip(xpos,vals,pcols): gradient_bar(ax,x0,v,bw,c,edgecolor=c)
    pad=18 if metric=='migration_count' else 1.1
    for x0,v in zip(xpos,vals): ax.text(x0,v+pad,f'{v:.0f}' if metric=='migration_count' else f'{v:.2f}',ha='center',fontsize=6.8)
    ax.set_ylim(0,max(vals)*(1.12 if metric=='migration_count' else 1.16)); ax.set_xlim(xpos[0]-.55,xpos[-1]+.55); ax.set_xticks(xpos,plabels,rotation=15); ax.set_ylabel(ylabel); style(ax); panel(ax,f'({chr(97+i)})')
fig.subplots_adjust(left=.065,right=.995,bottom=.24,top=.94,wspace=.30)
save(fig,'Figure_5_multiseed_migration_risk')

# ---------- Figure 6 cumulative behavior ----------
selected=r1c[(r1c.trace=='R1_C_noisy_bursts')&(r1c.chunk_size==16384)&r1c.policy.isin(['best_static','initial_best','afhc_d_frozen','shape_ewma_be_frozen','repressdb_frozen'])].copy()
fig,axes=plt.subplots(1,2,figsize=(7.15,2.85))
line_specs=[('best_static','Best static',PAL['olive'],'-'),('initial_best','Initial-best',PAL['gray'],'--'),('afhc_d_frozen','AF-HC-D',PAL['yellow'],'-'),('shape_ewma_be_frozen','Shape-EWMA-BE',PAL['coral'],'-'),('repressdb_frozen','RePressDB',PAL['blue'],'-')]
for p,l,c,ls in line_specs:
    sub=selected[selected.policy==p].sort_values('global_event_index'); axes[0].plot(sub.global_event_index/1000,sub.cumulative_total_time_ns/1e9,color=c,ls=ls,lw=1.25 if p=='repressdb_frozen' else 1.0,label=l)
axes[0].set_xlabel('Processed queries (thousands)'); axes[0].set_ylabel('Cumulative measured cost (s)'); style(axes[0]); panel(axes[0],'(a)'); axes[0].legend(frameon=False,ncol=2,loc='upper left')
for p,l,c in [('afhc_d_frozen','AF-HC-D',PAL['yellow']),('shape_ewma_be_frozen','Shape-EWMA-BE',PAL['coral']),('repressdb_frozen','RePressDB',PAL['blue'])]:
    sub=selected[selected.policy==p].sort_values('global_event_index'); axes[1].step(sub.global_event_index/1000,sub.cumulative_migrations,where='post',color=c,lw=1.15,label=l)
axes[1].set_xlabel('Processed queries (thousands)'); axes[1].set_ylabel('Cumulative migrations'); style(axes[1]); panel(axes[1],'(b)'); axes[1].legend(frameon=False,loc='upper left')
fig.subplots_adjust(left=.09,right=.99,bottom=.20,top=.93,wspace=.28)
save(fig,'Figure_6_cumulative_behavior')

# ---------- Figure 8 interference core ----------
order_scales=['logical_1GiB','logical_5GiB','logical_10GiB']; scale_labels={'logical_1GiB':'1 GiB','logical_5GiB':'5 GiB','logical_10GiB':'10 GiB'}
rows=[]
for s in order_scales:
    for threads in [1,4,8]: rows.append((s,threads,b3c[(b3c.scale_name==s)&(b3c.thread_count==threads)].iloc[0]))
y=np.arange(len(rows))[::-1]; yl=[f"{scale_labels[s]} / {t} thread{'s' if t>1 else ''}" for s,t,_ in rows]
fig,axes=plt.subplots(1,2,figsize=(7.15,3.15))
vals=np.array([r.throughput_ratio_geometric_mean for _,_,r in rows]); lo=np.array([r.throughput_ratio_ci_low for _,_,r in rows]); hi=np.array([r.throughput_ratio_ci_high for _,_,r in rows])
axes[0].errorbar(vals,y,xerr=[vals-lo,hi-vals],fmt='o',color=PAL['blue'],ecolor=PAL['blue'],capsize=2.5,ms=4.2,lw=1.0); axes[0].axvline(1,color=PAL['spine'],ls=(0,(4,2)),lw=.9); axes[0].set_yticks(y,yl); axes[0].set_xlabel('Background / foreground throughput'); axes[0].set_xlim(.95,1.005); style(axes[0],grid='x'); panel(axes[0],'(a)')
vals=np.array([r.p99_amplification_geometric_mean for _,_,r in rows]); lo=np.array([r.p99_amplification_ci_low for _,_,r in rows]); hi=np.array([r.p99_amplification_ci_high for _,_,r in rows])
axes[1].errorbar(vals,y,xerr=[vals-lo,hi-vals],fmt='s',color=PAL['coral'],ecolor=PAL['coral'],capsize=2.5,ms=4.0,lw=1.0); axes[1].axvline(1,color=PAL['spine'],ls=(0,(4,2)),lw=.9); axes[1].set_yticks(y,['']*len(y)); axes[1].set_xlabel('Background / foreground p99 latency'); axes[1].set_xlim(.99,1.16); style(axes[1],grid='x'); panel(axes[1],'(b)')
fig.subplots_adjust(left=.18,right=.99,bottom=.18,top=.93,wspace=.24)
save(fig,'Figure_8_interference_core')

# ---------- Figure 9 active view ----------
def prow(df,num,den): return df[(df.numerator_policy==num)&(df.denominator_policy==den)].iloc[0]
def irow(df,p): return df[df.policy==p].iloc[0]
cached_service=vpol[(vpol.implementation=='cached')&(vpol.scope=='aggregate')&(vpol.metric=='service_total_time_ns')]; cached_p99=vpol[(vpol.implementation=='cached')&(vpol.scope=='aggregate')&(vpol.metric=='query_p99_ns')]
impl_service=vstats[(vstats.scope=='policy_aggregate')&(vstats.metric=='service_total_time_ns')]; impl_dispatch=vstats[(vstats.scope=='policy_aggregate')&(vstats.metric=='query_catalog_dispatch_time_ns')]
rs=prow(cached_service,'repressdb_frozen','shape_ewma_be_frozen'); ri=prow(cached_service,'repressdb_frozen','initial_best'); rsp=prow(cached_p99,'repressdb_frozen','shape_ewma_be_frozen'); rip=prow(cached_p99,'repressdb_frozen','initial_best')
fig,axes=plt.subplots(1,3,figsize=(7.15,2.85)); pls=['initial_best','shape_ewma_be_frozen','repressdb_frozen']; labs=['Initial','Shape','RePressDB']; cc=[PAL['gray'],PAL['coral'],PAL['blue']]
for k,(ax,df,ylab) in enumerate([(axes[0],impl_service,'Service-time ratio'),(axes[1],impl_dispatch,'Dispatch-time ratio')]):
    for x0,p,c in zip(range(3),pls,cc):
        r=irow(df,p); ax.errorbar(x0,r.geometric_mean_ratio,yerr=[[r.geometric_mean_ratio-r.ci_low],[r.ci_high-r.geometric_mean_ratio]],fmt='o',color=c,ecolor=c,capsize=3,ms=4.6,lw=1.0)
    ax.axhline(1,color=PAL['spine'],ls=(0,(4,2)),lw=.9); ax.set_xticks(range(3),labs); ax.set_ylabel(ylab); style(ax); panel(ax,f'({chr(97+k)})')
comp=[('RePress/Shape',rs,PAL['blue']),('RePress/Initial',ri,PAL['blue']),('RePress/Shape p99',rsp,PAL['coral']),('RePress/Initial p99',rip,PAL['coral'])]
for x0,(lab,r,c) in enumerate(comp): axes[2].errorbar(x0,r.geometric_mean_ratio,yerr=[[r.geometric_mean_ratio-r.ci_low],[r.ci_high-r.geometric_mean_ratio]],fmt='o',color=c,ecolor=c,capsize=3,ms=4.5,lw=1.0)
axes[2].axhline(1,color=PAL['spine'],ls=(0,(4,2)),lw=.9); axes[2].set_xticks(range(4),[x[0] for x in comp],rotation=19,ha='right'); axes[2].set_ylabel('Paired ratio'); style(axes[2]); panel(axes[2],'(c)')
fig.subplots_adjust(left=.09,right=.99,bottom=.24,top=.93,wspace=.34)
save(fig,'Figure_9_active_view_end_to_end')

# ---------- Figure 10 calibration sensitivity ----------
fig,axes=plt.subplots(1,2,figsize=(7.0,2.75))
for p,l,c in [('repressdb_frozen','RePressDB',PAL['blue']),('shape_ewma_be_frozen','Shape-EWMA-BE',PAL['coral'])]:
    g=sp[sp.policy==p]; axes[0].plot(100*g.sample_fraction,100*(g.geometric_mean_total_ratio-1),marker='o',ms=3.8,color=c,lw=1.1,label=l)
axes[0].axhline(0,color=PAL['spine'],lw=.9,ls=(0,(4,2))); axes[0].set_xlabel('Chunks used for generic profile (%)'); axes[0].set_ylabel('True-cost change (%)'); style(axes[0]); panel(axes[0],'(a)'); axes[0].legend(frameon=False)
for p,l,c in [('repressdb_frozen','RePressDB',PAL['blue']),('shape_ewma_be_frozen','Shape-EWMA-BE',PAL['coral'])]:
    g=ce[(ce.policy==p)&(ce.error_level>0)]; axes[1].plot(100*g.error_level,100*(g.geometric_mean_total_ratio-1),marker='o',ms=3.8,color=c,lw=1.1,label=l)
axes[1].axhline(0,color=PAL['spine'],lw=.9,ls=(0,(4,2))); axes[1].set_xlabel('Independent bounded error (±%)'); axes[1].set_ylabel('True-cost change (%)'); style(axes[1]); panel(axes[1],'(b)')
fig.subplots_adjust(left=.09,right=.99,bottom=.21,top=.93,wspace=.28)
save(fig,'Figure_10_recovery_calibration')

# ---------- graphical abstract ----------
fig,ax=plt.subplots(figsize=(9.5,3.4)); fig.patch.set_facecolor('white'); ax.set_xlim(0,12); ax.set_ylim(0,4.2); ax.axis('off')
# three-stage control path, limited color
for x,title,desc,c in [(0.55,'Observe','query shapes\nshort + long windows',PAL['blue']),(4.55,'Decide','confirmation\nbreak-even\nwrite budget',PAL['coral']),(8.55,'Reconfigure','prepare\npublish\nrecover',PAL['blue'])]:
    ax.add_patch(Rectangle((x,2.05),2.85,1.25,fc='white',ec=c,lw=1.2)); ax.text(x+1.425,2.84,title,ha='center',va='center',fontsize=10,fontweight='bold'); ax.text(x+1.425,2.40,desc,ha='center',va='center',fontsize=8.2,linespacing=1.15)
arrow(ax,(3.40,2.68),(4.55,2.68),lw=1.2); arrow(ax,(7.40,2.68),(8.55,2.68),lw=1.2)
ax.text(1.05,.92,'0.9617',fontsize=14,fontweight='bold',color=PAL['blue']); ax.text(.55,.45,'measured-cost ratio\nRePressDB / Shape-EWMA-BE',fontsize=7.8)
ax.text(5.02,.92,'185 vs 361',fontsize=14,fontweight='bold',color=PAL['coral']); ax.text(4.55,.45,'physical migrations\nRePressDB vs Shape-EWMA-BE',fontsize=7.8)
ax.text(9.05,.92,'1,100',fontsize=14,fontweight='bold',color=PAL['blue']); ax.text(8.55,.45,'tested interruption / corruption\ntrials completed with valid state',fontsize=7.8)
save(fig,'graphical_abstract',dpi=320)

print('Cleaned figures generated in',ROOT)

# Restore final author-supplied architecture/reconfiguration diagrams after regenerating data figures.
# These assets were provided by the authors and are intentionally not reconstructed by this script.
from shutil import copy2 as _copy2
_AUTHOR_DIAGRAM_DIR = ROOT / 'author_supplied_diagrams'
for _stem in ['Figure_1_architecture', 'Figure_1_architecture_state_machine', 'Figure_2_reconfiguration_lifecycle']:
    for _ext in ['.png', '.pdf']:
        _src = _AUTHOR_DIAGRAM_DIR / f'{_stem}{_ext}'
        if _src.exists():
            _copy2(_src, ROOT / f'{_stem}{_ext}')
