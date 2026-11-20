"""Figure 8B: surrogate-normalised tgPAC, model 7.
Plots participant-level distributions after averaging across ROIs.
Run in the Python environment containing pandas and matplotlib.
"""
from pathlib import Path
import re
import warnings
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ---------------- Configuration ----------------
DATA_DIR = Path(r"F:\research project\# study 1\eeg_data\set\Y group\roi_source_analysis\surrogate_analysis\model7_surrogate_z")
PREFIX = 'model7_surrogate_z'
ALPHA = 0.01
PLOT_STYLE = "box"  # "bar" or "box"
ERROR_STYLE = "SD"  # "SD" or "SEM" (bar only)
SHOW_POINTS = False
SUMMARY_FILE = DATA_DIR.parent / "study1_surrogate_roi_summary.csv"
# Set SUMMARY_FILE to your actual CSV location if needed.
# The significance annotations come from Model 7, not tests of the plotted points.
ALLOW_REFERENCE_CONTRASTS = False
# The supplied CSVs may contain contrasts only. If so, use the EMMs
# transcribed from the Model 7 text output supplied on 20 Nov 2026.
# Set False to require EMMs in the CSV exports instead.
ALLOW_REFERENCE_EMMS = True

CONDITIONS = ['FTT', 'De-CRAT baseline', 'De-CRAT adaptation']
STAGES = ['Planning', 'Execution']
COLORS = dict(zip(CONDITIONS, ['#5267D9', '#9950D1', '#D94FAE']))
# stage, condition, EMM, lower 95% CI, upper 95% CI
REFERENCE_EMMS = [
    ('Planning',  'FTT',                 .7998, .542, 1.058),
    ('Planning',  'De-CRAT baseline',    .0402, -.217, .298),
    ('Planning',  'De-CRAT adaptation',  .6418, .384, .900),
    ('Execution', 'FTT',                 .9276, .670, 1.185),
    ('Execution', 'De-CRAT baseline',    .5255, .268, .783),
    ('Execution', 'De-CRAT adaptation', 1.2948, 1.037, 1.552),
]


def norm(s):
    return re.sub(r'[^a-z0-9]', '', str(s).lower())


def condition(s):
    n = norm(s)
    mapping = {'bl': CONDITIONS[0], 'ftt': CONDITIONS[0],
               'mainbaseline': CONDITIONS[1], 'decratbaseline': CONDITIONS[1], 'baseline': CONDITIONS[1],
               'mainadaptation': CONDITIONS[2], 'decratadaptation': CONDITIONS[2], 'adaptation': CONDITIONS[2]}
    if n not in mapping:
        raise ValueError(f'Unknown condition: {s!r}')
    return mapping[n]


def stage(s):
    n = norm(s)
    mapping = {'plan': STAGES[0], 'planning': STAGES[0], 'go': STAGES[1], 'execution': STAGES[1]}
    if n not in mapping:
        raise ValueError(f'Unknown task stage: {s!r}')
    return mapping[n]


def col(df, *names):
    lookup = {norm(c): c for c in df.columns}
    for name in names:
        if norm(name) in lookup:
            return lookup[norm(name)]
    return None


def read_csv(name):
    df = pd.read_csv(DATA_DIR / f'{PREFIX}_{name}.csv')
    print(f'{name}: {list(df.columns)}')
    return df


def numeric_p(s):
    # emmeans may export a string such as '<.0001'. Use that upper bound.
    # Here it is safely below the manuscript's alpha=.01.
    return float(str(s).strip().lstrip('<').strip())


def get_emms(tables):
    records = {}
    for df in tables:
        c, s, m = col(df, 'condition'), col(df, 'task_stage', 'stage'), col(df, 'emmean', 'emm')
        lo = col(df, 'asymp.LCL', 'lower.CL', 'lower_ci', 'LCL')
        hi = col(df, 'asymp.UCL', 'upper.CL', 'upper_ci', 'UCL')
        if any(v is None for v in (c, s, m, lo, hi)):
            continue
        for _, r in df.dropna(subset=[c, s, m, lo, hi]).iterrows():
            key = (stage(r[s]), condition(r[c]))
            val = (float(r[m]), float(r[lo]), float(r[hi]))
            if key in records and not np.allclose(records[key], val, atol=1e-4):
                raise ValueError(f'Conflicting EMM rows for {key}')
            records[key] = val
    expected = {(s, c) for s in STAGES for c in CONDITIONS}
    if not records:
        if not ALLOW_REFERENCE_EMMS:
            raise ValueError('No EMMs with confidence intervals found. Export summary(emmeans(...), infer=c(TRUE,FALSE)).')
        warnings.warn('No EMM rows found in CSVs; using the six EMMs and 95% CIs from the supplied Model 7 text output. Check they match the current model before publication.')
        records = {(s,c):(m,lo,hi) for s,c,m,lo,hi in REFERENCE_EMMS}
    if set(records) != expected:
        raise ValueError(f'Missing or unexpected EMMs: {expected - set(records)}; {set(records) - expected}')
    for key, (m, lo, hi) in records.items():
        if not np.isfinite([m,lo,hi]).all() or not lo <= m <= hi:
            raise ValueError(f'Invalid EMM/CI for {key}')
    return records


def get_contrasts(df, by_stage):
    group_col = col(df, 'task_stage', 'stage') if by_stage else col(df, 'condition')
    contrast_col, p_col = col(df, 'contrast'), col(df, 'p.value', 'p_value', 'p_adj', 'p_adjusted', 'p')
    if any(x is None for x in (group_col, contrast_col, p_col)):
        raise ValueError(f'Expected grouping, contrast and p.value columns. Found {list(df.columns)}. Export the emmeans contrast table.')
    out = {}
    for _, r in df.dropna(subset=[group_col, contrast_col, p_col]).iterrows():
        parts = re.split(r'\s+[-−–]\s+', str(r[contrast_col]).strip())
        if len(parts) != 2:
            raise ValueError(f'Cannot parse contrast {r[contrast_col]!r}; expected e.g. _BL - _MAIN_baseline')
        g = stage(r[group_col]) if by_stage else condition(r[group_col])
        a, b = map(condition if by_stage else stage, parts)
        key = (g, frozenset((a,b)))
        p = numeric_p(r[p_col])
        if not 0 <= p <= 1:
            raise ValueError(f'Invalid p-value: {p}')
        if key in out and not np.isclose(out[key], p):
            raise ValueError(f'Conflicting contrast p-values for {key}')
        out[key] = p
    expected = ({(s,frozenset(pair)) for s in STAGES for pair in combinations(CONDITIONS,2)} if by_stage
                else {(c,frozenset(STAGES)) for c in CONDITIONS})
    if set(out) != expected:
        raise ValueError(f'Missing/unexpected contrasts: {expected-set(out)}; {set(out)-expected}')
    return out


def bracket(ax, x1, x2, y, h, label='*'):
    ax.plot([x1,x1,x2,x2], [y,y+h,y+h,y], color='#606060', lw=1.15, clip_on=False)
    ax.text((x1+x2)/2, y+h+.022, label, ha='center', va='bottom', fontsize=12, color='#505050')


def main(summary_file=SUMMARY_FILE, output_dir=DATA_DIR):
    df = pd.read_csv(summary_file)
    keys = ['sub', 'condition', 'task_stage', 'roi']
    if df.duplicated(keys).any() or not np.isfinite(df.surrogate_z).all():
        raise ValueError('Duplicate observations or invalid surrogate_z values')
    df['condition'] = df.condition.map(condition)
    df['task_stage'] = df.task_stage.map(stage)
    counts = df.groupby(keys[:3]).roi.nunique()
    if not (counts == 24).all():
        raise ValueError('Expected 24 ROIs per participant and condition-stage')
    subjects = df.groupby(keys[:3], as_index=False).surrogate_z.mean()
    if len(subjects) != 138 or subjects['sub'].nunique() != 23 or not (subjects.groupby('sub').size() == 6).all():
        raise ValueError('Expected 23 participants with six condition-stage observations each')
    groups = {(s,c): subjects.loc[(subjects.task_stage == s) & (subjects.condition == c), 'surrogate_z'].to_numpy() for s in STAGES for c in CONDITIONS}
    if (DATA_DIR / f'{PREFIX}_condition_by_stage.csv').exists() and (DATA_DIR / f'{PREFIX}_stage_by_condition.csv').exists():
        between = get_contrasts(read_csv('condition_by_stage'), True)
        within = get_contrasts(read_csv('stage_by_condition'), False)
    elif ALLOW_REFERENCE_CONTRASTS:
        # Verified from the supplied Model 7 text; <.0001 represented by its upper bound.
        between = {(s, frozenset(pair)): p for s, ps in zip(STAGES, [[.0001,.1981,.0001],[.0001,.0001,.0001]]) for pair,p in zip(combinations(CONDITIONS,2),ps)}
        within = {(c,frozenset(STAGES)): p for c,p in zip(CONDITIONS,[.1368,.0001,.0001])}
    else:
        raise FileNotFoundError('Model 7 contrast tables missing. Check DATA_DIR.')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.labelsize':12,'svg.fonttype':'none'})
    fig, ax = plt.subplots(figsize=(6.5,6.3))
    centers = [0, 1.42]
    offsets = [-.32,0,.32]
    xpos = {(s,c):x+d for s,x in zip(STAGES,centers) for c,d in zip(CONDITIONS,offsets)}
    ax.axhline(0,color='#999999',ls='--',lw=.9,zorder=0)
    rng = np.random.default_rng(42)
    tops, bottoms = {}, []
    for (s,c),v in groups.items():
        x=xpos[s,c]; m=v.mean(); sd=v.std(ddof=1)
        err=sd if ERROR_STYLE=='SD' else sd/np.sqrt(len(v))
        if PLOT_STYLE=='bar':
            ax.bar(x,m,width=.25,color=COLORS[c],alpha=.67,edgecolor='#555555',linewidth=.75,zorder=2)
            ax.errorbar(x,m,yerr=err,fmt='none',ecolor='#454545',capsize=4,lw=1.2,zorder=4)
        elif PLOT_STYLE=='box':
            ax.boxplot(
                        v,
                        positions=[x],
                        widths=.25,
                        patch_artist=True,
                        showfliers=False,
                        boxprops={'facecolor': COLORS[c], 'alpha': .65, 'edgecolor': '#555555'},
                        medianprops={'color': '#444444'},
                        whiskerprops={'color': '#666666'},
                        capprops={'color': '#666666'}
                    )
        else: raise ValueError('PLOT_STYLE must be bar or box')
        if SHOW_POINTS:
            ax.scatter(x+rng.uniform(-.085,.085,len(v)),v,s=15,c=COLORS[c],edgecolors='#333333',linewidths=.35,alpha=.8,zorder=3)
        q1, q3 = np.percentile(v, [25, 75])
        iqr = q3 - q1
        upper_whisker = min(v.max(), q3 + 1.5 * iqr)
        lower_whisker = max(v.min(), q1 - 1.5 * iqr)

        tops[s,c] = upper_whisker
        bottoms.append(lower_whisker)
        if s=='Execution' and within[c,frozenset(STAGES)]<ALPHA:
            ax.text(x, tops[s,c] + .15, '★', ha='center', va='bottom', fontsize=13, color='#555555')
            tops[s,c] += .45
    highest=max(tops.values())
    for s in STAGES:
        placed=[]
        for a,b in [(CONDITIONS[0],CONDITIONS[1]),(CONDITIONS[1],CONDITIONS[2]),(CONDITIONS[0],CONDITIONS[2])]:
            if between[s,frozenset((a,b))]>=ALPHA: continue
            i,j=CONDITIONS.index(a),CONDITIONS.index(b)
            y = max(tops[s,c] for c in CONDITIONS[i:j+1]) + .16
            for l, r, h in placed:
                if i <= r and j >= l:
                    y = max(y, h + .40)
            bracket(ax, xpos[s,a], xpos[s,b], y, .07)
            placed.append((i,j,y)); highest=max(highest,y+.22)
    ax.set_ylim(min(bottoms)-.35,highest+.45)
    ax.set_xlim(-.57,1.99); ax.set_xticks(centers,STAGES)
    ax.set_xlabel('Movement stage'); ax.set_ylabel('Surrogate-normalised tgPAC (zMI)')
    ax.spines[['top','right']].set_visible(False); ax.tick_params(axis='x',length=0)
    fig.suptitle('PAC comparison across tasks and conditions',fontsize=13,fontweight='bold',y=.975)
    fig.legend(handles=[Patch(facecolor=COLORS[c],label=c) for c in CONDITIONS],loc='upper center',bbox_to_anchor=(.5,.925),ncol=3,frameon=False,fontsize=9,handlelength=1.2,columnspacing=1)
    desc = f'Bars: participant means ± {ERROR_STYLE}.' if PLOT_STYLE=='bar' else 'Boxes: median and IQR; whiskers: 1.5 × IQR.'
    fig.text(.5,.025,desc+'  Points: individual participant ROI means (n = 23).\n* Between conditions; ★ planning vs execution (Model 7, p < 0.01).',ha='center',fontsize=8.5)
    fig.subplots_adjust(left=.15,right=.97,bottom=.17,top=.84)
    output_dir=Path(output_dir); output_dir.mkdir(parents=True,exist_ok=True)
    for ext in ['png','svg']:
        path=output_dir / f'Figure8B_participant_distribution_{PLOT_STYLE}.{ext}'
        fig.savefig(path,dpi=300,facecolor='white'); print('Saved:',path)
    plt.show()

if __name__=='__main__':
    main()
