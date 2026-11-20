"""Figure 8B: surrogate-normalised tgPAC, model 7.
Plots model EMMs and 95% CIs, not raw-data boxplots.
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
PLOT_STYLE = "bar"  # "bar" or "point"
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
    ax.text((x1+x2)/2, y+h+.018, label, ha='center', va='bottom', fontsize=12, color='#505050')


def main():
    anova = read_csv('anova')
    print('\nModel ANOVA (for verification):\n', anova.to_string(index=False))
    condition_table = read_csv('condition_by_stage')
    stage_table = read_csv('stage_by_condition')
    # Optional separate EMM file, useful if the two contrast CSVs contain no means.
    emm_path = DATA_DIR / f'{PREFIX}_emms.csv'
    tables = [condition_table, stage_table]
    if emm_path.exists():
        tables.append(pd.read_csv(emm_path))
    emms = get_emms(tables)
    between = get_contrasts(condition_table, by_stage=True)
    within = get_contrasts(stage_table, by_stage=False)

    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':11, 'axes.labelsize':13,
                         'axes.titlesize':14, 'svg.fonttype':'none', 'pdf.fonttype':42})
    fig, ax = plt.subplots(figsize=(6.1,5.8))
    centers = [0, 1.28]
    offsets = [-.31, 0, .31]
    xpos = {(s,c):center+offset for s,center in zip(STAGES,centers) for c,offset in zip(CONDITIONS,offsets)}
    ax.axhline(0, color='#969696', lw=.9, ls='--', zorder=0)

    # Coloured points represent model-estimated means, whiskers are actual 95% CIs.
    stage_star_y = {}
    for s in STAGES:
        for c in CONDITIONS:
            x = xpos[s,c]
            m, lo, hi = emms[s,c]
            if PLOT_STYLE == 'bar':
                ax.bar(x, m, width=.25, color=COLORS[c], alpha=.85, edgecolor='#666666', linewidth=.8, zorder=2)
            ax.errorbar(x, m, yerr=[[m-lo],[hi-m]], fmt='none' if PLOT_STYLE == 'bar' else 'o', markersize=7,
                        markerfacecolor=COLORS[c], markeredgecolor='#555555', markeredgewidth=.8,
                        ecolor='#777777', elinewidth=1.3, capsize=6, capthick=1.2, zorder=3)
            if s == 'Execution' and within[c,frozenset(STAGES)] < ALPHA:
                y = hi + .045
                ax.text(x, y, '★', ha='center', va='bottom', color='#606060', fontsize=15)
                stage_star_y[s,c] = y+.075

    # Stack significant pairwise brackets above each movement-stage group.
    highest = 0
    pairs = [(CONDITIONS[0], CONDITIONS[1]), (CONDITIONS[1], CONDITIONS[2]), (CONDITIONS[0], CONDITIONS[2])]
    for s in STAGES:
        y = max(max(emms[s,c][2], stage_star_y.get((s,c),0)) for c in CONDITIONS) + .09
        for a,b in pairs:
            if between[s,frozenset((a,b))] < ALPHA:
                bracket(ax, xpos[s,a], xpos[s,b], y, .035)
                highest = max(highest,y+.09)
                y += .15
    ymin = min(lo for m,lo,hi in emms.values())
    ax.set_ylim(min(-.1, ymin-.10), max(highest+.17, max(v[2] for v in emms.values())+.25))
    ax.set_xlim(-.52, 1.80)
    ax.set_xticks(centers, STAGES)
    ax.set_xlabel('Movement stage')
    ax.set_ylabel('Surrogate-normalised tgPAC (zMI)')
    fig.suptitle('PAC comparison across tasks and conditions', fontsize=13, fontweight='bold', y=.975)
    ax.spines[['top','right']].set_visible(False)
    ax.tick_params(axis='x', length=0)
    handles = [Patch(facecolor=COLORS[c], edgecolor='none', label=c) for c in CONDITIONS]
    # Legend above the data to avoid overlap with statistical brackets.
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5,.925),
               ncol=3, frameon=False, fontsize=9, handlelength=1.3, columnspacing=1.0)
    fig.text(.5,.025,'* Between conditions; ★ planning vs execution (p < 0.01).\n' + ('Bars' if PLOT_STYLE == 'bar' else 'Points') + ': EMMs; whiskers: 95% confidence intervals.',
             ha='center', va='bottom', fontsize=9, color='#444444')
    fig.subplots_adjust(left=.16,right=.97,bottom=.19,top=.84)
    for ext in ['png','svg','pdf']:
        path = DATA_DIR / f'Figure8B_surrogate_normalised_tgPAC_{PLOT_STYLE}.{ext}'
        fig.savefig(path, dpi=300, facecolor='white')
        print('Saved:', path)
    plt.show()


if __name__ == '__main__':
    main()
