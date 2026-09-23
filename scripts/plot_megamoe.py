#!/usr/bin/env python3
"""Render MegaMoE end-to-end charts from the article tables (requires matplotlib)."""
from pathlib import Path
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parents[1]
POST = ROOT / '_posts/2026-09-22-mega-moe.md'
OUT = ROOT / 'assets/imgs/mega-moe'
OUT.mkdir(parents=True, exist_ok=True)
text = POST.read_text()
section = text.split('## End-to-end results in vLLM\n', 1)[1].split('## Kernel microbenchmarks', 1)[0]
charts = [('sm100-flash', 'DeepSeek-V4-Flash · 1×8 SM100 · EP8'),
          ('sm100-pro', 'DeepSeek-V4-Pro · 1×8 SM100 · EP8'),
          ('sm103-flash', 'DeepSeek-V4-Flash · 1×4 SM103 · EP4/TP4')]
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'svg.fonttype': 'none'})
for block, (slug, title) in zip(section.split('### ')[1:], charts):
    rows = [line.strip('| ').split('|') for line in block.splitlines() if re.match(r'^\| (prefill|decode|100K|32K)', line)]
    assert len(rows) == 4
    labels = [row[0].strip() for row in rows]
    values = [[int(cell.strip().split()[0]) for cell in row[1:]] for row in rows]
    fig, ax = plt.subplots(figsize=(9, 5.5), layout='constrained')
    colors = ['#b5b5b5', '#333333', '#76b900']
    names = ['native (MX)', 'fi_dg (MX)', 'fi_cutedsl (NVFP4)']
    for j, (name, color) in enumerate(zip(names, colors)):
        bars = ax.barh([i + (j - 1) * .23 for i in range(4)], [v[j] for v in values],
                       height=.20, color=color, label=name)
        ax.bar_label(bars, labels=[f'{v[j]:,}' for v in values], padding=4, fontsize=10)
    ax.set_yticks(range(4), labels)
    ax.invert_yaxis()
    ax.set_xlim(0, max(max(v) for v in values) * 1.18)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x / 1000:g}k' if x else '0'))
    ax.set_xlabel('Throughput (tokens/s) · Higher is better', labelpad=12)
    ax.set_title(title, loc='left', fontsize=15, weight='bold', pad=45)
    ax.legend(loc='lower left', bbox_to_anchor=(0, 1.015), ncol=3, frameon=False, fontsize=10)
    ax.set_axisbelow(True)
    ax.grid(axis='x', color='#e5e5e5', linewidth=.7)
    ax.tick_params(axis='both', length=0)
    for spine in ax.spines.values(): spine.set_visible(False)
    fig.savefig(OUT / (slug + '.svg'), metadata={'Title': title, 'Description': 'Reported Draft-4 vLLM throughput. Values reproduced from the adjacent table; checkpoint precision differs across backends.'})
    fig.savefig('/tmp/flashinfer-' + slug + '.png', dpi=140)
    plt.close(fig)
    print(slug, values)
