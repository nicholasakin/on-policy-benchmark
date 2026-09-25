#!/usr/bin/env python
"""Progress table for run_compare.sh log files (they're stdout, not metrics.csv)."""
import re
import sys
from glob import glob

STEP = re.compile(r'total num timesteps (\d+)/(\d+)')
SUCC = re.compile(r'eval success rate: ([\d.]+)')


def parse(path):
    steps = total = None
    succ = []
    with open(path) as handle:
        for line in handle:
            if m := STEP.search(line):
                steps, total = int(m.group(1)), int(m.group(2))
            if m := SUCC.search(line):
                succ.append(float(m.group(1)))
    return steps, total, succ


def sparkline(values, width=20):
    blocks = ' .:-=+*#'
    if not values:
        return ''
    step = max(1, len(values) / width)
    picked = [values[min(len(values) - 1, int(i * step))] for i in range(min(width, len(values)))]
    return ''.join(blocks[min(len(blocks) - 1, int(v / max(max(picked), 1e-9) * (len(blocks) - 1)))] for v in picked)


for path in sorted(glob(sys.argv[1] if len(sys.argv) > 1 else '*.log')):
    if 'driver' in path:
        continue
    steps, total, succ = parse(path)
    if steps is None:
        print(f'{path:<28} (not started / no output yet)')
        continue
    recent = sum(succ[-3:]) / max(1, len(succ[-3:])) if succ else 0.0
    print(f'{path:<28}{steps:>10}/{total:<10} succ(recent avg)={recent:>5.2f} best={max(succ, default=0):>5.2f}  {sparkline(succ)}')
