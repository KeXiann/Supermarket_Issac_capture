"""Keep latency-sensitive simulation off slower hybrid CPU cores when detected.

Only this process and future children change affinity. No governor, frequency,
priority, system configuration, or other running process is modified.
"""
import os
from pathlib import Path


def fast_cpu_set(available, maximum_khz):
    available = set(available)
    if not available or any(cpu not in maximum_khz for cpu in available):
        return available
    fastest = max(maximum_khz[cpu] for cpu in available)
    slowest = min(maximum_khz[cpu] for cpu in available)
    if slowest <= 0 or fastest / slowest < 1.10:
        return available
    selected = {cpu for cpu in available if maximum_khz[cpu] >= .95 * fastest}
    return selected if len(selected) >= 2 else available


def configure_cpu_affinity():
    setting = os.environ.get('PICO_CPU_AFFINITY', 'auto').strip()
    report = {'policy': setting, 'changed': False}
    if not hasattr(os, 'sched_getaffinity') or setting == 'off':
        return report
    available = set(os.sched_getaffinity(0))
    report['inherited_cpus'] = sorted(available)
    if setting == 'auto':
        frequencies = {}
        for cpu in available:
            try:
                frequencies[cpu] = int(Path(f'/sys/devices/system/cpu/cpu{cpu}/cpufreq/cpuinfo_max_freq').read_text())
            except (FileNotFoundError, PermissionError, ValueError):
                pass
        selected = fast_cpu_set(available, frequencies)
        report['cpuinfo_max_freq_khz'] = frequencies
    else:
        selected = set()
        for part in setting.split(','):
            if '-' in part:
                first, last = map(int, part.split('-'))
                selected.update(range(first, last+1))
            else:
                selected.add(int(part))
        if not selected or not selected.issubset(available):
            raise ValueError('PICO_CPU_AFFINITY must select available CPUs, auto, or off')
    if selected != available:
        os.sched_setaffinity(0, selected)
        report['changed'] = True
        print('[CPU] latency-sensitive runtime CPUs: ' + ','.join(map(str, sorted(selected))), flush=True)
    report['selected_cpus'] = sorted(selected)
    return report
