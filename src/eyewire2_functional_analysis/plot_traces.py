from matplotlib import pyplot as plt
import numpy as np


def plot_trace_and_trigger(time, trace, triggertimes, trace_norm=None, title=None, ax=None, label=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 2))
    if title is not None:
        ax.set_title(title)
    ax.plot(time, trace, label=label)
    ax.set(xlabel='time', ylabel='trace')
    if len(triggertimes) > 0:
        vmin, vmax = np.nanmin(trace), np.nanmax(trace)
        vrng = vmax - vmin
        ax.vlines(triggertimes, vmin - 0.22 * vrng, vmin - 0.02 * vrng, color='r', label='trigger', zorder=-2)
    ax.legend(loc='upper right')

    if trace_norm is not None:
        tax = ax.twinx()
        tax.plot(time, trace_norm, ':')
        if len(triggertimes) > 0:
            vmin, vmax = np.nanmin(trace_norm), np.nanmax(trace_norm)
            vrng = vmax - vmin
            tax.vlines(triggertimes, vmin - 0.22 * vrng, vmin - 0.02 * vrng, color='r', label='trigger', ls=':',
                       zorder=-1)
        tax.set(ylabel='normalized')

    return ax


def plot_traces(time, traces, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 2))
    if title is not None:
        ax.set_title(title)
    for trace in traces:
        ax.plot(time, trace, alpha=np.maximum(1. / len(traces), 0.3))


def get_aligned_snippets_times(snippets_times, raise_error=True, tol=1e-4):
    snippets_times = snippets_times - snippets_times[0, :]

    is_inconsistent = np.any(np.std(snippets_times, axis=1) > tol)
    if is_inconsistent:
        if raise_error:
            raise ValueError(f'Failed to snippet times: max_std={np.max(np.std(snippets_times, axis=1))}')
        else:
            warnings.warn(f'Snippet times are inconsistent: max_std={np.max(np.std(snippets_times, axis=1))}')

    aligned_times = np.mean(snippets_times, axis=1)
    return aligned_times