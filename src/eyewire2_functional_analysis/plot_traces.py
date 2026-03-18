from matplotlib import pyplot as plt
import numpy as np


def plot_trace_and_trigger(time, trace, triggertimes, trace_norm=None, title=None, ax=None, label=None):
    """Plot a fluorescence trace with trigger-time markers.

    Optionally overlays a normalised version of the trace on a twin y-axis.

    Args:
        time: 1-D time array.
        trace: 1-D fluorescence trace array aligned with ``time``.
        triggertimes: 1-D array of trigger times (in the same units as ``time``).
            Empty arrays are handled gracefully.
        trace_norm: Optional normalised trace to overlay on a twin y-axis.
        title: Optional axes title string.
        ax: Existing Matplotlib Axes to plot on. If ``None``, a new figure is created.
        label: Legend label for the main trace.

    Returns:
        matplotlib.axes.Axes: The primary axes containing the plot.
    """
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
    """Plot multiple traces on a single axes with automatic alpha scaling.

    Args:
        time: 1-D time array.
        traces: 2-D array of shape ``(n_traces, time)`` or iterable of 1-D arrays.
        ax: Existing Matplotlib Axes to plot on. If ``None``, a new figure is created.
        title: Optional axes title string.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 2))
    if title is not None:
        ax.set_title(title)
    for trace in traces:
        ax.plot(time, trace, alpha=np.maximum(1. / len(traces), 0.3))


def get_aligned_snippets_times(snippets_times, raise_error=True, tol=1e-4):
    """Return a single aligned time vector from a 2-D array of snippet time stamps.

    Subtracts the per-snippet offset (first row), checks consistency across snippets,
    and returns the mean time axis.

    Args:
        snippets_times: 2-D array of shape ``(time_points, n_snippets)`` containing
            absolute time stamps for each snippet.
        raise_error: If ``True``, raise a ``ValueError`` when the standard deviation
            across snippets exceeds ``tol``; otherwise issue a warning.
        tol: Maximum acceptable per-sample standard deviation across snippets.

    Returns:
        numpy.ndarray: 1-D array of aligned (mean) time values.

    Raises:
        ValueError: If snippet times are inconsistent and ``raise_error`` is ``True``.
    """
    snippets_times = snippets_times - snippets_times[0, :]

    is_inconsistent = np.any(np.std(snippets_times, axis=1) > tol)
    if is_inconsistent:
        if raise_error:
            raise ValueError(f'Failed to snippet times: max_std={np.max(np.std(snippets_times, axis=1))}')
        else:
            warnings.warn(f'Snippet times are inconsistent: max_std={np.max(np.std(snippets_times, axis=1))}')

    aligned_times = np.mean(snippets_times, axis=1)
    return aligned_times