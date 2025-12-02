import numpy as np
import skeliner as sk
from matplotlib import patches as patches

MB_DIRS              = (0,  180,   45,  225,  90, 270, 135, 315)
MB_DIRS_SYMBOLS_V_UP = ('↓', '↑', '↙', '↗', '←', '→', '↖', '↘')
MB_DIRS_SYMBOLS_D_UP = ('↑', '↓', '↗', '↙', '→', '←', '↘', '↖')


def plot_chirp(ax, row, stimulus_ms=None, plot_hline=True, plot_vlines=False):
    snippets = row['chirp_snippets']
    for i, trace in enumerate(snippets.T):
        ax.plot(np.arange(0, len(trace)) * row['chirp_snippets_dt'], trace / np.max(np.abs(trace)), color='dimgray',
                alpha=0.5, clip_on=False)
    ax.plot(np.arange(0, len(row['chirp_average_norm'])) * row['chirp_average_dt'], row['chirp_average_norm'],
            color='darkred', alpha=0.8, clip_on=False, )
    if plot_hline:
        ax.axhline(0, c='dimgray', ls='--')
    if plot_vlines:
        for t in [2, 5, 8, 30]:
            ax.axvline(t, c='dimgray', ls='--')
    if stimulus_ms is not None:
        y0 = np.max(row['chirp_average_norm'])
        yrng = np.max(row['chirp_average_norm']) - np.min(row['chirp_average_norm'])
        stimulus_ms_norm = (stimulus_ms - stimulus_ms.min()) / (stimulus_ms.max() - stimulus_ms.min())
        ax.plot(np.arange(len(stimulus_ms)) * 1e-3, stimulus_ms_norm * 0.2 * yrng + y0 * 1.1,
                c='k', clip_on=False, lw=1, solid_capstyle='butt')


def plot_bar(ax, row, annotate_dirs=False, annotate_symbols=False, ventral_up=True):
    vmax = np.max(row['bar_snippets'])

    if ventral_up:
        mb_symbols = MB_DIRS_SYMBOLS_V_UP
    else:
        mb_symbols = MB_DIRS_SYMBOLS_D_UP
    
    for i, (dir_deg, symbol) in enumerate(zip(MB_DIRS, mb_symbols)):
        snippets = row['bar_snippets'][:, np.array([0, 8, 16]) + i]
        time = (np.arange(0, snippets.shape[0]) + (snippets.shape[0] * 1.2 * i)) * row['bar_snippets_dt']
        for trace in snippets.T:
            ax.plot(time, trace / vmax, color='dimgray', alpha=0.5, clip_on=False, )
        ax.plot(time, np.mean(snippets, axis=1) / vmax, color='darkred', alpha=0.8, clip_on=False)
        ax.axhline(0, c='dimgray', ls='--')
        if annotate_dirs or annotate_symbols:
            x = time[0] + 0.5 * (time[-1] - time[0])
            y = 1.15

            if annotate_dirs:
                ax.text(x, y, f'{dir_deg}°', ha='center', va='top', fontsize=8)
            else:
                ax.text(
                    x, y, symbol,
                    ha='center', va='top',
                    fontsize=10,
                    fontweight='bold',
                    fontname='DejaVu Sans',
                )


def plot_bar_dir(ax, row, ventral_up=True):
    if np.any(~np.isfinite(row['bar_snippets'])):
        raise ValueError('bar_snippets not finite')

    sorted_directions, sorted_responses, sorted_averages = preprocess_mb_snippets(snippets=row['bar_snippets'])
    time_component, dir_component = get_time_dir_kernels(sorted_averages, dt=row['bar_snippets_dt'])
    sorted_directions = np.append(sorted_directions, sorted_directions[0])
    dir_component = np.append(dir_component, dir_component[0])

    if np.any(~np.isfinite(dir_component)):
        raise ValueError('dir_component not finite')
    
    ax.plot(sorted_directions, np.clip(dir_component, 0, None), color='darkred', alpha=0.8, lw=2)
    
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    
    ax.xaxis.set_tick_params(pad=-20)
    dirs = [0, 90, 180, 270]
    mb_symbols = MB_DIRS_SYMBOLS_V_UP if ventral_up else MB_DIRS_SYMBOLS_D_UP

    ax.set(xlabel=None, ylabel=None, yticks=[0, np.max(dir_component)])
    ax.set_ylim(0, np.max(dir_component))
    ax.set_xticks(np.deg2rad(dirs))
    ax.set_xticklabels([mb_symbols[np.argmax(np.array(MB_DIRS) == d)] for d in dirs],
                       fontsize=10, fontweight='bold', fontname='DejaVu Sans', )
    ax.set_yticklabels([])


def plot_bar_block(ax, row, i, show_symbol=True, ventral_up=False):
    """
    Plot ONE direction block (index i: 0..7) on the given Cartesian axes.
    Matches your original styling.
    """
    snippets = row['bar_snippets'][:, np.array([0, 8, 16]) + i]
    time = (np.arange(0, snippets.shape[0]) + (snippets.shape[0] * 1.2 * i)) * row['bar_snippets_dt']

    # traces
    for trace in snippets.T:
        ax.plot(time, trace, color='dimgray', alpha=0.5)
    # mean
    ax.plot(time, np.mean(snippets, axis=1), color='darkred', alpha=0.8)
    ax.axhline(0, c='dimgray', ls='--')

    mb_symbols = MB_DIRS_SYMBOLS_V_UP if ventral_up else MB_DIRS_SYMBOLS_D_UP

    # label
    if show_symbol:
        x = time[0] + 0.5 * (time[-1] - time[0])
        y_max = np.max(row['bar_snippets'])
        y = y_max + 0.25 * (np.max(row['bar_snippets']) - np.min(row['bar_snippets']))  # relative offset
        ax.text(
            x, y, mb_symbols[i],
            ha='center', va='top',
            fontsize=10,
            fontweight='bold',
            fontname='DejaVu Sans',
        )

    # clean look (like your grid cells)
    ax.set(xlabel=None, ylabel=None, xticks=[], yticks=[])
    ax.axis('off')


def plot_bar_split(ax_map, row, labels=('C', 'D', 'E', 'F', 'H', 'I', 'J', 'K'),
                   dir_idx_order=(0, 1, 2, 3, 4, 5, 6, 7)):
    """
    Plot the 8 direction blocks into named axes in a *specified* order.
    dir_idx_order: which data block (0..7) goes to each label, same length as labels.
    """
    y_max = float(np.nanmax(row['bar_snippets']))
    y_min = float(np.nanmin(row['bar_snippets']))
    y_span = (y_max - y_min) if (y_max > y_min) else 1.0
    y_top = y_max + 0.15 * y_span

    for lab, idx in zip(labels, dir_idx_order):
        ax = ax_map[lab]
        plot_bar_block(ax, row, idx, show_symbol=True)
        ax.set_ylim(y_min, y_top)


def draw_scale_bar(ax, length_data, label="2 mm",
                   where="lower center",
                   y_frac=0.06, x_pad_axes=0.0,  # x pad as fraction of axis width
                   lw=1.5, fontsize=8, label_above=False):
    xlo, xhi = ax.get_xlim()
    xspan = xhi - xlo
    if xspan == 0:
        return None
    w_frac = length_data / xspan  # width of bar in axes fraction

    if where == "lower left":
        x0_frac = x_pad_axes
    elif where == "lower right":
        x0_frac = 1.0 - x_pad_axes - w_frac
    elif where == "lower center":
        x0_frac = 0.5 - w_frac / 2
    else:
        raise ValueError("where must be 'lower left' | 'lower center' | 'lower right'.")

    x1_frac = x0_frac + w_frac
    line = ax.plot([x0_frac, x1_frac], [y_frac, y_frac],
                   transform=ax.transAxes, color='k', lw=lw,
                   solid_capstyle='butt', clip_on=False, zorder=10)[0]
    dy_pts = 4 if not label_above else -4
    va = 'top' if not label_above else 'bottom'
    ax.annotate(label, xy=((x0_frac + x1_frac) / 2, y_frac), xycoords=ax.transAxes,
                xytext=(0, -dy_pts), textcoords='offset points',
                ha='center', va=va, fontsize=fontsize, color='k',
                zorder=11, clip_on=False)
    return line


def plot_morph(ax, row, rad=200):
    sk.plot.projection(row.skel, ax=ax, plane='xy')  # , color_by="ntype", skel_cmap='Grays')
    sx, sy, sz = row.skel.soma.center
    ax.set_xlim(sx - rad, sx + rad)
    ax.set_ylim(sy + rad, sy - rad)
    return sx, sy, sz


def plot_mosaic(df, extent=(350, 1000, 0, 650)):
    import cell_mosaics
    
    assert df.shape[0] > 0, "No data to plot"
    mapper = cell_mosaics.CoverageDensityMapper(field_bounds=extent, resolution=500)
    for i, (seg_id, row) in enumerate(df.iterrows()):
        mapper.add_convex_hull(row.skel.nodes[row.skel.nodes[:, 2] > -10, :2])
    fig, ax, im = mapper.plot_coverage(colormap='bone_r', plot_cell_outlines=True)
    for i, (seg_id, row) in enumerate(df.sample(1).iterrows()):
        plot_morph(ax=ax, row=row)
    ax.set(xlim=extent[:2], ylim=extent[2:])
    return fig, ax


# DSI / OSI from djimaging
def get_dir_idx(snippets, dir_order=MB_DIRS):
    """
    snippets: np.ndarray (times, dirs*reps)
    dir_order: np.ndarray (dirs, ) or (dirs*reps, )
    """
    dir_order = np.asarray(dir_order).squeeze()
    assert dir_order.ndim == 1, dir_order.shape
    assert snippets.ndim == 2, snippets.shape
    n_snippets = snippets.shape[-1]
    assert (n_snippets % dir_order.size) == 0, f"Snippet length {n_snippets} is not a multiple of {dir_order.size}"
    dir_order = np.tile(dir_order, n_snippets // dir_order.size)
    assert n_snippets == dir_order.size

    dir_deg = dir_order[:8]  # get the directions of the bars in degree
    dir_rad = np.deg2rad(dir_deg)  # convert to radians
    dir_idx = [list(np.where(dir_order == d)[0]) for d in dir_deg]

    return dir_idx, dir_rad


def sort_response_matrix(snippets: np.ndarray, idxs: list, directions: np.ndarray):
    """
    Sorts the snippets according to stimulus condition and repetition into a time x direction x repetition matrix
    Inputs:
    snippets    list or array, time x (directions*repetitions)
    idxs        list of lists giving idxs into last axis of snippets. idxs[0] gives the indexes of rows in snippets
                which are responses to the direction directions[0]
    Outputs:
    sorted_responses   array, time x direction x repetitions, with directions sorted(!) (0, 45, 90, ..., 315) degrees
    sorted_directions   array, sorted directions
    """
    structured_responses = snippets[:, idxs]
    sorting = np.argsort(directions)
    sorted_responses = structured_responses[:, sorting, :]
    sorted_directions = directions[sorting]
    return sorted_responses, sorted_directions


def preprocess_mb_snippets(snippets, dir_order=MB_DIRS):
    dir_idx, dir_rad = get_dir_idx(snippets, dir_order)

    sorted_responses, sorted_directions = sort_response_matrix(snippets, dir_idx, dir_rad)
    sorted_averages = np.mean(sorted_responses, axis=-1)
    return sorted_directions, sorted_responses, sorted_averages


def get_time_dir_kernels(sorted_responses: np.ndarray, dt: float):
    """
    Performs singular value decomposition on the time x direction matrix (averaged across repetitions)
    Uses a heuristic to try to determine whether a sign flip occurred during svd
    For the time course, the mean of the first second is subtracted and then the vector is divided by the maximum
    absolute value.
    For the direction/orientation tuning curve, the vector is normalized to the range (0,1)

    Parameters:
    sorted_responses (array): Time x direction matrix.
    dt (float): 1 / sampling_rate of trace.

    Returns:
    tuple: Contains time_kernel (array, time x 1), direction_tuning (array, directions x 1), and singular_value (float).
    """
    U, S, Vh = np.linalg.svd(sorted_responses)

    time_component = U[:, 0]
    dir_component = Vh[0, :]

    # the time_kernel determined by SVD should be correlated to the average response across all directions. if the
    # correlation is negative, U is likely flipped

    if np.mean((-1 * time_component - np.mean(sorted_responses, axis=-1)) ** 2) < np.mean(
            (time_component - np.mean(sorted_responses, axis=-1)) ** 2
    ):
        su = -1
    else:
        su = 1

    sv = np.sign(np.mean(np.sign(dir_component)))
    if sv == 1 and su == 1:
        s = 1
    elif sv == -1 and su == -1:
        s = -1
    elif sv == 1 and su == -1:
        s = 1
    elif sv == 0:
        s = su
    else:
        s = 1

    time_component *= s
    dir_component *= s

    # determine which entries correspond to the first second, assuming 4 seconds presentation time
    first_second_idx = np.maximum(int(np.floor(1.0 / dt)), 1)
    time_component -= np.mean(time_component[:first_second_idx])
    time_component = time_component / np.max(np.abs(time_component))

    # dir_component -= np.min(dir_component)
    # dir_component = dir_component / np.max(dir_component)

    return time_component, dir_component


def plot_retina_orientation(ax, tdist=50, x0=0, y0=0, size=1000, fontsize=14):
    ax.plot([x0 - size / 2, x0 + size / 2], [y0, y0], c='k', solid_capstyle='butt', clip_on=False)
    ax.plot([x0, x0], [y0 - size / 2, y0 + size / 2], c='k', solid_capstyle='butt', clip_on=False)
    ax.text(x0 - size / 2 - tdist, y0, 'N', c='k', va='center', ha='right', fontsize=fontsize)
    ax.text(x0 + size / 2 + tdist, y0, 'T', c='k', va='center', ha='left', fontsize=fontsize)
    ax.text(x0, y0 - size / 2 - tdist, 'V', c='k', va='top', ha='center', fontsize=fontsize)
    ax.text(x0, y0 + size / 2 + tdist, 'D', c='k', va='bottom', ha='center', fontsize=fontsize)


def plot_scale_bar(
        ax, x0=0, y0=0, size=1000, tdist=70,
        fontsize=10, text=True, unit="µm", orientation='h'
):
    """
    Draws a horizontal or vertical scale bar.

    orientation: 'h' for horizontal, 'v' for vertical
    """

    if orientation == 'h':
        # horizontal bar
        ax.plot(
            [-size / 2 + x0, +size / 2 + x0],
            [y0, y0],
            c='k', solid_capstyle='butt', clip_on=False
        )
        if text:
            ax.text(
                x0, y0 - tdist,
                f'{size:.0f} {unit}',
                c='k', va='top', ha='center', fontsize=fontsize
            )

    elif orientation == 'v':
        # vertical bar
        ax.plot(
            [x0, x0],
            [-size / 2 + y0, +size / 2 + y0],
            c='k', solid_capstyle='butt', clip_on=False
        )
        if text:
            ax.text(
                x0 + tdist, y0,
                f'{size:.0f} {unit}',
                c='k', va='center', ha='left', fontsize=fontsize
            )

    else:
        raise ValueError("orientation must be 'h' or 'v'")


def plot_mean_and_sd(ax, traces, time, color='black', alt_color='dimgray', facealpha=0.2, offset=0.0):
    if traces.shape[0] <= 2:
        ax.plot(time, traces[0] - np.mean(traces[0]) + offset, color=color)
        if len(traces) == 2:
            ax.plot(time, traces[1] - np.mean(traces[1]) + offset, color=alt_color)
    else:
        mu = np.mean(traces, axis=0)
        mu = mu - np.mean(mu) + offset
        sd = np.std(traces, axis=0)

        ax.plot(time, mu, color=color)
        ax.fill_between(time, mu - sd, mu + sd, color=color, alpha=facealpha)


def get_extent(stack_avg, pixel_size_um, x_offset, y_offset):
    ps = pixel_size_um
    w, h = stack_avg.shape[:2]
    assert w == h
    extent = np.array([-w / 2 * ps, +w / 2 * ps, -w / 2 * ps, +w / 2 * ps])
    extent += (x_offset, x_offset, y_offset, y_offset)
    return extent


def plot_stack_average(ax, stack_avg, pixel_size_um, x_offset, y_offset, cmap='viridis', alpha=0.7, gamma=0.5):
    extent = get_extent(stack_avg, pixel_size_um, x_offset, y_offset)

    im = stack_avg.astype(float)
    vmin = np.percentile(im, q=5, axis=(0, 1))
    vmax = np.percentile(im, q=99, axis=(0, 1))
    im = (im - vmin) / (vmax - vmin)
    im = np.clip(im, 0, 1) ** gamma

    ax.imshow(im.T, extent=extent, cmap=cmap, interpolation='none', alpha=alpha)

    return extent


def make_square_bounding_box(xs, ys):
    # Step 1: Find initial min and max
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)

    # Step 2: Determine width and height
    width = xmax - xmin
    height = ymax - ymin

    # Step 3: Expand the smaller side to match the larger
    if width > height:
        # Increase height
        diff = width - height
        ymin -= diff / 2
        ymax += diff / 2
    elif height > width:
        # Increase width
        diff = height - width
        xmin -= diff / 2
        xmax += diff / 2

    return xmin, xmax, ymin, ymax


def plot_roi_mask(ax, rois, extent):
    _rois = -rois.copy()
    _rois = _rois.astype(float)
    _rois[_rois <= 0] = np.nan
    _rois = np.repeat(np.repeat(_rois, 3, axis=0), 3, axis=0)
    ax.imshow(_rois.T, cmap='jet', extent=extent)


def add_rect(ax, box_xlim, box_ylim, color_crop, linewidth=1.2):
    rect = patches.Rectangle(
        (box_xlim[0], box_ylim[0]), box_xlim[1] - box_xlim[0], box_ylim[1] - box_ylim[0],
        linewidth=linewidth, edgecolor=color_crop, facecolor='none', linestyle='--', clip_on=False
    )
    ax.add_patch(rect)


def plot_sac_lines(ax, xlim, text=True):
    ax.plot(xlim, [0, 0], c='darkblue', ls='--', lw=1.5)
    ax.plot(xlim, [12, 12], c='darkgreen', ls='--', lw=1.5)
    if text:
        ax.text(xlim[1], 0, '  ON', va='center', ha='left', color='darkblue', fontsize=8)
        ax.text(xlim[1], 12, '  OFF', va='center', ha='left', color='darkgreen', fontsize=8)


def plot_ipl_profile(ax, row):
    from pywarper.warpers import get_z_profile
    import skeliner as sk
    from copy import deepcopy
    
    skel = deepcopy(row['skel'])
    skel.node2verts = None
    sk.post.prune(
        skel=skel,
        kind="nodes",
        nodes=np.where(skel.ntype == 2)[0]
    )
    zlim = (-30, 30)
    
    z_dict = get_z_profile(
        skel=skel,
        extent=zlim,
    )
    ipl = z_dict['x']
    dens = z_dict['distribution']

    vmax = dens.max()
    xlim = -0.1*vmax, vmax*1.1

    ax.set_aspect('auto', 'box')
    ax.set_ylim(zlim)

    plot_sac_lines(ax, xlim, text=False)

    ax.set_xlim(xlim)
    ax.plot(dens, ipl, c='darkred', lw=2)
    #ax.set_title('xz', loc='left', y=0.9, va='top', fontsize=20)
    ax.set(xticks=[], yticks=[], xlabel=None, ylabel=None)
    ax.axis('off')