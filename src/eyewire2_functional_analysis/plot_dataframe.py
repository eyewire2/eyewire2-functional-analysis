import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt


def plot_df_chirp_and_bar(df, id_col='Nuc ID', type_col='Cell Type', title='',
                          chirp_q_thresh=0.45, bar_q_thresh=0.6):
    if df.shape[0] == 0:
        raise Exception('No data found')
    if id_col not in df.columns:
        raise Exception('ID column not found')
    if type_col not in df.columns:
        raise Exception('Type column not found')

    t_chirp = np.arange(df.chirp_average_norm.iloc[0].size) * df.chirp_average_dt.iloc[0]
    t_bar = np.arange(df.bar_time_component.iloc[0].size) * df.bar_snippets_dt.iloc[0]

    fig, axs = plt.subplots(1, 7, figsize=(14, df.shape[0] * 0.5 + 1),
                            width_ratios=(5, 4, 2, 1, 1, 1, 1), sharey=True)
    sns.despine()

    ax_annot = axs[0]
    ax_chirp = axs[1]
    ax_bar = axs[2]
    ax_soma = axs[3]
    ax_qidx = axs[4]
    ax_ds = axs[5]
    ax_os = axs[6]

    offset_scale = 2
    for ii, (i, row) in enumerate(df.iterrows()):
        offset = (df.shape[0] - 1 - ii) * offset_scale
        c = f"C{ii}"

        ax_chirp.plot(t_chirp, offset + row.chirp_average_norm, c=c, alpha=1 if row.qfilt else 0.5)
        ax_chirp.fill_between(t_chirp, offset + row.chirp_average_norm, np.full_like(row.chirp_average_norm, offset),
                            color=c, alpha=0.5 if row.qfilt else 0.25)

        ax_bar.plot(t_bar, offset + row.bar_time_component, c=c, alpha=1 if row.qfilt else 0.5)
        ax_bar.fill_between(t_bar, offset + row.bar_time_component, np.full_like(row.bar_time_component, offset),
                            color=c, alpha=0.5 if row.qfilt else 0.25)

        ax_soma.plot(row.roi_dia_um, offset, 'o', c=c, clip_on=False)

        ax_ds.plot(np.maximum(1e-2, row.bar_ds_pvalue), offset, 'o', c=c, clip_on=False,
                    mfc='none' if row.bar_ds_pvalue >= 0.05 else c)

        ax_os.plot(np.maximum(1e-2, row.bar_os_pvalue), offset, 'o', c=c, clip_on=False,
                    mfc='none' if row.bar_os_pvalue >= 0.05 else c)

        ax_qidx.plot(row.chirp_qidx - chirp_q_thresh, offset, marker='D', c=c, clip_on=False, alpha=0.8,
                mfc='none' if not row.qfilt else c)
        ax_qidx.plot(row.bar_qidx - bar_q_thresh, offset, 's', c=c, clip_on=False, alpha=0.8, mfc='none' if not row.qfilt else c)

        ax_annot.text(0.1, offset, f"{row['field']} | {row['roi_id']:02} | {row[id_col]} | {str(row[type_col])[:10]}",
                      ha='left', va='center', c=c)

    ax = axs[0]
    ax.set_yticks([])

    ax = ax_annot
    ax.set_xticks([])
    ax.text(0.1, df.shape[0] * offset_scale, title, ha='left', va='center', fontsize=12)
    ax.set_ylim(-offset_scale, (df.shape[0] + 1) * offset_scale)

    ax = ax_chirp
    ax.set(xlabel='Time [s]')
    for t in [2, 5, 8, 30]:
        ax.axvline(t, c='k', alpha=0.5, ls='--')

    ax = ax_bar
    ax.set(xlabel='Time [s]')
    for t in [1.152, 2.432, 3.712]:
        ax.axvline(t, c='k', alpha=0.5, ls='--')

    ax = ax_soma
    ax.set(xlabel='ØROI\n[µm]')
    ax.grid()
    ax.set_xlim(4, 20)

    ax = ax_ds
    ax.set(xlabel='p-val (DS)')
    ax.grid()
    ax.set(xscale='log')
    ax.axvline(0.05, c='k', alpha=0.5, ls='--')
    ax.set_xlim(1e-2 / 2, 1)

    ax = ax_os
    ax.set(xlabel='p-val (OS)')
    ax.grid()
    ax.set(xscale='log')
    ax.axvline(0.05, c='k', alpha=0.5, ls='--')
    ax.set_xlim(1e-2 / 2, 1)

    ax = ax_qidx
    ax.set(xlabel=r'QI-QI$_{0}$' + '\nMB: ■')
    ax.grid()
    ax.set_xlim(-0.6, 0.6)
    ax.set_xticks([-0.5, 0.5])
    ax.axvline(0, c='k', alpha=0.5, ls='--')


    plt.tight_layout()
    return fig, axs
