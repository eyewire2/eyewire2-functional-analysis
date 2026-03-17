import os

import pandas as pd
import numpy as np
from eyewire2_functional_analysis.io import restore_numpy_arrays


def load_parquet_df(filepath):
    df_flat = pd.read_parquet(filepath)
    df = restore_numpy_arrays(df_flat)
    df = df.map(lambda x: np.array(x) if isinstance(x, list) else x)
    return df


def load_df_rois(data_folder):
    df_rois = pd.concat([
        load_parquet_df(os.path.join(data_folder, f'df_eyewire2_roi_level_GCL{i}.parquet'))
        for i in range(5)])
    df_rois['qfilt'] = (df_rois['bar_qidx'] > 0.6) | (df_rois['chirp_qidx'] > 0.45)
    return df_rois


def load_df_fields(data_folder):
    df_fields = load_parquet_df(os.path.join(data_folder, 'df_eyewire2_field_level.parquet'))
    return df_fields


def load_df_outline(data_folder):
    df_outline = load_parquet_df(os.path.join(data_folder, 'df_eyewire2_outline.parquet'))
    return df_outline


def load_all_dfs(data_folder):
    df_rois = load_df_rois(data_folder)
    df_fields = load_df_fields(data_folder)
    df_outline = load_df_outline(data_folder)
    return df_rois, df_fields, df_outline


def load_df_rois_morph(morph_folder, nuc_col_master, seg_col_master, data_folder=None, df_rois=None,
                       morph_spreadsheet_filename="Eyewire II Proofread Cells Master List - All Cells 2025-11-24.csv"):
    if df_rois is None:
        assert data_folder is not None, "data_folder must be provided if df_rois is None"
        df_rois = load_df_rois(data_folder)

    df_master = pd.read_csv(os.path.join(os.path.join(morph_folder, morph_spreadsheet_filename)), dtype=str).dropna(axis=1, how='all')

    assert seg_col_master in df_master.columns, f"Column '{seg_col_master}' not found in df_master {list(df_master.columns)}"
    assert nuc_col_master in df_master.columns, f"Column '{nuc_col_master}' not found in df_master {list(df_master.columns)}"

    df_merged = pd.merge(
        df_master.set_index(nuc_col_master),
        df_rois.set_index(df_rois['nuc_id'].astype(str)),
        left_index=True, right_index=True, how='inner'
    ).reset_index()

    return df_merged
