import os

import numpy as np
import pandas as pd

from eyewire2_functional_analysis.io import restore_numpy_arrays


def load_parquet_df(filepath):
    """Load a parquet file and restore any serialised numpy arrays.

    Args:
        filepath: Path to the ``.parquet`` file.

    Returns:
        pandas.DataFrame: DataFrame with numpy arrays restored in object columns.
    """
    df_flat = pd.read_parquet(filepath)
    df = restore_numpy_arrays(df_flat)
    df = df.map(lambda x: np.array(x) if isinstance(x, list) else x)
    return df


def load_df_rois(data_folder):
    """Load and concatenate all GCL ROI-level parquet files from ``data_folder``.

    Reads five parquet files named ``df_eyewire2_roi_level_GCL{0..4}.parquet``,
    concatenates them, and adds a boolean ``qfilt`` column based on quality indices.

    Args:
        data_folder: Path to the directory containing the parquet files.

    Returns:
        pandas.DataFrame: Combined ROI-level DataFrame with ``qfilt`` column.
    """
    df_rois = pd.concat([
        load_parquet_df(os.path.join(data_folder, f'df_eyewire2_roi_level_GCL{i}.parquet'))
        for i in range(5)])
    df_rois['qfilt'] = (df_rois['bar_qidx'] > 0.6) | (df_rois['chirp_qidx'] > 0.45)
    return df_rois


def load_df_fields(data_folder):
    """Load the field-level parquet file from ``data_folder``.

    Args:
        data_folder: Path to the directory containing ``df_eyewire2_field_level.parquet``.

    Returns:
        pandas.DataFrame: Field-level DataFrame.
    """
    df_fields = load_parquet_df(os.path.join(data_folder, 'df_eyewire2_field_level.parquet'))
    return df_fields


def load_df_outline(data_folder):
    """Load the outline parquet file from ``data_folder``.

    Args:
        data_folder: Path to the directory containing ``df_eyewire2_outline.parquet``.

    Returns:
        pandas.DataFrame: Outline DataFrame.
    """
    df_outline = load_parquet_df(os.path.join(data_folder, 'df_eyewire2_outline.parquet'))
    return df_outline


def load_all_dfs(data_folder):
    """Load ROI-level, field-level, and outline DataFrames in one call.

    Args:
        data_folder: Path to the directory containing all required parquet files.

    Returns:
        tuple: ``(df_rois, df_fields, df_outline)`` – the three DataFrames.
    """
    df_rois = load_df_rois(data_folder)
    df_fields = load_df_fields(data_folder)
    df_outline = load_df_outline(data_folder)
    return df_rois, df_fields, df_outline


def load_df_rois_morph(morph_folder, nuc_col_master,
                       seg_col_master=(
                               'Updated Seg ID\n(Feb 04, 2026)\nIF YOU UPDATE THIS COLUMN, ALSO UPDATE Final SegID!',
                               'Updated Seg ID (Sept 2)',
                               'Final SegID'),
                       data_folder=None, df_rois=None,
                       morph_spreadsheet_filename="Eyewire II Proofread Cells Master List - All Cells 2025-11-24.csv"):
    """Load and merge the ROI-level DataFrame with the morphology master spreadsheet.

    Reads a CSV master list of proofread cells and performs an inner join with the
    ROI-level DataFrame on the nucleus ID column.

    Args:
        morph_folder: Path to the directory containing the morphology spreadsheet.
        nuc_col_master: Column name in the master CSV used as the nucleus ID key.
        seg_col_master : Column or tuple of columns used to find Final SegID in the master spreadsheet.
            The function will check these columns in order and use the first non-null value as the 'Latest SegID'.
        data_folder: Path to the directory containing parquet files. Required when
            ``df_rois`` is ``None``.
        df_rois: Pre-loaded ROI-level DataFrame. If ``None``, it is loaded from
            ``data_folder``.
        morph_spreadsheet_filename: Filename of the CSV master list within
            ``morph_folder``.

    Returns:
        pandas.DataFrame: Merged DataFrame containing columns from both the master
        spreadsheet and the ROI-level data, indexed by nucleus ID.

    Raises:
        AssertionError: If required columns are missing from the master spreadsheet,
            or if ``data_folder`` is not provided when ``df_rois`` is ``None``.
    """
    if df_rois is None:
        assert data_folder is not None, "data_folder must be provided if df_rois is None"
        df_rois = load_df_rois(data_folder)

    df_master = pd.read_csv(os.path.join(os.path.join(morph_folder, morph_spreadsheet_filename)), dtype=str).dropna(
        axis=1, how='all')

    assert nuc_col_master in df_master.columns, f"Column '{nuc_col_master}' not found in df_master {list(df_master.columns)}"

    # Normalise seg_col_master to a tuple so it works whether a single string or tuple was passed
    if isinstance(seg_col_master, str):
        seg_col_master = (seg_col_master,)

    missing = [c for c in seg_col_master if c not in df_master.columns]
    assert not missing, f"seg_col_master column(s) not found in df_master: {missing}"

    # Drop duplicates in df_master based on the nucleus ID column, keeping the first occurrence
    df_master = df_master.drop_duplicates(subset=nuc_col_master, keep='first')

    df_merged = pd.merge(
        df_master.set_index(nuc_col_master),
        df_rois.set_index(df_rois['nuc_id'].astype(str)),
        left_index=True, right_index=True, how='inner'
    ).reset_index()

    # Fold seg_col_master columns in order, taking the first non-null value as Latest SegID
    latest = df_merged[seg_col_master[0]]
    for col in seg_col_master[1:]:
        latest = latest.combine_first(df_merged[col])
    df_merged['Latest SegID'] = latest

    return df_merged