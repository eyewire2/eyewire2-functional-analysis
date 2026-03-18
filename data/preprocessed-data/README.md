Preprocessed calcium traces from OGB-1 recordings, stored as parquet files.

One parquet file per GCL recording field (5 total), plus field-level and outline DataFrames:

- `df_eyewire2_roi_level_GCL{0..4}.parquet` — ROI-level data (traces, quality indices, trigger times, …)
- `df_eyewire2_field_level.parquet` — field-level metadata
- `df_eyewire2_outline.parquet` — retinal outline data

Use `eyewire2_functional_analysis.data_loader` to load these files; see the [README](../../README.md) for examples.
