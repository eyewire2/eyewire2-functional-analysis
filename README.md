## Principles of Collaboration

**All data and code shared here is subject to the [Eyewire II Principles of Collaboration](https://eyewire.ai/principles).** By the use of this repository, you agree to be bound by these Principles.

# Eyewire II: Functional data

This repository hosts the functional data from OGB-1 recordings in the Eyewire II dataset.

This is work-in-progress and currently holds 
- pre-processed Calcium traces from recorded over five recording fields: [h5 files](https://github.com/eulerlab/eyewire2-functional-analysis/tree/main/data/preprocessed-data)
- example code to [load](https://github.com/eulerlab/eyewire2-functional-analysis/blob/main/notebooks/load_data.ipynb) and [plot](https://github.com/eulerlab/eyewire2-functional-analysis/blob/main/notebooks/plot_data_overview/plot_data_overview.ipynb) the pre-processed data
- [summary figures](https://github.com/eulerlab/eyewire2-functional-analysis/tree/main/notebooks/plot_data_overview/figures) of all cells, grouped by a preliminary clustering and recording field

Stimuli can be found here: https://huggingface.co/datasets/open-retina/open-retina/tree/main/euler_lab/hoefling_2024/stimuli.  
Documentation will be added soon.

Feel free to open issues to ask questions and request features!

### Setup

To use the code in this repository out of the box, you can use [uv](https://docs.astral.sh/uv/) to reproduce our python environment.
