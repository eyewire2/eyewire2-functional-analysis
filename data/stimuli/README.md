# Disclaimer
- TODO: Check if exact details match the stimuli that were actually used.
- TODO: Add orientation of MB, corrected for setup.
- TODO: Add open-retina examples.
- TODO: Improve MC train/test order explanation.
- TODO: Add MC test sequence order.

# Stimulus description
The stimuli, chirp, moving bar (MB) and mouse camera movies (MCs), used here are the same as in [Höfling et al. 2024](https://elifesciences.org/articles/86860).  
Pickled movie files of the stimuli, that can be used e.g. for model training, can be downloaded [here](https://huggingface.co/datasets/open-retina/open-retina/tree/main/euler_lab/hoefling_2024/stimuli).

## Chirp and moving bar
There are also QDSpy stimulator scripts for [chirp](https://github.com/eulerlab/QDSpy/blob/master/Stimuli/RGC_Chirp_2.py) and [MB](https://github.com/eulerlab/QDSpy/blob/master/Stimuli/RGC_MovingBar_2.py).

## Mouse camera movies
MCs were the same as in [Höfling et al. 2024](https://elifesciences.org/articles/86860) which were derived from recordings by [Qiu et al. 2021](https://www.cell.com/current-biology/fulltext/S0960-9822(21)00676-X).  
There are different MC stimuli, e.g. MC-16 and MC-20, that are all based on the same 113 five-second movie clips.  
Each MC stimulus MC-X is described by a sequence [test/train1-X/test/train2-X/test](https://iiif.elifesciences.org/lax:86860%2Felife-86860-fig1-v1.tif/full/1500,/0/default.jpg),
where:
- the test sequence, consisting of five movie clips, is the same for all MC versions and is shown three times (the beginning, the middle and the end),
- and the train sequences train1-X and train2-X consist of 54 movie clips, each, that are differently ordered for each MC stimuli MC-X.

Each MC therefore consists of a sequence of 123 five-second movie clips, with 123 corresponding triggertimes.
The order of the train sequences is stored in `mc_train_sequences.npy`.
