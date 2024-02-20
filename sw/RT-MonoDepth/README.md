## Training using RT-MonoDepth

Evaluation has not yet been implemented. To train RT-Monodepth, call ```python runner.py``` followed by the customized arguments that are located in ```run_config.txt```. Some arguments specified in ```runner.py``` are not used or are repetitive, but are implemented in case they are required in future runs.

It is also worth noting that ```dataloader.py``` is adapted from [here](https://github.com/ShuweiShao/NDDepth/blob/main/Estimation/nddepth/dataloaders/dataloader.py), so the implemented data augmentations are not from the RT-MonoDepth paper.

## Changing the model