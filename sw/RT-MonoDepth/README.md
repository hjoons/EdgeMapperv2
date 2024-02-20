# Training using RT-MonoDepth

Evaluation has not yet been implemented. To train RT-Monodepth, call ```python runner.py``` followed by the customized arguments that are located in ```run_config.txt```. Some arguments specified in ```runner.py``` are not used or are repetitive, but are implemented in case they are required in future runs.

It is also worth noting that ```dataloader.py``` is adapted from [here](https://github.com/ShuweiShao/NDDepth/blob/main/Estimation/nddepth/dataloaders/dataloader.py), so the implemented data augmentations are not from the *RT-MonoDepth (2023)* which may cause some deviations from the paper's published results (in the case of successful training).

# Changing the model
```layer.py``` contains all the necessary building blocks for *RT-MonoDepth (2023)*.

### DownBlock and DoubleConv

DoubleConv is the double 3x3 convolution used in *RT-MonoDepth (2023)*. This DoubleConv module takes an argument ```dw``` that specifies whether the model will utilize a depthwise separable convolution, or a normal 2D convolution. Notice that *there are no batch normalization calls* as BatchNorm is unnecessary and hinders latency according to *RT-MonoDepth (2023)*. This could be toyed with as we are unsure if a lack of batch normalization is causing the training and testing loss to plateau.

DownBlock creates a skip connection **prior** to down sampling; this feature map is used when upsampling with a pointwise convolution for the channels to match.

### NNConv and Fusions

NNConv comes from *FastDepth (2019)* and is a depthwise separable convolution followed by NN upsampling. This kernel size is customizable and could potentially be changed. *RT-MonoDepth (2023)* states NNConv3 is sufficient, while *FastDepth (2019)* utilizes NNConv5. The ```depthwise()``` and ```pointwise``` functions use ```BatchNorm2D()``` which is primarily why our encoder layers do not utilize these methods.

The fusion blocks are very straight forward element-wise addition and concat techniques that utilize the skip connections from previous layers. These should not be subject to much change. We could potentially implement the self-supervised approach in *RT-MonoDepth (2023)* by feeding each output from fusion blocks into the decoder architecture that's defined in the paper. 