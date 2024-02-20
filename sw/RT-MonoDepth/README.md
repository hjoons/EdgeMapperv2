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

# Output of summary.py

```----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1          [-1, 3, 480, 640]              27
              ReLU-2          [-1, 3, 480, 640]               0
            Conv2d-3          [-1, 3, 480, 640]              27
              ReLU-4          [-1, 3, 480, 640]               0
            Conv2d-5         [-1, 64, 480, 640]             192
              ReLU-6         [-1, 64, 480, 640]               0
        DoubleConv-7         [-1, 64, 480, 640]               0
            Conv2d-8         [-1, 64, 240, 320]          36,928
         DownBlock-9  [[-1, 64, 240, 320], [-1, 64, 480, 640]]               0
           Conv2d-10         [-1, 64, 240, 320]             576
             ReLU-11         [-1, 64, 240, 320]               0
           Conv2d-12         [-1, 64, 240, 320]             576
             ReLU-13         [-1, 64, 240, 320]               0
           Conv2d-14        [-1, 128, 240, 320]           8,192
             ReLU-15        [-1, 128, 240, 320]               0
       DoubleConv-16        [-1, 128, 240, 320]               0
           Conv2d-17        [-1, 128, 120, 160]         147,584
        DownBlock-18  [[-1, 128, 120, 160], [-1, 128, 240, 320]]               0
           Conv2d-19        [-1, 128, 120, 160]           1,152
             ReLU-20        [-1, 128, 120, 160]               0
           Conv2d-21        [-1, 128, 120, 160]           1,152
             ReLU-22        [-1, 128, 120, 160]               0
           Conv2d-23        [-1, 256, 120, 160]          32,768
             ReLU-24        [-1, 256, 120, 160]               0
       DoubleConv-25        [-1, 256, 120, 160]               0
           Conv2d-26          [-1, 256, 60, 80]         590,080
        DownBlock-27  [[-1, 256, 60, 80], [-1, 256, 120, 160]]               0
           Conv2d-28          [-1, 256, 60, 80]           2,304
             ReLU-29          [-1, 256, 60, 80]               0
           Conv2d-30          [-1, 256, 60, 80]           2,304
             ReLU-31          [-1, 256, 60, 80]               0
           Conv2d-32          [-1, 512, 60, 80]         131,072
             ReLU-33          [-1, 512, 60, 80]               0
       DoubleConv-34          [-1, 512, 60, 80]               0
           Conv2d-35          [-1, 512, 30, 40]       2,359,808
        DownBlock-36  [[-1, 512, 30, 40], [-1, 512, 60, 80]]               0
           Conv2d-37          [-1, 512, 30, 40]           4,608
      BatchNorm2d-38          [-1, 512, 30, 40]           1,024
             ReLU-39          [-1, 512, 30, 40]               0
           Conv2d-40          [-1, 256, 30, 40]         131,072
      BatchNorm2d-41          [-1, 256, 30, 40]             512
             ReLU-42          [-1, 256, 30, 40]               0
           NNConv-43          [-1, 256, 60, 80]               0
           Conv2d-44          [-1, 256, 60, 80]         131,072
      BatchNorm2d-45          [-1, 256, 60, 80]             512
             ReLU-46          [-1, 256, 60, 80]               0
           Conv2d-47          [-1, 256, 60, 80]         589,824
      BatchNorm2d-48          [-1, 256, 60, 80]             512
             ReLU-49          [-1, 256, 60, 80]               0
    FusionElement-50          [-1, 256, 60, 80]               0
           Conv2d-51          [-1, 256, 60, 80]           2,304
      BatchNorm2d-52          [-1, 256, 60, 80]             512
             ReLU-53          [-1, 256, 60, 80]               0
           Conv2d-54          [-1, 128, 60, 80]          32,768
      BatchNorm2d-55          [-1, 128, 60, 80]             256
             ReLU-56          [-1, 128, 60, 80]               0
           NNConv-57        [-1, 128, 120, 160]               0
           Conv2d-58        [-1, 128, 120, 160]          32,768
      BatchNorm2d-59        [-1, 128, 120, 160]             256
             ReLU-60        [-1, 128, 120, 160]               0
           Conv2d-61        [-1, 128, 120, 160]         147,456
      BatchNorm2d-62        [-1, 128, 120, 160]             256
             ReLU-63        [-1, 128, 120, 160]               0
    FusionElement-64        [-1, 128, 120, 160]               0
           Conv2d-65        [-1, 128, 120, 160]           1,152
      BatchNorm2d-66        [-1, 128, 120, 160]             256
             ReLU-67        [-1, 128, 120, 160]               0
           Conv2d-68         [-1, 64, 120, 160]           8,192
      BatchNorm2d-69         [-1, 64, 120, 160]             128
             ReLU-70         [-1, 64, 120, 160]               0
           NNConv-71         [-1, 64, 240, 320]               0
           Conv2d-72         [-1, 64, 240, 320]         110,592
      BatchNorm2d-73         [-1, 64, 240, 320]             128
             ReLU-74         [-1, 64, 240, 320]               0
     FusionConcat-75         [-1, 64, 240, 320]               0
           Conv2d-76         [-1, 64, 240, 320]             576
      BatchNorm2d-77         [-1, 64, 240, 320]             128
             ReLU-78         [-1, 64, 240, 320]               0
           Conv2d-79          [-1, 1, 240, 320]              64
      BatchNorm2d-80          [-1, 1, 240, 320]               2
             ReLU-81          [-1, 1, 240, 320]               0
           NNConv-82          [-1, 1, 480, 640]               0
================================================================
Total params: 4,511,672
Trainable params: 4,511,672
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 3.52
Forward/backward pass size (MB): 8025.45
Params size (MB): 17.21
Estimated Total Size (MB): 8046.17
----------------------------------------------------------------
[INFO] Register count_convNd() for <class 'torch.nn.modules.conv.Conv2d'>.
[INFO] Register zero_ops() for <class 'torch.nn.modules.activation.ReLU'>.
[INFO] Register zero_ops() for <class 'torch.nn.modules.container.Sequential'>.
[INFO] Register count_normalization() for <class 'torch.nn.modules.batchnorm.BatchNorm2d'>.
MACs: 29.511168 billion
Parameters: 4.511672 million```
