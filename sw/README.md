# Dataset Links

[Download NYUv2](https://www.kaggle.com/datasets/soumikrakshit/nyu-depth-v2) (50K Dataset)
[Download Guided Decoding (GuideDepth) Testset](https://drive.google.com/file/d/1hXvznCAa26bNBPGZJH1DI2siVxmQlm0W/view) (For evaluation only)

## Using NYUv2

Place NYUv2.zip (DO NOT EXTRACT) one level outside of where ```run_config``` is called. Rename ```archive.zip``` to ```nyuv2.zip```. Currently, the given ```run_config``` script should be able to find ```nyuv2.zip``` if it is located directly one directory level above. You can change this *easily* by by changing ```train_path``` in the ```run_config```.

## Using GuideDepth

*Extract* and place ```NYU_Testset``` inside of ```GuideDepth```. 
In command line run 
```python main.py --eval --dataset nyu_reduced --resolution full --model GuideDepth --test_path ./NYU_Testset --num_workers=4 --save_results results --weight_path <yourckpt>```.
If any errors arise with ```.pth```, locate ```DDRNet23s_imagenet.pth``` and ```NYU_Full_GuideDepth.pth```, and replace the two locations ```.pth``` files are used with the absolute path on your local machine.
```.pth``` is not used **anywhere** else in this repository, so replacing should be trivial.
