# Dataset Generation
Large Scale Dataset Generation for Face Recognition Applications.

# Datasets
Download the files [Texture Files](https://drive.google.com/drive/folders/1YNFWIvkAA_bqVOwRuBqaMKUq2j00zskd?usp=drive_link), [Segmentation Files](https://drive.google.com/drive/folders/1iOfkRUSQfS8caQMzIQNNjEcAxEFdq8IO?usp=drive_link), [Light Files](https://drive.google.com/drive/folders/1D41KHkGJgZX5DFojrhHkFATaulXzJ2wf?usp=drive_link) and put them in the root in their respective folders.

# Models and Weights
Download the [Models](https://drive.google.com/drive/folders/1r23bDC2fJGEKt5X3w1-P6iQkvcPmxMoP?usp=drive_link) from the models folder and [Weights](https://drive.google.com/drive/folders/1Cghp-y-OfUC29MpP6r9J3W2d-SzorIQk?usp=drive_link) from the weights folder and put them in the /models and /weights in root.

# How to use
We can either generate with random attributes or manually generate an image.

## Randomized Attributes
In generate.py in main uncomment generate() on line 492, this will generate images from a combination of random attributes.

## Controlled Attribute Generation
in generate.py in main, manual() function on line 493 will lead to the manual function in line 450. This is an example code of how to generate an image given constraints; this can be modifiable to be able to read txt files and the like.

# Process Visualization
## Texture Infusion -> Inversion -> Pose Transfer -> Image Enhancement -> Age Manipulation
<img src="https://github.com/Laudwika/Dataset-Generation/blob/main/test/textured.jpg" width="256" height="256" /><img src="https://github.com/Laudwika/Dataset-Generation/blob/main/test/style.jpg" width="256" height="256" /><img src="https://github.com/Laudwika/Dataset-Generation/blob/main/test/pose.jpg" width="256" height="256" /><img src="https://github.com/Laudwika/Dataset-Generation/blob/main/test/enhanced.jpg" width="256" height="256" /><img src="https://github.com/Laudwika/Dataset-Generation/blob/main/test/age.jpg" width="256" height="256" />

# Other Examples
## Custom Attribute Manipulation
![custom](/test/customized.png)

## Random Attribute Generation
![attribute](/test/attributes.png)

## Variety of Different Identities
![multiple](/test/multiple.png)

## Variety of Same Identities
![same](/test/one.png)

## Repositories Used
[Inversion](https://github.com/bryandlee/stylegan2-encoder-pytorch), [Relighting](https://github.com/zhhoper/DPR), [Pose Transfer](https://github.com/zhengkw18/face-vid2vid), [Image Enhancement](https://github.com/yangxy/GPEN/blob/main/README.md), [Age Manipulation](https://github.com/yuval-alaluf/SAM)
