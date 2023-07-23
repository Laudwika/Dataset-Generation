# Dataset Generation
Large Scale Dataset Generation for Face Recognition Applications.

# Datasets
Download the files [Texture Files](), [Segmentation Files](), [Light Files]() and put them in root.

# Models and Weights
Download the [Models]() from the models folder and [Weights]() from the weights folder and put them in root.

# How to use
We can either generate with random attributes or manually generate an image

## Randomized Attributes
In generate.py in main uncomment generate() on line 492, this will generate images from a combination of random attributes.

## Controlled Attribute Generation
in generate.py in main, manual() function on line 493 will lead to the manual function in line 450. This is an example code of how to generate an image given constraints, this can be modifieable to be able to read txt files and the like.

# Process Visualization
## Texture Infusion -> Inversion -> Pose Transfer -> Image Enhancement -> Age Manipulation
<img src="https://github.com/Laudwika/Dataset-Generation/blob/main/test/age.jpg" width="256" height="256" /> <img src="https://github.com/Laudwika/Dataset-Generation/blob/main/test/style.jpg" width="256" height="256" /><img src="https://github.com/Laudwika/Dataset-Generation/blob/main/test/pose.jpg" width="256" height="256" /><img src="https://github.com/Laudwika/Dataset-Generation/blob/main/test/enhanced.jpg" width="256" height="256" /><img src="https://github.com/Laudwika/Dataset-Generation/blob/main/test/age.jpg" width="256" height="256" />

# Other Examples
## Custom Attribute Manipulation
![custom](/test/customized.jpg)

## Random Attribute Generation
![attribute](/test/attributes.jpg)

## Variety of Different Identities
![multiple](/test/multiple.jpg)

## Variety of Same Identities
![same](/test/one.jpg)
