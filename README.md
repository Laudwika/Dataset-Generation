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
## 1. Texture Infusion
![texture](/test/textured.jpg)

## 2. Inversion
![style](/test/style.jpg)

## 3. Pose Transfer
![pose](/test/pose.jpg)

## 4. Image Enhancement
![enhanced](/test/enhanced.jpg)

## 5. Age Manipulation
![age](/test/age.jpg)

# Other Examples
## Custom Attribute Manipulation
![custom](/test/customized.png)

## Random Attribute Generation
![attribute](/test/attributes.png)

## Variety of Different Identities
![multiple](/test/multiple.png)

## Variety of Same Identities
![same](/test/one.png)
