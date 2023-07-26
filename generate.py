from __future__ import print_function
from binascii import b2a_base64
from netrc import netrc
from re import A
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import torch.utils.data as data
import torch.nn.functional as F
import torch.optim as optim
from torchvision.models import vgg19
import torchvision.transforms.functional as TF

import numpy as np
import os
import random
import cv2
from PIL import Image
from skimage import io, img_as_float32


from networks.customspade import CustomSPADE
from networks.generator import G
from networks.tex_generator import G_tex
from SPADE.options.train_options import TrainOptions

from model import Encoder as Encoder, Generator as Generator_style
from vid2vid_models import AFE, CKD, HPE_EDE, MFE, Generator

from utility import transform_kp

# from evaluate import eval
from utils.augmentations import AgeTransformer
from utils.common import tensor2im
from utils.utils_SH import *

from networks.psp import pSp
from argparse import Namespace
from face_enhance import enhance
from networks.defineHourglass_512_gray_skip import *



###########################################################
#####################INITIALIZATION########################
topil = transforms.ToPILImage()
device = torch.device("cuda")
def preprocess(img_path, output_size):
    transform = transforms.Compose([
        transforms.Resize(output_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
    image = Image.open(img_path)
    return transform(image).unsqueeze(0)
transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])


###########################################################
##########################LIGHT############################
def get_normal(img_size=256):
    x = np.linspace(-1, 1, img_size)
    z = np.linspace(1, -1, img_size)
    x, z = np.meshgrid(x, z)

    mag = np.sqrt(x**2 + z**2)
    valid = mag <=1
    y = -np.sqrt(1 - (x*valid)**2 - (z*valid)**2)
    x = x * valid
    y = y * valid
    z = z * valid
    normal = np.concatenate((x[...,None], y[...,None], z[...,None]), axis=2)
    normal = np.reshape(normal, (-1, 3))

    return normal, valid

def load_model(model_folder='models/', model_name='trained_model_03.t7'):
    my_network = HourglassNet()
    my_network.load_state_dict(torch.load(os.path.join(model_folder, model_name)))
    my_network.cuda()
    my_network.train(False)
    return my_network


###########################################################
###########################AGE#############################
model_path = 'models/sam_ffhq_aging.pt'
ckpt = torch.load(model_path, map_location='cuda')
opts = ckpt['opts']
opts['checkpoint_path'] = model_path
opts = Namespace(**opts)
net = pSp(opts)
net.eval()
net.cuda()


###########################################################
##########################POSE#############################

def eval(ckp_dir = 'ckp', output = '', ckp = 100, source = '', driving = ''):
    g_models = {"afe": AFE(), "ckd": CKD(), "hpe_ede": HPE_EDE(), "mfe": MFE(), "generator": Generator()}
    # ckp_path = os.path.join(ckp_dir, "%s-checkpoint.pth.tar" % str(ckp).zfill(8))
    checkpoint = torch.load('models/00000100-checkpoint.pth.tar', map_location=torch.device("cuda"))

    for k, v in g_models.items():
        v.to(device)
        v.load_state_dict(checkpoint[k])
        v.eval()
    output_frames = []

    s = img_as_float32(np.array(source))[:, :, :3]
    s = np.array(s, dtype="float32").transpose((2, 0, 1))
    s = torch.from_numpy(s).to(device).unsqueeze(0)
    s = F.interpolate(s, size=(256, 256))
    fs = g_models["afe"](s)
    kp_c = g_models["ckd"](s)
    yaw, pitch, roll, t, delta = g_models["hpe_ede"](s)
    kp_s, Rs = transform_kp(kp_c, yaw, pitch, roll, t, delta)

    num_frames = 1
    img = img_as_float32(io.imread(driving))
    img = cv2.resize(img, (256, 256))
    # img = drivin
    # img = img.resize((256, 256))
    img = np.array(img, dtype="float32").transpose((2, 0, 1))
    img = torch.from_numpy(img).to(device).unsqueeze(0)
    yaw, pitch, roll, t, delta = g_models["hpe_ede"](img)
    kp_d, Rd = transform_kp(kp_c, yaw, pitch, roll, t, delta)
    deformation, occlusion = g_models["mfe"](fs, kp_s, kp_d, Rs, Rd)
    generated_d = g_models["generator"](fs, deformation, occlusion)
    # generated_d = torch.cat((img, generated_d), dim=3)
    generated_d = generated_d.squeeze(0).data.cpu().numpy()
    generated_d = np.transpose(generated_d, [1, 2, 0])
    generated_d = generated_d.clip(0, 1)
    generated_d = (255 * generated_d).astype(np.uint8)
    # imageio.imwrite(output, generated_d)
    return(generated_d)

###########################################################
##########################INVERSION########################
def run_on_batch(inputs, net):
    result_batch = net(inputs.to("cuda").float(), randomize_noise=False, resize=False)
    return result_batch


def offset_crop(img, new_width, new_height, offset):
    width, height = img.size   # Get dimensions

    left = (width - new_width)/2
    top = (height - new_height)/2 + offset
    right = (width + new_width)/2
    bottom = (height + new_height)/2 + offset

    # Crop the center of the image
    img = img.crop((left, top, right, bottom))
    return img

e_ckpt = torch.load('models/encoder_ffhq.pt', map_location=lambda storage, loc: storage)
encoder = Encoder(256, 512).to(device)

encoder.load_state_dict(e_ckpt["e"])

g_ckpt = torch.load('models/generator_ffhq.pt', map_location=lambda storage, loc: storage)
g_args = g_ckpt['args']
size = g_args.size
latent = g_args.latent
n_mlp = g_args.n_mlp
channel_multiplier = g_args.channel_multiplier


generator = Generator_style(size, latent, n_mlp, channel_multiplier=channel_multiplier).to(device)
generator.load_state_dict(g_ckpt["g_ema"])
trunc = generator.mean_latent(4096).detach()
trunc.requires_grad = False 


###########################################################
#########################SEGMENTATION######################
vgg_model = torch.load('models/vgg_normalized.pth')

segmentation_model = G(vgg_model).to(device)
segmentation_model.load_state_dict(torch.load('models/encoder_13.pt', map_location=device))
segmentation_model.eval()
# model = nn.DataParallel(model)
segmentation_model.to(device)


###########################################################
#########################TEXTURE###########################
# texture_model = CustomSPADE(num_classes=20)
# texture_model.load_state_dict(torch.load('models/usable.pt', map_location=device))
# texture_model.eval()
# # texture_model = nn.DataParallel(texture_model)

# texture_model.to(device)


###########################################################
####################TEXTURE ADAIN##########################
texture_model = G_tex(vgg_model)
texture_model.load_state_dict(torch.load('models/encoder_10.pt'))
texture_model.eval()
texture_model = texture_model.cuda()


###########################################################
#####################DIRECTORIES###########################
skin_dir = 'segmentation data/skin'
hair_dir = 'segmentation data/hair'
mouth_dir = 'segmentation data/mouth'

skin_tex_dir = 'textured data/skin'
hair_tex_dir = 'textured data/hair'
l_eye_tex_dir = 'textured data/l_eye'
r_eye_tex_dir = 'textured data/r_eye'
mouth_tex_dir = 'textured data/mouth'
real_tex_dir = 'textured data/real'
full_dir = 'textured data/full'

light_folder='light_data/example_light/'
model_folder='trained_model/'

###########################################################
#######################FUNCTIONS###########################
def create_folders(i):
    new_folder_cropped = os.path.join('cropped',str(i))

    if not os.path.exists(new_folder_cropped):
        os.makedirs(new_folder_cropped)


    new_folder_age = os.path.join('age',str(i))

    if not os.path.exists(new_folder_age):
        os.makedirs(new_folder_age)
    return new_folder_age, new_folder_cropped



def get_segmentation_map(i = 0, imgs = None, mode = 'auto', skin_path = '', hair_path = '', mouth_path = ''):
    if mode == 'auto':
        skin_path = os.path.join(skin_dir, imgs)

        hair_dir_list = random.choice(os.listdir(hair_dir))
        hair_path = os.path.join(hair_dir, hair_dir_list)


        mouth_dir_list = random.choice(os.listdir(mouth_dir))
        mouth_path = os.path.join(mouth_dir, mouth_dir_list)
    elif mode == 'manual':
        skin_path = skin_path
        hair_path = hair_path
        mouth_path = mouth_path
    # Pre-process the input images
    skin = preprocess(skin_path, 256).to(device)
    hair = preprocess(hair_path, 256).to(device)
    mouth = preprocess(mouth_path, 256).to(device)
    imgsfile = []
    imgsfile.append(skin)
    imgsfile.append(hair)
    imgsfile.append(mouth)

    with torch.no_grad():
        output_image = segmentation_model(imgsfile[0], imgsfile[1:])
        new_folder = os.path.join('segmentations')
        if not os.path.exists(new_folder):
            os.makedirs(new_folder)
    if mode == 'auto':
        seg_string = os.path.join(new_folder, str(i) + '.jpg')
    if mode =='manual':
        seg_string = 'segmentation.jpg'
    vutils.save_image(output_image.data, seg_string, normalize = True)
    return seg_string

def infuse_textures(seg_string = '', mode = 'auto', skin_tex_path = '', hair_tex_path = '', l_eye_tex_path = '',r_eye_tex_path = '', mouth_tex_path= '', seg_map_path = ''):
    if mode == 'auto':
        

        skin_dir_list = random.choice(os.listdir(skin_tex_dir))
        skin_tex_path = os.path.join(skin_tex_dir, skin_dir_list)

        l_eye_dir_list = random.choice(os.listdir(l_eye_tex_dir))
        l_eye_tex_path = os.path.join(l_eye_tex_dir, l_eye_dir_list)
        
        r_eye_tex_path = l_eye_tex_path.replace("l_eye", "r_eye")

        hair_dir_list = random.choice(os.listdir(hair_tex_dir))
        hair_tex_path = os.path.join(hair_tex_dir, hair_dir_list)

        mouth_dir_list = random.choice(os.listdir(mouth_tex_dir))
        mouth_tex_path = os.path.join(mouth_tex_dir, mouth_dir_list)

        seg_dir_list = random.choice(os.listdir(full_dir))
        seg_map_path = os.path.join(full_dir, seg_dir_list)
        # seg_map_path = seg_string
        
    elif mode == 'manual':
        skin_tex_path = skin_tex_path
        hair_tex_path = hair_tex_path
        l_eye_tex_path = l_eye_tex_path
        r_eye_tex_path = r_eye_tex_path
        mouth_tex_path = mouth_tex_path
        seg_map_path = seg_string

    imgs = []
    seg_map = preprocess(seg_map_path, 256).to(device)
    skin = preprocess(skin_tex_path, 256).to(device)
    imgs.append(skin)

    hair = preprocess(hair_tex_path, 256).to(device)
    imgs.append(hair)

    left_eye = preprocess(l_eye_tex_path, 256).to(device)
    imgs.append(left_eye)

    right_eye = preprocess(r_eye_tex_path, 256).to(device)
    imgs.append(right_eye)

    mouth = preprocess(mouth_tex_path, 256).to(device)
    imgs.append(mouth)


    with torch.no_grad():
        textured_image = texture_model(seg_map, imgs)

    min_val = textured_image.min()
    max_val = textured_image.max()
    textured_image = (textured_image - min_val) / (max_val - min_val)
    textured_image = textured_image.squeeze()
    return seg_map, imgs, TF.to_pil_image(textured_image)

def change_mouth_or_hair(seg_map, imgs):

    random_number = random.randint(1, 3)
    imgs = imgs
    if random_number==1:
        mouth_dir_list = random.choice(os.listdir(mouth_tex_dir))
        mouth_tex_path = os.path.join(mouth_tex_dir, mouth_dir_list)
        mouth = preprocess(mouth_tex_path, 256).to(device)
        imgs[4] = mouth
    if random_number==2:
        hair_dir_list = random.choice(os.listdir(hair_tex_dir))
        hair_tex_path = os.path.join(hair_tex_dir, hair_dir_list)
        hair = preprocess(hair_tex_path, 256).to(device)
        imgs[1] = hair
    if random_number==3:
        mouth_dir_list = random.choice(os.listdir(mouth_tex_dir))
        mouth_tex_path = os.path.join(mouth_tex_dir, mouth_dir_list)
        mouth = preprocess(mouth_tex_path, 256).to(device)
        imgs[4] = mouth
        hair_dir_list = random.choice(os.listdir(hair_tex_dir))
        hair_tex_path = os.path.join(hair_tex_dir, hair_dir_list)
        hair = preprocess(hair_tex_path, 256).to(device)
        imgs[1] = hair
    with torch.no_grad():
        textured_image = texture_model(seg_map, imgs)

    min_val = textured_image.min()
    max_val = textured_image.max()
    textured_image = (textured_image - min_val) / (max_val - min_val)
    textured_image = textured_image.squeeze()
    return TF.to_pil_image(textured_image)


def celeba():
    for i, full in enumerate(os.listdir(full_dir)):
        seg_map_path = os.path.join(full_dir, full)

        skin_tex_path = seg_map_path.replace("full", "skin")
        hair_tex_path = seg_map_path.replace("full", "hair")
        l_eye_tex_path = seg_map_path.replace("full", "l_eye")
        r_eye_tex_path = seg_map_path.replace("full", "r_eye")
        mouth_tex_path = seg_map_path.replace("full", "mouth")


        imgs = []
        seg_map = preprocess(seg_map_path, 256).to(device)
        skin = preprocess(skin_tex_path, 256).to(device)
        imgs.append(skin)

        hair = preprocess(hair_tex_path, 256).to(device)
        imgs.append(hair)

        left_eye = preprocess(l_eye_tex_path, 256).to(device)
        imgs.append(left_eye)

        right_eye = preprocess(r_eye_tex_path, 256).to(device)
        imgs.append(right_eye)

        mouth = preprocess(mouth_tex_path, 256).to(device)
        imgs.append(mouth)


        with torch.no_grad():
            textured_image = texture_model(seg_map, imgs)
        min_val = textured_image.min()
        max_val = textured_image.max()
        textured_image = (textured_image - min_val) / (max_val - min_val)
        textured_image = textured_image.squeeze()
        textured_image =  TF.to_pil_image(textured_image)


        image = enhance(textured_image)
        os.makedirs('celeba_syn', exist_ok=True)
        new = full.split('_full.png')
        string = 'celeba_syn/' + str(int(new[0])) + '.jpg'
        image.save(string)
        print(i)



    

    

def inversion(image, truncation):
    image = transform(image).unsqueeze(0).to(device)
            
    latents = encoder(image)

    fake, _ = generator([latents],
                input_is_latent=True,
                truncation=truncation,
                truncation_latent=trunc,
                randomize_noise=False)


    min_val = fake.min()
    max_val = fake.max()
    fake = (fake - min_val) / (max_val - min_val)

    fake = fake.squeeze()
    return TF.to_pil_image(fake)


def relighting(image, mode = 'auto', loc = 0):

    model_light = load_model(model_folder='models')

    normal, valid = get_normal()

    img = image.resize((512, 512))

    img = np.array(img)
    row, col, _ = img.shape
    Lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    inputL = Lab[:,:,0]
    inputL = inputL.astype(np.float32)/255.0
    inputL = inputL.transpose((0,1))
    inputL = inputL[None,None,...]
    inputL = Variable(torch.from_numpy(inputL).cuda())
    if mode == 'auto':
        light = random.randint(0, 6)
    elif mode == 'manual':
        light = loc

    base_sh_coeff = np.loadtxt(os.path.join(light_folder, 'rotate_light_{:02d}.txt'.format(light)))
    base_sh_coeff = base_sh_coeff[0:9]

    # Generate some random perturbations
    perturbations = np.random.normal(scale=0.05, size=base_sh_coeff.shape)

    # Make perturbations smaller for higher frequency coefficients
    weights = np.linspace(1.0, 0.1, num=len(base_sh_coeff))
    perturbations *= weights

    # Apply perturbations to base coefficients
    sh = base_sh_coeff + perturbations

    sh = sh * 0.7
    shading = get_shading(normal, sh)
    value = np.percentile(shading, 95)
    ind = shading > value
    shading[ind] = value
    shading = (shading - np.min(shading))/(np.max(shading) - np.min(shading))
    shading = (shading *255.0).astype(np.uint8)
    shading = np.reshape(shading, (256, 256))
    shading = shading * valid

    sh = np.reshape(sh, (1,9,1,1)).astype(np.float32)
    sh = Variable(torch.from_numpy(sh).cuda())
    outputImg, outputSH  = model_light(inputL, sh, 0)
    outputImg = outputImg[0].cpu().data.numpy()
    outputImg = outputImg.transpose((1,2,0))
    outputImg = np.squeeze(outputImg)
    outputImg = (outputImg*255.0).astype(np.uint8)
    Lab[:,:,0] = outputImg
    resultLab = cv2.cvtColor(Lab, cv2.COLOR_LAB2BGR)
    resultLab = cv2.resize(resultLab, (col, row))
    return Image.fromarray(resultLab)

    
def pose_transfer(image, mode = 'auto', driving = ''):
    if mode == 'auto':
        driving = os.path.join('textured data/real',random.choice(os.listdir('textured data/real')))
    elif mode == 'manual':
        driving = driving
    return eval(output = '', source = image,driving = driving)


def age_changer(image, mode = 'auto', age = 0):
    if mode == 'auto':
        target_ages = [ 20, 30, 40, 50, 60]

        age = random.choice(target_ages)
    elif mode == 'manual':
        age = age

    age_transformer = AgeTransformer(target_age=age)

   
    with torch.no_grad():
        input_image_age = [age_transformer(image.cpu()).to('cuda')]
        input_image_age = torch.stack(input_image_age)
        result_tensor = run_on_batch(input_image_age, net)[0]
        return tensor2im(result_tensor)


def save_and_crop(image, uncropped, cropped):

    image.save(uncropped)

    img = image.resize((256,256))

    img = offset_crop(img, 170, 170, 20)

    img.save(cropped)



#CUDA_VISIBLE_DEVICES=0 python generate.py
#CUDA_VISIBLE_DEVICES=1 python generate.py
#CUDA_VISIBLE_DEVICES=2 python generate.py
#CUDA_VISIBLE_DEVICES=3 python generate.py

cuda_mode = 3
repeat = 1
#Automated generation example
def generate():
    mode = 'auto'
    for j, imgs in enumerate(os.listdir('segmentation data/skin')):
        if j < 7500 * cuda_mode:
            pass
        elif j == 7500 * (cuda_mode + 1):
            break
        else:
            i = j + (30000 * repeat)
            uncropped_folder, cropped_folder = create_folders(i)
            seg_string = get_segmentation_map(i, imgs, mode = 'auto')
            seg_map, images, base_texture = infuse_textures(seg_string, mode = 'auto')
            # style_image = inversion(textured_image, 0.6)
            for p in range(100):
                random_number = random.randint(1, 2)
                if random_number == 1:
                    textured_image = base_texture
                elif random_number == 2:
                    textured_image = change_mouth_or_hair(seg_map, images)
                light_image = relighting(textured_image)
                new_pose = pose_transfer(light_image, mode = 'auto')
                image = Image.fromarray(new_pose.astype('uint8'), 'RGB')
                image = enhance(image)
                image = transform(image)
                new_age = age_changer(image, mode = 'auto')
                uncropped = os.path.join(uncropped_folder, str(p) + '.jpg')
                cropped = os.path.join(cropped_folder, str(p) + '.jpg')
                save_and_crop(new_age, uncropped, cropped)


##Manual generation example
def manual():
    mode = 'manual'
    skin_path  = 'segmentation data/skin/00001_skin.png.png'
    hair_path = 'segmentation data/hair/00005_hair.png'
    mouth_path = 'segmentation data/mouth/00002_mouth.png'
    seg_string = get_segmentation_map(mode = 'manual', skin_path=skin_path, hair_path=hair_path, mouth_path=mouth_path)
    
    skin_tex_path = 'textured data/skin/00001_skin.png'
    hair_tex_path = 'textured data/hair/07682_hair.png'
    l_eye_tex_path = 'textured data/l_eye/09872_l_eye.png'
    r_eye_tex_path = 'textured data/r_eye/02131_r_eye.png'
    mouth_tex_path = 'textured data/mouth/00981_mouth.png'
    seg_map_path = 'textured data/full/00001_full.png'
    #or use seg string from the previously generated segmentation map

    _,_,textured_image = infuse_textures(mode = 'manual',skin_tex_path=skin_tex_path, hair_tex_path=hair_tex_path, l_eye_tex_path=l_eye_tex_path, r_eye_tex_path=r_eye_tex_path, mouth_tex_path=mouth_tex_path,seg_string=seg_map_path )
    textured_image.save('test/textured1.jpg')
    # exit()

    # style_image = inversion(textured_image, 0.6)
    # style_image.save('test/style.jpg')

    light_image = relighting(textured_image, mode = 'manual', loc = 1)
    light_image.save('test/light1.jpg')

    driving = 'textured data/real/0.jpg'

    new_pose = pose_transfer(light_image, mode = 'manual', driving = driving)
    new_pose = Image.fromarray(new_pose.astype('uint8'), 'RGB')

    new_pose.save('test/pose1.jpg')


    enhanced = enhance(new_pose)
    enhanced.save('test/enhanced1.jpg')
    
    new_age = transform(enhanced)
    new_age = age_changer(new_age, mode = 'manual', age = 20)
    new_age.save('test/age1.jpg')

def lighting_change():
    mode = 'manual'
    skin_tex_path = 'textured data/skin/00001_skin.png'
    hair_tex_path = 'textured data/hair/00001_hair.png'
    l_eye_tex_path = 'textured data/l_eye/00001_l_eye.png'
    r_eye_tex_path = 'textured data/r_eye/00001_r_eye.png'
    mouth_tex_path = 'textured data/mouth/00001_mouth.png'
    seg_map_path = 'textured data/full/00001_full.png'
    #or use seg string from the previously generated segmentation map

    _,_,textured_image = infuse_textures(mode = 'manual',skin_tex_path=skin_tex_path, hair_tex_path=hair_tex_path, l_eye_tex_path=l_eye_tex_path, r_eye_tex_path=r_eye_tex_path, mouth_tex_path=mouth_tex_path,seg_string=seg_map_path )

    style_image = inversion(textured_image, 0.6)
    os.makedirs('slight_changes/light',exist_ok=True)

    for i in range(20):
        light_image = relighting(style_image, mode = 'auto')
        light_string = 'slight_changes/light/' + str(i) +'.jpg'
        light_image = enhance(light_image)
        light_image.save(light_string)

def pose_change():
    mode = 'manual'
    skin_tex_path = 'textured data/skin/00001_skin.png'
    hair_tex_path = 'textured data/hair/00001_hair.png'
    l_eye_tex_path = 'textured data/l_eye/00001_l_eye.png'
    r_eye_tex_path = 'textured data/r_eye/00001_r_eye.png'
    mouth_tex_path = 'textured data/mouth/00001_mouth.png'
    seg_map_path = 'textured data/full/00001_full.png'
    #or use seg string from the previously generated segmentation map

    _,_,textured_image = infuse_textures(mode = 'manual',skin_tex_path=skin_tex_path, hair_tex_path=hair_tex_path, l_eye_tex_path=l_eye_tex_path, r_eye_tex_path=r_eye_tex_path, mouth_tex_path=mouth_tex_path,seg_string=seg_map_path )

    style_image = inversion(textured_image, 0.6)
    os.makedirs('slight_changes/pose',exist_ok=True)

    for i in range(20):
        new_pose = pose_transfer(style_image, mode = 'auto')
        pose_image = Image.fromarray(new_pose.astype('uint8'), 'RGB')
        pose_string = 'slight_changes/pose/' + str(i) +'.jpg'
        pose_image = enhance(pose_image)
        pose_image.save(pose_string)

def age_change():
    mode = 'manual'
    skin_tex_path = 'textured data/skin/00001_skin.png'
    hair_tex_path = 'textured data/hair/00001_hair.png'
    l_eye_tex_path = 'textured data/l_eye/00001_l_eye.png'
    r_eye_tex_path = 'textured data/r_eye/00001_r_eye.png'
    mouth_tex_path = 'textured data/mouth/00001_mouth.png'
    seg_map_path = 'textured data/full/00001_full.png'
    #or use seg string from the previously generated segmentation map

    _,_,textured_image = infuse_textures(mode = 'manual',skin_tex_path=skin_tex_path, hair_tex_path=hair_tex_path, l_eye_tex_path=l_eye_tex_path, r_eye_tex_path=r_eye_tex_path, mouth_tex_path=mouth_tex_path,seg_string=seg_map_path )

    style_image = inversion(textured_image, 0.6)
    os.makedirs('slight_changes/age',exist_ok=True)
    style_image = transform(style_image)
    ages = [10, 20, 30,40,50,60,70,80]
    for i in ages:
        new_age = age_changer(style_image, mode = 'manual', age = i)
        age_string = 'slight_changes/age/' + str(i) +'.jpg'
        new_age.save(age_string)

def slight_changes():
    for i in range(100):
        _,_,textured_image = infuse_textures(mode = 'auto')
        style_image = inversion(textured_image, 0.6)

        os.makedirs('slight_changes/{}/age'.format(str(i)),exist_ok=True)
        os.makedirs('slight_changes/{}/pose'.format(str(i)),exist_ok=True)
        os.makedirs('slight_changes/{}/light'.format(str(i)),exist_ok=True)



        for j in range(10):
            new_pose = pose_transfer(style_image, mode = 'auto')
            pose_image = Image.fromarray(new_pose.astype('uint8'), 'RGB')
            pose_string = 'slight_changes/{}/pose/'.format(str(i)) + str(j) +'.jpg'
            pose_image = enhance(pose_image)
            pose_image.save(pose_string)
        print(i, ': Pose')

        for j in range(7):
            light_image = relighting(style_image, mode = 'manual', loc=j)
            light_string = 'slight_changes/{}/light/'.format(str(i)) + str(j) +'.jpg'
            light_image = enhance(light_image)
            light_image.save(light_string)
        print(i, ': Light')
            
        style_image = transform(style_image)
        ages = [10, 20, 30,40,50,60,70,80]
        for j in ages:
            new_age = age_changer(style_image, mode = 'manual', age = j)
            age_string = 'slight_changes/{}/age/'.format(str(i)) + str(j) +'.jpg'
            new_age.save(age_string)

        print(i, ': Age')



if __name__ == "__main__":
    # generate()
    # manual()
    # lighting_change()
    # pose_change()
    # age_change()
    # slight_changes()
    celeba()




