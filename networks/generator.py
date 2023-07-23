from matplotlib.cbook import flatten
import torch.nn as nn
import torch.nn.functional as F
import torch

mse_criterion = nn.MSELoss()
def calc_mean_std(feat, eps=1e-5):
	size = feat.size()
	assert(len(size)==4)
	N, C = size[:2]
	feat_var = feat.view(N, C, -1).var(dim=2) +eps
	feat_std = feat_var.sqrt().view(N, C, 1, 1)
	feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
	return feat_mean, feat_std

def calc_content_loss(input_im, target):
	assert (input_im.size() == target.size())
	assert (target.requires_grad is False)
	return mse_criterion(input_im, target)

def calc_style_loss(input_im, target):
	assert (input_im.size() == target.size())
	assert (target.requires_grad is False)
	input_mean, input_std = calc_mean_std(input_im)
	target_mean, target_std = calc_mean_std(target)
	return mse_criterion(input_mean, target_mean) + \
			mse_criterion(input_std, target_std)


vggnet = nn.Sequential(
			# encode 1-1
			nn.Conv2d(3, 3, kernel_size=(1,1), stride= (1, 1)),
			nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
			nn.ReLU(inplace=True), # relu 1-1
			# encode 2-1
			nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),

			nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
			nn.ReLU(inplace=True), # relu 2-1
			# encoder 3-1
			nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
			nn.ReLU(inplace=True),

			nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
			nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
			nn.ReLU(inplace=True), # relu 3-1
			# encoder 4-1
			nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
			nn.ReLU(inplace=True),
			nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
			nn.ReLU(inplace=True),
			nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),

			nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
			nn.ReLU(inplace=True), # relu 4-1
			# rest of vgg not used
			nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
			nn.ReLU(inplace=True),
			nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
			nn.ReLU(inplace=True),
			nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),

			nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
			nn.ReLU(inplace=True),
			nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
			nn.ReLU(inplace=True),
			nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
			nn.ReLU(inplace=True),
			nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
			nn.ReLU(inplace=True)
			)			


class AdaIN(nn.Module):
	def __init__(self):
		super().__init__()

	def forward(self, x, y):
		eps = 1e-5	
		mean_x = torch.mean(x, dim=[2,3])
		mean_y = torch.mean(y, dim=[2,3])

		std_x = torch.std(x, dim=[2,3])
		std_y = torch.std(y, dim=[2,3])

		mean_x = mean_x.unsqueeze(-1).unsqueeze(-1)
		mean_y = mean_y.unsqueeze(-1).unsqueeze(-1)

		std_x = std_x.unsqueeze(-1).unsqueeze(-1) + eps
		std_y = std_y.unsqueeze(-1).unsqueeze(-1) + eps

		out = (x - mean_x)/ std_x * std_y + mean_y


		return out



class G(nn.Module):
	def __init__(self, vgg_model):
		super().__init__()

		vggnet.load_state_dict(vgg_model)

		self.encodera = nn.Sequential(
			nn.Conv2d(3, 3, kernel_size=(1,1), stride= (1, 1)),
			nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
			nn.ReLU(inplace=True),

			nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),

			nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
			nn.ReLU(inplace=True),


			nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
			
			nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
			nn.ReLU(inplace=True),


			nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
			nn.ReLU(inplace=True),
			nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
			nn.ReLU(inplace=True),
			nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
			
			nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
			nn.ReLU(inplace=True)
			)

		self.encoderb = nn.Sequential(
			nn.Conv2d(3, 3, kernel_size=(1,1), stride= (1, 1)),
			nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
			nn.ReLU(inplace=True),

			nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),

			nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
			nn.ReLU(inplace=True),


			nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
			
			nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
			nn.ReLU(inplace=True),


			nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
			nn.ReLU(inplace=True),
			nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
			nn.ReLU(inplace=True),
			nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
			
			nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
			nn.ReLU(inplace=True)
			)

		

        # self.encoder.load_state_dict(vggnet[:21].state_dict(), strict=False)
        # for parameter in self.encoder.parameters():
        #     parameter.requires_grad = False
		
		self.encoder2 = nn.Sequential(
			nn.Conv2d(3, 3, kernel_size=(1,1), stride= (1, 1)),
			nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
			nn.ReLU(inplace=True),

			nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),

			nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
			nn.ReLU(inplace=True),


			nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
			
			nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
			nn.ReLU(inplace=True),


			nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
			nn.ReLU(inplace=True),
			nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
			nn.ReLU(inplace=True),
			nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
			
			nn.Conv2d(256, 512*2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
			nn.ReLU(inplace=True)
			)
            

        # self.encoder2.load_state_dict(vggnet[:21].state_dict(), strict=False)
        # for parameter in self.encoder2.parameters():
        #     parameter.requires_grad = False
		# self.pose_estimator1 = nn.Sequential(
		# 	nn.Linear(3, 128),
		# 	nn.Linear(128, 256),
		# 	nn.Linear(256, 512),
		# 	nn.Linear(512, 1024),
		# 	nn.Linear(1024, 2048),
		# 	# nn.Linear(2048, 32768),
			
		# 	# nn.Linear(2048, 6144)



			
		# 	# nn.Linear(2048, 6144)


		# )
		self.decoder = nn.Sequential(
			nn.Conv2d(512*2, 2048, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
			nn.ReLU(inplace=True),

			nn.Upsample(scale_factor=2, mode='nearest'),
			nn.Conv2d(2048, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
			nn.ReLU(inplace=True),
			nn.Conv2d(1024, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
			nn.ReLU(inplace=True),
			nn.Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
			nn.ReLU(inplace=True),
			nn.Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
			nn.ReLU(inplace=True),

			nn.Upsample(scale_factor=2, mode='nearest'),
			nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
			nn.ReLU(inplace=True),
			nn.Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
			nn.ReLU(inplace=True),

			nn.Upsample(scale_factor=2, mode='nearest'),
			nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
			nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
            nn.Tanh()            
            )
		self.adaIN = AdaIN()
		self.mse_criterion = nn.MSELoss()
		

	def forward(self,  x, y, alpha=1.0):
		content_img = x
        # style_img = y
		# pose_out = self.pose_estimator1(z)

		

		encode_style = torch.cat((self.encodera(y[0]),self.encoderb(y[1])), 1)
		encode_content = self.encoder2(content_img)
		encode_out = self.adaIN(encode_content, encode_style)
		# pose_out = pose_out.view(1, 2048, 1, 1)
		# p32d = (1,30,2,29)
		# pose_out = F.pad(pose_out, p32d, "constant", 0)

		# gen_img = self.decoder(encode_out)
        # encode_gen = self.encoder(gen_img)

        # fm11_style = self.encoder[:3](style_img)
        # fm11_gen = self.encoder[:3](gen_img)

        # fm21_style = self.encoder[3:8](fm11_style)
        # fm21_gen = self.encoder[3:8](fm11_gen)

        # fm31_style = self.encoder[8:13](fm21_style)
        # fm31_gen = self.encoder[8:13](fm21_gen)
        
        # loss_content = self.mse_criterion(encode_gen, encode_out)

        # loss_style = self.mse_criterion(torch.mean(fm11_gen, dim=[2,3]), torch.mean(fm11_style, dim=[2,3])) +	\
        #             self.mse_criterion(torch.mean(fm21_gen, dim=[2,3]), torch.mean(fm21_style, dim=[2,3])) +	\
        #             self.mse_criterion(torch.mean(fm31_gen, dim=[2,3]), torch.mean(fm31_style, dim=[2,3])) +	\
        #             self.mse_criterion(torch.mean(encode_gen, dim=[2,3]), torch.mean(encode_style, dim=[2,3])) +	\
        #             self.mse_criterion(torch.std(fm11_gen, dim=[2,3]), torch.std(fm11_style, dim=[2,3])) +	\
        #             self.mse_criterion(torch.std(fm21_gen, dim=[2,3]), torch.std(fm21_style, dim=[2,3])) +	\
        #             self.mse_criterion(torch.std(fm31_gen, dim=[2,3]), torch.std(fm31_style, dim=[2,3])) +	\
        #             self.mse_criterion(torch.std(encode_gen, dim=[2,3]), torch.std(encode_style, dim=[2,3])) 

        # return loss_content, loss_style
		encode_out = alpha * encode_out + (1-alpha)* encode_content 
		# encode_out= torch.cat((encode_out, pose_out), dim = 1)
		gen_img = self.decoder(encode_out)
		return gen_img

# z_dim = 32
# h_dim = 1024

# class Flatten(nn.Module):
#     def forward(self, input):
#         return input.view(input.size(0), -1)


# class UnFlatten(nn.Module):
#     def forward(self, input, size=1024):
#         return input.view(input.size(0), size, 1, 1)



# class Interpolate(nn.Module):
#     def __init__(self, scale_factor, mode):
#         super(Interpolate, self).__init__()
#         self.interp = nn.functional.interpolate
#         self.scale_factor = scale_factor
#         self.mode = mode
#     def forward(self, x):
#         x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=False)
#         return x
# class G(nn.Module):
#     def __init__(self):
#         super(G, self).__init__()



#         self.main = nn.Sequential(
            
#                 # UnFlatten(),
#                 nn.Conv2d(in_channels=128*6, out_channels=2048, kernel_size=(5,5), stride=1, bias=True, padding = 2), ##1x1
#                 nn.BatchNorm2d(2048),
#                 nn.ReLU(True),
#                 Interpolate(scale_factor=2, mode='bilinear'),
#                 nn.Conv2d(in_channels=2048, out_channels=2048, kernel_size=(5,5), stride=1, bias=True, padding=2), ##2x2
#                 nn.BatchNorm2d(2048),
#                 nn.ReLU(True),
#                 Interpolate(scale_factor=2, mode='bilinear'),
#                 nn.Conv2d(in_channels=2048, out_channels=2048, kernel_size=(5,5), stride=1, bias=True, padding=2, padding_mode='reflect'), ##4x4
#                 nn.BatchNorm2d(2048),
#                 nn.ReLU(True),
#                 Interpolate(scale_factor=2, mode='bilinear'),
#                 nn.Conv2d(in_channels=2048, out_channels=2048,kernel_size=(5,5), stride=1, bias=True, padding=2, padding_mode='reflect'), ## 8x8
#                 nn.BatchNorm2d(2048),
#                 nn.ReLU(True),
#                 Interpolate(scale_factor=2, mode='bilinear'),
#                 nn.Conv2d(in_channels=2048, out_channels=1024, kernel_size=(5,5), stride=1, bias=True, padding=2, padding_mode='reflect'), ## 16x16
#                 nn.BatchNorm2d(1024),
#                 nn.ReLU(True),
#                 Interpolate(scale_factor=2, mode='bilinear'),
#                 nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=(5,5), stride=1, bias=True, padding=2, padding_mode='reflect'), ## 32x32x
# 				nn.BatchNorm2d(512),
#                 nn.ReLU(True),
#                 Interpolate(scale_factor=2, mode='bilinear'),
#                 nn.Conv2d(in_channels=512, out_channels=256,  kernel_size=(5,5), stride=1, bias=True, padding=2, padding_mode='reflect'), ## 64x64
# 				nn.BatchNorm2d(256),
#                 nn.ReLU(True),	
#                 Interpolate(scale_factor=2, mode='bilinear'),
#                 nn.Conv2d(in_channels=256, out_channels=128,kernel_size=(5,5), stride=1, bias=True, padding=2, padding_mode='reflect'), ## 128x128
# 				nn.BatchNorm2d(128),
#                 nn.ReLU(True),		
#                 Interpolate(scale_factor=2, mode='bilinear'),
#                 nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(5,5), stride=1, bias=True, padding=2, padding_mode='reflect'), ## 256x256
# 				nn.BatchNorm2d(64),
#                 nn.ReLU(True),
# 				nn.Conv2d(in_channels=64, out_channels=3, kernel_size=(5,5), stride=1 , bias=True, padding = 2, padding_mode='reflect'),
#                 # nn.Tanh()
#                 )
#         self.adaIN = AdaIN()
#         self.mse_criterion = nn.MSELoss()
#     def forward(self, x, y, alpha = 1.0):
#         encode_out = self.adaIN(x, y)
#         encode_out = alpha * encode_out + (1-alpha)* x 
#         gen_img = self.main(encode_out)
#         return gen_img


# class G_enc(nn.Module):
#     def __init__(self):
#         super(G_enc, self).__init__()
#         z_dim = 32
#         h_dim = 1024
#         self.main = nn.Sequential(
#             nn.Conv2d(3, 64, 5 ,1,2),
#             nn.ReLU(True),
#             nn.MaxPool2d(kernel_size = 2),
#             nn.Conv2d(64, 128, 5,1,2),
#             nn.ReLU(True),
#             nn.MaxPool2d(kernel_size = 2),
#             nn.Conv2d(128, 256, 5,1,2),
#             nn.ReLU(True),
#             nn.MaxPool2d(kernel_size = 2),
#             nn.Conv2d(256, 512, 5,1,2),
#             nn.ReLU(True),
#             nn.MaxPool2d(kernel_size = 2),
#             nn.Conv2d(512, 1024, 5,1,2),
#             nn.ReLU(True),
#             # nn.MaxPool2d(kernel_size = 2),
#             # nn.Conv2d(1024, 1024, 5,1,2),
#             # nn.ReLU(True),
#             nn.MaxPool2d(kernel_size = 2),
#             nn.Conv2d(1024, 2048, 5,1,2),
#             # nn.ReLU(True),
#             # nn.MaxPool2d(kernel_size = 2),
#             # nn.Conv2d(2048, 2048, 5,1,2),
#             # nn.ReLU(True),
#             # nn.MaxPool2d(kernel_size = 2),
#             # nn.Conv2d(2048, 2048, 5,1,2),
#             # nn.ReLU(True),
#             # nn.MaxPool2d(kernel_size = 2),
#             # nn.BatchNorm2d(256),
#             # nn.BatchNorm2d(256),
#             # nn.ReLU(True),
#             # nn.Conv2d(256,512, 3,1,1),
#             # nn.BatchNorm2d(512),
#             # nn.ReLU(True),
#             # nn.Conv2d(512, 1024, 3,1,1),
#             # nn.BatchNorm2d(1024),
#             # nn.ReLU(True)
#             # ,nn.Conv2d(1024, 2048, 3,1,0),
#             # nn.BatchNorm2d(2048),
#             # nn.ReLU(True),
#             # Flatten()
#             # nn.Linear(8*64*32*32, 64)
#             # nn.Conv2d(128, 128, 3,1),
#             # nn.BatchNorm2d(128),
#             # nn.ReLU(True),
#                 )
#         self.fc = nn.Sequential(
#             nn.Linear(640, 64),
#             nn.Linear(64, 128),
#             nn.Linear(128, 256),
#             nn.Linear(256, 512),
#             nn.Linear(512, 1024)

#         )
# #         self.fc1 = nn.Linear(h_dim, z_dim)
# #         self.fc2 = nn.Linear(h_dim, z_dim)
# #         self.fc3 = nn.Linear(z_dim, h_dim)


#     def forward(self, x):
#     # h = self.encoder(a)
#         # x = torch.cat((a,b,c,d,e), 1)
#         x = self.main(x)
#         # x = self.fc(x)
#         # a = self.fc(self.main(a).reshape(4,-1))
#         # b = self.fc(self.main(b).reshape(4,-1))
#         # c = self.fc(self.main(c).reshape(4,-1))
#         # d = self.fc(self.main(d).reshape(4,-1))
#         # e = self.fc(self.main(e).reshape(4,-1))
#         return x

# from .unet_parts import *


# class G(nn.Module):
#     def __init__(self, bilinear=True):
#         super(G, self).__init__()
#         self.bilinear = bilinear

#         self.inc = DoubleConv(18, 64)
#         self.down1 = Down(64, 128)
#         self.down2 = Down(128, 256)
#         self.down3 = Down(256, 512)
#         factor = 2 if bilinear else 1
#         self.down4 = Down(512, 1024 //factor)
#         # self.down5 = Down(1024, 2048)
#         # self.down6 = Down(2048, 4096)
#         # self.down7 = Down(4096, 8192 // factor)
#         # self.up1 = Up(8192, 4096 // factor, bilinear)
#         # self.up2 = Up(4096, 2048 // factor, bilinear)
#         # self.up3 = Up(2048, 1024 // factor, bilinear)
#         self.up4 = Up(1024, 512 // factor, bilinear)
#         self.up5 = Up(512, 256 // factor, bilinear)
#         self.up6 = Up(256, 128 // factor, bilinear)
#         self.up7 = Up(128, 64, bilinear)
#         self.outc = OutConv(64, 3)

#     def forward(self, x):
#         x1 = self.inc(x)
#         x2 = self.down1(x1)
#         x3 = self.down2(x2)
#         x4 = self.down3(x3)
#         x5 = self.down4(x4)
#         # x6 = self.down5(x5)
#         # x7 = self.down6(x6)
#         # x8 = self.down7(x7)
#         # x9 = self.down7(x8)


#         # x = self.up1(x8, x7)
#         # x = self.up2(x, x6)
#         # x = self.up3(x, x5)
#         x = self.up4(x5, x4)
#         x = self.up5(x, x3)
#         x = self.up6(x, x2)
#         x = self.up7(x, x1)

#         logits = self.outc(x)
#         return logits