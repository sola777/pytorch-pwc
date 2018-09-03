#!/usr/bin/env python

import getopt
import math
import numpy
import os
import PIL
import PIL.Image
import sys
import torch
import torch.utils.serialization
import cv2

# from correlation import correlation # the custom cost volume layer
from correlation import correlation

##########################################################

assert(int(torch.__version__.replace('.', '')) >= 40) # requires at least pytorch version 0.4.0

torch.set_grad_enabled(False) # make sure to not compute gradients for computational performance

# torch.cuda.device(1) # change this if you have a multiple graphics cards and you want to utilize them

torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance

##########################################################

arguments_strModel = 'sintel'
arguments_strFirst = './images/first.png'
arguments_strSecond = './images/second.png'
arguments_strOut = './result.flo'

for strOption, strArgument in getopt.getopt(sys.argv[1:], '', [ strParameter[2:] + '=' for strParameter in sys.argv[1::2] ])[0]:
	if strOption == '--model':
		arguments_strModel = strArgument # which model to use, see below

	elif strOption == '--first':
		arguments_strFirst = strArgument # path to the first frame

	elif strOption == '--second':
		arguments_strSecond = strArgument # path to the second frame

	elif strOption == '--out':
		arguments_strOut = strArgument # path to where the output should be stored

	# end
# end

##########################################################


class Network(torch.nn.Module):
	def __init__(self):
		super(Network, self).__init__()

		class Extractor(torch.nn.Module):
			def __init__(self):
				super(Extractor, self).__init__()

				self.moduleOne = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.moduleTwo = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.moduleThr = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.moduleFou = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=64, out_channels=96, kernel_size=3, stride=2, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.moduleFiv = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=96, out_channels=128, kernel_size=3, stride=2, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.moduleSix = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=128, out_channels=196, kernel_size=3, stride=2, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)
			# end

			def forward(self, tensorInput):
				tensorOne = self.moduleOne(tensorInput)
				tensorTwo = self.moduleTwo(tensorOne)
				tensorThr = self.moduleThr(tensorTwo)
				tensorFou = self.moduleFou(tensorThr)
				tensorFiv = self.moduleFiv(tensorFou)
				tensorSix = self.moduleSix(tensorFiv)

				return [ tensorOne, tensorTwo, tensorThr, tensorFou, tensorFiv, tensorSix ]
			# end
		# end

		class Backward(torch.nn.Module):
			def __init__(self):
				super(Backward, self).__init__()
			# end

			def forward(self, tensorInput, tensorFlow):
				if hasattr(self, 'tensorPartial') == False or self.tensorPartial.size(0) != tensorFlow.size(0) or self.tensorPartial.size(2) != tensorFlow.size(2) or self.tensorPartial.size(3) != tensorFlow.size(3):
					self.tensorPartial = torch.FloatTensor().resize_(tensorFlow.size(0), 1, tensorFlow.size(2), tensorFlow.size(3)).fill_(1.0).cuda()
				# end

				if hasattr(self, 'tensorGrid') == False or self.tensorGrid.size(0) != tensorFlow.size(0) or self.tensorGrid.size(2) != tensorFlow.size(2) or self.tensorGrid.size(3) != tensorFlow.size(3):
					torchHorizontal = torch.linspace(-1.0, 1.0, tensorFlow.size(3)).view(1, 1, 1, tensorFlow.size(3)).expand(tensorFlow.size(0), 1, tensorFlow.size(2), tensorFlow.size(3))
					torchVertical = torch.linspace(-1.0, 1.0, tensorFlow.size(2)).view(1, 1, tensorFlow.size(2), 1).expand(tensorFlow.size(0), 1, tensorFlow.size(2), tensorFlow.size(3))

					self.tensorGrid = torch.cat([ torchHorizontal, torchVertical ], 1).cuda()
				# end

				tensorInput = torch.cat([ tensorInput, self.tensorPartial ], 1)
				tensorFlow = torch.cat([ tensorFlow[:, 0:1, :, :] / ((tensorInput.size(3) - 1.0) / 2.0), tensorFlow[:, 1:2, :, :] / ((tensorInput.size(2) - 1.0) / 2.0) ], 1)

				tensorOutput = torch.nn.functional.grid_sample(input=tensorInput, grid=(self.tensorGrid + tensorFlow).permute(0, 2, 3, 1), mode='bilinear', padding_mode='zeros')

				tensorMask = tensorOutput[:, -1:, :, :]; tensorMask[tensorMask > 0.999] = 1.0; tensorMask[tensorMask < 1.0] = 0.0

				return tensorOutput[:, :-1, :, :] * tensorMask
			# end
		# end

		class Decoder(torch.nn.Module):
			def __init__(self, intLevel):
				super(Decoder, self).__init__()

				intPrevious = [ None, None, 81 + 32 + 2 + 2, 81 + 64 + 2 + 2, 81 + 96 + 2 + 2, 81 + 128 + 2 + 2, 81, None ][intLevel + 1]
				intCurrent = [ None, None, 81 + 32 + 2 + 2, 81 + 64 + 2 + 2, 81 + 96 + 2 + 2, 81 + 128 + 2 + 2, 81, None ][intLevel + 0]

				if intLevel < 6: self.moduleUpflow = torch.nn.ConvTranspose2d(in_channels=2, out_channels=2, kernel_size=4, stride=2, padding=1)
				if intLevel < 6: self.moduleUpfeat = torch.nn.ConvTranspose2d(in_channels=intPrevious + 128 + 128 + 96 + 64 + 32, out_channels=2, kernel_size=4, stride=2, padding=1)

				if intLevel < 6: self.dblBackward = [ None, None, None, 5.0, 2.5, 1.25, 0.625, None ][intLevel + 1]
				if intLevel < 6: self.moduleBackward = Backward()

				self.moduleCorrelation = correlation.ModuleCorrelation()
				self.moduleCorreleaky = torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)

				self.moduleOne = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=intCurrent, out_channels=128, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.moduleTwo = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=intCurrent + 128, out_channels=128, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.moduleThr = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=intCurrent + 128 + 128, out_channels=96, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.moduleFou = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=intCurrent + 128 + 128 + 96, out_channels=64, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.moduleFiv = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=intCurrent + 128 + 128 + 96 + 64, out_channels=32, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.moduleSix = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=intCurrent + 128 + 128 + 96 + 64 + 32, out_channels=2, kernel_size=3, stride=1, padding=1)
				)
			# end

			def forward(self, tensorFirst, tensorSecond, objectPrevious):
				tensorFlow = None
				tensorFeat = None

				if objectPrevious is None:
					tensorFlow = None
					tensorFeat = None

					tensorVolume = self.moduleCorreleaky(self.moduleCorrelation(tensorFirst, tensorSecond))

					tensorFeat = torch.cat([ tensorVolume ], 1)

				elif objectPrevious is not None:
					tensorFlow = self.moduleUpflow(objectPrevious['tensorFlow'])
					tensorFeat = self.moduleUpfeat(objectPrevious['tensorFeat'])

					tensorVolume = self.moduleCorreleaky(self.moduleCorrelation(tensorFirst, self.moduleBackward(tensorSecond, tensorFlow * self.dblBackward)))

					tensorFeat = torch.cat([ tensorVolume, tensorFirst, tensorFlow, tensorFeat ], 1)

				# end

				tensorFeat = torch.cat([ self.moduleOne(tensorFeat), tensorFeat ], 1)
				tensorFeat = torch.cat([ self.moduleTwo(tensorFeat), tensorFeat ], 1)
				tensorFeat = torch.cat([ self.moduleThr(tensorFeat), tensorFeat ], 1)
				tensorFeat = torch.cat([ self.moduleFou(tensorFeat), tensorFeat ], 1)
				tensorFeat = torch.cat([ self.moduleFiv(tensorFeat), tensorFeat ], 1)

				tensorFlow = self.moduleSix(tensorFeat)

				return {
					'tensorFlow': tensorFlow,
					'tensorFeat': tensorFeat
				}
			# end
		# end

		class Refiner(torch.nn.Module):
			def __init__(self):
				super(Refiner, self).__init__()

				self.moduleMain = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=81 + 32 + 2 + 2 + 128 + 128 + 96 + 64 + 32, out_channels=128, kernel_size=3, stride=1, padding=1,  dilation=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=2,  dilation=2),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=4,  dilation=4),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=128, out_channels=96, kernel_size=3, stride=1, padding=8,  dilation=8),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=96, out_channels=64, kernel_size=3, stride=1, padding=16,  dilation=16),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1,  dilation=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=32, out_channels=2, kernel_size=3, stride=1, padding=1,  dilation=1)
				)
			# end

			def forward(self, tensorInput):
				return self.moduleMain(tensorInput)
			# end
		# end

		self.moduleExtractor = Extractor()

		self.moduleTwo = Decoder(2)
		self.moduleThr = Decoder(3)
		self.moduleFou = Decoder(4)
		self.moduleFiv = Decoder(5)
		self.moduleSix = Decoder(6)

		self.moduleRefiner = Refiner()

		self.load_state_dict(torch.load('./models/' + arguments_strModel + '.pytorch'))
	# end

	def forward(self, tensorFirst, tensorSecond):
		tensorFirst = self.moduleExtractor(tensorFirst)
		tensorSecond = self.moduleExtractor(tensorSecond)

		objectEstimate = self.moduleSix(tensorFirst[-1], tensorSecond[-1], None)
		objectEstimate = self.moduleFiv(tensorFirst[-2], tensorSecond[-2], objectEstimate)
		objectEstimate = self.moduleFou(tensorFirst[-3], tensorSecond[-3], objectEstimate)
		objectEstimate = self.moduleThr(tensorFirst[-4], tensorSecond[-4], objectEstimate)
		objectEstimate = self.moduleTwo(tensorFirst[-5], tensorSecond[-5], objectEstimate)

		return objectEstimate['tensorFlow'] + self.moduleRefiner(objectEstimate['tensorFeat'])
	# end
# end

moduleNetwork = Network().cuda()

##########################################################

def estimate(tensorInputFirst, tensorInputSecond):
	tensorOutput = torch.FloatTensor()

	assert(tensorInputFirst.size(1) == tensorInputSecond.size(1))
	assert(tensorInputFirst.size(2) == tensorInputSecond.size(2))

	intWidth = tensorInputFirst.size(2)
	intHeight = tensorInputFirst.size(1)

# 	assert(intWidth == 1024) # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue
# 	assert(intHeight == 436) # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue

	if True:
		tensorInputFirst = tensorInputFirst.cuda()
		tensorInputSecond = tensorInputSecond.cuda()
		tensorOutput = tensorOutput.cuda()
	# end

	if True:
		tensorPreprocessedFirst = tensorInputFirst.view(1, 3, intHeight, intWidth)
		tensorPreprocessedSecond = tensorInputSecond.view(1, 3, intHeight, intWidth)

		intPreprocessedWidth = int(math.floor(math.ceil(intWidth / 64.0) * 64.0))
		intPreprocessedHeight = int(math.floor(math.ceil(intHeight / 64.0) * 64.0))

		tensorPreprocessedFirst = torch.nn.functional.upsample(input=tensorPreprocessedFirst, size=(intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)
		tensorPreprocessedSecond = torch.nn.functional.upsample(input=tensorPreprocessedSecond, size=(intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)

		tensorFlow = 20.0 * torch.nn.functional.upsample(input=moduleNetwork(tensorPreprocessedFirst, tensorPreprocessedSecond), size=(intHeight, intWidth), mode='bilinear', align_corners=False)

		tensorFlow[:, 0, :, :] *= float(intWidth) / float(intPreprocessedWidth)
		tensorFlow[:, 1, :, :] *= float(intHeight) / float(intPreprocessedHeight)

		tensorOutput.resize_(2, intHeight, intWidth).copy_(tensorFlow[0, :, :, :])
	# end

	if True:
		tensorInputFirst = tensorInputFirst.cpu()
		tensorInputSecond = tensorInputSecond.cpu()
		tensorOutput = tensorOutput.cpu()
	# end

	return tensorOutput
# end

##########################################################

def compute_flow_pair(img1, img2):
    tensorInputFirst = torch.FloatTensor(numpy.array(img1)[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) / 255.0)
    tensorInputSecond = torch.FloatTensor(numpy.array(img2)[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) / 255.0)
    tensorOutput = estimate(tensorInputFirst, tensorInputSecond)
    return tensorOutput.permute(1, 2, 0)


def warp(x, flo):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow

    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow

    """
    B, C, H, W = x.size()
    # mesh grid 
    xx = torch.arange(0, W).view(1,-1).repeat(H,1)
    yy = torch.arange(0, H).view(-1,1).repeat(1,W)
    xx = xx.view(1,1,H,W).repeat(B,1,1,1)
    yy = yy.view(1,1,H,W).repeat(B,1,1,1)
    grid = torch.cat((xx,yy),1).float()

    if x.is_cuda:
        grid = grid.cuda().detach()
    vgrid = grid + flo

    # scale grid to [-1,1] 
    vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:]/max(W-1,1)-1.0
    vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:]/max(H-1,1)-1.0

    vgrid = vgrid.permute(0,2,3,1)        
    output = torch.nn.functional.grid_sample(x, vgrid)
    print(x.type(), vgrid.type())
    mask = torch.ones(x.size()).cuda().detach()
    mask = torch.nn.functional.grid_sample(mask, vgrid)
    print(mask.type(), vgrid.type())

    mask[mask<0.999] = 0
    mask[mask>0] = 1

    return output*mask 
    

if __name__ == '__main__':
    
    
    # test frame pair
#     img1 = PIL.Image.open(arguments_strFirst)
#     img2 = PIL.Image.open(arguments_strSecond)
#     flow = compute_flow_pair(img1, img2)
    
#     outfile = open(out, 'wb')
#     numpy.array(flow.shape, numpy.float32).tofile(outfile)
#     numpy.array(flow, numpy.float32).tofile(outfile)
#     outfile.close()
    
    
#     img1_tensor = torch.from_numpy(numpy.array(img1)).permute(2, 0, 1).unsqueeze(0).float().cuda().detach()
#     flow = flow.permute(2, 0, 1).unsqueeze(0).cuda().detach()
#     out = warp(img1_tensor, flow)
#     out_np = out.cpu().squeeze(0).permute(1,2,0).numpy()
#     cv2.imwrite('../../tmp/warp.jpg', out_np)
    

    frame_dir = sys.argv[1]
    print(frame_dir)
    outfile = open(sys.argv[2], 'wb')
    mode = sys.argv[3]
    
    frame_sum = len(os.listdir(frame_dir))
    img_refer = PIL.Image.open(os.path.join(frame_dir, '{:05}.jpg'.format(10)))
    flow_list = []
    if mode == 'fb1':
        step = 1
    elif mode == 'fb5':
        step = 5
        
    for fid in range(frame_sum):

        frame_path = os.path.join(frame_dir, '{:05}.jpg'.format(fid))
        if not os.path.exists(frame_path):
            break
        img = PIL.Image.open(frame_path)
        ### middle flow
        if mode == 'mid':
            flow = compute_flow_pair(img, img_refer).numpy()
            flow_list.append(flow)
            print('Computing flow for frame ' + str(fid) + '...')
        ###
        else:
        ### forward backward flow
            frame_path = os.path.join(frame_dir, '{:05}.jpg'.format(fid - step))
            if not os.path.exists(frame_path):
                flow_for = None
            else:
                img_for = PIL.Image.open(frame_path)
                print('Computing forward flow for frame ' + str(fid) + '...')
                flow_for = compute_flow_pair(img, img_for).numpy()

            frame_path = os.path.join(frame_dir, '{:05}.jpg'.format(fid + step))
            if not os.path.exists(frame_path):
                flow_back = None
            else:
                img_back = PIL.Image.open(frame_path)
                print('Computing backward flow for frame ' + str(fid) + '...')
                flow_back = compute_flow_pair(img, img_back).numpy()

            if flow_for is None:
                flow_for = flow_back
            if flow_back is None:
                flow_back = flow_for
            flow_list.append(flow_for)
            flow_list.append(flow_back)
        ###
       
    flow_list = numpy.array(flow_list)
    numpy.array(flow_list.shape, numpy.float32).tofile(outfile)
    numpy.array(flow_list, numpy.float32).tofile(outfile)
    outfile.close()
    