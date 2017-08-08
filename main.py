import torch
import torch.nn as nn
import torch.nn.init as winit
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, utils
import torch.utils.data as data
from pytorch_dataloader.dataloader import Dataset
from multiprocessing import cpu_count
import math
import numpy as np
import os,sys
import argparse
from tqdm import tqdm

#winit.xavier_normal


def save_checkpoint(state, is_best, filename='checkpoint.pth'):
	torch.save(state, filename)
	if is_best:
		shutil.copyfile(filename, 'model_best.pth')


def psnr(img1, img2):
	mse = torch.mean( torch.pow( (img1 - img2),2 ) )
	if mse == 0:
		return 100
	PIXEL_MAX = 1.0
	return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def batch2poli(batch, deg):
	if deg > 3: raise ValueError('deg > 2 not implemented yet.')
	if deg == 1: return batch
	r,g,b = torch.unsqueeze(batch[:,0,:,:],1), torch.unsqueeze(batch[:,1,:,:],1), torch.unsqueeze(batch[:,2,:,:],1)
	# r + g + b
	ris = batch
	if deg > 1:
		# r^2 + g^2 + b^2 + bg + br + gr
		ris = torch.cat((ris,r.pow(2)),1) # r^2
		ris = torch.cat((ris,g.pow(2)),1) # g^2
		ris = torch.cat((ris,b.pow(2)),1) # b^2
		ris = torch.cat((ris,b*g),1) # bg
		ris = torch.cat((ris,b*r),1) # br
		ris = torch.cat((ris,g*r),1) # gr
	if deg > 2:
		# (r^3 + g^3 + b^3) + (gb^2 + rb^2) + (bg^2 + rg^2) + (br^2  + gr^2) + bgr
		ris = torch.cat((ris,r.pow(3)),1) # r^3
		ris = torch.cat((ris,g.pow(3)),1) # g^3
		ris = torch.cat((ris,b.pow(3)),1) # b^3
		ris = torch.cat((ris,g*b.pow(2)),1) # gb^2
		ris = torch.cat((ris,r*b.pow(2)),1) # rb^2
		ris = torch.cat((ris,b*g.pow(2)),1) # bg^2
		ris = torch.cat((ris,r*g.pow(2)),1) # rg^2
		ris = torch.cat((ris,b*r.pow(2)),1) # br^2
		ris = torch.cat((ris,g*r.pow(2)),1) # gr^2
		ris = torch.cat((ris,b*g*r),1) # bgr
	return ris



# ------------------ NET ------------------
class Net(nn.Module):
	def __init__(self, img_dim, patchSize, nc, nf, deg_poly_in, deg_poly_out):
		super(Net, self).__init__()
		self.img_dim = img_dim
		self.patchSize = patchSize
		self.deg_poly_in = deg_poly_in
		self.deg_poly_out = deg_poly_out
		# calculate number of channels
		self.nch_in = 3
		if deg_poly_in > 1: self.nch_in = self.nch_in + 6
		if deg_poly_in > 2: self.nch_in = self.nch_in + 10
		if deg_poly_in > 3: raise ValueError('deg > 3 not implemented yet.')
		self.nch_out = 3
		if deg_poly_out > 1: self.nch_out = self.nch_out + 6
		if deg_poly_out > 2: self.nch_out = self.nch_out + 10
		if deg_poly_out > 3: raise ValueError('deg > 3 not implemented yet.')
		# calculate number of patches
		self.hpatches = int(math.floor(img_dim[0]/patchSize))
		self.wpatches = int(math.floor(img_dim[1]/patchSize))
		self.npatches = self.hpatches *self.wpatches
		#self.npatches = int(math.floor(img_dim[0]/patchSize)*math.floor(img_dim[1]/patchSize))
		# create layers
		self.b1 = nn.BatchNorm2d(self.nch_in)
		self.c1 = nn.Conv2d(self.nch_in, nc, kernel_size=3, stride=2, padding=0)
		self.b2 = nn.BatchNorm2d(nc)
		self.c2 = nn.Conv2d(nc, nc, kernel_size=3, stride=2, padding=0)
		self.b3 = nn.BatchNorm2d(nc)
		self.c3 = nn.Conv2d(nc, nc, kernel_size=3, stride=2, padding=0)
		self.b4 = nn.BatchNorm2d(nc)
		self.c4 = nn.Conv2d(nc, nc, kernel_size=3, stride=2, padding=0)
		self.b5 = nn.BatchNorm2d(nc)
		self.c5 = nn.Conv2d(nc, nc, kernel_size=3, stride=2, padding=0)

		self.l1 = nn.Linear(nc*7*7, nf)
		self.l2 = nn.Linear(nf, nf)
		self.l3 = nn.Linear(nf, self.npatches*(self.nch_out*3+3)) # 2000 -> 21504   1->21

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()

	def forward(self,x):
		# create poly input
		img = batch2poli(x, self.deg_poly_out)

		# convert net input to poly
		x = batch2poli(x, self.deg_poly_in)
		# convert poly input as array b x #px x 3 x 1
		img = img.view(-1,self.nch_out, self.img_dim[0]*self.img_dim[1],1)
		img = img.clone().permute(0,2,1,3)
		# print(img.size())
		# print(x[1,:,0,2])
		# print(img[1,2,:,:])

		# calculate filters
		x = F.relu(self.c1(self.b1(x)))
		x = F.relu(self.c2(self.b2(x)))
		x = F.relu(self.c3(self.b3(x)))
		x = F.relu(self.c4(self.b4(x)))
		x = F.relu(self.c5(self.b5(x)))
		x = x.view(-1, x.size(1)*x.size(2)*x.size(3))
		x = F.relu(self.l1(x))
		x = F.relu(self.l2(x))
		x = self.l3(x)
		# x = x.view(-1, 9, self.hpatches, self.wpatches)
		x = x.view(-1, self.nch_out*3+3, self.hpatches, self.wpatches)
		# upsample
		x = F.upsample_bilinear(x,scale_factor=self.patchSize) # (36L, 18L+3, 256L, 256L)
		# unroll
		#x = x.view(-1,9,self.img_dim[0]*self.img_dim[1])
		x = x.view(-1,self.nch_out*3+3,self.img_dim[0]*self.img_dim[1]) # (36L, 18L+3, 65536L)
		# swap axes
		x = x.permute(0,2,1) # (36L, 65536L, 18L+3)
		# expand 3xnch
		x = x.contiguous().view(-1,x.size(1),3,self.nch_out+1) # (36L, 65536L, 3L, 6L+1)
		# add white channels to image
		w = Variable( torch.ones(img.size(0),img.size(1),1,img.size(3)) ).cuda()
		img = torch.cat((img,w),2)
		# prepare output variable
		ris = Variable(torch.zeros(img.size(0),img.size(1),3,img.size(3))).cuda()
		# multiply pixels for filters
		for bn in range(x.size(0)):
			ris[bn,:,:,:] = torch.bmm(x[bn,:,:,:].clone(),img[bn,:,:,:].clone())
		# convert images back to original shape
		ris = ris.permute(0,2,1,3)
		ris = ris.contiguous()
		ris = ris.view(-1,3, self.img_dim[0], self.img_dim[1])
		# apply tanh
		# img = F.tanh(img)

		return ris


# ---------------- parse args ---------------
parser = argparse.ArgumentParser()
parser.add_argument("-r", "--regen", help="Regenerate images using the model specified.",
					default="")
parser.add_argument("-ps", "--patchsize", help="Dimension of the patch.",
					default=32)
parser.add_argument("-nrow", "--nrow", help="Batchsize will be nrow*nrow.",
					default=5)
parser.add_argument("-di", "--degin", help="Degree of net input.",
					default=1)
parser.add_argument("-do", "--degout", help="Degree of polynomial regressor.",
					default=1)
parser.add_argument("-dir", "--dir", help="Folder containing images.",
					default='/media/flavio/Volume/datasets/places-instagram/')
args = parser.parse_args()

# set args to int
args.patchsize = int(args.patchsize)
args.nrow = int(args.nrow)
args.degin = int(args.degin)
args.degout = int(args.degout)
# print args
conf_txt = ''
for arg in vars(args):
	conf_txt = conf_txt + '{:>10} = '.format(arg) + str(getattr(args, arg)) + '\n'
print(conf_txt)
# write args on file
out_file = open("config.txt","w")
out_file.write(conf_txt)
out_file.close()
# for a in args:
# 	import ipdb; ipdb.set_trace()
# ------------------ TRAIN ------------------
# set parameters
img_dim = [256,256]
patchSize = args.patchsize
nRow = args.nrow
batchSize = nRow*nRow
batchSizeVal = 50
nepochs = 4
saveImagesEvery = 1
saveModelEvery = 200
nc = 200
nf = 2000
lr = 0.0001 #0.0002
deg_poly_in = args.degin
deg_poly_out = args.degout
# init net
net = Net(img_dim, patchSize, nc, nf, deg_poly_in, deg_poly_out).cuda()

# init loss
Loss = nn.MSELoss()
#Loss = nn.L1Loss()

# init optimizer
#optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
optimizer = optim.Adam(net.parameters(), lr=lr)

# create dataloaders
base_dir = args.dir
img_dirs = [os.path.join(base_dir,'images_orig/'), os.path.join(base_dir,'images/')]
gt_train = os.path.join(base_dir,'train-list.txt')
gt_valid = os.path.join(base_dir,'smallvalidation-list.txt')
gt_test  = os.path.join(base_dir,'test-list.txt')
# create loaders
train_loader = data.DataLoader(
		Dataset(img_dirs, gt_train, [256,256], [256,256], sep=','),
		batch_size = batchSize,
		shuffle = True,
		num_workers = cpu_count(),
)
valid_loader = data.DataLoader(
		Dataset(img_dirs, gt_valid, [256,256], [256,256], sep=','),
		batch_size = batchSize,
		shuffle = True,
		num_workers = cpu_count(),
)
test_loader = data.DataLoader(
		Dataset(img_dirs, gt_test, [256,256], [256,256], sep=',', include_filenames=True),
		batch_size = batchSize,
		shuffle = True,
		num_workers = cpu_count(),
)

best_psnr = np.inf
if not args.regen:
	# train
	for epoch in range(nepochs):
		print('------------------------------- EPOCH #' + str(epoch) + ' -------------------------------')
		for bn, (data, target) in enumerate(train_loader):
			# split images in orig and filt
			orig, filt = data
			# convert to float
			# orig, filt = orig.float(), filt.float()
			# convert in autograd variables
			orig, filt, target = Variable(orig), Variable(filt), Variable(target)
			# move in GPU
			orig, filt, target = orig.cuda(), filt.cuda(), target.cuda()
			# reset gradients
			optimizer.zero_grad()
			# calculate filters and multiply them to patches
			output = net.forward(filt)
			# apply loss
			loss = Loss(output, orig)
			# backward
			loss.backward()
			# optimizer step
			optimizer.step()
			# make grid
			# grid = utils.make_grid(output.data, nrow=nRow)
			# print(grid.size())
			if bn%saveImagesEvery == 0:
				utils.save_image(orig.data, './gt.png', nrow=nRow)
				utils.save_image(filt.data, './input.png', nrow=nRow)
				utils.save_image(output.data, './output.png', nrow=nRow)


			if bn%saveModelEvery == 0:
				save_checkpoint({
					'epoch': epoch + 1,
					'state_dict': net.state_dict(),
					'optimizer' : optimizer.state_dict(),
					'best_psnr' : best_psnr
				}, False)

			#print(grid.size())
			# pretty printings
			col = '\033[92m'
			endCol = '\033[0m'
			print('Epoch: [' + str(epoch+1) + '][' + str(bn+1) + '/' + str(len(train_loader)) + '] Loss = ' + col + str(round(loss.data[0],4)) + endCol)
			#sys.exit()
			#print(loss.data[0])
			# if bn >= 100:
			#	 sys.exit()
else:
	# regenerate test set
	print('Regenerating')
	# load checkpoint
	net.load_state_dict(torch.load(args.regen)['state_dict'])
	# set network in test mode
	net.train(False)
	for bn, (data, target, fns) in enumerate(tqdm(test_loader)):
		#for f in fns: print(f)
		# split images in orig and filt
		orig, filt = data
		# convert in autograd variables
		orig, filt, target = Variable(orig, requires_grad=False), Variable(filt, requires_grad=False), Variable(target, requires_grad=False)
		# move in GPU
		orig, filt, target = orig.cuda(), filt.cuda(), target.cuda()
		# regenerate
		output = net.forward(filt)
		# save images
		for i in range(output.size(0)):
			cur_img = output[i,:,:,:].data
			cur_fn = fns[i]
			if not os.path.isdir('./restored/'): os.makedirs('./restored/')
			utils.save_image(cur_img, os.path.join('./restored/', cur_fn), nrow=1, padding=0)
