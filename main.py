import torch
import torch.nn as nn
import torch.nn.init as winit
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, utils
import torch.utils.data as data
from Dataset import Dataset
from multiprocessing import cpu_count
import math
import numpy as np
import os,sys
import argparse
from tqdm import tqdm


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
		x = F.upsample(x,scale_factor=self.patchSize,  mode='bilinear') # (36L, 18L+3, 256L, 256L)
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
		return ris


# ---------------- parse args ---------------
parser = argparse.ArgumentParser()
parser.add_argument("-r", "--regen", help="Regenerate images using the model specified.",
					default="")
parser.add_argument("-ps", "--patchsize", help="Dimension of the patch.",
					default=8, type=int)
parser.add_argument("-nrow", "--nrow", help="Batchsize will be nrow*nrow.",
					default=5, type=int)
parser.add_argument("-di", "--degin", help="Degree of net input.",
					default=3, type=int)
parser.add_argument("-do", "--degout", help="Degree of polynomial regressor.",
					default=3, type=int)
parser.add_argument("-indir", "--indir", help="Folder containing filtered images.",
					default='./datasets/places-instagram/images/')
parser.add_argument("-gtdir", "--gtdir", help="Folder containing ground-truth images.",
					default='./datasets/places-instagram/images_orig/')
parser.add_argument("-trl", "--train_list", help="Train list.",
					default='./datasets/places-instagram/train-list.txt')
parser.add_argument("-val", "--validation_list", help="Validation list.",
					default='./datasets/places-instagram/smallvalidation-list.txt')
parser.add_argument("-tsl", "--test_list", help="Test list.",
					default='./datasets/places-instagram/test-list.txt')
args = parser.parse_args()

# print args
conf_txt = ''
for arg in vars(args):
	conf_txt = conf_txt + '{:>10} = '.format(arg) + str(getattr(args, arg)) + '\n'
print(conf_txt)
# write args on file
out_file = open("config.txt","w")
out_file.write(conf_txt)
out_file.close()
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

# init optimizer
optimizer = optim.Adam(net.parameters(), lr=lr)

# create dataloaders
img_dirs = [args.gtdir, args.indir]
gt_train = args.train_list
gt_valid = args.validation_list
gt_test = args.test_list
# create loaders
train_loader = data.DataLoader(
		Dataset(img_dirs, gt_train, sep=','),
		batch_size = batchSize,
		shuffle = True,
		num_workers = cpu_count(),
)
valid_loader = data.DataLoader(
		Dataset(img_dirs, gt_valid, sep=','),
		batch_size = batchSize,
		shuffle = True,
		num_workers = cpu_count(),
)
test_loader = data.DataLoader(
		Dataset(img_dirs, gt_test, sep=',', include_filenames=True),
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
			# convert in autograd variables
			orig, filt, target = Variable(orig), Variable(filt), Variable(target)
			# move in GPU
			orig, filt, target = orig.cuda(), filt.cuda(), target.cuda()
			# reset gradients
			optimizer.zero_grad()
			# calculate filters and multiply them to patches
			output = net(filt)
			# apply loss
			loss = Loss(output, orig)
			# backward
			loss.backward()
			# optimizer step
			optimizer.step()

			# save images
			if bn%saveImagesEvery == 0:
				utils.save_image(orig.data, './gt.png', nrow=nRow)
				utils.save_image(filt.data, './input.png', nrow=nRow)
				utils.save_image(output.data, './output.png', nrow=nRow)

			# save model
			if bn%saveModelEvery == 0:
				save_checkpoint({
					'epoch': epoch + 1,
					'state_dict': net.state_dict(),
					'optimizer' : optimizer.state_dict(),
					'best_psnr' : best_psnr
				}, False)

			# pretty printings
			col = '\033[92m'
			endCol = '\033[0m'
			print('Epoch: [' + str(epoch+1) + '][' + str(bn+1) + '/' + str(len(train_loader)) + '] Loss = ' + col + str(round(loss.data[0],4)) + endCol)
else:
	# regenerate test set
	print('Regenerating')
	# load checkpoint
	net.load_state_dict(torch.load(args.regen)['state_dict'])
	# set network in test mode
	net.train(False)
	for bn, (data, target, fns) in enumerate(tqdm(test_loader)):
		# split images in orig and filt
		orig, filt = data
		# convert in autograd variables
		orig, filt, target = Variable(orig, requires_grad=False), Variable(filt, requires_grad=False), Variable(target, requires_grad=False)
		# move in GPU
		orig, filt, target = orig.cuda(), filt.cuda(), target.cuda()
		# regenerate
		output = net(filt)
		# save images
		for i in range(output.size(0)):
			cur_img = output[i,:,:,:].data
			cur_fn = fns[i]
			cur_fn = os.path.splitext(cur_fn)[0] + '.png'
			if not os.path.isdir('./restored/'): os.makedirs('./restored/')
			utils.save_image(cur_img, os.path.join('./restored/', cur_fn), nrow=1, padding=0)
