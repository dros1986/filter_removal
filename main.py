import torch
import torch.nn as nn
import torch.nn.init as winit
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, utils
from dataloader_python import dataloader
import math
import numpy as np
import os,sys

#winit.xavier_normal

# ----------------- PATCHES ----------------

def batch2patch(batch, patchSize):
    # init res var
    if isinstance(batch, Variable):
        patches = Variable(torch.Tensor().type_as(batch.data))
    else:
        patches = torch.Tensor().type_as(batch)
    # call over all batch
    for i in range(batch.size(0)):
        cur_patch, patches_per_row = im2patch(batch[i,:,:,:], patchSize)
        cur_patch = torch.unsqueeze(cur_patch, 0)
        if patches.numel() == 0:
            patches = cur_patch
        else:
            patches = torch.cat((patches,cur_patch),0)
    return patches, patches_per_row

def patches2batch(patches, patchSize, patches_per_row):
    # init result var
    if isinstance(patches, Variable):
        batch = Variable(torch.Tensor().type_as(patches.data))
    else:
        batch = torch.Tensor().type_as(patches)
    # call over all patches
    for i in range(patches.size(0)):
        cur_img = torch.unsqueeze(patches2im(patches[i,:,:,:], patchSize, patches_per_row), 0)
        if batch.numel() == 0:
            batch = cur_img
        else:
            batch = torch.cat((batch,cur_img), 0)
    return batch

def im2patch(img, patchSize):
    # init res var
    if isinstance(img, Variable):
        patches = Variable(torch.Tensor().type_as(img.data))
    else:
        patches = torch.Tensor().type_as(img)
    patches_per_row = 0
    for r in np.arange(0,img.size(1),patchSize):
        patches_per_row = patches_per_row + 1
        for c in np.arange(0,img.size(2),patchSize):
            cur_patch = img[:, r:r+patchSize, c:c+patchSize].contiguous()
            cur_patch = cur_patch.view(1,3,-1)
            if patches.numel() == 0:
                patches = cur_patch
            else:
                patches = torch.cat((patches, cur_patch),0)
    return patches, patches_per_row

def patches2im(patches, patchSize, patches_per_row):
    # count number of patches
    npatches = patches.size(0)
    cpatches = patches_per_row
    rpatches = int(npatches/cpatches)
    # init img
    if isinstance(patches, Variable):
        #patches = Variable(torch.Tensor().type_as(img.data))
        img = Variable(torch.zeros(3,rpatches*patchSize, cpatches*patchSize).type_as(patches.data))
    else:
        img = torch.zeros(3,rpatches*patchSize, cpatches*patchSize).type_as(patches)
    # paste into
    cur_row, cur_col = 0,0
    for i in range(patches.size(0)):
        # increase row if needed
        if not i==0 and i%cpatches==0:
            cur_row = cur_row + patchSize
            cur_col = 0
        # vec2img
        cur_patch = patches[i,:,:]
        cur_patch = cur_patch.view(3,patchSize,patchSize)
        # paste into image
        img[:,cur_row:cur_row+patchSize, cur_col:cur_col+patchSize] = cur_patch
        # increase col
        cur_col = cur_col + patchSize
    # return img
    return img


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


# ------------------ NET ------------------
class Net(nn.Module):
    def __init__(self, img_dim, patchSize, nc, nf):
        super(Net, self).__init__()
        self.img_dim = img_dim
        self.patchSize = patchSize
        # calculate number of patches
        self.hpatches = int(math.floor(img_dim[0]/patchSize))
        self.wpatches = int(math.floor(img_dim[1]/patchSize))
        self.npatches = self.hpatches *self.wpatches
        #self.npatches = int(math.floor(img_dim[0]/patchSize)*math.floor(img_dim[1]/patchSize))
        # create layers
        self.b1 = nn.BatchNorm2d(3)
        self.c1 = nn.Conv2d(3, nc, kernel_size=3, stride=2, padding=0)
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
        self.l3 = nn.Linear(nf, self.npatches*9)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self,x):
        # take x between -1 and 1
        x = x*2-1

        # convert images as array b x #px x 3 x 1
        img = x.view(-1,3, self.img_dim[0]*self.img_dim[1],1)
        img = img.clone().permute(0,2,1,3)
        # print(img.size())
        # print(x[1,:,0,2])
        # print(img[1,2,:,:])

        # convert input in patches
        #patches, patches_per_row = batch2patch(x, self.patchSize)

        # calculate filters
        x = F.relu(self.c1(self.b1(x)))
        x = F.relu(self.c2(self.b2(x)))
        x = F.relu(self.c3(self.b3(x)))
        x = F.relu(self.c4(self.b4(x)))
        x = F.relu(self.c5(self.b5(x)))
        x = x.view(-1, x.size(1)*x.size(2)*x.size(3))
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        # x = x.view(-1, self.npatches, 3, 3)
        x = x.view(-1, 9, self.hpatches, self.wpatches)
        # upsample
        x = F.upsample_bilinear(x,scale_factor=self.patchSize)
        # unroll
        x = x.view(-1,9,self.img_dim[0]*self.img_dim[1])
        # swap axes
        x = x.permute(0,2,1)
        # expand 3x3
        x = x.contiguous().view(-1,x.size(1),3,3)

        # multiply pixels for filters
        for bn in range(x.size(0)):
            img[bn,:,:,:] = torch.bmm(x[bn,:,:,:].clone(),img[bn,:,:,:].clone())

        # convert images back to original shape
        img = img.permute(0,2,1,3)
        img = img.view(-1,3, self.img_dim[0], self.img_dim[1])

        # apply tanh
        img = F.tanh(img)

        # convert back the image between 0-1
        img = (img+1)/2

        return img


# ------------------ TRAIN ------------------
# set parameters
img_dim = [256,256]
patchSize = 8 #8 #64 #16 #128
nRow = 8 #10
batchSize = nRow*nRow
batchSizeVal = 50
nepochs = 1000
saveImagesEvery = 1
saveModelEvery = 200
nc = 200
nf = 2000
lr = 0.0001 #0.0002
# init net
net = Net(img_dim, patchSize, nc, nf).cuda()

# init loss
Loss = nn.MSELoss()
#Loss = nn.L1Loss()

# init optimizer
#optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
optimizer = optim.Adam(net.parameters(), lr=lr)

# create dataloaders
base_dir = '/media/flavio/Volume/datasets/places-instagram/'
img_dirs = [os.path.join(base_dir,'images_orig/'), os.path.join(base_dir,'images/')]
gt_train = os.path.join(base_dir,'train-list.txt')
gt_valid = os.path.join(base_dir,'smallvalidation-list.txt')
train_data = dataloader.Dataloader(img_dirs, gt_train, batchSize, [256,256], [256,256], convert2torch=True, sep=',')
valid_data = dataloader.Dataloader(img_dirs, gt_valid, batchSize, [256,256], [256,256], convert2torch=True, sep=',')

# train
for epoch in range(nepochs):
    print('------------------------------- EPOCH #' + str(epoch) + ' -------------------------------')
    for bn in range(train_data.nBatches()):
        # load data
        orig, filt, target = train_data.getBatch()
        # convert to float
        orig, filt = orig.float(), filt.float()
        # convert in autograd variables
        orig, filt, target = Variable(orig), Variable(filt), Variable(target)
        # move in GPU
        orig, filt, target = orig.cuda(), filt.cuda(), target.cuda()
        # convert input in patches
        #patches, patches_per_row = batch2patch(filt, patchSize)
        #patches, patches_per_row = batch2patch(filt.data, patchSize)
        #test
        # utils.save_image(filt.data, './filt.png', nrow=nRow)
        # batch = patches2batch(patches, patchSize, patches_per_row)
        # utils.save_image(batch, './recombined.png', nrow=nRow)
        # sys.exit()
        # end test
        #patches = Variable(patches, requires_grad=False)
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
        print('Epoch: [' + str(epoch+1) + '][' + str(bn+1) + '/' + str(train_data.nBatches()) + '] Loss = ' + col + str(round(loss.data[0],4)) + endCol)
        #sys.exit()
        #print(loss.data[0])
        # if bn >= 100:
        #     sys.exit()



        #x = torch.bmm(x, patches)
        # # multiply patches for filters
        # for bn in range(x.size(0)):
        #     for pn in range(x.size(1)):
        #         # patches[bn,pn,:,:] = torch.mm(x[bn,pn,:,:], patches[bn,pn,:,:])
        #         a = x[bn,pn,:,:].clone()
        #         b = patches[bn,pn,:,:].clone()
        #         patches[bn,pn,:,:] = torch.mm(a,b)
