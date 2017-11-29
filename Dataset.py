import os,sys
import torch.utils.data as data
from torchvision import transforms
from PIL import Image

class Dataset(data.Dataset):
	def __init__(self,dirs,listfn, include_filenames=False, sep=','):
		# save dirs
		if not isinstance(dirs, list): dirs = [dirs]
		self.dirs = dirs
		self.include_filenames = include_filenames
		self.sep = sep
		# get filenames in listfn
		in_file = open(listfn,"r")
		self.lines = in_file.read().split('\n')
		in_file.close()
		self.fns = [l for l in self.lines if l]

	def __getitem__(self, index):
		# split line to get filename and label if exist
		parts = self.fns[index].split(self.sep)
		fn, label = parts[0], int(parts[1]) if len(parts)>1 else 0
		out = []
		# for each dir
		for d in self.dirs:
			# read image
			cur_img = Image.open(os.path.join(d,fn))
			# convert to rgb if needed
			if not cur_img.mode == 'RGB':
				cur_img = cur_img.convert('RGB')
			# scale image if needed
			width, height = cur_img.size
			if not (width==256 and height==256):
				cur_img.resize((256,256), Image.ANTIALIAS)
			# convert to pytorch
			cur_img = transforms.ToTensor()(cur_img)
			# add to output
			out.append(cur_img)
		# return images and labels
		if self.include_filenames:
			return out, label, fn
		else:
			return out, label

	def __len__(self):
		return len(self.fns)
