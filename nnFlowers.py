import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import random
import torchvision.transforms as transforms
import os.path
from math import ceil
#from sklearn.cross_validation import train_test_split
import torch.utils.data as data_utils
from torch.optim.lr_scheduler import MultiStepLR
from matplotlib import pyplot as plt
torch.backends.cudnn.benchmark=True

def plot(data, name, title):
	plt.clf()
	plt.ion()
	print('plotting', title, name)
	#with open(title+'.txt','w') as f:
		#f.write(str(title)+'\n\n\n')
	for i in range(len(data)):
		plt.plot(data[i], label=name[i])
#		f.write(str(name[i])+'\n'+str(data[i])+'\n\n\n')

	plt.legend()
	plt.title(title)
	plt.show()
	#plt.savefig(title+'.png', bbox_inches='tight')

class JulianNet(nn.Module):

	
	def __init__(self, num_cls, trainset, testset, tot_epochs, loss_f, bs, trn_c, tst_c):
		super(JulianNet, self).__init__()
		self.num_classes= num_cls
		self.train_set 	= trainset
		self.test_set	= testset
		self.epochs 	= tot_epochs
		self.loss_function = loss_f
		self.batch_size	= bs 
		self.train_count = trn_c
		self.test_count	 = tst_c

		self.pad=1
		self.softmax 	= nn.LogSoftmax(dim=1)
		#self.pool 		= nn.MaxPool2d(kernel_size=2, stride=2)
		self.features 	= nn.Sequential(
			#[64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
			#[nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
			#             	nn.BatchNorm2d(x),
			#				nn.ReLU(inplace=True)]
			# 'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],


			nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=self.pad),
			nn.BatchNorm2d(64),
			nn.ReLU(inplace=True),

			nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=self.pad),
			nn.BatchNorm2d(64),
			nn.ReLU(inplace=True),

			nn.MaxPool2d(kernel_size=3, stride=2),

			nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=self.pad),
			nn.BatchNorm2d(128),
			nn.ReLU(inplace=True),

			nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=self.pad),
			nn.BatchNorm2d(128),
			nn.ReLU(inplace=True),

			nn.MaxPool2d(kernel_size=3, stride=2),

			nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=self.pad),
			nn.BatchNorm2d(256),
			nn.ReLU(inplace=True),

			nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=self.pad),
			nn.BatchNorm2d(256),
			nn.ReLU(inplace=True),


			nn.MaxPool2d(kernel_size=3, stride=2),

			nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=self.pad),
			nn.BatchNorm2d(512),
			nn.ReLU(inplace=True),

			nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=self.pad),
			nn.BatchNorm2d(512),
			nn.ReLU(inplace=True),

			nn.MaxPool2d(kernel_size=3, stride=2),


			)
		self.classify = nn.Sequential(
			nn.Dropout(),
			nn.Linear(512, 64),
			nn.ReLU(inplace=True),
			nn.BatchNorm1d(64),
			nn.Dropout(),
			nn.Linear(64, self.num_classes)
			)

	def flatten(self, input):
		return input.view(input.size(0), -1)

	def forward(self, input):
		output = self.features(input)
		output= self.flatten(output)
		output = self.classify(output)
		output = self.softmax(output)
		return output

def progress_bar(curr_epoch, tot_epoch, accuracy, loss, test_acc, test_loss):
	divider= ceil(tot_epoch/50)
	tot_len = ceil(tot_epoch/divider)
	epoch_len = ceil(curr_epoch/divider)
	print('\r['+ '#'*int(epoch_len) + '-'*int(tot_len-epoch_len) + ']', 'epoch:',str(curr_epoch)+'/'+str(tot_epoch), '  accuracy:', format(accuracy, '0.6f'), '  loss:', format(float(loss), '0.6f'), '  test_acc:', format(test_acc, '0.6f'), '  test_loss:', format(test_loss, '0.6f')   , end='')


def train( model, optimizer, scheduler):
	trn_loss=[]
	tst_loss=[]
	trn_acc =[]
	tst_acc =[]

	progress_bar(0, model.epochs, 0,0,0,0)
	for e in range(model.epochs):
		model.train()
		avg_loss = 0.0
		corr=0.0
		for i, (x, y) in enumerate(model.train_set):
			optimizer.zero_grad()
			input 	= Variable(x).cuda()
			out 	= model.forward(input)
			target 	= Variable(y).cuda()
			loss 	= model.loss_function(out, target)
			avg_loss += loss

			for tr in range(len(out)):
				out_val, out_index = out[tr].max(0)
				if out_index == target[tr]:
					corr+=1.0

			loss.backward()
			optimizer.step()			

		test_acc, test_loss, fails = test(model)
		train_acc = corr / model.train_count
		train_loss = float(avg_loss) / model.train_count
		scheduler.step()#train_acc)
		trn_acc.append(train_acc)
		trn_loss.append(train_loss)
		tst_acc.append(test_acc)
		tst_loss.append(test_loss)

		progress_bar(e, model.epochs, train_acc, train_loss, test_acc, test_loss)
		if e%30==0:
			for param_group in optimizer.param_groups:
				print(i,'LR:',  param_group['lr'])

	return trn_acc, trn_loss, tst_acc, tst_loss, fails

def test( model ):
	model.eval()
	with torch.no_grad():
		avg_loss = 0.0
		corr=0.0
		failures = []
		for i, (x, y) in enumerate(model.test_set):

			input 	= Variable(x).cuda()
			out 	= model.forward(input)
			target 	= Variable(y).cuda()
			loss 	= model.loss_function(out, target)
			avg_loss += loss

			for tr in range(len(out)):
				out_val, out_index = out[tr].max(0)
				if out_index == target[tr]:
					corr+=1.0
				else:
					failures.append((int(target[tr]), int(out_index)))
	return corr/model.test_count,   float(avg_loss)/model.test_count,	failures[:min(len(failures), 20)]

def load_split_data(batch_size):
	imgs = np.load("flower_imgs.npy")
	labels = np.load("flower_labels.npy")
	num_categories = int(np.max(labels))+1
	labels = torch.LongTensor(labels)
	arr = np.arange(len(imgs))
	np.random.seed(1234)
	np.random.shuffle(arr)
	split = int(0.85 * len(arr))

	trainX, trainY = imgs[arr[:split]], labels[arr[:split]]
	testX, testY = imgs[arr[split:]], labels[arr[split:]]
	train_count = len(trainY)
	test_count	= len(testY)
	img_mean = np.mean(np.swapaxes(imgs/255.0,0,1).reshape(3, -1), 1)
	img_std = np.std(np.swapaxes(imgs/255.0,0,1).reshape(3, -1), 1)


	class FlowerLoader(torch.utils.data.Dataset):
		def __init__(self, x_arr, y_arr, transform=None):
			self.x_arr = x_arr
			self.y_arr = y_arr
			self.transform = transform

		def __len__(self):
			return self.x_arr.shape[0]

		def __getitem__(self, index):
			img = self.x_arr[index]
			label = self.y_arr[index]
			if self.transform is not None:
				img = self.transform(img)

			return img, label

	normalize = transforms.Normalize(mean=list(img_mean),
									 std=list(img_std))

	train_loader = torch.utils.data.DataLoader(
		FlowerLoader(trainX, trainY, transforms.Compose([
			transforms.ToPILImage(),
			transforms.RandomCrop(32, padding=4),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			normalize,
		])),
		batch_size=batch_size, shuffle=True)
	test_loader = torch.utils.data.DataLoader(
		FlowerLoader(testX, testY, transforms.Compose([
			transforms.ToPILImage(),
			transforms.ToTensor(),
			normalize,
		])),
		batch_size=batch_size, shuffle=False)

	return train_loader, test_loader, num_categories, train_count, test_count

num_epochs = 94
batch_size = 64
train_set, test_set, n_categories, train_count, test_count = load_split_data(batch_size)

loss_function = torch.nn.CrossEntropyLoss().cuda()
model = JulianNet(n_categories, train_set, test_set, num_epochs, loss_function, batch_size, train_count, test_count)
model.cuda()

optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, nesterov=True, weight_decay=5e-4)
scheduler = MultiStepLR(optimizer, gamma=0.1, milestones=[num_epochs/2, num_epochs*3/4])#, mode='max')


trn_acc, trn_loss, tst_acc, tst_loss, fails = train(model, optimizer, scheduler)
plot([trn_acc, trn_loss, tst_acc, tst_loss], ['trn_acc', 'trn_loss', 'tst_acc', 'tst_loss'], 'Flowers')

# with open('flowerfails.txt','w') as f:
# 	for fail in fails:
# 		f.write('\ntarget:'+str(fail[0])+'\toutput:'+str(fail[1]))

# with open('flowerweights.txt','w') as f:
# 	for weight in list(model.parameters()):
# 		f.write(str(list(weight))+'\n\n')

'''
*1. Make sure you shuffle your training data before each epoch.

2. Decay the learning rate if you find accuracy saturates. I usually multiply the learning rate by 0.1 at 50% and 75% of the training progress.

3. Make sure your optimizer is defined only once, i.e., it should be defined outside of the training loop. It keeps the momentum of the gradients.

4. Flowers dataset is a bit difficult to train, so here I'm giving you more instructions. You can refer to the vgg (or other) models in this 
repository to design your network, but never copy directly. With 85% training data, a vgg13 from this repo, a SGD optimizer with momentum=0.9, 
weight_decay=5e-4, (the lr is for you to find, but remember to tune that!), a batch size in the range (16, 100), total epochs < 200, I was 
able to achieve 82.5% accuracy. You should use data augmentation. Here is the sample code for implementing dataloader with PyTorch and data 
augmentation.

5. The most direct reference for writing PyTorch should be it's examples.
'''