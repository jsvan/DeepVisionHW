import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import random
import torchvision.transforms as transforms
import os.path
from sklearn import preprocessing
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
	#	f.write(str(title)+'\n\n\n')
	for i in range(len(data)):
		plt.plot(data[i], label=name[i])
	#		f.write(str(name[i])+'\n'+str(data[i])+'\n\n\n')

	plt.legend()
	plt.title(title)
	plt.show()
	#plt.savefig(title+'.png', bbox_inches='tight')

class JulianNet(nn.Module):

	
	def __init__(self, trainset, testset, tot_epochs, loss_f, bs, trn_c, tst_c):
		super(JulianNet, self).__init__()
		self.train_set 	= trainset
		self.test_set = testset
		self.epochs 	= tot_epochs
		self.loss_function = loss_f
		self.batch_size	= bs 
		self.train_count = trn_c
		self.test_count = tst_c


		self.classify = nn.Sequential(

			nn.Linear(33, 24),
			nn.ReLU(inplace=True),
			nn.Linear(24, 16),
			nn.Linear(16,24),
			nn.ReLU(inplace=True),
			nn.Linear(24,33)


			)


	def forward(self, input):
		return self.classify(input)

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
		out = []
		loss=[]
		x=[]

		for i, x in enumerate(model.train_set):
			optimizer.zero_grad()
			min_vals, min_idxes = torch.min(x, 0, keepdim=True)
			max_vals, max_idxes = torch.max(x, 0, keepdim=True)

			target_scaled = (x - min_vals) / (max_vals-min_vals)
			input 	= Variable(x).cuda()
			target_scaled = Variable(target_scaled).cuda()
			out 	= model.forward(input)
			#loss 	= model.loss_function(out, input)
			loss =  F.l1_loss(out, target_scaled)#torch.mean(torch.abs(input-out))

			loss.backward()
			optimizer.step()	
			
			#print(float(loss))
			if float(loss) < 0.1:
				corr+=1
			avg_loss+=loss

		train_acc = corr / model.train_count
		avg_loss = float(avg_loss/i)
		test_acc, test_loss, fails = test(model)
		scheduler.step()#train_acc)
		trn_acc.append(train_acc)
		tst_acc.append(test_acc)
		tst_loss.append(test_loss)
		trn_loss.append(avg_loss)
		progress_bar(e, model.epochs, trn_acc[-1], trn_loss[-1], test_acc, test_loss)
		if e%(int(model.epochs/5))==0:
			for param_group in optimizer.param_groups:
				print(i,'LR:',  param_group['lr'])
	return trn_acc, trn_loss, test_acc, test_loss, fails

def test( model ):
	model.eval()
	with torch.no_grad():
		outV=[]
		failures=[]
		avg_loss=0.0
		preV=[]
		corr=0.0
		for i, x in enumerate(model.test_set):
			min_vals, min_idxes = torch.min(x, 0, keepdim=True)
			max_vals, max_idxes = torch.max(x, 0, keepdim=True)

			target_scaled = (x - min_vals) / (max_vals-min_vals)
			input 	= Variable(x).cuda()
			target_scaled = Variable(target_scaled).cuda()
			out 	= model.forward(input)
			#loss 	= model.loss_function(out, input)
			loss =  F.l1_loss(out, target_scaled)#torch.mean(torch.abs(input-out))
			if float( loss) < 0.1:
				corr+=1.0
			else:
				if(len(failures)<20):
					failures.append((x, loss))
			avg_loss+=loss
	return corr/model.test_count,   float(avg_loss/i),	failures

def load_split_data(batch_size):
	imgs = np.genfromtxt("data.csv", delimiter=',')
	arr = np.arange(len(imgs))
	#mean = np.mean(imgs, 0)
	#std = np.std(imgs,0)
	#imgs = (imgs-mean)/std
	quantile_transformer = preprocessing.QuantileTransformer(output_distribution='normal', random_state=0)
	imgs = quantile_transformer.fit_transform(imgs)
	np.random.seed(1234)
	np.random.shuffle(arr)
	split = int(0.85 * len(arr))

	trainX = imgs[arr[:split]]
	testX = imgs[arr[split:]]
	train_count = len(trainX)
	test_count	= len(trainX)
	trainX = torch.FloatTensor(trainX)
	testX = torch.FloatTensor(testX)

	train_loader = torch.utils.data.DataLoader(trainX,
		batch_size=batch_size, 
		shuffle=True)

	test_loader = torch.utils.data.DataLoader(testX,
		batch_size=batch_size, 
		shuffle=False)

	return train_loader, test_loader, train_count, test_count

num_epochs = 32
batch_size = 64
train_set, test_set, train_count, test_count = load_split_data(batch_size)

loss_function = torch.nn.MSELoss(size_average=True).cuda()
model = JulianNet(train_set, test_set, num_epochs, loss_function, batch_size, train_count, test_count)
model.cuda()

optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, nesterov=True)#, weight_decay=5e-4)
scheduler = MultiStepLR(optimizer, gamma=0.1, milestones=[num_epochs/2, num_epochs*3/4])#, mode='max')


trn_acc, trn_loss, tst_acc, tst_loss, fails = train(model, optimizer, scheduler)
plot([trn_acc, trn_loss, tst_acc, tst_loss], ['trn_acc', 'trn_loss', 'tst_acc', 'tst_loss'], 'ThreeMeter')
#plot([trn_loss], ['trn_loss'], 'Threemeter')
# with open('ThreeMeterfails.txt','w') as f:
# 	for fail in fails:
# 		f.write('\ntarget:'+str(fail[0])+'\toutput:'+str(fail[1]))

# with open('ThreeMeterWeights.txt','w') as f:
# 	for weight in list(model.parameters()):
#		f.write(str(list(weight))+'\n\n')

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