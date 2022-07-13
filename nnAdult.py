import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler as MMS
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

			nn.Linear(67, 128),
			nn.ReLU(inplace=True),
			
			nn.Dropout(.2),
			nn.Linear(128, 128),
			nn.ReLU(inplace=True),

			nn.Dropout(.2),
			nn.Linear(128, 128),
			nn.ReLU(inplace=True),
			
			nn.Dropout(.2),
			nn.Linear(128, 1),
			nn.Sigmoid(),


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

		for i, (x,y) in enumerate(model.train_set):
			optimizer.zero_grad()
			
			input 	= Variable(x).cuda()
			target 	= Variable(y).cuda()

			out 	= model.forward(input)
			loss 	= model.loss_function(out, target)

			loss.backward()
			optimizer.step()	
			
			for iii in range(len(out)):	
				if (out[iii].item()>0.5)==(y[iii].item()>0.5):
					corr+=1
			avg_loss+=loss

		test_acc, test_loss, fails = test(model)
		scheduler.step()#train_acc)
		tst_acc.append(test_acc)
		tst_loss.append(test_loss)
		trn_acc.append(corr/model.train_count)
		trn_loss.append(float(avg_loss/model.train_count))
		progress_bar(e, model.epochs, trn_acc[-1], trn_loss[-1], test_acc, test_loss)
		if e%(int(model.epochs/5))==0:
			for param_group in optimizer.param_groups:
				print(i,'LR:',  param_group['lr'])
	return trn_acc, trn_loss, tst_acc, tst_loss, fails

def test( model ):
	model.eval()
	with torch.no_grad():

		outV=[]
		preV=[]
		corr =0.0
		failures = []
		avg_loss = 0.0
		for i, (x, y) in enumerate(model.test_set):

			input 	= Variable(x).cuda()
			target 	= Variable(y).cuda()

			out 	= model.forward(input)
			loss 	= model.loss_function(out, target)

			for iii in range(len(out)):
				if (out[iii].item()>0.5)==(y[iii]>0.5):
					corr+=1
				else:
					failures.append((input[iii], loss))
			avg_loss+=loss
	return corr/model.test_count,  float( avg_loss/model.test_count),	failures[:min(len(failures), 20)]

def load_split_data(batch_size):
	imgs = np.load("data.npy")
	labels = np.load('labels.npy')
	imgs=MMS().fit_transform(imgs)
	arr = np.arange(len(imgs))

	np.random.seed(1234)
	np.random.shuffle(arr)
	split = int(0.85 * len(arr))

	trainX, trainY = imgs[arr[:split]], labels[arr[:split]]
	trainX=list(trainX)
	trainY=list(trainY)
	fin = len(trainX)
	# for i in range(fin):
	# if trainY[i] == 0:
	# trainX.append(trainX[i])
	# trainY.append(trainY[i])
	# #trainX.append(trainX[i])
	# #trainY.append(trainY[i])

	trainX=np.array(trainX)
	trainY=np.array(trainY)
	print(str(trainY.sum()/len(trainY)))
	testX, testY = imgs[arr[split:]], labels[arr[split:]]
	train_count = len(trainY)
	test_count	= len(testY)

	trainX = torch.FloatTensor(torch.from_numpy(trainX).float())
	testX = torch.FloatTensor(torch.from_numpy(testX).float())
	trainY = torch.from_numpy(trainY).float()
	testY = torch.from_numpy(testY).float()

	dstrain = torch.utils.data.TensorDataset(trainX, trainY)
	dstest	= torch.utils.data.TensorDataset(testX,testY)
	train_loader = torch.utils.data.DataLoader(dstrain,
		batch_size=batch_size, 
		shuffle=True)

	test_loader = torch.utils.data.DataLoader(dstest,
		batch_size=batch_size, 
		shuffle=False)

	return train_loader, test_loader, train_count, test_count

num_epochs = 10
batch_size = 64
train_set, test_set, train_count, test_count = load_split_data(batch_size)

loss_function = torch.nn.BCELoss().cuda()
model = JulianNet(train_set, test_set, num_epochs, loss_function, batch_size, train_count, test_count)
model.cuda()

optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)#, nesterov=True, weight_decay=5e-5)
scheduler = MultiStepLR(optimizer, gamma=0.1, milestones=[num_epochs/2, num_epochs*3/4])#, mode='max')


trn_acc, trn_loss, tst_acc, tst_loss, fails = train(model, optimizer, scheduler)
plot([trn_acc, trn_loss, tst_acc, tst_loss], ['trn_acc', 'trn_loss', 'tst_acc', 'tst_loss'], 'Adult')
#plot([trn_loss, trn_acc], ['trn_loss', 'trn_acc'], 'Adult')
# with open('AdultFails.txt','w') as f:
# 	for fail in fails:
# 		f.write('\ntarget:'+str(fail[0])+'\toutput:'+str(fail[1]))
# print(fail.sum())
# with open('AdultWeights.txt','w') as f:
# 	for weight in list(model.parameters()):
# 		f.write(str(list(weight))+'\n\n')

