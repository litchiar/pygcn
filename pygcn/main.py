import utils
import models
import torch.optim as optim
import torch.nn.functional as F
import time
from matplotlib import pyplot as plt

train_loss = []
train_accurate = []
test_loss = []
test_accurate = []


def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])#nllloss
    acc_train = utils.accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()
    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = utils.accuracy(output[idx_val], labels[idx_val])
    train_loss.append(loss_val.item())
    train_accurate.append(acc_val.item())
    print('Epoch: {:04d}'.format(epoch + 1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))


def test():
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = utils.accuracy(output[idx_test], labels[idx_test])
    test_loss.append(loss_test.item())
    test_accurate.append(acc_test.item())
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
adj, features, labels, idx_train, idx_val, idx_test = utils.load_data(path="./data/cora/")
model = models.GCN(nfeat=features.shape[1],
                   nhid=16,
                   nclass=labels.max().item() + 1)
optimizer = optim.Adam(model.parameters(),
                       lr=0.01, weight_decay=0.001)
print(model)
t_total = time.time()
i = 200
for epoch in range(i):
    train(epoch)
    test()
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
x = list(range(0, i))
plt.figure()
plt.title('train')
plt.plot(x,train_loss,'r')
plt.plot(x,train_accurate,'b')
plt.show()

plt.figure()
plt.title('test')
plt.plot(x,test_loss,'r')
plt.plot(x,test_accurate,'b')
plt.show()
#
#
# model.eval()
# output = model(features, adj)
# loss_test = F.nll_loss(output[idx_test], labels[idx_test])
# acc_test = accuracy(output[idx_test], labels[idx_test])
# print("Test set results:",
#           "loss= {:.4f}".format(loss_test.item()),
#           "accuracy= {:.4f}".format(acc_test.item()))
