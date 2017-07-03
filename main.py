import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils, iterators
from chainer.training import extensions
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from mock import MagicMock
from scipy import misc
from import_data import *



# -- GPU --#

class CNN_classification(chainer.Chain):
    def __init__(self, n_channels, filter_size):
        super(CNN_classification, self).__init__(
            conv=L.Convolution2D(1, n_channels, filter_size),
            l1=L.Linear(1369 * n_channels, 6)
        )

    def __call__(self, x):
        h = F.relu(F.max_pooling_2d(self.conv(x), 2))
        h = self.l1(h)
        return h


class MLP(chainer.Chain):
    def __init__(self, n_units):
        super(MLP, self).__init__(
            l1=L.Linear(49*49, n_units),
            l2=L.Linear(n_units, 4),
        )

    def __call__(self, x):
        h1 = self.l1(x)
        y = self.l2(h1)

        return y



filename = "../Sampled"
training_size = 3500
test_size = 3000
load_model = False
n_filter = 2
filter_size = 3
n_epoch = 15
batchsize = 10
n_units = 40
gpu = -1  # If gpu>0 then we utilise CPU

model = L.Classifier(MLP(n_units))
#model = L.Classifier(CNN_classification(n_filter, filter_size))

optimizer = optimizers.Adam()
optimizer.setup(model)

if gpu > 0:
    cuda.get_device().use()
    model.to_gpu()
    xp = cuda.cupy
else:
    xp = np


# -- Import data --#
[Xtr, Ytr, Xte, Yte] = get_randomized_classification_data(filename, xp, training_size, test_size)

train = zip(Xtr, Ytr)
test = zip(Xte, Yte)

train_iter = iterators.SerialIterator(train, batch_size=4, shuffle=True)
test_iter = iterators.SerialIterator(test, batch_size=20, repeat=False, shuffle=False)

updater = chainer.training.StandardUpdater(train_iter, optimizer)
trainer = chainer.training.Trainer(updater, (n_epoch, 'epoch'), out='result')

trainer.extend(extensions.Evaluator(test_iter, model))
trainer.extend(extensions.LogReport())
trainer.extend(extensions.PrintReport(['epoch', 'main/accuracy', 'validation/main/accuracy']))
trainer.extend(extensions.snapshot())
trainer.extend(extensions.ProgressBar())
trainer.run()

chainer.serializers.save_npz("result/perfect", model)

#print "Y test 2 : " +str(Yte[2])
#img=misc.imread(filename + "/ValidationL/" + str(1) + ".jpg",flatten=True)
#X=np.zeros((1,50,50))
#X[0]=img
#x=Variable(Xte[0:1])
#print model.predictor(x).data
