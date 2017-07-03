import chainer
import chainer.functions as F
import chainer.links as L


import matplotlib.pyplot as plt

class MLP(chainer.Chain):
    def __init__(self):
        super(MLP, self).__init__(
            l1=L.Linear(76*76, 100),
            l2=L.Linear(100, 100),
            l3=L.Linear(100, 6),
        )
        
    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        y = self.l3(h2)
        return y

    def forward(self,x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        y = self.l3(h2)
        return y
            


class CNN_regression(chainer.Chain):
    def __init__(self,linear_n_units,n_channels,filter_size,image_type):
        n_input_channels=1
        if image_type=="color":
            n_input_channels=3
        super(CNN_regression, self).__init__(
            conv=L.Convolution2D(n_input_channels,n_channels,filter_size),
            #l1=L.Linear((76-filter_size+1)*(76-filter_size+1)*n_channels,linear_n_units),
            l1=L.Linear(1369*n_channels,linear_n_units),
            l2=L.Linear(linear_n_units,1)
        )
        self.train = True

    def clear(self):
        self.loss = None
        self.accuracy = None

    def forward(self, x):
        h = F.relu(F.max_pooling_2d(self.conv(x),2))
        h = F.relu(self.l1(h))
        h = self.l2(h)
        return h
    
    def __call__(self, x, t):
        self.clear()
        h=self.forward(x)
        self.loss = F.mean_squared_error(h, t)

        return self.loss


class CNN_classification(chainer.Chain):
    def __init__(self,linear_n_units,n_channels,filter_size,image_type):
        n_input_channels=1
        if image_type=="color":
            n_input_channels=3
        super(CNN_classification, self).__init__(
            conv=L.Convolution2D(n_input_channels,n_channels,filter_size),
            l1=L.Linear((76-filter_size+1)*(76-filter_size+1)*n_channels,linear_n_units),
            l2=L.Linear(linear_n_units,50)
        )
        self.train = True

    def clear(self):
        self.loss = None
        self.accuracy = None

    def forward(self, x):
        h = F.relu(self.conv(x))
        h = F.relu(self.l1(h))
        h = F.relu(self.l2(h))
        return h
    
    def __call__(self, x, t):
        self.clear()
        h=self.forward(x)
        #print h.data
        self.loss = F.softmax_cross_entropy(h, t)
        #self.loss = F.mean_squared_error(h, t)
        #print "Loss : " +str(self.loss.data)
        return self.loss


class CNN_deep(chainer.Chain):
    def __init__(self):
        n_input_channels=1
        super(CNN_deep, self).__init__(
            conv1=L.Convolution2D(n_input_channels,2,3,pad=1),
            conv2=L.Convolution2D(1,4,3,pad=1),
            conv3=L.Convolution2D(1,2,3,pad=1),
            l1=L.Linear(2888,1)
        )
        self.train = True

    def clear(self):
        self.loss = None
        self.accuracy = None

    def forward(self, x):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(x))
        h = F.relu(F.max_pooling_2d(self.conv3(x),2))
        h = self.l1(h)
        return h
    
    def __call__(self, x, t):
        self.clear()
        h=self.forward(x)
        self.loss = F.mean_squared_error(h, t)

        return self.loss
