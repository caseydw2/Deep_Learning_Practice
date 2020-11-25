
from d2l import mxnet as d2l
from mxnet import autograd, np,npx
import random
npx.set_np()



def synthetic_data(w,b,num_examples):
    """Generate y = Xw + b + noise"""
    X = np.random.normal(0,1,(num_examples, len(w)))
    y = np.dot(X,w)+b
    y += np.random.normal(0,0.1,y.shape)
    return X,y.reshape((-1,1))

true_w = np.array([2,-3.4])
true_b = 4.2
features, labels = synthetic_data(true_w,true_b, 1000)

#print('features:', features[0],'\nlabel:', labels[0])

#d2l.set_figsize()
#d2l.plt.scatter(d2l.numpy(features[:,1]),d2l.numpy(labels),1)
#d2l.plt.show()

def data_iter(batch_size,features,labels):
    num_examples = len(labels)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0,num_examples,batch_size):
        batch_indices = np.array(
            indices[i:min(i+batch_size,num_examples)]
        )
        yield features[batch_indices],labels[batch_indices]

batch_size = 10
#for X,y in data_iter(batch_size,features,labels):
#    print(X,'\n', y)
#    break

w = np.random.normal(0,0.1,(2,1))
b = np.zeros(1)
w.attach_grad()
b.attach_grad()

def linreg(X,w,b):
    return np.dot(X,w)+b

def squared_loss(y_hat,y):
    return (y_hat - y.reshape(y_hat.shape)) **2 / 2

def stochastic_gradient_descent(params,lr,batch_size):
    for param in params:
        param[:] = param - lr * param.grad/batch_size

##Training
lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss

for epoch in range(num_epochs):
    for X,y in data_iter(batch_size,features,labels):
        with autograd.record():
            l = loss(net(X,w,b),y)
        l.backward()
        stochastic_gradient_descent([w,b],lr,batch_size)
    train_l = loss(net(features,w,b),labels)
    print(f'epoch {epoch +1}, loss {float(train_l.mean()):f}')

## error
print(f'error in estimating w: {true_w - w.reshape(true_w.shape)}')
print(f'error in estimating b: {true_b - b}')


