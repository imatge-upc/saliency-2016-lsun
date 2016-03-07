import numpy as np
import lasagne
import theano
import theano.tensor as T
import gzip
import pickle
from scipy.misc import imsave
import cPickle as pickle

train, val, test = pickle.load(gzip.open('mnist.pkl.gz'))

X_train, y_train = train
X_val, y_val = val

# The original 28x28 pixel images are flattened into 784 dimensional feature vectors
X_train.shape

# For training, we want to sample examples at random in small batches
def batch_gen(X, y, N):
    while True:
        idx = np.random.choice(len(y), N)
        yield X[idx].astype('float32'), y[idx].astype('int32')

# A very simple network, a single layer with one neuron per target class.
# Using the softmax activation function gives us a probability distribution at the output.
l_in = lasagne.layers.InputLayer(shape=(None, 784))
l_out = lasagne.layers.DenseLayer(
    l_in,
    num_units=10,
    nonlinearity=lasagne.nonlinearities.softmax)

# Symbolic variables for our input features and targets
X_sym = T.matrix()
y_sym = T.ivector()


# Theano expressions for the output distribution and predicted class
output = lasagne.layers.get_output(l_out, X_sym)
pred = output.argmax(-1)


# The loss function is cross-entropy averaged over a minibatch, we also compute accuracy as an evaluation metric
loss = T.mean(lasagne.objectives.categorical_crossentropy(output, y_sym))
acc = T.mean(T.eq(pred, y_sym))


# We retrieve all the trainable parameters in our network - a single weight matrix and bias vector
params = lasagne.layers.get_all_params(l_out)
print(params)

# Compute the gradient of the loss function with respect to the parameters.
# The stochastic gradient descent algorithm produces updates for each param
grad = T.grad(loss, params)
updates = lasagne.updates.sgd(grad, params, learning_rate=0.05)
print(updates)


# We define a training function that will compute the loss and accuracy, and take a single optimization step
f_train = theano.function([X_sym, y_sym], [loss, acc], updates=updates)


# The validation function is similar, but does not update the parameters
f_val = theano.function([X_sym, y_sym], [loss, acc])


# The prediction function doesn't require targets, and outputs only the predicted class values
f_predict = theano.function([X_sym], pred)


# We'll choose a batch size, and calculate the number of batches in an "epoch"
# (approximately one pass through the data).
BATCH_SIZE = 64
N_BATCHES = len(X_train) // BATCH_SIZE
N_VAL_BATCHES = len(X_val) // BATCH_SIZE


# Minibatch generators for the training and validation sets
train_batches = batch_gen(X_train, y_train, BATCH_SIZE)
val_batches = batch_gen(X_val, y_val, BATCH_SIZE)

X, y = next(train_batches)

p=X[10]
#imsave('num.'+str(y[0])+'.png',X[10].reshape((28, 28)))

# For each epoch, we call the training function N_BATCHES times,
# accumulating an estimate of the training loss and accuracy.
# Then we do the same thing for the validation set.
# Plotting the ratio of val to train loss can help recognize overfitting.
for epoch in range(3):
    train_loss = 0
    train_acc = 0
    for _ in range(N_BATCHES):
        X, y = next(train_batches)
        loss, acc = f_train(X, y)
        train_loss += loss
        train_acc += acc
    train_loss /= N_BATCHES
    train_acc /= N_BATCHES

    val_loss = 0
    val_acc = 0
    for _ in range(N_VAL_BATCHES):
        X, y = next(val_batches)
        loss, acc = f_val(X, y)
        val_loss += loss
        val_acc += acc
    val_loss /= N_VAL_BATCHES
    val_acc /= N_VAL_BATCHES
    
    print('Epoch {}, Train (val) loss {:.03f} ({:.03f}) ratio {:.03f}'.format(
            epoch, train_loss, val_loss, val_loss/train_loss))
    print('Train (val) accuracy {:.03f} ({:.03f})'.format(train_acc, val_acc))


# We can retrieve the value of the trained weight matrix from the output layer.
# It can be interpreted as a collection of images, one per class
weights = l_out.W.get_value()
print(weights.shape)



get_output = theano.function(inputs = [l_in.input_var],
                             outputs=lasagne.layers.get_output(l_out, deterministic=True))

val_output = get_output(X[0:9])

# The predicted class is just the index of the largest probability in the output
val_predictions = np.argmax(val_output, axis=1)
print 'the prediction is:'+str(val_predictions)


values = lasagne.layers.get_all_param_values(l_out)
pickle.dump(values, open('model.pkl', 'w'))

'''
for i in range(10):
    imsave(str(i)+'.png',weights[:,i].reshape((28, 28)))
'''

f = open('model.pkl', 'rb')
net=pickle.load(f)
f.close()


l_input = lasagne.layers.InputLayer(shape=(None, 784))
l_output = lasagne.layers.DenseLayer(
    l_input,
    num_units=10,
    nonlinearity=lasagne.nonlinearities.softmax)


lasagne.layers.set_all_param_values(l_output, net)

predict = theano.function(inputs = [l_input.input_var],
                             outputs=lasagne.layers.get_output(l_output, deterministic=True))

val_output = predict(X[0:9])
val_predictions = np.argmax(val_output, axis=1)
print 'the prediction is:'+str(val_predictions)
