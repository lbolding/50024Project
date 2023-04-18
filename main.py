import random
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import torch.utils.data
from torchvision import datasets
from torchvision.transforms import ToTensor

#This function does SGD on a set of MNIST data points and ensures that there is always a minority point
#Utilzing the average of the cost functions of a batch
#Generally ninsize will be 1, but this allows for custom ratios
def MNISTconstantEnsure(foursize, ninesize, numrounds):

    #Ensures size is at least 1
    if ninesize < 1:
        ninesize = 1
    if foursize < 1:
        foursize = 1

    #Gets MNIST DATA
    mnist_train = datasets.MNIST(root='data', train=True, transform=ToTensor(), download=True)
    mnist_test = datasets.MNIST(root='data', train=False, transform=ToTensor(), download=True)

    #Target data labels
    indices4 = mnist_train.targets == 4
    indices9 = mnist_train.targets == 9

    #Gets data at target labels
    four_train_data, four_train_labels = mnist_train.data[indices4], mnist_train.targets[indices4]
    nine_train_data, nine_train_labels = mnist_train.data[indices9], mnist_train.targets[indices9]

    #indicies_train = torch.logical_or(indices4, indices9)
    #mnist_train.data, mnist_train.targets = mnist_train.data[indicies_train], mnist_train.targets[indicies_train]

    #Test target indicies
    indices4 = mnist_test.targets == 4
    indices9 = mnist_test.targets == 9

    #Combines to make one large test set
    indicies_test = torch.logical_or(indices4, indices9)
    test_x, test_y = mnist_test.data[indicies_test], mnist_test.targets[indicies_test]

    #Gets MLP model
    model = getTrainModel()

    #Configures optimizer
    opt = tf.keras.optimizers.SGD(learning_rate = 0.001)

    #Loop for number of rounds of SGD
    for __ in range(numrounds):
        #Grabs minibatchs of both labels for training
        b_x_4, b_y_4 = getMiniBatch(four_train_data, four_train_labels, int(foursize), len(four_train_labels))
        b_x_9, b_y_9 = getMiniBatch(nine_train_data, nine_train_labels, int(ninesize), len(nine_train_labels))

        #Combines into one training set
        b_x = tf.Variable(tf.concat([b_x_4, b_x_9], 0))
        b_y = tf.Variable(tf.concat([b_y_4, b_y_9], 0))

        b_x = tf.expand_dims(b_x, axis=1)

        #Binary encoding of training
        b_y = [1 if x == 9 else 0 for x in b_y]
        b_y = np.asarray(b_y).astype('float32').reshape((-1, 1))

        size = ninesize + foursize

        #Constant equal weights normalized to produce mean
        example_weights = tf.ones([size], dtype = tf.float32) / float(size)

        #Gets gradients and updates model
        with tf.GradientTape() as tape:
            output1 = model(b_x)
            loss = tf.reduce_sum(tf.keras.losses.binary_crossentropy(b_y, output1) * example_weights)
            grads = tape.gradient(loss, model.trainable_weights)
        opt.apply_gradients(zip(grads, model.trainable_weights))

    #Tests training model and returns accuracy
    acc = testModel(model, test_x, test_y)
    return acc


#This function does SGD on a set of MNIST data points and creates a large batch from which mini batches are taken
#Utilzing the average of the cost functions of a batch
#Generally ninesize will be 1, but this allows for custom ratios
def MNISTconstantNormal(foursize, ninesize, numrounds):

    #Ensures size is at least 1
    if ninesize < 1:
        ninesize = 1
    if foursize < 1:
        foursize = 1

    #Loads data
    mnist_train = datasets.MNIST(root='data', train=True, transform=ToTensor(), download=True)
    mnist_test = datasets.MNIST(root='data', train=False, transform=ToTensor(), download=True)

    #Target training labels
    indices4 = mnist_train.targets == 4
    indices9 = mnist_train.targets == 9

    #Grab training data with specific labels
    four_train_data, four_train_labels = mnist_train.data[indices4], mnist_train.targets[indices4]
    nine_train_data, nine_train_labels = mnist_train.data[indices9], mnist_train.targets[indices9]

    #Used to calculate how many minority should be involved
    lenfour = int(ninesize * (len(four_train_labels) + 1) / foursize)

    if (lenfour < 1):
        lenfour = 1

    #Randomizes which minority points to be utilizing
    offset = random.randint(0, int(len(nine_train_labels)) - foursize)

    #Combines into one large training set
    nine_train_data, nine_train_labels = nine_train_data[offset:offset+lenfour], nine_train_labels[offset:offset+lenfour]

    train_X = np.concatenate([nine_train_data, four_train_data], 0)
    train_Y = np.concatenate([nine_train_labels, four_train_labels], 0)

    #Gets testing target labels
    indices4 = mnist_test.targets == 4
    indices9 = mnist_test.targets == 9

    #Gets testing data with specified labels
    indicies_test = torch.logical_or(indices4, indices9)
    test_x, test_y = mnist_test.data[indicies_test], mnist_test.targets[indicies_test]

    #Grabs MLP model
    model = getTrainModel()

    #Creates the optimizer
    opt = tf.keras.optimizers.SGD(learning_rate = 0.001)

    #For loop for the number of sgd rounds
    for __ in range(numrounds):

        #Grabs a mini batch of training points from large training set, always of size 100
        b_x, b_y = getMiniBatch(train_X, train_Y, 100, len(train_Y))

        b_x = tf.Variable(tf.expand_dims(b_x, axis=1))

        #Binary encoding for labels
        b_y = tf.Variable([1 if x == 9 else 0 for x in b_y])

        b_y = np.asarray(b_y).astype('float32').reshape((-1, 1))

        #Creates equal weighting for cost avergage
        example_weights = tf.ones([100], dtype = tf.float32) / float(100)

        #Generates gradients from loss function and updates model using optimizer
        with tf.GradientTape() as tape:
            output1 = model(b_x)
            loss = tf.reduce_sum(tf.keras.losses.binary_crossentropy(b_y, output1) * example_weights)
            grads = tape.gradient(loss, model.trainable_weights)
        opt.apply_gradients(zip(grads, model.trainable_weights))

    #Tests model and returns accuracy
    acc = testModel(model, test_x, test_y)
    return acc

#This function does SGD on a set of MNIST data points of which the minority class is to be ensured in each mini batch
#Utilzing random weights to generate overall loss
#Generally ninesize will be 1, but this allows for custom ratios
def MNISTrandomEnsure(foursize, ninesize, numrounds):

    #Ensures size is at least 1
    if ninesize < 1:
        ninesize = 1
    if foursize < 1:
        foursize = 1

    #Downloads data
    mnist_train = datasets.MNIST(root='data', train=True, transform=ToTensor(), download=True)
    mnist_test = datasets.MNIST(root='data', train=False, transform=ToTensor(), download=True)

    #Specifies target labels for training
    indices4 = mnist_train.targets == 4
    indices9 = mnist_train.targets == 9

    #Grabs specified training data
    four_train_data, four_train_labels = mnist_train.data[indices4], mnist_train.targets[indices4]
    nine_train_data, nine_train_labels = mnist_train.data[indices9], mnist_train.targets[indices9]

    #Specifies target labels for testing
    indices4 = mnist_test.targets == 4
    indices9 = mnist_test.targets == 9

    #Grabs total testing data
    indicies_test = torch.logical_or(indices4, indices9)
    test_x, test_y = mnist_test.data[indicies_test], mnist_test.targets[indicies_test]

    #Creates model
    model = getTrainModel()

    #Creates the optimizer
    opt = tf.keras.optimizers.SGD(learning_rate = 0.001)

    #This is the overall batch size
    size = ninesize + foursize

    for __ in range(numrounds):
        #Grabs training mini batch of both classes, ensuring that minority is always represented
        b_x_4, b_y_4 = getMiniBatch(four_train_data, four_train_labels, foursize, len(four_train_labels))
        b_x_9, b_y_9 = getMiniBatch(nine_train_data, nine_train_labels, ninesize, len(nine_train_labels))

        #Combines training batches into 1
        b_x = tf.Variable(tf.concat([b_x_4, b_x_9], 0))
        b_y = tf.Variable(tf.concat([b_y_4, b_y_9], 0))

        b_x = tf.expand_dims(b_x, axis = 1)

        #Binary encoding for labels
        b_y = [1 if x == 9 else 0 for x in b_y]
        b_y = np.asarray(b_y).astype('float32').reshape((-1, 1))

        #Generates random weights for cost of each trainign point towards overall loss
        rand_weights = tf.random.normal([size], mean=0.0, stddev=1.0)
        wp = tf.maximum(rand_weights, 0.0)
        ws = tf.reduce_sum(wp)
        ws += tf.cast(tf.equal(ws, 0.0), float)
        example_weights = wp / ws

        #Generates gradient of the model with respect to the loss and updates model
        with tf.GradientTape() as tape:
            output1 = model(b_x)
            loss = tf.reduce_sum(tf.keras.losses.binary_crossentropy(b_y, output1) * example_weights)
        grads = tape.gradient(loss, model.trainable_weights)
        opt.apply_gradients(zip(grads, model.trainable_weights))

    #Tests model and returns accuracy
    acc = testModel(model, test_x, test_y)
    return acc

#This function does SGD on a set of MNIST data points and creates a large batch from which mini batches are taken
#Utilzing random weights for each cost function towards the overall batch loss
#Generally ninesize will be 1, but this allows for custom ratios
def MNISTrandomNormal(foursize, ninesize, numrounds):

    #Ensures size is at least 1
    if ninesize < 1:
        ninesize = 1
    if foursize < 10:
        foursize = 10

    #Downloads training
    mnist_train = datasets.MNIST(root='data', train=True, transform=ToTensor(), download=True)
    mnist_test = datasets.MNIST(root='data', train=False, transform=ToTensor(), download=True)

    #Specifies labels of training data
    indices4 = mnist_train.targets == 4
    indices9 = mnist_train.targets == 9

    #Grabs specificed labeled training data
    four_train_data, four_train_labels = mnist_train.data[indices4], mnist_train.targets[indices4]
    nine_train_data, nine_train_labels = mnist_train.data[indices9], mnist_train.targets[indices9]

    #Calculates the amount of minority class to include
    lenfour = int(ninesize * (len(four_train_labels) + 1) / foursize)

    if (lenfour < 1):
        lenfour = 1

    #Generates random points to ensure random minority class points
    offset = random.randint(0, int(len(nine_train_labels)) - foursize)

    nine_train_data, nine_train_labels = nine_train_data[offset:offset+lenfour], nine_train_labels[offset:offset+lenfour]

    #Combines into one large training set
    train_X = np.concatenate([nine_train_data, four_train_data], 0)
    train_Y = np.concatenate([nine_train_labels, four_train_labels], 0)

    #Specifies target testing labels
    indices4 = mnist_test.targets == 4
    indices9 = mnist_test.targets == 9

    #Grabs testing data from labels
    indicies_test = torch.logical_or(indices4, indices9)
    test_x, test_y = mnist_test.data[indicies_test], mnist_test.targets[indicies_test]

    #Grabs the MLP model
    model = getTrainModel()

    #Creates the optimizer
    opt = tf.keras.optimizers.SGD(learning_rate = 0.001)

    #Runs SGD for numrounds
    for __ in range(numrounds):
        #Grabs minibatch of size 100 from training
        b_x, b_y = getMiniBatch(train_X, train_Y, 100, len(train_Y))

        b_x = tf.Variable(tf.expand_dims(b_x, axis = 1))

        #Binary encoding for labels
        b_y = tf.Variable([1 if x == 9 else 0 for x in b_y])
        b_y = np.asarray(b_y).astype('float32').reshape((-1, 1))

        #Generates random weights for each cost towards the overall loss
        rand_weights = tf.random.normal([100], mean=0.0, stddev=1.0)
        wp = tf.maximum(rand_weights, 0.0)
        ws = tf.reduce_sum(wp)
        ws += tf.cast(tf.equal(ws, 0.0), float)
        example_weights = wp / ws

        #Calculates model gradient with respect to total loss and updates using optimizer
        with tf.GradientTape() as tape:
            output1 = model(b_x)
            loss = tf.reduce_sum(tf.keras.losses.binary_crossentropy(b_y, output1) * example_weights)
        grads = tape.gradient(loss, model.trainable_weights)
        opt.apply_gradients(zip(grads, model.trainable_weights))

    #Tests model and returns accuracy
    acc = testModel(model, test_x, test_y)
    return acc

def MNISTautomatic(foursize, ninesize, numrounds, valsize):
    mnist_train = datasets.MNIST(root='data', train=True, transform=ToTensor(), download=True)
    mnist_test = datasets.MNIST(root='data', train=False, transform=ToTensor(), download=True)
    mnist_val = datasets.MNIST(root = 'data', train = True, transform = ToTensor())

    indices4 = mnist_train.targets == 4
    indices9 = mnist_train.targets == 9

    four_train_data, four_train_labels = mnist_train.data[indices4], mnist_train.targets[indices4]
    nine_train_data, nine_train_labels = mnist_train.data[indices9], mnist_train.targets[indices9]

    indices4 = mnist_test.targets == 4
    indices9 = mnist_test.targets == 9
    indicies_test = torch.logical_or(indices4, indices9)
    test_X, test_Y = mnist_test.data[indicies_test], mnist_test.targets[indicies_test]

    indices4 = mnist_val.targets == 4
    indices9 = mnist_val.targets == 9
    indicies_val = torch.logical_or(indices4, indices9)
    val_X, val_Y = mnist_val.data[indicies_val], mnist_val.targets[indicies_val]

    #loaders = {
    #    'train': torch.utils.data.DataLoader(mnist_train, batch_size=100, shuffle=True, num_workers=1),
    #    'test': torch.utils.data.DataLoader(mnist_test, batch_size=5, shuffle=True, num_workers=1),
    #    'val': torch.utils.data.DataLoader(mnist_val, batch_size=5, shuffle=True, num_workers=1),
    #}

    model = getTrainModel()

    optimizer = tf.keras.optimizers.SGD(learning_rate = 0.001)

    for __ in range(numrounds):
        b_x_4, b_y_4 = getMiniBatch(four_train_data, four_train_labels, foursize, len(four_train_labels))
        b_x_9, b_y_9 = getMiniBatch(nine_train_data, nine_train_labels, ninesize, len(nine_train_labels))

        b_x = tf.Variable(tf.concat([b_x_4, b_x_9], 0))
        b_y = tf.Variable(tf.concat([b_y_4, b_y_9], 0))

        b_x = tf.expand_dims(b_x, axis=1)

        b_y = tf.Variable([1 if x == 9 else 0 for x in b_y])

        b_val_x, b_val_y = getMiniBatch(val_X, val_Y, valsize, len(val_Y))

        b_val_x = tf.expand_dims(b_val_x, axis=1)
        print(b_val_x.shape)
        #print(b_val_x.shape)
        b_val_y = tf.Variable([1 if x == 9 else 0 for x in b_val_y])

        with tf.GradientTape() as tape2:
            ex_wts_a = tf.zeros(tf.shape(b_y), dtype=tf.float32)
            ex_wts_b = tf.ones(tf.shape(b_val_y), dtype=tf.float32) / float(tf.size(b_val_y))
            tape2.watch(ex_wts_a)
            with tf.GradientTape() as tape1:
                w_dict, loss_a, logits_a = getMNIST(b_x, b_y, ex_wts=ex_wts_a, is_training=True)
                var_names = w_dict.keys()
                var_list = [w_dict[kk] for kk in var_names]
            grads = tape1.gradient(loss_a, var_list)

            var_list_new = [vv - gg for gg, vv in zip(grads, var_list)]
            w_dict_new = dict(zip(var_names, var_list_new))
            _, loss_b, logits_b = getMNIST(b_val_x, b_val_y, ex_wts=ex_wts_b, is_training=True, w_dict=w_dict_new)

        grads_ex_weights = tape2.gradient(loss_b, [ex_wts_a])
        grads_ex_weights = np.array(grads_ex_weights)
        ex_weight = -grads_ex_weights
        ex_weight_plus = tf.maximum(ex_weight, 0.0)
        ex_weight_sum = tf.reduce_sum(ex_weight_plus)
        ex_weight_sum += tf.cast(tf.equal(ex_weight_sum, 0.0), tf.float32)
        ex_weights_norm = ex_weight_plus / ex_weight_sum

        with tf.GradientTape() as tape3:
            training_out = model(b_x, training=True)
            b_y = np.asarray(b_y).astype('float32').reshape((-1, 1))
            loss_training = tf.reduce_sum(tf.keras.losses.binary_crossentropy(b_y, training_out) * ex_weights_norm)
            #loss_training = tf.reduce_sum(
            #    tf.transpose(
            #        tf.nn.sigmoid_cross_entropy_with_logits(logits=training_out, labels=b_y)) * ex_weights_norm)
        grad_train = tape3.gradient(loss_training, model.trainable_variables)
        optimizer.apply_gradients(zip(grad_train, model.trainable_variables))
        print(loss_training)
    #print("done")

    acc = testModel(model, test_X, test_Y)
    return acc


#This function does SGD on a set of MNIST data points and ensures minority class always represented in minibatch
#Utilzing backward on backward automatic differentiation to generate cost weights
#Generally ninesize will be 1, but this allows for custom ratios
def MNISTautomaticEnsure(foursize, ninesize, numrounds, valsize):
    #Ensures size is at least 1
    if ninesize < 1:
        ninesize = 1
    if foursize < 1:
        foursize = 1

    #Downloads data
    mnist_train = datasets.MNIST(root='data', train=True, transform=ToTensor(), download=True)
    mnist_test = datasets.MNIST(root='data', train=False, transform=ToTensor(), download=True)

    #Specifies target training labels
    indices4 = mnist_train.targets == 4
    indices9 = mnist_train.targets == 9

    #Grabs target training data
    four_train_data, four_train_labels = mnist_train.data[indices4], mnist_train.targets[indices4]
    nine_train_data, nine_train_labels = mnist_train.data[indices9], mnist_train.targets[indices9]

    #Specifies target testing labels
    indices4 = mnist_test.targets == 4
    indices9 = mnist_test.targets == 9

    #Grabs target testing data and combiens
    indicies_test = torch.logical_or(indices4, indices9)
    test_X, test_Y = mnist_test.data[indicies_test], mnist_test.targets[indicies_test]

    #Grabs model
    model = getTrainModel()

    #Creates optimizer
    optimizer = tf.keras.optimizers.SGD(learning_rate = 0.001)

    #Runs SGD for numrounds
    for __ in range(numrounds):
        #Grabs minibatch of both classes to ensure minority represented
        b_x_4, b_y_4 = getMiniBatch(four_train_data, four_train_labels, foursize, len(four_train_labels))
        b_x_9, b_y_9 = getMiniBatch(nine_train_data, nine_train_labels, ninesize, len(nine_train_labels))

        #Combines into total training minibatch
        b_x = tf.Variable(tf.concat([b_x_4, b_x_9], 0))
        b_y = tf.Variable(tf.concat([b_y_4, b_y_9], 0))

        b_x = tf.expand_dims(b_x, axis=1)

        #Binary encoding of labels
        b_y = tf.Variable([1 if x == 9 else 0 for x in b_y])

        #Grabs minibatch of each class for validation
        #Same size
        b_val_x_4, b_val_y_4 = getMiniBatch(four_train_data, four_train_labels, int(valsize/2), len(four_train_labels))
        b_val_x_9, b_val_y_9 = getMiniBatch(nine_train_data, nine_train_labels, int(valsize/2), len(nine_train_labels))

        #Combines into total validation minibatch
        b_val_x = tf.Variable(tf.concat([b_val_x_4, b_val_x_9], 0))
        b_val_y = tf.Variable(tf.concat([b_val_y_4, b_val_y_9], 0))

        b_val_x = tf.expand_dims(b_val_x, axis=1)

        #Binary encoding for labels
        b_val_y = tf.Variable([1 if x == 9 else 0 for x in b_val_y])

        with tf.GradientTape() as tape2:
            #Initializes perturbing variable to be zero
            ex_wts_a = tf.zeros(tf.shape(b_y), dtype=tf.float32)
            #Creates weights for averaging cost
            ex_wts_b = tf.ones(tf.shape(b_val_y), dtype=tf.float32) / float(tf.size(b_val_y))
            tape2.watch(ex_wts_a)

            #Uses backward on backward AD to generate the gradient of perturbing variable
            with tf.GradientTape() as tape1:
                w_dict, loss_a, logits_a = getMNIST(b_x, b_y, ex_wts=ex_wts_a, is_training=True)
                var_names = w_dict.keys()
                var_list = [w_dict[kk] for kk in var_names]
            grads = tape1.gradient(loss_a, var_list)

            var_list_new = [vv - gg for gg, vv in zip(grads, var_list)]
            w_dict_new = dict(zip(var_names, var_list_new))
            _, loss_b, logits_b = getMNIST(b_val_x, b_val_y, ex_wts=ex_wts_b, is_training=True, w_dict=w_dict_new)

        #Normalizes perturbing variable gradient to create weighting of costs
        grads_ex_weights = tape2.gradient(loss_b, [ex_wts_a])
        grads_ex_weights = np.array(grads_ex_weights)
        ex_weight = -grads_ex_weights
        ex_weight_plus = tf.maximum(ex_weight, 0.0)
        ex_weight_sum = tf.reduce_sum(ex_weight_plus)
        ex_weight_sum += tf.cast(tf.equal(ex_weight_sum, 0.0), tf.float32)
        ex_weights_norm = ex_weight_plus / ex_weight_sum

        #Calculates gradient of model with respect to loss and updates model
        with tf.GradientTape() as tape3:
            training_out = model(b_x, training=True)
            b_y = np.asarray(b_y).astype('float32').reshape((-1, 1))
            loss_training = tf.reduce_sum(tf.keras.losses.binary_crossentropy(b_y, training_out) * ex_weights_norm)
        grad_train = tape3.gradient(loss_training, model.trainable_variables)
        optimizer.apply_gradients(zip(grad_train, model.trainable_variables))

    #Tests model and returns accuracy
    acc = testModel(model, test_X, test_Y)
    return acc

#This function does SGD on a set of MNIST data points and creates large trainign set where minibatches are taken
#Utilzing backward on backward automatic differentiation to generate cost weights
#Generally ninesize will be 1, but this allows for custom ratios
def MNISTautomaticNormal(foursize, ninesize, numrounds, valsize):
    #Makes sure size is at least 1
    if ninesize < 1:
        ninesize = 1
    if foursize < 10:
        foursize = 10

    #Downloads data
    mnist_train = datasets.MNIST(root='data', train=True, transform=ToTensor(), download=True)
    mnist_test = datasets.MNIST(root='data', train=False, transform=ToTensor(), download=True)

    #Specifies training labels
    indices4 = mnist_train.targets == 4
    indices9 = mnist_train.targets == 9

    #Grabs training data
    four_train_data, four_train_labels = mnist_train.data[indices4], mnist_train.targets[indices4]
    nine_train_data, nine_train_labels = mnist_train.data[indices9], mnist_train.targets[indices9]

    #Used to calculate number of minority to be included
    lenfour = int(ninesize*(len(four_train_labels) + 1) / foursize)

    if (lenfour < 1):
        lenfour = 1

    #Generates random offset to grab minority data
    offset = random.randint(0, int(len(nine_train_labels)) - foursize)

    nine_train_data, nine_train_labels = nine_train_data[offset:lenfour+offset], nine_train_labels[offset:lenfour+offset]

    #Combines two classes into one large training data set
    train_X = np.concatenate([nine_train_data, four_train_data], 0)
    train_Y = np.concatenate([nine_train_labels, four_train_labels], 0)

    #Specifies target testing labels
    indices4 = mnist_test.targets == 4
    indices9 = mnist_test.targets == 9

    #Grabs target testing data
    indicies_test = torch.logical_or(indices4, indices9)
    test_X, test_Y = mnist_test.data[indicies_test], mnist_test.targets[indicies_test]

    #Grabs model
    model = getTrainModel()

    #Creates optimizer
    optimizer = tf.keras.optimizers.SGD(learning_rate = 0.001)

    #SGD runs for numrounds
    for __ in range(numrounds):
                #Grabs minibatch of training data
                b_x, b_y = getMiniBatch(train_X, train_Y, 100, len(train_Y))
                b_x = tf.Variable(tf.expand_dims(b_x, axis=1))

                #Binary encoding for labels
                b_y = tf.Variable([1 if x == 9 else 0 for x in b_y])

                #Grabs minibatches for validation data
                #Both classes equally represented
                b_val_x_4, b_val_y_4 = getMiniBatch(four_train_data, four_train_labels, int(valsize/2), len(four_train_labels))
                b_val_x_9, b_val_y_9 = getMiniBatch(nine_train_data, nine_train_labels, int(valsize/2), len(nine_train_labels))

                #Combines validation data
                b_val_x = tf.Variable(tf.concat([b_val_x_4, b_val_x_9], 0))
                b_val_y = tf.Variable(tf.concat([b_val_y_4, b_val_y_9], 0))

                b_val_x = tf.expand_dims(b_val_x, axis=1)

                #Binary encoding of labels
                b_val_y = tf.Variable([1 if x == 9 else 0 for x in b_val_y])

                with tf.GradientTape() as tape2:
                    #Initializes perturbing variable to be 0
                    ex_wts_a = tf.zeros(tf.shape(b_y), dtype=tf.float32)
                    #Initializes weight variable for average cost
                    ex_wts_b = tf.ones(tf.shape(b_val_y), dtype=tf.float32) / float(tf.size(b_val_y))
                    tape2.watch(ex_wts_a)
                    # Uses backward on backward AD to generate the gradient of perturbing variable
                    with tf.GradientTape() as tape1:
                        w_dict, loss_a, logits_a = getMNIST(b_x, b_y, ex_wts=ex_wts_a, is_training=True)
                        var_names = w_dict.keys()
                        var_list = [w_dict[kk] for kk in var_names]
                    grads = tape1.gradient(loss_a, var_list)

                    var_list_new = [vv - gg for gg, vv in zip(grads, var_list)]
                    w_dict_new = dict(zip(var_names, var_list_new))
                    _, loss_b, logits_b = getMNIST(b_val_x, b_val_y, ex_wts=ex_wts_b, is_training=True, w_dict=w_dict_new)

                #Normalizes gradient of perturbing variable to generate cost weights
                grads_ex_weights = tape2.gradient(loss_b, [ex_wts_a])
                grads_ex_weights = np.array(grads_ex_weights)
                ex_weight = -grads_ex_weights
                ex_weight_plus = tf.maximum(ex_weight, 0.0)
                ex_weight_sum = tf.reduce_sum(ex_weight_plus)
                ex_weight_sum += tf.cast(tf.equal(ex_weight_sum, 0.0), tf.float32)
                ex_weights_norm = ex_weight_plus / ex_weight_sum

                #Calculates gradient of model with respect to loss and updates model weights
                with tf.GradientTape() as tape3:
                    training_out = model(b_x, training=True)
                    b_y = np.asarray(b_y).astype('float32').reshape((-1, 1))
                    loss_training = tf.reduce_sum(tf.keras.losses.binary_crossentropy(b_y, training_out) * ex_weights_norm)
                grad_train = tape3.gradient(loss_training, model.trainable_variables)
                optimizer.apply_gradients(zip(grad_train, model.trainable_variables))

    #Tests model and returns accuracy
    acc = testModel(model, test_X, test_Y)
    return acc

#This function is an expanded form of the model, such that we can use layerwise
#pre and post activations to calculate gradients with respect to the layers
#This is an updated function for tensorflow 2.0

def getMNIST(inputs, labels, is_training = True, dtype = tf.float32, w_dict = None, ex_wts = None):

    #Checks if we have a weight dictionary
    if w_dict is None:
        w_dict = {}

    #This function returns weight of layers and may creates dictionary to store them
    def _get_var(name, shape, dtype, initializer):
        if name in w_dict:
            return w_dict[name]
        else:
            var = tf.Variable(initializer(shape=shape, dtype = dtype))
            w_dict[name] = var
            return var

    #Changes shape of inputs
    inputs_ = tf.cast(tf.reshape(inputs, [-1, 28, 28, 1]), dtype)
    labels = tf.cast(labels, dtype)

    #Uses to create weights of model
    w_init = tf.keras.initializers.truncated_normal(stddev=0.1)

    #Grabs weight of each layer of model
    w1 = _get_var('w1', [5, 5, 1, 16], dtype, initializer=w_init)
    w2 = _get_var('w2', [5, 5, 16, 32], dtype, initializer=w_init)
    w3 = _get_var('w3', [5, 5, 32, 64], dtype, initializer=w_init)
    w4 = _get_var('w4', [1024, 100], dtype, initializer=w_init)
    w5 = _get_var('w5', [100, 1], dtype, initializer=w_init)

    b_init = tf.constant_initializer(0.0)

    b1 = _get_var('b1', [16], dtype, initializer=b_init)
    b2 = _get_var('b2', [32], dtype, initializer=b_init)
    b3 = _get_var('b3', [64], dtype, initializer=b_init)
    b4 = _get_var('b4', [100], dtype, initializer=b_init)
    b5 = _get_var('b5', [1], dtype, initializer=b_init)

    #Creates activation of model
    act = tf.nn.relu

    #Creates the layers of the model and calculates their activations
    l0 = tf.identity(inputs_, name='l0')
    z1 = tf.add(tf.nn.conv2d(inputs_, w1, [1, 1, 1, 1], 'SAME'), b1, name='z1')
    l1 = act(tf.nn.max_pool(z1, [1, 3, 3, 1], [1, 2, 2, 1], 'SAME'), name='l1')

    z2 = tf.add(tf.nn.conv2d(l1, w2, [1, 1, 1, 1], 'SAME'), b2, name='z2')
    l2 = act(tf.nn.max_pool(z2, [1, 3, 3, 1], [1, 2, 2, 1], 'SAME'), name='l2')

    z3 = tf.add(tf.nn.conv2d(l2, w3, [1, 1, 1, 1], 'SAME'), b3, name='z3')
    l3 = act(tf.nn.max_pool(z3, [1, 3, 3, 1], [1, 2, 2, 1], 'SAME'), name='l3')

    z4 = tf.add(tf.matmul(tf.reshape(l3, [-1, 1024]), w4), b4, name='z4')
    l4 = act(z4, name='l4')

    z5 = tf.add(tf.matmul(l4, w5), b5, name='z5')

    logits = tf.squeeze(z5)

    #Calculates the loss of the input data through the model
    if ex_wts is None:
        loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))
    else:
        loss = tf.reduce_sum(
            tf.nn.sigmoid_cross_entropy_with_logits(logits = logits, labels = labels) * ex_wts)

    #Returns the weight dictionary, the loss, and the predicted labels
    return w_dict, loss, logits

#def getMNISTOLD(inputs, labels, is_training = True, dtype = tf.float32, w_dict = None, ex_wts = None):
#    if w_dict is None:
#        w_dict = {}
#
#    def _get_var(name, shape, dtype, initializer):
#        key = tf.compat.v1.get_variable_scope().name + '/' + name
#        if key in w_dict:
#            return w_dict[key]
#        else:
#            var = tf.compat.v1.get_variable(name, shape, dtype, initializer=initializer)
#            w_dict[key] = var
#            return var
#
#    with tf.compat.v1.variable_scope('Model', reuse = True):
#        inputs_ = tf.cast(tf.reshape(inputs, [-1, 28, 28, 1]), dtype)
#        labels = tf.cast(labels, dtype)
#
#        w_init = tf.compat.v1.truncated_normal_initializer(stddev=0.1)
#        w1 = _get_var('w1', [5, 5, 1, 16], dtype, initializer=w_init)
#        w2 = _get_var('w2', [5, 5, 16, 32], dtype, initializer=w_init)
#        w3 = _get_var('w3', [5, 5, 32, 64], dtype, initializer=w_init)
#        w4 = _get_var('w4', [1024, 100], dtype, initializer=w_init)
#        w5 = _get_var('w5', [100, 1], dtype, initializer=w_init)
#
#        b_init = tf.constant_initializer(0.0)
#
#        b1 = _get_var('b1', [16], dtype, initializer=b_init)
#        b2 = _get_var('b2', [32], dtype, initializer=b_init)
#        b3 = _get_var('b3', [64], dtype, initializer=b_init)
#        b4 = _get_var('b4', [100], dtype, initializer=b_init)
#        b5 = _get_var('b5', [1], dtype, initializer=b_init)

 #       act = tf.nn.relu

#        l0 = tf.identity(inputs_, name='l0')
#        z1 = tf.add(tf.nn.conv2d(inputs_, w1, [1, 1, 1, 1], 'SAME'), b1, name='z1')
#        l1 = act(tf.nn.max_pool(z1, [1, 3, 3, 1], [1, 2, 2, 1], 'SAME'), name='l1')

        # Conv-2
#        z2 = tf.add(tf.nn.conv2d(l1, w2, [1, 1, 1, 1], 'SAME'), b2, name='z2')
#        l2 = act(tf.nn.max_pool(z2, [1, 3, 3, 1], [1, 2, 2, 1], 'SAME'), name='l2')

        # Conv-3
#        z3 = tf.add(tf.nn.conv2d(l2, w3, [1, 1, 1, 1], 'SAME'), b3, name='z3')
#        l3 = act(tf.nn.max_pool(z3, [1, 3, 3, 1], [1, 2, 2, 1], 'SAME'), name='l3')

        # FC-4
#        z4 = tf.add(tf.matmul(tf.reshape(l3, [-1, 1024]), w4), b4, name='z4')
#        l4 = act(z4, name='l4')

        # FC-5
#        z5 = tf.add(tf.matmul(l4, w5), b5, name='z5')

#        logits = tf.squeeze(z5)

#        if ex_wts is None:
            # Average loss.
#            loss = tf.reduce_mean(
 #               tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))
 #       else:
            # Weighted loss.
 #           loss = tf.reduce_sum(
 #               tf.nn.sigmoid_cross_entropy_with_logits(logits = logits, labels = labels) * ex_wts)

 #   return w_dict, loss, logits

#This function just grabs a minibatch from the overall data set
def getMiniBatch(X, y, batchSize, length):
    a = np.random.randint(low = 0, high = length-1, size = batchSize)
    X_batch = X[a]
    y_batch = y[a]
    return X_batch, y_batch

#This function returns the MLP
def getTrainModel():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Permute((2, 3, 1), input_shape=(1, 28, 28)))

    model.add(tf.keras.layers.Convolution2D(16, kernel_size=(5, 5), strides=(1, 1), input_shape=(28, 28, 1)))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Convolution2D(32, kernel_size=(5, 5), strides=(1, 1)))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
    return model

#This function tests the accuracy of the model
def testModel(model, X, y):
    b_x = tf.expand_dims(X, axis=1)
    b_y = [1 if x == 9 else 0 for x in y]
    b_y = np.asarray(b_y).astype('float32').reshape((-1, 1))
    pred_y = model(b_x)
    pred_y = [1 if x > 0.5 else 0 for x in pred_y]
    total = len(pred_y)
    correct = 0
    for i in range(total):
        if (pred_y[i] == b_y[i]):
            correct += 1
    return correct/total

def NormalCompare():
    foursize = np.linspace(100, 200, 6)
    allCon = []
    avgCon = []
    allRan = []
    avgRan = []
    allAD = []
    avgAD = []
    bsize = []

    for size in foursize:
        conTotal = 0
        ranTotal = 0
        adTotal = 0
        for _ in range(5):
            bsize.append(size)
            con = MNISTconstantNormal(int(size), 1, 8000)
            conTotal += con
            allCon.append((1 - con) * 100)
            ran = MNISTrandomNormal(int(size), 1, 8000)
            ranTotal += ran
            allRan.append((1 - ran) * 100)
            AD = MNISTautomaticNormal(int(size), 1, 8000, 10)
            adTotal += AD
            allAD.append((1 - AD) * 100)

        conTotal = conTotal / 5
        ranTotal = ranTotal / 5
        adTotal = adTotal / 5
        avgCon.append((1 - conTotal) * 100)
        avgRan.append((1 - ranTotal) * 100)
        avgAD.append((1 - adTotal) * 100)

    plt.scatter(bsize, allCon, color='r', s=25)
    plt.scatter(bsize, allRan, color='g', s=25)
    plt.scatter(bsize, allAD, color='b', s=25)
    plt.legend(["Constant", "Random", "Reweighted"])
    plt.plot(foursize, avgCon, color='r', marker=".", markersize=7)
    plt.plot(foursize, avgRan, color='g', marker=".", markersize=7)
    plt.plot(foursize, avgAD, color='b', marker=".", markersize=7)
    plt.xlabel("Majority Class Ratio X:1")
    plt.ylabel("Test Error %")
    plt.grid()
    plt.show()

def EnsureCompare():
    foursize = np.linspace(100, 200, 6)
    allCon = []
    avgCon = []
    allRan = []
    avgRan = []
    allAD = []
    avgAD = []
    bsize = []

    for size in foursize:
        conTotal = 0
        ranTotal = 0
        adTotal = 0
        for _ in range(5):
            bsize.append(size)
            con = MNISTconstantEnsure(int(size), 1, 8000)
            conTotal += con
            allCon.append((1-con) * 100)
            ran = MNISTrandomEnsure(int(size), 1, 8000)
            ranTotal += ran
            allRan.append((1-ran) * 100)
            AD = MNISTautomaticEnsure(int(size), 1, 8000, 10)
            adTotal += AD
            allAD.append((1-AD) * 100)

        conTotal = conTotal / 5
        ranTotal = ranTotal / 5
        adTotal = adTotal / 5
        avgCon.append((1-conTotal) * 100)
        avgRan.append((1-ranTotal) * 100)
        avgAD.append((1-adTotal) * 100)

    plt.scatter(bsize, allCon, color='r', s=25)
    plt.scatter(bsize, allRan, color='g', s=25)
    plt.scatter(bsize, allAD, color='b', s=25)
    plt.legend(["Constant", "Random", "Reweighted"])
    plt.plot(foursize, avgCon, color = 'r', marker=".", markersize = 7)
    plt.plot(foursize, avgRan, color = 'g', marker=".", markersize = 7)
    plt.plot(foursize, avgAD, color = 'b', marker=".", markersize = 7)
    plt.xlabel("Majority Class Ratio X:1")
    plt.ylabel("Test Error %")
    plt.grid()
    plt.show()

def testValSize():
    valsize = np.linspace(4, 20, 5)

    accAD = []
    avgAD = []
    vsize = []

    for size in valsize:
        total = 0
        for _ in range(5):
            vsize.append(size)
            AD = MNISTautomaticEnsure(150, 1, 4000, int(size))
            accAD.append((1 - AD) * 100)
            print(AD)
            total += AD
        total = total / 5
        avgAD.append((1-total) * 100)

    plt.scatter(vsize, accAD, color='r', s=25)
    plt.plot(valsize, avgAD,  marker=".", color = 'r', markersize = 5)
    plt.xlabel("Validation MiniBatch Size")
    plt.ylabel("Test error %")
    plt.show()

if __name__ == '__main__':
    NormalCompare()
    EnsureCompare()
    testValSize()



