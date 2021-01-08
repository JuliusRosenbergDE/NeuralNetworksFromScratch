from classes import *

# test of the mnist dataset


# os.chdir('C:/Users/Julius/Desktop/Python/nnfs')
os.chdir('./')

# load a mnist dataset


def load_data_mnist(dataset, path):

    # scan all the directories and create a list of labels
    labels = os.listdir(os.path.join(path, dataset))
    # create lists for samples and labels
    X = []
    y = []
    # for each label folder
    # and for each image in given folder
    # read the image
    for label in labels:
        for file in os.listdir(os.path.join(path, dataset, label)):
            image = cv2.imread(file, cv2.IMREAD_UNCHANGED)
            image = cv2.imread(os.path.join(
                path, dataset, label, file), cv2.IMREAD_UNCHANGED)

            # append file and label to the lists
            X.append(image)
            y.append(label)

    # convert the data to a proper numpy array
    return np.array(X), np.array(y).astype('uint8')


def create_data_mnist(path):

    X, y = load_data_mnist('train', path)
    X_test, y_test = load_data_mnist('test', path)

    return X, y, X_test, y_test


# Create dataset
X, y, X_test, y_test = create_data_mnist('fashion_mnist_images')

# Shuffle the training dataset
keys = np.array(range(X.shape[0]))
np.random.shuffle(keys)
X = X[keys]
y = y[keys]

# Scale and reshape samples
X = (X.reshape(X.shape[0], -1).astype(np.float32) - 127.5) / 127.5
X_test = (X_test.reshape(X_test.shape[0], -1).astype(np.float32) -
          127.5) / 127.5


# instantiate the model
model = Model()
"""
model has:
2 hidden layers
    64 neurons
    ReLU activation function
    no regularization
outputlayer with 10 neurons
    softmax activation function
categorical crossentropy loss
adam optimizer

"""

# add layers
model.add(Layer_Dense(X.shape[1], 64, reg_l2=5e-4))
model.add(Activation_ReLU())
model.add(Layer_Dense(64, 64))
model.add(Activation_ReLU())
model.add(Layer_Dropout(0.2))
model.add(Layer_Dense(64, 10))
model.add(Activation_Softmax())

# set loss optimizer and accuracy objects
model.set(
    optimizer=Optimizer_AdamW(decay=5e-5),
    loss=Loss_CategoricalCrossentropy(),
    accuracy=Accuracy_Classification()
)

# finalize
model.finalize()

# train!
model.train(X, y, validation_data=(X_test, y_test),
            epochs=5, batch_size=128, print_every=100)
