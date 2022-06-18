from tensorflow import keras
import matplotlib.pyplot as plt

def preprocess(location, res, b_size, isTanh):

    dataset = keras.preprocessing.image_dataset_from_directory(
        location,
        label_mode=None,
        image_size=(res, res),
        batch_size=b_size)

    if isTanh:

        dataset = dataset.map(lambda x: x / 127.5 - 1.0) #normalize data to -1 - 1

    else:

        dataset = dataset.map(lambda x: x / 255.) #normaliye data to 0 - 1

    return dataset

#show sample from dataset
def showSample(data):

    for x in data:

        plt.axis("off")

        plt.imshow(((x.numpy() + 1) * 127.5).astype("int32")[8])

        break

# custom preprocessing for imagedatagenerator to normalize data to -1 - 1 for tanh activation
def tanh_preprocess(i):

    i = i / 127.5 - 1.0

    return i
