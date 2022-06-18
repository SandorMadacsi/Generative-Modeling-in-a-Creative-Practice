import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import os, datetime

#creates directory of the given models output
def create_dir(location: str,
              name: str
              ):

    dir = f"{name}"
    parent = location
    out_dir = os.path.join(parent,dir)

    if not os.path.exists(out_dir):

        os.mkdir(out_dir)

    return out_dir

#Plot grid of images during training
def plot_images(images,
                freq,
                epoch,
                dir
                ):

    date = datetime.datetime.now().strftime("%m%d%H%M")

    num_images = int(np.sqrt(images.shape[0]))

    fig = plt.figure(figsize=(16, 16))#size of grid
    fig.patch.set_facecolor('xkcd:black')#black background for grid


    for i in range(images.shape[0]):

        img = keras.preprocessing.image.array_to_img(images[i])
        plt.subplot(num_images, num_images, i + 1)#orders image in grid
        plt.imshow(img)#show image
        plt.axis('off')
    #saves grid
    if (epoch + 1) % freq == 0 or epoch == 0:

        plt.savefig(dir + '/' + date+'_image_at_epoch_{:04d}.png'.format(epoch))

    plt.show()#show grid


#Used to display and save trained model outputs
def plot_results(images,dir,iteration):

    #date is necessary to not overwrite previous images
    date = datetime.datetime.now().strftime("%m%d%H%M")
    num_images = int(np.sqrt(images.shape[0]))

    fig = plt.figure(figsize=(16, 16)) #size of grid
    fig.patch.set_facecolor('xkcd:black') #black background for grid

    # Saves individual images from a grid of images
    for i in range(images.shape[0]):

        img = keras.preprocessing.image.array_to_img(images[i])
        img.save(dir+ date + f"{iteration+i}.png") #save image
        plt.subplot(num_images, num_images, i + 1) #orders image in grid
        plt.imshow(img) #show image
        plt.axis('off')

    plt.show() # show grid

