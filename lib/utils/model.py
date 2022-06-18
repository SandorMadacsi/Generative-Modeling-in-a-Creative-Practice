# %%
from tensorflow import keras
import tensorflow as tf
from keras import layers

class ModelSettings:

    def __init__(
            self,
            model_name,
            num_conv_layers: int = 2,
            img_shape: tuple = (28, 28, 1),
            dim: int = 7,
            gen_filter: int = 7,
            disc_filter: int = 7,
            kernel: int = 4,
            stride: int = 2,
            is_batchnorm: bool = False,
            is_multi_drop: bool = False,
            output: int = 1,
            latent_dim: int = 100,
            filter_mode: int = 1,
            is_tanh: bool = True,
            **kwargs
    ):
        self.disc = None #discriminator
        self.gen = None #generator
        self.initWeights = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02, seed=None) #weight initialization
        self.model_name = model_name # model name
        self.num_conv_layers = num_conv_layers # number of convolutional layers
        self.img_shape = img_shape # image shape
        self.dim = dim # initial dimention (resolution)
        self.kernel = kernel # kernel size
        self.stride = stride # stride
        self.output = output # number of output channels
        self.latent_dim = latent_dim # latent dimension
        self.gen_filter = gen_filter # initial generator filter
        self.disc_filter = disc_filter # initial discriminator filter
        self.is_batchnorm = is_batchnorm # boolean option for batch normalization
        self.is_multi_drop = is_multi_drop # multi dropout layers or single
        self.filter_mode = filter_mode # filter modes
        self.is_tanh = is_tanh # option for tanh  or sigmoid activation



def build_discriminator(settings: ModelSettings):
    conv2d = [] #convolutional layer array

    for i in range(settings.num_conv_layers - 2):
        #same filter mode
        if settings.filter_mode == 3:
            disc_filter = settings.disc_filter
        else:
            disc_filter = settings.disc_filter + i #on default discriminator model has an ascending architecture of filters

        conv2d.append(
            layers.Conv2D(pow(2, disc_filter), settings.kernel, strides=settings.stride, padding="same",
                          kernel_initializer=settings.initWeights
                          )) #convolutional layers

    leak = [layers.LeakyReLU() for _ in range(settings.num_conv_layers - 1)] #leakyRELU  layers
    if settings.is_multi_drop:
        drop = [layers.Dropout(0.25) for _ in range(settings.num_conv_layers - 1)] #dropout layers
    if settings.is_batchnorm:
        batch = [layers.BatchNormalization() for _ in range(settings.num_conv_layers - 1)] # batchnorm layers

    disc_input = keras.Input(shape=settings.img_shape) #input

    if settings.filter_mode == 3:
        disc_filter = settings.disc_filter #filters are the same in filter mode 3
    else:
        disc_filter = settings.disc_filter - 1 #on default reducing the initial filter to accomodate the ascending setup

    #first stack of layers
    x = layers.Conv2D(pow(2, disc_filter), kernel_size=settings.kernel, strides=settings.stride,
                      padding="same", kernel_initializer=settings.initWeights
                      )(disc_input) # first convolutional layer
    if settings.is_batchnorm:
        x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    #logic for various setups such as multi dropout, batch normalization
    if settings.is_multi_drop:
        x = layers.Dropout(0.25)(x)
    if settings.is_batchnorm and settings.is_multi_drop:
        for layer1, layer2, layer3, layer4 in zip(conv2d, batch, leak, drop):
            x = layer1(x)
            x = layer2(x)
            x = layer3(x)
            x = layer4(x)
    elif settings.is_batchnorm and not settings.is_multi_drop:
        for layer1, layer2, layer3 in zip(conv2d, batch, leak):
            x = layer1(x)
            x = layer2(x)
            x = layer3(x)
    elif not settings.is_batchnorm and settings.is_multi_drop:
        for layer1, layer2, layer3 in zip(conv2d, leak, drop):
            x = layer1(x)
            x = layer2(x)
            x = layer3(x)
    else:
        for layer1, layer2 in zip(conv2d, leak):
            x = layer1(x)
            x = layer2(x)
    x = layers.Flatten()(x)

    #single dropout
    if not settings.is_multi_drop:
        x = layers.Dropout(0.5)(x)
    x = layers.Dense(1, activation='sigmoid')(x)
    disc = keras.models.Model(disc_input, x, name="discriminator")
    return disc


def build_generator(settings: ModelSettings):
    conv2d = []

    for i in range(settings.num_conv_layers):
        if settings.filter_mode == 1:
            gfilter = settings.gen_filter + i
        elif settings.filter_mode == 2:
            gfilter = settings.gen_filter - i
        else:
            gfilter = settings.gen_filter
        conv2d.append(layers.Conv2DTranspose(pow(2, gfilter), settings.kernel, strides=settings.stride, padding='same',
                                             kernel_initializer=settings.initWeights
                                             ))

    leak = [layers.LeakyReLU() for _ in range(settings.num_conv_layers)]
    if settings.is_batchnorm:
        batch = [layers.BatchNormalization() for _ in range(settings.num_conv_layers)]

    generator_input = keras.Input(shape=(settings.latent_dim))
    #Ascending mode decreases the filter siye to acomodate increasing it dinamically
    if settings.filter_mode == 1:
        gfilter = settings.gen_filter - 1
    #Descending mode  increases the filter size to acomodate reducing it dinamically
    elif settings.filter_mode == 2:
        gfilter = settings.gen_filter + 1
    #Same filter mode ,all the filter sizes are the same in the model
    else:
        gfilter = settings.gen_filter

    #first stack of layers
    x = layers.Dense(settings.dim * settings.dim * pow(2, gfilter))(generator_input)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Reshape((settings.dim, settings.dim, pow(2, gfilter)))(x)
    #iterating trough arrays and adding layers
    for layer1, layer2, layer3 in zip(conv2d, batch, leak):
        x = layer1(x)
        x = layer2(x)
        x = layer3(x)

    #last layers
    if settings.is_tanh:
        x = layers.Conv2DTranspose(settings.output, 4, activation='tanh', padding='same',
                                   kernel_initializer=settings.initWeights
                                   )(x)
    else:
        x = layers.Conv2DTranspose(settings.output, 4, activation='sigmoid', padding='same',
                                   kernel_initializer=settings.initWeights
                                   )(x)
    gen = keras.Model(generator_input, x, name="generator")
    return gen


#building models
def build(settings: ModelSettings):
    settings.gen = build_generator(settings)
    settings.disc = build_discriminator(settings)
    settings.gen.summary()
    settings.disc.summary()
    return settings.gen, settings.disc
