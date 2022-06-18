from lib.utils import model
from tensorflow import keras
import os

def buildModel(n, f, ncl, k, s, isBatch,isT, dataf, o, shape, d, l, df, mDrop, fm):

    #model settings object
    data = model.ModelSettings(model_name=n,
                               dfilter=df,
                               filter=f,
                               number_of_conv_layers=ncl,
                               kernel=k,
                               stride=s,
                               isBatch=isBatch,
                               output=o,
                               img_shape=shape,
                               dim=d,
                               latent_dim=l,
                               isMDrop=mDrop,
                               filterMode=fm,
                               isTanh=isT
                               )
    generator = model.buildGenerator(data) #build generator
    generator.summary() #generator summary
    discriminator = model.buildDiscriminator(data) #build discriminator
    discriminator.summary() #discriminator summary
    dataframe = model.buildDataframe(data, dataf)  #dataframe for model setups
    output_path = '../model_settings.csv'
    dataframe.to_csv(output_path, mode='a', header=not os.path.exists(output_path))
    return generator, discriminator

#optimizers
def d_optimizer(dlr):
    return keras.optimizers.Adam(learning_rate=dlr, beta_1=0.5)

def g_optimizer(glr):
    return keras.optimizers.Adam(learning_rate=glr, beta_1=0.5)


