from lib.utils import model
from utils import checkpoints, newGan
from tensorflow import keras


class GANBuilder:

    def __init__(self,
                 atrib: model.ModelSettings,
                 name: str,
                 check_location: str,
                 b_size: int,
                 l_dim: int,
                 epochs: int,
                 **kwargs):

        self.atrib = atrib #model attributes as a modelSetting object
        self.name = name
        self.check_location = check_location #checkpoint location
        self.b_size = b_size #batch size
        self.l_dim = l_dim #latent_dimention
        self.epochs = epochs #epoch

        self.models = None #storing models
        self.checkpoint = None #storing checkpoints
        self.gan = None #storing gan

        #optimizers
        self.gopt = keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
        self.dopt = keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)


def make_gan(settings: GANBuilder, **kwargs):
    settings.models = model.build(settings.atrib) #building models

    #making / loading checkpoints
    settings.checkpoint = checkpoints.createCheck(settings.name,
                                                  settings.check_location,
                                                  gen=settings.models[0],
                                                  disc=settings.models[1],
                                                  gen_opt=settings.gopt,
                                                  disc_opt=settings.dopt
                                                  )
    #building GAN object
    g = newGan.GAN(g=settings.models[0],
                   d=settings.models[1],
                   b_size=settings.b_size,
                   l_dim=settings.l_dim,
                   check=settings.checkpoint[0],
                   prefix=settings.checkpoint[1])
    
    # Compile GAN
    g.compile(d_optimizer=settings.dopt, g_optimizer=settings.gopt)
    return g
