
from tensorflow import keras
import tensorflow as tf
from lib.utils import sampling
from IPython import display

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

class GAN(keras.Model):


    def __init__(self,
                 g: keras.models.Model = None,
                 d: keras.models.Model = None,
                 l_dim: int = 100,
                 b_size: int = 4,
                 check: tf.train.Checkpoint = None,
                 prefix: str = None):

        super(GAN, self).__init__()
        self.g = g #generator
        self.d = d #discriminator
        self.l_dim = l_dim #latent dimension
        self.b_size = b_size #batch size
        self.check = check #chekpoint object
        self.prefix = prefix #checkpoint prefix

        # custom loss metrics
        self.d_loss_metric = keras.metrics.Mean(name="d_loss")
        self.g_loss_metric = keras.metrics.Mean(name="g_loss")

    def compile(self,d_optimizer, g_optimizer):
        super(GAN, self).compile()
        self.d_optimizer = d_optimizer #discriminator optimizer
        self.g_optimizer = g_optimizer #generator optimizer

    @property
    # return custom metrics
    def metrics(self):
        return [self.d_loss_metric, self.g_loss_metric]

    # creates a normal gaussian noise matrix
    def generate_sample(self, batch):

      return tf.random.normal(shape=(batch, self.l_dim))

    # summing labels for the discriminator
    def sum_labels(self,batch):
        real_y = tf.zeros((batch,1))#real labels
        fake_y = tf.ones((batch,1))#fake labels
        sum_y = tf.concat([real_y,fake_y],axis=0)#concatinated labels into one matrix
        sum_y += 0.05 * tf.random.uniform(tf.shape(sum_y))#adding random noise to the labels
        return sum_y

    # summing images for the dicsriminator
    def sum_images(self, real, fake):
        sum_x = tf.concat([real, fake],axis=0)
        return sum_x

    # discriminator loss
    def disc_loss(self,x,y):
        pred = self.d(x)#predictions of the summed iamges
        d_loss = cross_entropy(y, pred)#loss of the prediction to the summed labels
        return d_loss

    # generator loss
    def gen_loss(self,random_sample,labels):
        pred = self.d(self.g(random_sample))#predictions of only generated samples
        g_loss = cross_entropy(labels, pred)#pred compared to misleading labels
        return g_loss

    def train_step(self, real_images):
        b = tf.shape(real_images)[0]#batch size
        sample = self.generate_sample(batch=b) #random generated sample
        generated_images = self.g(sample) #generated image
        sum_x = self. sum_images(real=real_images, fake=generated_images)#summed images for dicsriminator training
        sum_y = self.sum_labels(b) #summed labels for discriminator training
        with tf.GradientTape() as tape:
            d_loss = self.disc_loss(x=sum_x, y=sum_y)#discriminator loss

        # gradient descent
        grads = tape.gradient(d_loss, self.d.trainable_weights)
        self.d_optimizer.apply_gradients(zip(grads, self.d.trainable_weights))

        sample = self.generate_sample(batch=b) #new sample
        miss_labels = tf.zeros((b, 1)) #misleading labels for generator training

        with tf.GradientTape() as tape:
            g_loss = self.gen_loss(sample, miss_labels)#generator loss

        # gradient descent
        grads = tape.gradient(g_loss, self.g.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.g.trainable_weights))

        # update loss metrics
        self.d_loss_metric.update_state(d_loss)
        self.g_loss_metric.update_state(g_loss)
        return {"d_loss": self.d_loss_metric.result(), "g_loss": self.g_loss_metric.result()}

class GANCallback(keras.callbacks.Callback):
    def __init__(self,
                 num_img=8,
                 seed=8,
                 loc="",
                 cFreq=1,
                 iFreq=1,
                 name="model"):
        self.num_img = num_img
        self.seed = seed
        self.loc = loc
        self.cFreq = cFreq
        self.iFreq = iFreq

        self. loc = sampling.create_dir(loc, name)

    def on_epoch_end(self,
                     epoch,
                     logs=None):

        generated_images = self.model.g(self.seed, training=False)#generator creates sample
        display.clear_output(wait=True)#clear output in notebook
        # save and plot images
        sampling.plot_images(images=generated_images, freq=self.iFreq, epoch=epoch, dir=self.loc)
        # sawing checkpoint at given intervals
        if self.cFreq is not None and (epoch + 1) % self.cFreq == 0:
            self.model.check.save(file_prefix=self.model.prefix)