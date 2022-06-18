import tensorflow as tf
import os

def createCheck(model: str,
                location: str,
                gen: tf.keras.models.Model,
                disc: tf.keras.models.Model,
                gen_opt: tf.keras.optimizers.Adam,
                disc_opt: tf.keras.optimizers.Adam
                ):
    #created checkpoint dir if applicable
    dir = f"{model}"
    parent = location
    check_dir = os.path.join(parent,dir)
    if not os.path.exists(check_dir):
        os.mkdir(check_dir)

    #creates checkpoint object and prefix
    checkpoint_prefix = os.path.join(check_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=gen_opt,
                                     discriminator_optimizer=disc_opt,
                                     generator=gen,
                                     discriminator=disc)

    #checkpoint manager
    ckpt_manager = tf.train.CheckpointManager(checkpoint, check_dir, max_to_keep=5)

    #load checkpoint if applicable
    if ckpt_manager.latest_checkpoint:
        checkpoint.restore(ckpt_manager.latest_checkpoint).expect_partial()
        print('Checkpoint restored')

    return checkpoint, checkpoint_prefix
