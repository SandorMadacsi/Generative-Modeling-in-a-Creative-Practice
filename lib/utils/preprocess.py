# Data Preprocessing utility by Sandor Madacsi
# Takes a keras dataset of integers
#  - normalizes values (from 0 to 255) to values between -1 and 1
#  (The discriminator uses a tanh activation as the last function, and it works best in this range.)
#  - extend the dimension to take number of samples
#  - reshape the dataset
#  - if we need batching it slices the dataset in the given batch_size

import numpy as np
import tensorflow as tf

def preprocess_Data(x, y, isBatching, res, channel, b_size, isTanh):

    dataset = np.concatenate([x, y], axis=0)

    if(isTanh):

        dataset = dataset / 127.5 - 1.0 #normalize data for tanh activation to -1 and 1

    else:

        dataset = dataset / 255.  #normalize data for sigmoid activation to 0 and 1

    dataset = np.expand_dims(dataset, 3)

    dataset = np.array(dataset).reshape(-1, res, res, channel)

    if isBatching:

        dataset = tf.data.Dataset.from_tensor_slices(dataset).batch(b_size) #batch dataset

    return dataset