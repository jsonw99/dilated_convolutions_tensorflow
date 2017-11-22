import tensorflow as tf
import pickle
import os
import os.path as path
from model import dilation_model_pretrained
from datasets import CONFIG


if __name__ == '__main__':

    # Choose between 'cityscapes' and 'camvid'
    dataset = 'camvid'

    # Load dict of pretrained weights
    print("Loading pre-trained weights...")
    with open(CONFIG[dataset]['weights_file'], 'rb') as f:
        w_pretrained = pickle.load(f)
    print("Loading completed.")

    # Create checkpoint directory
    checkpoint_dir = path.join('data/checkpoint', 'dilation_' + dataset)
    if not path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # Build pretrained model and save it as TF checkpoint
    with tf.Session() as sess:
        print("Converting the pre-trained weights...")
        # Choose input shape according to dataset characteristics
        input_h, input_w, input_c = CONFIG[dataset]['input_shape']
        input_tensor = tf.placeholder(tf.float32, shape=(None, input_h, input_w, input_c), name='input_placeholder')

        # Create pretrained model
        model = dilation_model_pretrained(dataset, input_tensor, w_pretrained, trainable=False)

        sess.run(tf.global_variables_initializer())

        # Save both graph and weights
        saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
        saver.save(sess, path.join(checkpoint_dir, 'dilation'))
        print("Conversion task completed.")




