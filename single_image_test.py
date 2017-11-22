import tensorflow as tf
import cv2
import os
import os.path as path
import argparse
from utils import predict
from datasets import CONFIG

def get_arguments():
    """
    Parse all the arguments provide from the CLI.
    :return:
        A list of parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--img_path", type=str, help="Path to the test image.", required=True)
    return parser.parse_args()


def main():
    # Choose the model trained from 'camvid' dataset
    dataset = 'camvid'

    # Load checkpoint directory
    checkpoint_dir = path.join('data/checkpoint', 'dilation_' + dataset)

    # Image in / out parameters
    args = get_arguments();
    input_image_path = args.img_path
    image_name = input_image_path.split("/")[-1].split(".")[0]
    if not path.exists("./output"):
        os.makedirs("./output")
    output_image_path = path.join('./output', image_name + '_mask.png')

    # Restore both graph and weights from TF checkpoint
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(path.join(checkpoint_dir, 'dilation.meta'))
        saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir))

        graph = tf.get_default_graph()
        model = graph.get_tensor_by_name('softmax:0')
        model = tf.reshape(model, shape=(1,) + CONFIG[dataset]['output_shape'])

        # Read and predict on a test image
        input_image = cv2.imread(input_image_path)
        input_tensor = graph.get_tensor_by_name('input_placeholder:0')
        predicted_image = predict(input_image, input_tensor, model, dataset, sess)

        # Convert colorspace (palette is in RGB) and save prediction result
        predicted_image = cv2.cvtColor(predicted_image, cv2.COLOR_BGR2RGB)
        cv2.imwrite(output_image_path, predicted_image)
        cv2.imshow('image', predicted_image)
        cv2.waitKey()
        print('The output file has been saved as {}'.format(output_image_path))


if __name__ == '__main__':
    main()




