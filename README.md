## The implement of dilated convolution in tensorflow
This is an example of semantic image segmentation using the pre-trained dilated convolution model. the original post can be found [here]( https://github.com/ndrplz/dilation-tensorflow), by @ndrplz. <br />
The code had been tested with python3.5.

### 1. preparation
Download the pre-tained weights, which were trained by Caffe implementation, with the following [link]( https://drive.google.com/file/d/0Bx9YaGcDPu3Xd0JrcXZpTEpkb0U/view). And save it to the note the directory "./data/". Note that, the model was trained with CamVid dataset.

### 2. converge the pre-trained weights into tensorflow check point.
```bash
python pickle2tf.py
```

### 3. test a single image
```bash
python single_image_test.py -i ./test_image/test1.png
```
The result image file will use saved in the directory "./output/".
