import mxnet as mx
from mxnet import image
from mxnet.gluon.data.vision import transforms
import gluoncv
from gluoncv.data.transforms.presets.segmentation import test_transform
from gluoncv.utils.viz import get_color_pallete
import matplotlib.pyplot as plt
import streamlit as st


ctx = mx.cpu(0)

@st.cache()
def segmentation(img):
    img = mx.nd.array(img)
    img = test_transform(img, ctx)
    model = gluoncv.model_zoo.get_model('fcn_resnet101_voc', pretrained=True)
    output = model.predict(img)
    predict = mx.nd.squeeze(mx.nd.argmax(output, 1)).asnumpy()
    mask = get_color_pallete(predict, 'pascal_voc')

    output = plt.imshow(mask)

    return output
