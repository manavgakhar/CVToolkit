from matplotlib import pyplot as plt
import gluoncv
from gluoncv import model_zoo, data, utils
import mxnet as mx
import cv2
import streamlit as st

@st.cache()
def detect(img):
    
    img = mx.nd.array(img)
    net = model_zoo.get_model('faster_rcnn_resnet50_v1b_voc', pretrained=True)
    x = data.transforms.presets.rcnn.transform_test(img)
    
    box_ids, scores, bboxes = net(x[0])
    
    ax = utils.viz.plot_bbox(x[1], bboxes[0], scores[0], box_ids[0], class_names=net.classes)
    
    return ax
    
    
