import streamlit as st
from PIL import Image
import cv2
from img_clf import predict
from PIL import Image
from obj_det import detect
import matplotlib.pyplot as plt
from numpy import asarray
from sem_seg import segmentation

st.image("data/logo.png")
st.write("")
st.markdown(" ### Inference is CPU based so processing might take some time.")

st.set_option('deprecation.showfileUploaderEncoding', False)
st.set_option('deprecation.showPyplotGlobalUse', False)

file_up = st.file_uploader("Upload an image", type="jpg")



option = st.selectbox('Which feature would you like to use?',('Image Classification','Object Detection','Semantic Segmentation'))



if file_up is not None and option == "Image Classification":

    image = Image.open(file_up)
    st.image(image, caption='Uploaded Image.', use_column_width=False)
    st.write("")
    st.write("Just a second...")
    labels = predict(file_up)

    # print out the top 5 prediction labels with scores
    for i in labels:
        st.write("Prediction: ", i[0].split(",")[1], ",   Score: ", i[1])




if file_up is not None and option == "Object Detection":

    img = Image.open(file_up)
    img = asarray(img)

    plot = detect(img)

    st.pyplot()


if file_up is not None and option == "Semantic Segmentation":

    img = Image.open(file_up)
    img = asarray(img)

    plot = segmentation(img)

    st.pyplot()
