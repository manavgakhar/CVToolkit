import streamlit as st
from PIL import Image
import cv2
from img_clf import predict
from PIL import Image
from obj_det import detect
import matplotlib.pyplot as plt
from numpy import asarray


st.title("Computer Vision Toolkit")
st.write("")

st.set_option('deprecation.showfileUploaderEncoding', False)
st.set_option('deprecation.showPyplotGlobalUse', False)

file_up = st.file_uploader("Upload an image", type="jpg")



option = st.selectbox('Which feature would you like to use?',('Image Classification','Object Detection'))



if file_up is not None and option == "Image Classification":

    image = Image.open(file_up)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
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
    
    
