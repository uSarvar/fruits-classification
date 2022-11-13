import streamlit as st
from fastai.vision.all import *
import pathlib
import plotly.express as px
import platform

plt = platform.system()
if plt == 'Linux' : pathlib.WindowsPath = pathlib.PosixPath

# title
st.title('Classification of fruits')
st.subheader('This app predicts which of the classes (strawberry, lemon, grape, banana and apple) an image uploaded to the app belongs to. You can upload an image or take a pic from your device.')

# upload pics
file = st.file_uploader('Upload image', type=['png','jpeg','jpg','gif','svg'])
pic = st.camera_input('Take a picture')
# model
model = load_learner('fruit_model1.pkl')
if file:
   # PIL convert
   img = PILImage.create(file)
   pred, pred_id, probs = model.predict(img)
else:
   img = PILImage.create(pic)
   pred, pred_id, probs = model.predict(img)

# plotting
fig = px.bar(x=probs*100, y=model.dls.vocab)
st.plotly_chart(fig)

# prediction
st.success(f'Prediction: {pred}')
st.info(f'Probability: {probs[pred_id]*100:.2f}%')
st.image(file)


   
