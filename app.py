import streamlit as st
from PIL import Image
import joblib
import numpy as np
import pandas as pd
# load model
model=joblib.load('bone_fracture_xgb_model.pt')
# getLable
def get_label(img):
    # convert into gray scale
    img_g=img.convert("L")
    # resize into 100*100
    img_res=img_g.resize((100,100))
    #convert into numpy
    img_a=np.array(img_res).flatten() 
    # convert in to df and transpose
    img_df=pd.DataFrame(img_a).T
    # predict with the model
    pre=model.predict(img_df)
    # return the value
    return pre
def predict (pred):
    if pred == 0 :
        st.write("Fractured")
    elif pred == 1 :
        st.write("Non fractured")
    else :
        st.write("error")
st.title("Bone Fracture Prediction")
st.header("Computer vision project")
file=st.file_uploader("Upload your file ",type="png")
try:
    if file is not None:
        # read img
        img=Image.open(file)
        # show Image
        prediction=get_label(img)
        st.image(img,"Uploaded Image")
       
        st.write(f"Bone is ")
        predict(prediction)
        
    else:
        st.write("Empty file can't be read")
except Exception as e:
    st.write(f"Error : {e}")

finally:
    st.write("Thank for using")