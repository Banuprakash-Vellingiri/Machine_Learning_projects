import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
#------------------------------------------------------------------------------------------------------------
# Set page configuration
st.set_page_config(
                    page_title="Kannada MNIST Classification",
                    page_icon="	:mag:",
                    layout="wide",  # or "centered"
                  )
st.markdown(' # :books: :orange[Kannada MNIST] Classification &nbsp;')
st.markdown("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
show_images = st.checkbox(':orange[Show Digits With Classes]')
if show_images:
    image=Image.open( r"C:\Users\banup\Desktop\NLP_guvi_project\Project _1 _Kannada_MNIST _ Classification_Problem\sample_images\output.png")
    rezied_image=image.resize((4000,3000))
    st.image( image)
st.write("### :blue[Upload Valid Image For Classification ] :arrow_down_small: ")
#---------------------------------------------------------------------------------------------------------------
#Image Uploader
uploaded_image= st.file_uploader("", type="jpg")
#Submit Button
submit_button=st.button(":orange[Submit Image]")
if submit_button and uploaded_image:
            
            col1,col2,col3=st.columns(3)
            
            with col1:
                st.text("Uploaded Image:")
                # Load the image only for display purpose 
                image = Image.open(uploaded_image)
                #Resizing for display purpose
                resized_image=image.resize((350,350))
                st.image(resized_image)
            with col2:
                st.text(" Processed Image:") 
                image = Image.open(uploaded_image) 
                resized_image = image.resize((28,28))
                # Convert the image to grayscale
                grey_image = resized_image .convert("L")
                # Convert the grayscale image to a NumPy array
                image_array = np.array(grey_image)
                #Reshaping the array for input
                final_image=image_array.reshape(1,784)
                #-------------------------------------------------------------------------------------------------
                plt.figure(figsize=(5,5))
                plt.imshow(final_image.reshape(28, 28),cmap="gray") #Reshaping it into  28x28 image
                plt.axis("off")
                plt.show()
                #Saving the processed image to local directory
                file_location=r"C:\Users\banup\Desktop\NLP_guvi_project\Project _1 _Kannada_MNIST _ Classification_Problem\sample_images\processed_image.jpg"
                plt.savefig(file_location, format="png")
                #Opening the saved image 
                processed_image=Image.open(r"C:\Users\banup\Desktop\NLP_guvi_project\Project _1 _Kannada_MNIST _ Classification_Problem\sample_images\processed_image.jpg")
                #Resizing for display purpose
                resized_image_2=processed_image.resize((350,350))
                st.image(resized_image_2)

                
            with col3:
                #Loading Models
                with open("svc_model.pkl","rb")as file:
                    model=pickle.load(file)
                with open("pca_svc.pkl","rb")as file:
                    pca=pickle.load(file)
                with open("mms_svc.pkl","rb")as file:
                    mms=pickle.load(file)

                sample = pca.transform(final_image)
                sample= mms.transform(sample)
                predict = model.predict(sample)
                st.markdown("### The Digit Belongs to :")
                st.info(f'# Class - :red[{predict[0]}]')

#-----------------------------------------------------------------------------------------------------
st.markdown("# ")       
st.markdown("# ")     
st.markdown("# ")    
st.markdown("# ")       
st.markdown("# ")     
      



st.text("-created by banuprakash vellingiri ❤️")