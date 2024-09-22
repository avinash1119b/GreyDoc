import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import RMSprop
from io import BytesIO
from PIL import Image


page_bg_img = """
<style>
[data-testid="stAppViewContainer"]{
    background-image: url("https://media.licdn.com/dms/image/D5622AQEWJG7WHfT7jg/feedshare-shrink_2048_1536/0/1715094770811?e=1718236800&v=beta&t=daBNLiuvguxJ4Dr63vS2jv34YJmXbEyLhKTvY4A7wuM");
    background-size: cover;
}

[data-testid="stHeader"]{
    background-color: rgba(0,0,0,0);
}


[class="css-17ziqus e1fqkh3o3"]{
    background-image: url("https://www.pixelstalk.net/wp-content/uploads/images1/Medical-Wallpapers-HD.jpg");
    background-size: cover;
}

</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)

# loading the saved models

diabetes_model = pickle.load(open('C:/Mine/Deploy/Models/diabetes_model.sav', 'rb'))

heart_disease_model = pickle.load(open('C:/Mine/Deploy/Models/heart_disease_model.sav', 'rb'))

parkinsons_model = pickle.load(open('C:/Mine/Deploy/Models/parkinsons_model.sav', 'rb'))

kidney_model = pickle.load(open('C:/Mine/Deploy/Models/kidney_model.sav', 'rb'))

model_path = 'C:/Mine/Deploy/Models/Tumor.h5'
Brain_model = load_model(model_path)


# Define custom objects
custom_objects = {
    'RMSprop': RMSprop
}

# Load the model without compiling (bypasses optimizer issue)
Blood_model = load_model("C:/Mine/Deploy/Models/Blood_Cancer.h5", compile=False)

# sidebar for navigation
with st.sidebar:
    selected = option_menu('GreyDoc',

                           ['Diabetes Prediction',
                            'Heart Disease Prediction',
                            'Parkinsons Prediction',
                            'Chronic Kidney Cancer   Prediction',
                            'Brain Tumor Detection',
                            'Blood cell Image Classifier'],

                           menu_icon='hospital',
                           icons=['activity', 'heart', 'person', 'meta', 'braces-asterisk', 'droplet-fill'],
                           default_index=0)


# Diabetes Prediction Page
if selected == 'Diabetes Prediction':
    # page title
    st.title('Diabetes Prediction using ML')

    # getting the input data from the user
    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')

    with col2:
        Glucose = st.text_input('Glucose Level')

    with col3:
        BloodPressure = st.text_input('Blood Pressure value')

    with col1:
        SkinThickness = st.text_input('Skin Thickness value')

    with col2:
        Insulin = st.text_input('Insulin Level')

    with col3:
        BMI = st.text_input('BMI value')

    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')

    with col2:
        Age = st.text_input('Age of the Person')


    # code for Prediction
    diab_diagnosis = ''

    # creating a button for Prediction

    if st.button('Diabetes Test Result'):

        user_input = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
                      BMI, DiabetesPedigreeFunction, Age]

        user_input = [float(x) for x in user_input]

        diab_prediction = diabetes_model.predict([user_input])

        if diab_prediction[0] == 1:
            diab_diagnosis = 'The person is diabetic'
        else:
            diab_diagnosis = 'The person is not diabetic'

    st.success(diab_diagnosis)

# Heart Disease Prediction Page
if selected == 'Heart Disease Prediction':

    # page title
    st.title('Heart Disease Prediction using ML')

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.text_input('Age')

    with col2:
        sex = st.text_input('Sex')

    with col3:
        cp = st.text_input('Chest Pain types')

    with col1:
        trestbps = st.text_input('Resting Blood Pressure')

    with col2:
        chol = st.text_input('Serum Cholestoral in mg/dl')

    with col3:
        fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl')

    with col1:
        restecg = st.text_input('Resting Electrocardiographic results')

    with col2:
        thalach = st.text_input('Maximum Heart Rate achieved')

    with col3:
        exang = st.text_input('Exercise Induced Angina')

    with col1:
        oldpeak = st.text_input('ST depression induced by exercise')

    with col2:
        slope = st.text_input('Slope of the peak exercise ST segment')

    with col3:
        ca = st.text_input('Major vessels colored by flourosopy')

    with col1:
        thal = st.text_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect')

    # code for Prediction
    heart_diagnosis = ''

    # creating a button for Prediction

    if st.button('Heart Disease Test Result'):

        user_input = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]

        user_input = [float(x) for x in user_input]

        heart_prediction = heart_disease_model.predict([user_input])

        if heart_prediction[0] == 1:
            heart_diagnosis = 'The person is having heart disease'
        else:
            heart_diagnosis = 'The person does not have any heart disease'

    st.success(heart_diagnosis)

# Parkinson's Prediction Page
if selected == "Parkinsons Prediction":

    # page title
    st.title("Parkinson's Disease Prediction using ML")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        fo = st.text_input('MDVP:Fo(Hz)')

    with col2:
        fhi = st.text_input('MDVP:Fhi(Hz)')

    with col3:
        flo = st.text_input('MDVP:Flo(Hz)')

    with col4:
        Jitter_percent = st.text_input('MDVP:Jitter(%)')

    with col5:
        Jitter_Abs = st.text_input('MDVP:Jitter(Abs)')

    with col1:
        RAP = st.text_input('MDVP:RAP')

    with col2:
        PPQ = st.text_input('MDVP:PPQ')

    with col3:
        DDP = st.text_input('Jitter:DDP')

    with col4:
        Shimmer = st.text_input('MDVP:Shimmer')

    with col5:
        Shimmer_dB = st.text_input('MDVP:Shimmer(dB)')

    with col1:
        APQ3 = st.text_input('Shimmer:APQ3')

    with col2:
        APQ5 = st.text_input('Shimmer:APQ5')

    with col3:
        APQ = st.text_input('MDVP:APQ')

    with col4:
        DDA = st.text_input('Shimmer:DDA')

    with col5:
        NHR = st.text_input('NHR')

    with col1:
        HNR = st.text_input('HNR')

    with col2:
        RPDE = st.text_input('RPDE')

    with col3:
        DFA = st.text_input('DFA')

    with col4:
        spread1 = st.text_input('spread1')

    with col5:
        spread2 = st.text_input('spread2')

    with col1:
        D2 = st.text_input('D2')

    with col2:
        PPE = st.text_input('PPE')

    # code for Prediction
    parkinsons_diagnosis = ''

    # creating a button for Prediction    
    if st.button("Parkinson's Test Result"):

        user_input = [fo, fhi, flo, Jitter_percent, Jitter_Abs,
                      RAP, PPQ, DDP,Shimmer, Shimmer_dB, APQ3, APQ5,
                      APQ, DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]

        user_input = [float(x) for x in user_input]

        parkinsons_prediction = parkinsons_model.predict([user_input])

        if parkinsons_prediction[0] == 1:
            parkinsons_diagnosis = "The person has Parkinson's disease"
        else:
            parkinsons_diagnosis = "The person does not have Parkinson's disease"

    st.success(parkinsons_diagnosis)

#Kidney Disease Prediction
if selected == "Chronic Kidney Cancer Prediction":

    # page title
    st.title("Chronic Kidney Disease Prediction")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        age = st.text_input('Age')

    with col2:
        blood_pressure = st.text_input('Blood Pressure')

    with col3:
        specific_gravity = st.text_input('Specific_Gravity')

    with col4:
        albumin = st.text_input('Albumin')

    with col5:
        sugar = st.text_input('Sugar')

    with col1:
        red_blood_cells = st.text_input('Red Blood Cells: Normal=1, Abnorm=0')

    with col2:
        pus_cell = st.text_input('Pus Cells: Normal=1, Abnormal=0')

    with col3:
        pus_cell_clumps = st.text_input('Pus Cells Clumps: Present=1, NotPres=0')

    with col4:
        bacteria = st.text_input('Bacteria: Present=1, NotPresent=0')

    with col5:
        diabetes_mellitus = st.text_input('Diabetes Millitus: Yes=1, No=0')

    with col1:
        blood_urea = st.text_input('Blood Urea')

    with col2:
        serum_creatinine = st.text_input('Serum Creatinine')

    with col3:
        sodium = st.text_input('Sodium')

    with col4:
        potassium = st.text_input('Potassium')

    with col5:
        haemoglobin = st.text_input('Heamoglobin')

    with col1:
        coronary_artery_disease = st.text_input('Coronary Artery Disease: Yes=1, No=0')

    with col2:
        white_blood_cell_count = st.text_input('White Blood Cell Count')

    with col3:
        appetite = st.text_input('Appetite: Good=0, Poor=1')

    with col4:
        hypertension = st.text_input('Hypertension: Yes=1, No=0')

    with col5:
        peda_edema = st.text_input('Peda Edema: Yes=1, No=0')

    with col1:
        packed_cell_volume = st.text_input('Packed Cell Volume')

    with col2:
        red_blood_cell_count = st.text_input('Red Blood Cell Count')

    with col3:
        blood_glucose_random = st.text_input('Blood Glucose')

    with col4:
        aanemia = st.text_input('Aanemia: Yes=1, No=0')

    # code for Prediction
    kidney_diagnosis = ''

    # creating a button for Prediction
    if st.button("Kidney Test Result"):

        user_input = [age,	blood_pressure,	specific_gravity,	albumin,	sugar,	red_blood_cells,	pus_cell,	pus_cell_clumps,	bacteria,
        	diabetes_mellitus,	blood_urea,	serum_creatinine,	sodium,	potassium,	haemoglobin,	coronary_artery_disease,	white_blood_cell_count,
            appetite,	hypertension,	peda_edema,	packed_cell_volume,	red_blood_cell_count,	blood_glucose_random,	aanemia]

        user_input = [float(x) for x in user_input]

        kidney_prediction = kidney_model.predict([user_input])

        if kidney_prediction[0] == 0:
            kidney_diagnosis = "The person has Cronic Kidney disease"
        else:
            kidney_diagnosis = "The person does not have Cronic Kidney disease"

    st.success(kidney_diagnosis)


if selected == 'Brain Tumor Detection':
    # Function to preprocess the image
    def preprocess_image(image_file, target_size):
        img = Image.open(image_file)
        img = img.resize(target_size)
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        return img_array

    # Main Streamlit app
    st.title('Brain Tumor Detection')

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption='Image Uploaded.', use_column_width=False)
        st.write("")
        st.write("Identifying...")

        # Preprocess the uploaded image
        img = preprocess_image(uploaded_file, target_size=(224, 224))

        # Make predictions
        prediction = Brain_model.predict(img)
        if prediction[0][0] > 0.5:
            st.write("The Person have Brain Tumor")  # Update with your class labels
        else:
            st.write("The Person does not have Brain Tumor")  # Update with your class labels

if selected == 'Blood cell Image Classifier':
    class_labels = ['Basophil', 'Eosinophil', 'Erythroblast', 'IG', 'Lymphocyte', 'Monocyte', 'Neutrophil', 'Platelet']

    # Function to preprocess image
    def preprocess_image(img):
        img = img.resize((256, 256))  # Resize the image
        img_array = np.array(img)  # Convert to numpy array
        img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions to create batch size of 1
        img_array = img_array / 255.0  # Normalize pixel values
        return img_array

    # Streamlit UI
    st.title("Blood Cell Image Classifier")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png"])

    if uploaded_file is not None:
        # Open the image using PIL
        img = Image.open(BytesIO(uploaded_file.read()))

        # Display the uploaded image
        st.image(img, caption='Uploaded Image.', use_column_width=True)

        # Preprocess the image
        img_array = preprocess_image(img)

        # Make predictions
        predictions = Blood_model.predict(img_array)

        # Get the predicted class index
        predicted_class_index = np.argmax(predictions)
        predicted_probability = predictions[0][predicted_class_index]

        if predicted_probability >= 1:
            decoded_predictions = class_labels[predicted_class_index]
            st.write(f"The person has: {decoded_predictions} Cancer (Probability: {predicted_probability*100:.2f})")
        else:
            st.write("The image only has RBC.")

import time
import numpy as np
import pandas as pd
import streamlit as st

_LOREM_IPSUM = """
This is a student developed website so please consult the doctor for further instructions.\n
Diabetes Prediction has 98% accuracy.\n
Heart Disease Prediction has 98% accuracy.\n
Parkinson's Prediction has 97% accuracy.\n
Chronic Kidney Cancer Prediction has 96% accuracy.\n
Brain Tumour Detection have 92% accuracy.\n
Blood Cancer Prediction have 99% accuracy.\n
"""


if st.button(" Terms & Conditions "):
    st.write(_LOREM_IPSUM)

Credits = "Designed and Coded by Avinash😊"
st.write(Credits)


