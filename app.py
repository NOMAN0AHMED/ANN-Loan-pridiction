import pandas as pd 
import tensorflow as tf 
from sklearn.preprocessing import OneHotEncoder,StandardScaler,LabelEncoder
import pickle
import streamlit as st
import streamlit as st



st.cache_data.clear()  # For cached data
st.cache_resource.clear()  # For cached resources

st.write("loan approval pridection")

from tensorflow.keras.models import load_model

model = load_model("model_clean.keras")






with open("Gender.pkl","rb") as file:
   LE_Gender= pickle.load(file)
with open("Loan_Status.pkl","rb") as file:
   LE_Loan_Status= pickle.load(file)
with open("SelfEmployed.pkl","rb") as file:
  LE_SelfEmployeds= pickle.load(file)
with open("OHE_PD.pkl","rb") as file:
  OHE_PD= pickle.load(file)
with open("scalar.pkl","rb") as file:
 scalar= pickle.load(file)
   
#user input                                                  #Ye lazmi nahi hai, lekin:

                                                              #✅ Ye safer aur dynamic hai.
Gender=st.selectbox("Select your Gender",LE_Gender.classes_) # classes_ Ye guarantee karta hai ke input values 
                                                            #exactly wahi hain jo model ne training ke time dekhi thi.
SelfEmployed =st.selectbox("SelfEmployed",LE_SelfEmployeds.classes_)
ApplicantIncome=st.number_input("ApplicantIncome")
LoanAmount=st.number_input("LoanAmount")
Loan_Amount_Term=st.slider("Loan_Amount_Term",1,360)
Credit_History=st.selectbox("Credit_History",[0,1])
Property_Area=st.selectbox("Property_Area",OHE_PD.categories_[0])



gender_encoded = LE_Gender.transform([Gender])[0]
SelfEmployeds_encoded= LE_SelfEmployeds.transform([SelfEmployed])[0]


input_data=pd.DataFrame([{
   
   "SelfEmployed": SelfEmployeds_encoded,
   "Gender":gender_encoded,
   "ApplicantIncome":ApplicantIncome,
   "LoanAmount":LoanAmount,
   "Credit_History":Credit_History,
   "Property_Area":Property_Area,
   "Loan_Amount_Term":Loan_Amount_Term
}])
encodeds=OHE_PD.transform([[Property_Area]]).toarray()

end_encoding=pd.DataFrame(encodeds,columns=(OHE_PD.get_feature_names_out(["Property_Area"])))

input_data = pd.concat([input_data.drop("Property_Area", axis=1), end_encoding], axis=1)

# Step 3: Add any missing columns
#scalar.feature_names_in_ Ye line guarantee karti hai ke tumhara data exactly usi column order 
# me ho jis order me StandardScaler ne training ke waqt fit kiya tha.
expected_cols = scalar.feature_names_in_ 
for col in expected_cols:
    if col not in input_data.columns:
        input_data[col] = 0

# Step 4: Ensure correct order
input_data = input_data[expected_cols]


input_data_encoded=scalar.transform(input_data)
prediction=model.predict(input_data_encoded)
pridiction_proba=prediction[0][0]
st.write(f"pridiction_proba {pridiction_proba:.2f}")
print(pridiction_proba)
if  pridiction_proba>0.5:
    st.success("Loan is very likely to be approved ✅")

else:
    st.error("Loan is unlikely to be approved ❌")
