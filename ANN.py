import pandas as pd 
from sklearn.preprocessing import OneHotEncoder,StandardScaler,LabelEncoder
from sklearn.model_selection import train_test_split

import pickle

data=pd.read_csv("/Users/nomanmacbook/Downloads/loan_approval_data.csv")
print(data['Loan_Status'].value_counts())
print(data.head)
data = data.drop(["ApplicantID", "Married", "Education", "CoapplicantIncome"], axis=1)
print(data.head)
LED_Gender=LabelEncoder()
data["Gender"]=LED_Gender.fit_transform(data["Gender"])
print(data)
with open("Gender.pkl","wb") as file:
    pickle.dump(LED_Gender,file)
LED_Self=LabelEncoder()
data["SelfEmployed"]=LED_Self.fit_transform(data["SelfEmployed"])
print(data)
with open("SelfEmployed.pkl","wb") as file:
    pickle.dump(LED_Self,file)
le_status = LabelEncoder()
data["Loan_Status"] = le_status.fit_transform(data["Loan_Status"])
with open("Loan_Status.pkl","wb") as file:
        pickle.dump(le_status,file)
OHE_PD=OneHotEncoder()
encodeds=OHE_PD.fit_transform(data[["Property_Area"]])
end_encoding=pd.DataFrame(encodeds.toarray(),columns=(OHE_PD.get_feature_names_out(["Property_Area"])))
data=pd.concat([data.drop("Property_Area",axis=1),end_encoding],axis=1)
print(data.head)
with open("OHE_PD.pkl","wb") as file:
    pickle.dump(OHE_PD,file)
print(data.head)
x=data.drop("Loan_Status",axis=1)#feauter mean input
y=data["Loan_Status"]#output mean lable

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
scalar=StandardScaler()
x_train=scalar.fit_transform(x_train)
x_test=scalar.fit_transform(x_test)
print(x_test.shape)
with open("scalar.pkl","wb") as file:
        pickle.dump(scalar,file)
#bulid ANN model
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import TensorBoard , EarlyStopping
import datetime
model=Sequential([
     Dense(64,activation="relu",input_shape=( x_train.shape[1],)),
     Dense(32,activation="relu"),
     Dense(1,activation="sigmoid")
])
print(model.summary())
opr=tf.keras.optimizers.Adam(learning_rate=0.01)
loss=tf.keras.losses.BinaryCrossentropy()
model.compile(optimizer=opr,loss="binary_crossentropy",metrics=["accuracy"])

log_dir= "logs/fit" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensor_call=TensorBoard(log_dir=log_dir,histogram_freq=1)
EarlyStopping_calls=EarlyStopping(monitor="val_loss",patience=5,restore_best_weights=True)

#traning ANN model
history=model.fit(
     x_train,y_train,
     validation_data=(x_test,y_test),
     epochs=100,
     callbacks=[tensor_call,EarlyStopping_calls]
)
model.save("model_clean.keras")



  # ✅ Use ONLY this for Keras 3
  # ✅ This is Keras 3 compatible
 

 

#run vs code    python3 -m tensorboard.main --logdir=logs/fit20250726-140509

