!pip install --upgrade -q gspread

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from google.colab import auth
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
from keras import optimizers


auth.authenticate_user()
from google.colab import auth
auth.authenticate_user()

import gspread
from oauth2client.client import GoogleCredentials

gc = gspread.authorize(GoogleCredentials.get_application_default())


worksheet = gc.open('StockDataWeek.csv').sheet1

# get_all_values gives a list of rows.
rows = worksheet.get_all_values()

np.random.seed(7)
data = np.array(rows)
print(data.shape)


scaler = MinMaxScaler()
#scaler = StandardScaler()

data = scaler.fit_transform(data)


X = data[:,0:11]
Y = data[:,11]


model = Sequential()
model.add(Dense(32,input_dim=11,activation='relu',kernel_initializer='normal'))
model.add(Dense(16, activation = 'relu',kernel_initializer='normal'))
model.add(Dense(12, activation = 'relu',kernel_initializer='normal'))
model.add(Dense(1,kernel_initializer='normal'))

sgd = optimizers.SGD(lr=0.09, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(loss='mean_squared_error',optimizer='adam' ,metrics = ['mse'])

worksheet1 = gc.open('StockTestWeek.csv').sheet1

# get_all_values gives a list of rows.
t = worksheet1.get_all_values()

test = np.array(t)

print(t)

testdata = scaler.fit_transform(test)
Xtest = testdata[:,:11]
Ytest = testdata[:,11]

model.summary()
history = model.fit(X,Y,epochs=50,batch_size=5,validation_data=(Xtest,Ytest))

predict = model.predict(Xtest)
plt.plot(predict, label = "Predicted Value")
plt.plot(Ytest, label = "True Value")
plt.legend()
plt.title("Predicted Value over 10 days")
plt.xlabel("Days after training date")
plt.ylabel("Normalized Stock Price")


outArray = np.concatenate((Xtest,predict),axis=1)
val = scaler.inverse_transform(outArray)

predVal = val[:,11]
test = test.astype(float)
trueVal = test[:,11]

plt.plot(predVal,label="Predicted Value")
plt.plot(trueVal,label="True Value")
plt.legend()
plt.title("Predicted vs True Value")
plt.xlabel("Days after training data")
plt.ylabel("Google Stock Value")



plt.plot(history.history['loss'])
plt.title("Loss Function")
plt.xlabel("Epochs")
plt.ylabel("MSE")

print(min(history.history['val_loss']))
print(min(history.history['loss']))

plt.plot(history.history['val_loss'])
plt.title('Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('MSE')


