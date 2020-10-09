# Importing the Libraries

import json
import numpy as np
import pandas as pd



'''
# data load
# data split
# artitecture
# train
# eval
#predication
'''



# Loading Data

def Load_Data(Path):
  with open(Path,"r") as fp:
    data=json.loads(fp.read())
    return data

data=Load_Data(r'data_10.json')
X=data['mfcc']
Y=data['labels']
X=np.array(X)
Y=np.array(Y)


print(len(X))
print(len(Y))

print(X.shape)
print(Y.shape)
Y=Y.reshape((999,1))
# Splitting the Data

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=20,random_state=10)



import tensorflow.keras as keras


classifier=keras.Sequential()
classifier.add(keras.layers.Flatten(input_shape=(X.shape[1],X.shape[2])))

classifier.add(keras.layers.Dense(512,activation='relu',kernel_regularizer=keras.regularizers.l2(0.001)))
classifier.add(keras.layers.Dense(256,activation='relu',kernel_regularizer=keras.regularizers.l2(0.001)))
classifier.add(keras.layers.Dense(64,activation='relu',kernel_regularizer=keras.regularizers.l2(0.001)))
classifier.add(keras.layers.Dense(10,activation='softmax'))

optimizer=keras.optimizers.Adam(learning_rate=0.001)

classifier.compile(optimizer=optimizer,loss='sparse_categorical_crossentropy',
                   metrics=['accuracy'])



his=classifier.fit(x_train,y_train,validation_data=(x_test,y_test),
               epochs=40,batch_size=32)


classifier.summary()


his.history.keys()



# Acurracy
import matplotlib.pyplot as plt

plt.plot(his.epoch,his.history['accuracy'],label='acc')
plt.plot(his.epoch,his.history['val_accuracy'],label='val')
plt.legend()
plt.show()

# Validation Loss

plt.plot(his.epoch,his.history['loss'],label='acc')
plt.plot(his.epoch,his.history['val_loss'],label='val')
plt.legend()
plt.show()