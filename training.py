import tensorflow as tf
import pickle
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten,Conv2D,MaxPooling2D

pickle_in = open(r"X.pickle","rb")
X = pickle.load(pickle_in)

pickle_in = open(r"y.pickle","rb")
y = pickle.load(pickle_in)

#print(X)
#print()
#print(y)

X = X/255.0

model = Sequential()

model.add(Conv2D(256, (3,3), input_shape=X.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(256, (3,3), input_shape=X.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(64))
model.add(Activation('relu'))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss="binary_crossentropy",optimizer="adam",metrics=["accuracy"])
model.fit(X,y,batch_size=4,epochs=10,validation_split=0.3)
model.save("CNN.model")
