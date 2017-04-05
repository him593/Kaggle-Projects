from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers import Dense
from keras.metrics import categorical_accuracy as accuracy
from keras.models import Sequential






def batch_iter(data,batch_size,num_epochs,shuffle=True):
    data=np.array(data)

    data_size=len(data)
    num_batches_per_epoch = int(len(data) / batch_size) + 1

    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

train=pd.read_csv('train.csv')

images=train.ix[:,1:].values

images=np.multiply(images,1./255.)

labels=train.ix[:,0].values

labels_encoded=np.zeros((len(images),10))

for i in range(len(images)):
    labels_encoded[i][labels[i]]=1

itrain,itest,jtrain,jtest=train_test_split(images,labels_encoded)

model = Sequential()
model.add(Dense(512,input_dim=784,init='uniform',activation='relu'))
model.add(Dense(512,init='uniform',activation='relu'))
model.add(Dense(512,init='uniform',activation='relu'))
model.add(Dense(512,init='uniform',activation='relu'))
model.add((Dense(512,init='uniform',activation='relu')))
model.add(Dense(10,init='uniform',activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(itrain, jtrain, nb_epoch=30, batch_size=64)


scores = model.evaluate(itest, jtest)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

test_images=pd.read_csv('test.csv')
test_images=test_images.astype(np.float)

test_images = np.multiply(test_images, 1.0 / 255.0)
test_images=np.array(test_images)
print test_images.shape
predictions=model.predict(test_images)
predicts= np.argmax(predictions,axis=1)
index=np.arange(1,len(predictions)+1,1)
results=pd.DataFrame(index,columns=['ImageId'])
results['Label']=predicts
results.to_csv('results_cnn.csv')