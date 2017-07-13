from keras.applications.inception_v3 import InceptionV3

from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout,Conv2D, BatchNormalization, MaxPooling2D, Flatten
from keras import backend as K
from DataGenerator import CarData
import sys
import cv2
import numpy as np
from keras.models import model_from_yaml

nbr_classes=196

Cars_Data=CarData()
Cars_Data.loadData()

generator=Cars_Data.TrainDataGenerate(400, aug=True)
generator2=Cars_Data.TrainDataGenerate(400, aug=False)

valGenerator=Cars_Data.ValidateDataGenerate(400)
testData,testLabels,labelNames,fileNames=Cars_Data.testData()


base_model = InceptionV3(weights='imagenet', include_top=False)
x = base_model.output
x = GlobalAveragePooling2D(name='mg1')(x)
x = BatchNormalization(name='mg33')(x)
x = Dropout(0.7)(x)
x = Dense(256, activation='sigmoid', name='mg2')(x)
x = BatchNormalization(name='mg3')(x)
x = Dropout(0.5, name='mg4')(x)
x = Dense(784, activation='relu', name='mg5')(x)

predictions = Dense(196, activation='softmax',name='mg6')(x)
model = Model(inputs=base_model.input, outputs=predictions)

for layer in model.layers[:310]:
   layer.trainable = False
for layer in model.layers[310:]:
   layer.trainable = True

model.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics=['categorical_accuracy'])
model.fit_generator(generator, samples_per_epoch=18,  verbose=2, 
		        nb_epoch=15,validation_data=valGenerator,
			validation_steps=16,max_q_size=1,workers=1)

model_yaml = model.to_yaml()
with open("cars.yaml", "w") as yaml_file:
    yaml_file.write(model_yaml)
model.save_weights("cars_weights.h5")
print("Saved model to disk")



yaml_file = open('cars.yaml', 'r')
loaded_model_yaml = yaml_file.read()
yaml_file.close()
loaded_model = model_from_yaml(loaded_model_yaml)
loaded_model.load_weights("cars_weights.h5")
print("Loaded model from disk")
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
 
for layer in loaded_model.layers[:100]:
   layer.trainable = False
for layer in loaded_model.layers[100:]:
   layer.trainable = True

from keras.optimizers import SGD
loaded_model.compile(optimizer=SGD(lr=0.005, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

loaded_model.fit_generator(generator, steps_per_epoch=60,  verbose=2, 
                        nb_epoch=4,validation_data=valGenerator,
                        validation_steps=45,max_q_size=1,workers=1)

model_yaml = loaded_model.to_yaml()
with open("cars.yaml", "w") as yaml_file:
    yaml_file.write(model_yaml)
loaded_model.save_weights("cars_weights.h5")
print("Saved model to disk")



yaml_file = open('cars.yaml', 'r')
loaded_model_yaml = yaml_file.read()
yaml_file.close()
loaded_model = model_from_yaml(loaded_model_yaml)
loaded_model.load_weights("cars_weights.h5")
print("Loaded model from disk")
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


loaded_model.compile(optimizer=SGD(lr=0.005, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

loaded_model.fit_generator(generator2, steps_per_epoch=60,  verbose=2, 
                        nb_epoch=2,validation_data=valGenerator,
                        validation_steps=45,max_q_size=1,workers=1)

right=0

for i in range(0,len(testData)):
        probs = loaded_model.predict(testData[np.newaxis, i])
        prediction = probs.argmax(axis=1)
        print("Predicted: {}, Actual: {},Name:{}, Image:{}".format(prediction[0],
                np.argmax(testLabels[i]), labelNames[i], fileNames[i]))
	if prediction[0]==np.argmax(testLabels[i]):
		right=right+1	

print "Totals"
print "Right:" 
print right
print "Out of:"
print len(testData)
print float(right)/float(len(testData))

