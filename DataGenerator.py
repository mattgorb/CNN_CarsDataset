import numpy as np
import cv2
import sys
import scipy.io
import numpy as np
import os.path
from keras.utils import np_utils
from string import strip
import math
from keras.preprocessing.image import ImageDataGenerator
import os.path


class CarData:
   

   def __init__(self):
        loadMatCarNames=scipy.io.loadmat("meta/cars_annos.mat")
	self.carNames=np.array(loadMatCarNames['class_names'][0])


   def separateTrainTestData(self):
	loadMat = scipy.io.loadmat("meta/cars_annos.mat")
	self.imageAnnotation=np.array(loadMat['annotations'][0])
        np.random.shuffle(self.imageAnnotation)
	self.Cars85th= round((self.imageAnnotation.size-1)*.85)
	testData=self.imageAnnotation[int(self.Cars85th)+1:]
	trainData=self.imageAnnotation[:int(self.Cars85th)]
	scipy.io.savemat("meta/cars_annosTrain.mat", mdict={'data':np.array(trainData)})
	scipy.io.savemat("meta/cars_annosTest.mat", mdict={'data':np.array(testData)})
	self.imageAnnotation=np.array([])
	self.Cars85th=0;
		

   def loadData(self):
	if os.path.isfile('meta/cars_annosTrain.mat')==False or os.path.isfile('meta/cars_annosTest.mat')==False:
		self.separateTrainTestData()
	loadMat = scipy.io.loadmat("meta/cars_annosTest.mat")
	self.imageAnnotationTest=np.array(loadMat['data'])

	loadMat2 = scipy.io.loadmat("meta/cars_annosTrain.mat")
	self.imageAnnotation=np.array(loadMat2['data'])
	np.random.shuffle(self.imageAnnotation)
	self.Cars85th= round((self.imageAnnotation.size-1)*.85)

   def testData(self):
	self.dataArray=[]
	self.resultArray=[]
	self.resultNames=[]
	self.imageName=[]
	for index in range(0, int(self.imageAnnotationTest.size-1)):
		filename=self.imageAnnotationTest[0][index][0][0].astype(str).tostring()
		self.imageName.append(filename)
		img = cv2.imread(filename, cv2.IMREAD_COLOR)
		crop_img = img[self.imageAnnotationTest[0][index][2][0][0]:self.imageAnnotationTest[0][index][4][0][0], self.imageAnnotationTest[0][index][1][0][0]:self.imageAnnotationTest[0][index][3][0][0]]
		resized_image = cv2.resize(crop_img, (224, 224)) 
		#resized_image=np.transpose(resized_image)
		resized_image = resized_image.astype('float32')
		resized_image/=255
		self.dataArray.append(resized_image)
		self.resultArray.append(self.imageAnnotationTest[0][index][5][0][0])
#		print self.imageAnnotation[index][5][0][0]
		self.resultNames.append(self.carNames[self.imageAnnotationTest[0][index][5][0][0]-1][0].astype(str).tostring())
	self.resultArray=np.array(self.resultArray)
	self.resultArray=self.resultArray-1
	self.resultArray=np_utils.to_categorical(self.resultArray, 196)
	return np.array(self.dataArray), np.array(self.resultArray), np.array(self.resultNames), np.array(self.imageName)

   def TrainDataGenerate(self,batch_size, aug=True):
	rounds=4
        datagen = ImageDataGenerator(rotation_range=10,
		                width_shift_range=0.1,
		                height_shift_range=0.1,
		                horizontal_flip=False)
	batch_features=np.zeros((batch_size,224,224,3))
	batch_labels=np.zeros((batch_size,196))
	while True:
		batch_features=np.zeros(batch_features.shape)
		batch_labels=np.zeros(batch_labels.shape)
		for i in range(batch_size):
			index= int(np.random.choice(int(self.Cars85th),1)[0])
			filename=self.imageAnnotation[0][index][0][0].astype(str).tostring()
			img = cv2.imread(filename, cv2.IMREAD_COLOR)
			crop_img = img[self.imageAnnotation[0][index][2][0][0]:self.imageAnnotation[0][index][4][0][0], self.imageAnnotation[0][index][1][0][0]:self.imageAnnotation[0][index][3][0][0]]
			resized_image = cv2.resize(crop_img, (224, 224)) 
			resized_image=resized_image
			resized_image= resized_image.astype('float32')
			resized_image=resized_image = resized_image[np.newaxis, :, :, :]
			classNumber=self.imageAnnotation[0][index][5][0][0]-1
			out=np_utils.to_categorical(classNumber, 196)

			if(aug):
				x, y=next(datagen.flow(resized_image,out,batch_size=rounds))			
				x/=255
			else:
				resized_image/=255
				x=resized_image
				y=out
			batch_features[i] = x
			batch_labels[i] = y
		yield batch_features, batch_labels



   def ValidateDataGenerate(self,batch_size):
	while True:
		batch_features = np.zeros((batch_size, 224, 224, 3))
		batch_labels = np.zeros((batch_size,1))
		for i in range(batch_size):
			index= int(self.Cars85th+np.random.choice(int((int(self.imageAnnotation.size)-self.Cars85th)),1)[0])
			filename=self.imageAnnotation[0][index][0][0].astype(str).tostring()
			img = cv2.imread(filename, cv2.IMREAD_COLOR)
			crop_img = img[self.imageAnnotation[0][index][2][0][0]:self.imageAnnotation[0][index][4][0][0], self.imageAnnotation[0][index][1][0][0]:self.imageAnnotation[0][index][3][0][0]]
			resized_image = cv2.resize(crop_img, (224, 224)) 
			resized_image=resized_image
			resized_image = resized_image.astype('float32')
			resized_image/=255
			batch_features[i] = resized_image
			batch_labels[i] = self.imageAnnotation[0][index][5][0][0]

		batch_labels=batch_labels-1
		batch_labels=np_utils.to_categorical(batch_labels, 196)
		yield batch_features, batch_labels



