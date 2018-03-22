import csv

lines = []
with open('./data/driving_log.csv') as csvfile:
  reader = csv.reader(csvfile)
  for line in reader:
    lines.append(line)

lines = lines[1:]

import numpy as np

low = [line for line in lines if abs(float(line[3])) < 0.05]
print(len(low))
print('##################')
print(len(lines))
lines = [line for line in lines if not (abs(float(line[3])) < 0.05 and np.random.randint(10) < 8)]


from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(lines, test_size = 0.2)

print('Total trian samples: {}\nTotal valid samples: {}'.format(len(train_samples),len(validation_samples)))

import cv2
from sklearn.utils import shuffle

def generator(samples, batch_size = 32):
  num_samples = len(samples)
  while 1: # Loop forever so the generator nerver terminates
    shuffle(samples)
    for offset in range(0, num_samples, batch_size):
      batch_samples = samples[offset: offset + batch_size]

      images = []
      measurements = []
      for batch_sample in batch_samples:
        for i in range(3):
          source_path = batch_sample[i]
          filename = source_path.split('/')[-1]
          current_path = './data/IMG/' + filename
          image = cv2.cvtColor(cv2.imread(current_path), cv2.COLOR_BGR2RGB)
          images.append(image)
          measurement = float(line[3]) + 0.2 * (-1.5 * i * i + 2.5 * i)
          measurements.append(measurement)
          #augment the data set with flipping of the centre image
          if i == 0:
            images.append(cv2.flip(image, 1))
            measurements.append(measurement * -1.0)

      X_train = np.array(images)
      y_train = np.array(measurements)
      yield shuffle(X_train, y_train)

train_generator = generator(train_samples, batch_size = 32)
validation_generator = generator(validation_samples, batch_size = 32)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout, ELU, Conv2D
from keras.layers import BatchNormalization, Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

def LeNet():
  model = Sequential()
  model.add(Lambda(lambda x : (x / 255.0) - 0.5, input_shape = (160, 320, 3)))
  model.add(Cropping2D(cropping = ((70, 25), (0, 0))))
  model.add(Convolution2D(6, 5, 5, activation = 'relu'))
  model.add(MaxPooling2D())
  model.add(Convolution2D(6, 5, 5, activation = 'relu'))
  model.add(MaxPooling2D())
  model.add(Flatten())
  model.add(Dense(120))
  model.add(Dense(84))
  model.add(Dense(1))
  return model

def NvidiaNet():
  model = Sequential()
  model.add(Lambda(lambda x : (x / 127.5) - 1.0, input_shape = (160, 320, 3)))
  model.add(Cropping2D(cropping = ((70, 25), (0, 0))))
  model.add(Convolution2D(24, 5, 5, subsample = (2, 2), activation = 'relu')) 
   
  model.add(Convolution2D(36, 5, 5, subsample = (2, 2), activation = 'relu'))

  model.add(Convolution2D(48, 5, 5, subsample = (2, 2), activation = 'relu'))
 
  model.add(Convolution2D(64, 3, 3, activation = 'relu'))
  
  model.add(Convolution2D(64, 3, 3, activation = 'relu'))
  
  model.add(Flatten())
  
  model.add(Dense(100))
  model.add(Dropout(0.7))
  model.add(Activation('relu'))
  
  model.add(Dense(50))
  model.add(Activation('relu'))
  
  model.add(Dense(10))
  model.add(Activation('relu'))
  
  model.add(Dense(1))
  return model

from keras.models import load_model

model = NvidiaNet()
model.compile(loss = 'mse', optimizer = 'adam')
model.fit_generator(train_generator, samples_per_epoch = 4 * len(train_samples)  , validation_data = validation_generator, nb_val_samples = len(validation_samples) * 4, nb_epoch = 50)

model.save('model.h5')
exit()
