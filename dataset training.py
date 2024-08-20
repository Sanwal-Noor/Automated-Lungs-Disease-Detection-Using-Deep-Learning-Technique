

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

        
# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# check GPU  (I got a Tesla P100 today)
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras import Model

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# simply show a picture
plt.figure(figsize= (10, 10))
img = mpimg.imread("../input/lungs-disease-dataset-4-types/Lung Disease Dataset/train/Normal/test_0_9774.jpeg")
plt.imshow(img)
plt.show()

# ImageDataGenerator (only can adjust on training data)
traingen = ImageDataGenerator(rescale= 1./255,
                             width_shift_range=0.2 , 
                             height_shift_range=0.2 ,
                             zoom_range=0.2)
valgen = ImageDataGenerator(rescale= 1./255)
testgen = ImageDataGenerator(rescale= 1./255)
# flow_from_directory
train_it = traingen.flow_from_directory("../input/lungs-disease-dataset-4-types/Lung Disease Dataset/train", target_size = (224, 224))
val_it = traingen.flow_from_directory("../input/lungs-disease-dataset-4-types/Lung Disease Dataset/val", target_size = (224, 224))
test_it = traingen.flow_from_directory("../input/lungs-disease-dataset-4-types/Lung Disease Dataset/test", target_size = (224, 224))
Found 6054 images belonging to 5 classes.
Found 2016 images belonging to 5 classes.
Found 2025 images belonging to 5 classes.
# show the picture after ImageDataGenerator
plt.figure()
plt.imshow(next(train_it)[0][0])
plt.show()

# use pre-train model of DenseNet201
base_model_201 = tf.keras.applications.DenseNet201(input_shape = (224, 224, 3),
    include_top=False,
    weights="imagenet"
)

# lock layers
for layer in base_model_201.layers:
  layer.trainable = False

# set full connect layers
x = layers.Flatten()(base_model_201.output)  # base_model_201.output
x = layers.Dropout(0.5)(x) # 
x = layers.Dense(512, activation= 'relu')(x)
x = layers.Dense(5, activation = 'softmax')(x)

model2 = tf.keras.models.Model(base_model_201.input, x)  # keras.models not keras.model

# compile
model2.compile('adam', loss = 'categorical_crossentropy',metrics = ['acc'])


Earlystop
This time I forget to use earlystop. Maybe I need to use it next time (because loss and acc don't change at finall)

from tensorflow.keras.callbacks import ReduceLROnPlateau

del_dense = keras.models.load_model('densenet201.hdf5')
model_dense.evaluate(test_it, steps= 1)
 