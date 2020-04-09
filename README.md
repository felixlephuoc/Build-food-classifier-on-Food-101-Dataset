
# Food Classification using Keras and TensorFlow on Food-101 Dataset

## Overview


   * Download and extract Food 101 dataset
   * Understand dataset structure and files
   * Visualize random image from each class in the dataset
   * Split the image data into train and test using train.txt and test.txt
   * Create a subset of data with few classes(3) - train_mini and test_mini for experimenting
   * Fine tune Inception Pretrained model using Food 101 dataset
   * Train model
   * Visualize accuracy and loss plots
   * Predicting classes for new images from internet
   * Scale up and fine tune Inceptionv3 model with 11 classes of data
   * Testing the trained model using images downloaded from Internet
   * Visualization of intermediate activations
   * Heat Map Visualizations of Class Activation
   


```python
# Restore previous notebook session
dill.load_session('notebook_env.db')
```


```python
import tensorflow as tf
import matplotlib.image as img
%matplotlib inline
import numpy as np
from collections import defaultdict
import collections
from shutil import copy
from shutil import copytree, rmtree
import keras.backend as K
from keras.models import load_model
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import os
import random
from keras import regularizers
from keras.applications.inception_v3 import InceptionV3
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D, AveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.optimizers import SGD
from keras.regularizers import l2
from tensorflow import keras
from keras import models
import cv2
```

    Using TensorFlow backend.


Check if GPU is enabled


```python
# Check if GPU is enabled
import tensorflow as tf
print(tf.__version__)
print(tf.test.gpu_device_name())
```

    1.8.0
    /device:GPU:0



```python
import tensorflow as tf
with tf.device('/gpu:0'):
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
    c = tf.matmul(a, b)

with tf.Session() as sess:
    print (sess.run(c))
```

    [[22. 28.]
     [49. 64.]]


## Download and Extract Food 101 Dataset


```python
# Helper function to download data and extract
def get_data_extract():
    if "food_101" in os.listdir("."):
        print("Dataset already exists")
    else:
        print("Downloading the data...")
        !wget http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz
        print("Dataset downloaded")
        print("Extracting data..")
        !tar xzvf food-101.tar.gz | head -10
        print("Extraction done!")
```

Since the datasdet can be downloaded from Kaggle, the following cell is commented


```python
# get_data_extract() 
```

## Understand dataset structure and files


The dataset being used is Food 101

   * This dataset has 101000 images in total. It's a food dataset with 101 categories(multiclass)
   * Each type of food has 750 training samples and 250 test samples
   * Note found on the webpage of the dataset :
   * On purpose, the training images were not cleaned, and thus still contain some amount of noise. This comes mostly in the form of intense colors and sometimes wrong labels. All images were rescaled to have a maximum side length of 512 pixels.
   * The entire dataset is 5GB in size




```python
# Check the extracted dataset folder
!ls food-101
```

    README.txt  images  license_agreement.txt  meta


- **images** folder contains 101 folders with 1000 images each
Each folder contains images of a specific food class, which are shown below:


```python
os.listdir('food-101/images')[0:20] # Showing first 20 folders only
```




    ['steak',
     'grilled_cheese_sandwich',
     'hamburger',
     'fish_and_chips',
     'red_velvet_cake',
     'deviled_eggs',
     'pizza',
     'donuts',
     'eggs_benedict',
     'pulled_pork_sandwich',
     'french_onion_soup',
     'creme_brulee',
     'lobster_roll_sandwich',
     'ceviche',
     'ice_cream',
     'cheesecake',
     'baklava',
     'escargots',
     'croque_madame',
     'scallops']



- **meta** folder contains the text files - train.txt and test.txt
- **train.txt** contains the list of images that belong to training set
- **test.txt** contains the list of images that belong to test set
- **classes.txt** contains the list of all classes of food


```python
os.listdir('food-101/meta')
```




    ['test.json',
     'train.json',
     'test.txt',
     'train.txt',
     'labels.txt',
     'classes.txt']




```python
!head food-101/meta/train.txt
```

    apple_pie/1005649
    apple_pie/1014775
    apple_pie/1026328
    apple_pie/1028787
    apple_pie/1043283
    apple_pie/1050519
    apple_pie/1057749
    apple_pie/1057810
    apple_pie/1072416
    apple_pie/1074856



```python
!head food-101/meta/test.txt
```

    apple_pie/1011328
    apple_pie/101251
    apple_pie/1034399
    apple_pie/103801
    apple_pie/1038694
    apple_pie/1047447
    apple_pie/1068632
    apple_pie/110043
    apple_pie/1106961
    apple_pie/1113017



```python
!head food-101/meta/classes.txt
```

    apple_pie
    baby_back_ribs
    baklava
    beef_carpaccio
    beef_tartare
    beet_salad
    beignets
    bibimbap
    bread_pudding
    breakfast_burrito


## Visualize random image from each of the 101 classes¶


```python
# Visualize the data, showing one image per class from 101 classes
rows = 26
cols = 4
fig, ax = plt.subplots(rows, cols, figsize=(20,50))
fig.suptitle("Showing one random image from each class", y=1.05, fontsize=28, fontweight='bold')
data_dir = "food_101/images/"
food_sorted = sorted(os.listdir(data_dir))
food_id = 1
for i in range(rows):
    for j in range(cols):
        try:
            food_selected = food_sorted[food_id]
            food_id += 1
        except:
            break
        if food_selected == '.DS_Store':
            continue
        food_selected_images = os.listdir(os.path.join(data_dir, food_selected)) # returns the
        # list of all files present in each food category
        food_selected_random = np.random.choice(food_selected_images) #  picks one food item from 
        # the list as choice, takes a list and returns one random item
        img = plt.imread(os.path.join(data_dir, food_selected, food_selected_random))
        ax[i][j].imshow(img)
        ax[i][j].set_title(food_selected, pad = 10, fontsize=18)
        
plt.setp(ax, xticks =[], yticks=[])
plt.tight_layout()
```


![png](2.%20Multiclass%20Classification%20using%20Keras%20and%20TensorFlow%20on%20Food-101%20Dataset_files/2.%20Multiclass%20Classification%20using%20Keras%20and%20TensorFlow%20on%20Food-101%20Dataset_21_0.png)


## Split the image data into train and test using train.txt and test.txt


```python
# Helper method to split dataset into train and test folders
def prepare_data(filepath, src, dest):
    classes_images = defaultdict(list)
    with open(filepath, 'r') as txt:
        paths = [read.strip() for read in txt.readlines()]
        for p in paths:
            food = p.split('/')
            classes_images[food[0]].append(food[1] + '.jpg')
    
    for food in classes_images.keys():
        print "Copying images into ", food
        if not os.path.exists(os.path.join(dest, food)):
            os.makedirs(os.path.join(dest, food))
        for i in classes_images[food]:
            copy(os.path.join(src, food, i), os.path.join(dest, food, i))
    print("Copying Done!")
```


```python
# Prepare train dataset by copying images from food-101/images to food-101/train 
# using the file train.txt
print("Creating train data...")
prepare_data('./food_101/meta/train.txt', './food_101/images/', 'train')
```

    Creating train data...
    Copying images into  churros
    Copying images into  hot_and_sour_soup
    Copying images into  samosa
    Copying images into  sashimi
    Copying images into  pork_chop
    Copying images into  spring_rolls
    Copying images into  panna_cotta
    Copying images into  beef_tartare
    Copying images into  greek_salad
    Copying images into  foie_gras
    Copying images into  tacos
    Copying images into  pad_thai
    Copying images into  poutine
    Copying images into  ramen
    Copying images into  pulled_pork_sandwich
    Copying images into  bibimbap
    Copying images into  beignets
    Copying images into  apple_pie
    Copying images into  crab_cakes
    Copying images into  risotto
    Copying images into  paella
    Copying images into  steak
    Copying images into  baby_back_ribs
    Copying images into  miso_soup
    Copying images into  frozen_yogurt
    Copying images into  club_sandwich
    Copying images into  carrot_cake
    Copying images into  falafel
    Copying images into  bread_pudding
    Copying images into  chicken_wings
    Copying images into  gnocchi
    Copying images into  caprese_salad
    Copying images into  creme_brulee
    Copying images into  escargots
    Copying images into  chocolate_cake
    Copying images into  tiramisu
    Copying images into  spaghetti_bolognese
    Copying images into  mussels
    Copying images into  scallops
    Copying images into  baklava
    Copying images into  edamame
    Copying images into  macaroni_and_cheese
    Copying images into  pancakes
    Copying images into  garlic_bread
    Copying images into  beet_salad
    Copying images into  onion_rings
    Copying images into  red_velvet_cake
    Copying images into  grilled_salmon
    Copying images into  chicken_curry
    Copying images into  deviled_eggs
    Copying images into  caesar_salad
    Copying images into  hummus
    Copying images into  fish_and_chips
    Copying images into  lasagna
    Copying images into  peking_duck
    Copying images into  guacamole
    Copying images into  strawberry_shortcake
    Copying images into  clam_chowder
    Copying images into  croque_madame
    Copying images into  french_onion_soup
    Copying images into  beef_carpaccio
    Copying images into  fried_rice
    Copying images into  donuts
    Copying images into  gyoza
    Copying images into  ravioli
    Copying images into  fried_calamari
    Copying images into  spaghetti_carbonara
    Copying images into  french_toast
    Copying images into  lobster_bisque
    Copying images into  ceviche
    Copying images into  bruschetta
    Copying images into  french_fries
    Copying images into  shrimp_and_grits
    Copying images into  filet_mignon
    Copying images into  hamburger
    Copying images into  dumplings
    Copying images into  tuna_tartare
    Copying images into  sushi
    Copying images into  cheese_plate
    Copying images into  eggs_benedict
    Copying images into  cup_cakes
    Copying images into  takoyaki
    Copying images into  chocolate_mousse
    Copying images into  breakfast_burrito
    Copying images into  hot_dog
    Copying images into  macarons
    Copying images into  waffles
    Copying images into  seaweed_salad
    Copying images into  cannoli
    Copying images into  huevos_rancheros
    Copying images into  pizza
    Copying images into  chicken_quesadilla
    Copying images into  pho
    Copying images into  prime_rib
    Copying images into  cheesecake
    Copying images into  ice_cream
    Copying images into  omelette
    Copying images into  grilled_cheese_sandwich
    Copying images into  lobster_roll_sandwich
    Copying images into  nachos
    Copying images into  oysters
    Copying Done!



```python
# Prepare test data by copying images from food-101/images to food-101/
# test using the file test.txt
print("Creating test data...")
prepare_data('./food_101/meta/test.txt', './food_101/images', 'test')
```

    Creating test data...
    Copying images into  churros
    Copying images into  hot_and_sour_soup
    Copying images into  samosa
    Copying images into  sashimi
    Copying images into  pork_chop
    Copying images into  spring_rolls
    Copying images into  panna_cotta
    Copying images into  beef_tartare
    Copying images into  greek_salad
    Copying images into  foie_gras
    Copying images into  tacos
    Copying images into  pad_thai
    Copying images into  poutine
    Copying images into  ramen
    Copying images into  pulled_pork_sandwich
    Copying images into  bibimbap
    Copying images into  beignets
    Copying images into  apple_pie
    Copying images into  crab_cakes
    Copying images into  risotto
    Copying images into  paella
    Copying images into  steak
    Copying images into  baby_back_ribs
    Copying images into  miso_soup
    Copying images into  frozen_yogurt
    Copying images into  club_sandwich
    Copying images into  carrot_cake
    Copying images into  falafel
    Copying images into  bread_pudding
    Copying images into  chicken_wings
    Copying images into  gnocchi
    Copying images into  caprese_salad
    Copying images into  creme_brulee
    Copying images into  escargots
    Copying images into  chocolate_cake
    Copying images into  tiramisu
    Copying images into  spaghetti_bolognese
    Copying images into  mussels
    Copying images into  scallops
    Copying images into  baklava
    Copying images into  edamame
    Copying images into  macaroni_and_cheese
    Copying images into  pancakes
    Copying images into  garlic_bread
    Copying images into  beet_salad
    Copying images into  onion_rings
    Copying images into  red_velvet_cake
    Copying images into  grilled_salmon
    Copying images into  chicken_curry
    Copying images into  deviled_eggs
    Copying images into  caesar_salad
    Copying images into  hummus
    Copying images into  fish_and_chips
    Copying images into  lasagna
    Copying images into  peking_duck
    Copying images into  guacamole
    Copying images into  strawberry_shortcake
    Copying images into  clam_chowder
    Copying images into  croque_madame
    Copying images into  french_onion_soup
    Copying images into  beef_carpaccio
    Copying images into  fried_rice
    Copying images into  donuts
    Copying images into  gyoza
    Copying images into  ravioli
    Copying images into  fried_calamari
    Copying images into  spaghetti_carbonara
    Copying images into  french_toast
    Copying images into  lobster_bisque
    Copying images into  ceviche
    Copying images into  bruschetta
    Copying images into  french_fries
    Copying images into  shrimp_and_grits
    Copying images into  filet_mignon
    Copying images into  hamburger
    Copying images into  dumplings
    Copying images into  tuna_tartare
    Copying images into  sushi
    Copying images into  cheese_plate
    Copying images into  eggs_benedict
    Copying images into  cup_cakes
    Copying images into  takoyaki
    Copying images into  chocolate_mousse
    Copying images into  breakfast_burrito
    Copying images into  hot_dog
    Copying images into  macarons
    Copying images into  waffles
    Copying images into  seaweed_salad
    Copying images into  cannoli
    Copying images into  huevos_rancheros
    Copying images into  pizza
    Copying images into  chicken_quesadilla
    Copying images into  pho
    Copying images into  prime_rib
    Copying images into  cheesecake
    Copying images into  ice_cream
    Copying images into  omelette
    Copying images into  grilled_cheese_sandwich
    Copying images into  lobster_roll_sandwich
    Copying images into  nachos
    Copying images into  oysters
    Copying Done!



```python
# Check how many images in the train folder
print("Total number of samples in train folder")
!find train -type d -or -type f -printf '.' |wc -c
```

    Total number of samples in train folder
    75750



```python
# Check how many files are in the test folder
print("Total number of samples in test folder")
!find test -type d -or -type f -printf '.' | wc -c
```

    Total number of samples in test folder
    25250


## Create a subset of data with few classes(3) - train_mini and test_mini for experimenting



   * We now have train and test data ready
   * But to experiment and try different architectures, working on the whole data with 101 classes takes a lot of time and computation
   * To proceed with further experiments, I am creating train_min and test_mini, limiting the dataset to 3 classes
   * Since the original problem is multiclass classification which makes key aspects of architectural decisions different from that of binary classification, choosing 3 classes is a good start instead of 2




```python
# List of all 101 types of foods(sorted alphabetically)
del food_sorted[0] # remove .DS_Store from the list
food_sorted
```




    ['apple_pie',
     'baby_back_ribs',
     'baklava',
     'beef_carpaccio',
     'beef_tartare',
     'beet_salad',
     'beignets',
     'bibimbap',
     'bread_pudding',
     'breakfast_burrito',
     'bruschetta',
     'caesar_salad',
     'cannoli',
     'caprese_salad',
     'carrot_cake',
     'ceviche',
     'cheese_plate',
     'cheesecake',
     'chicken_curry',
     'chicken_quesadilla',
     'chicken_wings',
     'chocolate_cake',
     'chocolate_mousse',
     'churros',
     'clam_chowder',
     'club_sandwich',
     'crab_cakes',
     'creme_brulee',
     'croque_madame',
     'cup_cakes',
     'deviled_eggs',
     'donuts',
     'dumplings',
     'edamame',
     'eggs_benedict',
     'escargots',
     'falafel',
     'filet_mignon',
     'fish_and_chips',
     'foie_gras',
     'french_fries',
     'french_onion_soup',
     'french_toast',
     'fried_calamari',
     'fried_rice',
     'frozen_yogurt',
     'garlic_bread',
     'gnocchi',
     'greek_salad',
     'grilled_cheese_sandwich',
     'grilled_salmon',
     'guacamole',
     'gyoza',
     'hamburger',
     'hot_and_sour_soup',
     'hot_dog',
     'huevos_rancheros',
     'hummus',
     'ice_cream',
     'lasagna',
     'lobster_bisque',
     'lobster_roll_sandwich',
     'macaroni_and_cheese',
     'macarons',
     'miso_soup',
     'mussels',
     'nachos',
     'omelette',
     'onion_rings',
     'oysters',
     'pad_thai',
     'paella',
     'pancakes',
     'panna_cotta',
     'peking_duck',
     'pho',
     'pizza',
     'pork_chop',
     'poutine',
     'prime_rib',
     'pulled_pork_sandwich',
     'ramen',
     'ravioli',
     'red_velvet_cake',
     'risotto',
     'samosa',
     'sashimi',
     'scallops',
     'seaweed_salad',
     'shrimp_and_grits',
     'spaghetti_bolognese',
     'spaghetti_carbonara',
     'spring_rolls',
     'steak',
     'strawberry_shortcake',
     'sushi',
     'tacos',
     'takoyaki',
     'tiramisu',
     'tuna_tartare',
     'waffles']




```python
# Helper method to create train_mini and test_mini data samples
def dataset_mini(food_list, src, dest):
    if os.path.exists(dest):
        rmtree(dest) # # removing dataset_mini(if it already exists) 
        # folders so that we will have only the classes that we want
        os.makedirs(dest)
    for food_item in food_list:
        print "Copying images into ", food_item
        copytree(os.path.join(src, food_item), os.path.join(dest, food_item))   
```


```python
# picking 3 food items and generating separate data folders for the same
food_list = ['apple_pie', 'pizza', 'omelette']
src_train = 'train'
dest_train = 'train_mini'
src_test = 'test'
dest_test = 'test_mini'
```


```python
print("Create train data folder with new classes")
dataset_mini(food_list, src_train, dest_train)
```

    Create train data folder with new classes
    Copying images into  apple_pie
    Copying images into  pizza
    Copying images into  omelette



```python
print("Create test data folder with new classes")
dataset_mini(food_list, src_test, dest_test)
```

    Create test data folder with new classes
    Copying images into  apple_pie
    Copying images into  pizza
    Copying images into  omelette



```python
print("Total number of samples in mini train folder")
!find train_mini -type d -or -type f -printf '.' | wc -c
```

    Total number of samples in mini train folder
    2250



```python
print("Total number of samples in mini test folder")
!find test_mini -type d -or -type f -printf '.' | wc -c
```

    Total number of samples in mini test folder
    750


## Fine tune Inception Pretrained model using Food 101 dataset

   * Keras and other Deep Learning libraries provide pretrained models
   * These are deep neural networks with efficient architectures(like VGG,Inception,ResNet) that are already trained on datasets like ImageNet
   * Using these pretrained models, we can use the already learned weights and add few layers on top to finetune the model to our new data
   * This helps in faster convergance and saves time and computation when compared to models trained from scratch
   * We currently have a subset of dataset with 3 classes - samosa, pizza and omelette
    Use the below code to finetune Inceptionv3 pretrained model




```python
K.clear_session()
n_classes = 3
img_width, img_height = 299, 299 
train_data_dir = 'train_mini'
validation_data_dir = 'test_mini'
nb_train_samples = 2250 # 75,750
nb_validation_samples = 750 # 25,250
batch_size = 16

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size = batch_size,
    class_mode = 'categorical')
validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size = (img_height, img_width),
    batch_size = batch_size,
    class_mode = 'categorical')

inception = InceptionV3(weights='imagenet', include_top=False)
x = inception.output
# Add one FC layer
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.2)(x)

predictions = Dense(3, kernel_regularizer= regularizers.l2(0.005), activation='softmax')(x)

model = Model(input = inception.input, output = predictions)
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', 
              metrics=['accuracy'])
checkpoint = ModelCheckpoint(filepath='best_model_3class.hdf5', verbose=1, save_best_only=True)
csv_logger = CSVLogger('history_3class.log')
```

    Found 2250 images belonging to 3 classes.
    Found 750 images belonging to 3 classes.


    /usr/local/lib/python2.7/dist-packages/ipykernel_launcher.py:37: UserWarning: Update your `Model` call to the Keras 2 API: `Model(outputs=Tensor("de..., inputs=Tensor("in...)`



```python
## Check model architecture
print(model.summary())
```

    __________________________________________________________________________________________________
    Layer (type)                    Output Shape         Param #     Connected to                     
    ==================================================================================================
    input_1 (InputLayer)            (None, None, None, 3 0                                            
    __________________________________________________________________________________________________
    conv2d_1 (Conv2D)               (None, None, None, 3 864         input_1[0][0]                    
    __________________________________________________________________________________________________
    batch_normalization_1 (BatchNor (None, None, None, 3 96          conv2d_1[0][0]                   
    __________________________________________________________________________________________________
    activation_1 (Activation)       (None, None, None, 3 0           batch_normalization_1[0][0]      
    __________________________________________________________________________________________________
    conv2d_2 (Conv2D)               (None, None, None, 3 9216        activation_1[0][0]               
    __________________________________________________________________________________________________
    batch_normalization_2 (BatchNor (None, None, None, 3 96          conv2d_2[0][0]                   
    __________________________________________________________________________________________________
    activation_2 (Activation)       (None, None, None, 3 0           batch_normalization_2[0][0]      
    __________________________________________________________________________________________________
    conv2d_3 (Conv2D)               (None, None, None, 6 18432       activation_2[0][0]               
    __________________________________________________________________________________________________
    batch_normalization_3 (BatchNor (None, None, None, 6 192         conv2d_3[0][0]                   
    __________________________________________________________________________________________________
    activation_3 (Activation)       (None, None, None, 6 0           batch_normalization_3[0][0]      
    __________________________________________________________________________________________________
    max_pooling2d_1 (MaxPooling2D)  (None, None, None, 6 0           activation_3[0][0]               
    __________________________________________________________________________________________________
    conv2d_4 (Conv2D)               (None, None, None, 8 5120        max_pooling2d_1[0][0]            
    __________________________________________________________________________________________________
    batch_normalization_4 (BatchNor (None, None, None, 8 240         conv2d_4[0][0]                   
    __________________________________________________________________________________________________
    activation_4 (Activation)       (None, None, None, 8 0           batch_normalization_4[0][0]      
    __________________________________________________________________________________________________
    conv2d_5 (Conv2D)               (None, None, None, 1 138240      activation_4[0][0]               
    __________________________________________________________________________________________________
    batch_normalization_5 (BatchNor (None, None, None, 1 576         conv2d_5[0][0]                   
    __________________________________________________________________________________________________
    activation_5 (Activation)       (None, None, None, 1 0           batch_normalization_5[0][0]      
    __________________________________________________________________________________________________
    max_pooling2d_2 (MaxPooling2D)  (None, None, None, 1 0           activation_5[0][0]               
    __________________________________________________________________________________________________
    conv2d_9 (Conv2D)               (None, None, None, 6 12288       max_pooling2d_2[0][0]            
    __________________________________________________________________________________________________
    batch_normalization_9 (BatchNor (None, None, None, 6 192         conv2d_9[0][0]                   
    __________________________________________________________________________________________________
    activation_9 (Activation)       (None, None, None, 6 0           batch_normalization_9[0][0]      
    __________________________________________________________________________________________________
    conv2d_7 (Conv2D)               (None, None, None, 4 9216        max_pooling2d_2[0][0]            
    __________________________________________________________________________________________________
    conv2d_10 (Conv2D)              (None, None, None, 9 55296       activation_9[0][0]               
    __________________________________________________________________________________________________
    batch_normalization_7 (BatchNor (None, None, None, 4 144         conv2d_7[0][0]                   
    __________________________________________________________________________________________________
    batch_normalization_10 (BatchNo (None, None, None, 9 288         conv2d_10[0][0]                  
    __________________________________________________________________________________________________
    activation_7 (Activation)       (None, None, None, 4 0           batch_normalization_7[0][0]      
    __________________________________________________________________________________________________
    activation_10 (Activation)      (None, None, None, 9 0           batch_normalization_10[0][0]     
    __________________________________________________________________________________________________
    average_pooling2d_1 (AveragePoo (None, None, None, 1 0           max_pooling2d_2[0][0]            
    __________________________________________________________________________________________________
    conv2d_6 (Conv2D)               (None, None, None, 6 12288       max_pooling2d_2[0][0]            
    __________________________________________________________________________________________________
    conv2d_8 (Conv2D)               (None, None, None, 6 76800       activation_7[0][0]               
    __________________________________________________________________________________________________
    conv2d_11 (Conv2D)              (None, None, None, 9 82944       activation_10[0][0]              
    __________________________________________________________________________________________________
    conv2d_12 (Conv2D)              (None, None, None, 3 6144        average_pooling2d_1[0][0]        
    __________________________________________________________________________________________________
    batch_normalization_6 (BatchNor (None, None, None, 6 192         conv2d_6[0][0]                   
    __________________________________________________________________________________________________
    batch_normalization_8 (BatchNor (None, None, None, 6 192         conv2d_8[0][0]                   
    __________________________________________________________________________________________________
    batch_normalization_11 (BatchNo (None, None, None, 9 288         conv2d_11[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_12 (BatchNo (None, None, None, 3 96          conv2d_12[0][0]                  
    __________________________________________________________________________________________________
    activation_6 (Activation)       (None, None, None, 6 0           batch_normalization_6[0][0]      
    __________________________________________________________________________________________________
    activation_8 (Activation)       (None, None, None, 6 0           batch_normalization_8[0][0]      
    __________________________________________________________________________________________________
    activation_11 (Activation)      (None, None, None, 9 0           batch_normalization_11[0][0]     
    __________________________________________________________________________________________________
    activation_12 (Activation)      (None, None, None, 3 0           batch_normalization_12[0][0]     
    __________________________________________________________________________________________________
    mixed0 (Concatenate)            (None, None, None, 2 0           activation_6[0][0]               
                                                                     activation_8[0][0]               
                                                                     activation_11[0][0]              
                                                                     activation_12[0][0]              
    __________________________________________________________________________________________________
    conv2d_16 (Conv2D)              (None, None, None, 6 16384       mixed0[0][0]                     
    __________________________________________________________________________________________________
    batch_normalization_16 (BatchNo (None, None, None, 6 192         conv2d_16[0][0]                  
    __________________________________________________________________________________________________
    activation_16 (Activation)      (None, None, None, 6 0           batch_normalization_16[0][0]     
    __________________________________________________________________________________________________
    conv2d_14 (Conv2D)              (None, None, None, 4 12288       mixed0[0][0]                     
    __________________________________________________________________________________________________
    conv2d_17 (Conv2D)              (None, None, None, 9 55296       activation_16[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_14 (BatchNo (None, None, None, 4 144         conv2d_14[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_17 (BatchNo (None, None, None, 9 288         conv2d_17[0][0]                  
    __________________________________________________________________________________________________
    activation_14 (Activation)      (None, None, None, 4 0           batch_normalization_14[0][0]     
    __________________________________________________________________________________________________
    activation_17 (Activation)      (None, None, None, 9 0           batch_normalization_17[0][0]     
    __________________________________________________________________________________________________
    average_pooling2d_2 (AveragePoo (None, None, None, 2 0           mixed0[0][0]                     
    __________________________________________________________________________________________________
    conv2d_13 (Conv2D)              (None, None, None, 6 16384       mixed0[0][0]                     
    __________________________________________________________________________________________________
    conv2d_15 (Conv2D)              (None, None, None, 6 76800       activation_14[0][0]              
    __________________________________________________________________________________________________
    conv2d_18 (Conv2D)              (None, None, None, 9 82944       activation_17[0][0]              
    __________________________________________________________________________________________________
    conv2d_19 (Conv2D)              (None, None, None, 6 16384       average_pooling2d_2[0][0]        
    __________________________________________________________________________________________________
    batch_normalization_13 (BatchNo (None, None, None, 6 192         conv2d_13[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_15 (BatchNo (None, None, None, 6 192         conv2d_15[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_18 (BatchNo (None, None, None, 9 288         conv2d_18[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_19 (BatchNo (None, None, None, 6 192         conv2d_19[0][0]                  
    __________________________________________________________________________________________________
    activation_13 (Activation)      (None, None, None, 6 0           batch_normalization_13[0][0]     
    __________________________________________________________________________________________________
    activation_15 (Activation)      (None, None, None, 6 0           batch_normalization_15[0][0]     
    __________________________________________________________________________________________________
    activation_18 (Activation)      (None, None, None, 9 0           batch_normalization_18[0][0]     
    __________________________________________________________________________________________________
    activation_19 (Activation)      (None, None, None, 6 0           batch_normalization_19[0][0]     
    __________________________________________________________________________________________________
    mixed1 (Concatenate)            (None, None, None, 2 0           activation_13[0][0]              
                                                                     activation_15[0][0]              
                                                                     activation_18[0][0]              
                                                                     activation_19[0][0]              
    __________________________________________________________________________________________________
    conv2d_23 (Conv2D)              (None, None, None, 6 18432       mixed1[0][0]                     
    __________________________________________________________________________________________________
    batch_normalization_23 (BatchNo (None, None, None, 6 192         conv2d_23[0][0]                  
    __________________________________________________________________________________________________
    activation_23 (Activation)      (None, None, None, 6 0           batch_normalization_23[0][0]     
    __________________________________________________________________________________________________
    conv2d_21 (Conv2D)              (None, None, None, 4 13824       mixed1[0][0]                     
    __________________________________________________________________________________________________
    conv2d_24 (Conv2D)              (None, None, None, 9 55296       activation_23[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_21 (BatchNo (None, None, None, 4 144         conv2d_21[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_24 (BatchNo (None, None, None, 9 288         conv2d_24[0][0]                  
    __________________________________________________________________________________________________
    activation_21 (Activation)      (None, None, None, 4 0           batch_normalization_21[0][0]     
    __________________________________________________________________________________________________
    activation_24 (Activation)      (None, None, None, 9 0           batch_normalization_24[0][0]     
    __________________________________________________________________________________________________
    average_pooling2d_3 (AveragePoo (None, None, None, 2 0           mixed1[0][0]                     
    __________________________________________________________________________________________________
    conv2d_20 (Conv2D)              (None, None, None, 6 18432       mixed1[0][0]                     
    __________________________________________________________________________________________________
    conv2d_22 (Conv2D)              (None, None, None, 6 76800       activation_21[0][0]              
    __________________________________________________________________________________________________
    conv2d_25 (Conv2D)              (None, None, None, 9 82944       activation_24[0][0]              
    __________________________________________________________________________________________________
    conv2d_26 (Conv2D)              (None, None, None, 6 18432       average_pooling2d_3[0][0]        
    __________________________________________________________________________________________________
    batch_normalization_20 (BatchNo (None, None, None, 6 192         conv2d_20[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_22 (BatchNo (None, None, None, 6 192         conv2d_22[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_25 (BatchNo (None, None, None, 9 288         conv2d_25[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_26 (BatchNo (None, None, None, 6 192         conv2d_26[0][0]                  
    __________________________________________________________________________________________________
    activation_20 (Activation)      (None, None, None, 6 0           batch_normalization_20[0][0]     
    __________________________________________________________________________________________________
    activation_22 (Activation)      (None, None, None, 6 0           batch_normalization_22[0][0]     
    __________________________________________________________________________________________________
    activation_25 (Activation)      (None, None, None, 9 0           batch_normalization_25[0][0]     
    __________________________________________________________________________________________________
    activation_26 (Activation)      (None, None, None, 6 0           batch_normalization_26[0][0]     
    __________________________________________________________________________________________________
    mixed2 (Concatenate)            (None, None, None, 2 0           activation_20[0][0]              
                                                                     activation_22[0][0]              
                                                                     activation_25[0][0]              
                                                                     activation_26[0][0]              
    __________________________________________________________________________________________________
    conv2d_28 (Conv2D)              (None, None, None, 6 18432       mixed2[0][0]                     
    __________________________________________________________________________________________________
    batch_normalization_28 (BatchNo (None, None, None, 6 192         conv2d_28[0][0]                  
    __________________________________________________________________________________________________
    activation_28 (Activation)      (None, None, None, 6 0           batch_normalization_28[0][0]     
    __________________________________________________________________________________________________
    conv2d_29 (Conv2D)              (None, None, None, 9 55296       activation_28[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_29 (BatchNo (None, None, None, 9 288         conv2d_29[0][0]                  
    __________________________________________________________________________________________________
    activation_29 (Activation)      (None, None, None, 9 0           batch_normalization_29[0][0]     
    __________________________________________________________________________________________________
    conv2d_27 (Conv2D)              (None, None, None, 3 995328      mixed2[0][0]                     
    __________________________________________________________________________________________________
    conv2d_30 (Conv2D)              (None, None, None, 9 82944       activation_29[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_27 (BatchNo (None, None, None, 3 1152        conv2d_27[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_30 (BatchNo (None, None, None, 9 288         conv2d_30[0][0]                  
    __________________________________________________________________________________________________
    activation_27 (Activation)      (None, None, None, 3 0           batch_normalization_27[0][0]     
    __________________________________________________________________________________________________
    activation_30 (Activation)      (None, None, None, 9 0           batch_normalization_30[0][0]     
    __________________________________________________________________________________________________
    max_pooling2d_3 (MaxPooling2D)  (None, None, None, 2 0           mixed2[0][0]                     
    __________________________________________________________________________________________________
    mixed3 (Concatenate)            (None, None, None, 7 0           activation_27[0][0]              
                                                                     activation_30[0][0]              
                                                                     max_pooling2d_3[0][0]            
    __________________________________________________________________________________________________
    conv2d_35 (Conv2D)              (None, None, None, 1 98304       mixed3[0][0]                     
    __________________________________________________________________________________________________
    batch_normalization_35 (BatchNo (None, None, None, 1 384         conv2d_35[0][0]                  
    __________________________________________________________________________________________________
    activation_35 (Activation)      (None, None, None, 1 0           batch_normalization_35[0][0]     
    __________________________________________________________________________________________________
    conv2d_36 (Conv2D)              (None, None, None, 1 114688      activation_35[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_36 (BatchNo (None, None, None, 1 384         conv2d_36[0][0]                  
    __________________________________________________________________________________________________
    activation_36 (Activation)      (None, None, None, 1 0           batch_normalization_36[0][0]     
    __________________________________________________________________________________________________
    conv2d_32 (Conv2D)              (None, None, None, 1 98304       mixed3[0][0]                     
    __________________________________________________________________________________________________
    conv2d_37 (Conv2D)              (None, None, None, 1 114688      activation_36[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_32 (BatchNo (None, None, None, 1 384         conv2d_32[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_37 (BatchNo (None, None, None, 1 384         conv2d_37[0][0]                  
    __________________________________________________________________________________________________
    activation_32 (Activation)      (None, None, None, 1 0           batch_normalization_32[0][0]     
    __________________________________________________________________________________________________
    activation_37 (Activation)      (None, None, None, 1 0           batch_normalization_37[0][0]     
    __________________________________________________________________________________________________
    conv2d_33 (Conv2D)              (None, None, None, 1 114688      activation_32[0][0]              
    __________________________________________________________________________________________________
    conv2d_38 (Conv2D)              (None, None, None, 1 114688      activation_37[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_33 (BatchNo (None, None, None, 1 384         conv2d_33[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_38 (BatchNo (None, None, None, 1 384         conv2d_38[0][0]                  
    __________________________________________________________________________________________________
    activation_33 (Activation)      (None, None, None, 1 0           batch_normalization_33[0][0]     
    __________________________________________________________________________________________________
    activation_38 (Activation)      (None, None, None, 1 0           batch_normalization_38[0][0]     
    __________________________________________________________________________________________________
    average_pooling2d_4 (AveragePoo (None, None, None, 7 0           mixed3[0][0]                     
    __________________________________________________________________________________________________
    conv2d_31 (Conv2D)              (None, None, None, 1 147456      mixed3[0][0]                     
    __________________________________________________________________________________________________
    conv2d_34 (Conv2D)              (None, None, None, 1 172032      activation_33[0][0]              
    __________________________________________________________________________________________________
    conv2d_39 (Conv2D)              (None, None, None, 1 172032      activation_38[0][0]              
    __________________________________________________________________________________________________
    conv2d_40 (Conv2D)              (None, None, None, 1 147456      average_pooling2d_4[0][0]        
    __________________________________________________________________________________________________
    batch_normalization_31 (BatchNo (None, None, None, 1 576         conv2d_31[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_34 (BatchNo (None, None, None, 1 576         conv2d_34[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_39 (BatchNo (None, None, None, 1 576         conv2d_39[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_40 (BatchNo (None, None, None, 1 576         conv2d_40[0][0]                  
    __________________________________________________________________________________________________
    activation_31 (Activation)      (None, None, None, 1 0           batch_normalization_31[0][0]     
    __________________________________________________________________________________________________
    activation_34 (Activation)      (None, None, None, 1 0           batch_normalization_34[0][0]     
    __________________________________________________________________________________________________
    activation_39 (Activation)      (None, None, None, 1 0           batch_normalization_39[0][0]     
    __________________________________________________________________________________________________
    activation_40 (Activation)      (None, None, None, 1 0           batch_normalization_40[0][0]     
    __________________________________________________________________________________________________
    mixed4 (Concatenate)            (None, None, None, 7 0           activation_31[0][0]              
                                                                     activation_34[0][0]              
                                                                     activation_39[0][0]              
                                                                     activation_40[0][0]              
    __________________________________________________________________________________________________
    conv2d_45 (Conv2D)              (None, None, None, 1 122880      mixed4[0][0]                     
    __________________________________________________________________________________________________
    batch_normalization_45 (BatchNo (None, None, None, 1 480         conv2d_45[0][0]                  
    __________________________________________________________________________________________________
    activation_45 (Activation)      (None, None, None, 1 0           batch_normalization_45[0][0]     
    __________________________________________________________________________________________________
    conv2d_46 (Conv2D)              (None, None, None, 1 179200      activation_45[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_46 (BatchNo (None, None, None, 1 480         conv2d_46[0][0]                  
    __________________________________________________________________________________________________
    activation_46 (Activation)      (None, None, None, 1 0           batch_normalization_46[0][0]     
    __________________________________________________________________________________________________
    conv2d_42 (Conv2D)              (None, None, None, 1 122880      mixed4[0][0]                     
    __________________________________________________________________________________________________
    conv2d_47 (Conv2D)              (None, None, None, 1 179200      activation_46[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_42 (BatchNo (None, None, None, 1 480         conv2d_42[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_47 (BatchNo (None, None, None, 1 480         conv2d_47[0][0]                  
    __________________________________________________________________________________________________
    activation_42 (Activation)      (None, None, None, 1 0           batch_normalization_42[0][0]     
    __________________________________________________________________________________________________
    activation_47 (Activation)      (None, None, None, 1 0           batch_normalization_47[0][0]     
    __________________________________________________________________________________________________
    conv2d_43 (Conv2D)              (None, None, None, 1 179200      activation_42[0][0]              
    __________________________________________________________________________________________________
    conv2d_48 (Conv2D)              (None, None, None, 1 179200      activation_47[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_43 (BatchNo (None, None, None, 1 480         conv2d_43[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_48 (BatchNo (None, None, None, 1 480         conv2d_48[0][0]                  
    __________________________________________________________________________________________________
    activation_43 (Activation)      (None, None, None, 1 0           batch_normalization_43[0][0]     
    __________________________________________________________________________________________________
    activation_48 (Activation)      (None, None, None, 1 0           batch_normalization_48[0][0]     
    __________________________________________________________________________________________________
    average_pooling2d_5 (AveragePoo (None, None, None, 7 0           mixed4[0][0]                     
    __________________________________________________________________________________________________
    conv2d_41 (Conv2D)              (None, None, None, 1 147456      mixed4[0][0]                     
    __________________________________________________________________________________________________
    conv2d_44 (Conv2D)              (None, None, None, 1 215040      activation_43[0][0]              
    __________________________________________________________________________________________________
    conv2d_49 (Conv2D)              (None, None, None, 1 215040      activation_48[0][0]              
    __________________________________________________________________________________________________
    conv2d_50 (Conv2D)              (None, None, None, 1 147456      average_pooling2d_5[0][0]        
    __________________________________________________________________________________________________
    batch_normalization_41 (BatchNo (None, None, None, 1 576         conv2d_41[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_44 (BatchNo (None, None, None, 1 576         conv2d_44[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_49 (BatchNo (None, None, None, 1 576         conv2d_49[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_50 (BatchNo (None, None, None, 1 576         conv2d_50[0][0]                  
    __________________________________________________________________________________________________
    activation_41 (Activation)      (None, None, None, 1 0           batch_normalization_41[0][0]     
    __________________________________________________________________________________________________
    activation_44 (Activation)      (None, None, None, 1 0           batch_normalization_44[0][0]     
    __________________________________________________________________________________________________
    activation_49 (Activation)      (None, None, None, 1 0           batch_normalization_49[0][0]     
    __________________________________________________________________________________________________
    activation_50 (Activation)      (None, None, None, 1 0           batch_normalization_50[0][0]     
    __________________________________________________________________________________________________
    mixed5 (Concatenate)            (None, None, None, 7 0           activation_41[0][0]              
                                                                     activation_44[0][0]              
                                                                     activation_49[0][0]              
                                                                     activation_50[0][0]              
    __________________________________________________________________________________________________
    conv2d_55 (Conv2D)              (None, None, None, 1 122880      mixed5[0][0]                     
    __________________________________________________________________________________________________
    batch_normalization_55 (BatchNo (None, None, None, 1 480         conv2d_55[0][0]                  
    __________________________________________________________________________________________________
    activation_55 (Activation)      (None, None, None, 1 0           batch_normalization_55[0][0]     
    __________________________________________________________________________________________________
    conv2d_56 (Conv2D)              (None, None, None, 1 179200      activation_55[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_56 (BatchNo (None, None, None, 1 480         conv2d_56[0][0]                  
    __________________________________________________________________________________________________
    activation_56 (Activation)      (None, None, None, 1 0           batch_normalization_56[0][0]     
    __________________________________________________________________________________________________
    conv2d_52 (Conv2D)              (None, None, None, 1 122880      mixed5[0][0]                     
    __________________________________________________________________________________________________
    conv2d_57 (Conv2D)              (None, None, None, 1 179200      activation_56[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_52 (BatchNo (None, None, None, 1 480         conv2d_52[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_57 (BatchNo (None, None, None, 1 480         conv2d_57[0][0]                  
    __________________________________________________________________________________________________
    activation_52 (Activation)      (None, None, None, 1 0           batch_normalization_52[0][0]     
    __________________________________________________________________________________________________
    activation_57 (Activation)      (None, None, None, 1 0           batch_normalization_57[0][0]     
    __________________________________________________________________________________________________
    conv2d_53 (Conv2D)              (None, None, None, 1 179200      activation_52[0][0]              
    __________________________________________________________________________________________________
    conv2d_58 (Conv2D)              (None, None, None, 1 179200      activation_57[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_53 (BatchNo (None, None, None, 1 480         conv2d_53[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_58 (BatchNo (None, None, None, 1 480         conv2d_58[0][0]                  
    __________________________________________________________________________________________________
    activation_53 (Activation)      (None, None, None, 1 0           batch_normalization_53[0][0]     
    __________________________________________________________________________________________________
    activation_58 (Activation)      (None, None, None, 1 0           batch_normalization_58[0][0]     
    __________________________________________________________________________________________________
    average_pooling2d_6 (AveragePoo (None, None, None, 7 0           mixed5[0][0]                     
    __________________________________________________________________________________________________
    conv2d_51 (Conv2D)              (None, None, None, 1 147456      mixed5[0][0]                     
    __________________________________________________________________________________________________
    conv2d_54 (Conv2D)              (None, None, None, 1 215040      activation_53[0][0]              
    __________________________________________________________________________________________________
    conv2d_59 (Conv2D)              (None, None, None, 1 215040      activation_58[0][0]              
    __________________________________________________________________________________________________
    conv2d_60 (Conv2D)              (None, None, None, 1 147456      average_pooling2d_6[0][0]        
    __________________________________________________________________________________________________
    batch_normalization_51 (BatchNo (None, None, None, 1 576         conv2d_51[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_54 (BatchNo (None, None, None, 1 576         conv2d_54[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_59 (BatchNo (None, None, None, 1 576         conv2d_59[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_60 (BatchNo (None, None, None, 1 576         conv2d_60[0][0]                  
    __________________________________________________________________________________________________
    activation_51 (Activation)      (None, None, None, 1 0           batch_normalization_51[0][0]     
    __________________________________________________________________________________________________
    activation_54 (Activation)      (None, None, None, 1 0           batch_normalization_54[0][0]     
    __________________________________________________________________________________________________
    activation_59 (Activation)      (None, None, None, 1 0           batch_normalization_59[0][0]     
    __________________________________________________________________________________________________
    activation_60 (Activation)      (None, None, None, 1 0           batch_normalization_60[0][0]     
    __________________________________________________________________________________________________
    mixed6 (Concatenate)            (None, None, None, 7 0           activation_51[0][0]              
                                                                     activation_54[0][0]              
                                                                     activation_59[0][0]              
                                                                     activation_60[0][0]              
    __________________________________________________________________________________________________
    conv2d_65 (Conv2D)              (None, None, None, 1 147456      mixed6[0][0]                     
    __________________________________________________________________________________________________
    batch_normalization_65 (BatchNo (None, None, None, 1 576         conv2d_65[0][0]                  
    __________________________________________________________________________________________________
    activation_65 (Activation)      (None, None, None, 1 0           batch_normalization_65[0][0]     
    __________________________________________________________________________________________________
    conv2d_66 (Conv2D)              (None, None, None, 1 258048      activation_65[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_66 (BatchNo (None, None, None, 1 576         conv2d_66[0][0]                  
    __________________________________________________________________________________________________
    activation_66 (Activation)      (None, None, None, 1 0           batch_normalization_66[0][0]     
    __________________________________________________________________________________________________
    conv2d_62 (Conv2D)              (None, None, None, 1 147456      mixed6[0][0]                     
    __________________________________________________________________________________________________
    conv2d_67 (Conv2D)              (None, None, None, 1 258048      activation_66[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_62 (BatchNo (None, None, None, 1 576         conv2d_62[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_67 (BatchNo (None, None, None, 1 576         conv2d_67[0][0]                  
    __________________________________________________________________________________________________
    activation_62 (Activation)      (None, None, None, 1 0           batch_normalization_62[0][0]     
    __________________________________________________________________________________________________
    activation_67 (Activation)      (None, None, None, 1 0           batch_normalization_67[0][0]     
    __________________________________________________________________________________________________
    conv2d_63 (Conv2D)              (None, None, None, 1 258048      activation_62[0][0]              
    __________________________________________________________________________________________________
    conv2d_68 (Conv2D)              (None, None, None, 1 258048      activation_67[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_63 (BatchNo (None, None, None, 1 576         conv2d_63[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_68 (BatchNo (None, None, None, 1 576         conv2d_68[0][0]                  
    __________________________________________________________________________________________________
    activation_63 (Activation)      (None, None, None, 1 0           batch_normalization_63[0][0]     
    __________________________________________________________________________________________________
    activation_68 (Activation)      (None, None, None, 1 0           batch_normalization_68[0][0]     
    __________________________________________________________________________________________________
    average_pooling2d_7 (AveragePoo (None, None, None, 7 0           mixed6[0][0]                     
    __________________________________________________________________________________________________
    conv2d_61 (Conv2D)              (None, None, None, 1 147456      mixed6[0][0]                     
    __________________________________________________________________________________________________
    conv2d_64 (Conv2D)              (None, None, None, 1 258048      activation_63[0][0]              
    __________________________________________________________________________________________________
    conv2d_69 (Conv2D)              (None, None, None, 1 258048      activation_68[0][0]              
    __________________________________________________________________________________________________
    conv2d_70 (Conv2D)              (None, None, None, 1 147456      average_pooling2d_7[0][0]        
    __________________________________________________________________________________________________
    batch_normalization_61 (BatchNo (None, None, None, 1 576         conv2d_61[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_64 (BatchNo (None, None, None, 1 576         conv2d_64[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_69 (BatchNo (None, None, None, 1 576         conv2d_69[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_70 (BatchNo (None, None, None, 1 576         conv2d_70[0][0]                  
    __________________________________________________________________________________________________
    activation_61 (Activation)      (None, None, None, 1 0           batch_normalization_61[0][0]     
    __________________________________________________________________________________________________
    activation_64 (Activation)      (None, None, None, 1 0           batch_normalization_64[0][0]     
    __________________________________________________________________________________________________
    activation_69 (Activation)      (None, None, None, 1 0           batch_normalization_69[0][0]     
    __________________________________________________________________________________________________
    activation_70 (Activation)      (None, None, None, 1 0           batch_normalization_70[0][0]     
    __________________________________________________________________________________________________
    mixed7 (Concatenate)            (None, None, None, 7 0           activation_61[0][0]              
                                                                     activation_64[0][0]              
                                                                     activation_69[0][0]              
                                                                     activation_70[0][0]              
    __________________________________________________________________________________________________
    conv2d_73 (Conv2D)              (None, None, None, 1 147456      mixed7[0][0]                     
    __________________________________________________________________________________________________
    batch_normalization_73 (BatchNo (None, None, None, 1 576         conv2d_73[0][0]                  
    __________________________________________________________________________________________________
    activation_73 (Activation)      (None, None, None, 1 0           batch_normalization_73[0][0]     
    __________________________________________________________________________________________________
    conv2d_74 (Conv2D)              (None, None, None, 1 258048      activation_73[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_74 (BatchNo (None, None, None, 1 576         conv2d_74[0][0]                  
    __________________________________________________________________________________________________
    activation_74 (Activation)      (None, None, None, 1 0           batch_normalization_74[0][0]     
    __________________________________________________________________________________________________
    conv2d_71 (Conv2D)              (None, None, None, 1 147456      mixed7[0][0]                     
    __________________________________________________________________________________________________
    conv2d_75 (Conv2D)              (None, None, None, 1 258048      activation_74[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_71 (BatchNo (None, None, None, 1 576         conv2d_71[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_75 (BatchNo (None, None, None, 1 576         conv2d_75[0][0]                  
    __________________________________________________________________________________________________
    activation_71 (Activation)      (None, None, None, 1 0           batch_normalization_71[0][0]     
    __________________________________________________________________________________________________
    activation_75 (Activation)      (None, None, None, 1 0           batch_normalization_75[0][0]     
    __________________________________________________________________________________________________
    conv2d_72 (Conv2D)              (None, None, None, 3 552960      activation_71[0][0]              
    __________________________________________________________________________________________________
    conv2d_76 (Conv2D)              (None, None, None, 1 331776      activation_75[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_72 (BatchNo (None, None, None, 3 960         conv2d_72[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_76 (BatchNo (None, None, None, 1 576         conv2d_76[0][0]                  
    __________________________________________________________________________________________________
    activation_72 (Activation)      (None, None, None, 3 0           batch_normalization_72[0][0]     
    __________________________________________________________________________________________________
    activation_76 (Activation)      (None, None, None, 1 0           batch_normalization_76[0][0]     
    __________________________________________________________________________________________________
    max_pooling2d_4 (MaxPooling2D)  (None, None, None, 7 0           mixed7[0][0]                     
    __________________________________________________________________________________________________
    mixed8 (Concatenate)            (None, None, None, 1 0           activation_72[0][0]              
                                                                     activation_76[0][0]              
                                                                     max_pooling2d_4[0][0]            
    __________________________________________________________________________________________________
    conv2d_81 (Conv2D)              (None, None, None, 4 573440      mixed8[0][0]                     
    __________________________________________________________________________________________________
    batch_normalization_81 (BatchNo (None, None, None, 4 1344        conv2d_81[0][0]                  
    __________________________________________________________________________________________________
    activation_81 (Activation)      (None, None, None, 4 0           batch_normalization_81[0][0]     
    __________________________________________________________________________________________________
    conv2d_78 (Conv2D)              (None, None, None, 3 491520      mixed8[0][0]                     
    __________________________________________________________________________________________________
    conv2d_82 (Conv2D)              (None, None, None, 3 1548288     activation_81[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_78 (BatchNo (None, None, None, 3 1152        conv2d_78[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_82 (BatchNo (None, None, None, 3 1152        conv2d_82[0][0]                  
    __________________________________________________________________________________________________
    activation_78 (Activation)      (None, None, None, 3 0           batch_normalization_78[0][0]     
    __________________________________________________________________________________________________
    activation_82 (Activation)      (None, None, None, 3 0           batch_normalization_82[0][0]     
    __________________________________________________________________________________________________
    conv2d_79 (Conv2D)              (None, None, None, 3 442368      activation_78[0][0]              
    __________________________________________________________________________________________________
    conv2d_80 (Conv2D)              (None, None, None, 3 442368      activation_78[0][0]              
    __________________________________________________________________________________________________
    conv2d_83 (Conv2D)              (None, None, None, 3 442368      activation_82[0][0]              
    __________________________________________________________________________________________________
    conv2d_84 (Conv2D)              (None, None, None, 3 442368      activation_82[0][0]              
    __________________________________________________________________________________________________
    average_pooling2d_8 (AveragePoo (None, None, None, 1 0           mixed8[0][0]                     
    __________________________________________________________________________________________________
    conv2d_77 (Conv2D)              (None, None, None, 3 409600      mixed8[0][0]                     
    __________________________________________________________________________________________________
    batch_normalization_79 (BatchNo (None, None, None, 3 1152        conv2d_79[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_80 (BatchNo (None, None, None, 3 1152        conv2d_80[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_83 (BatchNo (None, None, None, 3 1152        conv2d_83[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_84 (BatchNo (None, None, None, 3 1152        conv2d_84[0][0]                  
    __________________________________________________________________________________________________
    conv2d_85 (Conv2D)              (None, None, None, 1 245760      average_pooling2d_8[0][0]        
    __________________________________________________________________________________________________
    batch_normalization_77 (BatchNo (None, None, None, 3 960         conv2d_77[0][0]                  
    __________________________________________________________________________________________________
    activation_79 (Activation)      (None, None, None, 3 0           batch_normalization_79[0][0]     
    __________________________________________________________________________________________________
    activation_80 (Activation)      (None, None, None, 3 0           batch_normalization_80[0][0]     
    __________________________________________________________________________________________________
    activation_83 (Activation)      (None, None, None, 3 0           batch_normalization_83[0][0]     
    __________________________________________________________________________________________________
    activation_84 (Activation)      (None, None, None, 3 0           batch_normalization_84[0][0]     
    __________________________________________________________________________________________________
    batch_normalization_85 (BatchNo (None, None, None, 1 576         conv2d_85[0][0]                  
    __________________________________________________________________________________________________
    activation_77 (Activation)      (None, None, None, 3 0           batch_normalization_77[0][0]     
    __________________________________________________________________________________________________
    mixed9_0 (Concatenate)          (None, None, None, 7 0           activation_79[0][0]              
                                                                     activation_80[0][0]              
    __________________________________________________________________________________________________
    concatenate_1 (Concatenate)     (None, None, None, 7 0           activation_83[0][0]              
                                                                     activation_84[0][0]              
    __________________________________________________________________________________________________
    activation_85 (Activation)      (None, None, None, 1 0           batch_normalization_85[0][0]     
    __________________________________________________________________________________________________
    mixed9 (Concatenate)            (None, None, None, 2 0           activation_77[0][0]              
                                                                     mixed9_0[0][0]                   
                                                                     concatenate_1[0][0]              
                                                                     activation_85[0][0]              
    __________________________________________________________________________________________________
    conv2d_90 (Conv2D)              (None, None, None, 4 917504      mixed9[0][0]                     
    __________________________________________________________________________________________________
    batch_normalization_90 (BatchNo (None, None, None, 4 1344        conv2d_90[0][0]                  
    __________________________________________________________________________________________________
    activation_90 (Activation)      (None, None, None, 4 0           batch_normalization_90[0][0]     
    __________________________________________________________________________________________________
    conv2d_87 (Conv2D)              (None, None, None, 3 786432      mixed9[0][0]                     
    __________________________________________________________________________________________________
    conv2d_91 (Conv2D)              (None, None, None, 3 1548288     activation_90[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_87 (BatchNo (None, None, None, 3 1152        conv2d_87[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_91 (BatchNo (None, None, None, 3 1152        conv2d_91[0][0]                  
    __________________________________________________________________________________________________
    activation_87 (Activation)      (None, None, None, 3 0           batch_normalization_87[0][0]     
    __________________________________________________________________________________________________
    activation_91 (Activation)      (None, None, None, 3 0           batch_normalization_91[0][0]     
    __________________________________________________________________________________________________
    conv2d_88 (Conv2D)              (None, None, None, 3 442368      activation_87[0][0]              
    __________________________________________________________________________________________________
    conv2d_89 (Conv2D)              (None, None, None, 3 442368      activation_87[0][0]              
    __________________________________________________________________________________________________
    conv2d_92 (Conv2D)              (None, None, None, 3 442368      activation_91[0][0]              
    __________________________________________________________________________________________________
    conv2d_93 (Conv2D)              (None, None, None, 3 442368      activation_91[0][0]              
    __________________________________________________________________________________________________
    average_pooling2d_9 (AveragePoo (None, None, None, 2 0           mixed9[0][0]                     
    __________________________________________________________________________________________________
    conv2d_86 (Conv2D)              (None, None, None, 3 655360      mixed9[0][0]                     
    __________________________________________________________________________________________________
    batch_normalization_88 (BatchNo (None, None, None, 3 1152        conv2d_88[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_89 (BatchNo (None, None, None, 3 1152        conv2d_89[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_92 (BatchNo (None, None, None, 3 1152        conv2d_92[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_93 (BatchNo (None, None, None, 3 1152        conv2d_93[0][0]                  
    __________________________________________________________________________________________________
    conv2d_94 (Conv2D)              (None, None, None, 1 393216      average_pooling2d_9[0][0]        
    __________________________________________________________________________________________________
    batch_normalization_86 (BatchNo (None, None, None, 3 960         conv2d_86[0][0]                  
    __________________________________________________________________________________________________
    activation_88 (Activation)      (None, None, None, 3 0           batch_normalization_88[0][0]     
    __________________________________________________________________________________________________
    activation_89 (Activation)      (None, None, None, 3 0           batch_normalization_89[0][0]     
    __________________________________________________________________________________________________
    activation_92 (Activation)      (None, None, None, 3 0           batch_normalization_92[0][0]     
    __________________________________________________________________________________________________
    activation_93 (Activation)      (None, None, None, 3 0           batch_normalization_93[0][0]     
    __________________________________________________________________________________________________
    batch_normalization_94 (BatchNo (None, None, None, 1 576         conv2d_94[0][0]                  
    __________________________________________________________________________________________________
    activation_86 (Activation)      (None, None, None, 3 0           batch_normalization_86[0][0]     
    __________________________________________________________________________________________________
    mixed9_1 (Concatenate)          (None, None, None, 7 0           activation_88[0][0]              
                                                                     activation_89[0][0]              
    __________________________________________________________________________________________________
    concatenate_2 (Concatenate)     (None, None, None, 7 0           activation_92[0][0]              
                                                                     activation_93[0][0]              
    __________________________________________________________________________________________________
    activation_94 (Activation)      (None, None, None, 1 0           batch_normalization_94[0][0]     
    __________________________________________________________________________________________________
    mixed10 (Concatenate)           (None, None, None, 2 0           activation_86[0][0]              
                                                                     mixed9_1[0][0]                   
                                                                     concatenate_2[0][0]              
                                                                     activation_94[0][0]              
    __________________________________________________________________________________________________
    global_average_pooling2d_1 (Glo (None, 2048)         0           mixed10[0][0]                    
    __________________________________________________________________________________________________
    dense_1 (Dense)                 (None, 128)          262272      global_average_pooling2d_1[0][0] 
    __________________________________________________________________________________________________
    dropout_1 (Dropout)             (None, 128)          0           dense_1[0][0]                    
    __________________________________________________________________________________________________
    dense_2 (Dense)                 (None, 3)            387         dropout_1[0][0]                  
    ==================================================================================================
    Total params: 22,065,443
    Trainable params: 22,031,011
    Non-trainable params: 34,432
    __________________________________________________________________________________________________
    None


## Train our model


```python
history = model.fit_generator(train_generator,
                             steps_per_epoch=nb_train_samples // batch_size,
                             validation_data = validation_generator,
                             validation_steps = nb_validation_samples//batch_size,
                             epochs = 30,
                             verbose=1,
                             callbacks=[csv_logger, checkpoint])
model.save('model_trained_3class.hdf5')
```

    Epoch 1/30
    140/140 [==============================] - 115s 818ms/step - loss: 1.0822 - acc: 0.4680 - val_loss: 0.9090 - val_acc: 0.7038
    
    Epoch 00001: val_loss improved from inf to 0.90902, saving model to best_model_3class.hdf5
    Epoch 2/30
    140/140 [==============================] - 94s 670ms/step - loss: 0.8565 - acc: 0.6708 - val_loss: 0.7083 - val_acc: 0.8139
    
    Epoch 00002: val_loss improved from 0.90902 to 0.70829, saving model to best_model_3class.hdf5
    Epoch 3/30
    140/140 [==============================] - 93s 664ms/step - loss: 0.6816 - acc: 0.7770 - val_loss: 0.5451 - val_acc: 0.8614
    
    Epoch 00003: val_loss improved from 0.70829 to 0.54505, saving model to best_model_3class.hdf5
    Epoch 4/30
    140/140 [==============================] - 90s 641ms/step - loss: 0.5882 - acc: 0.7953 - val_loss: 0.4513 - val_acc: 0.8791
    
    Epoch 00004: val_loss improved from 0.54505 to 0.45127, saving model to best_model_3class.hdf5
    Epoch 5/30
    140/140 [==============================] - 91s 651ms/step - loss: 0.5087 - acc: 0.8285 - val_loss: 0.3785 - val_acc: 0.8899
    
    Epoch 00005: val_loss improved from 0.45127 to 0.37850, saving model to best_model_3class.hdf5
    Epoch 6/30
    140/140 [==============================] - 88s 631ms/step - loss: 0.4507 - acc: 0.8499 - val_loss: 0.3320 - val_acc: 0.8995
    
    Epoch 00006: val_loss improved from 0.37850 to 0.33200, saving model to best_model_3class.hdf5
    Epoch 7/30
    140/140 [==============================] - 90s 646ms/step - loss: 0.3940 - acc: 0.8693 - val_loss: 0.2996 - val_acc: 0.9076
    
    Epoch 00007: val_loss improved from 0.33200 to 0.29958, saving model to best_model_3class.hdf5
    Epoch 8/30
    140/140 [==============================] - 88s 627ms/step - loss: 0.3785 - acc: 0.8763 - val_loss: 0.2771 - val_acc: 0.9239
    
    Epoch 00008: val_loss improved from 0.29958 to 0.27711, saving model to best_model_3class.hdf5
    Epoch 9/30
    140/140 [==============================] - 87s 619ms/step - loss: 0.3415 - acc: 0.8917 - val_loss: 0.2593 - val_acc: 0.9321
    
    Epoch 00009: val_loss improved from 0.27711 to 0.25928, saving model to best_model_3class.hdf5
    Epoch 10/30
    140/140 [==============================] - 87s 620ms/step - loss: 0.3157 - acc: 0.8999 - val_loss: 0.2439 - val_acc: 0.9307
    
    Epoch 00010: val_loss improved from 0.25928 to 0.24386, saving model to best_model_3class.hdf5
    Epoch 11/30
    140/140 [==============================] - 88s 625ms/step - loss: 0.2881 - acc: 0.9149 - val_loss: 0.2330 - val_acc: 0.9293
    
    Epoch 00011: val_loss improved from 0.24386 to 0.23303, saving model to best_model_3class.hdf5
    Epoch 12/30
    140/140 [==============================] - 86s 616ms/step - loss: 0.2893 - acc: 0.9067 - val_loss: 0.2260 - val_acc: 0.9280
    
    Epoch 00012: val_loss improved from 0.23303 to 0.22601, saving model to best_model_3class.hdf5
    Epoch 13/30
    140/140 [==============================] - 88s 625ms/step - loss: 0.2531 - acc: 0.9243 - val_loss: 0.2166 - val_acc: 0.9348
    
    Epoch 00013: val_loss improved from 0.22601 to 0.21660, saving model to best_model_3class.hdf5
    Epoch 14/30
    140/140 [==============================] - 85s 604ms/step - loss: 0.2547 - acc: 0.9160 - val_loss: 0.2090 - val_acc: 0.9321
    
    Epoch 00014: val_loss improved from 0.21660 to 0.20904, saving model to best_model_3class.hdf5
    Epoch 15/30
    140/140 [==============================] - 87s 621ms/step - loss: 0.2353 - acc: 0.9274 - val_loss: 0.2075 - val_acc: 0.9293
    
    Epoch 00015: val_loss improved from 0.20904 to 0.20749, saving model to best_model_3class.hdf5
    Epoch 16/30
    140/140 [==============================] - 84s 601ms/step - loss: 0.2250 - acc: 0.9337 - val_loss: 0.2020 - val_acc: 0.9348
    
    Epoch 00016: val_loss improved from 0.20749 to 0.20204, saving model to best_model_3class.hdf5
    Epoch 17/30
    140/140 [==============================] - 85s 609ms/step - loss: 0.2118 - acc: 0.9437 - val_loss: 0.1981 - val_acc: 0.9361
    
    Epoch 00017: val_loss improved from 0.20204 to 0.19810, saving model to best_model_3class.hdf5
    Epoch 18/30
    140/140 [==============================] - 85s 605ms/step - loss: 0.1992 - acc: 0.9451 - val_loss: 0.1962 - val_acc: 0.9389
    
    Epoch 00018: val_loss improved from 0.19810 to 0.19624, saving model to best_model_3class.hdf5
    Epoch 19/30
    140/140 [==============================] - 86s 611ms/step - loss: 0.2120 - acc: 0.9338 - val_loss: 0.1991 - val_acc: 0.9334
    
    Epoch 00019: val_loss did not improve from 0.19624
    Epoch 20/30
    140/140 [==============================] - 83s 595ms/step - loss: 0.1828 - acc: 0.9465 - val_loss: 0.1957 - val_acc: 0.9361
    
    Epoch 00020: val_loss improved from 0.19624 to 0.19568, saving model to best_model_3class.hdf5
    Epoch 21/30
    140/140 [==============================] - 84s 599ms/step - loss: 0.1662 - acc: 0.9502 - val_loss: 0.1964 - val_acc: 0.9375
    
    Epoch 00021: val_loss did not improve from 0.19568
    Epoch 22/30
    140/140 [==============================] - 83s 596ms/step - loss: 0.1822 - acc: 0.9479 - val_loss: 0.1941 - val_acc: 0.9389
    
    Epoch 00022: val_loss improved from 0.19568 to 0.19412, saving model to best_model_3class.hdf5
    Epoch 23/30
    140/140 [==============================] - 84s 601ms/step - loss: 0.1633 - acc: 0.9527 - val_loss: 0.1899 - val_acc: 0.9361
    
    Epoch 00023: val_loss improved from 0.19412 to 0.18994, saving model to best_model_3class.hdf5
    Epoch 24/30
    140/140 [==============================] - 84s 599ms/step - loss: 0.1521 - acc: 0.9585 - val_loss: 0.1860 - val_acc: 0.9402
    
    Epoch 00024: val_loss improved from 0.18994 to 0.18599, saving model to best_model_3class.hdf5
    Epoch 25/30
    140/140 [==============================] - 82s 586ms/step - loss: 0.1394 - acc: 0.9663 - val_loss: 0.1882 - val_acc: 0.9389
    
    Epoch 00025: val_loss did not improve from 0.18599
    Epoch 26/30
    140/140 [==============================] - 84s 597ms/step - loss: 0.1450 - acc: 0.9616 - val_loss: 0.1842 - val_acc: 0.9402
    
    Epoch 00026: val_loss improved from 0.18599 to 0.18420, saving model to best_model_3class.hdf5
    Epoch 27/30
    140/140 [==============================] - 84s 602ms/step - loss: 0.1368 - acc: 0.9646 - val_loss: 0.1866 - val_acc: 0.9429
    
    Epoch 00027: val_loss did not improve from 0.18420
    Epoch 28/30
    140/140 [==============================] - 83s 591ms/step - loss: 0.1188 - acc: 0.9750 - val_loss: 0.1888 - val_acc: 0.9389
    
    Epoch 00028: val_loss did not improve from 0.18420
    Epoch 29/30
    140/140 [==============================] - 83s 590ms/step - loss: 0.1264 - acc: 0.9682 - val_loss: 0.1901 - val_acc: 0.9361
    
    Epoch 00029: val_loss did not improve from 0.18420
    Epoch 30/30
    140/140 [==============================] - 82s 587ms/step - loss: 0.1181 - acc: 0.9732 - val_loss: 0.1891 - val_acc: 0.9402
    
    Epoch 00030: val_loss did not improve from 0.18420



```python
class_map_3 = train_generator.class_indices
class_map_3
```




    {'apple_pie': 0, 'omelette': 1, 'pizza': 2}



### Visualize the accuracy and loss plots


```python
def plot_accuracy(history, title):
    plt.title(title)
    line1 = plt.plot(history.history['acc'])
    line2 = plt.plot(history.history['val_acc'])
    #plt.setp(line1, linewidth=2.0, marker = '+', markersize=10.0)
    #plt.setp(line2, linewidth=2.0, marker = '4', markersize=10.0)
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train_accuracy', 'validation_accuracy'], loc='best')
    plt.show()
    
def plot_loss(history, title):
    plt.title(title)
    line1 = plt.plot(history.history['loss'])
    line2 = plt.plot(history.history['val_loss'])
    #plt.setp(line1, linewidth=2.0, marker = '+', markersize=10.0)
    #plt.setp(line2, linewidth=2.0, marker = '4', markersize=10.0)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train_loss', 'validation_loss'], loc='best')
    plt.show()
```


```python
plot_accuracy(history, 'FOOD101-Inceptionv3-3classes')
plot_loss(history, 'FOOD101-Inceptionv3-3classes')
```


![png](2.%20Multiclass%20Classification%20using%20Keras%20and%20TensorFlow%20on%20Food-101%20Dataset_files/2.%20Multiclass%20Classification%20using%20Keras%20and%20TensorFlow%20on%20Food-101%20Dataset_44_0.png)



![png](2.%20Multiclass%20Classification%20using%20Keras%20and%20TensorFlow%20on%20Food-101%20Dataset_files/2.%20Multiclass%20Classification%20using%20Keras%20and%20TensorFlow%20on%20Food-101%20Dataset_44_1.png)


   * **The plots show that the accuracy of the model increased with epochs and the loss has decreased**
   * **Validation accuracy has been on the higher side than training accuracy for many epochs**
   * This could be for several reasons:
      * We used a pretrained model trained on ImageNet which contains data from a variety of classes
      *  Using dropout can lead to a higher validation accuracy



## Predicting classes for new images from internet using the best trained model


```python
%%time
# Loading the best saved model to make predictions
K.clear_session()
model_best = load_model('best_model_3class.hdf5', compile=False)
```

    CPU times: user 13.5 s, sys: 85.1 ms, total: 13.5 s
    Wall time: 13.5 s


* Setting compile=False and clearing the session leads to faster loading of the saved model

* Without the above addiitons, model loading was taking more than a minute!


```python
# Helper function to predict class on a new image
def predict_class(model, images, show=True):
    for img in images:
        img = image.load_img(img, target_size=(299,299))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img /= 255
        
        pred = model.predict(img)
        index = np.argmax(pred)
        food_list.sort()
        pred_value = food_list[index]
        if show:
            plt.imshow(img[0])
            plt.axis('off')
            plt.title(pred_value)
            plt.show()
```


```python
# Downloading images from internet using the URLs
!wget -O apple_pie.jpg https://acleanbake.com/wp-content/uploads/2017/10/Paleo-Apple-Pie-with-Crumb-Topping-gluten-free-grain-free-dairy-free-15.jpg

!wget -O samosa.jpg http://veggiefoodrecipes.com/wp-content/uploads/2016/05/lentil-samosa-recipe-01.jpg

!wget -O pizza.jpg http://104.130.3.186/assets/itemimages/400/400/3/default_9b4106b8f65359684b3836096b4524c8_pizza%20dreamstimesmall_94940296.jpg

!wget -O omelette.jpg https://www.incredibleegg.org/wp-content/uploads/basic-french-omelet-930x550.jpg
    
```

    --2020-04-08 06:51:48--  https://acleanbake.com/wp-content/uploads/2017/10/Paleo-Apple-Pie-with-Crumb-Topping-gluten-free-grain-free-dairy-free-15.jpg
    Resolving acleanbake.com (acleanbake.com)... 209.151.228.206
    Connecting to acleanbake.com (acleanbake.com)|209.151.228.206|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 114276 (112K) [image/jpeg]
    Saving to: 'apple_pie.jpg'
    
    apple_pie.jpg       100%[===================>] 111.60K  --.-KB/s    in 0.1s    
    
    2020-04-08 06:51:49 (908 KB/s) - 'apple_pie.jpg' saved [114276/114276]
    
    --2020-04-08 06:51:49--  http://veggiefoodrecipes.com/wp-content/uploads/2016/05/lentil-samosa-recipe-01.jpg
    Resolving veggiefoodrecipes.com (veggiefoodrecipes.com)... 62.75.168.50
    Connecting to veggiefoodrecipes.com (veggiefoodrecipes.com)|62.75.168.50|:80... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 75560 (74K) [image/jpeg]
    Saving to: 'samosa.jpg'
    
    samosa.jpg          100%[===================>]  73.79K   312KB/s    in 0.2s    
    
    2020-04-08 06:51:50 (312 KB/s) - 'samosa.jpg' saved [75560/75560]
    
    --2020-04-08 06:51:50--  http://104.130.3.186/assets/itemimages/400/400/3/default_9b4106b8f65359684b3836096b4524c8_pizza%20dreamstimesmall_94940296.jpg
    Connecting to 104.130.3.186:80... ^C
    --2020-04-08 06:52:07--  https://www.incredibleegg.org/wp-content/uploads/basic-french-omelet-930x550.jpg
    Resolving www.incredibleegg.org (www.incredibleegg.org)... 35.224.134.160
    Connecting to www.incredibleegg.org (www.incredibleegg.org)|35.224.134.160|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 109569 (107K) [image/jpeg]
    Saving to: 'omelette.jpg'
    
    omelette.jpg        100%[===================>] 107.00K  --.-KB/s    in 0.07s   
    
    2020-04-08 06:52:07 (1.43 MB/s) - 'omelette.jpg' saved [109569/109569]
    



```python
# Make a list of downloaded images and test the trained model
images = []
images.append('apple_pie.jpg')
images.append('pizza.jpg')
images.append('omelette.jpg')

predict_class(model_best, images, True)
```


![png](2.%20Multiclass%20Classification%20using%20Keras%20and%20TensorFlow%20on%20Food-101%20Dataset_files/2.%20Multiclass%20Classification%20using%20Keras%20and%20TensorFlow%20on%20Food-101%20Dataset_51_0.png)



![png](2.%20Multiclass%20Classification%20using%20Keras%20and%20TensorFlow%20on%20Food-101%20Dataset_files/2.%20Multiclass%20Classification%20using%20Keras%20and%20TensorFlow%20on%20Food-101%20Dataset_51_1.png)



![png](2.%20Multiclass%20Classification%20using%20Keras%20and%20TensorFlow%20on%20Food-101%20Dataset_files/2.%20Multiclass%20Classification%20using%20Keras%20and%20TensorFlow%20on%20Food-101%20Dataset_51_2.png)


**All three pictures are predicted correctly!**

## Fine tune Inceptionv3 model with 11 classes of data

   * We trained a model on 3 classes and tested it using new data
   * The model was able to predict the classes of all three test images correctly
   * Will it be able to perform at the same level of accuracy for more classes?
   * FOOD-101 dataset has 101 classes of data
   * Even with fine tuning using a pre-trained model, each epoch was taking more than an hour when all 101 classes of data is used(tried this on both Colab and on a Deep Learning VM instance with P100 GPU on GCP)
   * But to check how the model performs when more classes are included, I'm using the same model to fine tune and train on 11 randomly chosen classes




```python
# Helper function to select n random food classes
def pick_n_random_classes(n):
    food_list = []
    random_food_indices = random.sample(range(len(food_sorted)), n) # pick n random food classes
    for i in random_food_indices:
        food_list.append(food_sorted[i])
    food_list.sort()
    return food_list
```


```python
# randomly pick the food classes
# n = 11
# food_list = pick_n_random_classes(n)

# To facilitate the testing procedure later, we manually select the classes
# Ideally, they should be selected randomly
food_list =  ['apple_pie', 'beef_carpaccio', 'bibimbap', 'cup_cakes', 'foie_gras', 
              'french_fries', 'garlic_bread', 'pizza', 'spring_rolls', 
              'spaghetti_carbonara', 'strawberry_shortcake']
print "These are the randomly picked food classes we will be training the model on:\n", 
food_list
```

    These are the randomly picked food classes we will be training the model on:





    ['apple_pie',
     'beef_carpaccio',
     'bibimbap',
     'cup_cakes',
     'foie_gras',
     'french_fries',
     'garlic_bread',
     'pizza',
     'spring_rolls',
     'spaghetti_carbonara',
     'strawberry_shortcake']




```python
# Create the new data subset of n classes
print("Creating training data folder with new classes...")
dataset_mini(food_list, src_train, dest_train)
```

    Creating training data folder with new classes...



    

    NameErrorTraceback (most recent call last)

    <ipython-input-7-d0cf87b09319> in <module>()
          1 # Create the new data subset of n classes
          2 print("Creating training data folder with new classes...")
    ----> 3 dataset_mini(food_list, src_train, dest_train)
    

    NameError: name 'dataset_mini' is not defined



```python
print("Total number of samples in train folder")
!find train_mini -type d -or -type f -printf '.' | wc -c
```

    Total number of samples in train folder
    8250



```python
print("Creating test data folder with new classes")
dataset_mini(food_list, src_test, dest_test)
```

    Creating test data folder with new classes
    Copying images into  apple_pie
    Copying images into  beef_carpaccio
    Copying images into  bibimbap
    Copying images into  cup_cakes
    Copying images into  foie_gras
    Copying images into  french_fries
    Copying images into  garlic_bread
    Copying images into  pizza
    Copying images into  spring_rolls
    Copying images into  spaghetti_carbonara
    Copying images into  strawberry_shortcake



```python
print("Total number of samples in test folder")
!find test_mini -type d -or -type f -printf '.' | wc -c
```

    Total number of samples in test folder
    2750



```python
# Let's use a pretrained Inceptionv3 model on subset of data with 11 food classes
K.clear_session()
n_classes = n
img_width, img_height = 299, 299 
train_data_dir = 'train_mini'
validation_data_dir = 'test_mini'
nb_train_samples = 8250 # 75,750
nb_validation_samples = 2750 # 25,250
batch_size = 16

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size = batch_size,
    class_mode = 'categorical')
validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size = (img_height, img_width),
    batch_size = batch_size,
    class_mode = 'categorical')

inception = InceptionV3(weights='imagenet', include_top=False)
x = inception.output
# Add FC layer
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.2)(x)

predictions = Dense(n, kernel_regularizer= regularizers.l2(0.005), activation='softmax')(x)

model = Model(input = inception.input, output = predictions)
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', 
              metrics=['accuracy'])
checkpoint = ModelCheckpoint(filepath='best_model_11class.hdf5', verbose=1, save_best_only=True)
csv_logger = CSVLogger('history_11class.log')

history_11class = model.fit_generator(train_generator,
                             steps_per_epoch=nb_train_samples // batch_size,
                             validation_data = validation_generator,
                             validation_steps = nb_validation_samples//batch_size,
                             epochs = 30,
                             verbose=1,
                             callbacks=[csv_logger, checkpoint])
model.save('model_trained_11class.hdf5')
```

    Found 8250 images belonging to 11 classes.
    Found 2750 images belonging to 11 classes.


    /usr/local/lib/python2.7/dist-packages/ipykernel_launcher.py:38: UserWarning: Update your `Model` call to the Keras 2 API: `Model(outputs=Tensor("de..., inputs=Tensor("in...)`


    Epoch 1/30
    515/515 [==============================] - 301s 585ms/step - loss: 2.0991 - acc: 0.3669 - val_loss: 1.3876 - val_acc: 0.7211
    
    Epoch 00001: val_loss improved from inf to 1.38763, saving model to best_model_11class.hdf5
    Epoch 2/30
    515/515 [==============================] - 269s 523ms/step - loss: 1.2997 - acc: 0.6703 - val_loss: 0.7898 - val_acc: 0.8377
    
    Epoch 00002: val_loss improved from 1.38763 to 0.78985, saving model to best_model_11class.hdf5
    Epoch 3/30
    515/515 [==============================] - 278s 539ms/step - loss: 0.9424 - acc: 0.7579 - val_loss: 0.5723 - val_acc: 0.8765
    
    Epoch 00003: val_loss improved from 0.78985 to 0.57231, saving model to best_model_11class.hdf5
    Epoch 4/30
    515/515 [==============================] - 267s 517ms/step - loss: 0.7751 - acc: 0.7977 - val_loss: 0.4710 - val_acc: 0.8999
    
    Epoch 00004: val_loss improved from 0.57231 to 0.47099, saving model to best_model_11class.hdf5
    Epoch 5/30
    515/515 [==============================] - 270s 525ms/step - loss: 0.6828 - acc: 0.8244 - val_loss: 0.4200 - val_acc: 0.9137
    
    Epoch 00005: val_loss improved from 0.47099 to 0.42000, saving model to best_model_11class.hdf5
    Epoch 6/30
    515/515 [==============================] - 275s 535ms/step - loss: 0.6203 - acc: 0.8413 - val_loss: 0.3863 - val_acc: 0.9240
    
    Epoch 00006: val_loss improved from 0.42000 to 0.38628, saving model to best_model_11class.hdf5
    Epoch 7/30
    515/515 [==============================] - 268s 520ms/step - loss: 0.5862 - acc: 0.8567 - val_loss: 0.3682 - val_acc: 0.9254
    
    Epoch 00007: val_loss improved from 0.38628 to 0.36815, saving model to best_model_11class.hdf5
    Epoch 8/30
    515/515 [==============================] - 267s 518ms/step - loss: 0.5307 - acc: 0.8764 - val_loss: 0.3499 - val_acc: 0.9287
    
    Epoch 00008: val_loss improved from 0.36815 to 0.34986, saving model to best_model_11class.hdf5
    Epoch 9/30
    515/515 [==============================] - 267s 518ms/step - loss: 0.4935 - acc: 0.8845 - val_loss: 0.3358 - val_acc: 0.9309
    
    Epoch 00009: val_loss improved from 0.34986 to 0.33582, saving model to best_model_11class.hdf5
    Epoch 10/30
    515/515 [==============================] - 267s 518ms/step - loss: 0.4643 - acc: 0.8899 - val_loss: 0.3292 - val_acc: 0.9298
    
    Epoch 00010: val_loss improved from 0.33582 to 0.32921, saving model to best_model_11class.hdf5
    Epoch 11/30
    515/515 [==============================] - 265s 515ms/step - loss: 0.4271 - acc: 0.8994 - val_loss: 0.3190 - val_acc: 0.9346
    
    Epoch 00011: val_loss improved from 0.32921 to 0.31903, saving model to best_model_11class.hdf5
    Epoch 12/30
    515/515 [==============================] - 266s 516ms/step - loss: 0.4149 - acc: 0.9061 - val_loss: 0.3154 - val_acc: 0.9331
    
    Epoch 00012: val_loss improved from 0.31903 to 0.31541, saving model to best_model_11class.hdf5
    Epoch 13/30
    515/515 [==============================] - 267s 518ms/step - loss: 0.4009 - acc: 0.9078 - val_loss: 0.3127 - val_acc: 0.9306
    
    Epoch 00013: val_loss improved from 0.31541 to 0.31271, saving model to best_model_11class.hdf5
    Epoch 14/30
    515/515 [==============================] - 265s 514ms/step - loss: 0.3710 - acc: 0.9201 - val_loss: 0.3085 - val_acc: 0.9353
    
    Epoch 00014: val_loss improved from 0.31271 to 0.30854, saving model to best_model_11class.hdf5
    Epoch 15/30
    515/515 [==============================] - 265s 515ms/step - loss: 0.3472 - acc: 0.9258 - val_loss: 0.3025 - val_acc: 0.9397
    
    Epoch 00015: val_loss improved from 0.30854 to 0.30252, saving model to best_model_11class.hdf5
    Epoch 16/30
    515/515 [==============================] - 264s 513ms/step - loss: 0.3503 - acc: 0.9288 - val_loss: 0.3037 - val_acc: 0.9382
    
    Epoch 00016: val_loss did not improve from 0.30252
    Epoch 17/30
    515/515 [==============================] - 266s 517ms/step - loss: 0.3065 - acc: 0.9388 - val_loss: 0.3006 - val_acc: 0.9379
    
    Epoch 00017: val_loss improved from 0.30252 to 0.30056, saving model to best_model_11class.hdf5
    Epoch 18/30
    515/515 [==============================] - 264s 513ms/step - loss: 0.2928 - acc: 0.9416 - val_loss: 0.3051 - val_acc: 0.9397
    
    Epoch 00018: val_loss did not improve from 0.30056
    Epoch 19/30
    515/515 [==============================] - 266s 516ms/step - loss: 0.2882 - acc: 0.9466 - val_loss: 0.2983 - val_acc: 0.9397
    
    Epoch 00019: val_loss improved from 0.30056 to 0.29834, saving model to best_model_11class.hdf5
    Epoch 20/30
    515/515 [==============================] - 264s 513ms/step - loss: 0.2728 - acc: 0.9492 - val_loss: 0.2909 - val_acc: 0.9419
    
    Epoch 00020: val_loss improved from 0.29834 to 0.29090, saving model to best_model_11class.hdf5
    Epoch 21/30
    515/515 [==============================] - 264s 513ms/step - loss: 0.2666 - acc: 0.9525 - val_loss: 0.2930 - val_acc: 0.9401
    
    Epoch 00021: val_loss did not improve from 0.29090
    Epoch 22/30
    515/515 [==============================] - 265s 515ms/step - loss: 0.2577 - acc: 0.9520 - val_loss: 0.2953 - val_acc: 0.9379
    
    Epoch 00022: val_loss did not improve from 0.29090
    Epoch 23/30
    515/515 [==============================] - 266s 517ms/step - loss: 0.2454 - acc: 0.9548 - val_loss: 0.2921 - val_acc: 0.9404
    
    Epoch 00023: val_loss did not improve from 0.29090
    Epoch 24/30
    515/515 [==============================] - 264s 512ms/step - loss: 0.2416 - acc: 0.9577 - val_loss: 0.2914 - val_acc: 0.9404
    
    Epoch 00024: val_loss did not improve from 0.29090
    Epoch 25/30
    515/515 [==============================] - 265s 514ms/step - loss: 0.2275 - acc: 0.9627 - val_loss: 0.2939 - val_acc: 0.9404
    
    Epoch 00025: val_loss did not improve from 0.29090
    Epoch 26/30
    515/515 [==============================] - 265s 514ms/step - loss: 0.2123 - acc: 0.9685 - val_loss: 0.2950 - val_acc: 0.9386
    
    Epoch 00026: val_loss did not improve from 0.29090
    Epoch 27/30
    515/515 [==============================] - 262s 509ms/step - loss: 0.2106 - acc: 0.9668 - val_loss: 0.2926 - val_acc: 0.9412
    
    Epoch 00027: val_loss did not improve from 0.29090
    Epoch 28/30
    515/515 [==============================] - 264s 512ms/step - loss: 0.2117 - acc: 0.9684 - val_loss: 0.2905 - val_acc: 0.9415
    
    Epoch 00028: val_loss improved from 0.29090 to 0.29052, saving model to best_model_11class.hdf5
    Epoch 29/30
    515/515 [==============================] - 263s 511ms/step - loss: 0.1959 - acc: 0.9732 - val_loss: 0.2897 - val_acc: 0.9415
    
    Epoch 00029: val_loss improved from 0.29052 to 0.28970, saving model to best_model_11class.hdf5
    Epoch 30/30
    515/515 [==============================] - 262s 509ms/step - loss: 0.1977 - acc: 0.9709 - val_loss: 0.2915 - val_acc: 0.9426
    
    Epoch 00030: val_loss did not improve from 0.28970



```python
class_map_11 = train_generator.class_indices
class_map_11
```




    {'apple_pie': 0,
     'beef_carpaccio': 1,
     'bibimbap': 2,
     'cup_cakes': 3,
     'foie_gras': 4,
     'french_fries': 5,
     'garlic_bread': 6,
     'pizza': 7,
     'spaghetti_carbonara': 8,
     'spring_rolls': 9,
     'strawberry_shortcake': 10}




```python
plot_accuracy(history_11class, 'FOOD101-Inceptionv3-11classes')
plot_loss(history_11class, 'FOOD101-Inceptionv3-11classes')
```


![png](2.%20Multiclass%20Classification%20using%20Keras%20and%20TensorFlow%20on%20Food-101%20Dataset_files/2.%20Multiclass%20Classification%20using%20Keras%20and%20TensorFlow%20on%20Food-101%20Dataset_62_0.png)



![png](2.%20Multiclass%20Classification%20using%20Keras%20and%20TensorFlow%20on%20Food-101%20Dataset_files/2.%20Multiclass%20Classification%20using%20Keras%20and%20TensorFlow%20on%20Food-101%20Dataset_62_1.png)


## Testing the trained model using images downloaded from Internet


```python
%%time
# Load Model
K.clear_session()
model_best = load_model('best_model_11class.hdf5', compile=False)
```

    CPU times: user 12.2 s, sys: 242 ms, total: 12.4 s
    Wall time: 12.6 s



```python
# Downloading images from internet using the URLs
!wget -O cupcakes.jpg https://www.publicdomainpictures.net/pictures/110000/nahled/halloween-witch-cupcakes.jpg
!wget -O springrolls.jpg https://upload.wikimedia.org/wikipedia/commons/6/6f/Vietnamese_spring_rolls.jpg
!wget -O pizza.jpg http://104.130.3.186/assets/itemimages/400/400/3/default_9b4106b8f65359684b3836096b4524c8_pizza%20dreamstimesmall_94940296.jpg
!wget -O garlicbread.jpg https://c1.staticflickr.com/1/84/262952165_7ba3466108_z.jpg?zz=1
```

    --2020-04-08 09:59:33--  https://www.publicdomainpictures.net/pictures/110000/nahled/halloween-witch-cupcakes.jpg
    Resolving www.publicdomainpictures.net (www.publicdomainpictures.net)... 104.20.45.162, 104.20.44.162, 2606:4700:10::6814:2da2, ...
    Connecting to www.publicdomainpictures.net (www.publicdomainpictures.net)|104.20.45.162|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 51698 (50K) [image/jpeg]
    Saving to: 'cupcakes.jpg'
    
    cupcakes.jpg        100%[===================>]  50.49K  --.-KB/s    in 0.002s  
    
    2020-04-08 09:59:33 (22.9 MB/s) - 'cupcakes.jpg' saved [51698/51698]
    
    --2020-04-08 09:59:33--  https://upload.wikimedia.org/wikipedia/commons/6/6f/Vietnamese_spring_rolls.jpg
    Resolving upload.wikimedia.org (upload.wikimedia.org)... 208.80.154.240, 2620:0:861:ed1a::2:b
    Connecting to upload.wikimedia.org (upload.wikimedia.org)|208.80.154.240|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 4154557 (4.0M) [image/jpeg]
    Saving to: 'springrolls.jpg'
    
    springrolls.jpg     100%[===================>]   3.96M  15.4MB/s    in 0.3s    
    
    2020-04-08 09:59:34 (15.4 MB/s) - 'springrolls.jpg' saved [4154557/4154557]
    
    --2020-04-08 09:59:34--  http://104.130.3.186/assets/itemimages/400/400/3/default_9b4106b8f65359684b3836096b4524c8_pizza%20dreamstimesmall_94940296.jpg
    Connecting to 104.130.3.186:80... ^C
    --2020-04-08 09:59:56--  https://c1.staticflickr.com/1/84/262952165_7ba3466108_z.jpg?zz=1
    Resolving c1.staticflickr.com (c1.staticflickr.com)... 13.225.211.163, 2600:9000:202c:7000:0:5a51:64c9:c681, 2600:9000:202c:b600:0:5a51:64c9:c681, ...
    Connecting to c1.staticflickr.com (c1.staticflickr.com)|13.225.211.163|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: unspecified [image/jpeg]
    Saving to: 'garlicbread.jpg'
    
    garlicbread.jpg         [ <=>                ]  57.53K  --.-KB/s    in 0.002s  
    
    2020-04-08 09:59:56 (29.7 MB/s) - 'garlicbread.jpg' saved [58907]
    



```python
# Make a list of downloaded images and test the trained model
images = []
images.append('cupcakes.jpg')
images.append('pizza.jpg')
images.append('springrolls.jpg')
images.append('garlicbread.jpg')
predict_class(model_best, images, True)
```


![png](2.%20Multiclass%20Classification%20using%20Keras%20and%20TensorFlow%20on%20Food-101%20Dataset_files/2.%20Multiclass%20Classification%20using%20Keras%20and%20TensorFlow%20on%20Food-101%20Dataset_66_0.png)



![png](2.%20Multiclass%20Classification%20using%20Keras%20and%20TensorFlow%20on%20Food-101%20Dataset_files/2.%20Multiclass%20Classification%20using%20Keras%20and%20TensorFlow%20on%20Food-101%20Dataset_66_1.png)



![png](2.%20Multiclass%20Classification%20using%20Keras%20and%20TensorFlow%20on%20Food-101%20Dataset_files/2.%20Multiclass%20Classification%20using%20Keras%20and%20TensorFlow%20on%20Food-101%20Dataset_66_2.png)



![png](2.%20Multiclass%20Classification%20using%20Keras%20and%20TensorFlow%20on%20Food-101%20Dataset_files/2.%20Multiclass%20Classification%20using%20Keras%20and%20TensorFlow%20on%20Food-101%20Dataset_66_3.png)


* The model did well even when the number of classes are increased to 11
* Model training on all 101 classes takes some time
* It was taking more than an hour for one epoch when the full dataset is used for fine tuning




## Visualization of intermediate activations

* Load the saved model and a test image


```python
K.clear_session()
print("Loading the model...")
model = load_model('best_model_3class.hdf5', compile=False)
print("Done!")
```

    Loading the model...
    Done!


Define helper functions to visualize the activations


```python
def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1
    
    # clip to [0,1]
    x += 0.5
    x = np.clip(x, 0, 1) # value outside interval are clipped to the interval edge
    
    # convert to RGB array
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x
```


```python
def generate_pattern(layer_name, filter_index, size=150):
    # Build a loss function that maximizes the activation
    # of the nth filter of the layer considered
    layer_output = model.get_layer(layer_name).output
    loss = K.mean(layer_output[:,:,:,filter_index])
    
    # compute the gradient of the input picture wrt this loss
    grads = K.gradients(loss, model.input)[0]
    
    # Normalization trick: we normalize the gradient
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

    # This function returns the loss and grads given the input picture
    iterate = K.function([model.input], [loss, grads])
    
    # We start from a gray image with some noise
    input_img_data = np.random.random((1, size, size, 3)) * 20 + 128.

    # Run gradient ascent for 40 steps
    step = 1.
    for i in range(40):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step
        
    img = input_img_data[0]
    return deprocess_image(img)
```


```python
def get_activation(img, model_activation):
    img = image.load_img(img, target_size=(299,299))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img /= 255
    plt.imshow(img[0])
    plt.show()
    return model_activation.predict(img)
```


```python
def show_activations(activations, layer_names):
    images_per_row = 16
    # Display feature map
    for layer_name, layer_activation in zip(layer_names, activations):
        # This is the number of features in the feature map
        n_features = layer_activation.shape[-1]
        # The feature map has shape (1, size, size, n_features)
        size = layer_activation.shape[1]
        n_cols = n_features //images_per_row
        display_grid = np.zeros((size*n_cols, images_per_row*size))
        
        for col in range(n_cols):
            for row in range(images_per_row):
                channel_image = layer_activation[0,:,:,col*images_per_row+row]
                # Post process feature to make it visually palatable
                channel_image -= channel_image.mean()
                channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                display_grid[col * size : (col + 1) * size,
                             row * size : (row + 1) * size] = channel_image
        # Display the grid
        scale = 1./size
        plt.figure(figsize=(scale * display_grid.shape[1],
                          scale * display_grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto',cmap='viridis')
    plt.show()
```

**Select and visualize activations of first 10 layers**


```python
layers = [layer.output for layer in model.layers[1:11]]
activations_output = models.Model(input=model.input, outputs=layers)
```

    /usr/local/lib/python2.7/dist-packages/ipykernel_launcher.py:2: UserWarning: Update your `Model` call to the Keras 2 API: `Model(outputs=[<tf.Tenso..., inputs=Tensor("in...)`
      



```python
layers
```




    [<tf.Tensor 'conv2d_1/convolution:0' shape=(?, ?, ?, 32) dtype=float32>,
     <tf.Tensor 'batch_normalization_1/cond/Merge:0' shape=(?, ?, ?, 32) dtype=float32>,
     <tf.Tensor 'activation_1/Relu:0' shape=(?, ?, ?, 32) dtype=float32>,
     <tf.Tensor 'conv2d_2/convolution:0' shape=(?, ?, ?, 32) dtype=float32>,
     <tf.Tensor 'batch_normalization_2/cond/Merge:0' shape=(?, ?, ?, 32) dtype=float32>,
     <tf.Tensor 'activation_2/Relu:0' shape=(?, ?, ?, 32) dtype=float32>,
     <tf.Tensor 'conv2d_3/convolution:0' shape=(?, ?, ?, 64) dtype=float32>,
     <tf.Tensor 'batch_normalization_3/cond/Merge:0' shape=(?, ?, ?, 64) dtype=float32>,
     <tf.Tensor 'activation_3/Relu:0' shape=(?, ?, ?, 64) dtype=float32>,
     <tf.Tensor 'max_pooling2d_1/MaxPool:0' shape=(?, ?, ?, 64) dtype=float32>]



As seen below, the 10 chosen layers contain 3 convolution, 3 batch normalization, 3 activation and 1 max pooling layers

**Get names of selected layers**


```python
layer_names = []
for layer in model.layers[1:11]:
    layer_names.append(layer.name)
print layer_names
```

    [u'conv2d_1', u'batch_normalization_1', u'activation_1', u'conv2d_2', u'batch_normalization_2', u'activation_2', u'conv2d_3', u'batch_normalization_3', u'activation_3', u'max_pooling2d_1']


**Provide an input to the model and get the activations of all the 10 chosen layers**


```python
food = 'apple_pie.jpg'
activations = get_activation(food, activations_output)
```


![png](2.%20Multiclass%20Classification%20using%20Keras%20and%20TensorFlow%20on%20Food-101%20Dataset_files/2.%20Multiclass%20Classification%20using%20Keras%20and%20TensorFlow%20on%20Food-101%20Dataset_82_0.png)



```python
show_activations(activations, layer_names)
```

    /usr/local/lib/python2.7/dist-packages/ipykernel_launcher.py:17: RuntimeWarning: invalid value encountered in divide



![png](2.%20Multiclass%20Classification%20using%20Keras%20and%20TensorFlow%20on%20Food-101%20Dataset_files/2.%20Multiclass%20Classification%20using%20Keras%20and%20TensorFlow%20on%20Food-101%20Dataset_83_1.png)



![png](2.%20Multiclass%20Classification%20using%20Keras%20and%20TensorFlow%20on%20Food-101%20Dataset_files/2.%20Multiclass%20Classification%20using%20Keras%20and%20TensorFlow%20on%20Food-101%20Dataset_83_2.png)



![png](2.%20Multiclass%20Classification%20using%20Keras%20and%20TensorFlow%20on%20Food-101%20Dataset_files/2.%20Multiclass%20Classification%20using%20Keras%20and%20TensorFlow%20on%20Food-101%20Dataset_83_3.png)



![png](2.%20Multiclass%20Classification%20using%20Keras%20and%20TensorFlow%20on%20Food-101%20Dataset_files/2.%20Multiclass%20Classification%20using%20Keras%20and%20TensorFlow%20on%20Food-101%20Dataset_83_4.png)



![png](2.%20Multiclass%20Classification%20using%20Keras%20and%20TensorFlow%20on%20Food-101%20Dataset_files/2.%20Multiclass%20Classification%20using%20Keras%20and%20TensorFlow%20on%20Food-101%20Dataset_83_5.png)



![png](2.%20Multiclass%20Classification%20using%20Keras%20and%20TensorFlow%20on%20Food-101%20Dataset_files/2.%20Multiclass%20Classification%20using%20Keras%20and%20TensorFlow%20on%20Food-101%20Dataset_83_6.png)



![png](2.%20Multiclass%20Classification%20using%20Keras%20and%20TensorFlow%20on%20Food-101%20Dataset_files/2.%20Multiclass%20Classification%20using%20Keras%20and%20TensorFlow%20on%20Food-101%20Dataset_83_7.png)



![png](2.%20Multiclass%20Classification%20using%20Keras%20and%20TensorFlow%20on%20Food-101%20Dataset_files/2.%20Multiclass%20Classification%20using%20Keras%20and%20TensorFlow%20on%20Food-101%20Dataset_83_8.png)



![png](2.%20Multiclass%20Classification%20using%20Keras%20and%20TensorFlow%20on%20Food-101%20Dataset_files/2.%20Multiclass%20Classification%20using%20Keras%20and%20TensorFlow%20on%20Food-101%20Dataset_83_9.png)



![png](2.%20Multiclass%20Classification%20using%20Keras%20and%20TensorFlow%20on%20Food-101%20Dataset_files/2.%20Multiclass%20Classification%20using%20Keras%20and%20TensorFlow%20on%20Food-101%20Dataset_83_10.png)


* What we see in the above plots are the activations or the outputs of each of the 11 layers we chose
* The activations or the outputs from the 1st layer(conv2d_1) don't lose much information of the original input
* They are the results of applying several edge detecting filters on the input image
* With each added layer, the activations lose visual/input information and keeps building on the class/ouput information
* As the depth increases, the layers activations become less visually interpretabale and more abstract
* By doing so, they learn to detect more specific features of the class rather than just edges and curves
* We plotted just 10 out of 314 intermediate layers. We already have in these few layers, activations which are blank/sparse(for ex: the 2 blank activations in the layer activation_2)
* These blank/sparse activations are caused when any of the filters used in that layer didn't find a matching pattern in the input given to it
* By plotting more layers(specially those towards the end of the network), we can observe more of these sparse activations and how the layers get more abstract



**Plot the activations for another input image**


```python
food = 'pizza.jpg'
activations = get_activation(food, activations_output)
show_activations(activations, layer_names)
```


![png](2.%20Multiclass%20Classification%20using%20Keras%20and%20TensorFlow%20on%20Food-101%20Dataset_files/2.%20Multiclass%20Classification%20using%20Keras%20and%20TensorFlow%20on%20Food-101%20Dataset_86_0.png)


    /usr/local/lib/python2.7/dist-packages/ipykernel_launcher.py:17: RuntimeWarning: invalid value encountered in divide



![png](2.%20Multiclass%20Classification%20using%20Keras%20and%20TensorFlow%20on%20Food-101%20Dataset_files/2.%20Multiclass%20Classification%20using%20Keras%20and%20TensorFlow%20on%20Food-101%20Dataset_86_2.png)



![png](2.%20Multiclass%20Classification%20using%20Keras%20and%20TensorFlow%20on%20Food-101%20Dataset_files/2.%20Multiclass%20Classification%20using%20Keras%20and%20TensorFlow%20on%20Food-101%20Dataset_86_3.png)



![png](2.%20Multiclass%20Classification%20using%20Keras%20and%20TensorFlow%20on%20Food-101%20Dataset_files/2.%20Multiclass%20Classification%20using%20Keras%20and%20TensorFlow%20on%20Food-101%20Dataset_86_4.png)



![png](2.%20Multiclass%20Classification%20using%20Keras%20and%20TensorFlow%20on%20Food-101%20Dataset_files/2.%20Multiclass%20Classification%20using%20Keras%20and%20TensorFlow%20on%20Food-101%20Dataset_86_5.png)



![png](2.%20Multiclass%20Classification%20using%20Keras%20and%20TensorFlow%20on%20Food-101%20Dataset_files/2.%20Multiclass%20Classification%20using%20Keras%20and%20TensorFlow%20on%20Food-101%20Dataset_86_6.png)



![png](2.%20Multiclass%20Classification%20using%20Keras%20and%20TensorFlow%20on%20Food-101%20Dataset_files/2.%20Multiclass%20Classification%20using%20Keras%20and%20TensorFlow%20on%20Food-101%20Dataset_86_7.png)



![png](2.%20Multiclass%20Classification%20using%20Keras%20and%20TensorFlow%20on%20Food-101%20Dataset_files/2.%20Multiclass%20Classification%20using%20Keras%20and%20TensorFlow%20on%20Food-101%20Dataset_86_8.png)



![png](2.%20Multiclass%20Classification%20using%20Keras%20and%20TensorFlow%20on%20Food-101%20Dataset_files/2.%20Multiclass%20Classification%20using%20Keras%20and%20TensorFlow%20on%20Food-101%20Dataset_86_9.png)



![png](2.%20Multiclass%20Classification%20using%20Keras%20and%20TensorFlow%20on%20Food-101%20Dataset_files/2.%20Multiclass%20Classification%20using%20Keras%20and%20TensorFlow%20on%20Food-101%20Dataset_86_10.png)



![png](2.%20Multiclass%20Classification%20using%20Keras%20and%20TensorFlow%20on%20Food-101%20Dataset_files/2.%20Multiclass%20Classification%20using%20Keras%20and%20TensorFlow%20on%20Food-101%20Dataset_86_11.png)


* The feature maps in the above activations are for a different input image
* We see the same patterns discussed for the previous input image
* It is interesting to see the blank/sparse activations in the same layer(activation_2) and for same filters when a different image is passed to the network
* Remember we used a pretrained Inceptionv3 model. All the filters that are used in different layers come from this pretrained model



## Heat Map Visualizations of Class Activation

* So far we were doing activation maps visualization
* This helps in understanding how the input is transformed from one layer to another as it goes through several operations
* At the end of training, we want the model to classify or detect objects based on features which are specific to the class
* For example, when training a dogs vs cats classifier, the model should detect dogs based on features relevant to dog but not cats
* To validate how model attributes the features to class output, we can generate heat maps using gradients to find out which regions in the input images were instrumental in determining the class



```python
def get_attribution(food):
    img = image.load_img(food, target_size=(299, 299))
    img = image.img_to_array(img) 
    img /= 255. 
    f,ax = plt.subplots(1,3, figsize=(15,15))
    ax[0].imshow(img)
    
    img = np.expand_dims(img, axis=0) 
    
    preds = model.predict(img)
    class_id = np.argmax(preds[0])
    ax[0].set_title("Input Image")
    class_output = model.output[:, class_id]
    last_conv_layer = model.get_layer("mixed10")
    
    grads = K.gradients(class_output, last_conv_layer.output)[0]
    pooled_grads = K.mean(grads, axis=(0, 1, 2))
    iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
    pooled_grads_value, conv_layer_output_value = iterate([img])
    for i in range(2048):
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
    
    heatmap = np.mean(conv_layer_output_value, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    ax[1].imshow(heatmap)
    ax[1].set_title("Heat map")
    
    
    act_img = cv2.imread(food)
    heatmap = cv2.resize(heatmap, (act_img.shape[1], act_img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed = cv2.addWeighted(act_img, 0.6, heatmap, 0.4, 0)
    cv2.imwrite('classactivation.png', superimposed)
    img_act = image.load_img('classactivation.png', target_size=(299, 299))
    ax[2].imshow(img_act)
    ax[2].set_title("Class Activation")
    plt.show()
    return preds
```


```python
print("Showing the class map..")
print(class_map_3)
```

    Showing the class map..
    {'omelette': 1, 'apple_pie': 0, 'pizza': 2}



```python
pred = get_attribution('apple_pie.jpg')
print "Softmax predictions: ", pred
```


![png](2.%20Multiclass%20Classification%20using%20Keras%20and%20TensorFlow%20on%20Food-101%20Dataset_files/2.%20Multiclass%20Classification%20using%20Keras%20and%20TensorFlow%20on%20Food-101%20Dataset_92_0.png)


    Softmax predictions:  [[9.9428177e-01 5.5873864e-03 1.3088895e-04]]


* In the above plot, we see on left the input image passed to the model, heat map in the middle and the class activation map on right
* Heat map gives a visual of what regions in the image were used in determining the class of the image
* Now it's clearly visible what a model looks for in an image if it has to be classified as an applepie!



**See how the class activation map looks for a different image**


```python
pred = get_attribution('omelette.jpg')
print("Here are softmax predictions..",pred)
```


![png](2.%20Multiclass%20Classification%20using%20Keras%20and%20TensorFlow%20on%20Food-101%20Dataset_files/2.%20Multiclass%20Classification%20using%20Keras%20and%20TensorFlow%20on%20Food-101%20Dataset_95_0.png)


    ('Here are softmax predictions..', array([[0.00220886, 0.9944366 , 0.00335447]], dtype=float32))


We can see how the heat map is different for a different image i.e the model looks for a totally different features/regions if it has to classify it as a pizza

**Save notebook session**


```python
import dill
dill.dump_session('notebook_env.db')
```
