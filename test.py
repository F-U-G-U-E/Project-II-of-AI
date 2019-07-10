
# coding: utf-8

# # Test.py
# To predict the vein image based on leaf. 

# In[114]:


import os
import sys
import numpy as np
import tensorflow as tf
from PIL import Image, ExifTags
import matplotlib.pyplot as plt
from keras.models import Model, load_model

import keras.backend as K
from keras.initializers import Initializer
from keras.models import Model, load_model
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import RMSprop, Adam
from keras.layers import Input, Conv3D, Conv2D, Dense, MaxPooling3D, Reshape,                        AveragePooling2D, MaxPooling2D, BatchNormalization, TimeDistributed, Flatten,                        Bidirectional, GRU, Masking, Dropout, GlobalMaxPool1D, Activation,                         dot, multiply, Lambda, GlobalAveragePooling1D, Permute, Multiply, Concatenate,                         SpatialDropout2D, Softmax


# In[4]:


### 使得这里的编号和nvidia-smi看到的编号是一样的
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
### 指定只能看到编号为0的GPwU
os.environ["CUDA_VISIBLE_DEVICES"]="3"
### 设定显存按需分配
config = tf.ConfigProto()  
config.gpu_options.allow_growth=True   
session = tf.Session(config=config)
K.tensorflow_backend.set_session(session)


# ## 0. Functions

# In[5]:


# clippling for RGB image
# pil_image: pil image. After converting to numpy array, shape (h, w, 3), e.g. (4032, 3024, 3)
# size: dest pil size (300, 400), after convering to numpy array, shape (h = 400, w = 300, c = 3)

def edge_padding(pil_image, size = (300, 400)): 
    
    raw_size = pil_image.size
    
    image = np.array(pil_image)
    height = image.shape[0]
    width = image.shape[1]
    
    # padding 
    if height * 3 == width * 4:
        # this is already the dest size
        padding_setting = [None, 0, 0, 0, 0]
        pass
    
    elif height * 3 > width * 4:
        # height too great, width too small
        dest_width = int(height * 3 / 4)
        pad_left = int((dest_width - width) / 2)
        pad_right = dest_width - width - pad_left
        image = np.pad(image, ((0, 0), (pad_left, pad_right), (0, 0)), 'edge')
        padding_setting = ['LR', 0, 0, pad_left, pad_right]
        
    elif height * 3 < width * 4:
        # width too great, height too small
        dest_height = int(width * 4 / 3)
        pad_top = int((dest_height - height) / 2)
        pad_down = dest_height - height - pad_top
        image = np.pad(image, ((pad_top, pad_down), (0, 0), (0, 0)), 'edge')
        padding_setting = ['UD', pad_top, pad_down, 0, 0]
        
    pil_image = Image.fromarray(image)
    temp_size = pil_image.size
    
    image = np.array(pil_image.resize(size, Image.ANTIALIAS))
        
        
    return image, temp_size, raw_size, padding_setting


# In[90]:


# clipping for SINGLE image
# np_image: numpy array with shape (h, w, 1)
# padding_setting = [How, Top, Bottom, Left, Right]
# temp_size = (h_temp, w_temp)
# raw_size = (h_raw, w_raw)

def edge_clippling(np_image, temp_size, raw_size, padding_setting = [None, 0, 0, 0, 0]):
    pil_image = Image.fromarray(np_image).resize(temp_size, Image.ANTIALIAS)
    image = np.array(pil_image)
    
    How, Top, Bottom, Left, Right = padding_setting
    w_raw, h_raw = raw_size
    
    
    if How == None: # without any padding
        return image
    elif How == 'LR': # padding at left and right
        
        return image[:, Left:(Left+w_raw)]
    elif How == 'UD': # padding at top and bottom
        return image[Top:Top+h_raw, :]
    
    


# ## 1. Preprocessing
# To the size and shape that network can process
# Record: Raw Size, Padding Size, 

# In[121]:


Path_Path = sys.argv[1]
try:
    Dest_Path = sys.argv[2]
except:
    Dest_Path = ''


# In[105]:


with open(Path_Path) as f:
    lines = f.readlines()
# print(lines)

X = []
Name = []
Temp = []
Raw = []
PadSet = []

print('Loading Images...')

for line in lines:
    name = line.strip('\n')
    
    try:
        pil_image = Image.open(name)
    except:
        print('Failed to open', name)
        continue
    
    name = os.path.basename(name).split('.')[0]
    Name.append(name)
    
    image, temp_size, raw_size, padding_setting = edge_padding(pil_image)
    X.append(image)
    Temp.append(temp_size)
    Raw.append(raw_size)
    PadSet.append(padding_setting)
    
X = np.array(X)
print(X.shape)


# In[66]:


X = X / 255.0


# ## 2. Predict
# Predict the Vein

# In[80]:


THRESHOLD = 0.50


# In[68]:


print('Predicting...')
model = load_model('test_model.h5', custom_objects={'tf': tf})
model_pred = Model(inputs=model.input,outputs=model.layers[8].output)


# In[69]:


pred = model_pred.predict([X, np.zeros((X.shape[0], 400, 300, 1))], batch_size = 8)


# In[74]:





# In[81]:


vein_pred = (pred > THRESHOLD).astype(np.uint8)


# In[82]:


vein_pred = 255 - 255 * vein_pred


# ## 3. Postprocessing

# In[100]:



dest_path = os.path.join(Dest_Path, 'vein_pred_pics')
if not os.path.exists(dest_path):
    os.makedirs(dest_path)
    print('Create path:', dest_path)
else:
    print('Path already exists:', dest_path)


# In[113]:


print('Generating Images...')
for i in range(len(vein_pred)):
    array = edge_clippling(np_image = vein_pred[i, :, :, 0], temp_size = Temp[i], raw_size = Raw[i], padding_setting = PadSet[i])
    pil_image = Image.fromarray(array)
    name = Name[i]
    print(os.path.join(dest_path, name)+'.jpg')
    pil_image.save(os.path.join(dest_path, name)+'.jpg', quality=95)
    

