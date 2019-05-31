from keras.layers import Conv2D, UpSampling2D, MaxPooling2D, BatchNormalization, Permute, ConvLSTM2D, Reshape, Conv3D, Lambda, Add, Input, TimeDistributed, Multiply, ReLU, concatenate, GaussianNoise, Conv2DTranspose
from keras.models import Sequential, Model
from keras import regularizers
from keras.callbacks import Callback
import random
import glob
import wandb
from wandb.keras import WandbCallback
import subprocess
import os
from PIL import Image
import numpy as np
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
import pdb

run = wandb.init(project='catz-may13', entity='qualcomm')
config = run.config

config.num_epochs = 20
config.batch_size = 32
config.img_dir = "images"
config.height = 96
config.width = 96

val_dir = 'catz/test'
train_dir = 'catz/train'

img_gen = ImageDataGenerator()
transform_parameters = {'theta':10,
                        'tx':0.1,
                        'ty':0.05,
                        'shear':25,
                        'zx':1.1,
                        'zy':1.1,
                        'channel_shift_intencity':0.75,
                        'brightness':0.75 }

# automatically get the data if it doesn't exist
if not os.path.exists("catz"):
    print("Downloading catz dataset...")
    subprocess.check_output(
        "curl https://storage.googleapis.com/wandb/catz.tar.gz | tar xz", shell=True)


class ImageCallback(Callback):
    def on_epoch_end(self, epoch, logs):
        validation_X, validation_y = next(
            my_generator(32, val_dir))
        output = self.model.predict(validation_X)
        wandb.log({
            "input": [wandb.Image(np.concatenate(np.split(c, 5, axis=2), axis=1)) for c in validation_X],
            "output": [wandb.Image(np.concatenate([validation_y[i], o], axis=1)) for i, o in enumerate(output)]
        }, commit=False)


def my_generator(batch_size, img_dir):
    """A generator that returns 5 images plus a result image"""
    cat_dirs = glob.glob(img_dir + "/*")
    counter = 0
    while True:
        input_images = np.zeros(
            (batch_size*2, config.width, config.height, 3 * 5))
        output_images = np.zeros((batch_size*2, config.width, config.height, 3))
        random.shuffle(cat_dirs)
        if ((counter+1)*batch_size >= len(cat_dirs)):
            counter = 0
        for i in range(batch_size):
            input_imgs = glob.glob(cat_dirs[counter + i] + "/cat_[0-5]*")
            imgs = [Image.open(img) for img in sorted(input_imgs)]
            input_images[i] = np.concatenate(imgs, axis=2)
            transformed_imgs = []
            transformed_imgs.append ( img_gen.apply_transform(np.asarray(imgs[0]), transform_parameters) )
            transformed_imgs.append ( img_gen.apply_transform(np.asarray(imgs[1]), transform_parameters) )
            transformed_imgs.append ( img_gen.apply_transform(np.asarray(imgs[2]), transform_parameters) )
            transformed_imgs.append ( img_gen.apply_transform(np.asarray(imgs[3]), transform_parameters) )
            transformed_imgs.append ( img_gen.apply_transform(np.asarray(imgs[4]), transform_parameters) )
            input_images[i+batch_size] = np.concatenate(transformed_imgs, axis=2)
            output_images[i] = np.array(Image.open(cat_dirs[counter + i] + "/cat_result.jpg"))
            output_images[i+batch_size] = img_gen.apply_transform(np.asarray(output_images[i]), transform_parameters)            
        yield (input_images, output_images)
        counter += batch_size
        
def my_generator_no_aug(batch_size, img_dir):
    """A generator that returns 5 images plus a result image"""
    cat_dirs = glob.glob(img_dir + "/*")
    counter = 0
    while True:
        input_images = np.zeros(
            (batch_size, config.width, config.height, 3 * 5))
        output_images = np.zeros((batch_size, config.width, config.height, 3))
        random.shuffle(cat_dirs)
        if ((counter+1)*batch_size >= len(cat_dirs)):
            counter = 0
        for i in range(batch_size):
            input_imgs = glob.glob(cat_dirs[counter + i] + "/cat_[0-5]*")
            imgs = [Image.open(img) for img in sorted(input_imgs)]
            input_images[i] = np.concatenate(imgs, axis=2)
            output_images[i] = np.array(Image.open(cat_dirs[counter + i] + "/cat_result.jpg"))           
        yield (input_images, output_images)
        counter += batch_size

def extract_last(img_seq):
    tmp_in = img_seq[:,-1,:,:,:];
    #tmp_out = Conv2D(3, (1,1), padding='same')(tmp_in)
    return tmp_in

def per_feature_network (img_seq_per_feature):
    conv2d0_out = TimeDistributed(Conv2D(16, (3,3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01)))(img_seq_per_feature)
    #maxpool0_out = TimeDistributed(MaxPooling2D((2,2)))(conv2d0_out)
    #conv2d1_out = TimeDistributed(Conv2D(8, (5,5), padding='same', activation='relu'))(conv2d0_out)
    #maxpool1_out = TimeDistributed(MaxPooling2D((2,2)))(conv2d1_out)
    #batchnorm1_out = TimeDistributed(BatchNormalization())(conv2d0_out)

    convlstm2d_out = ConvLSTM2D(10, (3,3), padding='same', recurrent_dropout=0.2, dropout=0.2, 
    activation='tanh',recurrent_activation='hard_sigmoid', return_sequences=False)(conv2d0_out)

    conv2d2_out = Conv2D(16, (3,3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01))(convlstm2d_out)
    #upsample1_out = UpSampling2D((2,2))(conv2d2_out)
    #deconv2d1_out = Conv2DTranspose(8, (3,3), strides=(1,1), padding='same', output_padding=None, activation='relu')(convlstm2d_out)
    
    #batchnorm2_out = BatchNormalization()(conv2d1_out)
    conv2d3_out = Conv2D(1, (3,3), padding='same', kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01))(conv2d2_out)
    #upsample2_out = UpSampling2D((2,2))(conv2d3_out)
    #deconv2d2_out = Conv2DTranspose(8, (5,5), strides=(1,1), padding='same', output_padding=None, activation='relu')(deconv2d1_out)
    
    #conv2d4_out = Conv2D(1, (3,3), padding='same')(deconv2d2_out)
    relu_out = ReLU(max_value=255, negative_slope=0.0, threshold=0.0)(conv2d3_out)
    return relu_out

def extract_red (img_seq):
    tmp = img_seq[:,:,:,:,0]
    reshape_out = Reshape((5,96,96,1))(tmp) 
    return reshape_out

def extract_green (img_seq):
    tmp = img_seq[:,:,:,:,1]
    reshape_out = Reshape((5,96,96,1))(tmp) 
    return reshape_out

def extract_blue (img_seq):
    tmp = img_seq[:,:,:,:,2]
    reshape_out = Reshape((5,96,96,1))(tmp) 
    return reshape_out

def form_image (features_in):
    tmp = np.zeros((32,96,96,3))
    tmp[:,:,:,0] = features_in[0]
    tmp[:,:,:,1] = features_in[1]
    tmp[:,:,:,2] = features_in[2]
    return tmp

        
inp = Input(shape=(config.height, config.width, 5 * 3))
reshape_out = Reshape((config.height,config.width,5,3))(inp)
perm_out = Permute((3,1,2,4))(reshape_out)
#noise_out = GaussianNoise(2)(perm_out)

red_extract_layer = Lambda(extract_red, ((5,96,96,1)))
green_extract_layer = Lambda(extract_green, ((5,96,96,1)))
blue_extract_layer = Lambda(extract_blue, ((5,96,96,1)))
red_in = red_extract_layer(perm_out)
green_in = green_extract_layer(perm_out)
blue_in = blue_extract_layer(perm_out)

#red_network = Lambda(per_feature_network, ((96,96,1)))
#green_network = Lambda(per_feature_network, ((96,96,1)))
#blue_network = Lambda(per_feature_network, ((96,96,1)))

red_out = per_feature_network(red_in)
green_out = per_feature_network(green_in)
blue_out = per_feature_network(blue_in)

merged1 = concatenate([red_out, green_out])
merged2 = concatenate([merged1, blue_out])
#conv_color = Conv2D(3,(1,1),padding='same')(merged2)
#relu_out2 = ReLU(max_value=255, negative_slope=0.0, threshold=0.0)(conv_color)

layer = Lambda(extract_last, ((96,96,3)))
last_out = layer(perm_out)

concat_out =concatenate([merged2, last_out])
conv2d00_out = Conv2D(3, (1,1), padding='same')(concat_out)
relu00_out = ReLU(max_value=255, negative_slope=0.0, threshold=0.0)(conv2d00_out)

#add_out = Add()([merged2, last_out])
#relu01_out = ReLU(max_value=255, negative_slope=0.0, threshold=0.0)(add_out)


model = Model(inp, merged2)



def perceptual_distance(y_true, y_pred):
    #y_true = 256.0 * y_true
    #y_pred = 256.0 * y_pred
    rmean = (y_true[:, :, :, 0] + y_pred[:, :, :, 0]) / 2
    r = y_true[:, :, :, 0] - y_pred[:, :, :, 0]
    g = y_true[:, :, :, 1] - y_pred[:, :, :, 1]
    b = y_true[:, :, :, 2] - y_pred[:, :, :, 2]
    return K.mean(K.sqrt((((512+rmean)*r*r)/256) + 4*g*g + (((767-rmean)*b*b)/256)))


model.compile(optimizer='adam', loss=[perceptual_distance], metrics=[perceptual_distance])
model.summary()

model.fit_generator(my_generator(config.batch_size, train_dir),
                    steps_per_epoch=len(glob.glob(train_dir + "/*")) // config.batch_size,
                    epochs=config.num_epochs, callbacks=[
    ImageCallback(), WandbCallback()],
    validation_steps=len(glob.glob(val_dir + "/*")) // config.batch_size,
    validation_data=my_generator_no_aug(config.batch_size, val_dir))
