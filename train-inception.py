from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D, Reshape, BatchNormalization, Input, Concatenate, LeakyReLU, UpSampling2D
from keras.models import Sequential, Model
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

run = wandb.init(project='catz-may13', entity='qualcomm')
config = run.config

config.num_epochs = 20
config.batch_size = 32
config.img_dir = "images"
config.height = 96
config.width = 96

val_dir = 'catz/test'
train_dir = 'catz/train'

# automatically get the data if it doesn't exist
if not os.path.exists("catz"):
    print("Downloading catz dataset...")
    subprocess.check_output(
        "curl https://storage.googleapis.com/wandb/catz.tar.gz | tar xz", shell=True)


class ImageCallback(Callback):
    def on_epoch_end(self, epoch, logs):
        validation_X, validation_y = next(
            my_generator(15, val_dir))
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
            (batch_size, config.width, config.height, 3 * 5))
        output_images = np.zeros((batch_size, config.width, config.height, 3))
        random.shuffle(cat_dirs)
        if ((counter+1)*batch_size >= len(cat_dirs)):
            counter = 0
        for i in range(batch_size):
            input_imgs = glob.glob(cat_dirs[counter + i] + "/cat_[0-5]*")
            imgs = [Image.open(img) for img in sorted(input_imgs)]
            input_images[i] = np.concatenate(imgs, axis=2)
            output_images[i] = np.array(Image.open(
                cat_dirs[counter + i] + "/cat_result.jpg"))
        yield (input_images, output_images)
        counter += batch_size

def inception_block(ip):
    conv1x1_out = Conv2D(32, (1,1), padding='same', activation="relu")(ip)
    conv3x3_out = Conv2D(32, (3,3), padding='same', activation="relu")(ip)
    conv5x5_out = Conv2D(32, (5,5), padding='same', activation="relu")(ip)
    maxpool3x3_out = MaxPooling2D(pool_size=(3, 3), strides=(1,1), padding='same')(ip)
    concat_out = Concatenate()([conv1x1_out, conv3x3_out, conv5x5_out, maxpool3x3_out])
    return concat_out
    
inp = Input(shape=(config.height, config.width, 5 * 3))
#reshape_out = Reshape((img_width, img_height, 1))(inp)
incept1_out = inception_block(inp)
maxpool1_out = MaxPooling2D((4,4))(incept1_out)
incept2_out = inception_block(maxpool1_out)
batchnorm1_out = BatchNormalization()(incept2_out)
conv2d3_out = Conv2D(3,(3,3),padding='same')(batchnorm1_out)
upsample1_out = UpSampling2D((4,4))(conv2d3_out)
#flat1_out = Flatten()(batchnorm1_out)
#dense1_out = Dense(128, activation="relu")(flat1_out)
#dropout1_out = Dropout(0.25)(dense1_out)
#dense2_out = Dense(64, activation="relu")(dropout1_out)
#dropout2_out = Dropout(0.2)(dense2_out)
#dense3_out = Dense(num_classes, activation="softmax")(dropout2_out)
model = Model(inp, upsample1_out)

    
def perceptual_distance(y_true, y_pred):
    rmean = (y_true[:, :, :, 0] + y_pred[:, :, :, 0]) / 2
    r = y_true[:, :, :, 0] - y_pred[:, :, :, 0]
    g = y_true[:, :, :, 1] - y_pred[:, :, :, 1]
    b = y_true[:, :, :, 2] - y_pred[:, :, :, 2]

    return K.mean(K.sqrt((((512+rmean)*r*r)/256) + 4*g*g + (((767-rmean)*b*b)/256)))


model.compile(optimizer='adam', loss=[perceptual_distance], metrics=[perceptual_distance])
model.summary()

model.fit_generator(my_generator(config.batch_size, train_dir),
                    steps_per_epoch=len(
                        glob.glob(train_dir + "/*")) // config.batch_size,
                    epochs=config.num_epochs, callbacks=[
    ImageCallback(), WandbCallback()],
    validation_steps=len(glob.glob(val_dir + "/*")) // config.batch_size,
    validation_data=my_generator(config.batch_size, val_dir))
