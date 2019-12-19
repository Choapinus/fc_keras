
import cv2
import keras
import keras.backend as K
import matplotlib.pyplot as plt

import os
import glob
import random
import numpy as np
# import skimage.io.imread
# from utils import utils, helpers

from keras.models import Model
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.regularizers import l2
from keras.layers import (
	Input, Conv2D, Conv2DTranspose, BatchNormalization, 
	Dropout, AveragePooling2D, MaxPooling2D, Concatenate, Activation
)

import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
allow_growth_session = tf.Session(config=config)
tf.keras.backend.set_session(allow_growth_session)

def denseLayer(x, filters, kernel_size=[3, 3], dropout_p=0.2):
	x = BatchNormalization()(x)
	x = Activation('relu')(x)
	x = Conv2D(
		filters, kernel_size=kernel_size, padding='same',
		# kernel_initializer='he_normal', # kernel_regularizer=l2(1e-4) # usa lecun_normal
	)(x)
	x = Dropout(dropout_p)(x)
	return x

def denseBlock(x, n_layers, growth_rate, dropout_p):
	new_features = []
	
	for i in range(n_layers):
		layer = denseLayer(x, growth_rate, dropout_p=dropout_p)
		new_features.append(layer)
		x = Concatenate()([x, layer])
	new_features = Concatenate()(new_features)
	
	return x, new_features

def TD(x, n_filters, dropout_p=0.2):
	# x = BatchNormalization()(x)
	# x = Activation('relu')(x)
	# x = Conv2D(
	# 	n_filters, kernel_size=(1, 1), padding='same',
	# 	# kernel_initializer='he_normal', # kernel_regularizer=l2(1e-4)
	# )(x)
	# x = Dropout(dropout_p)(x)
	# # x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
	# x = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(x)
	
	x = denseLayer(x, n_filters, kernel_size=[1, 1], dropout_p=dropout_p)
	x = MaxPooling2D(pool_size=[2, 2], strides=[2, 2])(x)
	
	return x

def TU(x, skip_connection, n_filters):
	x = Conv2DTranspose(
		n_filters, kernel_size=(3, 3), padding='same', strides=(2, 2),
		# kernel_initializer='he_normal', # kernel_regularizer=l2(1e-4)
	)(x)
	x = Concatenate()([x, skip_connection])
	return x

def build(width, height, n_classes, n_filters_first_conv=48, dropout_p=0.2):
	# inputs, num_classes, n_filters_first_conv=48, 
	# n_pool=5, growth_rate=12, n_layers_per_block=4, dropout_p=0.2, scope=None
	input = Input(shape=(height, width, 3))
	
	n_pool=4
	growth_rate=6
	n_layers_per_block=3
	n_filters = n_filters_first_conv
	n_layers_per_block = [n_layers_per_block] * (2 * n_pool + 1)
	
	# first conv
	stack = Conv2D(
		n_filters_first_conv, kernel_size=(3, 3), padding='same', 
		name='first_conv', # kernel_initializer='he_normal',
	)(input)
	
	# downsampling
	skip_connection_list = []
	
	for i in range(n_pool):
		# denseBlock
		stack, _ = denseBlock(stack, n_layers_per_block[i], growth_rate, dropout_p)
		n_filters += growth_rate * n_layers_per_block[i]
		skip_connection_list.append(stack)
		
		# transition down
		stack = TD(stack, n_filters, dropout_p)
	
	skip_connection_list = skip_connection_list[::-1]
	
	# bottleneck
	stack, block_to_upsample = denseBlock(stack, n_layers_per_block[n_pool], growth_rate, dropout_p)
	
	# upsampling
	for i in range(n_pool):
		# transition up (upsampling + concatenation with the skip connection)
		n_filters_keep = growth_rate * n_layers_per_block[n_pool + i]
		stack = TU(block_to_upsample, skip_connection_list[i], n_filters_keep)
		
		# denseBlock
		stack, block_to_upsample = denseBlock(stack, n_layers_per_block[n_pool + i + 1], growth_rate, dropout_p)
		
	# softmax
	output = Conv2D(n_classes, kernel_size=(1, 1), activation='softmax')(stack)
	# output = Activation('softmax')(output)
	
	model = Model(inputs=input, outputs=output)
	
	return model

def load_image(path):
	image = cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
	return image

class DataGenerator(keras.utils.Sequence):
	# https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
	'Generates data for Keras'
	def __init__(
		self, folder, set_folder='train', batch_size=32, dim=(640,400), n_channels=3, n_classes=4, shuffle=True,
		augmentation={'h_flip': False, 'v_flip': False, 'brightness': None, 'rotation': None}
	):
		'Initialization'
		self.dim = dim
		self.batch_size = batch_size
		self.folder = folder
		self.images = self.get_files(set_folder)
		self.labels = self.get_files(set_folder+'_labels')
		self.n_channels = n_channels
		self.n_classes = n_classes
		self.shuffle = shuffle
		self.augmentation = augmentation
		self.on_epoch_end()
		
	def __len__(self):
		'Denotes the number of batches per epoch'
		return int(np.floor(len(self.images) / self.batch_size))

	def __getitem__(self, index):
		'Generate one batch of data'
		# Generate indexes of the batch
		indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

		# Generate data
		X, y = self.__data_generation(indexes)

		return X, y

	def on_epoch_end(self):
		'Updates indexes after each epoch'
		self.indexes = np.arange(self.__len__())
		if self.shuffle == True:
			np.random.shuffle(self.indexes)
			
	def get_files(self, type_folder):
		return sorted(glob.glob(os.path.join(self.folder, type_folder, '*')))
	
	def dataAugmentation(self, input_image, output_image):
		if self.augmentation['h_flip'] and random.randint(0,1):
			input_image = cv2.flip(input_image, 1)
			output_image = cv2.flip(output_image, 1)
		if self.augmentation['v_flip'] and random.randint(0,1):
			input_image = cv2.flip(input_image, 0)
			output_image = cv2.flip(output_image, 0)
		if self.augmentation['brightness']:
			factor = 1.0 + random.uniform(-1.0*self.augmentation['brightness'], self.augmentation['brightness'])
			table = np.array([((i / 255.0) * factor) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
			input_image = cv2.LUT(input_image, table)
		if self.augmentation['rotation']:
			angle = random.uniform(-1*self.augmentation['rotation'], self.augmentation['rotation'])
		if self.augmentation['rotation']:
			M = cv2.getRotationMatrix2D((input_image.shape[1]//2, input_image.shape[0]//2), angle, 1.0)
			input_image = cv2.warpAffine(input_image, M, (input_image.shape[1], input_image.shape[0]), flags=cv2.INTER_NEAREST)
			output_image = cv2.warpAffine(output_image, M, (output_image.shape[1], output_image.shape[0]), flags=cv2.INTER_NEAREST)

		return input_image, output_image
	
	def __data_generation(self, indexes):
		'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
		# Initialization
		X = np.empty((self.batch_size, *self.dim, self.n_channels), dtype=np.float32)
		y = np.empty((self.batch_size, *self.dim, self.n_classes), dtype=np.float32)
		# dividir y con to one hot de utils

		# Generate data
		for i, idx in enumerate(indexes):
			# Store sample
			# X[i,] = utils.load_image(self.images[idx])
			# lb = np.load(self.labels[idx])
			# for x in np.unique(lb):
			#     y[i,...,x] = np.where(lb == x, 1, 0)
			
			# load data
			im = load_image(self.images[idx])
			lb = np.load(self.labels[idx])
			im, lb = self.dataAugmentation(im, lb)
			im = np.float32(im) / 255.
			# lb = np.float32(lb)
			
			# assign data to batch
			X[i, ] = im
			for x in np.unique(lb):
				mask = np.where(lb == x, 1, 0)
				y[i, ..., x] = np.float32(mask)

		return X, y


if __name__ == "__main__":
	model = build(400, 640, 4, dropout_p=0.5) # dropout_p=0.2
	batch_size = 1
	# model.summary()
	# model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

	aug = {'h_flip': True, 'v_flip': True, 'brightness': .25, 'rotation': 25}
	trainG = DataGenerator('openeds_split', 'train', batch_size, augmentation=aug)
	# testG = DataGenerator('openeds_split', 'test', batch_size)
	valG = DataGenerator('openeds_split', 'val', batch_size)

	# loss function
	# loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=network, labels=net_output))
	def rmean_scwl_keras(target, output):
		# return keras.backend.mean(keras.backend.softmax(keras.backend.categorical_crossentropy(target, output, from_logits=True)))
		return keras.backend.mean(keras.backend.categorical_crossentropy(target, output, from_logits=True))

	def rmean_scwl_tf(target, output):
		return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=target))

	# opt = keras.optimizers.SGD(lr=1e-4, momentum=0.9, nesterov=True, decay=1e-4)
	opt = keras.optimizers.RMSprop(lr=0.0001, decay=0.995)
	model.compile(opt, loss='categorical_crossentropy', metrics=['accuracy', ])
	# model.compile(opt, loss=rmean_scwl_tf, metrics=['accuracy', ])

	mckpt = ModelCheckpoint('./models/densenet_epoch_{epoch:04d}.hdf5', monitor='val_loss', save_best_only=True, verbose=1)
	tensorboard = TensorBoard(log_dir='./logs')

	H = model.fit_generator(
		generator=trainG, validation_data=valG, 
		epochs=100, callbacks=[mckpt, tensorboard, ], 
		steps_per_epoch=len(trainG) // batch_size,
	)
	# H = model.fit_generator(generator=trainG, epochs=5, steps_per_epoch=10)

	fig, ax = plt.subplots()
	ax.plot(range(len(H.history['val_acc'])), H.history['val_acc'], label='val_acc')
	ax.plot(range(len(H.history['val_loss'])), H.history['val_loss'], label='val_loss')
	# ax.plot(range(len(H.history['acc'])), H.history['acc'], label='acc')
	# ax.plot(range(len(H.history['loss'])), H.history['loss'], label='loss')
	plt.legend(loc='upper left', borderaxespad=0.)

	ax.set(xlabel='Epochs', ylabel='val', title='Epochs vs val_loss')
	ax.grid()
	fig.savefig("performance.png")