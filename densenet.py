import os
import keras
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint, TensorBoard

from datagenerator import DataGenerator
from model import Tiramisu

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
allow_growth_session = tf.Session(config=config)
tf.keras.backend.set_session(allow_growth_session)

if __name__ == "__main__":
	if not os.path.exists('models') and not os.path.exists('logs'):
		os.path.mkdir('models')
		os.path.mkdir('logs')
	
	batch_size = 1
	model = Tiramisu(
		input_shape=(640, 400, 3), n_classes=4, n_filters_first_conv=48,
		n_pool=4, growth_rate=6, n_layers_per_block=3, dropout_p=0.2
	)
	print('Model params: %d' % model.count_params())
	
	aug = {'h_flip': True, 'v_flip': True, 'brightness': .25, 'rotation': 25}
	trainG = DataGenerator('openeds_split', 'train', batch_size, augmentation=aug)
	# testG = DataGenerator('openeds_split', 'test', batch_size)
	valG = DataGenerator('openeds_split', 'val', batch_size)

	# optimizer
	opt = keras.optimizers.SGD(lr=1e-4, momentum=0.9, nesterov=True, decay=1e-4)
	# opt = keras.optimizers.RMSprop(lr=0.0001, decay=0.995) 
	# opt = keras.optimizers.Adadelta(lr=1.0, rho=0.95)

	# model compile with loss function
	# model.compile(opt, loss='categorical_crossentropy', metrics=['accuracy', ])
	# model.compile(opt, loss=rmean_scwl_tf, metrics=['accuracy', ])
	model.compile(opt, loss='kullback_leibler_divergence', metrics=['accuracy', ])
	# model.compile(opt, loss='kullback_leibler_divergence', metrics=['accuracy', mean_iou(4)]) # no funca xd
	"""
	https://machinelearningmastery.com/how-to-choose-loss-functions-when-training-deep-learning-neural-networks/
	
	kullback_leibler_divergence used when using models that learn to approximate a more 
	complex function than simply multi-class classification, 
	such as in the case of an autoencoder used for learning 
	a dense feature representation under a model that must reconstruct the original input.
	"""

	mckpt = ModelCheckpoint('./models/densenet_epoch_{epoch:04d}.hdf5', monitor='val_loss', save_best_only=True, verbose=1)
	tensorboard = TensorBoard(log_dir='./logs')

	H = model.fit_generator(
		generator=trainG, validation_data=valG, 
		epochs=100, callbacks=[mckpt, tensorboard, ], 
		steps_per_epoch=np.ceil(len(trainG) // batch_size),
		validation_steps=np.floor(len(valG) // batch_size)
	)
	
	# H = model.fit_generator(
	# 	generator=trainG, validation_data=valG, 
	# 	epochs=10, callbacks=[mckpt, tensorboard, ], 
	# 	steps_per_epoch=100, validation_steps=100
	# )

	fig, ax = plt.subplots()
	ax.plot(range(len(H.history['val_acc'])), H.history['val_acc'], label='val_acc')
	ax.plot(range(len(H.history['val_loss'])), H.history['val_loss'], label='val_loss')
	# ax.plot(range(len(H.history['acc'])), H.history['acc'], label='acc')
	# ax.plot(range(len(H.history['loss'])), H.history['loss'], label='loss')
	plt.legend(loc='upper left', borderaxespad=0.)

	ax.set(xlabel='Epochs', ylabel='val', title='Epochs vs val_loss')
	ax.grid()
	fig.savefig("performance.png")