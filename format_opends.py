import os
import shutil
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset',
					help='Dir to opends_camvid dataset', required=True)
parser.add_argument('--out_dir', help='Dir to output', default='opends_format')
parser.add_argument('--bool', help='Minimum number of images', type=str, default='all')
parser.add_argument(
	'--test', help='Size of test folder', type=float, default=.2)
parser.add_argument(
	'--train_size', help='Number of train images', type=int, default=421)
parser.add_argument(
	'--test_size', help='Number of test images', type=int, default=168)
parser.add_argument(
	'--val_size', help='Number of val images', type=int, default=112)
args = parser.parse_args()

images_dir = os.path.join(args.dataset, 'images')
labels_dir = os.path.join(args.dataset, 'labels')

images_names = sorted(os.listdir(images_dir))
labels_names = sorted(os.listdir(labels_dir))

images = list(map(lambda x: os.path.join(images_dir, x), images_names))
labels = list(map(lambda x: os.path.join(labels_dir, x), labels_names))


if args.bool == 'all':
	train_size = 1 - (args.test + args.test) # 60%
	train_size = int( len(images) * train_size )
	train_images = images[:train_size]
	train_labels = labels[:train_size]

	test_size_end = train_size + int( len(images) * args.test ) # 60% + 20%
	test_images = images[train_size:test_size_end]
	test_labels = labels[train_size:test_size_end]

	# val_size_begin = train_size + test_size_end # remain
	val_images = images[test_size_end:]
	val_labels = labels[test_size_end:]

else:
	train_images = images[:args.train_size]
	train_labels = labels[:args.train_size]

	test_size_end = args.train_size + args.test_size
	test_images = images[args.train_size:test_size_end]
	test_labels = labels[args.train_size:test_size_end]

	val_size_begin = args.train_size + args.test_size
	val_size_end = args.train_size + args.test_size + args.val_size
	val_images = images[val_size_begin:val_size_end]
	val_labels = labels[val_size_begin:val_size_end]

if os.path.exists(args.out_dir):
	shutil.rmtree(args.out_dir)

os.makedirs(os.path.join(args.out_dir, 'train'))
os.makedirs(os.path.join(args.out_dir, 'train_labels'))
os.makedirs(os.path.join(args.out_dir, 'test'))
os.makedirs(os.path.join(args.out_dir, 'test_labels'))
os.makedirs(os.path.join(args.out_dir, 'val'))
os.makedirs(os.path.join(args.out_dir, 'val_labels'))
shutil.copyfile(os.path.join(args.dataset, 'class_dict.csv'),
				os.path.join(args.out_dir, 'class_dict.csv'))

for im, lb in zip(train_images, train_labels):
	im_name = im.split('/')[-1]
	lb_name = lb.split('/')[-1]
	im_dst = os.path.join(args.out_dir, 'train', im_name)
	lb_dst = os.path.join(args.out_dir, 'train_labels', lb_name)
	shutil.copyfile(im, im_dst)
	shutil.copyfile(lb, lb_dst)

for im, lb in zip(test_images, test_labels):
	im_name = im.split('/')[-1]
	lb_name = lb.split('/')[-1]
	im_dst = os.path.join(args.out_dir, 'test', im_name)
	lb_dst = os.path.join(args.out_dir, 'test_labels', lb_name)
	shutil.copyfile(im, im_dst)
	shutil.copyfile(lb, lb_dst)

for im, lb in zip(val_images, val_labels):
	im_name = im.split('/')[-1]
	lb_name = lb.split('/')[-1]
	im_dst = os.path.join(args.out_dir, 'val', im_name)
	lb_dst = os.path.join(args.out_dir, 'val_labels', lb_name)
	shutil.copyfile(im, im_dst)
	shutil.copyfile(lb, lb_dst)

