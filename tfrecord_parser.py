import tensorflow as tf
import cv2, os, glob
import numpy as np

# we will read the tfrecords
# preprocess and do augmentation
# return the dataloader
# return image, bbox, labels
def read_image(img_string):
	# Decode the string as an RGB JPEG.
    # Note that the resulting image contains an unknown height
    # and width that is set dynamically by decode_jpeg. In other
    # words, the height and width of image is unknown at compile-i
    # time.
	image = tf.image.decode_jpeg(img_string)
	# now here we can do all sorts of preprocessing and augmentations
	# image = preprocess(image)
	image = tf.image.convert_image_dtype(image, dtype=tf.float32)
	return image

def _parse_fn(serialized):
	"""Summary
        
        Args:
            serialized (TYPE): Description
        
        Returns:
            TYPE: Description
    """
	features = {
          'image/height': tf.io.FixedLenFeature([], tf.int64),
          'image/width': tf.io.FixedLenFeature([], tf.int64),
          'image/encoded': tf.io.FixedLenFeature([],tf.string),
          'image/object/bbox/xmin': tf.io.VarLenFeature(tf.keras.backend.floatx()),
          'image/object/bbox/xmax': tf.io.VarLenFeature(tf.keras.backend.floatx()),
          'image/object/bbox/ymin': tf.io.VarLenFeature(tf.keras.backend.floatx()),
          'image/object/bbox/ymax': tf.io.VarLenFeature(tf.keras.backend.floatx()),
          'image/f_id': tf.io.FixedLenFeature([], tf.int64),
          'image/object/class/label':tf.io.VarLenFeature(tf.int64)}


	parsed_example = tf.io.parse_example(serialized=serialized, features=features)
	image_batch = read_image(parsed_example['image/encoded'])

	xmin_batch = tf.sparse.to_dense(parsed_example['image/object/bbox/xmin'],default_value=-1)
	xmax_batch = tf.sparse.to_dense(parsed_example['image/object/bbox/xmax'],default_value=-1)
	ymin_batch = tf.sparse.to_dense(parsed_example['image/object/bbox/ymin'],default_value=-1)
	ymax_batch = tf.sparse.to_dense(parsed_example['image/object/bbox/ymax'],default_value=-1)

	label_batch = tf.sparse(parsed_example['image/object/class/label'], default_value=-1)

	bbox_batch = {
			'xmin': xmin_batch,
			'xmax': xmax_batch,
			'ymin': ymin_batch,
			'ymax': ymax_batch
    }

	return image_batch, bbox_batch, label_batch


def parse_tfrecords(filename, batch_size):

	dataset = tf.data.Dataset.list_files(filename).shuffle(buffer_size=256).repeat(-1)
	dataset = dataset.interleave(
				tf.data.TFRecordDataset,
				num_parallel_calls = tf.data.experimental.AUTOTUNE,
				deterministic=False)

	dataset = dataset.batch(
				batch_size,
				drop_remainder=True)
	
	dataset = dataset.map(
				_parse_fn,
				num_parallel_calls=tf.data.experimental.AUTOTUNE)

	dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

	return dataset


if __name__ == '__main__':
	
	dataset = parse_tfrecords(
        filename=os.path.join(os.getcwd(),'DATA','train*.tfrecord'), 
        batch_size=2)

	for data, bbox, label in data.take(5):
		image_batch = data.numpy()
		bbox_batch = bbox.numpy()
		label_batch = label.numpy()
		print(image_batch.shape,label_batch)