import tensorflow as tf
from tensorflow import keras
import cv2, os, glob
import numpy as np

from helpers import SSD300Config

config = SSD300Config()


def pad_resize(image, height, width, resize_width, resize_height):
    """Summary

    Args:
        image (TYPE): Description
        height (TYPE): Description
        width (TYPE): Description
        scale (TYPE): Description

    Returns:
        numpy nd.array: Description
    """
    padded_image = np.zeros(shape=(height.astype(int), width.astype(int),3), dtype=image.dtype)
    h,w,_ =  image.shape
    padded_image[:h,:w,:] = image
    resized_image = cv2.resize(padded_image, (resize_width, resize_height)).astype(keras.backend.floatx())
    return resized_image


@tf.function
def decode_pad_resize(image_string, pad_height, pad_width, resize_width, resize_height):
    """Summary

    Args:
      image_string (TYPE): Description
      pad_height (TYPE): Description
      pad_width (TYPE): Description
      esize_width, resize_height (TYPE): Description

    Returns:
      tf.tensor: Description
    """
    image = tf.image.decode_jpeg(image_string)
    image = tf.numpy_function(pad_resize, [image, pad_height, pad_width, resize_width, resize_height], Tout=keras.backend.floatx())
    #image.set_shape([None, None, 3])
    return image




def make_gt(bboxes, max_box_per_image, height_scale, width_scale):
    """Summary
    Create a bbox batch of the form [xmin,ymin, xmax,ymax,labels]
    """

    # delete bboxes containing [-1,-1,-1,-1, -1] added in **[1]
    bboxes = bboxes[~np.all(bboxes==-1, axis=1)]

    bboxes[:,0] *= width_scale
    bboxes[:,1] *= width_scale
    bboxes[:,2] *= height_scale
    bboxes[:,3] *= height_scale

    num_boxes, boxes_per_row = bboxes.shape

    assert boxes_per_row == 5

    arr = np.zeros((max_box_per_image,5), keras.backend.floatx())
    arr[:,-1] -= 1

    max_index = min(num_boxes, max_box_per_image)

    arr[:max_index, :5] = bboxes[:max_index, :5]

    return arr #.astype(keras.backend.floatx())

@tf.function
def tf_make_gt(xmin_batch, ymin_batch, xmax_batch, ymax_batch, label_batch, batch_size, max_box_per_image, height_scale, width_scale):
    
    annotation_batch = list()

    for index in range(batch_size):

        xmins, ymins, xmaxs, ymaxs, labels = xmin_batch[index], ymin_batch[index], xmax_batch[index], ymax_batch[index], label_batch[index]

        labels = tf.cast(labels, keras.backend.floatx())

        bboxes = tf.convert_to_tensor([xmins,ymins,xmaxs,ymaxs,labels], dtype=keras.backend.floatx())
        bboxes = tf.transpose(bboxes)

        bboxes = tf.numpy_function(make_gt, [bboxes, max_box_per_image, height_scale, width_scale], Tout=keras.backend.floatx())

        annotation_batch.append(bboxes)

    return tf.convert_to_tensor(annotation_batch)


class Tfrpaser(object):
    """docstring for Tfrpaser"""
    def __init__(self, batch_size):

        self.batch_size = batch_size
        

    def _parse_fn(self, serialized):
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
              'image/object/bbox/xmin': tf.io.VarLenFeature(keras.backend.floatx()),
              'image/object/bbox/xmax': tf.io.VarLenFeature(keras.backend.floatx()),
              'image/object/bbox/ymin': tf.io.VarLenFeature(keras.backend.floatx()),
              'image/object/bbox/ymax': tf.io.VarLenFeature(keras.backend.floatx()),
              'image/f_id': tf.io.FixedLenFeature([], tf.int64),
              'image/object/class/label':tf.io.VarLenFeature(tf.int64)}


        parsed_example = tf.io.parse_example(serialized=serialized, features=features)

        max_height = tf.cast(tf.keras.backend.max(parsed_example['image/height']), tf.int32)
        max_width = tf.cast(tf.keras.backend.max(parsed_example['image/width']), tf.int32)

        height_scale = config.width/max_height
        width_scale = config.width/max_width


        image_batch = tf.map_fn(lambda x: decode_pad_resize(x, max_height, max_width, config.width, config.width), parsed_example['image/encoded'], dtype=keras.backend.floatx())

        # **[1] pad with -1 to batch properly
        xmin_batch = tf.sparse.to_dense(parsed_example['image/object/bbox/xmin'], default_value=-1)
        xmax_batch = tf.sparse.to_dense(parsed_example['image/object/bbox/xmax'], default_value=-1)
        ymin_batch = tf.sparse.to_dense(parsed_example['image/object/bbox/ymin'], default_value=-1)
        ymax_batch = tf.sparse.to_dense(parsed_example['image/object/bbox/ymax'], default_value=-1)
        label_batch = tf.sparse.to_dense(parsed_example['image/object/class/label'], default_value=-1)

        annotation_batch = tf_make_gt(
            xmin_batch, 
            xmax_batch, 
            ymin_batch, 
            ymax_batch, 
            label_batch, 
            self.batch_size,
            config.max_boxes_per_image,
            height_scale, 
            width_scale)

        return image_batch/255, annotation_batch


    def parse_tfrecords(self, filename):

        dataset = tf.data.Dataset.list_files(filename).shuffle(buffer_size=8).repeat(-1)
        dataset = dataset.interleave(
                    tf.data.TFRecordDataset,
                    num_parallel_calls = tf.data.experimental.AUTOTUNE,
                    deterministic=False)

        dataset = dataset.batch(
                    self.batch_size,
                    drop_remainder=True)
        
        dataset = dataset.map(
                    self._parse_fn,
                    num_parallel_calls=tf.data.experimental.AUTOTUNE)

        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        return dataset


if __name__ == '__main__':

    from helpers import draw_boxes_on_image_v2

    parser = Tfrpaser(batch_size=2)
    
    dataset = parser.parse_tfrecords(filename=os.path.join(os.getcwd(),'DATA','train*.tfrecord'))

    for data, annotation in dataset.take(1):
        image_batch = data.numpy()
        abxs_batch = annotation.numpy()
        # print(image_batch.shape)
        # print(abxs_batch.shape)
        # # print(image_batch)
        # print(abxs_batch)

        for index in range(parser.batch_size):
            im = draw_boxes_on_image_v2(image_batch[index]*255, abxs_batch[index])
            cv2.imwrite(f"{index}.jpg", im)
        