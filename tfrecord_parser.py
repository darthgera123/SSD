import tensorflow as tf
import cv2, os, glob
import numpy as np

# we will read the tfrecords
# preprocess and do augmentation
# return the dataloader
# return image, bbox, labels
def _pad(image, height, width):
    """Summary
    
    Args:
        image (TYPE): Description
        height (TYPE): Description
        width (TYPE): Description
        scale (TYPE): Description
    
    Returns:
        numpy nd.array: Description
    """

    image = image.astype(np.uint8)
    padded_image = np.zeros(shape=(height.astype(int), width.astype(int),3), dtype=np.uint8)
    h,w,_ =  image.shape
    padded_image[:h,:w,:] = image
    return padded_image

@tf.function
def decode_pad(image_string, pad_height, pad_width):
  """Summary
  
  Args:
      image_string (TYPE): Description
      pad_height (TYPE): Description
      pad_width (TYPE): Description
      scale (TYPE): Description
  
  Returns:
      tf.tensor: Description
  """
  image = tf.image.decode_jpeg(image_string)
  image = tf.numpy_function(_pad, [image, pad_height, pad_width], Tout=tf.uint8)
  #image.set_shape([None, None, 3])
  return image


def process_bbox(xmin_batch, ymin_batch, xmax_batch, ymax_batch, label_batch,batch_size=2):
    regression_batch = list()
    classification_batch = list()

    for index in range(batch_size):
        xmins, ymins, xmaxs, ymaxs, labels = xmin_batch[index], ymin_batch[index], xmax_batch[index], ymax_batch[index], label_batch[index]
        bboxes = tf.convert_to_tensor([xmins,ymins,xmaxs,ymaxs], dtype=tf.keras.backend.floatx())
        bboxes = tf.transpose(bboxes)
        

        regression_batch.append(bboxes)
        classification_batch.append(labels)

    return tf.convert_to_tensor(regression_batch), tf.convert_to_tensor(classification_batch)

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

    max_height = tf.cast(tf.keras.backend.max(parsed_example['image/height']), tf.int32)
    max_width = tf.cast(tf.keras.backend.max(parsed_example['image/width']), tf.int32)

    image_batch = tf.map_fn(lambda x: decode_pad(x, max_height, max_width), parsed_example['image/encoded'], dtype=tf.uint8)
    
    xmin_batch = tf.sparse.to_dense(parsed_example['image/object/bbox/xmin'],default_value=-1)
    xmax_batch = tf.sparse.to_dense(parsed_example['image/object/bbox/xmax'],default_value=-1)
    ymin_batch = tf.sparse.to_dense(parsed_example['image/object/bbox/ymin'],default_value=-1)
    ymax_batch = tf.sparse.to_dense(parsed_example['image/object/bbox/ymax'],default_value=-1)

    label_batch = tf.sparse.to_dense(parsed_example['image/object/class/label'], default_value=-1)

    regression_batch,classification_batch = process_bbox(xmin_batch,xmax_batch,
                                                        ymin_batch,ymax_batch,label_batch)

    return image_batch, {'regression':regression_batch, 'classification':classification_batch}


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

    for data, annotation in dataset.take(1):
        image_batch = data.numpy()
        abxs_batch = annotation['regression'].numpy()
        labels_batch = annotation['classification'].numpy()

        print(image_batch.shape, abxs_batch.shape, labels_batch.shape)
        print(image_batch.dtype, abxs_batch.dtype, labels_batch.dtype)