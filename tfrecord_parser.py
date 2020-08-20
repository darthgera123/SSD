import tensorflow as tf
import cv2, os, glob
import numpy as np

IMAGE_WIDTH=300
IMAGE_HEIGHT=300
MAX_BOX_PER_IMAGE = 30
# Read the tfrecords
# preprocess and do augmentation
# return the dataloader
# return image, [bbox, labels]
def __resize(image):
    """Summary
    
    Args:
        image (TYPE): Description
        
    Returns:
        numpy nd.array: Description
    """

    image = image.astype(np.uint8)
    resized_image = cv2.resize(image,(IMAGE_HEIGHT,IMAGE_WIDTH))
    resized_image = resized_image/255
    # any other augmentation tricks and techniques should be added here.
    return resized_image.astype('float32')

@tf.function
def decode_resize(image_string):
  """Summary
  
  Args:
      image_string (TYPE): Description
      
  
  Returns:
      tf.tensor: Description
  """
  
  image = tf.image.decode_jpeg(image_string)
  # image = tf.image.resize(image, [IMAGE_HEIGHT, IMAGE_WIDTH])
  image = tf.numpy_function(__resize, [image], Tout=tf.keras.backend.floatx())
  #image.set_shape([None, None, 3])
  return image

def resize_boxes(xmins,xmaxs,ymins,ymaxs,h,w):
  """Summary
  
  Args:
      xmins: min x coordinate of bbox
      xmaxs: max x coordinate of bbox
      ymins: min y coordinate of bbox
      ymaxs: max y coordinate of bbox
      h: height of image
      w: width of image
      
  
  Returns:
      xmin,tmin,xmax,ymax (resized versions)

  """
  resize_ratio = [IMAGE_HEIGHT / h, IMAGE_WIDTH / w]
  xmin = (resize_ratio[1] * xmins).astype('uint32')
  xmax = (resize_ratio[1] * xmaxs).astype('uint32')
  ymin = (resize_ratio[0] * ymins).astype('uint32')
  ymax = (resize_ratio[0] * ymaxs).astype('uint32')
  return xmin.astype('float32'), ymin.astype('float32'), xmax.astype('float32'), ymax.astype('float32')

def create_tensor(xmins,xmaxs,ymins,ymaxs,labels):
  """Summary
  Create a bbox batch of the form [xmin,ymin, xmax,ymax,labels]
  """
  new_tensor = list(zip(xmins,ymins,xmaxs,ymaxs,labels))
  remainder = MAX_BOX_PER_IMAGE-len(new_tensor)
  if remainder > 0:
    for i in range(remainder):
      new_tensor.append([0,0,0,0,-1])
  else:
    new_tensor = new_tensor[:MAX_BOX_PER_IMAGE]
  return np.asarray(new_tensor).astype('float32')

def process_bbox(xmin_batch, ymin_batch, xmax_batch, ymax_batch, label_batch, heights, widths, batch_size):
    
    regression_batch = list()

    for index in range(batch_size):

        xmins, ymins, xmaxs, ymaxs, labels = xmin_batch[index], ymin_batch[index], xmax_batch[index], ymax_batch[index], label_batch[index]
        height,width = heights[index], widths[index]
        xmin,ymin,xmax,ymax = tf.numpy_function(resize_boxes, [xmins, xmaxs, ymins, ymaxs, height, width], 
          Tout=[tf.keras.backend.floatx(), tf.keras.backend.floatx(), tf.keras.backend.floatx(), tf.keras.backend.floatx()])
        bbox = tf.numpy_function(create_tensor,[xmin,xmax,ymin,ymax,labels],Tout=tf.keras.backend.floatx())
        # bboxes = tf.convert_to_tensor([xmin,ymin,xmax,ymax], dtype=tf.keras.backend.floatx())
        bboxes = tf.convert_to_tensor(bbox, dtype=tf.keras.backend.floatx())
        # bboxes = tf.transpose(bboxes)
        

        regression_batch.append(bboxes)

    return tf.convert_to_tensor(regression_batch)


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
              'image/object/bbox/xmin': tf.io.VarLenFeature(tf.keras.backend.floatx()),
              'image/object/bbox/xmax': tf.io.VarLenFeature(tf.keras.backend.floatx()),
              'image/object/bbox/ymin': tf.io.VarLenFeature(tf.keras.backend.floatx()),
              'image/object/bbox/ymax': tf.io.VarLenFeature(tf.keras.backend.floatx()),
              'image/f_id': tf.io.FixedLenFeature([], tf.int64),
              'image/object/class/label':tf.io.VarLenFeature(tf.int64)}


        parsed_example = tf.io.parse_example(serialized=serialized, features=features)

        # max_height = tf.cast(tf.keras.backend.max(parsed_example['image/height']), tf.int32)
        # max_width = tf.cast(tf.keras.backend.max(parsed_example['image/width']), tf.int32)
        heights = tf.cast(parsed_example['image/height'],tf.int32)
        widths = tf.cast(parsed_example['image/width'],tf.int32)
        image_batch = tf.map_fn(lambda x: decode_resize(x), parsed_example['image/encoded'], dtype=tf.keras.backend.floatx())
        
        xmin_batch = tf.sparse.to_dense(parsed_example['image/object/bbox/xmin'],default_value=0)
        xmax_batch = tf.sparse.to_dense(parsed_example['image/object/bbox/xmax'],default_value=0)
        ymin_batch = tf.sparse.to_dense(parsed_example['image/object/bbox/ymin'],default_value=0)
        ymax_batch = tf.sparse.to_dense(parsed_example['image/object/bbox/ymax'],default_value=0)

        label_batch = tf.sparse.to_dense(parsed_example['image/object/class/label'], default_value=-1)

        regression_batch = process_bbox(xmin_batch, xmax_batch, 
          ymin_batch, ymax_batch, label_batch, heights, widths, self.batch_size)

        return image_batch, regression_batch


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

    parser = Tfrpaser(batch_size=2)
    
    dataset = parser.parse_tfrecords(filename=os.path.join(os.getcwd(),'DATA','train*.tfrecord'))

    for data, annotation in dataset.take(1):
        image_batch = data.numpy()
        abxs_batch = annotation.numpy()
        print(image_batch)
        print(abxs_batch)
        