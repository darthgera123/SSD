import tensorflow as tf
from PIL import Image
import json, os, glob, random, io, tqdm
import math, sys
from pprint import pprint
import xml.etree.ElementTree as ET

def int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_list_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def findLabels(xml_dir):

  unique_labels = ['background']

  #print(os.path.join(xml_dir,'*.xml'))

  for xml in glob.glob(os.path.join(xml_dir,'*.xml')):
    #print(xml)
    tree = ET.parse(xml)
    root = tree.getroot()

    #labels = []

    for obj in root.findall('object'):
      label = obj.find('name').text
      #print(label)

      if label not in unique_labels:
        unique_labels.append(label)

  #print(unique_labels)
  label_dict = {}
  for i in unique_labels:
    label_dict[i]=unique_labels.index(i)
  return label_dict

def negative_data(filename):

    name = filename.split('/')[-1]
    assert filename.endswith('.jpg')
    with tf.gfile.GFile(filename, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = filename.split('/')[-1]

    #labels_text = ['background']
    #labels = [0]
    xmin = []
    ymin = []
    xmax = []
    ymax = []
    labels_text = []
    labels = []

    return {
            'height' : height,
            'width' : width,
            'xmin' : xmin,
            'ymin' : ymin,
            'xmax' : xmax,
            'ymax' : ymax,
            'filename' : name,
            'labels_text' : labels_text,
            'labels' : labels,
            'image' : encoded_jpg
            }

def positive_data(xml_path, image_file, VOC_LABELS):

    filename = os.path.join(xml_path,image_file.split('/')[-1].split('.')[0]+'.xml')
    tree = ET.parse(filename)
    root = tree.getroot()

    # Image shape.

    with tf.io.gfile.GFile(image_file, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image_data = Image.open(encoded_jpg_io)
    width, height = image_data.size
    # Find annotations.
    xmin = []
    ymin = []
    xmax = []
    ymax = []
    labels = []
    labels_text = []

    for obj in root.findall('object'):
        label = obj.find('name').text
        labels.append(int(VOC_LABELS[label]))
        labels_text.append(label.encode('ascii'))

        bbox = obj.find('bndbox')
        xmin.append(float(bbox.find('xmin').text))
        ymin.append(float(bbox.find('ymin').text))
        xmax.append(float(bbox.find('xmax').text))
        ymax.append(float(bbox.find('ymax').text))

    name = root.find('filename').text
    filename = image_file
    assert filename.endswith('.jpg')

    #with tf.gfile.GFile(filename, 'rb') as fid:
    #  image_data = fid.read()    

    return {
            'height' : height,
            'width' : width,
            'xmin' : xmin,
            'ymin' : ymin,
            'xmax' : xmax,
            'ymax' : ymax,
            'filename' : name,
            'labels_text' : labels_text,
            'labels' : labels,
            'image' : encoded_jpg,
            }

def get_data(xml_dir, filename, positive, VOC_LABELS, fid):

    name = filename.split('/')[-1].split('.')[0]

    if positive:
        data_dict = positive_data(xml_dir, filename, VOC_LABELS)
    else :
        data_dict = negative_data(filename)

    data_dict['fid'] = fid

    return create_tf_example(data_dict)

def create_tf_example(data):
  """Creates a tf.Example proto from sample cat image.

  Args:
    data: A dictionary containing required data. The image
          byte is expected to be jpeg encoded

  Returns:
    example: The created tf.Example.
  """

  height = data['height']
  width = data['width']
  filename = data['filename']
  #assert filename.endswith('.jpg')
  filename = data['filename'].encode('utf-8')
  image_format = b'jpg'

  xmins = data['xmin']
  xmaxs = data['xmax']
  ymins = data['ymin']
  ymaxs = data['ymax']
  #classes_text = data['labels_text']
  classes = data['labels']

  #print([xmins, ymins, xmaxs, ymaxs])

  bytes_image_data = data['image']

  tf_example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': int64_feature(height),
      'image/width': int64_feature(width),
      'image/encoded': bytes_feature(bytes_image_data),
      'image/object/bbox/xmin': float_list_feature(xmins),
      'image/object/bbox/xmax': float_list_feature(xmaxs),
      'image/object/bbox/ymin': float_list_feature(ymins),
      'image/object/bbox/ymax': float_list_feature(ymaxs),
      'image/f_id': int64_feature(int(data['fid'])),
      'image/object/class/label': int64_list_feature(classes),
  }))
  return tf_example


def files_to_retain(files, xml_dir):

    retain_list = []
    reject_list = []
    
    for f in files:
        name = f.split('/')[-1].split('.')[0]
        if os.path.isfile(os.path.join(xml_dir,name+'.xml')):
         retain_list.append((f, True))
        else :
         reject_list.append((f, False)) 
    #pprint(reject_list)
    #pprint(retain_list)

    return retain_list+random.sample(reject_list,min(len(retain_list),len(reject_list)))


_NUM_SHARDS = 4

def create_tfrecords(image_dir, xml_dir, outpath=os.path.join(os.getcwd(),'DATA'), outname='train.tfrecord'):

  if not outname.endswith('.tfrecord'):
    raise ValueError("outname should endwith '.tfrecord', got name %s "%(outname))

  if not os.path.isdir(image_dir):
    raise ValueError('image directory doesnt exist')

  if not os.path.isdir(xml_dir):
    raise ValueError('annotation directory doesnt exist')

  image_names = files_to_retain(glob.glob(os.path.join(image_dir,'*.jpg')),xml_dir)

  os.makedirs(outpath, exist_ok=True)

  #if not os.path.isdir('./DATA'): os.mkdir('./DATA')
  #writer = tf.python_io.TFRecordWriter(output_path)

  VOC_LABELS = findLabels(xml_dir)
  print(VOC_LABELS)

  print('creating tfrecords ..')

  num_images = len(image_names)
  num_per_shard = int(math.ceil(num_images / _NUM_SHARDS))

  fid=1

  for shard_id in range(_NUM_SHARDS):
    output_filename = os.path.join(
        outpath,
        '%s-%05d-of-%05d.tfrecord' % ('train', shard_id, _NUM_SHARDS))

    with tf.io.TFRecordWriter(output_filename) as tfrecord_writer:
      start_idx = shard_id * num_per_shard
      end_idx = min((shard_id + 1) * num_per_shard, num_images)
      for i in range(start_idx, end_idx):
        sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
            i + 1, num_images, shard_id))
        sys.stdout.flush()
        # Read the image.
        image_file, positive = image_names[i]

        tf_example = get_data(xml_dir, image_file, positive, VOC_LABELS, fid)
        tfrecord_writer.write(tf_example.SerializeToString())
        fid += 1

  sys.stdout.write('\n')
  sys.stdout.flush()

  #writer.close()


if __name__ == '__main__':
  image_dir="../images"
  xml_dir="../annotations/pascalvoc_xml"
  create_tfrecords(image_dir, xml_dir, outname='aerial-vehicles-dataset.tfrecord')
