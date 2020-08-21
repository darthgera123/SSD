import tensorflow as tf
import zipfile, tempfile, os, glob, tqdm
import numpy as np
import cv2
# from utils import get_random_data

def draw_boxes_on_image_v2(image, boxes):
    image = image.astype('uint8')
    # num_boxes = boxes.shape[0]
    for x1,x2,y1,y2,l in boxes:
        if x1==y1==x2==y2==-1:
            break

        class_and_score = f"label :{l}"
        cv2.rectangle(img=image, pt1=(int(x1), int(y1)), pt2=(int(x2), int(y2)), color=(255, 0, 0), thickness=2)
        cv2.putText(img=image, text=class_and_score, org=(int(x1), int(y1) - 10), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5, color=(0, 255, 255), thickness=1)
    return image


class SSD300Config(object):
    """docstring for SSDConfig"""
    def __init__(self, 
        aspect_ratios=([1.0, 2.0, 0.5],
                 [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                 [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                 [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                 [1.0, 2.0, 0.5],
                 [1.0, 2.0, 0.5]), 
        max_boxes=300, 
        # width=416,
        score=0.3,
        iou=0.6):

        self.height = 300
        self.width = 300

        self.input_shape = (self.height, self.width)

        self.iou = iou
        self.max_boxes_per_image = max_boxes

        self.aspect_ratios = aspect_ratios


def download_aerial_dataset(dataset_path=tempfile.gettempdir()):
    zip_url = 'https://segmind-data.s3.ap-south-1.amazonaws.com/edge/data/aerial-vehicles-dataset.zip'
    path_to_zip_file = tf.keras.utils.get_file(
        'aerial-vehicles-dataset.zip',
        zip_url,
        cache_dir=dataset_path, 
        cache_subdir='',
        extract=False)
    directory_to_extract_to = os.path.join(dataset_path,'aerial-vehicles-dataset')
    with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
        zip_ref.extractall(directory_to_extract_to)

    images_dir = os.path.join(dataset_path, 'aerial-vehicles-dataset','images')
    annotation_dir = os.path.join(dataset_path, 'aerial-vehicles-dataset','annotations','pascalvoc_xml')

    return images_dir, annotation_dir


def download_chess_dataset(dataset_path=tempfile.gettempdir()):
    zip_url = 'https://public.roboflow.ai/ds/uBYkFHtqpy?key=HZljsh2sXY'
    path_to_zip_file = tf.keras.utils.get_file(
        'chess_pieces.zip',
        zip_url,
        cache_dir=dataset_path, 
        cache_subdir='',
        extract=False)
    directory_to_extract_to = os.path.join(dataset_path)
    with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
        zip_ref.extractall(directory_to_extract_to)

    images_dir = os.path.join(dataset_path, 'chess_pieces','train')
    annotation_dir = os.path.join(dataset_path, 'chess_pieces','train')

    for image in tqdm.tqdm(glob.glob(os.path.join(images_dir, '*.jpg'))):
        new_name = image.replace('_jpg.rf.', '')
        os.rename(image, new_name)

        annotation = image.replace('.jpg', '.xml')
        new_name = annotation.replace('_jpg.rf.', '')
        os.rename(annotation, new_name)

    return images_dir, annotation_dir

