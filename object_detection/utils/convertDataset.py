import tensorflow as tf
import dataset_util
import numpy as np
import gzip
from tqdm import tqdm
import os
import glob
import re

classesid = {'banana': 1, 'bottle': 2, 'remote': 3, 'keyboard': 4, 'cell_phone': 5}


def create_tf_example(example):
    (obj_type, fileidx, annotations, data, sh) = example
    # TODO(user): Populate the following variables from your example.
    height = 240  # Image height
    width = 304  # Image width
    filename = str.encode(fileidx + 'npy.gz')  # Filename of the image. Empty if image is not from file
    encoded_image_data = data  # Encoded image bytes
    indices = np.int64(np.random.random((1000, 3))*[304,240,2])
    indices0, indices1, indices2 = indices.T
    indices = indices.flatten().tolist()
    # indices = np.int64(np.random.random((1, 3))*[304, 240, 3])
    # indices = indices.tobytes()
    
    sh = np.array([304, 240, 2]).astype(np.int64)
    
    values = (np.random.random(len(indices)) * 256).astype(np.float32).flatten().tolist()
    # values = values.tobytes()
    
    image_format = b'vEvent'  # b'jpeg' or b'png'
    
    xmins = [annotations[2]]  # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs = [annotations[4]]  # List of normalized right x coordinates in bounding box
    # (1 per box)
    ymins = [annotations[1]]  # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs = [annotations[3]]  # List of normalized bottom y coordinates in bounding box
    # (1 per box)
    classes_text = [str.encode(obj_type)]  # List of string class name of bounding box (1 per box)
    classes = [classesid[obj_type]]  # List of integer class id of bounding box (1 per box)
    
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/indices'           : dataset_util.int64_list_feature(indices),
        'image/indices0'           : dataset_util.int64_list_feature(indices0),
        'image/indices1'           : dataset_util.int64_list_feature(indices1),
        'image/indices2'           : dataset_util.int64_list_feature(indices2),
        'image/values'            : dataset_util.float_list_feature(values),
        'image/shape'             : dataset_util.int64_list_feature(sh),
        'image/height'            : dataset_util.int64_feature(height),
        'image/width'             : dataset_util.int64_feature(width),
        'image/filename'          : dataset_util.bytes_feature(filename),
        'image/source_id'         : dataset_util.bytes_feature(filename),
        'image/encoded'           : dataset_util.bytes_feature(encoded_image_data),
        'image/format'            : dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin'  : dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax'  : dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin'  : dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax'  : dataset_util.float_list_feature(ymaxs),
        'image/object/class/text' : dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def main(_):
    output_path = '/home/miacono/workspace/models/object_detection/data/train.records'
    writer = tf.python_io.TFRecordWriter(output_path)
    
    path = '/home/miacono/workspace/vObj_detection/data'
    files = glob.glob(path + '/splits/*/left/*.npy.gz')
    pattern = re.compile('.*/(.*)/left/(.*).npy.gz')
    boxpattern = re.compile('.*/boxes/(.*)/boxes.npy.gz')
    boxfiles = glob.glob(path + '/boxes/*/*.npy.gz')
    boxes_dict = {}
    for file in boxfiles:
        boxes = np.load(gzip.open(file, 'rb'))
        match = boxpattern.match(file)
        (obj_type,) = match.groups()
        boxes_dict[obj_type] = boxes
    for file in tqdm(files):
        match = pattern.match(file)
        obj_type, fileidx = match.groups()
        annotations = boxes_dict[obj_type][int(fileidx) - 1]
        with gzip.open(file, 'rb') as f:
            array = np.load(f)
            sh = array.shape
            data = array.tobytes()
        example = (obj_type, fileidx, annotations, data, sh)
        tf_example = create_tf_example(example)
        writer.write(tf_example.SerializeToString())
    
    writer.close()


if __name__ == '__main__':
    tf.app.run()
