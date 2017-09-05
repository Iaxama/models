import tensorflow as tf
import glob
import re
from tqdm import tqdm
import numpy as np
import gzip

from object_detection.utils import dataset_util


flags = tf.app.flags
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
FLAGS = flags.FLAGS

classesid = {'banana': 52, 'bottle': 44, 'remote': 75, 'keyboard': 76, 'cell_phone': 77}

def create_tf_example(example):
    (obj_type, fileidx, annotations, data) = example
    height = 240 # Image height
    width = 304 # Image width
    filename = str.encode('img_%s.png' %fileidx) # Filename of the image. Empty if image is not from file
    encoded_image_data = data # Encoded image bytes
    image_format = b'png' # b'jpeg' or b'png'
    
    normalized = annotations[1:] / [height, width, height, width]
    xmins = [normalized[1]]   # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs = [normalized[3]] # List of normalized right x coordinates in bounding box
    # (1 per box)
    ymins = [normalized[0]]# List of normalized top y coordinates in bounding box (1 per box)
    ymaxs = [normalized[2]] # List of normalized bottom y coordinates in bounding box
    # (1 per box)
    classes_text = [str.encode(obj_type)] # List of string class name of bounding box (1 per box)
    classes = [classesid[obj_type]] # List of integer class id of bounding box (1 per box)
    
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_image_data),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def main(_):

    output_path = '/home/miacono/workspace/models/object_detection/data/train.records'

    writer = tf.python_io.TFRecordWriter(output_path)
    
    path = '/home/miacono/workspace/vObj_detection/data/'
    pattern = re.compile('.*/frames/\d*ms/(.*)/left/img_(.*).png')
    boxpattern = re.compile('.*/boxes/(.*)/boxes.npy.gz')
    boxfiles = glob.glob(path + '/boxes/*/*.npy.gz')
    boxes_dict = {}
    for file in boxfiles:
        boxes = np.load(gzip.open(file, 'rb'))
        match = boxpattern.match(file)
        (obj_type,) = match.groups()
        boxes_dict[obj_type] = boxes
    

    for twindow in np.int32(np.logspace(np.log10(5), np.log10(1000.0), num=10)):
        files = glob.glob(path + "frames/%dms/*/left/*.png" % twindow)
        for file in tqdm(files):
            match = pattern.match(file)
            obj_type, fileidx = match.groups()
            annotations = boxes_dict[obj_type][int(fileidx) - 1]
            with open(file, 'rb') as f:
                example = (obj_type, fileidx, annotations, f.read())
            tf_example = create_tf_example(example)
            writer.write(tf_example.SerializeToString())

    writer.close()


if __name__ == '__main__':
    tf.app.run()