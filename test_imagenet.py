import pandas as pd
import numpy as np
import cv2
import googlenetbn
import chainer
import cupy as cp

model = googlentbn.GoogLeNetBN()
chainer.serializers.load_npz('', model)
chainer.cuda.get_device(0).use()
model.to_gpu()

def image_reader(root, path):
    name = root+'/'+path
    img = cv2.imread(name).astype(np.float32)
    h, w, _ = img.shape
    if h > w:
        new_h = int( 1.0 * h / w * 256.0)
        new_w = 256
    else:
        new_h = 256
        new_w = int( 1.0 * w / h * 256.0)
    img = cv2.resize(img,(new_w, new_h))
    top = (new_h - 224) / 2
    left = (new_w - 224) / 2
    img = img[top:224+top,left:224+left,:].transpose(2, 0, 1)
    img -= np.load('mean.npy')[:, :crop_size, :crop_size]
    return img

def class_check(img):
    img = cp.asarray(img)
    predict_class = np.argmax(model.predict(img).get())
    return predict_class

s = pd.read_csv( 'clf_test.tsv', delimiter='\t')
s = s.as_matrix()

predict=[]
for _, _path in s:
    predict_class = (image_reader('../test_images', _path))
    print (_path)
    predict.append(predict_class)
predict = np.asarray(predict)
output = pd.Series(predict)
output.to_csv( 'test.csv' )