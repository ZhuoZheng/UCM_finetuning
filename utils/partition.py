import numpy as np
import os
import sys
import shutil

for i in range(1000):
    with open(r"E:\ttt\{0}.txt".format(i), 'w') as f:
        f.write("111")

DATASET = r"E:\ttt"
train_ratio = 0.8
filenames = os.listdir(DATASET)
filenames = np.asarray(filenames)
np.random.shuffle(filenames)
train_sum = int(filenames.shape[0] * 0.8)
train_files = filenames[:train_sum]
test_files = filenames[train_sum:]

if not os.path.exists("{0}/train".format(DATASET)):
    os.makedirs("{0}/train".format(DATASET))
if not os.path.exists("{0}/test".format(DATASET)):
    os.makedirs("{0}/test".format(DATASET))
for i, filename in enumerate(train_files):
    shutil.move("{0}/{1}".format(DATASET, filename), "{0}/train/{1}".format(DATASET, filename))
    sys.stdout.write("train-processing[{0}/{1}]\n".format(i + 1, train_sum))
    sys.stdout.flush()
for i, filename in enumerate(test_files):
    shutil.move("{0}/{1}".format(DATASET, filename), "{0}/test/{1}".format(DATASET, filename))
    sys.stdout.write("test-processing[{0}/{1}]\n".format(i + 1, filenames.shape[0] - train_sum))
    sys.stdout.flush()
