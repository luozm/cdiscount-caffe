import lmdb
import numpy as np
from skimage.data import imread
import caffe
import bson
import pandas as pd
import io
import os
import tqdm
import utils


# Convert bson samples to datum
def sample2datum(bson_file, image_table, product_table, index, txn):

    image_row = image_table.iloc[index]
    product_id = image_row["product_id"]
    offset_row = product_table.loc[product_id]

    # Read this product's data from the BSON file.
    bson_file.seek(offset_row["offset"])
    item_data = bson_file.read(offset_row["length"])

    # Grab the image from the product.
    item = bson.BSON.decode(item_data)
    img_idx = image_row["img_idx"]
    bson_img = item["imgs"][img_idx]["picture"]

    image = imread(io.BytesIO(bson_img))

    # Transpose to CxHxW array, uint8
    image = np.transpose(image, [2, 0, 1])
    label = image_row["category_idx"]

    # save in datum
    datum = caffe.io.array_to_datum(image, label)
    keystr = '{:08}'.format(index)

    txn.put(keystr.encode("ascii"), datum.SerializeToString())


# Convert BSON to LMDB
def convert2lmdb(bson_file, image_table, product_table, lmdb_file):
    # create the lmdb file
    lmdb_env = lmdb.open(lmdb_file, map_size=int(1e12))
    lmdb_txn = lmdb_env.begin(write=True)
    print("Start converting {}...".format(lmdb_file))
    with tqdm.tqdm(total=10000) as bar:
        for idx in range(10000):

            # Put datum into txn
            sample2datum(bson_file,
                         image_table,
                         product_table,
                         idx,
                         lmdb_txn)

            # write batch
            if (idx + 1) % batch_size == 0:
                lmdb_txn.commit()
                lmdb_txn = lmdb_env.begin(write=True)

            bar.update()

    # write last batch
    if (idx + 1) % batch_size != 0:
        lmdb_txn.commit()


# Input data files are available in the "../data/input/" directory.
data_dir = utils.data_dir
utils_dir = utils.utils_dir

train_bson_path = os.path.join(data_dir, "train.bson")
lmdb_train_file = 'lmdb_train'
lmdb_val_file = 'lmdb_valid'

# First load the lookup tables from the CSV files.
train_product_table = pd.read_csv(utils_dir + "train_offsets.csv", index_col=0)
train_image_table = pd.read_csv(utils_dir + "train_images.csv", index_col=0)
val_image_table = pd.read_csv(utils_dir+"val_images.csv", index_col=0)

num_train_images = len(train_image_table)
num_val_images = len(val_image_table)
train_bson_file = open(train_bson_path, "rb")

batch_size = 8192


# Convert training set
convert2lmdb(
    train_bson_file,
    train_image_table,
    train_product_table,
    os.path.join(utils_dir, lmdb_train_file)
)

# Convert validation set
convert2lmdb(
    train_bson_file,
    val_image_table,
    train_product_table,
    os.path.join(utils_dir, lmdb_val_file)
)
