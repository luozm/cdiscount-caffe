"""
Pre process the data set
"""
import os
import struct
from collections import defaultdict
from tqdm import *

import numpy as np
import pandas as pd
import bson

import utils


# ## Read the BSON files
# 
# We store the offsets and lengths of all items, allowing us random access to the items later.
# 
# Inspired by code from: https://www.kaggle.com/vfdev5/random-item-access
# 
# Note: this takes a few minutes to execute,
# but we only have to do it once (we'll save the table to a CSV file afterwards).
def read_bson(bson_path, num_records, with_categories):
    rows = {}
    with open(bson_path, "rb") as f, tqdm(total=num_records) as pbar:
        offset = 0
        while True:
            # The length of the item is stored as 4 bytes in file header
            item_length_bytes = f.read(4)

            # Check if file reaches the EOF
            if len(item_length_bytes) == 0:
                break

            # Read item data by offset and length
            length = struct.unpack("<i", item_length_bytes)[0]
            f.seek(offset)
            item_data = f.read(length)
            assert len(item_data) == length

            # Decode bson data to get id and raw image
            item = bson.BSON.decode(item_data)
            product_id = item["_id"]
            num_imgs = len(item["imgs"])

            # Save the retrieved information as a row
            row = [num_imgs, offset, length]
            if with_categories:
                row += [item["category_id"]]

            # Add the row in rows
            rows[product_id] = row

            # Update offset and prepare for next retrieval
            offset += length
            f.seek(offset)
            pbar.update()

    # Set the table header
    columns = ["num_imgs", "offset", "length"]
    if with_categories:
        columns += ["category_id"]

    # Construct the product table and return it
    product_table = pd.DataFrame.from_dict(rows, orient="index")
    product_table.index.name = "product_id"
    product_table.columns = columns
    product_table.sort_index(inplace=True)
    return product_table


# Create a random train/validation split
# 
# We split on products, not on individual images.
# Since some of the categories only have a few products, we do the split separately for each category.
# 
# This creates two new tables, one for the training images and one for the validation images.
# There is a row for every single image, so if a product has more than one image it occurs more than once in the table.
def make_val_set(product_table, category2index, category2index_level1, category2index_level2,
                 split_percentage=0.2, drop_percentage=0.):
    # Find the product_ids for each category.
    category_dict = defaultdict(list)
    for ir in tqdm(product_table.itertuples()):
        category_dict[ir[4]].append(ir[0])

    train_list = []
    val_list = []
    with tqdm(total=int(len(product_table)*(1-drop_percentage))) as pbar:
        for category_id, product_ids in category_dict.items():
            category_idx = category2index[category_id]
            category_idx_level1 = category2index_level1[category_id]
            category_idx_level2 = category2index_level2[category_id]

            # Randomly remove products to make the dataset smaller.
            keep_size = int(len(product_ids) * (1. - drop_percentage))
            if keep_size < len(product_ids):
                product_ids = np.random.choice(product_ids, keep_size, replace=False)

            # Randomly choose the products that become part of the validation set.
            val_size = int(len(product_ids) * split_percentage)
            if val_size > 0:
                val_ids = np.random.choice(product_ids, val_size, replace=False)
            else:
                val_ids = []

            # Create a new row for each image.
            for product_id in product_ids:
                row = [product_id, category_idx, category_idx_level1, category_idx_level2]
                for img_idx in range(product_table.loc[product_id, "num_imgs"]):
                    if product_id in val_ids:
                        val_list.append(row + [img_idx])
                    else:
                        train_list.append(row + [img_idx])
                pbar.update()
                
    columns = ["product_id", "category_idx", "category_idx_level1", "category_idx_level2", "img_idx"]
    train_df = pd.DataFrame(train_list, columns=columns)
    # Shuffle the whole training set
    train_df_shuffle = train_df.sample(frac=1).reset_index(drop=True)

    val_df = pd.DataFrame(val_list, columns=columns)
    return train_df_shuffle, val_df


def make_test_set(df):
    test_list = []
    for ir in tqdm(df.itertuples()):
        product_id = ir[0]
        num_imgs = ir[1]
        for img_idx in range(num_imgs):
            test_list.append([product_id, img_idx])

    columns = ["product_id", "img_idx"]
    test_df = pd.DataFrame(test_list, columns=columns)
    return test_df


train_bson_path = os.path.join(utils.data_dir, "train.bson")
test_bson_path = os.path.join(utils.data_dir, "test.bson")


# # Part 1: Create lookup tables
# 
# The generator uses several lookup tables that describe the layout of the BSON file,
# which products and images are part of the training/validation sets, and so on.
# 
# You only need to generate these tables once, as they get saved to CSV files.

# ## Lookup table for categories
categories_path = os.path.join(utils.data_dir, "category_names.csv")
categories_df = pd.read_csv(categories_path, index_col="category_id")


# Maps the category_id to an integer index.
categories_df["category_idx"] = pd.Series(range(len(categories_df)), index=categories_df.index)

categories_df.to_csv(utils.utils_dir + "categories.csv")


# Create dictionaries for quick lookup of `category_id` to `category_idx` mapping.
cat2idx, idx2cat = utils.make_category_tables(categories_df)


# For level1

# Maps the category_id_level1 to an integer index (category_level1_index)
categories_level1_list = list(categories_df["category_level1"].unique())
categories_level1_df = pd.DataFrame(categories_level1_list, columns=["category_level1_names"])
categories_level1_df["category_level1_index"] = [i for i in range(len(categories_level1_list))]

# Create dictionaries for quick lookup of `category_id` to `category_level1_index` mapping.
cat2idx_level1 = utils.make_category_table_level1(categories_level1_df, categories_df)


# For level2

# Maps the category_id_level1 to an integer index (category_level1_index)
categories_level2_list = list(categories_df["category_level2"].unique())
categories_level2_df = pd.DataFrame(categories_level2_list, columns=["category_level2_names"])
categories_level2_df["category_level2_index"] = [i for i in range(len(categories_level2_list))]

# Create dictionaries for quick lookup of `category_id` to `category_level1_index` mapping.
cat2idx_level2 = utils.make_category_table_level2(categories_level2_df, categories_df)


# ## Read the BSON files
# 
# We store the offsets and lengths of all items, allowing us random access to the items later.
print("Scanning train file:")
train_offsets_df = read_bson(
    train_bson_path,
    num_records=utils.num_train_products,
    with_categories=True)

train_offsets_df.to_csv(utils.utils_dir + "train_offsets.csv")
print("Successfully save train_offsets.csv")

# How many images in total?
print("Number of total training images:", train_offsets_df["num_imgs"].sum())


# Also create a table for the offsets from the test set.
print("Scanning test file:")
test_offsets_df = read_bson(
    test_bson_path,
    num_records=utils.num_test_products,
    with_categories=False)

test_offsets_df.to_csv(utils.utils_dir + "test_offsets.csv")
print("Successfully save test_offsets.csv")


# ## Create a random train/validation split
# 
# We split on products, not on individual images.
# Since some of the categories only have a few products, we do the split separately for each category.
# 
# This creates two new tables, one for the training images and one for the validation images.
# There is a row for every single image, so if a product has more than one image it occurs more than once in the table.

# Create a 80/20 split. Also can drop some of all products to make the dataset more manageable.
print("Spliting train/val set:")
train_images_df, val_images_df = make_val_set(
    train_offsets_df,
    cat2idx,
    cat2idx_level1,
    cat2idx_level2,
    split_percentage=0.1,
    drop_percentage=0)

print("Number of training images:", len(train_images_df))
print("Number of validation images:", len(val_images_df))
print("Total images:", len(train_images_df) + len(val_images_df))


# Save the lookup tables as CSV so that we don't need to repeat the above procedure again.
train_images_df.to_csv(utils.utils_dir + "train_images.csv")
val_images_df.to_csv(utils.utils_dir + "val_images.csv")
print("Successfully save train_images.csv and val_images.csv")

# ## Lookup table for test set images
# 
# Create a list containing a row for each image.
# If a product has more than one image, it appears more than once in this list.
print("Making test set:")
test_images_df = make_test_set(test_offsets_df)

print("Number of test images:", len(test_images_df))

test_images_df.to_csv(utils.utils_dir + "test_images.csv")
print("Successfully save test_images.csv")
