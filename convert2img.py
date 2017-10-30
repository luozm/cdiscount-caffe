import bson
import pandas as pd
import os
import tqdm


# basic setting
utils_dir = ''
pickle_file = pd.read_pickle(utils_dir + "val_dataset.pkl")
output_image_dir = '../images/val'


# Create output folders
if not os.path.exists(output_image_dir):
    os.makedirs(output_image_dir)


with tqdm.tqdm(total=len(pickle_file)) as bar:
    for idx in range(len(pickle_file)):
        sample = pickle_file.iloc[idx]

        # prepare the data and label
        image = sample["image"]
        label = sample["label"]
        id = sample["product_id"]
        img_idx = sample["img_idx"]

        file_name = '{}-{}-{}.jpg'.format(id, img_idx, label)

        # write image to file
        with open(os.path.join(output_image_dir, file_name), 'wb') as f:
            f.write(image)
        # write label info to txt
        with open("Output.txt", "w") as text_file:
            print("{} {}".format(file_name, label), file=text_file)

        bar.update()

