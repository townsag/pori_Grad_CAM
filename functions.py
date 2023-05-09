import os
from PIL import Image
import pandas as pd
# for display img gird
from mpl_toolkits.axes_grid1 import ImageGrid
import math
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
# for plot confusion matrix
import itertools


def get_dataframe(parent_directory: str):
    temp = []

    for category in os.listdir(parent_directory):
        subdirectory_path = os.path.join(parent_directory, category)
        if os.path.isdir(subdirectory_path):

            for file_name in os.listdir(subdirectory_path):
                file_path = os.path.join(subdirectory_path, file_name)
                if os.path.isfile(file_path):

                    try:
                        with Image.open(file_path) as im:
                            temp.append({'file_name': os.path.join(category, file_name),
                                         'class_name': category,
                                         'height': im.size[1],
                                         'width': im.size[0]})
                    except OSError:
                        print("OSError")
                        pass

    df = pd.DataFrame(temp)
    # df.set_index("file_name", inplace=True)

    return df


def display_img_grid(df, num: int, image_class: str, parent_path: str, W: int, H: int, preprocess=None):
    filtered_df = df[df['class_name'] == image_class]
    num_to_display = min(filtered_df.shape[0], num)
    num_cols = math.ceil(math.sqrt(num_to_display))
    # image_sub_paths = filtered_df.iloc[0:num_to_display]['file_name'].tolist()
    # image_paths = [os.path.join()]
    # image_paths = [os.path.join(parent_path, i) for i in filtered_df.iloc[0:num_to_display].index.tolist()]
    image_paths = [os.path.join(parent_path, i) for i in filtered_df.iloc[0:num_to_display]['file_name'].tolist()]

    if preprocess is not None:
        images = [preprocess(np.expand_dims(img_to_array(load_img(i, target_size=(W, H))), axis=0))[0]
                  for i in image_paths]
    else:
        images = [img_to_array(load_img(i, target_size=(W, H))).astype(int) for i in image_paths]
    # images = [img_to_array(load_img(i)).astype(int) for i in image_paths]

    fig = plt.figure(figsize=(8., 8.))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(num_cols, num_cols),  # creates 2x2 grid of axes
                     axes_pad=0.1,  # pad between axes in inch.
                     )

    for ax, im in zip(grid, images):
        # Iterating over the grid returns the Axes.
        ax.imshow(im)

    plt.show()


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    # plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
