"""
This script predict the number of mitoses for the input slide images
"""
import os
import socket
from collections import Counter
from time import gmtime, strftime

import numpy as np
import openslide
import pandas as pd
from pathlib import Path
from PIL import Image
import tensorflow as tf

from deephistopath.preprocessing import create_tile_generator, get_20x_zoom_level
from deephistopath.visualization import add_mark, Shape
from preprocess_mitoses import create_mask, gen_dense_coords, extract_patch, gen_patches
from train_mitoses import create_augmented_batch, marginalize, normalize

MASK_FILE_PREFIX = "result/prediction/mitosis_location_img/"
LOCATION_FILE_PREFIX = "result/prediction/mitosis_location_csv/"


def save_array_2_image(img_path, data):
  """ save the numpy array as an image

  Args:
    img_path: image path, and its file extension will determine the
      image format
    data: the numpy array
  """
  data = np.asarray(data, dtype=np.uint8)
  dims = len(data.shape)
  if dims == 2:
     data = (1 - data) * 255
  im = Image.fromarray(data)
  im.save(img_path)

def save_mitosis_locations_2_csv(file_path, mitosis_locations):
  """ save the coordinate locations of mitoses into csv file

  Args:
    file_path: the output csv file path
    mitosis_locations: list of mitoses' coordinate locations
  """
  df = pd.DataFrame(mitosis_locations)
  df.columns = ["row", "col", "score"]
  df.to_csv(file_path, index=False)
  # skip the header and the score column to match the requirement of the
  # preprocess_mitoses.py
  #df[["row", "col"]].to_csv(file_path, index=False, header=False)

def check_subsetting(ROI, ROI_size, tiles, tile_size, tile_overlap, channel=3):
  """ check if the generation of tiles is right by re-combine the tiles

  Args:
    ROI: The original ROI image (h, w, c).
    ROI_size: The size of ROI.
    tiles: The tiles (n, h, w, c).
    tile_size: The size of tile.
    tile_overlap: The overlap between tiles.
    channel: the number of channel; ROI and tiles should have the same
      number of channel.
  """
  try:
    stride = tile_size - tile_overlap

    # compute the number of rows and columns when splitting ROI into tiles.
    col_num = row_num = (ROI_size - tile_size) // stride + 1
    height = width = stride * (col_num - 1) + tile_size
    tile_recombination = np.zeros((0, width, channel), dtype=np.uint8)
    for row in range(row_num):
      # for the last row, its height will be equal to the tile size;
      # Otherwise, it will be the stride of window moving.
      h = stride if row < row_num - 1 else tile_size
      cur = np.zeros((h, 0, channel), dtype=np.uint8)
      for col in range(col_num):
        # in each row, the width of last tile will be equal to tile size;
        # for other tiles, their width will be equal to stride.
        w = stride if col < col_num - 1 else tile_size
        index = col_num * row + col
        cur = np.concatenate((cur, tiles[index][0:h, 0:w, ]), axis=1)
      tile_recombination = np.concatenate((tile_recombination, cur), axis=0)
    return np.array_equal(ROI[0:height, 0:width, ], tile_recombination)
  except:
    return False

def pad_tile_on_edge(tile, tile_row, tile_col, tile_size, ROI):
  """ add the padding to the tile on the edges. If the tile's center is
    outside of ROI, move it back to the edge

  Args:
    tile: tile value
    tile_row: row number of the tile relative to its ROI
    tile_col: col number of the tile relative to its ROI
    tile_size: default tile size which may be different from the input
      tile
    ROI: ROI value which contains the input tile

  Return:
    the padded tile
  """

  tile_height, tile_width, tile_channel = tile.shape
  tile_row_lower = tile_row
  tile_row_upper = tile_row + tile_height
  tile_col_lower = tile_col
  tile_col_upper = tile_col + tile_width
  # if the tile's center is outside of ROI, move it back to the edge,
  # and then add the padding
  if tile_height < tile_size / 2:
    tile_row_lower = tile_row_upper - tile_size // 2
    tile_height = tile_size // 2
  if tile_width < tile_size / 2:
    tile_col_lower = tile_col_upper - tile_size // 2
    tile_width = tile_size // 2
  tile = ROI[tile_row_lower: tile_row_upper, tile_col_lower: tile_col_upper, ]
  padding = ((0, tile_size - tile_height), (0, tile_size - tile_width), (0, 0))
  return np.pad(tile, padding, "reflect")

def gen_batches(iterator, batch_size, include_partial=True):
  """ generate the tile batches from the tile iterator
  Args:
    iterator: the tile iterator
    batch_size: batch size
    include_partial: boolean value to keep the partial batch or not

  Return:
    the iterator for the tile batches
  """
  batch = []
  for item in iterator:
    batch.append(item)
    if len(batch) == batch_size:
      yield batch
      batch = []
  if len(batch) > 0 and include_partial:
    yield batch


def predict_mitoses_num_locations(model, model_name, threshold, ROI, tile_size=64, tile_overlap=0,
                                  tile_channel=3, batch_size=128, marginalization=False):
  """ Predict the number of mitoses with the detected mitosis locations
    for each input ROI.

  Args:
    model: model loaded from the model file.
    model_name: name of the input model, e.g. vgg, resnet.
    threshold: threshold for the output of last sigmoid layer.
    ROI: ROI in numpy array.
    ROI_size: size of ROI.
    ROI_overlap: overlap between ROIs.
    ROI_row: row number of the ROI in the input slide image. If setting
      it 0, the original coordination will be the left-upper corner of
      the input ROI.
    ROI_col: col number of the ROI in the input slide image. If setting
      it 0, the original coordination will be the left-upper corner of
      the input ROI.
    tile_size: tile size.
    tile_overlap: overlap between tiles.
    tile_channel: channel of tiles.
    batch_size: the batch_size for prediction.
    marginalization: Boolean for whether or not to use noise
      marginalization when making predictions.  If True, then
      each image will be expanded to a batch of size `batch_size` of
      augmented versions of that image, and predicted probabilities for
      each batch will be averaged to yield a single noise-marginalized
      prediction for each image.  Note: if this is True, then
      `batch_size` must be divisible by 4, or equal to 1 for a special
      debugging case of no augmentation.

  Return:
     the prediction result for the input ROI, (mitosis_num,
     mitosis_location_scores).
  """
  ROI_height, ROI_width, ROI_channel = ROI.shape

  # gen_dense_coords function will handle the cases that the tile center point is outside of the ROI
  tile_indices = list(gen_dense_coords(ROI_height, ROI_width, tile_size - tile_overlap))

  mitosis_location_scores = []
  predictions = np.empty((0, 1))

  if marginalization:
    # create tiles larger than the intended size so that we can perform random rotations and
    # random translations via cropping
    d = 72  # TODO: keep this in sync with the training augmentation code
    tiles = (element[0] for element in gen_patches(ROI, tile_indices, tile_size+d, rotations=0,
        translations=0, max_shift=0, p=1))

    # create marginalization graph
    # NOTE: averaging over sigmoid outputs vs. logits may yield slightly different results, due
    # to numerical precision
    prep_tile = tf.placeholder(tf.float32, shape=[tile_size+d, tile_size+d, tile_channel])
    aug_tiles = create_augmented_batch(prep_tile, batch_size, tile_size)  # create aug batch
    norm_tiles = normalize(aug_tiles, model_name)  # normalize augmented tiles
    aug_preds = model(norm_tiles)  # make predictions on normalized and augmented batch
    pred = marginalize(aug_preds)  # average predictions

    # make predictions
    sess = tf.keras.backend.get_session()
    for tile in tiles:
      prep_tile_np = (tile / 255).astype(np.float32)  # convert to values in [0,1]
      pred_np, aug_preds_np = sess.run((pred, aug_preds),
          feed_dict={prep_tile: prep_tile_np, tf.keras.backend.learning_phase(): 0})
      predictions = np.concatenate((predictions, pred_np), axis=0)

      print (f"The {predictions.shape[0]}th prediction: max: {np.max(aug_preds_np)}, min: "\
             f"{np.min(aug_preds_np)}, avg: {pred_np}")

  else:
    tiles = (element[0] for element in gen_patches(ROI, tile_indices, tile_size, rotations=0,
        translations=0, max_shift=0, p=1))
    tile_batches = gen_batches(tiles, batch_size, include_partial=True)
    for tile_batch in tile_batches:
      tile_stack = np.stack(tile_batch, axis=0)
      tile_stack = normalize((tile_stack / 255).astype(dtype=np.float32), model_name)
      pred_np = model.predict(tile_stack, batch_size)
      predictions = np.concatenate((predictions, pred_np), axis=0)

  isMitoses = predictions > threshold
  for i in range(isMitoses.shape[0]):
    if isMitoses[i]:
      tile_row_index, tile_col_index = tile_indices[i]
      mitosis_location_scores.append((tile_row_index, tile_col_index,
                                      np.asscalar(predictions[i])))

  mitosis_num = len(mitosis_location_scores)
  return (mitosis_num, mitosis_location_scores)


def predict_mitoses_help(model_file, model_name, index, file_partition,
                         ROI_size, ROI_overlap, ROI_channel=3, skipROI=False,
                         tile_size=64, tile_overlap=0, tile_channel=3,
                         threshold=0.5, isGPU=True, batch_size=128, marginalization=False,
                         save_mitosis_locations=False, save_mask=False, isDebug=False):
  """ Predict the number of mitoses for each input slide image.

  Args:
    model_file: file path for the input model (.hdf5).
    model_name: name of the input model, e.g. vgg, resnet.
    index: if using GPU, it will be the assigned gpu id for this
      partition; if using cpu, it will the split index.
    file_partition: The partition of input files.
    ROI_size: The ROI size.
    ROI_overlap: The overlap between ROIs.
    ROI_channel: The channel of ROI.
    skipROI: True if skipping the ROI layer; False if adding the ROI layer.
    tile_size: The tile siz.
    tile_overlap: The overlap between tiles.
    tile_channel: The channel of tiles.
    threshold: The threshold for the output of last sigmoid layer.
    isGPU: true if running on tensorflow-GPU; false if running on
      tensorflow-CPU.
    batch_size: the batch_size for prediction.
    marginalization: Boolean for whether or not to use noise
      marginalization when making predictions.  If True, then
      each image will be expanded to a batch of size `batch_size` of
      augmented versions of that image, and predicted probabilities for
      each batch will be averaged to yield a single noise-marginalized
      prediction for each image.  Note: if this is True, then
      `batch_size` must be divisible by 4, or equal to 1 for a special
      debugging case of no augmentation.
    save_mitosis_locations: bool value to determine if save the detected
      mitosis locations into csv file.
    save_mask: bool value to determine if saving the mask as an image.
    isDebug: if true, print out the debug information.

  Return:
     A list of prediction result tuples (slide_id, ROI_id, mitoses_sum,
     mitosis_locations).
  """

  # configure GPU
  if isGPU:
    gpu_id = str(index)   #get_gpus(1)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    # conf = tf.ConfigProto()
    # conf.allow_soft_placement = True
    # conf.gpu_options.visible_device_list = gpu_id
    # tf_session = tf.Session(config=conf)
    # K.set_session(tf_session)
    if isDebug:
      print(f"GPU_ID: {gpu_id}")

  config = tf.ConfigProto(allow_soft_placement=True)#, log_device_placement=True)
  sess = tf.Session(config=config)
  tf.keras.backend.set_session(sess)

  # load the model and add the sigmoid layer
  base_model = tf.keras.models.load_model(model_file, compile=False)

  # specify the name of the added activation layer to avoid the name
  # conflict in ResNet
  probs = tf.keras.layers.Activation('sigmoid', name="sigmoid")(base_model.output)
  model = tf.keras.models.Model(inputs=base_model.input, outputs=probs)

  result = []

  for file_path in file_partition:
    # generate the ROI indices for each input file

    # use the file path as the slide id
    slide_id = str(file_path).replace("/", "_").replace(".", "_")
    slide = openslide.open_slide(str(file_path))
    # for the case that does not need the ROI layer between the input
    # image and tiles, set the ROI_size to be as large as the input
    # image to make ROI to be same with the input image
    if skipROI:
      ROI_size = max(slide.dimensions)
    generator = create_tile_generator(slide, ROI_size, ROI_overlap)
    zoom_level = get_20x_zoom_level(slide, generator)
    cols, rows = generator.level_tiles[zoom_level]
    ROI_indices = [(zoom_level, col, row) for col in range(cols) for row in range(rows)]

    for ROI_index in ROI_indices:
      # get the ROI
      zl, col, row = ROI_index
      ROI = np.asarray(generator.get_tile(zl, (col, row)))
      # if we are using ROIs, skip the ROIs whose size is smaller than the tile size
      if not skipROI and (ROI.shape[0] < ROI_size or ROI.shape[1] < ROI_size):
        continue
      mitosis_location_score_list = []
      # predict the mitoses with location information
      mitosis_num, mitosis_location_scores  = predict_mitoses_num_locations(model=model,
                                                    model_name=model_name, threshold=threshold,
                                                    ROI=ROI, tile_size=tile_size,
                                                    tile_overlap=tile_overlap,
                                                    tile_channel=tile_channel,
                                                    batch_size=batch_size,
                                                    marginalization=marginalization)

      mitosis_location_score_list += mitosis_location_scores
      result.append((slide_id, f"ROI_{row}_{col}", mitosis_num, mitosis_location_scores))

      if isDebug:
        cur_time = strftime("%Y-%m-%d %H:%M:%S", gmtime())
        info = f"Slide: {slide_id}, ROI_ID: {row}_{col}, Mitosis_Num: {mitosis_num}; " \
               f"Time: {cur_time}"
        print(info)

      if save_mitosis_locations and len(mitosis_location_score_list) > 0:
        csv_path = str(file_path).replace("data", "result").split(".")[0]

        if skipROI:
          csv_path = csv_path +  ".csv"
        else:
          csv_path = csv_path + f"_ROI_{row}_{col}" + ".csv"

        csv_path = Path(csv_path)
        dir = os.path.dirname(csv_path)
        os.makedirs(dir, exist_ok=True)
        save_mitosis_locations_2_csv(csv_path, mitosis_location_score_list)

      if save_mask and len(mitosis_location_score_list) > 0:
        mitosis_rows, mitosis_cols, pred_scores = zip(*mitosis_location_score_list)
        mitosis_location_list = list(zip(mitosis_rows, mitosis_cols))
        mask = create_mask(ROI_size, ROI_size, mitosis_location_list, 30)  # TODO: radius hyperparam
        mask_path = str(file_path).replace("data", "result").split(".")[0]
        pred_mark_ROI_path = str(file_path).replace("data", "result").split(".")[0]

        if skipROI:
          mask_path = mask_path + "_mask.tif"
          pred_mark_ROI_path = pred_mark_ROI_path + "_mark.tif"
        else:
          mask_path = mask_path + f"_ROI_{row}_{col}" + "_mask.tif"
          pred_mark_ROI_path = pred_mark_ROI_path + f"_ROI_{row}_{col}" + "_mark.tif"

        mask_path = Path(mask_path)
        pred_mark_ROI_path = Path(pred_mark_ROI_path)
        dir = os.path.dirname(mask_path)
        os.makedirs(dir, exist_ok=True)
        save_array_2_image(mask_path, mask)
        # add the detected mitoses into the original image
        im_ROI = Image.fromarray(ROI)
        #add_mark(im_ROI, mitosis_location_list, Shape.SQUARE)
        im_ROI.save(pred_mark_ROI_path)

  return result

def predict_mitoses(sparkContext, model_path, model_name, input_dir, file_suffix, partition_num,
                    ROI_size, ROI_overlap, ROI_channel=3, skipROI=False, tile_size=64,
                    tile_overlap=0, tile_channel=3, threshold=0.5, isGPU=True, batch_size=128,
                    marginalization=False, save_mitosis_locations=False, save_mask=False,
                    isDebug=False):
    """ Predict the number of mitoses for the input slide images. It
      supports both GPUs and CPUs.

    Args:
      sparkContext: Spark context.
      model_path: file path for the input model (.hdf5).
      model_name: model name for the input model, e.g. vgg, resnet.
      input_dir: directory for the input slide images.
      file_suffix: the suffix for the slide image file path. It can be
        used to filter out the inputs.
      partition_num: number of the partitions of input slide images.
      ROI_size: size of region of interest. ROI will be a square.
      ROI_overlap: over lap between ROIs
      ROI_channel: channel of ROI.
      skipROI: True if skipping the ROI layer; False if adding the ROI
        layer.
      tile_size: size of tile. The tile will be a square.
      tile_overlap: overlap between tiles.
      tile_channel: channel of tiles
      threshold: threshold for the output of last sigmoid layer.
      isGPU: true if running on tensorflow-GPU; false if running on
        tensorflow-CPU
      batch_size: the batch_size for prediction.
      marginalization: Boolean for whether or not to use noise
        marginalization when making predictions.  If True, then
        each image will be expanded to a batch of size `batch_size` of
        augmented versions of that image, and predicted probabilities for
        each batch will be averaged to yield a single noise-marginalized
        prediction for each image.  Note: if this is True, then
        `batch_size` must be divisible by 4, or equal to 1 for a special
        debugging case of no augmentation.
      save_mitosis_locations: bool value to determine if save the
        detected mitosis locations into csv file.
      save_mask: bool value to determine if saving the mask as an image.
      isDebug: if true, print out the debug information

    Return:
       A list of prediction result tuple (slide_id, ROI_id, mitoses_sum,
       mitosis_locations)
    """
    if isGPU:
        predictions_rdd = predict_mitoses_gpu(sparkContext=sparkContext, model_path=model_path,
                                              model_name=model_name,
                                              input_dir=input_dir, file_suffix=file_suffix,
                                              partition_num=partition_num,
                                              ROI_size=ROI_size, ROI_overlap=ROI_overlap,
                                              ROI_channel=ROI_channel, skipROI=skipROI,
                                              tile_size=tile_size, tile_overlap=tile_overlap,
                                              tile_channel=tile_channel, threshold=threshold,
                                              batch_size=batch_size,
                                              marginalization=marginalization,
                                              save_mitosis_locations=save_mitosis_locations,
                                              save_mask=save_mask, isDebug=isDebug)
    else:
        predictions_rdd = predict_mitoses_cpu(sparkContext=sparkContext, model_path=model_path,
                                              model_name=model_name,
                                              input_dir=input_dir, file_suffix=file_suffix,
                                              partition_num=partition_num,
                                              ROI_size=ROI_size, ROI_overlap=ROI_overlap,
                                              ROI_channel=ROI_channel, skipROI=skipROI,
                                              tile_size=tile_size, tile_overlap=tile_overlap,
                                              tile_channel=tile_channel, threshold=threshold,
                                              batch_size=batch_size,
                                              marginalization=marginalization,
                                              save_mitosis_locations=save_mitosis_locations,
                                              save_mask=save_mask, isDebug=isDebug)
    return predictions_rdd


def predict_mitoses_gpu(sparkContext, model_path, model_name, input_dir, file_suffix, partition_num,
                    ROI_size, ROI_overlap, ROI_channel=3, skipROI=False, tile_size=64,
                    tile_overlap=0, tile_channel=3, threshold=0.5, batch_size=128,
                    marginalization=False, save_mitosis_locations=False, save_mask=False,
                    isDebug=False):
  """ Predict the number of mitoses for the input slide images using GPU.

    Args:
      sparkContext: Spark context.
      model_path: file path for the input model (.hdf5).
      model_name: model name for the input model, e.g. vgg, resnet.
      input_dir: directory for the input slide images.
      file_suffix: the suffix for the slide image file path. It can be
        used to filter out the inputs.
      partition_num: number of the partitions of input slide images.
      ROI_size: size of region of interest. ROI will be a square.
      ROI_overlap: over lap between ROIs.
      ROI_channel: channel of ROI.
      skipROI: True if skipping the ROI layer; False if adding the ROI
        layer.
      tile_size: size of tile. The tile will be a square.
      tile_overlap: overlap between tiles.
      tile_channel: channel of tiles.
      threshold: threshold for the output of last sigmoid layer.
      batch_size: the batch_size for prediction.
      marginalization: Boolean for whether or not to use noise
        marginalization when making predictions.  If True, then
        each image will be expanded to a batch of size `batch_size` of
        augmented versions of that image, and predicted probabilities for
        each batch will be averaged to yield a single noise-marginalized
        prediction for each image.  Note: if this is True, then
        `batch_size` must be divisible by 4, or equal to 1 for a special
        debugging case of no augmentation.
      save_mitosis_locations: bool value to determine if save the
        detected mitosis locations into csv file.
      save_mask: bool value to determine if saving the mask as an image.
      isDebug: if true, print out the debug information.

    Return:
       A list of prediction result tuple (slide_id, ROI_id, mitoses_sum,
       mitosis_locations).
    """

  input_dir = Path(input_dir)
  input_imgs = [str(x) for x in input_dir.rglob(file_suffix)]

  # parallel the input slide images and split it to several partitions
  # specified by the input At the same time, cache this rdd to keep the
  # location of each partition
  slide_rdd = sparkContext.parallelize(input_imgs, partition_num).cache()

  # get the hostname for each partition
  get_hostname = lambda split_index, partitions : [(split_index, socket.gethostname())]
  split_index_2_host = slide_rdd.mapPartitionsWithIndex(get_hostname).collect()

  # TODO: there is an assumption here that the number of partitions on
  # each node is same with the number of available GPUs
  # get the number of available GPUs on each node
  host_2_gpu_num = Counter(t[1] for t in split_index_2_host)

  # assign the gpu id to each partition
  split_index_2_gpu_id = []
  for tuple in split_index_2_host:
        host_2_gpu_num[tuple[1]] -= 1
        map = (tuple[0], host_2_gpu_num[tuple[1]])
        split_index_2_gpu_id.append(map)
  split_index_2_gpu_id = dict(split_index_2_gpu_id)

  if isDebug:
    print (host_2_gpu_num)
    print(split_index_2_host)
    print(split_index_2_gpu_id)

  # assign GPU id to each partition, and then run the predict function for each partition.
  predictions_rdd = slide_rdd.mapPartitionsWithIndex(lambda index, p: [(split_index_2_gpu_id[index], p)])\
    .flatMap(lambda t : predict_mitoses_help(model_file=model_path, model_name = model_name,
                                          index=t[0], file_partition=t[1], ROI_size=ROI_size,
                                         ROI_overlap=ROI_overlap, ROI_channel=ROI_channel,
                                         skipROI=skipROI,
                                         tile_size=tile_size, tile_overlap=tile_overlap,
                                         tile_channel=tile_channel, threshold=threshold,
                                         isGPU=True, batch_size=batch_size,
                                         marginalization=marginalization,
                                         save_mitosis_locations=save_mitosis_locations,
                                         save_mask=save_mask, isDebug=isDebug))

  return predictions_rdd

def predict_mitoses_cpu(sparkContext, model_path, model_name, input_dir, file_suffix, partition_num,
                        ROI_size, ROI_overlap, ROI_channel=3, skipROI=False, tile_size=64,
                        tile_overlap=0, tile_channel=3, threshold=0.5, batch_size=128,
                        marginalization=False, save_mitosis_locations=False, save_mask=False,
                        isDebug=False):
  """ Predict the number of mitoses for the input slide images using CPU.

    Args:
      sparkContext: Spark context.
      model_path: file path for the input model (.hdf5).
      model_name: model name for the input model, e.g. vgg, resnet.
      input_dir: directory for the input slide images.
      file_suffix: the suffix for the slide image file path. It can be
        used to filter out the inputs.
      partition_num: number of the partitions of input slide images.
      ROI_size: size of region of interest. ROI will be a square.
      ROI_overlap: over lap between ROIs.
      ROI_channel: channel of ROI.
      skipROI: True if skipping the ROI layer; False if adding the ROI
        layer.
      tile_size: size of tile. The tile will be a square.
      tile_overlap: overlap between tiles.
      tile_channel: channel of tiles.
      threshold: threshold for the output of last sigmoid layer.
      batch_size: the batch_size for prediction.
      marginalization: Boolean for whether or not to use noise
        marginalization when making predictions.  If True, then
        each image will be expanded to a batch of size `batch_size` of
        augmented versions of that image, and predicted probabilities for
        each batch will be averaged to yield a single noise-marginalized
        prediction for each image.  Note: if this is True, then
        `batch_size` must be divisible by 4, or equal to 1 for a special
        debugging case of no augmentation.
      save_mitosis_locations: bool value to determine if save the
        detected mitosis locations into csv file.
      save_mask: bool value to determine if saving the mask as an image.
      isDebug: if true, print out the debug information.

    Return:
       A list of prediction result tuple (slide_id, ROI_id, mitoses_sum, mitosis_locations).
    """

  input_dir = Path(input_dir)
  input_imgs = [str(x) for x in input_dir.rglob(file_suffix)]

  # parallel the input slide images and repartition it to be same with the number of input images
  rdd = sparkContext.parallelize(input_imgs, partition_num)

  # run the predict function for each partition.
  predictions_rdd = rdd.mapPartitionsWithIndex(lambda index, p: predict_mitoses_help(
                                                    model_file=model_path, model_name = model_name,
                                                    index=index, file_partition=p,
                                                    ROI_size=ROI_size, ROI_overlap=ROI_overlap,
                                                    ROI_channel=ROI_channel, skipROI=skipROI,
                                                    tile_size=tile_size, tile_overlap=tile_overlap,
                                                    tile_channel=tile_channel,
                                                    threshold=threshold, isGPU=False,
                                                    batch_size=batch_size,
                                                    marginalization=marginalization,
                                                    save_mitosis_locations=save_mitosis_locations,
                                                    save_mask=save_mask, isDebug=isDebug))
  return predictions_rdd

def test_predict_mitoses_num_locations():
  #TODO: change the model file path
  #model_file = 'model/0.74172_f1_1.7319_loss_8_epoch_model.hdf5'
  model_name = 'vgg'
  model_file = 'experiments/models/deep_histopath_model.hdf5'
  #model_name = 'resnet'
  marginalization = True
  threshold = 0
  tile_overlap = 0
  batch_size = 128
  ROI = np.asarray(Image.open("data/mitoses/patches/val/mitosis/1_04_21_723_1803_0_0_0_7.png"), dtype=np.int)

  # the expected probability
  base_model = tf.keras.models.load_model(model_file  , compile=False)
  probs = tf.keras.layers.Activation('sigmoid', name="sigmoid")(base_model.output)
  model = tf.keras.models.Model(inputs=base_model.input, outputs=probs)
  norm_ROI = normalize((ROI / 255).astype(dtype=np.float32), model_name)
  prob = model.predict(np.expand_dims(norm_ROI, axis=0))
  print(f"The expected probability: {prob}")

  # the predicted result
  pred_result = predict_mitoses_num_locations(model, model_name, threshold, ROI,
                                                    tile_size=64, tile_overlap=tile_overlap, tile_channel=3,
                                                    batch_size=batch_size, marginalization=marginalization)
  print(f"The predicted probability: {pred_result}")


