#-------------------------------------------------------------
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#
#-------------------------------------------------------------

"""
Preprocessing -- Predicting Breast Cancer Proliferation Scores with
Apache SystemML

This module contains functions for the preprocessing phase of the
breast cancer project.
"""

import math
import os

import numpy as np
import openslide
from PIL import Image
from openslide import OpenSlideError
from openslide.deepzoom import DeepZoomGenerator
import pandas as pd
from pyspark.ml.linalg import Vectors
import pyspark.sql.functions as F
from scipy.ndimage.morphology import binary_fill_holes
from skimage.color import rgb2gray
from skimage.feature import canny
from skimage.morphology import binary_closing, binary_dilation, disk


# Open Whole-Slide Image

def open_slide(slide_num, folder, training):
  """
  Open a whole-slide image, given an image number.

  Args:
    slide_num: Slide image number as an integer.
    folder: Directory in which the slides folder is stored, as a string.
      This should contain either a `training_image_data` folder with
      images in the format `TUPAC-TR-###.svs`, or a `testing_image_data`
      folder with images in the format `TUPAC-TE-###.svs`.
    training: Boolean for training or testing datasets.

  Returns:
    An OpenSlide object representing a whole-slide image.
  """
  if training:
    filename = os.path.join(folder, "training_image_data",
                            "TUPAC-TR-{}.svs".format(str(slide_num).zfill(3)))
  else:
    # Testing images
    filename = os.path.join(folder, "testing_image_data",
                            "TUPAC-TE-{}.svs".format(str(slide_num).zfill(3)))
  try:
    slide = openslide.open_slide(filename)
  except OpenSlideError:
    slide = None
  except FileNotFoundError:
    slide = None
  return slide


# Create Tile Generator

def create_tile_generator(slide, tile_size, overlap):
  """
  Create a tile generator for the given slide.

  This generator is able to extract tiles from the overall
  whole-slide image.

  Args:
    slide: An OpenSlide object representing a whole-slide image.
    tile_size: The width and height of a square tile to be generated.
    overlap: Number of pixels by which to overlap the tiles.

  Returns:
    A DeepZoomGenerator object representing the tile generator. Each
    extracted tile is a PIL Image with shape
    (tile_size, tile_size, channels).
    Note: This generator is not a true "Python generator function", but
    rather is an object that is capable of extracting individual tiles.
  """
  generator = DeepZoomGenerator(slide, tile_size=tile_size, overlap=overlap, limit_bounds=True)
  return generator


# Determine 20x Magnification Zoom Level

def get_20x_zoom_level(slide, generator):
  """
  Return the zoom level that corresponds to a 20x magnification.

  The generator can extract tiles from multiple zoom levels,
  downsampling by a factor of 2 per level from highest to lowest
  resolution.

  Args:
    slide: An OpenSlide object representing a whole-slide image.
    generator: A DeepZoomGenerator object representing a tile generator.
      Note: This generator is not a true "Python generator function",
      but rather is an object that is capable of extracting individual
      tiles.

  Returns:
    Zoom level corresponding to a 20x magnification, or as close as
    possible.
  """
  highest_zoom_level = generator.level_count - 1  # 0-based indexing
  try:
    mag = int(slide.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER])
    # `mag / 20` gives the downsampling factor between the slide's
    # magnification and the desired 20x magnification.
    # `(mag / 20) / 2` gives the zoom level offset from the highest
    # resolution level, based on a 2x downsampling factor in the
    # generator.
    offset = math.floor((mag / 20) / 2)
    level = highest_zoom_level - offset
  except (ValueError, KeyError) as e:
    # In case the slide magnification level is unknown, just
    # use the highest resolution.
    level = highest_zoom_level
  return level


# Generate Tile Indices For Whole-Slide Image.

def process_slide(slide_num, folder, training, tile_size, overlap):
  """
  Generate all possible tile indices for a whole-slide image.

  Given a slide number, tile size, and overlap, generate
  all possible (slide_num, tile_size, overlap, zoom_level, col, row)
  indices.

  Args:
    slide_num: Slide image number as an integer.
    folder: Directory in which the slides folder is stored, as a string.
      This should contain either a `training_image_data` folder with
      images in the format `TUPAC-TR-###.svs`, or a `testing_image_data`
      folder with images in the format `TUPAC-TE-###.svs`.
    training: Boolean for training or testing datasets.
    tile_size: The width and height of a square tile to be generated.
    overlap: Number of pixels by which to overlap the tiles.

  Returns:
    A list of (slide_num, tile_size, overlap, zoom_level, col, row)
    integer index tuples representing possible tiles to extract.
  """
  # Open slide.
  slide = open_slide(slide_num, folder, training)
  # Create tile generator.
  generator = create_tile_generator(slide, tile_size, overlap)
  # Get 20x zoom level.
  zoom_level = get_20x_zoom_level(slide, generator)
  # Generate all possible (zoom_level, col, row) tile index tuples.
  cols, rows = generator.level_tiles[zoom_level]
  tile_indices = [(slide_num, tile_size, overlap, zoom_level, col, row)
                  for col in range(cols) for row in range(rows)]
  return tile_indices


# Generate Tile From Tile Index

def process_tile_index(tile_index, folder, training):
  """
  Generate a tile from a tile index.

  Given a (slide_num, tile_size, overlap, zoom_level, col, row) tile
  index, generate a (slide_num, tile) tuple.

  Args:
    tile_index: A (slide_num, tile_size, overlap, zoom_level, col, row)
      integer index tuple representing a tile to extract.
    folder: Directory in which the slides folder is stored, as a string.
      This should contain either a `training_image_data` folder with
      images in the format `TUPAC-TR-###.svs`, or a `testing_image_data`
      folder with images in the format `TUPAC-TE-###.svs`.
    training: Boolean for training or testing datasets.

  Returns:
    A (slide_num, tile) tuple, where slide_num is an integer, and tile
    is a 3D NumPy array of shape (tile_size, tile_size, channels) in
    RGB format.
  """
  slide_num, tile_size, overlap, zoom_level, col, row = tile_index
  # Open slide.
  slide = open_slide(slide_num, folder, training)
  # Create tile generator.
  generator = create_tile_generator(slide, tile_size, overlap)
  # Generate tile.
  tile = np.asarray(generator.get_tile(zoom_level, (col, row)))
  return (slide_num, tile)


# Filter Tile For Dimensions & Tissue Threshold

def optical_density(tile):
  """
  Convert a tile to optical density values.

  Args:
    tile: A 3D NumPy array of shape (tile_size, tile_size, channels).

  Returns:
    A 3D NumPy array of shape (tile_size, tile_size, channels)
    representing optical density values.
  """
  tile = tile.astype(np.float64)
  #od = -np.log10(tile/255 + 1e-8)
  od = -np.log((tile+1)/240)
  return od


def keep_tile(tile_tuple, tile_size, tissue_threshold):
  """
  Determine if a tile should be kept.

  This filters out tiles based on size and a tissue percentage
  threshold, using a custom algorithm. If a tile has height &
  width equal to (tile_size, tile_size), and contains greater
  than or equal to the given percentage, then it will be kept;
  otherwise it will be filtered out.

  Args:
    tile_tuple: A (slide_num, tile) tuple, where slide_num is an
      integer, and tile is a 3D NumPy array of shape
      (tile_size, tile_size, channels).
    tile_size: The width and height of a square tile to be generated.
    tissue_threshold: Tissue percentage threshold.

  Returns:
    A Boolean indicating whether or not a tile should be kept for
    future usage.
  """
  slide_num, tile = tile_tuple
  if tile.shape[0:2] == (tile_size, tile_size):
    tile_orig = tile

    # Check 1
    # Convert 3D RGB image to 2D grayscale image, from
    # 0 (dense tissue) to 1 (plain background).
    tile = rgb2gray(tile)
    # 8-bit depth complement, from 1 (dense tissue)
    # to 0 (plain background).
    tile = 1 - tile
    # Canny edge detection with hysteresis thresholding.
    # This returns a binary map of edges, with 1 equal to
    # an edge. The idea is that tissue would be full of
    # edges, while background would not.
    tile = canny(tile)
    # Binary closing, which is a dilation followed by
    # an erosion. This removes small dark spots, which
    # helps remove noise in the background.
    tile = binary_closing(tile, disk(10))
    # Binary dilation, which enlarges bright areas,
    # and shrinks dark areas. This helps fill in holes
    # within regions of tissue.
    tile = binary_dilation(tile, disk(10))
    # Fill remaining holes within regions of tissue.
    tile = binary_fill_holes(tile)
    # Calculate percentage of tissue coverage.
    percentage = tile.mean()
    check1 = percentage >= tissue_threshold

    # Check 2
    # Convert to optical density values
    tile = optical_density(tile_orig)
    # Threshold at beta
    beta = 0.15
    tile = np.min(tile, axis=2) >= beta
    # Apply morphology for same reasons as above.
    tile = binary_closing(tile, disk(2))
    tile = binary_dilation(tile, disk(2))
    tile = binary_fill_holes(tile)
    percentage = tile.mean()
    check2 = percentage >= tissue_threshold

    return check1 and check2
  else:
    return False


# Generate Samples From Tile

def process_tile(tile_tuple, sample_size, grayscale):
  """
  Process a tile into a group of smaller samples.

  Cut up a tile into smaller blocks of sample_size x sample_size pixels,
  change the shape of each sample from (H, W, channels) to
  (channels, H, W), then flatten each into a vector of length
  channels*H*W.

  Args:
    tile_tuple: A (slide_num, tile) tuple, where slide_num is an
      integer, and tile is a 3D NumPy array of shape
      (tile_size, tile_size, channels).
    sample_size: The new width and height of the square samples to be
      generated.
    grayscale: Whether or not to generate grayscale samples, rather
      than RGB.

  Returns:
    A list of (slide_num, sample) tuples representing cut up tiles,
    where each sample is a 3D NumPy array of shape
    (sample_size_x, sample_size_y, channels).
  """
  slide_num, tile = tile_tuple
  if grayscale:
    tile = rgb2gray(tile)[:, :, np.newaxis]  # Grayscale
    # Save disk space and future IO time by converting from [0,1] to [0,255],
    # at the expense of some minor loss of information.
    tile = np.round(tile * 255).astype("uint8")
  x, y, ch = tile.shape
  # 1. Reshape into a 5D array of (num_x, sample_size_x, num_y, sample_size_y, ch), where
  # num_x and num_y are the number of chopped tiles on the x and y axes, respectively.
  # 2. Swap sample_size_x and num_y axes to create
  # (num_x, num_y, sample_size_x, sample_size_y, ch).
  # 3. Combine num_x and num_y into single axis, returning
  # (num_samples, sample_size_x, sample_size_y, ch).
  samples = (tile.reshape((x // sample_size, sample_size, y // sample_size, sample_size, ch))
                 .swapaxes(1,2)
                 .reshape((-1, sample_size, sample_size, ch)))
  samples = [(slide_num, sample) for sample in list(samples)]
  return samples


# Normalize staining

def normalize_staining(sample_tuple, beta=0.15, alpha=1, light_intensity=255):
  """
  Normalize the staining of H&E histology slides.

  This function normalizes the staining of H&E histology slides.

  References:
    - Macenko, Marc, et al. "A method for normalizing histology slides
    for quantitative analysis." Biomedical Imaging: From Nano to Macro,
    2009.  ISBI'09. IEEE International Symposium on. IEEE, 2009.
      - http://wwwx.cs.unc.edu/~mn/sites/default/files/macenko2009.pdf
    - https://github.com/mitkovetta/staining-normalization

  Args:
    sample_tuple: A (slide_num, sample) tuple, where slide_num is an
      integer, and sample is a 3D NumPy array of shape (H,W,C).

  Returns:
    A (slide_num, sample) tuple, where the sample is a 3D NumPy array
    of shape (H,W,C) that has been stain normalized.
  """
  # Setup.
  slide_num, sample = sample_tuple
  x = np.asarray(sample)
  h, w, c = x.shape
  x = x.reshape(-1, c).astype(np.float64)  # shape (H*W, C)

  # Reference stain vectors and stain saturations.  We will normalize all slides
  # to these references.  To create these, grab the stain vectors and stain
  # saturations from a desirable slide.

  # Values in reference implementation for use with eigendecomposition approach, natural log,
  # and `light_intensity=240`.
  #stain_ref = np.array([0.5626, 0.2159, 0.7201, 0.8012, 0.4062, 0.5581]).reshape(3,2)
  #max_sat_ref = np.array([1.9705, 1.0308]).reshape(2,1)

  # SVD w/ log10, and `light_intensity=255`.
  stain_ref = (np.array([0.54598845, 0.322116, 0.72385198, 0.76419107, 0.42182333, 0.55879629])
                 .reshape(3,2))
  max_sat_ref = np.array([0.82791151, 0.61137274]).reshape(2,1)

  # Convert RGB to OD.
  # Note: The original paper used log10, and the reference implementation used the natural log.
  #OD = -np.log((x+1)/light_intensity)  # shape (H*W, C)
  OD = -np.log10(x/light_intensity + 1e-8)

  # Remove data with OD intensity less than beta.
  # I.e. remove transparent pixels.
  # Note: This needs to be checked per channel, rather than
  # taking an average over all channels for a given pixel.
  OD_thresh = OD[np.all(OD >= beta, 1), :]  # shape (K, C)

  # Calculate eigenvectors.
  # Note: We can either use eigenvector decomposition, or SVD.
  #eigvals, eigvecs = np.linalg.eig(np.cov(OD_thresh.T))  # np.cov results in inf/nans
  U, s, V = np.linalg.svd(OD_thresh, full_matrices=False)

  # Extract two largest eigenvectors.
  # Note: We swap the sign of the eigvecs here to be consistent
  # with other implementations.  Both +/- eigvecs are valid, with
  # the same eigenvalue, so this is okay.
  #top_eigvecs = eigvecs[:, np.argsort(eigvals)[-2:]] * -1
  top_eigvecs = V[0:2, :].T * -1  # shape (C, 2)

  # Project thresholded optical density values onto plane spanned by
  # 2 largest eigenvectors.
  proj = np.dot(OD_thresh, top_eigvecs)  # shape (K, 2)

  # Calculate angle of each point wrt the first plane direction.
  # Note: the parameters are `np.arctan2(y, x)`
  angles = np.arctan2(proj[:, 1], proj[:, 0])  # shape (K,)

  # Find robust extremes (a and 100-a percentiles) of the angle.
  min_angle = np.percentile(angles, alpha)
  max_angle = np.percentile(angles, 100-alpha)

  # Convert min/max vectors (extremes) back to optimal stains in OD space.
  # This computes a set of axes for each angle onto which we can project
  # the top eigenvectors.  This assumes that the projected values have
  # been normalized to unit length.
  extreme_angles = np.array(
    [[np.cos(min_angle), np.cos(max_angle)],
     [np.sin(min_angle), np.sin(max_angle)]]
  )  # shape (2,2)
  stains = np.dot(top_eigvecs, extreme_angles)  # shape (C, 2)

  # Merge vectors with hematoxylin first, and eosin second, as a heuristic.
  if stains[0, 0] < stains[0, 1]:
    stains[:, [0, 1]] = stains[:, [1, 0]]  # swap columns

  # Calculate saturations of each stain.
  # Note: Here, we solve
  #    OD = VS
  #     S = V^{-1}OD
  # where `OD` is the matrix of optical density values of our image,
  # `V` is the matrix of stain vectors, and `S` is the matrix of stain
  # saturations.  Since this is an overdetermined system, we use the
  # least squares solver, rather than a direct solve.
  sats, _, _, _ = np.linalg.lstsq(stains, OD.T)

  # Normalize stain saturations to have same pseudo-maximum based on
  # a reference max saturation.
  max_sat = np.percentile(sats, 99, axis=1, keepdims=True)
  sats = sats / max_sat * max_sat_ref

  # Compute optimal OD values.
  OD_norm = np.dot(stain_ref, sats)

  # Recreate image.
  # Note: If the image is immediately converted to uint8 with `.astype(np.uint8)`, it will
  # not return the correct values due to the initital values being outside of [0,255].
  # To fix this, we round to the nearest integer, and then clip to [0,255], which is the
  # same behavior as Matlab.
  #x_norm = np.exp(OD_norm) * light_intensity  # natural log approach
  x_norm = 10**(-OD_norm) * light_intensity - 1e-8  # log10 approach
  x_norm = np.clip(np.round(x_norm), 0, 255).astype(np.uint8)
  x_norm = x_norm.astype(np.uint8)
  x_norm = x_norm.T.reshape(h,w,c)
  return (slide_num, x_norm)


def flatten_sample_tuple(sample_tuple):
  """
  Flatten a (H,W,C) sample into a (C*H*W) row vector.

  Transpose each sample from (H, W, channels) to (channels, H, W), then
  flatten each into a vector of length channels*H*W.

  Args:
    sample_tuple: A (slide_num, sample) tuple, where slide_num is an
      integer, and sample is a 3D NumPy array of shape (H,W,C).

  Returns:
    A (slide_num, sample) tuple, where the sample has been transposed
    from (H,W,C) to (C,H,W), and flattened to a vector of length
    (C*H*W).
  """
  slide_num, sample = sample_tuple
  # 1. Swap axes from (sample_size_x, sample_size_y, ch) to
  # (ch, sample_size_x, sample_size_y).
  # 2. Flatten sample into (ch*sample_size_x*sample_size_y).
  flattened_sample = sample.transpose(2,0,1).reshape(-1)
  return (slide_num, flattened_sample)

def flatten_sample(sample):
  """
    Flatten a (H,W,C) sample into a (C*H*W) row vector.

    Transpose each sample from (H, W, channels) to (channels, H, W), then
    flatten each into a vector of length channels*H*W.

    Args:
      sample: A sample, where slide_num is an
        integer, and sample is a 3D NumPy array of shape (H,W,C).

    Returns:
      A sample, where the sample has been transposed
      from (H,W,C) to (C,H,W), and flattened to a vector of length
      (C*H*W).
    """
  flattened_sample = sample.transpose(2, 0, 1).reshape(-1)
  return flattened_sample



# Get Ground Truth Labels

def get_labels_df(folder, filename="training_ground_truth.csv"):
  """
  Create a DataFrame with the ground truth labels for each slide.

  Args:
    folder: Directory containing a `training_ground_truth.csv` file
      containing the ground truth "tumor_score" and "molecular_score"
      labels for each slide.

  Returns:
    A Pandas DataFrame containing the ground truth labels for each
    slide.
  """
  filepath = os.path.join(folder, filename)
  labels_df = pd.read_csv(filepath, names=["tumor_score", "molecular_score"], header=None)
  labels_df["slide_num"] = labels_df.index + 1  # slide numbering starts at 1
  labels_df.set_index("slide_num", drop=False, inplace=True)  # use the slide num as index
  return labels_df


# Process All Slides Into A Spark RDD

def preprocess(spark, slide_nums, folder="data", training=True, tile_size=1024, overlap=0,
               tissue_threshold=0.9, sample_size=256, grayscale=False, normalize_stains=True,
               num_partitions=20000):
  """
  Preprocess a set of whole-slide images.

  Preprocess a set of whole-slide images as follows:
    1. Tile the slides into tiles of size (tile_size, tile_size, 3).
    2. Filter the tiles to remove unnecessary tissue.
    3. Cut the remaining tiles into samples of size
       (sample_size, sample_size, ch), where `ch` is 1 if `grayscale`
       is true, or 3 otherwise.

  Args:
    spark: SparkSession.
    slide_nums: List of whole-slide numbers to process.
    folder: Local directory in which the slides folder and ground truth
      file is stored, as a string. This should contain a
      `training_image_data` folder with images in the format
      `TUPAC-TR-###.svs`, as well as a `training_ground_truth.csv` file
      containing the ground truth "tumor_score" and "molecular_score"
      labels for each slide.  Alternatively, the folder should contain a
      `testing_image_data` folder with images in the format
      `TUPAC-TE-###.svs`.
    training: Boolean for training or testing datasets.
    tile_size: The width and height of a square tile to be generated.
    overlap: Number of pixels by which to overlap the tiles.
    tissue_threshold: Tissue percentage threshold for filtering.
    sample_size: The new width and height of the square samples to be
      generated.
    grayscale: Whether or not to generate grayscale samples, rather
      than RGB.
    normalize_stains: Whether or not to apply stain normalization.
    num_partitions: Number of partitions to use during processing.

  Returns:
    A Spark RDD in which, for training data sets, each element contains the slide number, tumor
    score, molecular score, and the sample as a numpy array; for validation data set,
    each element contains the slide number and the sample as a numpy array
  """
  # Filter out broken slides
  # Note: "Broken" here is due to a "version of OpenJPEG with broken support for chroma-subsampled
  # images".
  slides = (spark.sparkContext
      .parallelize(slide_nums)
      .filter(lambda slide: open_slide(slide, folder, training) is not None))

  # Create DataFrame of all tile locations and increase number of partitions
  # to avoid OOM during subsequent processing.
  tile_indices = (slides.flatMap(
      lambda slide: process_slide(slide, folder, training, tile_size, overlap)))
  # TODO: Explore computing the ideal paritition sizes based on projected number
  #   of tiles after filtering.  I.e. something like the following:
  #rows = tile_indices.count()
  #part_size = 128
  #channels = 1 if grayscale else 3
  #row_mb = tile_size * tile_size * channels * 8 / 1024 / 1024  # size of one row in MB
  #rows_per_part = round(part_size / row_mb)
  #num_parts = rows / rows_per_part
  tile_indices = tile_indices.repartition(num_partitions)
  tile_indices.cache()

  # Extract all tiles into an RDD, filter, cut into smaller samples, apply stain
  # normalization, and flatten.
  tiles = tile_indices.map(lambda tile_index: process_tile_index(tile_index, folder, training))
  filtered_tiles = tiles.filter(lambda tile: keep_tile(tile, tile_size, tissue_threshold))
  samples = filtered_tiles.flatMap(lambda tile: process_tile(tile, sample_size, grayscale))
  if normalize_stains:
    samples = samples.map(lambda sample: normalize_staining(sample))

  # Convert to a DataFrame
  if training:
    # Append labels
    labels_df = get_labels_df(folder)
    samples_with_labels = (samples.map(
        lambda tup: (int(tup[0]), int(labels_df.at[tup[0],"tumor_score"]),
                     float(labels_df.at[tup[0],"molecular_score"]), tup[1])))
    return samples_with_labels
  else:  # testing data -- no labels
    return samples


# Save DataFrame

def save_df(df, filepath, sample_size, grayscale, mode="error", format="parquet", file_size=128):
  """
  Save a preprocessed DataFrame with a constraint on the file sizes.

  Args:
    df: A Spark DataFrame.
    filepath: Hadoop-supported path at which to save `df`.
    sample_size: The width and height of the square samples.
    grayscale: Whether or not to the samples are in grayscale format,
      rather than RGB.
    mode: Specifies the behavior of `df.write.mode` when the data
      already exists.  Options include:
        * `append`: Append contents of this DataFrame to
          existing data.
        * `overwrite`: Overwrite existing data.
        * `error`: Throw an exception if data already exists.
        * `ignore`: Silently ignore this operation if data already
          exists.
    format: The format in which to save the DataFrame.
    file_size: Size in MB of each saved file.  128 MB is an
      empirically ideal size.
  """
  channels = 1 if grayscale else 3
  row_mb = sample_size * sample_size * channels * 8 / 1024 / 1024  # size of one row in MB
  rows_per_file = round(file_size / row_mb)
  df.write.option("maxRecordsPerFile", rows_per_file).mode(mode).save(filepath, format=format)


# Utilities

def add_row_indices(df, training=True):
  """
  Add a row index column for faster data ingestion times with SystemML.

  Args:
    df: A Spark DataFrame in which each row contains the slide number,
      tumor score, molecular score, and the sample stretched out into a
      Vector.
    training: Boolean for training or testing datasets.

  Returns:
    The Spark DataFrame with a row index column called "__INDEX".
  """
  rdd = (df.rdd
           .zipWithIndex()
           .map(lambda r: (r[1] + 1, *r[0])))  # flatten & convert index to 1-based indexing
  if training:
    df = rdd.toDF(['__INDEX', 'slide_num', 'tumor_score', 'molecular_score', 'sample'])
    df = df.select(df["__INDEX"].astype("int"), df.slide_num.astype("int"),
                   df.tumor_score.astype("int"), df.molecular_score, df["sample"])
  else:  # testing data -- no labels
    df = rdd.toDF(["__INDEX", "slide_num", "sample"])
    df = df.select(df["__INDEX"].astype("int"), df.slide_num.astype("int"), df["sample"])
  return df


def sample(df, frac, training=True, seed=None):
  """
  Sample the DataFrame, stratified on the class.

  Args:
    df: A Spark DataFrame in which each row contains the slide number,
      tumor score, molecular score, and the sample stretched out into a
      Vector.
    frac: Fraction of rows to keep.
    training: Boolean for training or testing datasets.
    seed: Random seed used for the sampling.

  Returns:
    A stratified sample of the original Spark DataFrame.
  """
  df_sample = df.sampleBy("tumor_score", fractions={1: frac, 2: frac, 3: frac}, seed=seed)
  return df_sample


def rdd_2_df(rdd, training=True):
  """
  Convert the sample RDD to a Spark DataFrame.

  Args:
    rdd: A sample RDD with or without labels; for the RDD with labels,
      each element will be (slide_num, tumor_score, molecular_score,
      sample); for the RDD witout labels, each element will be (slide_num,
      sample).
    training: Boolean for training or testing datasets.

  Returns:
    A Spark DataFrame in which each row contains the slide number, tumor
    score, molecular score, and the sample stretched out into a Vector.
  """
  if training:
    # Append labels
    samples_with_labels = (rdd.map(
        lambda tup: (tup[0], tup[1], tup[2], Vectors.dense(flatten_sample(tup[3])))))
    df = samples_with_labels.toDF(["slide_num", "tumor_score", "molecular_score", "sample"])
    df = df.select(df.slide_num.astype("int"), df.tumor_score.astype("int"),
                     df.molecular_score, df["sample"])
  else:  # testing data -- no labels
    samples = (rdd.map(lambda tup: (tup[0], Vectors.dense(flatten_sample(tup[1])))))
    df = samples.toDF(["slide_num", "sample"])
    df = df.select(df.slide_num.astype("int"), df["sample"])
  return df


def save_rdd_2_jpeg(rdd, save_dir):
  """
  Save the Spark RDD into JPEG

  Args:
    rdd: the spark RDD with or without labels. The RDD element could be
     a tuple with labels (slide_num, tumor_score, molecular_score, sample)
     or a tuple without labels, e.g. (slide_num, sample)
    save_dir: the file directory at which to save JPEGs
  """
  rdd.foreach(lambda sample_element: save_2_jpeg(sample_element, save_dir))


def save_2_jpeg(sample_element, save_dir):
  """
  Save the sample with or without labels into JPEG

  Args:
    sample_element: it may be a sample tuple with labels, e.g. (slide_num, tumor_score, molecular_score, sample)
      or a sample tuple without labels, e.g. (slide_num, sample)
    save_dir: the file directory at which to save JPEGs
  """
  if len(sample_element) == 4: # for the sample element with labels
    save_labelled_sample_2_jpeg(sample_element, save_dir)
  elif len(sample_element) == 2: # for the sample element without labels
    save_nonlabelled_sample_2_jpeg(sample_element, save_dir)
  else:
    raise ValueError("This type of sample_element is not supported yet")


def save_nonlabelled_sample_2_jpeg(sample, save_dir):
  """
  Save the sample without labels into JPEG

  Args:
    sample_element: a sample tuple without labels, e.g. (slide_num, sample)
    save_dir: the file directory at which to save JPEGs
  """
  slide_num, img_value = sample
  filename = '{slide_num}_{hash}.jpeg'.format(
    slide_num=slide_num, hash=np.random.randint(1e4))
  filepath = os.path.join(save_dir, filename)
  save_jpeg_help(img_value, filepath)


def save_labelled_sample_2_jpeg(sample_with_label, save_dir):
  """
  Save the sample with labels into JPEG

  Args:
    sample_element: a sample tuple with labels, e.g. (slide_num, tumor_score, molecular_score, sample)
    save_dir: the file directory at which to save JPEGs
  """
  slide_num, tumor_score, molecular_score, img_value = sample_with_label
  filename = '{slide_num}_{hash}.jpeg'.format(
      slide_num=slide_num, hash=np.random.randint(1e4))
  class_dir = os.path.join(save_dir, str(tumor_score))
  filepath = os.path.join(class_dir, filename)
  save_jpeg_help(img_value, filepath)


def save_jpeg_help(img_value, filepath):
  """
   Save data into JPEG

   Args:
     img_value: the image value with the size (img_size_x, img_size_y, channels)
     file path: the file path at which to save JPEGs
   """
  dir = os.path.dirname(filepath)
  os.makedirs(dir, exist_ok=True)
  img = Image.fromarray(img_value.astype(np.uint8), 'RGB')
  img.save(filepath)

