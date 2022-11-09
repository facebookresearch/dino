# ------------------------------------------------------------------------
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ------------------------------------------------------------------------

import math
import multiprocessing
import numpy as np
import os
import scipy.ndimage.morphology as sc_morph
import skimage.color as sk_color
import skimage.exposure as sk_exposure
import skimage.feature as sk_feature
import skimage.filters as sk_filters
import skimage.future as sk_future
import skimage.morphology as sk_morphology
import skimage.segmentation as sk_segmentation

from deephistopath.wsi import slide
from deephistopath.wsi import util
from deephistopath.wsi.util import Time


def filter_rgb_to_grayscale(np_img, output_type="uint8"):
  """
  Convert an RGB NumPy array to a grayscale NumPy array.

  Shape (h, w, c) to (h, w).

  Args:
    np_img: RGB Image as a NumPy array.
    output_type: Type of array to return (float or uint8)

  Returns:
    Grayscale image as NumPy array with shape (h, w).
  """
  t = Time()
  # Another common RGB ratio possibility: [0.299, 0.587, 0.114]
  grayscale = np.dot(np_img[..., :3], [0.2125, 0.7154, 0.0721])
  if output_type != "float":
    grayscale = grayscale.astype("uint8")
  util.np_info(grayscale, "Gray", t.elapsed())
  return grayscale


def filter_complement(np_img, output_type="uint8"):
  """
  Obtain the complement of an image as a NumPy array.

  Args:
    np_img: Image as a NumPy array.
    type: Type of array to return (float or uint8).

  Returns:
    Complement image as Numpy array.
  """
  t = Time()
  if output_type == "float":
    complement = 1.0 - np_img
  else:
    complement = 255 - np_img
  util.np_info(complement, "Complement", t.elapsed())
  return complement


def filter_hysteresis_threshold(np_img, low=50, high=100, output_type="uint8"):
  """
  Apply two-level (hysteresis) threshold to an image as a NumPy array, returning a binary image.

  Args:
    np_img: Image as a NumPy array.
    low: Low threshold.
    high: High threshold.
    output_type: Type of array to return (bool, float, or uint8).

  Returns:
    NumPy array (bool, float, or uint8) where True, 1.0, and 255 represent a pixel above hysteresis threshold.
  """
  t = Time()
  hyst = sk_filters.apply_hysteresis_threshold(np_img, low, high)
  if output_type == "bool":
    pass
  elif output_type == "float":
    hyst = hyst.astype(float)
  else:
    hyst = (255 * hyst).astype("uint8")
  util.np_info(hyst, "Hysteresis Threshold", t.elapsed())
  return hyst


def filter_otsu_threshold(np_img, output_type="uint8"):
  """
  Compute Otsu threshold on image as a NumPy array and return binary image based on pixels above threshold.

  Args:
    np_img: Image as a NumPy array.
    output_type: Type of array to return (bool, float, or uint8).

  Returns:
    NumPy array (bool, float, or uint8) where True, 1.0, and 255 represent a pixel above Otsu threshold.
  """
  t = Time()
  otsu_thresh_value = sk_filters.threshold_otsu(np_img)
  otsu = (np_img > otsu_thresh_value)
  if output_type == "bool":
    pass
  elif output_type == "float":
    otsu = otsu.astype(float)
  else:
    otsu = otsu.astype("uint8") * 255
  util.np_info(otsu, "Otsu Threshold", t.elapsed())
  return otsu


def filter_local_otsu_threshold(np_img, disk_size=3, output_type="uint8"):
  """
  Compute local Otsu threshold for each pixel and return binary image based on pixels being less than the
  local Otsu threshold.

  Args:
    np_img: Image as a NumPy array.
    disk_size: Radius of the disk structuring element used to compute the Otsu threshold for each pixel.
    output_type: Type of array to return (bool, float, or uint8).

  Returns:
    NumPy array (bool, float, or uint8) where local Otsu threshold values have been applied to original image.
  """
  t = Time()
  local_otsu = sk_filters.rank.otsu(np_img, sk_morphology.disk(disk_size))
  if output_type == "bool":
    pass
  elif output_type == "float":
    local_otsu = local_otsu.astype(float)
  else:
    local_otsu = local_otsu.astype("uint8") * 255
  util.np_info(local_otsu, "Otsu Local Threshold", t.elapsed())
  return local_otsu


def filter_entropy(np_img, neighborhood=9, threshold=5, output_type="uint8"):
  """
  Filter image based on entropy (complexity).

  Args:
    np_img: Image as a NumPy array.
    neighborhood: Neighborhood size (defines height and width of 2D array of 1's).
    threshold: Threshold value.
    output_type: Type of array to return (bool, float, or uint8).

  Returns:
    NumPy array (bool, float, or uint8) where True, 1.0, and 255 represent a measure of complexity.
  """
  t = Time()
  entr = sk_filters.rank.entropy(np_img, np.ones((neighborhood, neighborhood))) > threshold
  if output_type == "bool":
    pass
  elif output_type == "float":
    entr = entr.astype(float)
  else:
    entr = entr.astype("uint8") * 255
  util.np_info(entr, "Entropy", t.elapsed())
  return entr


def filter_canny(np_img, sigma=1, low_threshold=0, high_threshold=25, output_type="uint8"):
  """
  Filter image based on Canny algorithm edges.

  Args:
    np_img: Image as a NumPy array.
    sigma: Width (std dev) of Gaussian.
    low_threshold: Low hysteresis threshold value.
    high_threshold: High hysteresis threshold value.
    output_type: Type of array to return (bool, float, or uint8).

  Returns:
    NumPy array (bool, float, or uint8) representing Canny edge map (binary image).
  """
  t = Time()
  can = sk_feature.canny(np_img, sigma=sigma, low_threshold=low_threshold, high_threshold=high_threshold)
  if output_type == "bool":
    pass
  elif output_type == "float":
    can = can.astype(float)
  else:
    can = can.astype("uint8") * 255
  util.np_info(can, "Canny Edges", t.elapsed())
  return can


def mask_percent(np_img):
  """
  Determine the percentage of a NumPy array that is masked (how many of the values are 0 values).

  Args:
    np_img: Image as a NumPy array.

  Returns:
    The percentage of the NumPy array that is masked.
  """
  if (len(np_img.shape) == 3) and (np_img.shape[2] == 3):
    np_sum = np_img[:, :, 0] + np_img[:, :, 1] + np_img[:, :, 2]
    mask_percentage = 100 - np.count_nonzero(np_sum) / np_sum.size * 100
  else:
    mask_percentage = 100 - np.count_nonzero(np_img) / np_img.size * 100
  return mask_percentage


def tissue_percent(np_img):
  """
  Determine the percentage of a NumPy array that is tissue (not masked).

  Args:
    np_img: Image as a NumPy array.

  Returns:
    The percentage of the NumPy array that is tissue.
  """
  return 100 - mask_percent(np_img)


def filter_remove_small_objects(np_img, min_size=3000, avoid_overmask=True, overmask_thresh=95, output_type="uint8"):
  """
  Filter image to remove small objects (connected components) less than a particular minimum size. If avoid_overmask
  is True, this function can recursively call itself with progressively smaller minimum size objects to remove to
  reduce the amount of masking that this filter performs.

  Args:
    np_img: Image as a NumPy array of type bool.
    min_size: Minimum size of small object to remove.
    avoid_overmask: If True, avoid masking above the overmask_thresh percentage.
    overmask_thresh: If avoid_overmask is True, avoid masking above this threshold percentage value.
    output_type: Type of array to return (bool, float, or uint8).

  Returns:
    NumPy array (bool, float, or uint8).
  """
  t = Time()

  rem_sm = np_img.astype(bool)  # make sure mask is boolean
  rem_sm = sk_morphology.remove_small_objects(rem_sm, min_size=min_size)
  mask_percentage = mask_percent(rem_sm)
  if (mask_percentage >= overmask_thresh) and (min_size >= 1) and (avoid_overmask is True):
    new_min_size = min_size / 2
    print("Mask percentage %3.2f%% >= overmask threshold %3.2f%% for Remove Small Objs size %d, so try %d" % (
      mask_percentage, overmask_thresh, min_size, new_min_size))
    rem_sm = filter_remove_small_objects(np_img, new_min_size, avoid_overmask, overmask_thresh, output_type)
  np_img = rem_sm

  if output_type == "bool":
    pass
  elif output_type == "float":
    np_img = np_img.astype(float)
  else:
    np_img = np_img.astype("uint8") * 255

  util.np_info(np_img, "Remove Small Objs", t.elapsed())
  return np_img


def filter_remove_small_holes(np_img, min_size=3000, output_type="uint8"):
  """
  Filter image to remove small holes less than a particular size.

  Args:
    np_img: Image as a NumPy array of type bool.
    min_size: Remove small holes below this size.
    output_type: Type of array to return (bool, float, or uint8).

  Returns:
    NumPy array (bool, float, or uint8).
  """
  t = Time()

  rem_sm = sk_morphology.remove_small_holes(np_img, min_size=min_size)

  if output_type == "bool":
    pass
  elif output_type == "float":
    rem_sm = rem_sm.astype(float)
  else:
    rem_sm = rem_sm.astype("uint8") * 255

  util.np_info(rem_sm, "Remove Small Holes", t.elapsed())
  return rem_sm


def filter_contrast_stretch(np_img, low=40, high=60):
  """
  Filter image (gray or RGB) using contrast stretching to increase contrast in image based on the intensities in
  a specified range.

  Args:
    np_img: Image as a NumPy array (gray or RGB).
    low: Range low value (0 to 255).
    high: Range high value (0 to 255).

  Returns:
    Image as NumPy array with contrast enhanced.
  """
  t = Time()
  low_p, high_p = np.percentile(np_img, (low * 100 / 255, high * 100 / 255))
  contrast_stretch = sk_exposure.rescale_intensity(np_img, in_range=(low_p, high_p))
  util.np_info(contrast_stretch, "Contrast Stretch", t.elapsed())
  return contrast_stretch


def filter_histogram_equalization(np_img, nbins=256, output_type="uint8"):
  """
  Filter image (gray or RGB) using histogram equalization to increase contrast in image.

  Args:
    np_img: Image as a NumPy array (gray or RGB).
    nbins: Number of histogram bins.
    output_type: Type of array to return (float or uint8).

  Returns:
     NumPy array (float or uint8) with contrast enhanced by histogram equalization.
  """
  t = Time()
  # if uint8 type and nbins is specified, convert to float so that nbins can be a value besides 256
  if np_img.dtype == "uint8" and nbins != 256:
    np_img = np_img / 255
  hist_equ = sk_exposure.equalize_hist(np_img, nbins=nbins)
  if output_type == "float":
    pass
  else:
    hist_equ = (hist_equ * 255).astype("uint8")
  util.np_info(hist_equ, "Hist Equalization", t.elapsed())
  return hist_equ


def filter_adaptive_equalization(np_img, nbins=256, clip_limit=0.01, output_type="uint8"):
  """
  Filter image (gray or RGB) using adaptive equalization to increase contrast in image, where contrast in local regions
  is enhanced.

  Args:
    np_img: Image as a NumPy array (gray or RGB).
    nbins: Number of histogram bins.
    clip_limit: Clipping limit where higher value increases contrast.
    output_type: Type of array to return (float or uint8).

  Returns:
     NumPy array (float or uint8) with contrast enhanced by adaptive equalization.
  """
  t = Time()
  adapt_equ = sk_exposure.equalize_adapthist(np_img, nbins=nbins, clip_limit=clip_limit)
  if output_type == "float":
    pass
  else:
    adapt_equ = (adapt_equ * 255).astype("uint8")
  util.np_info(adapt_equ, "Adapt Equalization", t.elapsed())
  return adapt_equ


def filter_local_equalization(np_img, disk_size=50):
  """
  Filter image (gray) using local equalization, which uses local histograms based on the disk structuring element.

  Args:
    np_img: Image as a NumPy array.
    disk_size: Radius of the disk structuring element used for the local histograms

  Returns:
    NumPy array with contrast enhanced using local equalization.
  """
  t = Time()
  local_equ = sk_filters.rank.equalize(np_img, selem=sk_morphology.disk(disk_size))
  util.np_info(local_equ, "Local Equalization", t.elapsed())
  return local_equ


def filter_rgb_to_hed(np_img, output_type="uint8"):
  """
  Filter RGB channels to HED (Hematoxylin - Eosin - Diaminobenzidine) channels.

  Args:
    np_img: RGB image as a NumPy array.
    output_type: Type of array to return (float or uint8).

  Returns:
    NumPy array (float or uint8) with HED channels.
  """
  t = Time()
  hed = sk_color.rgb2hed(np_img)
  if output_type == "float":
    hed = sk_exposure.rescale_intensity(hed, out_range=(0.0, 1.0))
  else:
    hed = (sk_exposure.rescale_intensity(hed, out_range=(0, 255))).astype("uint8")

  util.np_info(hed, "RGB to HED", t.elapsed())
  return hed


def filter_rgb_to_hsv(np_img, display_np_info=True):
  """
  Filter RGB channels to HSV (Hue, Saturation, Value).

  Args:
    np_img: RGB image as a NumPy array.
    display_np_info: If True, display NumPy array info and filter time.

  Returns:
    Image as NumPy array in HSV representation.
  """

  if display_np_info:
    t = Time()
  hsv = sk_color.rgb2hsv(np_img)
  if display_np_info:
    util.np_info(hsv, "RGB to HSV", t.elapsed())
  return hsv


def filter_hsv_to_h(hsv, output_type="int", display_np_info=True):
  """
  Obtain hue values from HSV NumPy array as a 1-dimensional array. If output as an int array, the original float
  values are multiplied by 360 for their degree equivalents for simplicity. For more information, see
  https://en.wikipedia.org/wiki/HSL_and_HSV

  Args:
    hsv: HSV image as a NumPy array.
    output_type: Type of array to return (float or int).
    display_np_info: If True, display NumPy array info and filter time.

  Returns:
    Hue values (float or int) as a 1-dimensional NumPy array.
  """
  if display_np_info:
    t = Time()
  h = hsv[:, :, 0]
  h = h.flatten()
  if output_type == "int":
    h *= 360
    h = h.astype("int")
  if display_np_info:
    util.np_info(hsv, "HSV to H", t.elapsed())
  return h


def filter_hsv_to_s(hsv):
  """
  Experimental HSV to S (saturation).

  Args:
    hsv:  HSV image as a NumPy array.

  Returns:
    Saturation values as a 1-dimensional NumPy array.
  """
  s = hsv[:, :, 1]
  s = s.flatten()
  return s


def filter_hsv_to_v(hsv):
  """
  Experimental HSV to V (value).

  Args:
    hsv:  HSV image as a NumPy array.

  Returns:
    Value values as a 1-dimensional NumPy array.
  """
  v = hsv[:, :, 2]
  v = v.flatten()
  return v


def filter_hed_to_hematoxylin(np_img, output_type="uint8"):
  """
  Obtain Hematoxylin channel from HED NumPy array and rescale it (for example, to 0 to 255 for uint8) for increased
  contrast.

  Args:
    np_img: HED image as a NumPy array.
    output_type: Type of array to return (float or uint8).

  Returns:
    NumPy array for Hematoxylin channel.
  """
  t = Time()
  hema = np_img[:, :, 0]
  if output_type == "float":
    hema = sk_exposure.rescale_intensity(hema, out_range=(0.0, 1.0))
  else:
    hema = (sk_exposure.rescale_intensity(hema, out_range=(0, 255))).astype("uint8")
  util.np_info(hema, "HED to Hematoxylin", t.elapsed())
  return hema


def filter_hed_to_eosin(np_img, output_type="uint8"):
  """
  Obtain Eosin channel from HED NumPy array and rescale it (for example, to 0 to 255 for uint8) for increased
  contrast.

  Args:
    np_img: HED image as a NumPy array.
    output_type: Type of array to return (float or uint8).

  Returns:
    NumPy array for Eosin channel.
  """
  t = Time()
  eosin = np_img[:, :, 1]
  if output_type == "float":
    eosin = sk_exposure.rescale_intensity(eosin, out_range=(0.0, 1.0))
  else:
    eosin = (sk_exposure.rescale_intensity(eosin, out_range=(0, 255))).astype("uint8")
  util.np_info(eosin, "HED to Eosin", t.elapsed())
  return eosin


def filter_binary_fill_holes(np_img, output_type="bool"):
  """
  Fill holes in a binary object (bool, float, or uint8).

  Args:
    np_img: Binary image as a NumPy array.
    output_type: Type of array to return (bool, float, or uint8).

  Returns:
    NumPy array (bool, float, or uint8) where holes have been filled.
  """
  t = Time()
  if np_img.dtype == "uint8":
    np_img = np_img / 255
  result = sc_morph.binary_fill_holes(np_img)
  if output_type == "bool":
    pass
  elif output_type == "float":
    result = result.astype(float)
  else:
    result = result.astype("uint8") * 255
  util.np_info(result, "Binary Fill Holes", t.elapsed())
  return result


def filter_binary_erosion(np_img, disk_size=5, iterations=1, output_type="uint8"):
  """
  Erode a binary object (bool, float, or uint8).

  Args:
    np_img: Binary image as a NumPy array.
    disk_size: Radius of the disk structuring element used for erosion.
    iterations: How many times to repeat the erosion.
    output_type: Type of array to return (bool, float, or uint8).

  Returns:
    NumPy array (bool, float, or uint8) where edges have been eroded.
  """
  t = Time()
  if np_img.dtype == "uint8":
    np_img = np_img / 255
  result = sc_morph.binary_erosion(np_img, sk_morphology.disk(disk_size), iterations=iterations)
  if output_type == "bool":
    pass
  elif output_type == "float":
    result = result.astype(float)
  else:
    result = result.astype("uint8") * 255
  util.np_info(result, "Binary Erosion", t.elapsed())
  return result


def filter_binary_dilation(np_img, disk_size=5, iterations=1, output_type="uint8"):
  """
  Dilate a binary object (bool, float, or uint8).

  Args:
    np_img: Binary image as a NumPy array.
    disk_size: Radius of the disk structuring element used for dilation.
    iterations: How many times to repeat the dilation.
    output_type: Type of array to return (bool, float, or uint8).

  Returns:
    NumPy array (bool, float, or uint8) where edges have been dilated.
  """
  t = Time()
  if np_img.dtype == "uint8":
    np_img = np_img / 255
  result = sc_morph.binary_dilation(np_img, sk_morphology.disk(disk_size), iterations=iterations)
  if output_type == "bool":
    pass
  elif output_type == "float":
    result = result.astype(float)
  else:
    result = result.astype("uint8") * 255
  util.np_info(result, "Binary Dilation", t.elapsed())
  return result


def filter_binary_opening(np_img, disk_size=3, iterations=1, output_type="uint8"):
  """
  Open a binary object (bool, float, or uint8). Opening is an erosion followed by a dilation.
  Opening can be used to remove small objects.

  Args:
    np_img: Binary image as a NumPy array.
    disk_size: Radius of the disk structuring element used for opening.
    iterations: How many times to repeat.
    output_type: Type of array to return (bool, float, or uint8).

  Returns:
    NumPy array (bool, float, or uint8) following binary opening.
  """
  t = Time()
  if np_img.dtype == "uint8":
    np_img = np_img / 255
  result = sc_morph.binary_opening(np_img, sk_morphology.disk(disk_size), iterations=iterations)
  if output_type == "bool":
    pass
  elif output_type == "float":
    result = result.astype(float)
  else:
    result = result.astype("uint8") * 255
  util.np_info(result, "Binary Opening", t.elapsed())
  return result


def filter_binary_closing(np_img, disk_size=3, iterations=1, output_type="uint8"):
  """
  Close a binary object (bool, float, or uint8). Closing is a dilation followed by an erosion.
  Closing can be used to remove small holes.

  Args:
    np_img: Binary image as a NumPy array.
    disk_size: Radius of the disk structuring element used for closing.
    iterations: How many times to repeat.
    output_type: Type of array to return (bool, float, or uint8).

  Returns:
    NumPy array (bool, float, or uint8) following binary closing.
  """
  t = Time()
  if np_img.dtype == "uint8":
    np_img = np_img / 255
  result = sc_morph.binary_closing(np_img, sk_morphology.disk(disk_size), iterations=iterations)
  if output_type == "bool":
    pass
  elif output_type == "float":
    result = result.astype(float)
  else:
    result = result.astype("uint8") * 255
  util.np_info(result, "Binary Closing", t.elapsed())
  return result


def filter_kmeans_segmentation(np_img, compactness=10, n_segments=800):
  """
  Use K-means segmentation (color/space proximity) to segment RGB image where each segment is
  colored based on the average color for that segment.

  Args:
    np_img: Binary image as a NumPy array.
    compactness: Color proximity versus space proximity factor.
    n_segments: The number of segments.

  Returns:
    NumPy array (uint8) representing 3-channel RGB image where each segment has been colored based on the average
    color for that segment.
  """
  t = Time()
  labels = sk_segmentation.slic(np_img, compactness=compactness, n_segments=n_segments)
  result = sk_color.label2rgb(labels, np_img, kind='avg')
  util.np_info(result, "K-Means Segmentation", t.elapsed())
  return result


def filter_rag_threshold(np_img, compactness=10, n_segments=800, threshold=9):
  """
  Use K-means segmentation to segment RGB image, build region adjacency graph based on the segments, combine
  similar regions based on threshold value, and then output these resulting region segments.

  Args:
    np_img: Binary image as a NumPy array.
    compactness: Color proximity versus space proximity factor.
    n_segments: The number of segments.
    threshold: Threshold value for combining regions.

  Returns:
    NumPy array (uint8) representing 3-channel RGB image where each segment has been colored based on the average
    color for that segment (and similar segments have been combined).
  """
  t = Time()
  labels = sk_segmentation.slic(np_img, compactness=compactness, n_segments=n_segments)
  g = sk_future.graph.rag_mean_color(np_img, labels)
  labels2 = sk_future.graph.cut_threshold(labels, g, threshold)
  result = sk_color.label2rgb(labels2, np_img, kind='avg')
  util.np_info(result, "RAG Threshold", t.elapsed())
  return result


def filter_threshold(np_img, threshold, output_type="bool"):
  """
  Return mask where a pixel has a value if it exceeds the threshold value.

  Args:
    np_img: Binary image as a NumPy array.
    threshold: The threshold value to exceed.
    output_type: Type of array to return (bool, float, or uint8).

  Returns:
    NumPy array representing a mask where a pixel has a value (T, 1.0, or 255) if the corresponding input array
    pixel exceeds the threshold value.
  """
  t = Time()
  result = (np_img > threshold)
  if output_type == "bool":
    pass
  elif output_type == "float":
    result = result.astype(float)
  else:
    result = result.astype("uint8") * 255
  util.np_info(result, "Threshold", t.elapsed())
  return result


def filter_green_channel(np_img, green_thresh=200, avoid_overmask=True, overmask_thresh=90, output_type="bool"):
  """
  Create a mask to filter out pixels with a green channel value greater than a particular threshold, since hematoxylin
  and eosin are purplish and pinkish, which do not have much green to them.

  Args:
    np_img: RGB image as a NumPy array.
    green_thresh: Green channel threshold value (0 to 255). If value is greater than green_thresh, mask out pixel.
    avoid_overmask: If True, avoid masking above the overmask_thresh percentage.
    overmask_thresh: If avoid_overmask is True, avoid masking above this threshold percentage value.
    output_type: Type of array to return (bool, float, or uint8).

  Returns:
    NumPy array representing a mask where pixels above a particular green channel threshold have been masked out.
  """
  t = Time()

  g = np_img[:, :, 1]
  gr_ch_mask = (g < green_thresh) & (g > 0)
  mask_percentage = mask_percent(gr_ch_mask)
  if (mask_percentage >= overmask_thresh) and (green_thresh < 255) and (avoid_overmask is True):
    new_green_thresh = math.ceil((255 - green_thresh) / 2 + green_thresh)
    print(
      "Mask percentage %3.2f%% >= overmask threshold %3.2f%% for Remove Green Channel green_thresh=%d, so try %d" % (
        mask_percentage, overmask_thresh, green_thresh, new_green_thresh))
    gr_ch_mask = filter_green_channel(np_img, new_green_thresh, avoid_overmask, overmask_thresh, output_type)
  np_img = gr_ch_mask

  if output_type == "bool":
    pass
  elif output_type == "float":
    np_img = np_img.astype(float)
  else:
    np_img = np_img.astype("uint8") * 255

  util.np_info(np_img, "Filter Green Channel", t.elapsed())
  return np_img


def filter_red(rgb, red_lower_thresh, green_upper_thresh, blue_upper_thresh, output_type="bool",
               display_np_info=False):
  """
  Create a mask to filter out reddish colors, where the mask is based on a pixel being above a
  red channel threshold value, below a green channel threshold value, and below a blue channel threshold value.

  Args:
    rgb: RGB image as a NumPy array.
    red_lower_thresh: Red channel lower threshold value.
    green_upper_thresh: Green channel upper threshold value.
    blue_upper_thresh: Blue channel upper threshold value.
    output_type: Type of array to return (bool, float, or uint8).
    display_np_info: If True, display NumPy array info and filter time.

  Returns:
    NumPy array representing the mask.
  """
  if display_np_info:
    t = Time()
  r = rgb[:, :, 0] > red_lower_thresh
  g = rgb[:, :, 1] < green_upper_thresh
  b = rgb[:, :, 2] < blue_upper_thresh
  result = ~(r & g & b)
  if output_type == "bool":
    pass
  elif output_type == "float":
    result = result.astype(float)
  else:
    result = result.astype("uint8") * 255
  if display_np_info:
    util.np_info(result, "Filter Red", t.elapsed())
  return result


def filter_red_pen(rgb, output_type="bool"):
  """
  Create a mask to filter out red pen marks from a slide.

  Args:
    rgb: RGB image as a NumPy array.
    output_type: Type of array to return (bool, float, or uint8).

  Returns:
    NumPy array representing the mask.
  """
  t = Time()
  result = filter_red(rgb, red_lower_thresh=150, green_upper_thresh=80, blue_upper_thresh=90) & \
           filter_red(rgb, red_lower_thresh=110, green_upper_thresh=20, blue_upper_thresh=30) & \
           filter_red(rgb, red_lower_thresh=185, green_upper_thresh=65, blue_upper_thresh=105) & \
           filter_red(rgb, red_lower_thresh=195, green_upper_thresh=85, blue_upper_thresh=125) & \
           filter_red(rgb, red_lower_thresh=220, green_upper_thresh=115, blue_upper_thresh=145) & \
           filter_red(rgb, red_lower_thresh=125, green_upper_thresh=40, blue_upper_thresh=70) & \
           filter_red(rgb, red_lower_thresh=200, green_upper_thresh=120, blue_upper_thresh=150) & \
           filter_red(rgb, red_lower_thresh=100, green_upper_thresh=50, blue_upper_thresh=65) & \
           filter_red(rgb, red_lower_thresh=85, green_upper_thresh=25, blue_upper_thresh=45)
  if output_type == "bool":
    pass
  elif output_type == "float":
    result = result.astype(float)
  else:
    result = result.astype("uint8") * 255
  util.np_info(result, "Filter Red Pen", t.elapsed())
  return result


def filter_green(rgb, red_upper_thresh, green_lower_thresh, blue_lower_thresh, output_type="bool",
                 display_np_info=False):
  """
  Create a mask to filter out greenish colors, where the mask is based on a pixel being below a
  red channel threshold value, above a green channel threshold value, and above a blue channel threshold value.
  Note that for the green ink, the green and blue channels tend to track together, so we use a blue channel
  lower threshold value rather than a blue channel upper threshold value.

  Args:
    rgb: RGB image as a NumPy array.
    red_upper_thresh: Red channel upper threshold value.
    green_lower_thresh: Green channel lower threshold value.
    blue_lower_thresh: Blue channel lower threshold value.
    output_type: Type of array to return (bool, float, or uint8).
    display_np_info: If True, display NumPy array info and filter time.

  Returns:
    NumPy array representing the mask.
  """
  if display_np_info:
    t = Time()
  r = rgb[:, :, 0] < red_upper_thresh
  g = rgb[:, :, 1] > green_lower_thresh
  b = rgb[:, :, 2] > blue_lower_thresh
  result = ~(r & g & b)
  if output_type == "bool":
    pass
  elif output_type == "float":
    result = result.astype(float)
  else:
    result = result.astype("uint8") * 255
  if display_np_info:
    util.np_info(result, "Filter Green", t.elapsed())
  return result


def filter_green_pen(rgb, output_type="bool"):
  """
  Create a mask to filter out green pen marks from a slide.

  Args:
    rgb: RGB image as a NumPy array.
    output_type: Type of array to return (bool, float, or uint8).

  Returns:
    NumPy array representing the mask.
  """
  t = Time()
  result = filter_green(rgb, red_upper_thresh=150, green_lower_thresh=160, blue_lower_thresh=140) & \
           filter_green(rgb, red_upper_thresh=70, green_lower_thresh=110, blue_lower_thresh=110) & \
           filter_green(rgb, red_upper_thresh=45, green_lower_thresh=115, blue_lower_thresh=100) & \
           filter_green(rgb, red_upper_thresh=30, green_lower_thresh=75, blue_lower_thresh=60) & \
           filter_green(rgb, red_upper_thresh=195, green_lower_thresh=220, blue_lower_thresh=210) & \
           filter_green(rgb, red_upper_thresh=225, green_lower_thresh=230, blue_lower_thresh=225) & \
           filter_green(rgb, red_upper_thresh=170, green_lower_thresh=210, blue_lower_thresh=200) & \
           filter_green(rgb, red_upper_thresh=20, green_lower_thresh=30, blue_lower_thresh=20) & \
           filter_green(rgb, red_upper_thresh=50, green_lower_thresh=60, blue_lower_thresh=40) & \
           filter_green(rgb, red_upper_thresh=30, green_lower_thresh=50, blue_lower_thresh=35) & \
           filter_green(rgb, red_upper_thresh=65, green_lower_thresh=70, blue_lower_thresh=60) & \
           filter_green(rgb, red_upper_thresh=100, green_lower_thresh=110, blue_lower_thresh=105) & \
           filter_green(rgb, red_upper_thresh=165, green_lower_thresh=180, blue_lower_thresh=180) & \
           filter_green(rgb, red_upper_thresh=140, green_lower_thresh=140, blue_lower_thresh=150) & \
           filter_green(rgb, red_upper_thresh=185, green_lower_thresh=195, blue_lower_thresh=195)
  if output_type == "bool":
    pass
  elif output_type == "float":
    result = result.astype(float)
  else:
    result = result.astype("uint8") * 255
  util.np_info(result, "Filter Green Pen", t.elapsed())
  return result


def filter_blue(rgb, red_upper_thresh, green_upper_thresh, blue_lower_thresh, output_type="bool",
                display_np_info=False):
  """
  Create a mask to filter out blueish colors, where the mask is based on a pixel being below a
  red channel threshold value, below a green channel threshold value, and above a blue channel threshold value.

  Args:
    rgb: RGB image as a NumPy array.
    red_upper_thresh: Red channel upper threshold value.
    green_upper_thresh: Green channel upper threshold value.
    blue_lower_thresh: Blue channel lower threshold value.
    output_type: Type of array to return (bool, float, or uint8).
    display_np_info: If True, display NumPy array info and filter time.

  Returns:
    NumPy array representing the mask.
  """
  if display_np_info:
    t = Time()
  r = rgb[:, :, 0] < red_upper_thresh
  g = rgb[:, :, 1] < green_upper_thresh
  b = rgb[:, :, 2] > blue_lower_thresh
  result = ~(r & g & b)
  if output_type == "bool":
    pass
  elif output_type == "float":
    result = result.astype(float)
  else:
    result = result.astype("uint8") * 255
  if display_np_info:
    util.np_info(result, "Filter Blue", t.elapsed())
  return result


def filter_blue_pen(rgb, output_type="bool"):
  """
  Create a mask to filter out blue pen marks from a slide.

  Args:
    rgb: RGB image as a NumPy array.
    output_type: Type of array to return (bool, float, or uint8).

  Returns:
    NumPy array representing the mask.
  """
  t = Time()
  result = filter_blue(rgb, red_upper_thresh=60, green_upper_thresh=120, blue_lower_thresh=190) & \
           filter_blue(rgb, red_upper_thresh=120, green_upper_thresh=170, blue_lower_thresh=200) & \
           filter_blue(rgb, red_upper_thresh=175, green_upper_thresh=210, blue_lower_thresh=230) & \
           filter_blue(rgb, red_upper_thresh=145, green_upper_thresh=180, blue_lower_thresh=210) & \
           filter_blue(rgb, red_upper_thresh=37, green_upper_thresh=95, blue_lower_thresh=160) & \
           filter_blue(rgb, red_upper_thresh=30, green_upper_thresh=65, blue_lower_thresh=130) & \
           filter_blue(rgb, red_upper_thresh=130, green_upper_thresh=155, blue_lower_thresh=180) & \
           filter_blue(rgb, red_upper_thresh=40, green_upper_thresh=35, blue_lower_thresh=85) & \
           filter_blue(rgb, red_upper_thresh=30, green_upper_thresh=20, blue_lower_thresh=65) & \
           filter_blue(rgb, red_upper_thresh=90, green_upper_thresh=90, blue_lower_thresh=140) & \
           filter_blue(rgb, red_upper_thresh=60, green_upper_thresh=60, blue_lower_thresh=120) & \
           filter_blue(rgb, red_upper_thresh=110, green_upper_thresh=110, blue_lower_thresh=175)
  if output_type == "bool":
    pass
  elif output_type == "float":
    result = result.astype(float)
  else:
    result = result.astype("uint8") * 255
  util.np_info(result, "Filter Blue Pen", t.elapsed())
  return result


def filter_grays(rgb, tolerance=15, output_type="bool"):
  """
  Create a mask to filter out pixels where the red, green, and blue channel values are similar.

  Args:
    np_img: RGB image as a NumPy array.
    tolerance: Tolerance value to determine how similar the values must be in order to be filtered out
    output_type: Type of array to return (bool, float, or uint8).

  Returns:
    NumPy array representing a mask where pixels with similar red, green, and blue values have been masked out.
  """
  t = Time()
  (h, w, c) = rgb.shape

  rgb = rgb.astype(np.int)
  rg_diff = abs(rgb[:, :, 0] - rgb[:, :, 1]) <= tolerance
  rb_diff = abs(rgb[:, :, 0] - rgb[:, :, 2]) <= tolerance
  gb_diff = abs(rgb[:, :, 1] - rgb[:, :, 2]) <= tolerance
  result = ~(rg_diff & rb_diff & gb_diff)

  if output_type == "bool":
    pass
  elif output_type == "float":
    result = result.astype(float)
  else:
    result = result.astype("uint8") * 255
  util.np_info(result, "Filter Grays", t.elapsed())
  return result


def uint8_to_bool(np_img):
  """
  Convert NumPy array of uint8 (255,0) values to bool (True,False) values

  Args:
    np_img: Binary image as NumPy array of uint8 (255,0) values.

  Returns:
    NumPy array of bool (True,False) values.
  """
  result = (np_img / 255).astype(bool)
  return result


def apply_image_filters(np_img, slide_num=None, info=None, save=False, display=False):
  """
  Apply filters to image as NumPy array and optionally save and/or display filtered images.

  Args:
    np_img: Image as NumPy array.
    slide_num: The slide number (used for saving/displaying).
    info: Dictionary of slide information (used for HTML display).
    save: If True, save image.
    display: If True, display image.

  Returns:
    Resulting filtered image as a NumPy array.
  """
  rgb = np_img
  save_display(save, display, info, rgb, slide_num, 1, "Original", "rgb")

  mask_not_green = filter_green_channel(rgb)
  rgb_not_green = util.mask_rgb(rgb, mask_not_green)
  save_display(save, display, info, rgb_not_green, slide_num, 2, "Not Green", "rgb-not-green")

  mask_not_gray = filter_grays(rgb)
  rgb_not_gray = util.mask_rgb(rgb, mask_not_gray)
  save_display(save, display, info, rgb_not_gray, slide_num, 3, "Not Gray", "rgb-not-gray")

  mask_no_red_pen = filter_red_pen(rgb)
  rgb_no_red_pen = util.mask_rgb(rgb, mask_no_red_pen)
  save_display(save, display, info, rgb_no_red_pen, slide_num, 4, "No Red Pen", "rgb-no-red-pen")

  mask_no_green_pen = filter_green_pen(rgb)
  rgb_no_green_pen = util.mask_rgb(rgb, mask_no_green_pen)
  save_display(save, display, info, rgb_no_green_pen, slide_num, 5, "No Green Pen", "rgb-no-green-pen")

  mask_no_blue_pen = filter_blue_pen(rgb)
  rgb_no_blue_pen = util.mask_rgb(rgb, mask_no_blue_pen)
  save_display(save, display, info, rgb_no_blue_pen, slide_num, 6, "No Blue Pen", "rgb-no-blue-pen")

  mask_gray_green_pens = mask_not_gray & mask_not_green & mask_no_red_pen & mask_no_green_pen & mask_no_blue_pen
  rgb_gray_green_pens = util.mask_rgb(rgb, mask_gray_green_pens)
  save_display(save, display, info, rgb_gray_green_pens, slide_num, 7, "Not Gray, Not Green, No Pens",
               "rgb-no-gray-no-green-no-pens")

  mask_remove_small = filter_remove_small_objects(mask_gray_green_pens, min_size=500, output_type="bool")
  rgb_remove_small = util.mask_rgb(rgb, mask_remove_small)
  save_display(save, display, info, rgb_remove_small, slide_num, 8,
               "Not Gray, Not Green, No Pens,\nRemove Small Objects",
               "rgb-not-green-not-gray-no-pens-remove-small")

  img = rgb_remove_small
  return img


def apply_filters_to_image(slide_num, save=True, display=False):
  """
  Apply a set of filters to an image and optionally save and/or display filtered images.

  Args:
    slide_num: The slide number.
    save: If True, save filtered images.
    display: If True, display filtered images to screen.

  Returns:
    Tuple consisting of 1) the resulting filtered image as a NumPy array, and 2) dictionary of image information
    (used for HTML page generation).
  """
  t = Time()
  print("Processing slide #%d" % slide_num)

  info = dict()

  if save and not os.path.exists(slide.FILTER_DIR):
    os.makedirs(slide.FILTER_DIR)
  img_path = slide.get_training_image_path(slide_num)
  np_orig = slide.open_image_np(img_path)
  filtered_np_img = apply_image_filters(np_orig, slide_num, info, save=save, display=display)

  if save:
    t1 = Time()
    result_path = slide.get_filter_image_result(slide_num)
    pil_img = util.np_to_pil(filtered_np_img)
    pil_img.save(result_path)
    print("%-20s | Time: %-14s  Name: %s" % ("Save Image", str(t1.elapsed()), result_path))

    t1 = Time()
    thumbnail_path = slide.get_filter_thumbnail_result(slide_num)
    slide.save_thumbnail(pil_img, slide.THUMBNAIL_SIZE, thumbnail_path)
    print("%-20s | Time: %-14s  Name: %s" % ("Save Thumbnail", str(t1.elapsed()), thumbnail_path))

  print("Slide #%03d processing time: %s\n" % (slide_num, str(t.elapsed())))

  return filtered_np_img, info


def save_display(save, display, info, np_img, slide_num, filter_num, display_text, file_text,
                 display_mask_percentage=True):
  """
  Optionally save an image and/or display the image.

  Args:
    save: If True, save filtered images.
    display: If True, display filtered images to screen.
    info: Dictionary to store filter information.
    np_img: Image as a NumPy array.
    slide_num: The slide number.
    filter_num: The filter number.
    display_text: Filter display name.
    file_text: Filter name for file.
    display_mask_percentage: If True, display mask percentage on displayed slide.
  """
  mask_percentage = None
  if display_mask_percentage:
    mask_percentage = mask_percent(np_img)
    display_text = display_text + "\n(" + mask_percentage_text(mask_percentage) + " masked)"
  if slide_num is None and filter_num is None:
    pass
  elif filter_num is None:
    display_text = "S%03d " % slide_num + display_text
  elif slide_num is None:
    display_text = "F%03d " % filter_num + display_text
  else:
    display_text = "S%03d-F%03d " % (slide_num, filter_num) + display_text
  if display:
    util.display_img(np_img, display_text)
  if save:
    save_filtered_image(np_img, slide_num, filter_num, file_text)
  if info is not None:
    info[slide_num * 1000 + filter_num] = (slide_num, filter_num, display_text, file_text, mask_percentage)


def mask_percentage_text(mask_percentage):
  """
  Generate a formatted string representing the percentage that an image is masked.

  Args:
    mask_percentage: The mask percentage.

  Returns:
    The mask percentage formatted as a string.
  """
  return "%3.2f%%" % mask_percentage


def image_cell(slide_num, filter_num, display_text, file_text):
  """
  Generate HTML for viewing a processed image.

  Args:
    slide_num: The slide number.
    filter_num: The filter number.
    display_text: Filter display name.
    file_text: Filter name for file.

  Returns:
    HTML for a table cell for viewing a filtered image.
  """
  filt_img = slide.get_filter_image_path(slide_num, filter_num, file_text)
  filt_thumb = slide.get_filter_thumbnail_path(slide_num, filter_num, file_text)
  img_name = slide.get_filter_image_filename(slide_num, filter_num, file_text)
  return "      <td>\n" + \
         "        <a target=\"_blank\" href=\"%s\">%s<br/>\n" % (filt_img, display_text) + \
         "          <img src=\"%s\" />\n" % (filt_thumb) + \
         "        </a>\n" + \
         "      </td>\n"


def html_header(page_title):
  """
  Generate an HTML header for previewing images.

  Returns:
    HTML header for viewing images.
  """
  html = "<!DOCTYPE html PUBLIC \"-//W3C//DTD XHTML 1.0 Strict//EN\" " + \
         "\"http://www.w3.org/TR/xhtml1/DTD/xhtml1-strict.dtd\">\n" + \
         "<html xmlns=\"http://www.w3.org/1999/xhtml\" lang=\"en\" xml:lang=\"en\">\n" + \
         "  <head>\n" + \
         "    <title>%s</title>\n" % page_title + \
         "    <style type=\"text/css\">\n" + \
         "     img { border: 2px solid black; }\n" + \
         "     td { border: 2px solid black; }\n" + \
         "    </style>\n" + \
         "  </head>\n" + \
         "  <body>\n"
  return html


def html_footer():
  """
  Generate an HTML footer for previewing images.

  Returns:
    HTML footer for viewing images.
  """
  html = "</body>\n" + \
         "</html>\n"
  return html


def save_filtered_image(np_img, slide_num, filter_num, filter_text):
  """
  Save a filtered image to the file system.

  Args:
    np_img: Image as a NumPy array.
    slide_num:  The slide number.
    filter_num: The filter number.
    filter_text: Descriptive text to add to the image filename.
  """
  t = Time()
  filepath = slide.get_filter_image_path(slide_num, filter_num, filter_text)
  pil_img = util.np_to_pil(np_img)
  pil_img.save(filepath)
  print("%-20s | Time: %-14s  Name: %s" % ("Save Image", str(t.elapsed()), filepath))

  t1 = Time()
  thumbnail_filepath = slide.get_filter_thumbnail_path(slide_num, filter_num, filter_text)
  slide.save_thumbnail(pil_img, slide.THUMBNAIL_SIZE, thumbnail_filepath)
  print("%-20s | Time: %-14s  Name: %s" % ("Save Thumbnail", str(t1.elapsed()), thumbnail_filepath))


def generate_filter_html_result(html_page_info):
  """
  Generate HTML to view the filtered images. If slide.FILTER_PAGINATE is True, the results will be paginated.

  Args:
    html_page_info: Dictionary of image information.
  """
  if not slide.FILTER_PAGINATE:
    html = ""
    html += html_header("Filtered Images")
    html += "  <table>\n"

    row = 0
    for key in sorted(html_page_info):
      value = html_page_info[key]
      current_row = value[0]
      if current_row > row:
        html += "    <tr>\n"
        row = current_row
      html += image_cell(value[0], value[1], value[2], value[3])
      next_key = key + 1
      if next_key not in html_page_info:
        html += "    </tr>\n"

    html += "  </table>\n"
    html += html_footer()
    text_file = open(os.path.join(slide.FILTER_HTML_DIR, "filters.html"), "w")
    text_file.write(html)
    text_file.close()
  else:
    slide_nums = set()
    for key in html_page_info:
      slide_num = math.floor(key / 1000)
      slide_nums.add(slide_num)
    slide_nums = sorted(list(slide_nums))
    total_len = len(slide_nums)
    page_size = slide.FILTER_PAGINATION_SIZE
    num_pages = math.ceil(total_len / page_size)

    for page_num in range(1, num_pages + 1):
      start_index = (page_num - 1) * page_size
      end_index = (page_num * page_size) if (page_num < num_pages) else total_len
      page_slide_nums = slide_nums[start_index:end_index]

      html = ""
      html += html_header("Filtered Images, Page %d" % page_num)

      html += "  <div style=\"font-size: 20px\">"
      if page_num > 1:
        if page_num == 2:
          html += "<a href=\"filters.html\">&lt;</a> "
        else:
          html += "<a href=\"filters-%d.html\">&lt;</a> " % (page_num - 1)
      html += "Page %d" % page_num
      if page_num < num_pages:
        html += " <a href=\"filters-%d.html\">&gt;</a> " % (page_num + 1)
      html += "</div>\n"

      html += "  <table>\n"
      for slide_num in page_slide_nums:
        html += "  <tr>\n"
        filter_num = 1

        lookup_key = slide_num * 1000 + filter_num
        while lookup_key in html_page_info:
          value = html_page_info[lookup_key]
          html += image_cell(value[0], value[1], value[2], value[3])
          lookup_key += 1
        html += "  </tr>\n"

      html += "  </table>\n"

      html += html_footer()
      if page_num == 1:
        text_file = open(os.path.join(slide.FILTER_HTML_DIR, "filters.html"), "w")
      else:
        text_file = open(os.path.join(slide.FILTER_HTML_DIR, "filters-%d.html" % page_num), "w")
      text_file.write(html)
      text_file.close()


def apply_filters_to_image_list(image_num_list, save, display):
  """
  Apply filters to a list of images.

  Args:
    image_num_list: List of image numbers.
    save: If True, save filtered images.
    display: If True, display filtered images to screen.

  Returns:
    Tuple consisting of 1) a list of image numbers, and 2) a dictionary of image filter information.
  """
  html_page_info = dict()
  for slide_num in image_num_list:
    _, info = apply_filters_to_image(slide_num, save=save, display=display)
    html_page_info.update(info)
  return image_num_list, html_page_info


def apply_filters_to_image_range(start_ind, end_ind, save, display):
  """
  Apply filters to a range of images.

  Args:
    start_ind: Starting index (inclusive).
    end_ind: Ending index (inclusive).
    save: If True, save filtered images.
    display: If True, display filtered images to screen.

  Returns:
    Tuple consisting of 1) staring index of slides converted to images, 2) ending index of slides converted to images,
    and 3) a dictionary of image filter information.
  """
  html_page_info = dict()
  for slide_num in range(start_ind, end_ind + 1):
    _, info = apply_filters_to_image(slide_num, save=save, display=display)
    html_page_info.update(info)
  return start_ind, end_ind, html_page_info


def singleprocess_apply_filters_to_images(save=True, display=False, html=True, image_num_list=None):
  """
  Apply a set of filters to training images and optionally save and/or display the filtered images.

  Args:
    save: If True, save filtered images.
    display: If True, display filtered images to screen.
    html: If True, generate HTML page to display filtered images.
    image_num_list: Optionally specify a list of image slide numbers.
  """
  t = Time()
  print("Applying filters to images\n")

  if image_num_list is not None:
    _, info = apply_filters_to_image_list(image_num_list, save, display)
  else:
    num_training_slides = slide.get_num_training_slides()
    (s, e, info) = apply_filters_to_image_range(1, num_training_slides, save, display)

  print("Time to apply filters to all images: %s\n" % str(t.elapsed()))

  if html:
    generate_filter_html_result(info)


def multiprocess_apply_filters_to_images(save=True, display=False, html=True, image_num_list=None):
  """
  Apply a set of filters to all training images using multiple processes (one process per core).

  Args:
    save: If True, save filtered images.
    display: If True, display filtered images to screen (multiprocessed display not recommended).
    html: If True, generate HTML page to display filtered images.
    image_num_list: Optionally specify a list of image slide numbers.
  """
  timer = Time()
  print("Applying filters to images (multiprocess)\n")

  if save and not os.path.exists(slide.FILTER_DIR):
    os.makedirs(slide.FILTER_DIR)

  # how many processes to use
  num_processes = multiprocessing.cpu_count()
  pool = multiprocessing.Pool(num_processes)

  if image_num_list is not None:
    num_train_images = len(image_num_list)
  else:
    num_train_images = slide.get_num_training_slides()
  if num_processes > num_train_images:
    num_processes = num_train_images
  images_per_process = num_train_images / num_processes

  print("Number of processes: " + str(num_processes))
  print("Number of training images: " + str(num_train_images))

  tasks = []
  for num_process in range(1, num_processes + 1):
    start_index = (num_process - 1) * images_per_process + 1
    end_index = num_process * images_per_process
    start_index = int(start_index)
    end_index = int(end_index)
    if image_num_list is not None:
      sublist = image_num_list[start_index - 1:end_index]
      tasks.append((sublist, save, display))
      print("Task #" + str(num_process) + ": Process slides " + str(sublist))
    else:
      tasks.append((start_index, end_index, save, display))
      if start_index == end_index:
        print("Task #" + str(num_process) + ": Process slide " + str(start_index))
      else:
        print("Task #" + str(num_process) + ": Process slides " + str(start_index) + " to " + str(end_index))

  # start tasks
  results = []
  for t in tasks:
    if image_num_list is not None:
      results.append(pool.apply_async(apply_filters_to_image_list, t))
    else:
      results.append(pool.apply_async(apply_filters_to_image_range, t))

  html_page_info = dict()
  for result in results:
    if image_num_list is not None:
      (image_nums, html_page_info_res) = result.get()
      html_page_info.update(html_page_info_res)
      print("Done filtering slides: %s" % image_nums)
    else:
      (start_ind, end_ind, html_page_info_res) = result.get()
      html_page_info.update(html_page_info_res)
      if (start_ind == end_ind):
        print("Done filtering slide %d" % start_ind)
      else:
        print("Done filtering slides %d through %d" % (start_ind, end_ind))

  if html:
    generate_filter_html_result(html_page_info)

  print("Time to apply filters to all images (multiprocess): %s\n" % str(timer.elapsed()))

# if __name__ == "__main__":
# slide.training_slide_to_image(2)
# singleprocess_apply_filters_to_images(image_num_list=[2], display=True)

# singleprocess_apply_filters_to_images()
# multiprocess_apply_filters_to_images()
