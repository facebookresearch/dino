"""
Evaluate the prediction results
"""
import numpy as np
import pandas as pd
from pathlib import Path
import re, os
from PIL import Image
from deephistopath.visualization import Shape, add_mark

GROUND_TRUTH_FILE_ID_RE = \
  "(\d+/\d+|\d+-\d+)[.csv|/clustered_mitosis_locations.csv|.tif|_mark.tif|_mask.tif]"

def prepare_f1_inputs(prediction, ground_truth, threshold=30, file_id=None):
  """ Prepare the input variables (TP, FP, and PN) for computing F1
  score.

  TP: its Eucledian distance to a ground truth location is less than
    threshold;
  FP: not within the threshold distance of a ground truth location;
  FN: all ground truth locations that do not have a detection within
    the threshold.
  file_id: file id for the input prediction and ground truth data,
    which should have the same file id

  Args:
    prediction: prediction result, a list of point locations, e.g.
      [(r0, c0, p0), (r1, c1, p1), (r2, c2, p2), ...].
    ground_truth: groud truth data, a list of point locations, e.g.
      [(r0, c0), (r1, c1), (r2, c2), ...].
    threshold: Eucledian distance to see if the predict and ground truth
      points are in the same circle.

  Return:
    A tuple of (FP, TP, FN), where TP will be a list of list of tuples like
    [[(file_id0, r0, c0, prob0, 'TP'),..., (file_id_n, r_n, c_n, prob_n,
     'TP')], [.....], ......, [......]], FP will be like [(file_id, r,
     c, prob, 'FP'), .....], FN will be like [(file_id, r, c, prob,
     'FN'), .....]
  """
  if len(prediction) == 0 and len(ground_truth) != 0:
    FP = []
    TP = []
    FN = [(file_id, r, c, 999, 'FN') for (r, c) in ground_truth]
  elif len(prediction) != 0 and len(ground_truth) == 0:
    FP = [(file_id, r, c, p, 'FP') for (r, c, p) in prediction]
    TP = []
    FN = []
  elif len(prediction) != 0 and len(ground_truth) != 0:
    # initialize the values.
    FP = [(file_id, r, c, p, 'FP') for (r, c, p) in prediction]
    TP = []
    FN = []
    # check which points in the ground truth are located with each other
    # less than `threshold`.
    for gt_r, gt_c in ground_truth:
      for r, c in ground_truth:
        dist = np.sqrt((gt_r - r) ** 2 + (gt_c - c) ** 2)
        if dist > 0 and dist < threshold * 2:
          print(f"Point ({r} , {c}) has multiple points in the circle")

    for gt_r, gt_c in ground_truth:
      # if several points fall within a single ground truth location,
      # they will be counted as one true positive. Here we use a list
      # to collect this kind of points for each ground truth point.
      tp = []
      for pred_r, pred_c, pre_prob in prediction:
        dist = np.sqrt((gt_r - pred_r) ** 2 + (gt_c - pred_c) ** 2)
        if dist <= threshold:
          tp.append((file_id, pred_r, pred_c, pre_prob, 'TP'))
          # remove the TP point from the FP list
          if (file_id, pred_r, pred_c, pre_prob, 'FP') in FP:
            FP.remove((file_id, pred_r, pred_c, pre_prob, 'FP'))
      if len(tp) == 0:
        # if no points fall within this ground truth point, the ground
        # truth point will be treated as FN. 999 does not mean the
        # prediction probability, it will indicate this is FN
        FN.append((file_id, gt_r, gt_c, 999, 'FN'))
      else:
        # the reason of using .append() here instead of += is to count
        # several points falling in a single ground truth location as
        # one true positive
        TP.append(tp)
  else:
    FP = []
    TP = []
    FN = []

  return (FP, TP, FN)

def compute_f1(FP, TP, FN):
  """ Compute the F1 score. The calculation equation can be found at
  http://amida13.isi.uu.nl/?q=node/4.

  Args:
    FP: a list of false positive predictions.
    TP: a list of true positive predictions.
    FN: a list of false negative predictions.

  Return:
    a tuple of (F1, precision, recall).
  """

  precision = len(TP) / (len(TP) + len(FP)) if len(TP) + len(FP) > 0 else 0
  recall = len(TP) / (len(TP) + len(FN)) if len(TP) + len(FN) > 0 else 0
  f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

  #print (f"TP: {len(TP)}; FP: {len(FP)}; FN: {len(FN)}")
  #print (f"precision: {precision}; recall: {recall}")

  return (f1, precision, recall)

def list_files(dir, file_suffix):
  """ recursively list all the files that have the same input file
    suffix under the input directory.

  Args
    dir: file directory.
    file_suffix: file suffix.

  Return:
    a list of file path.
  """
  dir = Path(dir)
  files = [str(x) for x in dir.rglob(file_suffix)]
  return files

def get_file_id(files, file_id_re):
  """ get the file id using the file id regular expression
  Args:
    files: list of input file paths
    file_id_re: regular expression string used to detect the file ID
      from the file path

  Return:
    a dictionary of file id and its full file path
  """
  id_files = {re.findall(file_id_re, x)[0]: x for x in files}
  return id_files


def get_locations_from_csv(file, hasHeader=False, hasProb=True):
  """ get the point locations from CSV file.

  Args:
    file: csv file, of which the first and second columns store the
      point coordinates.
    hasHeader: bool value to tell if the input csv file has a header or
      not.
    hasProb: bool value to tell if the input csv file has the
      probability column

  Return:
    a list of point locations, e.g. [(r0, c0, p0), (r1, c1, p1), ......].
  """
  # handle the case that the input file does not exist
  if file is None:
    return []

  if hasHeader:
    data = pd.read_csv(file)
  else:
    data = pd.read_csv(file, header=None)

  if hasProb:
    locations = [(int(x[0]), int(x[1]), float(x[2])) for x in data.values.tolist()]
  else:
    locations = [(int(x[0]), int(x[1])) for x in data.values.tolist()]
  return locations


def get_data_from_csv(file, hasHeader=False):
  """ get the data from CSV file.

  Args:
    file: csv file, of which the first and second columns store the
      point coordinates.
    hasHeader: bool value to tell if the input csv file has a header or
      not.

  Return:
    a list of rows.
  """
  if hasHeader:
    data = pd.read_csv(file)
  else:
    data = pd.read_csv(file, header=None)

  return data.values.tolist()


def evaluate_f1(pred_dir, ground_true_dir, threshold=30, prob_threshold=None):
  """ Evaluate the prediction result based on the ground truth data
    using F1 score. It will compute F1 score for each input file.

  Args:
    pred_dir: directory for the prediction csv files.
    ground_true_dir: directory for the ground truth csv files.
    threshold: Eucledian distance to see if the predict and ground
      truth points are at the same circle.
    prob_threshold: prediction probability threshold

  Return:
     a tuple of (a F1_score list for each input file; a list of files
      that are detected by model, but not in the ground truth data; a
      list of files that are not detected by model, but in the ground
      truth data).
  """
  pred_files = list_files(pred_dir, "*.csv")
  pred_files = get_file_id(pred_files, GROUND_TRUTH_FILE_ID_RE)
  ground_true_files = list_files(ground_true_dir, "*.csv")
  ground_true_files = get_file_id(ground_true_files, GROUND_TRUTH_FILE_ID_RE)

  union_file_keys = set(pred_files.keys()).union(set(ground_true_files.keys()))

  f1_list = []
  over_detected = []
  non_detected = []

  for key in union_file_keys:
    pred_file = pred_files[key] if key in pred_files else None
    ground_true_file = ground_true_files[key] if key in ground_true_files else None
    pred_locations = get_locations_from_csv(pred_file, hasHeader=True, hasProb=True)
    if prob_threshold is not None:
      pred_locations = [location for location in pred_locations if location[2] > prob_threshold]

    ground_true_locations = get_locations_from_csv(ground_true_file, hasHeader=False, hasProb=False)
    FP, TP, FN = prepare_f1_inputs(pred_locations, ground_true_locations, threshold, file_id=key)
    f1, precision, recall = compute_f1(FP, TP, FN)
    f1_list.append([f"{key}-{len(ground_true_locations)}-{len(pred_locations)}", f1])

    # record the image IDs that the model detected but not in the ground truth
    if pred_file is None:
      non_detected.append(ground_true_file)

    # record the image IDs that are not detected by the model
    if ground_true_file is None:
      over_detected.append(pred_file)

  return (f1_list, over_detected, non_detected)


def evaluate_global_f1(pred_dir, ground_true_dir, threshold=30, prob_threshold=None):
  """ Evaluate the prediction result based on the ground truth data
    using F1 score. It will compute a single F1 score for over all
    input files.

  Args:
    pred_dir: directory for the prediction csv files.
    ground_true_dir: directory for the ground truth csv files.
    threshold: Eucledian distance to see if the predict and ground
      truth points are at the same circle.
    prob_threshold: prediction probability threshold

  Return:
    A tuple of (a single F1_score; a single precision value; a single
    recall value; a list of files that are detected by model, but not in
    the ground truth data; a list of files that are not detected by
    model, but in the ground truth data; a list of FP; a list of TP; a
    list of FN).
  """
  pred_files = list_files(pred_dir, "*.csv")
  pred_files = get_file_id(pred_files, GROUND_TRUTH_FILE_ID_RE)
  ground_true_files = list_files(ground_true_dir, "*.csv")
  ground_true_files = get_file_id(ground_true_files, GROUND_TRUTH_FILE_ID_RE)

  union_file_keys = set(pred_files.keys()).union(set(ground_true_files.keys()))

  over_detected = []
  non_detected = []
  FP_list = []
  TP_list = []
  FN_list = []

  for key in union_file_keys:
    pred_file = pred_files[key] if key in pred_files else None
    ground_true_file = ground_true_files[key] if key in ground_true_files else None
    pred_locations = get_locations_from_csv(pred_file, hasHeader=True, hasProb=True)
    if prob_threshold is not None:
      pred_locations = [location for location in pred_locations if location[2] > prob_threshold]

    ground_true_locations = get_locations_from_csv(ground_true_file, hasHeader=False, hasProb=False)
    FP, TP, FN = prepare_f1_inputs(pred_locations, ground_true_locations, threshold, file_id=key)
    FP_list += FP
    TP_list += TP
    FN_list += FN

    # record the image IDs that the model detected but not in the ground truth
    if pred_file is None:
      non_detected.append(ground_true_file)

    # record the image IDs that are not detected by the model
    if ground_true_file is None:
      over_detected.append(pred_file)

  f1, precision, recall = compute_f1(FP_list, TP_list, FN_list)
  return (f1, precision, recall, over_detected, non_detected, FP_list, TP_list, FN_list)


def search_prob_threshold_for_f1(prob_thresholds, pred_dir, ground_truth_dir, dist_threshold=30):
  """ Search the best probability threshold for F1

  Args:
    prob_thresholds: a list of probability thresholds
    pred_dir: the directory for the prediction csv files
    ground_truth_dir: the directory for the ground truth files
    dist_threshold: Eucledian distance to see if the predict and ground truth
      points are at the same circle.

  Return:
    a data frame, each row of which will be ('prob_threshold', 'f1', 'precision', 'recall')
  """
  result = []
  for prob_threshold in prob_thresholds:
    f1, precision, recall, \
    over_detected, non_detected, \
    FP, TP, FN = evaluate_global_f1(pred_dir, ground_truth_dir, dist_threshold, prob_threshold)
    result.append((prob_threshold, f1, precision, recall))

  df = pd.DataFrame(result, columns=['prob_threshold', 'f1', 'precision', 'recall'])
  return df

def export_F1_inputs_TP_FP_FN(output_path, FP, TP, FN):
  """ Export TP, FP, FN predictions into a single csv file

  Args:
    output_path: output file path
    FP: a list of FP prediction tuples e.g. [(file_id, row, col,
     prob, label), ......]
    TP: a list of TP prediction tuples e.g. [(file_id, row, col,
     prob, label), ......]
    FN: a list of FN prediction tuples e.g. [(file_id, row, col,
     prob, label), ......]
  """
  TP_flat = []
  for tp_col in TP:
    for tp in tp_col:
      TP_flat.append(tp)

  result = []
  result += FP
  result += TP_flat
  result += FN
  df = pd.DataFrame(result, columns=['file_id', 'row', 'col', 'prob', 'label'])

  dir = os.path.dirname(output_path)
  os.makedirs(dir, exist_ok=True)
  df.to_csv(output_path, index=False)

def export_single_F1_input(output_dir, f1_parameter, f1_parameter_name):
  """ Export the prediction results with the label (e.g. FP, TP, FN)
    into the csv file
  Args:
    output_dir: directory for the output file
    f1_parameter: a list of prediction tuples e.g. [(file_id, row, col,
     prob, label), ......]
    f1_parameter_name: label string, which could be FP, TP, or FN
  """

  df = pd.DataFrame(f1_parameter, columns=['file_id', 'row', 'col', 'prob', 'label'])
  df = df.groupby(['file_id'])
  for group in df:
    file_path = group[0] + ".csv"
    file_path = os.path.join(output_dir, f1_parameter_name, file_path)
    dir = os.path.dirname(file_path)
    os.makedirs(dir, exist_ok=True)
    table = group[1][['row', 'col']]
    table.to_csv(file_path, index=False, header=False)


def add_ground_truth_mark_help(im_path, ground_truth_file_path, hasHeader=False,
                               shape=Shape.CROSS, mark_color=(0, 255, 127, 200), hasProb=False):
  """ add the mark for each point in the ground truth data.

  Args:
    im_path: image file path to add the marks.
    ground_truth_file_path: file path for the ground truth csv file.
    hasHeader: bool value to tell if the ground truth csv file has a header.
    shape: options for mark shape. It could be Shape.CROSS, Shape.SQUARE,
      or Shape.CIRCLE.
    mark_color: mark color, the default value is (0, 255, 127, 200)
    hasProb: bool value to tell if the input csv file has the
      probability column
  """
  im = Image.open(im_path)
  locations = get_locations_from_csv(ground_truth_file_path, hasHeader=hasHeader, hasProb=hasProb)
  add_mark(im, locations, shape, mark_color=mark_color, hasProb=hasProb)
  print(im_path)
  im.save(im_path)

def add_ground_truth_mark(sparkContext, partition_num, im_dir, im_suffix, ground_truth_dir,
                          ground_truth_file_suffix, shape=Shape.CROSS,
                          mark_color=(0, 255, 127, 200), hasHeader=False, hasProb=False):
  """ Add the ground truth data as marks into the images in parallel.

    Args:
      sparkContext: Spark context.
      partition_num: number of partitions for the input images
      im_dir: input image directory.
      im_suffix: input image suffix.
      ground_truth_dir: ground truth file directory
      ground_truth_file_suffix: ground truth file suffix
      shape: options for mark shape. It could be Shape.CROSS, Shape.SQUARE,
        or Shape.CIRCLE.
      hasHeader: bool value to tell if the ground truth csv file has a
        header.
      hasProb: bool value to tell if the input csv file has the
      probability column.
    """

  # assume that the image file path and ground truth file path are like
  # "/File/.../Path/ID1/ID2/FileName"
  input_imgs = list_files(im_dir, im_suffix)
  id_imgs = get_file_id(input_imgs, GROUND_TRUTH_FILE_ID_RE)

  ground_truth_files = list_files(ground_truth_dir, ground_truth_file_suffix)
  id_ground_truth_files = get_file_id(ground_truth_files, GROUND_TRUTH_FILE_ID_RE)

  # match the image data with the ground truth csv file
  img_ground_truth = []
  for id_ground_truth in id_ground_truth_files.keys():
    if id_ground_truth in id_imgs:
      img_ground_truth.append((id_imgs[id_ground_truth], id_ground_truth_files[id_ground_truth]))

  # parallel the input images, and add the marks to each image
  rdd = sparkContext.parallelize(img_ground_truth, partition_num)
  rdd.foreach(lambda t: add_ground_truth_mark_help(t[0], t[1], hasHeader=hasHeader, shape=shape,
                                                   mark_color=mark_color, hasProb=hasProb))


def test_compute_f1():
  threshold = 30

  # test case 1: the prediction result is the same with the ground truth
  predict1 = [(10, 10, 0.5), (20, 30, 0.5), (30, 60, 0.5), (100, 53, 0.5), (32, 26, 0.5),
      (120, 66, 0.5)]
  ground_true1 = [(10, 10), (20, 30), (30, 60), (100, 53), (32, 26), (120, 66)]
  FP, TP, FN = prepare_f1_inputs(predict1, ground_true1, threshold)
  f1, precision, recall = compute_f1(FP, TP, FN)
  assert f1 == 1

  # test case 2: the prediction result is totally different from the ground truth
  predict2 = [(10, 10, 0.5), (2, 3, 0.5), (3, 6, 0.5), (10, 5, 0.5), (3, 2, 0.5), (12, 6, 0.5)]
  ground_true2 = [(100, 100), (200, 300), (300, 600), (1000, 530), (320, 260), (1200, 660)]
  FP, TP, FN = prepare_f1_inputs(predict2, ground_true2, threshold)
  f1, precision, recall = compute_f1(FP, TP, FN)
  assert f1 == 0

  # test case 3: the prediction result partially matches the ground truth
  predict3 = [(2, 3, 0.5), (50, 180, 0.5), (66, 20, 0.5), (80, 70, 0.5), (100, 200, 0.5),
      (300, 400, 0.5)]
  ground_true3 = [(10, 10), (40, 50), (50, 200), (60, 110), (70, 10), (80, 80)]
  FP, TP, FN = prepare_f1_inputs(predict3, ground_true3, threshold)
  f1, precision, recall = compute_f1(FP, TP, FN)
  assert f1 == 2/3

  # test case 4: ground truth but no predictions
  predict4 = []
  ground_true4 = [(10, 10), (40, 50), (50, 200), (60, 110), (70, 10), (80, 80)]
  FP, TP, FN = prepare_f1_inputs(predict4, ground_true4, threshold)
  f1, precision, recall = compute_f1(FP, TP, FN)
  assert f1 == 0

  # test case 5: prediction but not ground truth
  predict5 = [(2, 3, 0.5), (50, 180, 0.5), (66, 20, 0.5), (80, 70, 0.5), (100, 200, 0.5),
      (300, 400, 0.5)]
  ground_true5 = []
  FP, TP, FN = prepare_f1_inputs(predict5, ground_true5, threshold)
  f1, precision, recall = compute_f1(FP, TP, FN)
  assert f1 == 0

  # test case 6: no prediction but also no ground truth
  predict6 = []
  ground_true6 = []
  FP, TP, FN = prepare_f1_inputs(predict6, ground_true6, threshold)
  f1, precision, recall = compute_f1(FP, TP, FN)
  assert f1 == 0


def test_add_ground_truth_mark():
  from pyspark.sql import SparkSession
  # Create new SparkSession
  spark = SparkSession.builder.appName("Add marks").getOrCreate()
  sparkContext = spark.sparkContext
  im_dir = "result/mitoses/mitoses_train_image_result"
  partition_num = 4
  im_suffix = "*mark.tif"
  ground_truth_dir = "data/mitoses/mitoses_ground_truth"
  ground_truth_file_suffix = "*.csv"
  hasHeader = False
  hasProb = True

  add_ground_truth_mark(sparkContext, partition_num, im_dir, im_suffix, ground_truth_dir,
                        ground_truth_file_suffix, Shape.SQUARE, mark_color=(0, 255, 127, 200),
                        hasHeader=hasHeader, hasProb=hasProb)

def test_img_quality():
  # this test shows the image quality difference between different image
  # formats (e.g., tif, png, jpeg) and different libraries (e.g., openslide,
  # PIL, Tensorflow) with different reading parameters. The experiment
  # results indicate that tif and png have the highest quality; JPEG changes
  # a lot of pixel values when it compresses the images, but has the smallest
  # file size.

  import os
  import openslide
  import tensorflow as tf
  from deephistopath.preprocessing import create_tile_generator, get_20x_zoom_level
  import pandas as pd

  # all the output images from these reader functions are in the type of
  # np.int instead of np.uint8, which avoids the invalid value when
  # computing the difference between two images

  def read_img_openslide(img_file):
    slide = openslide.open_slide(img_file)
    ROI_size = max(slide.dimensions)
    generator = create_tile_generator(slide, ROI_size, 0)
    zoom_level = get_20x_zoom_level(slide, generator)
    cols, rows = generator.level_tiles[zoom_level]
    ROI_indices = [(zoom_level, col, row) for col in range(cols) for row in range(rows)]
    zl, col, row = ROI_indices[0]
    img_openslide = np.asarray(generator.get_tile(zl, (col, row)), dtype=np.int)
    return img_openslide

  def read_img_PIL(img_file):
    img_PIL = Image.open(img_file)
    img_PIL = np.array(img_PIL, dtype=np.int)
    return img_PIL

  def read_jpeg_tf_fast(jpg_file):
    image_string = tf.read_file(jpg_file)
    img_tensor_fast = tf.image.decode_jpeg(image_string, channels=3, dct_method='INTEGER_FAST')

    with tf.Session() as sess:
      img_tf_fast = sess.run(img_tensor_fast).astype(np.int)
    return img_tf_fast

  def read_jpeg_tf_accurate(jpg_file):
    image_string = tf.read_file(jpg_file)
    img_tensor_accurate = tf.image.decode_jpeg(image_string, channels=3,
                                               dct_method='INTEGER_ACCURATE')
    with tf.Session() as sess:
      img_tf_accurate = sess.run(img_tensor_accurate).astype(np.int)
    return img_tf_accurate

  def read_png_tf(png_file):
    image_string = tf.read_file(png_file)
    img_tensor_png = tf.image.decode_png(image_string, channels=3, dtype=tf.uint8)
    with tf.Session() as sess:
      img_tf_png = sess.run(img_tensor_png).astype(np.int)
    return img_tf_png

  def compute_diff(img1, img2, threshold=0):
    diff = np.sum(np.abs(img1 - img2) > threshold)/img1.size
    return diff

  # generate an image array
  img_orig = np.random.randint(low=0, high=256, size=(2000, 2000, 3), dtype=np.uint8)

  # save the image array as the tif file
  tif_file = "test.tif"
  Image.fromarray(img_orig).save(tif_file, subsampling=0, quality=100)

  img_tif = Image.open(tif_file)
  # save the image array from the tif file into: 1) low-quality jpeg:
  # using the default configuration of PIL; 2) high-quality jpeg:
  # adding the customized configuration for PIL to generate a
  # high-quality JPG; 3) png
  jpg_lq_file = "test_lq.jpg"
  jpg_hq_file = "test_hq.jpg"
  png_file = "test.png"
  img_tif.save(jpg_lq_file)
  img_tif.save(jpg_hq_file, subsampling=0, quality=100)
  img_tif.save(png_file)
  img_tif.close()

  # start to test the reading result from OpenSlide, PIL, and Tensorflow

  # convert the data type of original image from uint8 to int
  img_orig = img_orig.astype(np.int)

  # read images using openslide
  img_openslide_jpg_lq = read_img_openslide(jpg_lq_file)
  img_openslide_jpg_hq = read_img_openslide(jpg_hq_file)
  img_openslide_tif = read_img_openslide(tif_file)
  img_openslide_png = read_img_openslide(png_file)

  # read images using PIL
  img_PIL_jpg_lq = read_img_PIL(jpg_lq_file)
  img_PIL_jpg_hq = read_img_PIL(jpg_hq_file)
  img_PIL_tif = read_img_PIL(tif_file)
  img_PIL_png = read_img_PIL(png_file)

  # read images using Tensorflow
  img_tf_jpg_lq_fast = read_jpeg_tf_fast(jpg_lq_file)
  img_tf_jpg_lq_accurate = read_jpeg_tf_accurate(jpg_lq_file)
  img_tf_jpg_hq_fast = read_jpeg_tf_fast(jpg_hq_file)
  img_tf_jpg_hq_accurate = read_jpeg_tf_accurate(jpg_hq_file)
  img_tf_png = read_png_tf(png_file)

  # evaluate the difference between different images using different
  # libraries
  img_name_list = ["orig_img", "openslide_tif", "PIL_tif", "openslide_png", "PIL_png", "tf_png",
                   "openslide_jpg_lq", "openslide_jpg_hq", "PIL_jpg_lq", "PIL_jpg_hq",
                   "tf_jpg_lq_fast", "tf_jpg_lq_accurate", "tf_jpg_hq_fast", "tf_jpg_hq_accurate"]

  img_dict = {"orig_img": img_orig,
               "openslide_tif": img_openslide_tif,
               "PIL_tif": img_PIL_tif,
               "openslide_png": img_openslide_png,
               "PIL_png": img_PIL_png,
               "tf_png": img_tf_png,
               "openslide_jpg_lq": img_openslide_jpg_lq,
               "openslide_jpg_hq": img_openslide_jpg_hq,
               "PIL_jpg_lq": img_PIL_jpg_lq,
               "PIL_jpg_hq": img_PIL_jpg_hq,
               "tf_jpg_lq_fast": img_tf_jpg_lq_fast,
               "tf_jpg_lq_accurate": img_tf_jpg_lq_accurate,
               "tf_jpg_hq_fast": img_tf_jpg_hq_fast,
               "tf_jpg_hq_accurate": img_tf_jpg_hq_accurate}

  num_img = len(img_name_list)
  diff = np.empty([num_img, num_img], dtype=np.float32)

  # compute the difference ratio between different images
  threshold = 0 # threshold to judge whether two pixels are different
  for row in range(num_img):
    for col in range(num_img):
      diff[row][col] = compute_diff(img_dict[img_name_list[row]],
                                    img_dict[img_name_list[col]],
                                    threshold=threshold)

  # organize the comparison result into data frame, and print it out as
  # table
  df = pd.DataFrame(diff, img_name_list, img_name_list)
  with pd.option_context("display.max_rows", num_img, "display.max_columns", num_img,
                         'expand_frame_repr', False):
    print(df)

  # delete the temporary file
  os.remove(tif_file)
  os.remove(jpg_hq_file)
  os.remove(jpg_lq_file)
  os.remove(png_file)


def test_evaluate_global_f1():
  # NOTE: `pred` refers to `/8tb/deep_histopath/pred` on the servers
  pred_dir = "pred/result/vgg/32_vgg_nm_0stride/"
  ground_true_dir = "pred/mitoses_ground_truth_test/val/"

  threshold = 30
  # implicit 0.5 threshold
  f1, ppv, sens, over_detected, non_detected, fp_list, tp_list, fn_list = evaluate_global_f1(
      pred_dir, ground_true_dir, threshold)
  assert np.allclose(f1, 0.518918918918919)

  # custom threshold
  prob_threshold = 0.6
  f1, ppv, sens, over_detected, non_detected, fp_list, tp_list, fn_list = evaluate_global_f1(
      pred_dir, ground_true_dir, threshold, prob_threshold)
  assert np.allclose(f1, 0.5445544554455445)

