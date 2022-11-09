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

# To get around renderer issue on macOS going from Matplotlib image to NumPy image.
import matplotlib

matplotlib.use('Agg')

import colorsys
import math
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont
from enum import Enum
from deephistopath.wsi import util
from deephistopath.wsi import filter
from deephistopath.wsi import slide
from deephistopath.wsi.util import Time

TISSUE_HIGH_THRESH = 80
TISSUE_LOW_THRESH = 10

ROW_TILE_SIZE = 1024
COL_TILE_SIZE = 1024
NUM_TOP_TILES = 50

DISPLAY_TILE_SUMMARY_LABELS = False
TILE_LABEL_TEXT_SIZE = 10
LABEL_ALL_TILES_IN_TOP_TILE_SUMMARY = False
BORDER_ALL_TILES_IN_TOP_TILE_SUMMARY = False

TILE_BORDER_SIZE = 2  # The size of the colored rectangular border around summary tiles.

HIGH_COLOR = (0, 255, 0)
MEDIUM_COLOR = (255, 255, 0)
LOW_COLOR = (255, 165, 0)
NONE_COLOR = (255, 0, 0)

FADED_THRESH_COLOR = (128, 255, 128)
FADED_MEDIUM_COLOR = (255, 255, 128)
FADED_LOW_COLOR = (255, 210, 128)
FADED_NONE_COLOR = (255, 128, 128)

FONT_PATH = "/Library/Fonts/Arial Bold.ttf"
SUMMARY_TITLE_FONT_PATH = "/Library/Fonts/Courier New Bold.ttf"
SUMMARY_TITLE_TEXT_COLOR = (0, 0, 0)
SUMMARY_TITLE_TEXT_SIZE = 24
SUMMARY_TILE_TEXT_COLOR = (255, 255, 255)
TILE_TEXT_COLOR = (0, 0, 0)
TILE_TEXT_SIZE = 36
TILE_TEXT_BACKGROUND_COLOR = (255, 255, 255)
TILE_TEXT_W_BORDER = 5
TILE_TEXT_H_BORDER = 4

HSV_PURPLE = 270
HSV_PINK = 330


def get_num_tiles(rows, cols, row_tile_size, col_tile_size):
  """
  Obtain the number of vertical and horizontal tiles that an image can be divided into given a row tile size and
  a column tile size.

  Args:
    rows: Number of rows.
    cols: Number of columns.
    row_tile_size: Number of pixels in a tile row.
    col_tile_size: Number of pixels in a tile column.

  Returns:
    Tuple consisting of the number of vertical tiles and the number of horizontal tiles that the image can be divided
    into given the row tile size and the column tile size.
  """
  num_row_tiles = math.ceil(rows / row_tile_size)
  num_col_tiles = math.ceil(cols / col_tile_size)
  return num_row_tiles, num_col_tiles


def get_tile_indices(rows, cols, row_tile_size, col_tile_size):
  """
  Obtain a list of tile coordinates (starting row, ending row, starting column, ending column, row number, column number).

  Args:
    rows: Number of rows.
    cols: Number of columns.
    row_tile_size: Number of pixels in a tile row.
    col_tile_size: Number of pixels in a tile column.

  Returns:
    List of tuples representing tile coordinates consisting of starting row, ending row,
    starting column, ending column, row number, column number.
  """
  indices = list()
  num_row_tiles, num_col_tiles = get_num_tiles(rows, cols, row_tile_size, col_tile_size)
  for r in range(0, num_row_tiles):
    start_r = r * row_tile_size
    end_r = ((r + 1) * row_tile_size) if (r < num_row_tiles - 1) else rows
    for c in range(0, num_col_tiles):
      start_c = c * col_tile_size
      end_c = ((c + 1) * col_tile_size) if (c < num_col_tiles - 1) else cols
      indices.append((start_r, end_r, start_c, end_c, r + 1, c + 1))
  return indices


def create_summary_pil_img(np_img, title_area_height, row_tile_size, col_tile_size, num_row_tiles, num_col_tiles):
  """
  Create a PIL summary image including top title area and right side and bottom padding.

  Args:
    np_img: Image as a NumPy array.
    title_area_height: Height of the title area at the top of the summary image.
    row_tile_size: The tile size in rows.
    col_tile_size: The tile size in columns.
    num_row_tiles: The number of row tiles.
    num_col_tiles: The number of column tiles.

  Returns:
    Summary image as a PIL image. This image contains the image data specified by the np_img input and also has
    potentially a top title area and right side and bottom padding.
  """
  r = row_tile_size * num_row_tiles + title_area_height
  c = col_tile_size * num_col_tiles
  summary_img = np.zeros([r, c, np_img.shape[2]], dtype=np.uint8)
  # add gray edges so that tile text does not get cut off
  summary_img.fill(120)
  # color title area white
  summary_img[0:title_area_height, 0:summary_img.shape[1]].fill(255)
  summary_img[title_area_height:np_img.shape[0] + title_area_height, 0:np_img.shape[1]] = np_img
  summary = util.np_to_pil(summary_img)
  return summary


def generate_tile_summaries(tile_sum, np_img, display=True, save_summary=False):
  """
  Generate summary images/thumbnails showing a 'heatmap' representation of the tissue segmentation of all tiles.

  Args:
    tile_sum: TileSummary object.
    np_img: Image as a NumPy array.
    display: If True, display tile summary to screen.
    save_summary: If True, save tile summary images.
  """
  z = 300  # height of area at top of summary slide
  slide_num = tile_sum.slide_num
  rows = tile_sum.scaled_h
  cols = tile_sum.scaled_w
  row_tile_size = tile_sum.scaled_tile_h
  col_tile_size = tile_sum.scaled_tile_w
  num_row_tiles, num_col_tiles = get_num_tiles(rows, cols, row_tile_size, col_tile_size)
  summary = create_summary_pil_img(np_img, z, row_tile_size, col_tile_size, num_row_tiles, num_col_tiles)
  draw = ImageDraw.Draw(summary)

  original_img_path = slide.get_training_image_path(slide_num)
  np_orig = slide.open_image_np(original_img_path)
  summary_orig = create_summary_pil_img(np_orig, z, row_tile_size, col_tile_size, num_row_tiles, num_col_tiles)
  draw_orig = ImageDraw.Draw(summary_orig)

  for t in tile_sum.tiles:
    border_color = tile_border_color(t.tissue_percentage)
    tile_border(draw, t.r_s + z, t.r_e + z, t.c_s, t.c_e, border_color)
    tile_border(draw_orig, t.r_s + z, t.r_e + z, t.c_s, t.c_e, border_color)

  summary_txt = summary_title(tile_sum) + "\n" + summary_stats(tile_sum)

  summary_font = ImageFont.truetype(SUMMARY_TITLE_FONT_PATH, size=SUMMARY_TITLE_TEXT_SIZE)
  draw.text((5, 5), summary_txt, SUMMARY_TITLE_TEXT_COLOR, font=summary_font)
  draw_orig.text((5, 5), summary_txt, SUMMARY_TITLE_TEXT_COLOR, font=summary_font)

  if DISPLAY_TILE_SUMMARY_LABELS:
    count = 0
    for t in tile_sum.tiles:
      count += 1
      label = "R%d\nC%d" % (t.r, t.c)
      font = ImageFont.truetype(FONT_PATH, size=TILE_LABEL_TEXT_SIZE)
      # drop shadow behind text
      draw.text(((t.c_s + 3), (t.r_s + 3 + z)), label, (0, 0, 0), font=font)
      draw_orig.text(((t.c_s + 3), (t.r_s + 3 + z)), label, (0, 0, 0), font=font)

      draw.text(((t.c_s + 2), (t.r_s + 2 + z)), label, SUMMARY_TILE_TEXT_COLOR, font=font)
      draw_orig.text(((t.c_s + 2), (t.r_s + 2 + z)), label, SUMMARY_TILE_TEXT_COLOR, font=font)

  if display:
    summary.show()
    summary_orig.show()
  if save_summary:
    save_tile_summary_image(summary, slide_num)
    save_tile_summary_on_original_image(summary_orig, slide_num)


def generate_top_tile_summaries(tile_sum, np_img, display=True, save_summary=False, show_top_stats=True,
                                label_all_tiles=LABEL_ALL_TILES_IN_TOP_TILE_SUMMARY,
                                border_all_tiles=BORDER_ALL_TILES_IN_TOP_TILE_SUMMARY):
  """
  Generate summary images/thumbnails showing the top tiles ranked by score.

  Args:
    tile_sum: TileSummary object.
    np_img: Image as a NumPy array.
    display: If True, display top tiles to screen.
    save_summary: If True, save top tiles images.
    show_top_stats: If True, append top tile score stats to image.
    label_all_tiles: If True, label all tiles. If False, label only top tiles.
  """
  z = 300  # height of area at top of summary slide
  slide_num = tile_sum.slide_num
  rows = tile_sum.scaled_h
  cols = tile_sum.scaled_w
  row_tile_size = tile_sum.scaled_tile_h
  col_tile_size = tile_sum.scaled_tile_w
  num_row_tiles, num_col_tiles = get_num_tiles(rows, cols, row_tile_size, col_tile_size)
  summary = create_summary_pil_img(np_img, z, row_tile_size, col_tile_size, num_row_tiles, num_col_tiles)
  draw = ImageDraw.Draw(summary)

  original_img_path = slide.get_training_image_path(slide_num)
  np_orig = slide.open_image_np(original_img_path)
  summary_orig = create_summary_pil_img(np_orig, z, row_tile_size, col_tile_size, num_row_tiles, num_col_tiles)
  draw_orig = ImageDraw.Draw(summary_orig)

  if border_all_tiles:
    for t in tile_sum.tiles:
      border_color = faded_tile_border_color(t.tissue_percentage)
      tile_border(draw, t.r_s + z, t.r_e + z, t.c_s, t.c_e, border_color, border_size=1)
      tile_border(draw_orig, t.r_s + z, t.r_e + z, t.c_s, t.c_e, border_color, border_size=1)

  tbs = TILE_BORDER_SIZE
  top_tiles = tile_sum.top_tiles()
  for t in top_tiles:
    border_color = tile_border_color(t.tissue_percentage)
    tile_border(draw, t.r_s + z, t.r_e + z, t.c_s, t.c_e, border_color)
    tile_border(draw_orig, t.r_s + z, t.r_e + z, t.c_s, t.c_e, border_color)
    if border_all_tiles:
      tile_border(draw, t.r_s + z + tbs, t.r_e + z - tbs, t.c_s + tbs, t.c_e - tbs, (0, 0, 0))
      tile_border(draw_orig, t.r_s + z + tbs, t.r_e + z - tbs, t.c_s + tbs, t.c_e - tbs, (0, 0, 0))

  summary_title = "Slide %03d Top Tile Summary:" % slide_num
  summary_txt = summary_title + "\n" + summary_stats(tile_sum)

  summary_font = ImageFont.truetype(SUMMARY_TITLE_FONT_PATH, size=SUMMARY_TITLE_TEXT_SIZE)
  draw.text((5, 5), summary_txt, SUMMARY_TITLE_TEXT_COLOR, font=summary_font)
  draw_orig.text((5, 5), summary_txt, SUMMARY_TITLE_TEXT_COLOR, font=summary_font)

  tiles_to_label = tile_sum.tiles if label_all_tiles else top_tiles
  h_offset = TILE_BORDER_SIZE + 2
  v_offset = TILE_BORDER_SIZE
  h_ds_offset = TILE_BORDER_SIZE + 3
  v_ds_offset = TILE_BORDER_SIZE + 1
  for t in tiles_to_label:
    label = "R%d\nC%d" % (t.r, t.c)
    font = ImageFont.truetype(FONT_PATH, size=TILE_LABEL_TEXT_SIZE)
    # drop shadow behind text
    draw.text(((t.c_s + h_ds_offset), (t.r_s + v_ds_offset + z)), label, (0, 0, 0), font=font)
    draw_orig.text(((t.c_s + h_ds_offset), (t.r_s + v_ds_offset + z)), label, (0, 0, 0), font=font)

    draw.text(((t.c_s + h_offset), (t.r_s + v_offset + z)), label, SUMMARY_TILE_TEXT_COLOR, font=font)
    draw_orig.text(((t.c_s + h_offset), (t.r_s + v_offset + z)), label, SUMMARY_TILE_TEXT_COLOR, font=font)

  if show_top_stats:
    summary = add_tile_stats_to_top_tile_summary(summary, top_tiles, z)
    summary_orig = add_tile_stats_to_top_tile_summary(summary_orig, top_tiles, z)

  if display:
    summary.show()
    summary_orig.show()
  if save_summary:
    save_top_tiles_image(summary, slide_num)
    save_top_tiles_on_original_image(summary_orig, slide_num)


def add_tile_stats_to_top_tile_summary(pil_img, tiles, z):
  np_sum = util.pil_to_np_rgb(pil_img)
  sum_r, sum_c, sum_ch = np_sum.shape
  np_stats = np_tile_stat_img(tiles)
  st_r, st_c, _ = np_stats.shape
  combo_c = sum_c + st_c
  combo_r = max(sum_r, st_r + z)
  combo = np.zeros([combo_r, combo_c, sum_ch], dtype=np.uint8)
  combo.fill(255)
  combo[0:sum_r, 0:sum_c] = np_sum
  combo[z:st_r + z, sum_c:sum_c + st_c] = np_stats
  result = util.np_to_pil(combo)
  return result


def np_tile_stat_img(tiles):
  """
  Generate tile scoring statistics for a list of tiles and return the result as a NumPy array image.

  Args:
    tiles: List of tiles (such as top tiles)

  Returns:
    Tile scoring statistics converted into an NumPy array image.
  """
  tt = sorted(tiles, key=lambda t: (t.r, t.c), reverse=False)
  tile_stats = "Tile Score Statistics:\n"
  count = 0
  for t in tt:
    if count > 0:
      tile_stats += "\n"
    count += 1
    tup = (t.r, t.c, t.rank, t.tissue_percentage, t.color_factor, t.s_and_v_factor, t.quantity_factor, t.score)
    tile_stats += "R%03d C%03d #%003d TP:%6.2f%% CF:%4.0f SVF:%4.2f QF:%4.2f S:%0.4f" % tup
  np_stats = np_text(tile_stats, font_path=SUMMARY_TITLE_FONT_PATH, font_size=14)
  return np_stats


def tile_border_color(tissue_percentage):
  """
  Obtain the corresponding tile border color for a particular tile tissue percentage.

  Args:
    tissue_percentage: The tile tissue percentage

  Returns:
    The tile border color corresponding to the tile tissue percentage.
  """
  if tissue_percentage >= TISSUE_HIGH_THRESH:
    border_color = HIGH_COLOR
  elif (tissue_percentage >= TISSUE_LOW_THRESH) and (tissue_percentage < TISSUE_HIGH_THRESH):
    border_color = MEDIUM_COLOR
  elif (tissue_percentage > 0) and (tissue_percentage < TISSUE_LOW_THRESH):
    border_color = LOW_COLOR
  else:
    border_color = NONE_COLOR
  return border_color


def faded_tile_border_color(tissue_percentage):
  """
  Obtain the corresponding faded tile border color for a particular tile tissue percentage.

  Args:
    tissue_percentage: The tile tissue percentage

  Returns:
    The faded tile border color corresponding to the tile tissue percentage.
  """
  if tissue_percentage >= TISSUE_HIGH_THRESH:
    border_color = FADED_THRESH_COLOR
  elif (tissue_percentage >= TISSUE_LOW_THRESH) and (tissue_percentage < TISSUE_HIGH_THRESH):
    border_color = FADED_MEDIUM_COLOR
  elif (tissue_percentage > 0) and (tissue_percentage < TISSUE_LOW_THRESH):
    border_color = FADED_LOW_COLOR
  else:
    border_color = FADED_NONE_COLOR
  return border_color


def summary_title(tile_summary):
  """
  Obtain tile summary title.

  Args:
    tile_summary: TileSummary object.

  Returns:
     The tile summary title.
  """
  return "Slide %03d Tile Summary:" % tile_summary.slide_num


def summary_stats(tile_summary):
  """
  Obtain various stats about the slide tiles.

  Args:
    tile_summary: TileSummary object.

  Returns:
     Various stats about the slide tiles as a string.
  """
  return "Original Dimensions: %dx%d\n" % (tile_summary.orig_w, tile_summary.orig_h) + \
         "Original Tile Size: %dx%d\n" % (tile_summary.orig_tile_w, tile_summary.orig_tile_h) + \
         "Scale Factor: 1/%dx\n" % tile_summary.scale_factor + \
         "Scaled Dimensions: %dx%d\n" % (tile_summary.scaled_w, tile_summary.scaled_h) + \
         "Scaled Tile Size: %dx%d\n" % (tile_summary.scaled_tile_w, tile_summary.scaled_tile_w) + \
         "Total Mask: %3.2f%%, Total Tissue: %3.2f%%\n" % (
           tile_summary.mask_percentage(), tile_summary.tissue_percentage) + \
         "Tiles: %dx%d = %d\n" % (tile_summary.num_col_tiles, tile_summary.num_row_tiles, tile_summary.count) + \
         " %5d (%5.2f%%) tiles >=%d%% tissue\n" % (
           tile_summary.high, tile_summary.high / tile_summary.count * 100, TISSUE_HIGH_THRESH) + \
         " %5d (%5.2f%%) tiles >=%d%% and <%d%% tissue\n" % (
           tile_summary.medium, tile_summary.medium / tile_summary.count * 100, TISSUE_LOW_THRESH,
           TISSUE_HIGH_THRESH) + \
         " %5d (%5.2f%%) tiles >0%% and <%d%% tissue\n" % (
           tile_summary.low, tile_summary.low / tile_summary.count * 100, TISSUE_LOW_THRESH) + \
         " %5d (%5.2f%%) tiles =0%% tissue" % (tile_summary.none, tile_summary.none / tile_summary.count * 100)


def tile_border(draw, r_s, r_e, c_s, c_e, color, border_size=TILE_BORDER_SIZE):
  """
  Draw a border around a tile with width TILE_BORDER_SIZE.

  Args:
    draw: Draw object for drawing on PIL image.
    r_s: Row starting pixel.
    r_e: Row ending pixel.
    c_s: Column starting pixel.
    c_e: Column ending pixel.
    color: Color of the border.
    border_size: Width of tile border in pixels.
  """
  for x in range(0, border_size):
    draw.rectangle([(c_s + x, r_s + x), (c_e - 1 - x, r_e - 1 - x)], outline=color)


def save_tile_summary_image(pil_img, slide_num):
  """
  Save a tile summary image and thumbnail to the file system.

  Args:
    pil_img: Image as a PIL Image.
    slide_num: The slide number.
  """
  t = Time()
  filepath = slide.get_tile_summary_image_path(slide_num)
  pil_img.save(filepath)
  print("%-20s | Time: %-14s  Name: %s" % ("Save Tile Sum", str(t.elapsed()), filepath))

  t = Time()
  thumbnail_filepath = slide.get_tile_summary_thumbnail_path(slide_num)
  slide.save_thumbnail(pil_img, slide.THUMBNAIL_SIZE, thumbnail_filepath)
  print("%-20s | Time: %-14s  Name: %s" % ("Save Tile Sum Thumb", str(t.elapsed()), thumbnail_filepath))


def save_top_tiles_image(pil_img, slide_num):
  """
  Save a top tiles image and thumbnail to the file system.

  Args:
    pil_img: Image as a PIL Image.
    slide_num: The slide number.
  """
  t = Time()
  filepath = slide.get_top_tiles_image_path(slide_num)
  pil_img.save(filepath)
  print("%-20s | Time: %-14s  Name: %s" % ("Save Top Tiles Image", str(t.elapsed()), filepath))

  t = Time()
  thumbnail_filepath = slide.get_top_tiles_thumbnail_path(slide_num)
  slide.save_thumbnail(pil_img, slide.THUMBNAIL_SIZE, thumbnail_filepath)
  print("%-20s | Time: %-14s  Name: %s" % ("Save Top Tiles Thumb", str(t.elapsed()), thumbnail_filepath))


def save_tile_summary_on_original_image(pil_img, slide_num):
  """
  Save a tile summary on original image and thumbnail to the file system.

  Args:
    pil_img: Image as a PIL Image.
    slide_num: The slide number.
  """
  t = Time()
  filepath = slide.get_tile_summary_on_original_image_path(slide_num)
  pil_img.save(filepath)
  print("%-20s | Time: %-14s  Name: %s" % ("Save Tile Sum Orig", str(t.elapsed()), filepath))

  t = Time()
  thumbnail_filepath = slide.get_tile_summary_on_original_thumbnail_path(slide_num)
  slide.save_thumbnail(pil_img, slide.THUMBNAIL_SIZE, thumbnail_filepath)
  print(
    "%-20s | Time: %-14s  Name: %s" % ("Save Tile Sum Orig T", str(t.elapsed()), thumbnail_filepath))


def save_top_tiles_on_original_image(pil_img, slide_num):
  """
  Save a top tiles on original image and thumbnail to the file system.

  Args:
    pil_img: Image as a PIL Image.
    slide_num: The slide number.
  """
  t = Time()
  filepath = slide.get_top_tiles_on_original_image_path(slide_num)
  pil_img.save(filepath)
  print("%-20s | Time: %-14s  Name: %s" % ("Save Top Orig", str(t.elapsed()), filepath))

  t = Time()
  thumbnail_filepath = slide.get_top_tiles_on_original_thumbnail_path(slide_num)
  slide.save_thumbnail(pil_img, slide.THUMBNAIL_SIZE, thumbnail_filepath)
  print(
    "%-20s | Time: %-14s  Name: %s" % ("Save Top Orig Thumb", str(t.elapsed()), thumbnail_filepath))


def summary_and_tiles(slide_num, display=True, save_summary=False, save_data=True, save_top_tiles=True):
  """
  Generate tile summary and top tiles for slide.

  Args:
    slide_num: The slide number.
    display: If True, display tile summary to screen.
    save_summary: If True, save tile summary images.
    save_data: If True, save tile data to csv file.
    save_top_tiles: If True, save top tiles to files.

  """
  img_path = slide.get_filter_image_result(slide_num)
  np_img = slide.open_image_np(img_path)

  tile_sum = score_tiles(slide_num, np_img)
  if save_data:
    save_tile_data(tile_sum)
  generate_tile_summaries(tile_sum, np_img, display=display, save_summary=save_summary)
  generate_top_tile_summaries(tile_sum, np_img, display=display, save_summary=save_summary)
  if save_top_tiles:
    for tile in tile_sum.top_tiles():
      tile.save_tile()
  return tile_sum


def save_tile_data(tile_summary):
  """
  Save tile data to csv file.

  Args
    tile_summary: TimeSummary object.
  """

  time = Time()

  csv = summary_title(tile_summary) + "\n" + summary_stats(tile_summary)

  csv += "\n\n\nTile Num,Row,Column,Tissue %,Tissue Quantity,Col Start,Row Start,Col End,Row End,Col Size,Row Size," + \
         "Original Col Start,Original Row Start,Original Col End,Original Row End,Original Col Size,Original Row Size," + \
         "Color Factor,S and V Factor,Quantity Factor,Score\n"

  for t in tile_summary.tiles:
    line = "%d,%d,%d,%4.2f,%s,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%4.0f,%4.2f,%4.2f,%0.4f\n" % (
      t.tile_num, t.r, t.c, t.tissue_percentage, t.tissue_quantity().name, t.c_s, t.r_s, t.c_e, t.r_e, t.c_e - t.c_s,
      t.r_e - t.r_s, t.o_c_s, t.o_r_s, t.o_c_e, t.o_r_e, t.o_c_e - t.o_c_s, t.o_r_e - t.o_r_s, t.color_factor,
      t.s_and_v_factor, t.quantity_factor, t.score)
    csv += line

  data_path = slide.get_tile_data_path(tile_summary.slide_num)
  csv_file = open(data_path, "w")
  csv_file.write(csv)
  csv_file.close()

  print("%-20s | Time: %-14s  Name: %s" % ("Save Tile Data", str(time.elapsed()), data_path))


def tile_to_pil_tile(tile):
  """
  Convert tile information into the corresponding tile as a PIL image read from the whole-slide image file.

  Args:
    tile: Tile object.

  Return:
    Tile as a PIL image.
  """
  t = tile
  slide_filepath = slide.get_training_slide_path(t.slide_num)
  s = slide.open_slide(slide_filepath)

  x, y = t.o_c_s, t.o_r_s
  w, h = t.o_c_e - t.o_c_s, t.o_r_e - t.o_r_s
  tile_region = s.read_region((x, y), 0, (w, h))
  # RGBA to RGB
  pil_img = tile_region.convert("RGB")
  return pil_img


def tile_to_np_tile(tile):
  """
  Convert tile information into the corresponding tile as a NumPy image read from the whole-slide image file.

  Args:
    tile: Tile object.

  Return:
    Tile as a NumPy image.
  """
  pil_img = tile_to_pil_tile(tile)
  np_img = util.pil_to_np_rgb(pil_img)
  return np_img


def save_display_tile(tile, save=True, display=False):
  """
  Save and/or display a tile image.

  Args:
    tile: Tile object.
    save: If True, save tile image.
    display: If True, dispaly tile image.
  """
  tile_pil_img = tile_to_pil_tile(tile)

  if save:
    t = Time()
    img_path = slide.get_tile_image_path(tile)
    dir = os.path.dirname(img_path)
    if not os.path.exists(dir):
      os.makedirs(dir)
    tile_pil_img.save(img_path)
    print("%-20s | Time: %-14s  Name: %s" % ("Save Tile", str(t.elapsed()), img_path))

  if display:
    tile_pil_img.show()


def score_tiles(slide_num, np_img=None, dimensions=None, small_tile_in_tile=False):
  """
  Score all tiles for a slide and return the results in a TileSummary object.

  Args:
    slide_num: The slide number.
    np_img: Optional image as a NumPy array.
    dimensions: Optional tuple consisting of (original width, original height, new width, new height). Used for dynamic
      tile retrieval.
    small_tile_in_tile: If True, include the small NumPy image in the Tile objects.

  Returns:
    TileSummary object which includes a list of Tile objects containing information about each tile.
  """
  if dimensions is None:
    img_path = slide.get_filter_image_result(slide_num)
    o_w, o_h, w, h = slide.parse_dimensions_from_image_filename(img_path)
  else:
    o_w, o_h, w, h = dimensions

  if np_img is None:
    np_img = slide.open_image_np(img_path)

  row_tile_size = round(ROW_TILE_SIZE / slide.SCALE_FACTOR)  # use round?
  col_tile_size = round(COL_TILE_SIZE / slide.SCALE_FACTOR)  # use round?

  num_row_tiles, num_col_tiles = get_num_tiles(h, w, row_tile_size, col_tile_size)

  tile_sum = TileSummary(slide_num=slide_num,
                         orig_w=o_w,
                         orig_h=o_h,
                         orig_tile_w=COL_TILE_SIZE,
                         orig_tile_h=ROW_TILE_SIZE,
                         scaled_w=w,
                         scaled_h=h,
                         scaled_tile_w=col_tile_size,
                         scaled_tile_h=row_tile_size,
                         tissue_percentage=filter.tissue_percent(np_img),
                         num_col_tiles=num_col_tiles,
                         num_row_tiles=num_row_tiles)

  count = 0
  high = 0
  medium = 0
  low = 0
  none = 0
  tile_indices = get_tile_indices(h, w, row_tile_size, col_tile_size)
  for t in tile_indices:
    count += 1  # tile_num
    r_s, r_e, c_s, c_e, r, c = t
    np_tile = np_img[r_s:r_e, c_s:c_e]
    t_p = filter.tissue_percent(np_tile)
    amount = tissue_quantity(t_p)
    if amount == TissueQuantity.HIGH:
      high += 1
    elif amount == TissueQuantity.MEDIUM:
      medium += 1
    elif amount == TissueQuantity.LOW:
      low += 1
    elif amount == TissueQuantity.NONE:
      none += 1
    o_c_s, o_r_s = slide.small_to_large_mapping((c_s, r_s), (o_w, o_h))
    o_c_e, o_r_e = slide.small_to_large_mapping((c_e, r_e), (o_w, o_h))

    # pixel adjustment in case tile dimension too large (for example, 1025 instead of 1024)
    if (o_c_e - o_c_s) > COL_TILE_SIZE:
      o_c_e -= 1
    if (o_r_e - o_r_s) > ROW_TILE_SIZE:
      o_r_e -= 1

    score, color_factor, s_and_v_factor, quantity_factor = score_tile(np_tile, t_p, slide_num, r, c)

    np_scaled_tile = np_tile if small_tile_in_tile else None
    tile = Tile(tile_sum, slide_num, np_scaled_tile, count, r, c, r_s, r_e, c_s, c_e, o_r_s, o_r_e, o_c_s,
                o_c_e, t_p, color_factor, s_and_v_factor, quantity_factor, score)
    tile_sum.tiles.append(tile)

  tile_sum.count = count
  tile_sum.high = high
  tile_sum.medium = medium
  tile_sum.low = low
  tile_sum.none = none

  tiles_by_score = tile_sum.tiles_by_score()
  rank = 0
  for t in tiles_by_score:
    rank += 1
    t.rank = rank

  return tile_sum


def score_tile(np_tile, tissue_percent, slide_num, row, col):
  """
  Score tile based on tissue percentage, color factor, saturation/value factor, and tissue quantity factor.

  Args:
    np_tile: Tile as NumPy array.
    tissue_percent: The percentage of the tile judged to be tissue.
    slide_num: Slide number.
    row: Tile row.
    col: Tile column.

  Returns tuple consisting of score, color factor, saturation/value factor, and tissue quantity factor.
  """
  color_factor = hsv_purple_pink_factor(np_tile)
  s_and_v_factor = hsv_saturation_and_value_factor(np_tile)
  amount = tissue_quantity(tissue_percent)
  quantity_factor = tissue_quantity_factor(amount)
  combined_factor = color_factor * s_and_v_factor * quantity_factor
  score = (tissue_percent ** 2) * np.log(1 + combined_factor) / 1000.0
  # scale score to between 0 and 1
  score = 1.0 - (10.0 / (10.0 + score))
  return score, color_factor, s_and_v_factor, quantity_factor


def tissue_quantity_factor(amount):
  """
  Obtain a scoring factor based on the quantity of tissue in a tile.

  Args:
    amount: Tissue amount as a TissueQuantity enum value.

  Returns:
    Scoring factor based on the tile tissue quantity.
  """
  if amount == TissueQuantity.HIGH:
    quantity_factor = 1.0
  elif amount == TissueQuantity.MEDIUM:
    quantity_factor = 0.2
  elif amount == TissueQuantity.LOW:
    quantity_factor = 0.1
  else:
    quantity_factor = 0.0
  return quantity_factor


def tissue_quantity(tissue_percentage):
  """
  Obtain TissueQuantity enum member (HIGH, MEDIUM, LOW, or NONE) for corresponding tissue percentage.

  Args:
    tissue_percentage: The tile tissue percentage.

  Returns:
    TissueQuantity enum member (HIGH, MEDIUM, LOW, or NONE).
  """
  if tissue_percentage >= TISSUE_HIGH_THRESH:
    return TissueQuantity.HIGH
  elif (tissue_percentage >= TISSUE_LOW_THRESH) and (tissue_percentage < TISSUE_HIGH_THRESH):
    return TissueQuantity.MEDIUM
  elif (tissue_percentage > 0) and (tissue_percentage < TISSUE_LOW_THRESH):
    return TissueQuantity.LOW
  else:
    return TissueQuantity.NONE


def image_list_to_tiles(image_num_list, display=False, save_summary=True, save_data=True, save_top_tiles=True):
  """
  Generate tile summaries and tiles for a list of images.

  Args:
    image_num_list: List of image numbers.
    display: If True, display tile summary images to screen.
    save_summary: If True, save tile summary images.
    save_data: If True, save tile data to csv file.
    save_top_tiles: If True, save top tiles to files.
  """
  tile_summaries_dict = dict()
  for slide_num in image_num_list:
    tile_summary = summary_and_tiles(slide_num, display, save_summary, save_data, save_top_tiles)
    tile_summaries_dict[slide_num] = tile_summary
  return image_num_list, tile_summaries_dict


def image_range_to_tiles(start_ind, end_ind, display=False, save_summary=True, save_data=True, save_top_tiles=True):
  """
  Generate tile summaries and tiles for a range of images.

  Args:
    start_ind: Starting index (inclusive).
    end_ind: Ending index (inclusive).
    display: If True, display tile summary images to screen.
    save_summary: If True, save tile summary images.
    save_data: If True, save tile data to csv file.
    save_top_tiles: If True, save top tiles to files.
  """
  image_num_list = list()
  tile_summaries_dict = dict()
  for slide_num in range(start_ind, end_ind + 1):
    tile_summary = summary_and_tiles(slide_num, display, save_summary, save_data, save_top_tiles)
    image_num_list.append(slide_num)
    tile_summaries_dict[slide_num] = tile_summary
  return image_num_list, tile_summaries_dict


def singleprocess_filtered_images_to_tiles(display=False, save_summary=True, save_data=True, save_top_tiles=True,
                                           html=True, image_num_list=None):
  """
  Generate tile summaries and tiles for training images using a single process.

  Args:
    display: If True, display tile summary images to screen.
    save_summary: If True, save tile summary images.
    save_data: If True, save tile data to csv file.
    save_top_tiles: If True, save top tiles to files.
    html: If True, generate HTML page to display tiled images
    image_num_list: Optionally specify a list of image slide numbers.
  """
  t = Time()
  print("Generating tile summaries\n")

  if image_num_list is not None:
    image_num_list, tile_summaries_dict = image_list_to_tiles(image_num_list, display, save_summary, save_data,
                                                              save_top_tiles)
  else:
    num_training_slides = slide.get_num_training_slides()
    image_num_list, tile_summaries_dict = image_range_to_tiles(1, num_training_slides, display, save_summary, save_data,
                                                               save_top_tiles)

  print("Time to generate tile summaries: %s\n" % str(t.elapsed()))

  if html:
    generate_tiled_html_result(image_num_list, tile_summaries_dict, save_data)


def multiprocess_filtered_images_to_tiles(display=False, save_summary=True, save_data=True, save_top_tiles=True,
                                          html=True, image_num_list=None):
  """
  Generate tile summaries and tiles for all training images using multiple processes (one process per core).

  Args:
    display: If True, display images to screen (multiprocessed display not recommended).
    save_summary: If True, save tile summary images.
    save_data: If True, save tile data to csv file.
    save_top_tiles: If True, save top tiles to files.
    html: If True, generate HTML page to display tiled images.
    image_num_list: Optionally specify a list of image slide numbers.
  """
  timer = Time()
  print("Generating tile summaries (multiprocess)\n")

  if save_summary and not os.path.exists(slide.TILE_SUMMARY_DIR):
    os.makedirs(slide.TILE_SUMMARY_DIR)

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
      tasks.append((sublist, display, save_summary, save_data, save_top_tiles))
      print("Task #" + str(num_process) + ": Process slides " + str(sublist))
    else:
      tasks.append((start_index, end_index, display, save_summary, save_data, save_top_tiles))
      if start_index == end_index:
        print("Task #" + str(num_process) + ": Process slide " + str(start_index))
      else:
        print("Task #" + str(num_process) + ": Process slides " + str(start_index) + " to " + str(end_index))

  # start tasks
  results = []
  for t in tasks:
    if image_num_list is not None:
      results.append(pool.apply_async(image_list_to_tiles, t))
    else:
      results.append(pool.apply_async(image_range_to_tiles, t))

  slide_nums = list()
  tile_summaries_dict = dict()
  for result in results:
    image_nums, tile_summaries = result.get()
    slide_nums.extend(image_nums)
    tile_summaries_dict.update(tile_summaries)
    print("Done tiling slides: %s" % image_nums)

  if html:
    generate_tiled_html_result(slide_nums, tile_summaries_dict, save_data)

  print("Time to generate tile previews (multiprocess): %s\n" % str(timer.elapsed()))


def image_row(slide_num, tile_summary, data_link):
  """
  Generate HTML for viewing a tiled image.

  Args:
    slide_num: The slide number.
    tile_summary: TileSummary object.
    data_link: If True, add link to tile data csv file.

  Returns:
    HTML table row for viewing a tiled image.
  """
  orig_img = slide.get_training_image_path(slide_num)
  orig_thumb = slide.get_training_thumbnail_path(slide_num)
  filt_img = slide.get_filter_image_result(slide_num)
  filt_thumb = slide.get_filter_thumbnail_result(slide_num)
  sum_img = slide.get_tile_summary_image_path(slide_num)
  sum_thumb = slide.get_tile_summary_thumbnail_path(slide_num)
  osum_img = slide.get_tile_summary_on_original_image_path(slide_num)
  osum_thumb = slide.get_tile_summary_on_original_thumbnail_path(slide_num)
  top_img = slide.get_top_tiles_image_path(slide_num)
  top_thumb = slide.get_top_tiles_thumbnail_path(slide_num)
  otop_img = slide.get_top_tiles_on_original_image_path(slide_num)
  otop_thumb = slide.get_top_tiles_on_original_thumbnail_path(slide_num)
  html = "    <tr>\n" + \
         "      <td style=\"vertical-align: top\">\n" + \
         "        <a target=\"_blank\" href=\"%s\">S%03d Original<br/>\n" % (orig_img, slide_num) + \
         "          <img src=\"%s\" />\n" % (orig_thumb) + \
         "        </a>\n" + \
         "      </td>\n" + \
         "      <td style=\"vertical-align: top\">\n" + \
         "        <a target=\"_blank\" href=\"%s\">S%03d Filtered<br/>\n" % (filt_img, slide_num) + \
         "          <img src=\"%s\" />\n" % (filt_thumb) + \
         "        </a>\n" + \
         "      </td>\n"

  html += "      <td style=\"vertical-align: top\">\n" + \
          "        <a target=\"_blank\" href=\"%s\">S%03d Tiles<br/>\n" % (sum_img, slide_num) + \
          "          <img src=\"%s\" />\n" % (sum_thumb) + \
          "        </a>\n" + \
          "      </td>\n"

  html += "      <td style=\"vertical-align: top\">\n" + \
          "        <a target=\"_blank\" href=\"%s\">S%03d Tiles<br/>\n" % (osum_img, slide_num) + \
          "          <img src=\"%s\" />\n" % (osum_thumb) + \
          "        </a>\n" + \
          "      </td>\n"

  html += "      <td style=\"vertical-align: top\">\n"
  if data_link:
    html += "        <div style=\"white-space: nowrap;\">S%03d Tile Summary\n" % slide_num + \
            "        (<a target=\"_blank\" href=\"%s\">Data</a>)</div>\n" % slide.get_tile_data_path(slide_num)
  else:
    html += "        <div style=\"white-space: nowrap;\">S%03d Tile Summary</div>\n" % slide_num

  html += "        <div style=\"font-size: smaller; white-space: nowrap;\">\n" + \
          "          %s\n" % summary_stats(tile_summary).replace("\n", "<br/>\n          ") + \
          "        </div>\n" + \
          "      </td>\n"

  html += "      <td style=\"vertical-align: top\">\n" + \
          "        <a target=\"_blank\" href=\"%s\">S%03d Top Tiles<br/>\n" % (top_img, slide_num) + \
          "          <img src=\"%s\" />\n" % (top_thumb) + \
          "        </a>\n" + \
          "      </td>\n"

  html += "      <td style=\"vertical-align: top\">\n" + \
          "        <a target=\"_blank\" href=\"%s\">S%03d Top Tiles<br/>\n" % (otop_img, slide_num) + \
          "          <img src=\"%s\" />\n" % (otop_thumb) + \
          "        </a>\n" + \
          "      </td>\n"

  top_tiles = tile_summary.top_tiles()
  num_tiles = len(top_tiles)
  score_num = 0
  for t in top_tiles:
    score_num += 1
    t.tile_num = score_num
  # sort top tiles by rows and columns to make them easier to locate on HTML page
  top_tiles = sorted(top_tiles, key=lambda t: (t.r, t.c), reverse=False)

  html += "      <td style=\"vertical-align: top\">\n" + \
          "        <div style=\"white-space: nowrap;\">S%03d Top %d Tile Scores</div>\n" % (slide_num, num_tiles) + \
          "        <div style=\"font-size: smaller; white-space: nowrap;\">\n"

  html += "        <table>\n"
  MAX_TILES_PER_ROW = 15
  num_cols = math.ceil(num_tiles / MAX_TILES_PER_ROW)
  num_rows = num_tiles if num_tiles < MAX_TILES_PER_ROW else MAX_TILES_PER_ROW
  for row in range(num_rows):
    html += "          <tr>\n"
    for col in range(num_cols):
      html += "            <td style=\"border: none;\">"
      tile_num = row + (col * num_rows) + 1
      if tile_num <= num_tiles:
        t = top_tiles[tile_num - 1]
        label = "R%03d C%03d %0.4f (#%02d)" % (t.r, t.c, t.score, t.tile_num)
        tile_img_path = slide.get_tile_image_path(t)
        html += "<a target=\"_blank\" href=\"%s\">%s</a>" % (tile_img_path, label)
      else:
        html += "&nbsp;"
      html += "</td>\n"
    html += "          </tr>\n"
  html += "        </table>\n"

  html += "        </div>\n"
  html += "      </td>\n"

  html += "    </tr>\n"
  return html


def generate_tiled_html_result(slide_nums, tile_summaries_dict, data_link):
  """
  Generate HTML to view the tiled images.

  Args:
    slide_nums: List of slide numbers.
    tile_summaries_dict: Dictionary of TileSummary objects keyed by slide number.
    data_link: If True, add link to tile data csv file.
  """
  slide_nums = sorted(slide_nums)
  if not slide.TILE_SUMMARY_PAGINATE:
    html = ""
    html += filter.html_header("Tiles")

    html += "  <table>\n"
    for slide_num in slide_nums:
      html += image_row(slide_num, data_link)
    html += "  </table>\n"

    html += filter.html_footer()
    text_file = open(os.path.join(slide.TILE_SUMMARY_HTML_DIR, "tiles.html"), "w")
    text_file.write(html)
    text_file.close()
  else:
    total_len = len(slide_nums)
    page_size = slide.TILE_SUMMARY_PAGINATION_SIZE
    num_pages = math.ceil(total_len / page_size)
    for page_num in range(1, num_pages + 1):
      start_index = (page_num - 1) * page_size
      end_index = (page_num * page_size) if (page_num < num_pages) else total_len
      page_slide_nums = slide_nums[start_index:end_index]

      html = ""
      html += filter.html_header("Tiles, Page %d" % page_num)

      html += "  <div style=\"font-size: 20px\">"
      if page_num > 1:
        if page_num == 2:
          html += "<a href=\"tiles.html\">&lt;</a> "
        else:
          html += "<a href=\"tiles-%d.html\">&lt;</a> " % (page_num - 1)
      html += "Page %d" % page_num
      if page_num < num_pages:
        html += " <a href=\"tiles-%d.html\">&gt;</a> " % (page_num + 1)
      html += "</div>\n"

      html += "  <table>\n"
      for slide_num in page_slide_nums:
        tile_summary = tile_summaries_dict[slide_num]
        html += image_row(slide_num, tile_summary, data_link)
      html += "  </table>\n"

      html += filter.html_footer()
      if page_num == 1:
        text_file = open(os.path.join(slide.TILE_SUMMARY_HTML_DIR, "tiles.html"), "w")
      else:
        text_file = open(os.path.join(slide.TILE_SUMMARY_HTML_DIR, "tiles-%d.html" % page_num), "w")
      text_file.write(html)
      text_file.close()


def np_hsv_hue_histogram(h):
  """
  Create Matplotlib histogram of hue values for an HSV image and return the histogram as a NumPy array image.

  Args:
    h: Hue values as a 1-dimensional int NumPy array (scaled 0 to 360)

  Returns:
    Matplotlib histogram of hue values converted to a NumPy array image.
  """
  figure = plt.figure()
  canvas = figure.canvas
  _, _, patches = plt.hist(h, bins=360)
  plt.title("HSV Hue Histogram, mean=%3.1f, std=%3.1f" % (np.mean(h), np.std(h)))

  bin_num = 0
  for patch in patches:
    rgb_color = colorsys.hsv_to_rgb(bin_num / 360.0, 1, 1)
    patch.set_facecolor(rgb_color)
    bin_num += 1

  canvas.draw()
  w, h = canvas.get_width_height()
  np_hist = np.fromstring(canvas.get_renderer().tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
  plt.close(figure)
  util.np_info(np_hist)
  return np_hist


def np_histogram(data, title, bins="auto"):
  """
  Create Matplotlib histogram and return it as a NumPy array image.

  Args:
    data: Data to plot in the histogram.
    title: Title of the histogram.
    bins: Number of histogram bins, "auto" by default.

  Returns:
    Matplotlib histogram as a NumPy array image.
  """
  figure = plt.figure()
  canvas = figure.canvas
  plt.hist(data, bins=bins)
  plt.title(title)

  canvas.draw()
  w, h = canvas.get_width_height()
  np_hist = np.fromstring(canvas.get_renderer().tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
  plt.close(figure)
  util.np_info(np_hist)
  return np_hist


def np_hsv_saturation_histogram(s):
  """
  Create Matplotlib histogram of saturation values for an HSV image and return the histogram as a NumPy array image.

  Args:
    s: Saturation values as a 1-dimensional float NumPy array

  Returns:
    Matplotlib histogram of saturation values converted to a NumPy array image.
  """
  title = "HSV Saturation Histogram, mean=%.2f, std=%.2f" % (np.mean(s), np.std(s))
  return np_histogram(s, title)


def np_hsv_value_histogram(v):
  """
  Create Matplotlib histogram of value values for an HSV image and return the histogram as a NumPy array image.

  Args:
    v: Value values as a 1-dimensional float NumPy array

  Returns:
    Matplotlib histogram of saturation values converted to a NumPy array image.
  """
  title = "HSV Value Histogram, mean=%.2f, std=%.2f" % (np.mean(v), np.std(v))
  return np_histogram(v, title)


def np_rgb_channel_histogram(rgb, ch_num, ch_name):
  """
  Create Matplotlib histogram of an RGB channel for an RGB image and return the histogram as a NumPy array image.

  Args:
    rgb: Image as RGB NumPy array.
    ch_num: Which channel (0=red, 1=green, 2=blue)
    ch_name: Channel name ("R", "G", "B")

  Returns:
    Matplotlib histogram of RGB channel converted to a NumPy array image.
  """

  ch = rgb[:, :, ch_num]
  ch = ch.flatten()
  title = "RGB %s Histogram, mean=%.2f, std=%.2f" % (ch_name, np.mean(ch), np.std(ch))
  return np_histogram(ch, title, bins=256)


def np_rgb_r_histogram(rgb):
  """
  Obtain RGB R channel histogram as a NumPy array image.

  Args:
    rgb: Image as RGB NumPy array.

  Returns:
     Histogram of RGB R channel as a NumPy array image.
  """
  hist = np_rgb_channel_histogram(rgb, 0, "R")
  return hist


def np_rgb_g_histogram(rgb):
  """
  Obtain RGB G channel histogram as a NumPy array image.

  Args:
    rgb: Image as RGB NumPy array.

  Returns:
     Histogram of RGB G channel as a NumPy array image.
  """
  hist = np_rgb_channel_histogram(rgb, 1, "G")
  return hist


def np_rgb_b_histogram(rgb):
  """
  Obtain RGB B channel histogram as a NumPy array image.

  Args:
    rgb: Image as RGB NumPy array.

  Returns:
     Histogram of RGB B channel as a NumPy array image.
  """
  hist = np_rgb_channel_histogram(rgb, 2, "B")
  return hist


def pil_hue_histogram(h):
  """
  Create Matplotlib histogram of hue values for an HSV image and return the histogram as a PIL image.

  Args:
    h: Hue values as a 1-dimensional int NumPy array (scaled 0 to 360)

  Returns:
    Matplotlib histogram of hue values converted to a PIL image.
  """
  np_hist = np_hsv_hue_histogram(h)
  pil_hist = util.np_to_pil(np_hist)
  return pil_hist


def display_image_with_hsv_hue_histogram(np_rgb, text=None, scale_up=False):
  """
  Display an image with its corresponding hue histogram.

  Args:
    np_rgb: RGB image tile as a NumPy array
    text: Optional text to display above image
    scale_up: If True, scale up image to display by slide.SCALE_FACTOR
  """
  hsv = filter.filter_rgb_to_hsv(np_rgb)
  h = filter.filter_hsv_to_h(hsv)
  np_hist = np_hsv_hue_histogram(h)
  hist_r, hist_c, _ = np_hist.shape

  if scale_up:
    np_rgb = np.repeat(np_rgb, slide.SCALE_FACTOR, axis=1)
    np_rgb = np.repeat(np_rgb, slide.SCALE_FACTOR, axis=0)

  img_r, img_c, img_ch = np_rgb.shape
  if text is not None:
    np_t = np_text(text)
    t_r, t_c, _ = np_t.shape
    t_i_c = max(t_c, img_c)
    t_i_r = t_r + img_r
    t_i = np.zeros([t_i_r, t_i_c, img_ch], dtype=np.uint8)
    t_i.fill(255)
    t_i[0:t_r, 0:t_c] = np_t
    t_i[t_r:t_r + img_r, 0:img_c] = np_rgb
    np_rgb = t_i  # for simplicity assign title+image to image
    img_r, img_c, img_ch = np_rgb.shape

  r = max(img_r, hist_r)
  c = img_c + hist_c
  combo = np.zeros([r, c, img_ch], dtype=np.uint8)
  combo.fill(255)
  combo[0:img_r, 0:img_c] = np_rgb
  combo[0:hist_r, img_c:c] = np_hist
  pil_combo = util.np_to_pil(combo)
  pil_combo.show()


def display_image(np_rgb, text=None, scale_up=False):
  """
  Display an image with optional text above image.

  Args:
    np_rgb: RGB image tile as a NumPy array
    text: Optional text to display above image
    scale_up: If True, scale up image to display by slide.SCALE_FACTOR
  """
  if scale_up:
    np_rgb = np.repeat(np_rgb, slide.SCALE_FACTOR, axis=1)
    np_rgb = np.repeat(np_rgb, slide.SCALE_FACTOR, axis=0)

  img_r, img_c, img_ch = np_rgb.shape
  if text is not None:
    np_t = np_text(text)
    t_r, t_c, _ = np_t.shape
    t_i_c = max(t_c, img_c)
    t_i_r = t_r + img_r
    t_i = np.zeros([t_i_r, t_i_c, img_ch], dtype=np.uint8)
    t_i.fill(255)
    t_i[0:t_r, 0:t_c] = np_t
    t_i[t_r:t_r + img_r, 0:img_c] = np_rgb
    np_rgb = t_i

  pil_img = util.np_to_pil(np_rgb)
  pil_img.show()


def display_image_with_hsv_histograms(np_rgb, text=None, scale_up=False):
  """
  Display an image with its corresponding HSV hue, saturation, and value histograms.

  Args:
    np_rgb: RGB image tile as a NumPy array
    text: Optional text to display above image
    scale_up: If True, scale up image to display by slide.SCALE_FACTOR
  """
  hsv = filter.filter_rgb_to_hsv(np_rgb)
  np_h = np_hsv_hue_histogram(filter.filter_hsv_to_h(hsv))
  np_s = np_hsv_saturation_histogram(filter.filter_hsv_to_s(hsv))
  np_v = np_hsv_value_histogram(filter.filter_hsv_to_v(hsv))
  h_r, h_c, _ = np_h.shape
  s_r, s_c, _ = np_s.shape
  v_r, v_c, _ = np_v.shape

  if scale_up:
    np_rgb = np.repeat(np_rgb, slide.SCALE_FACTOR, axis=1)
    np_rgb = np.repeat(np_rgb, slide.SCALE_FACTOR, axis=0)

  img_r, img_c, img_ch = np_rgb.shape
  if text is not None:
    np_t = np_text(text)
    t_r, t_c, _ = np_t.shape
    t_i_c = max(t_c, img_c)
    t_i_r = t_r + img_r
    t_i = np.zeros([t_i_r, t_i_c, img_ch], dtype=np.uint8)
    t_i.fill(255)
    t_i[0:t_r, 0:t_c] = np_t
    t_i[t_r:t_r + img_r, 0:img_c] = np_rgb
    np_rgb = t_i  # for simplicity assign title+image to image
    img_r, img_c, img_ch = np_rgb.shape

  hists_c = max(h_c, s_c, v_c)
  hists_r = h_r + s_r + v_r
  hists = np.zeros([hists_r, hists_c, img_ch], dtype=np.uint8)

  hists[0:h_r, 0:h_c] = np_h
  hists[h_r:h_r + s_r, 0:s_c] = np_s
  hists[h_r + s_r:h_r + s_r + v_r, 0:v_c] = np_v

  r = max(img_r, hists_r)
  c = img_c + hists_c
  combo = np.zeros([r, c, img_ch], dtype=np.uint8)
  combo.fill(255)
  combo[0:img_r, 0:img_c] = np_rgb
  combo[0:hists_r, img_c:c] = hists
  pil_combo = util.np_to_pil(combo)
  pil_combo.show()


def display_image_with_rgb_histograms(np_rgb, text=None, scale_up=False):
  """
  Display an image with its corresponding RGB histograms.

  Args:
    np_rgb: RGB image tile as a NumPy array
    text: Optional text to display above image
    scale_up: If True, scale up image to display by slide.SCALE_FACTOR
  """
  np_r = np_rgb_r_histogram(np_rgb)
  np_g = np_rgb_g_histogram(np_rgb)
  np_b = np_rgb_b_histogram(np_rgb)
  r_r, r_c, _ = np_r.shape
  g_r, g_c, _ = np_g.shape
  b_r, b_c, _ = np_b.shape

  if scale_up:
    np_rgb = np.repeat(np_rgb, slide.SCALE_FACTOR, axis=1)
    np_rgb = np.repeat(np_rgb, slide.SCALE_FACTOR, axis=0)

  img_r, img_c, img_ch = np_rgb.shape
  if text is not None:
    np_t = np_text(text)
    t_r, t_c, _ = np_t.shape
    t_i_c = max(t_c, img_c)
    t_i_r = t_r + img_r
    t_i = np.zeros([t_i_r, t_i_c, img_ch], dtype=np.uint8)
    t_i.fill(255)
    t_i[0:t_r, 0:t_c] = np_t
    t_i[t_r:t_r + img_r, 0:img_c] = np_rgb
    np_rgb = t_i  # for simplicity assign title+image to image
    img_r, img_c, img_ch = np_rgb.shape

  hists_c = max(r_c, g_c, b_c)
  hists_r = r_r + g_r + b_r
  hists = np.zeros([hists_r, hists_c, img_ch], dtype=np.uint8)

  hists[0:r_r, 0:r_c] = np_r
  hists[r_r:r_r + g_r, 0:g_c] = np_g
  hists[r_r + g_r:r_r + g_r + b_r, 0:b_c] = np_b

  r = max(img_r, hists_r)
  c = img_c + hists_c
  combo = np.zeros([r, c, img_ch], dtype=np.uint8)
  combo.fill(255)
  combo[0:img_r, 0:img_c] = np_rgb
  combo[0:hists_r, img_c:c] = hists
  pil_combo = util.np_to_pil(combo)
  pil_combo.show()


def pil_text(text, w_border=TILE_TEXT_W_BORDER, h_border=TILE_TEXT_H_BORDER, font_path=FONT_PATH,
             font_size=TILE_TEXT_SIZE, text_color=TILE_TEXT_COLOR, background=TILE_TEXT_BACKGROUND_COLOR):
  """
  Obtain a PIL image representation of text.

  Args:
    text: The text to convert to an image.
    w_border: Tile text width border (left and right).
    h_border: Tile text height border (top and bottom).
    font_path: Path to font.
    font_size: Size of font.
    text_color: Tile text color.
    background: Tile background color.

  Returns:
    PIL image representing the text.
  """

  font = ImageFont.truetype(font_path, font_size)
  x, y = ImageDraw.Draw(Image.new("RGB", (1, 1), background)).textsize(text, font)
  image = Image.new("RGB", (x + 2 * w_border, y + 2 * h_border), background)
  draw = ImageDraw.Draw(image)
  draw.text((w_border, h_border), text, text_color, font=font)
  return image


def np_text(text, w_border=TILE_TEXT_W_BORDER, h_border=TILE_TEXT_H_BORDER, font_path=FONT_PATH,
            font_size=TILE_TEXT_SIZE, text_color=TILE_TEXT_COLOR, background=TILE_TEXT_BACKGROUND_COLOR):
  """
  Obtain a NumPy array image representation of text.

  Args:
    text: The text to convert to an image.
    w_border: Tile text width border (left and right).
    h_border: Tile text height border (top and bottom).
    font_path: Path to font.
    font_size: Size of font.
    text_color: Tile text color.
    background: Tile background color.

  Returns:
    NumPy array representing the text.
  """
  pil_img = pil_text(text, w_border, h_border, font_path, font_size,
                     text_color, background)
  np_img = util.pil_to_np_rgb(pil_img)
  return np_img


def display_tile(tile, rgb_histograms=True, hsv_histograms=True):
  """
  Display a tile with its corresponding RGB and HSV histograms.

  Args:
    tile: The Tile object.
    rgb_histograms: If True, display RGB histograms.
    hsv_histograms: If True, display HSV histograms.
  """

  text = "S%03d R%03d C%03d\n" % (tile.slide_num, tile.r, tile.c)
  text += "Score:%4.2f Tissue:%5.2f%% CF:%2.0f SVF:%4.2f QF:%4.2f\n" % (
    tile.score, tile.tissue_percentage, tile.color_factor, tile.s_and_v_factor, tile.quantity_factor)
  text += "Rank #%d of %d" % (tile.rank, tile.tile_summary.num_tiles())

  np_scaled_tile = tile.get_np_scaled_tile()
  if np_scaled_tile is not None:
    small_text = text + "\n \nSmall Tile (%d x %d)" % (np_scaled_tile.shape[1], np_scaled_tile.shape[0])
    if rgb_histograms and hsv_histograms:
      display_image_with_rgb_and_hsv_histograms(np_scaled_tile, small_text, scale_up=True)
    elif rgb_histograms:
      display_image_with_rgb_histograms(np_scaled_tile, small_text, scale_up=True)
    elif hsv_histograms:
      display_image_with_hsv_histograms(np_scaled_tile, small_text, scale_up=True)
    else:
      display_image(np_scaled_tile, small_text, scale_up=True)

  np_tile = tile.get_np_tile()
  text += " based on small tile\n \nLarge Tile (%d x %d)" % (np_tile.shape[1], np_tile.shape[0])
  if rgb_histograms and hsv_histograms:
    display_image_with_rgb_and_hsv_histograms(np_tile, text)
  elif rgb_histograms:
    display_image_with_rgb_histograms(np_tile, text)
  elif hsv_histograms:
    display_image_with_hsv_histograms(np_tile, text)
  else:
    display_image(np_tile, text)


def display_image_with_rgb_and_hsv_histograms(np_rgb, text=None, scale_up=False):
  """
  Display a tile with its corresponding RGB and HSV histograms.

  Args:
    np_rgb: RGB image tile as a NumPy array
    text: Optional text to display above image
    scale_up: If True, scale up image to display by slide.SCALE_FACTOR
  """
  hsv = filter.filter_rgb_to_hsv(np_rgb)
  np_r = np_rgb_r_histogram(np_rgb)
  np_g = np_rgb_g_histogram(np_rgb)
  np_b = np_rgb_b_histogram(np_rgb)
  np_h = np_hsv_hue_histogram(filter.filter_hsv_to_h(hsv))
  np_s = np_hsv_saturation_histogram(filter.filter_hsv_to_s(hsv))
  np_v = np_hsv_value_histogram(filter.filter_hsv_to_v(hsv))

  r_r, r_c, _ = np_r.shape
  g_r, g_c, _ = np_g.shape
  b_r, b_c, _ = np_b.shape
  h_r, h_c, _ = np_h.shape
  s_r, s_c, _ = np_s.shape
  v_r, v_c, _ = np_v.shape

  if scale_up:
    np_rgb = np.repeat(np_rgb, slide.SCALE_FACTOR, axis=1)
    np_rgb = np.repeat(np_rgb, slide.SCALE_FACTOR, axis=0)

  img_r, img_c, img_ch = np_rgb.shape
  if text is not None:
    np_t = np_text(text)
    t_r, t_c, _ = np_t.shape
    t_i_c = max(t_c, img_c)
    t_i_r = t_r + img_r
    t_i = np.zeros([t_i_r, t_i_c, img_ch], dtype=np.uint8)
    t_i.fill(255)
    t_i[0:t_r, 0:t_c] = np_t
    t_i[t_r:t_r + img_r, 0:img_c] = np_rgb
    np_rgb = t_i  # for simplicity assign title+image to image
    img_r, img_c, img_ch = np_rgb.shape

  rgb_hists_c = max(r_c, g_c, b_c)
  rgb_hists_r = r_r + g_r + b_r
  rgb_hists = np.zeros([rgb_hists_r, rgb_hists_c, img_ch], dtype=np.uint8)
  rgb_hists[0:r_r, 0:r_c] = np_r
  rgb_hists[r_r:r_r + g_r, 0:g_c] = np_g
  rgb_hists[r_r + g_r:r_r + g_r + b_r, 0:b_c] = np_b

  hsv_hists_c = max(h_c, s_c, v_c)
  hsv_hists_r = h_r + s_r + v_r
  hsv_hists = np.zeros([hsv_hists_r, hsv_hists_c, img_ch], dtype=np.uint8)
  hsv_hists[0:h_r, 0:h_c] = np_h
  hsv_hists[h_r:h_r + s_r, 0:s_c] = np_s
  hsv_hists[h_r + s_r:h_r + s_r + v_r, 0:v_c] = np_v

  r = max(img_r, rgb_hists_r, hsv_hists_r)
  c = img_c + rgb_hists_c + hsv_hists_c
  combo = np.zeros([r, c, img_ch], dtype=np.uint8)
  combo.fill(255)
  combo[0:img_r, 0:img_c] = np_rgb
  combo[0:rgb_hists_r, img_c:img_c + rgb_hists_c] = rgb_hists
  combo[0:hsv_hists_r, img_c + rgb_hists_c:c] = hsv_hists
  pil_combo = util.np_to_pil(combo)
  pil_combo.show()


def rgb_to_hues(rgb):
  """
  Convert RGB NumPy array to 1-dimensional array of hue values (HSV H values in degrees).

  Args:
    rgb: RGB image as a NumPy array

  Returns:
    1-dimensional array of hue values in degrees
  """
  hsv = filter.filter_rgb_to_hsv(rgb, display_np_info=False)
  h = filter.filter_hsv_to_h(hsv, display_np_info=False)
  return h


def hsv_saturation_and_value_factor(rgb):
  """
  Function to reduce scores of tiles with narrow HSV saturations and values since saturation and value standard
  deviations should be relatively broad if the tile contains significant tissue.

  Example of a blurred tile that should not be ranked as a top tile:
    ../data/tiles_png/006/TUPAC-TR-006-tile-r58-c3-x2048-y58369-w1024-h1024.png

  Args:
    rgb: RGB image as a NumPy array

  Returns:
    Saturation and value factor, where 1 is no effect and less than 1 means the standard deviations of saturation and
    value are relatively small.
  """
  hsv = filter.filter_rgb_to_hsv(rgb, display_np_info=False)
  s = filter.filter_hsv_to_s(hsv)
  v = filter.filter_hsv_to_v(hsv)
  s_std = np.std(s)
  v_std = np.std(v)
  if s_std < 0.05 and v_std < 0.05:
    factor = 0.4
  elif s_std < 0.05:
    factor = 0.7
  elif v_std < 0.05:
    factor = 0.7
  else:
    factor = 1

  factor = factor ** 2
  return factor


def hsv_purple_deviation(hsv_hues):
  """
  Obtain the deviation from the HSV hue for purple.

  Args:
    hsv_hues: NumPy array of HSV hue values.

  Returns:
    The HSV purple deviation.
  """
  purple_deviation = np.sqrt(np.mean(np.abs(hsv_hues - HSV_PURPLE) ** 2))
  return purple_deviation


def hsv_pink_deviation(hsv_hues):
  """
  Obtain the deviation from the HSV hue for pink.

  Args:
    hsv_hues: NumPy array of HSV hue values.

  Returns:
    The HSV pink deviation.
  """
  pink_deviation = np.sqrt(np.mean(np.abs(hsv_hues - HSV_PINK) ** 2))
  return pink_deviation


def hsv_purple_pink_factor(rgb):
  """
  Compute scoring factor based on purple and pink HSV hue deviations and degree to which a narrowed hue color range
  average is purple versus pink.

  Args:
    rgb: Image an NumPy array.

  Returns:
    Factor that favors purple (hematoxylin stained) tissue over pink (eosin stained) tissue.
  """
  hues = rgb_to_hues(rgb)
  hues = hues[hues >= 260]  # exclude hues under 260
  hues = hues[hues <= 340]  # exclude hues over 340
  if len(hues) == 0:
    return 0  # if no hues between 260 and 340, then not purple or pink
  pu_dev = hsv_purple_deviation(hues)
  pi_dev = hsv_pink_deviation(hues)
  avg_factor = (340 - np.average(hues)) ** 2

  if pu_dev == 0:  # avoid divide by zero if tile has no tissue
    return 0

  factor = pi_dev / pu_dev * avg_factor
  return factor


def hsv_purple_vs_pink_average_factor(rgb, tissue_percentage):
  """
  Function to favor purple (hematoxylin) over pink (eosin) staining based on the distance of the HSV hue average
  from purple and pink.

  Args:
    rgb: Image as RGB NumPy array
    tissue_percentage: Amount of tissue on the tile

  Returns:
    Factor, where >1 to boost purple slide scores, <1 to reduce pink slide scores, or 1 no effect.
  """

  factor = 1
  # only applies to slides with a high quantity of tissue
  if tissue_percentage < TISSUE_HIGH_THRESH:
    return factor

  hues = rgb_to_hues(rgb)
  hues = hues[hues >= 200]  # Remove hues under 200
  if len(hues) == 0:
    return factor
  avg = np.average(hues)
  # pil_hue_histogram(hues).show()

  pu = HSV_PURPLE - avg
  pi = HSV_PINK - avg
  pupi = pu + pi
  # print("Av: %4d, Pu: %4d, Pi: %4d, PuPi: %4d" % (avg, pu, pi, pupi))
  # Av:  250, Pu:   20, Pi:   80, PuPi:  100
  # Av:  260, Pu:   10, Pi:   70, PuPi:   80
  # Av:  270, Pu:    0, Pi:   60, PuPi:   60 ** PURPLE
  # Av:  280, Pu:  -10, Pi:   50, PuPi:   40
  # Av:  290, Pu:  -20, Pi:   40, PuPi:   20
  # Av:  300, Pu:  -30, Pi:   30, PuPi:    0
  # Av:  310, Pu:  -40, Pi:   20, PuPi:  -20
  # Av:  320, Pu:  -50, Pi:   10, PuPi:  -40
  # Av:  330, Pu:  -60, Pi:    0, PuPi:  -60 ** PINK
  # Av:  340, Pu:  -70, Pi:  -10, PuPi:  -80
  # Av:  350, Pu:  -80, Pi:  -20, PuPi: -100

  if pupi > 30:
    factor *= 1.2
  if pupi < -30:
    factor *= .8
  if pupi > 0:
    factor *= 1.2
  if pupi > 50:
    factor *= 1.2
  if pupi < -60:
    factor *= .8

  return factor


class TileSummary:
  """
  Class for tile summary information.
  """

  slide_num = None
  orig_w = None
  orig_h = None
  orig_tile_w = None
  orig_tile_h = None
  scale_factor = slide.SCALE_FACTOR
  scaled_w = None
  scaled_h = None
  scaled_tile_w = None
  scaled_tile_h = None
  mask_percentage = None
  num_row_tiles = None
  num_col_tiles = None

  count = 0
  high = 0
  medium = 0
  low = 0
  none = 0

  def __init__(self, slide_num, orig_w, orig_h, orig_tile_w, orig_tile_h, scaled_w, scaled_h, scaled_tile_w,
               scaled_tile_h, tissue_percentage, num_col_tiles, num_row_tiles):
    self.slide_num = slide_num
    self.orig_w = orig_w
    self.orig_h = orig_h
    self.orig_tile_w = orig_tile_w
    self.orig_tile_h = orig_tile_h
    self.scaled_w = scaled_w
    self.scaled_h = scaled_h
    self.scaled_tile_w = scaled_tile_w
    self.scaled_tile_h = scaled_tile_h
    self.tissue_percentage = tissue_percentage
    self.num_col_tiles = num_col_tiles
    self.num_row_tiles = num_row_tiles
    self.tiles = []

  def __str__(self):
    return summary_title(self) + "\n" + summary_stats(self)

  def mask_percentage(self):
    """
    Obtain the percentage of the slide that is masked.

    Returns:
       The amount of the slide that is masked as a percentage.
    """
    return 100 - self.tissue_percentage

  def num_tiles(self):
    """
    Retrieve the total number of tiles.

    Returns:
      The total number of tiles (number of rows * number of columns).
    """
    return self.num_row_tiles * self.num_col_tiles

  def tiles_by_tissue_percentage(self):
    """
    Retrieve the tiles ranked by tissue percentage.

    Returns:
       List of the tiles ranked by tissue percentage.
    """
    sorted_list = sorted(self.tiles, key=lambda t: t.tissue_percentage, reverse=True)
    return sorted_list

  def tiles_by_score(self):
    """
    Retrieve the tiles ranked by score.

    Returns:
       List of the tiles ranked by score.
    """
    sorted_list = sorted(self.tiles, key=lambda t: t.score, reverse=True)
    return sorted_list

  def top_tiles(self):
    """
    Retrieve the top-scoring tiles.

    Returns:
       List of the top-scoring tiles.
    """
    sorted_tiles = self.tiles_by_score()
    top_tiles = sorted_tiles[:NUM_TOP_TILES]
    return top_tiles

  def get_tile(self, row, col):
    """
    Retrieve tile by row and column.

    Args:
      row: The row
      col: The column

    Returns:
      Corresponding Tile object.
    """
    tile_index = (row - 1) * self.num_col_tiles + (col - 1)
    tile = self.tiles[tile_index]
    return tile

  def display_summaries(self):
    """
    Display summary images.
    """
    summary_and_tiles(self.slide_num, display=True, save_summary=False, save_data=False, save_top_tiles=False)


class Tile:
  """
  Class for information about a tile.
  """

  def __init__(self, tile_summary, slide_num, np_scaled_tile, tile_num, r, c, r_s, r_e, c_s, c_e, o_r_s, o_r_e, o_c_s,
               o_c_e, t_p, color_factor, s_and_v_factor, quantity_factor, score):
    self.tile_summary = tile_summary
    self.slide_num = slide_num
    self.np_scaled_tile = np_scaled_tile
    self.tile_num = tile_num
    self.r = r
    self.c = c
    self.r_s = r_s
    self.r_e = r_e
    self.c_s = c_s
    self.c_e = c_e
    self.o_r_s = o_r_s
    self.o_r_e = o_r_e
    self.o_c_s = o_c_s
    self.o_c_e = o_c_e
    self.tissue_percentage = t_p
    self.color_factor = color_factor
    self.s_and_v_factor = s_and_v_factor
    self.quantity_factor = quantity_factor
    self.score = score

  def __str__(self):
    return "[Tile #%d, Row #%d, Column #%d, Tissue %4.2f%%, Score %0.4f]" % (
      self.tile_num, self.r, self.c, self.tissue_percentage, self.score)

  def __repr__(self):
    return "\n" + self.__str__()

  def mask_percentage(self):
    return 100 - self.tissue_percentage

  def tissue_quantity(self):
    return tissue_quantity(self.tissue_percentage)

  def get_pil_tile(self):
    return tile_to_pil_tile(self)

  def get_np_tile(self):
    return tile_to_np_tile(self)

  def save_tile(self):
    save_display_tile(self, save=True, display=False)

  def display_tile(self):
    save_display_tile(self, save=False, display=True)

  def display_with_histograms(self):
    display_tile(self, rgb_histograms=True, hsv_histograms=True)

  def get_np_scaled_tile(self):
    return self.np_scaled_tile

  def get_pil_scaled_tile(self):
    return util.np_to_pil(self.np_scaled_tile)


class TissueQuantity(Enum):
  NONE = 0
  LOW = 1
  MEDIUM = 2
  HIGH = 3


def dynamic_tiles(slide_num, small_tile_in_tile=False):
  """
  Generate tile summary with top tiles using original WSI training slide without intermediate image files saved to
  file system.

  Args:
    slide_num: The slide number.
    small_tile_in_tile: If True, include the small NumPy image in the Tile objects.

  Returns:
     TileSummary object with list of top Tile objects. The actual tile images are not retrieved until the
     Tile get_tile() methods are called.
  """
  np_img, large_w, large_h, small_w, small_h = slide.slide_to_scaled_np_image(slide_num)
  filt_np_img = filter.apply_image_filters(np_img)
  tile_summary = score_tiles(slide_num, filt_np_img, (large_w, large_h, small_w, small_h), small_tile_in_tile)
  return tile_summary


def dynamic_tile(slide_num, row, col, small_tile_in_tile=False):
  """
  Generate a single tile dynamically based on slide number, row, and column. If more than one tile needs to be
  retrieved dynamically, dynamic_tiles() should be used.

  Args:
    slide_num: The slide number.
    row: The row.
    col: The column.
    small_tile_in_tile: If True, include the small NumPy image in the Tile objects.

  Returns:
    Tile tile object.
  """
  tile_summary = dynamic_tiles(slide_num, small_tile_in_tile)
  tile = tile_summary.get_tile(row, col)
  return tile

# if __name__ == "__main__":
  # tile = dynamic_tile(2, 29, 16, True)
  # tile.display_with_histograms()

  # singleprocess_filtered_images_to_tiles()
  # multiprocess_filtered_images_to_tiles()
