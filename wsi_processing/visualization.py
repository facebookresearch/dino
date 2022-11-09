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
Visualization -- Predicting Breast Cancer Proliferation Scores with
Apache SystemML

This module contains functions for visualizing data for the breast
cancer project.
"""
import matplotlib.pyplot as plt
from PIL import ImageDraw
from enum import Enum

class Shape(Enum):
  CROSS = 0
  SQUARE = 1
  CIRCLE = 2


def visualize_tile(tile):
  """
  Plot a tissue tile.
  
  Args:
    tile: A 3D NumPy array of shape (tile_size, tile_size, channels).
  
  Returns:
    None
  """
  plt.imshow(tile)
  plt.show()


def visualize_sample(sample, size=256):
  """
  Plot a tissue sample.
  
  Args:
    sample: A square sample flattened to a vector of size
      (channels*size_x*size_y).
    size: The width and height of the square samples.
  
  Returns:
    None
  """
  # Change type, reshape, transpose to (size_x, size_y, channels).
  length = sample.shape[0]
  channels = int(length / (size * size))
  if channels > 1:
    sample = sample.astype('uint8').reshape((channels, size, size)).transpose(1,2,0)
    plt.imshow(sample)
  else:
    vmax = 255 if sample.max() > 1 else 1
    sample = sample.reshape((size, size))
    plt.imshow(sample, cmap="gray", vmin=0, vmax=vmax)
  plt.show()

def draw_circle(draw, center_location, color, radius=32, width=5):
  """
  Draw a circle on a image

  Args:
    draw: PIL.ImageDraw object
    center_location: center location of the circle
    color: color of the circle outline
    radius: circle radius
    width: width of the circle outline
  """
  r, c = center_location
  radius = int(radius)
  for i in range(width):
    r0 = r - (radius + i)
    c0 = c - (radius + i)
    r1 = r + (radius + i)
    c1 = c + (radius + i)
    draw.ellipse([(c0, r0), (c1, r1)], fill=None, outline=color)

def add_mark(im, locations, shape=Shape.SQUARE, mark_color=(0, 255, 127, 200), hasProb=False):
  """
  add the mark for each point of the input locations into the image

  Args:
    im: PIL.Image object
    locations: a list of point locations, such as [(r0, c0, p0),
      (r1, c1, p1), (r2, c2, p2), ...]
    shape: mark shape, it could be Shape.CROSS, Shape.SQUARE,
      Shape.CIRCLE
    mark_color: mark color, default value is (0, 255, 127, 200)
    hasProb: bool value to tell if the input csv file has the
      probability column.
  """
  init_radius = 32
  width = 8

  draw = ImageDraw.Draw(im, 'RGBA')
  for t in locations:
    # reset radius to the initial value for each loop
    radius = init_radius
    r = t[0]
    c = t[1]
    p = t[2] if hasProb else 1
    radius = radius * p
    r0 = r - radius
    c0 = c - radius
    r1 = r + radius
    c1 = c + radius

    # the default ImageDraw functions could not set the outline width
    # for rectangle, circle, and ellipse. Here we need to use the basic
    # functions to customize the drawing functions for cross, square,
    # circle
    if shape is Shape.CROSS:
      draw.line([(c, r0), (c, r1)], fill= mark_color, width=width)
      draw.line([(c0, r), (c1, r)], fill=mark_color, width=width)
    elif shape is Shape.SQUARE:
      draw.line([(c0, r0), (c1, r0)], fill=mark_color, width=width)
      draw.line([(c1, r0), (c1, r1)], fill=mark_color, width=width)
      draw.line([(c1, r1), (c0, r1)], fill=mark_color, width=width)
      draw.line([(c0, r1), (c0, r0)], fill=mark_color, width=width)
    elif shape is Shape.CIRCLE:
      draw_circle(draw, (r, c), mark_color, radius, width=5)
    else:
      raise ValueError("Do not support this shape yet")
  del draw
  return im
