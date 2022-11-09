"""
This script is from TensorFlowOnSpark project (https://github.com/yahoo/TensorFlowOnSpark/blob/master/
tensorflowonspark/gpu_info.py). It can be used to get the information of available GPUs
"""

import ctypes as ct
import logging
import platform
import random
import subprocess
import time

MAX_RETRIES=3

def get_gpu():
  """Allocates first available GPU using cudaSetDevice(), or returns 0 otherwise"""
  # Note: this code executes, but Tensorflow subsequently complains that the "current context was not created by
  # the StreamExecutor cuda_driver API"
  system = platform.system()
  if system == "Linux":
    libcudart = ct.cdll.LoadLibrary("libcudart.so")
  elif system == "Darwin":
    libcudart = ct.cdll.LoadLibrary("libcudart.dylib")
  elif system == "Windows":
    libcudart = ct.windll.LoadLibrary("libcudart.dll")
  else:
    raise NotImplementedError("Cannot identify system.")

  device_count = ct.c_int()
  libcudart.cudaGetDeviceCount(ct.byref(device_count))
  gpu = 0
  for i in range(device_count.value):
    if (0 == libcudart.cudaSetDevice(i) and 0 == libcudart.cudaFree(0)):
      gpu = i
      break;
  return gpu

def get_gpus(num_gpu=1):
  """Returns list of free GPUs according to nvidia-smi"""
  try:
    # get list of gpus (index, uuid)
    list_gpus = subprocess.check_output(["nvidia-smi", "--list-gpus"]).decode()
    logging.debug("all GPUs:\n{0}".format(list_gpus))

    # parse index and guid
    gpus = [ x for x in list_gpus.split('\n') if len(x) > 0 ]
    def parse_gpu(gpu_str):
      cols = gpu_str.split(' ')
      return cols[5].split(')')[0], cols[1].split(':')[0]
    gpu_list = [parse_gpu(gpu) for gpu in gpus]

    # randomize the search order to get a better distribution of GPUs
    random.shuffle(gpu_list)

    free_gpus = []
    retries = 0
    while len(free_gpus) < num_gpu and retries < MAX_RETRIES:
      smi_output = subprocess.check_output(["nvidia-smi", "--format=csv,noheader,nounits",
                                            "--query-compute-apps=gpu_uuid"]).decode()
      logging.debug("busy GPUs:\n{0}".format(smi_output))
      busy_uuids = [x for x in smi_output.split('\n') if len(x) > 0 ]
      for uuid, index in gpu_list:
        if not uuid in busy_uuids:
          free_gpus.append(index)

      if len(free_gpus) < num_gpu:
        # keep trying indefinitely
        logging.warn("Unable to find available GPUs: requested={0}, available={1}".format(num_gpu, len(free_gpus)))
        retries += 1
        time.sleep(30*retries)
        free_gpus = []

    # if still can't find GPUs, raise exception
    if len(free_gpus) < num_gpu:
      smi_output = subprocess.check_output(["nvidia-smi", "--format=csv",
                                            "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory"]).decode()
      logging.info(": {0}".format(smi_output))
      raise Exception("Unable to find free GPU:\n{0}".format(smi_output))

    return ','.join(free_gpus[:num_gpu])
  except subprocess.CalledProcessError as e:
    print ("nvidia-smi error", e.output)

# Function to get the gpu information
def get_free_gpu(max_gpu_utilization=40, min_free_memory=0.5, num_gpu=1):
  def get_gpu_info():
    # Get the gpu information
    gpu_info = subprocess.check_output(["nvidia-smi", "--format=csv,noheader,nounits",
                                        "--query-gpu=index,memory.total,memory.free,memory.used,utilization.gpu"]).decode()
    gpu_info = gpu_info.split('\n')

    gpu_info_array = []

    # Check each gpu
    for line in gpu_info:
      if len(line) > 0:
        val = line.split(',')
        gpu_id, total_memory, free_memory, used_memory, gpu_util = line.split(',')

        gpu_memory_util = float(used_memory)/float(total_memory)
        gpu_info_array.append((float(gpu_util), gpu_memory_util, gpu_id))

    return(gpu_info_array)

  # Read the gpu information multiple times
  num_times_to_average = 5
  current_array = []
  for ind in range(num_times_to_average):
    current_array.append(get_gpu_info())
    time.sleep(1)

  # Get number of gpus
  num_gpus = len(current_array[0])

  # Average the gpu information
  avg_array = [(0,0,str(x)) for x in range(num_gpus)]
  for ind in range(num_times_to_average):
    for gpu_ind in range(num_gpus):
      avg_array[gpu_ind] = (avg_array[gpu_ind][0] + current_array[ind][gpu_ind][0],
                            avg_array[gpu_ind][1] + current_array[ind][gpu_ind][1], avg_array[gpu_ind][2])

  for gpu_ind in range(num_gpus):
    avg_array[gpu_ind] = (float(avg_array[gpu_ind][0]) / num_times_to_average,
                          float(avg_array[gpu_ind][1]) / num_times_to_average, avg_array[gpu_ind][2])

  avg_array.sort()

  gpus_found = 0
  gpus_to_use = ""
  free_memory = 1.0
  # Return the least utilized GPUs if it's utilized less than max_gpu_utilization and amount of free memory is at most min_free_memory
  # Otherwise, run in cpu only mode
  for current_gpu in avg_array:
    if current_gpu[0] < max_gpu_utilization and (1-current_gpu[1]) > min_free_memory:
      if gpus_found == 0:
        gpus_to_use = current_gpu[2]
        free_memory = 1-current_gpu[1]
      else:
        gpus_to_use = gpus_to_use + "," + current_gpu[2]
        free_memory = min(free_memory, 1-current_gpu[1])

      gpus_found = gpus_found + 1

    if gpus_found == num_gpu:
      break

  return gpus_to_use, free_memory

