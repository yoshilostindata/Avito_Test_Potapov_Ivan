"""Общие константы решения."""

SEED = 42

MODEL_ID = "PaddlePaddle/PP-LCNet_x1_0_textline_ori_safetensors"
MODEL_REVISION = "9f1b0e272ba4e5d89aef7cf52998c6646862c5a7"

# Вход модели: как в PPLCNetImageProcessor для этой модели
INPUT_HEIGHT = 80
INPUT_WIDTH = 160
IMAGE_MEAN = [0.406, 0.456, 0.485]
IMAGE_STD = [0.225, 0.224, 0.229]
