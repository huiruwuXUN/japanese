from collections import namedtuple
from itertools import groupby
from pathlib import Path
import ipywidgets as widgets

import cv2
import matplotlib.pyplot as plt
import numpy as np
from openvino.runtime import Core

model_folder = "model"
data_folder = "seg_letter/J01"
charlist_folder = 'charlist'

# Precision used by the model.
precision = "FP16"

Language = namedtuple(
    typename="Language", field_names=["model_name", "charlist_name", "demo_image_name"]
)
chinese_files = Language(
    model_name="handwritten-simplified-chinese-recognition-0001",
    charlist_name="chinese_charlist.txt",
    demo_image_name="handwritten_chinese_test.jpg",
)
japanese_files = Language(
    model_name="handwritten-japanese-recognition-0001",
    charlist_name="japanese_charlist.txt",
    demo_image_name="handwritten_japanese_test.png",
)
language = "japanese"

languages = {"chinese": chinese_files, "japanese": japanese_files}

selected_language = languages.get(language)


path_to_model_weights = Path(f'{model_folder}/intel/{selected_language.model_name}/{precision}/{selected_language.model_name}.bin')
#if not path_to_model_weights.is_file():
    #download_command = f'omz_downloader --name {selected_language.model_name} --output_dir {model_folder} --precision {precision}'
    #print(download_command)


core = Core()
path_to_model = path_to_model_weights.with_suffix(".xml")
model = core.read_model(model=path_to_model)


core = Core()

device = widgets.Dropdown(
    options=core.available_devices + ["AUTO"],
    value='AUTO',
    description='Device:',
    disabled=False,
)

compiled_model = core.compile_model(model=model, device_name=device.value)
recognition_output_layer = compiled_model.output(0)
recognition_input_layer = compiled_model.input(0)

file_name = '3.jpg'

# Text detection models expect an image in grayscale format.
# IMPORTANT! This model enables reading only one line at time.

# Read the image.
image = cv2.imread(filename=f"{data_folder}/{file_name}", flags=cv2.IMREAD_GRAYSCALE)

# Fetch the shape.
image_height, _ = image.shape

# B,C,H,W = batch size, number of channels, height, width.
_, _, H, W = recognition_input_layer.shape

# Calculate scale ratio between the input shape height and image height to resize the image.
scale_ratio = H / image_height

# Resize the image to expected input sizes.
resized_image = cv2.resize(
    image, None, fx=scale_ratio, fy=scale_ratio, interpolation=cv2.INTER_AREA
)

# Pad the image to match input size, without changing aspect ratio.
resized_image = np.pad(
    resized_image, ((0, 0), (0, W - resized_image.shape[1])), mode="edge"
)

# Reshape to network input shape.
input_image = resized_image[None, None, :, :]

used_charlist = selected_language.charlist_name

# With both models, there should be blank symbol added at index 0 of each charlist.
blank_char = "~"

with open(f"{charlist_folder}/{used_charlist}", "r", encoding="utf-8") as charlist:
    letters = blank_char + "".join(line.strip() for line in charlist)

predictions = compiled_model([input_image])[recognition_output_layer]
predictions = np.squeeze(predictions)

# Run the `argmax` function to pick the symbols with the highest probability.
predictions_indexes = np.argmax(predictions, axis=1)
# Use the `groupby` function to remove concurrent letters, as required by CTC greedy decoding.
output_text_indexes = list(groupby(predictions_indexes))

# Remove grouper objects.
output_text_indexes, _ = np.transpose(output_text_indexes, (1, 0))

# Remove blank symbols.
output_text_indexes = output_text_indexes[output_text_indexes != 0]

# Assign letters to indexes from the output array.
output_text = [letters[letter_index] for letter_index in output_text_indexes]
print("finished")
print("".join(output_text))