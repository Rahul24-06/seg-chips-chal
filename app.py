# -*- coding: utf-8 -*-

# Commented out IPython magic to ensure Python compatibility.
# %pip install -q "transformers>=4.26.0" "openvino==2023.1.0.dev20230728" gradio torch scipy ipywidgets Pillow matplotlib

import warnings
from collections import defaultdict
from pathlib import Path
import sys

from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
from transformers.models.oneformer.modeling_oneformer import OneFormerForUniversalSegmentationOutput
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from PIL import ImageOps

import openvino

sys.path.append("../utils")
from notebook_utils import download_file

IR_PATH = Path("oneformer.xml")
OUTPUT_NAMES = ['class_queries_logits', 'masks_queries_logits']

"""## Load OneFormer fine-tuned on COCO for universal segmentation [$\Uparrow$](#Table-of-content:)
Here we use the `from_pretrained` method of `OneFormerForUniversalSegmentation` to load the [HuggingFace OneFormer model](https://huggingface.co/docs/transformers/model_doc/oneformer) based on Swin-L backbone and trained on [COCO](https://cocodataset.org/) dataset.

Also, we use HuggingFace processor to prepare the model inputs from images and post-process model outputs for visualization.
"""

processor = OneFormerProcessor.from_pretrained("shi-labs/oneformer_coco_swin_large")
model = OneFormerForUniversalSegmentation.from_pretrained(
    "shi-labs/oneformer_coco_swin_large",
)
id2label = model.config.id2label

task_seq_length = processor.task_seq_length
shape = (800, 800)
dummy_input = {
    "pixel_values": torch.randn(1, 3, *shape),
    "task_inputs": torch.randn(1, task_seq_length),
    "pixel_mask": torch.randn(1, *shape),
}

import ipywidgets as widgets

core = openvino.Core()

device = widgets.Dropdown(
    options=core.available_devices + ["AUTO"],
    value='AUTO',
    description='Device:',
    disabled=False,
)

device

def prepare_inputs(image: Image.Image, task: str):
    """Convert image to model input"""
    image = ImageOps.pad(image, shape)
    inputs = processor(image, [task], return_tensors="pt")
    converted = {
        'pixel_values': inputs['pixel_values'],
        'task_inputs': inputs['task_inputs']
    }
    return converted

def process_output(d):
    """Convert OpenVINO model output to HuggingFace representation for visualization"""
    hf_kwargs = {
        output_name: torch.tensor(d[output_name]) for output_name in OUTPUT_NAMES
    }

    return OneFormerForUniversalSegmentationOutput(**hf_kwargs)

# Read the model from files.
model = core.read_model(model=IR_PATH)
# Compile the model.
compiled_model = core.compile_model(model=model, device_name=device.value)

class Visualizer:
    @staticmethod
    def extract_legend(handles):
        fig = plt.figure()
        fig.legend(handles=handles, ncol=len(handles) // 20 + 1, loc='center')
        fig.tight_layout()
        return fig

    @staticmethod
    def predicted_semantic_map_to_figure(predicted_map):
        segmentation = predicted_map[0]
        # get the used color map
        viridis = plt.get_cmap('viridis', torch.max(segmentation))
        # get all the unique numbers
        labels_ids = torch.unique(segmentation).tolist()
        fig, ax = plt.subplots()
        ax.imshow(segmentation)
        ax.set_axis_off()
        handles = []
        for label_id in labels_ids:
            label = id2label[label_id]
            color = viridis(label_id)
            handles.append(mpatches.Patch(color=color, label=label))
        fig_legend = Visualizer.extract_legend(handles=handles)
        fig.tight_layout()
        return fig, fig_legend

    @staticmethod
    def predicted_instance_map_to_figure(predicted_map):
        segmentation = predicted_map[0]['segmentation']
        segments_info = predicted_map[0]['segments_info']
        # get the used color map
        viridis = plt.get_cmap('viridis', torch.max(segmentation))
        fig, ax = plt.subplots()
        ax.imshow(segmentation)
        ax.set_axis_off()
        instances_counter = defaultdict(int)
        handles = []
        # for each segment, draw its legend
        for segment in segments_info:
            segment_id = segment['id']
            segment_label_id = segment['label_id']
            segment_label = id2label[segment_label_id]
            label = f"{segment_label}-{instances_counter[segment_label_id]}"
            instances_counter[segment_label_id] += 1
            color = viridis(segment_id)
            handles.append(mpatches.Patch(color=color, label=label))

        fig_legend = Visualizer.extract_legend(handles)
        fig.tight_layout()
        return fig, fig_legend

    @staticmethod
    def predicted_panoptic_map_to_figure(predicted_map):
        segmentation = predicted_map[0]['segmentation']
        segments_info = predicted_map[0]['segments_info']
        # get the used color map
        viridis = plt.get_cmap('viridis', torch.max(segmentation))
        fig, ax = plt.subplots()
        ax.imshow(segmentation)
        ax.set_axis_off()
        instances_counter = defaultdict(int)
        handles = []
        # for each segment, draw its legend
        for segment in segments_info:
            segment_id = segment['id']
            segment_label_id = segment['label_id']
            segment_label = id2label[segment_label_id]
            label = f"{segment_label}-{instances_counter[segment_label_id]}"
            instances_counter[segment_label_id] += 1
            color = viridis(segment_id)
            handles.append(mpatches.Patch(color=color, label=label))

        fig_legend = Visualizer.extract_legend(handles)
        fig.tight_layout()
        return fig, fig_legend

def segment(img: Image.Image, task: str):
    """
    Apply segmentation on an image.

    Args:
        img: Input image. It will be resized to 800x800.
        task: String describing the segmentation task. Supported values are: "semantic", "instance" and "panoptic".
    Returns:
        Tuple[Figure, Figure]: Segmentation map and legend charts.
    """
    if img is None:
        raise gr.Error("Please load the image or use one from the examples list")
    inputs = prepare_inputs(img, task)
    outputs = compiled_model(inputs)
    hf_output = process_output(outputs)
    predicted_map = getattr(processor, f"post_process_{task}_segmentation")(
        hf_output, target_sizes=[img.size[::-1]]
    )
    return getattr(Visualizer, f"predicted_{task}_map_to_figure")(predicted_map)

image = download_file("http://images.cocodataset.org/val2017/000000439180.jpg", "sample.jpg")
image = Image.open("sample.jpg")
image

from ipywidgets import Dropdown

task = Dropdown(options=["semantic", "instance", "panoptic"], value="semantic")
task

import matplotlib
matplotlib.use("Agg")  # disable showing figures

def stack_images_horizontally(img1: Image, img2: Image):
    res = Image.new("RGB", (img1.width + img2.width, max(img1.height, img2.height)), (255, 255,255))
    res.paste(img1, (0, 0))
    res.paste(img2, (img1.width, 0))
    return res

result, legend = segment(image, task.value)

result.savefig("result.jpg", bbox_inches="tight")
legend.savefig("legend.jpg", bbox_inches="tight")
result = Image.open("result.jpg")
legend = Image.open("legend.jpg")
stack_images_horizontally(result, legend)

import gradio as gr


def compile_model(device):
    global compiled_model
    compiled_model = core.compile_model(model=model, device_name=device)


with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            inp_img = gr.Image(label="Image", type="pil")
            inp_task = gr.Radio(
                ["semantic", "instance", "panoptic"], label="Task", value="semantic"
            )
            inp_device = gr.Dropdown(
                label="Device", choices=core.available_devices + ["AUTO"], value="AUTO"
            )
        with gr.Column():
            out_result = gr.Plot(label="Result")
            out_legend = gr.Plot(label="Legend")
    btn = gr.Button()
    gr.Examples(
        examples=[["sample.jpg", "semantic"]], inputs=[inp_img, inp_task]
    )
    btn.click(segment, [inp_img, inp_task], [out_result, out_legend])

    def on_device_change_begin():
        return (
            btn.update(value="Changing device...", interactive=False),
            inp_device.update(interactive=False)
        )

    def on_device_change_end():
        return (btn.update(value="Run", interactive=True), inp_device.update(interactive=True))

    inp_device.change(on_device_change_begin, outputs=[btn, inp_device]).then(
        compile_model, inp_device
    ).then(on_device_change_end, outputs=[btn, inp_device])


try:
    demo.launch(debug=True)
except Exception:
    demo.launch(share=True, debug=True)
# if you are launching remotely, specify server_name and server_port
# demo.launch(server_name='your server name', server_port='server port in int')
# Read more in the docs: https://gradio.app/docs/
