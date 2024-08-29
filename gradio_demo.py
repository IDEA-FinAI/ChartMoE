import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import gradio as gr
from PIL import Image

from functools import partial
from copy import deepcopy

from chartmoe import ChartMoE_Robot

def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)

# ========================================
#             Model Initialization
# ========================================

print('Initializing Chat')


disable_torch_init()
chat_robot = ChartMoE_Robot()

print('Initialization Finished')

# ========================================
#             Gradio Setting
# ========================================


def gradio_reset(history, img_list):
    if history is not None:
        history = ""
    if img_list is not None:
        img_list = []
    return None, \
        gr.update(value=None, interactive=True), \
        gr.update(placeholder='Please upload your image first', interactive=False),\
        gr.update(value="Upload & Start Chat", interactive=True), \
        history, \
        img_list

def upload_img(gr_img, text_input, history, img_list):
    def load_img(image,img_list):
        if isinstance(image, str):  # is a image path
            image = Image.open(image).convert('RGB')
        elif isinstance(image, Image.Image):
            image = image.convert('RGB')

        img_list.append(image)
        msg = "Received."
        return msg
    if gr_img is None:
        return None, None, gr.update(interactive=True), history

    load_img(gr_img, img_list)
    return gr.update(interactive=False), \
        gr.update(interactive=True, placeholder='Type and press Enter'), \
        gr.update(value="Start Chatting", interactive=False), \
        history, \
        img_list

def gradio_ask(user_message, chatbot):
    if len(user_message) == 0:
        return gr.update(interactive=True, placeholder='Input should not be empty!'), chatbot, history
    chatbot = chatbot + [[user_message, None]]
    return user_message, chatbot


def gradio_answer(chatbot, text_input, history, img_list, do_sample,num_beams, temperature, max_new_tokens):
    generation_config = \
        {
            "do_sample": do_sample=='True',
            "num_beams": num_beams,
            'temperature': temperature,
            'max_new_tokens': max_new_tokens,
        }

    image =  img_list[0]
    with torch.cuda.amp.autocast():
        response, history = chat_robot.chat(image=image,question=text_input,history=history,**generation_config)
    chatbot[-1][1] = response
    text_input = ''
    return chatbot, history, img_list, text_input

title = """<h1 align="center">Demo of ChartMoE</h1>"""
description = """<h3>This is the demo of ChartMoE. Upload your images and start chatting! <br> To use
            example questions, click example image, hit upload, and press enter in the chatbox. </h3>"""

from transformers.trainer_utils import set_seed
set_seed(42)
#TODO show examples below

with gr.Blocks() as demo:
    gr.Markdown(title)
    gr.Markdown(description)

    with gr.Row():
        with gr.Column(scale=0.5):
            image = gr.Image(type="pil")
            upload_button = gr.Button(value="Upload & Start Chat", interactive=True, variant="primary")
            clear = gr.Button("Restart 🔄")
            do_sample = gr.components.Radio(['True', 'False'],
                            label='do_sample(If False, num_beams, temperature and so on cannot work!)',
                            value='False')

            num_beams = gr.Slider(
                minimum=1,
                maximum=5,
                value=3,
                step=1,
                interactive=True,
                label="beam search numbers",
            )

            temperature = gr.Slider(
                minimum=0.1,
                maximum=2.0,
                value=1.0,
                step=0.1,
                interactive=True,
                label="Temperature",
            )

            max_new_tokens = gr.Slider(
                minimum=128,
                maximum=4096,
                value=512,
                step=128,
                interactive=True,
                label="max new tokens",
            )

        with gr.Column():
            history = gr.State(value="")
            img_list = gr.State(value=[])
            chatbot = gr.Chatbot(label='ChartMoE')
            text_input = gr.Textbox(label='User', placeholder='Please upload your image first', interactive=False)

            gr.Examples(examples=[
                ["examples/bar2.png", "Redraw the chart with python matplotlib, giving the code to highlight the column corresponding to the year in which the student got the highest score (painting it red). Please keep the same colors and legend as the input chart."],
                ["examples/line3.png", "Redraw the chart with python matplotlib, giving the code to highlight data point with lowest growth rate (draw a horizontal dotted line parallel to the x-axi, through the lowest point and add \'lowest\' label in the legend anchor). Please keep the same colors and legend as the input chart."],
                ["examples/pie1.png", "Redraw the chart with python matplotlib, convert it into a bar chart, giving the code to reflect the fact that the price of \'Gold\' has been reduced to 27% and the \'Silver\' has been increased to 28%. Please keep the colors and legend according to the input chart."]
            ], inputs=[image, text_input])

    upload_button.click(upload_img, [image, text_input, history,img_list], [image, text_input, upload_button, history, img_list])

    # print(list(map(type,[text_input, chatbot])))
    # print(list(map(type,[chatbot, history, img_list, do_sample, num_beams, temperature, max_new_tokens])))
    text_input.submit(gradio_ask, [text_input, chatbot], [text_input, chatbot]).then(
        gradio_answer, [chatbot, text_input, history, img_list, do_sample, num_beams, temperature, max_new_tokens], [chatbot, history, img_list, text_input]
    )
    clear.click(gradio_reset, [history, img_list], [chatbot, image, text_input, upload_button, history, img_list], queue=False)

demo.launch(share=True,inbrowser=True)