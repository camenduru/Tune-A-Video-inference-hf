#!/usr/bin/env python

from __future__ import annotations

import os
from subprocess import getoutput

import gradio as gr
import torch

from app_inference import create_inference_demo
from app_training import create_training_demo
from app_upload import create_upload_demo
from inference import InferencePipeline
from trainer import Trainer

TITLE = '# [Tune-A-Video](https://tuneavideo.github.io/) UI'

ORIGINAL_SPACE_ID = 'Tune-A-Video-library/Tune-A-Video-Training-UI'
SPACE_ID = os.getenv('SPACE_ID', ORIGINAL_SPACE_ID)
GPU_DATA = getoutput('nvidia-smi')
SHARED_UI_WARNING = f'''## Attention - Training doesn't work in this shared UI. You can duplicate and use it with a paid private T4 GPU.

<center><a class="duplicate-button" style="display:inline-block" target="_blank" href="https://huggingface.co/spaces/{SPACE_ID}?duplicate=true"><img style="margin-top:0;margin-bottom:0" src="https://img.shields.io/badge/-Duplicate%20Space-blue?labelColor=white&style=flat&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAAAXNSR0IArs4c6QAAAP5JREFUOE+lk7FqAkEURY+ltunEgFXS2sZGIbXfEPdLlnxJyDdYB62sbbUKpLbVNhyYFzbrrA74YJlh9r079973psed0cvUD4A+4HoCjsA85X0Dfn/RBLBgBDxnQPfAEJgBY+A9gALA4tcbamSzS4xq4FOQAJgCDwV2CPKV8tZAJcAjMMkUe1vX+U+SMhfAJEHasQIWmXNN3abzDwHUrgcRGmYcgKe0bxrblHEB4E/pndMazNpSZGcsZdBlYJcEL9Afo75molJyM2FxmPgmgPqlWNLGfwZGG6UiyEvLzHYDmoPkDDiNm9JR9uboiONcBXrpY1qmgs21x1QwyZcpvxt9NS09PlsPAAAAAElFTkSuQmCC&logoWidth=14" alt="Duplicate Space"></a></center>
'''

if os.getenv('SYSTEM') == 'spaces' and SPACE_ID != ORIGINAL_SPACE_ID:
    SETTINGS = f'<a href="https://huggingface.co/spaces/{SPACE_ID}/settings">Settings</a>'
else:
    SETTINGS = 'Settings'

INVALID_GPU_WARNING = f'''## Attention - the specified GPU is invalid. Training may not work. Make sure you have selected a `T4 GPU` for this task.'''

CUDA_NOT_AVAILABLE_WARNING = f'''## Attention - Running on CPU.
<center>
You can assign a GPU in the {SETTINGS} tab if you are running this on HF Spaces.
You can use "T4 small/medium" to run this demo.
</center>
'''

HF_TOKEN_NOT_SPECIFIED_WARNING = f'''The environment variable `HF_TOKEN` is not specified. Feel free to specify your Hugging Face token with write permission if you don't want to manually provide it for every run.
<center>
You can check and create your Hugging Face tokens <a href="https://huggingface.co/settings/tokens" target="_blank">here</a>.
You can specify environment variables in the "Repository secrets" section of the {SETTINGS} tab.
</center>
'''

HF_TOKEN = os.getenv('HF_TOKEN')


def show_warning(warning_text: str) -> gr.Blocks:
    with gr.Blocks() as demo:
        with gr.Box():
            gr.Markdown(warning_text)
    return demo


pipe = InferencePipeline(HF_TOKEN)
trainer = Trainer(HF_TOKEN)

with gr.Blocks(css='style.css') as demo:
    if SPACE_ID == ORIGINAL_SPACE_ID:
        show_warning(SHARED_UI_WARNING)
    elif not torch.cuda.is_available():
        show_warning(CUDA_NOT_AVAILABLE_WARNING)
    elif (not 'T4' in GPU_DATA):
        show_warning(INVALID_GPU_WARNING)

    gr.Markdown(TITLE)
    with gr.Tabs():
        with gr.TabItem('Train'):
            create_training_demo(trainer, pipe)
        with gr.TabItem('Run'):
            create_inference_demo(pipe, HF_TOKEN)
        with gr.TabItem('Upload'):
            gr.Markdown('''
            - You can use this tab to upload models later if you choose not to upload models in training time or if upload in training time failed.
            ''')
            create_upload_demo(HF_TOKEN)

    if not HF_TOKEN:
        show_warning(HF_TOKEN_NOT_SPECIFIED_WARNING)

demo.queue(max_size=1).launch(share=False)
