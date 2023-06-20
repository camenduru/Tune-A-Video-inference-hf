#!/usr/bin/env python

from __future__ import annotations

import os

import gradio as gr
import torch

from inference import InferencePipeline


class InferenceUtil:
    def __init__(self, hf_token: str | None):
        self.hf_token = hf_token

    def load_model_info(self, model_id: str) -> tuple[str, str]:
        try:
            card = InferencePipeline.get_model_card(model_id, self.hf_token)
        except Exception:
            return '', ''
        base_model = getattr(card.data, 'base_model', '')
        training_prompt = getattr(card.data, 'training_prompt', '')
        return base_model, training_prompt


DESCRIPTION = '# [Tune-A-Video](https://tuneavideo.github.io/)'
if not torch.cuda.is_available():
    DESCRIPTION += '\n<p>Running on CPU 🥶 This demo does not work on CPU.</p>'

HF_TOKEN = os.getenv('HF_TOKEN')
pipe = InferencePipeline(HF_TOKEN)
app = InferenceUtil(HF_TOKEN)

with gr.Blocks(css='style.css') as demo:
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        with gr.Column():
            with gr.Box():
                model_id = gr.Dropdown(
                    label='Model ID',
                    choices=[
                        'Tune-A-Video-library/a-man-is-surfing',
                        'Tune-A-Video-library/mo-di-bear-guitar',
                        'Tune-A-Video-library/redshift-man-skiing',
                    ],
                    value='Tune-A-Video-library/a-man-is-surfing')
                with gr.Accordion(
                        label=
                        'Model info (Base model and prompt used for training)',
                        open=False):
                    with gr.Row():
                        base_model_used_for_training = gr.Text(
                            label='Base model', interactive=False)
                        prompt_used_for_training = gr.Text(
                            label='Training prompt', interactive=False)
            prompt = gr.Textbox(label='Prompt',
                                max_lines=1,
                                placeholder='Example: "A panda is surfing"')
            video_length = gr.Slider(label='Video length',
                                     minimum=4,
                                     maximum=12,
                                     step=1,
                                     value=8)
            fps = gr.Slider(label='FPS',
                            minimum=1,
                            maximum=12,
                            step=1,
                            value=1)
            seed = gr.Slider(label='Seed',
                             minimum=0,
                             maximum=100000,
                             step=1,
                             value=0)
            with gr.Accordion('Other Parameters', open=False):
                num_steps = gr.Slider(label='Number of Steps',
                                      minimum=0,
                                      maximum=100,
                                      step=1,
                                      value=50)
                guidance_scale = gr.Slider(label='CFG Scale',
                                           minimum=0,
                                           maximum=50,
                                           step=0.1,
                                           value=7.5)

            run_button = gr.Button('Generate')

            gr.Markdown('''
            - It takes a few minutes to download model first.
            - Expected time to generate an 8-frame video: 70 seconds with T4, 24 seconds with A10G, (10 seconds with A100)
            ''')
        with gr.Column():
            result = gr.Video(label='Result')
    with gr.Row():
        examples = [
            [
                'Tune-A-Video-library/a-man-is-surfing',
                'A panda is surfing.',
                8,
                1,
                3,
                50,
                7.5,
            ],
            [
                'Tune-A-Video-library/a-man-is-surfing',
                'A racoon is surfing, cartoon style.',
                8,
                1,
                3,
                50,
                7.5,
            ],
            [
                'Tune-A-Video-library/mo-di-bear-guitar',
                'a handsome prince is playing guitar, modern disney style.',
                8,
                1,
                123,
                50,
                7.5,
            ],
            [
                'Tune-A-Video-library/mo-di-bear-guitar',
                'a magical princess is playing guitar, modern disney style.',
                8,
                1,
                123,
                50,
                7.5,
            ],
            [
                'Tune-A-Video-library/mo-di-bear-guitar',
                'a rabbit is playing guitar, modern disney style.',
                8,
                1,
                123,
                50,
                7.5,
            ],
            [
                'Tune-A-Video-library/mo-di-bear-guitar',
                'a baby is playing guitar, modern disney style.',
                8,
                1,
                123,
                50,
                7.5,
            ],
            [
                'Tune-A-Video-library/redshift-man-skiing',
                '(redshift style) spider man is skiing.',
                8,
                1,
                123,
                50,
                7.5,
            ],
            [
                'Tune-A-Video-library/redshift-man-skiing',
                '(redshift style) black widow is skiing.',
                8,
                1,
                123,
                50,
                7.5,
            ],
            [
                'Tune-A-Video-library/redshift-man-skiing',
                '(redshift style) batman is skiing.',
                8,
                1,
                123,
                50,
                7.5,
            ],
            [
                'Tune-A-Video-library/redshift-man-skiing',
                '(redshift style) hulk is skiing.',
                8,
                1,
                123,
                50,
                7.5,
            ],
        ]
        gr.Examples(examples=examples,
                    inputs=[
                        model_id,
                        prompt,
                        video_length,
                        fps,
                        seed,
                        num_steps,
                        guidance_scale,
                    ],
                    outputs=result,
                    fn=pipe.run,
                    cache_examples=os.getenv('CACHE_EXAMPLES') == '1')

    model_id.change(fn=app.load_model_info,
                    inputs=model_id,
                    outputs=[
                        base_model_used_for_training,
                        prompt_used_for_training,
                    ])
    inputs = [
        model_id,
        prompt,
        video_length,
        fps,
        seed,
        num_steps,
        guidance_scale,
    ]
    prompt.submit(fn=pipe.run, inputs=inputs, outputs=result)
    run_button.click(fn=pipe.run, inputs=inputs, outputs=result)

demo.queue().launch()
