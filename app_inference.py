#!/usr/bin/env python

from __future__ import annotations

import enum

import gradio as gr
from huggingface_hub import HfApi

from constants import MODEL_LIBRARY_ORG_NAME, UploadTarget
from inference import InferencePipeline
from utils import find_exp_dirs


class ModelSource(enum.Enum):
    HUB_LIB = UploadTarget.MODEL_LIBRARY.value
    LOCAL = 'Local'


class InferenceUtil:
    def __init__(self, hf_token: str | None):
        self.hf_token = hf_token

    def load_hub_model_list(self) -> dict:
        api = HfApi(token=self.hf_token)
        choices = [
            info.modelId
            for info in api.list_models(author=MODEL_LIBRARY_ORG_NAME)
        ]
        return gr.update(choices=choices,
                         value=choices[0] if choices else None)

    @staticmethod
    def load_local_model_list() -> dict:
        choices = find_exp_dirs()
        return gr.update(choices=choices,
                         value=choices[0] if choices else None)

    def reload_model_list(self, model_source: str) -> dict:
        if model_source == ModelSource.HUB_LIB.value:
            return self.load_hub_model_list()
        elif model_source == ModelSource.LOCAL.value:
            return self.load_local_model_list()
        else:
            raise ValueError

    def load_model_info(self, model_id: str) -> tuple[str, str]:
        try:
            card = InferencePipeline.get_model_card(model_id, self.hf_token)
        except Exception:
            return '', ''
        base_model = getattr(card.data, 'base_model', '')
        training_prompt = getattr(card.data, 'training_prompt', '')
        return base_model, training_prompt

    def reload_model_list_and_update_model_info(
            self, model_source: str) -> tuple[dict, str, str]:
        model_list_update = self.reload_model_list(model_source)
        model_list = model_list_update['choices']
        model_info = self.load_model_info(model_list[0] if model_list else '')
        return model_list_update, *model_info


def create_inference_demo(pipe: InferencePipeline,
                          hf_token: str | None = None) -> gr.Blocks:
    app = InferenceUtil(hf_token)

    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                with gr.Box():
                    model_source = gr.Radio(
                        label='Model Source',
                        choices=[_.value for _ in ModelSource],
                        value=ModelSource.HUB_LIB.value)
                    reload_button = gr.Button('Reload Model List')
                    model_id = gr.Dropdown(label='Model ID',
                                           choices=None,
                                           value=None)
                    with gr.Accordion(
                            label=
                            'Model info (Base model and prompt used for training)',
                            open=False):
                        with gr.Row():
                            base_model_used_for_training = gr.Text(
                                label='Base model', interactive=False)
                            prompt_used_for_training = gr.Text(
                                label='Training prompt', interactive=False)
                prompt = gr.Textbox(
                    label='Prompt',
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
                - After training, you can press "Reload Model List" button to load your trained model names.
                - It takes a few minutes to download model first.
                - Expected time to generate an 8-frame video: 70 seconds with T4, 24 seconds with A10G, (10 seconds with A100)
                ''')
            with gr.Column():
                result = gr.Video(label='Result')

        model_source.change(fn=app.reload_model_list_and_update_model_info,
                            inputs=model_source,
                            outputs=[
                                model_id,
                                base_model_used_for_training,
                                prompt_used_for_training,
                            ])
        reload_button.click(fn=app.reload_model_list_and_update_model_info,
                            inputs=model_source,
                            outputs=[
                                model_id,
                                base_model_used_for_training,
                                prompt_used_for_training,
                            ])
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
    return demo


if __name__ == '__main__':
    import os

    hf_token = os.getenv('HF_TOKEN')
    pipe = InferencePipeline(hf_token)
    demo = create_inference_demo(pipe, hf_token)
    demo.queue(max_size=10).launch(share=False)
