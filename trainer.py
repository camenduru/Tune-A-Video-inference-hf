from __future__ import annotations

import datetime
import os
import pathlib
import shlex
import shutil
import subprocess
import sys

import gradio as gr
import slugify
import torch
from huggingface_hub import HfApi
from omegaconf import OmegaConf

from app_upload import ModelUploader
from utils import save_model_card

sys.path.append('Tune-A-Video')

URL_TO_JOIN_MODEL_LIBRARY_ORG = 'https://huggingface.co/organizations/Tune-A-Video-library/share/YjTcaNJmKyeHFpMBioHhzBcTzCYddVErEk'
ORIGINAL_SPACE_ID = 'Tune-A-Video-library/Tune-A-Video-Training-UI'
SPACE_ID = os.getenv('SPACE_ID', ORIGINAL_SPACE_ID)


class Trainer:
    def __init__(self, hf_token: str | None = None):
        self.hf_token = hf_token
        self.model_uploader = ModelUploader(hf_token)

        self.checkpoint_dir = pathlib.Path('checkpoints')
        self.checkpoint_dir.mkdir(exist_ok=True)

    def download_base_model(self, base_model_id: str) -> str:
        model_dir = self.checkpoint_dir / base_model_id
        if not model_dir.exists():
            org_name = base_model_id.split('/')[0]
            org_dir = self.checkpoint_dir / org_name
            org_dir.mkdir(exist_ok=True)
            subprocess.run(shlex.split(
                f'git clone https://huggingface.co/{base_model_id}'),
                           cwd=org_dir)
        return model_dir.as_posix()

    def join_model_library_org(self, token: str) -> None:
        subprocess.run(
            shlex.split(
                f'curl -X POST -H "Authorization: Bearer {token}" -H "Content-Type: application/json" {URL_TO_JOIN_MODEL_LIBRARY_ORG}'
            ))

    def run(
        self,
        training_video: str,
        training_prompt: str,
        output_model_name: str,
        overwrite_existing_model: bool,
        validation_prompt: str,
        base_model: str,
        resolution_s: str,
        n_steps: int,
        learning_rate: float,
        gradient_accumulation: int,
        seed: int,
        fp16: bool,
        use_8bit_adam: bool,
        checkpointing_steps: int,
        validation_epochs: int,
        upload_to_hub: bool,
        use_private_repo: bool,
        delete_existing_repo: bool,
        upload_to: str,
        remove_gpu_after_training: bool,
        input_token: str,
    ) -> str:
        if SPACE_ID == ORIGINAL_SPACE_ID:
            raise gr.Error(
                'This Space does not work on this Shared UI. Duplicate the Space and attribute a GPU'
            )
        if not torch.cuda.is_available():
            raise gr.Error('CUDA is not available.')
        if training_video is None:
            raise gr.Error('You need to upload a video.')
        if not training_prompt:
            raise gr.Error('The training prompt is missing.')
        if not validation_prompt:
            raise gr.Error('The validation prompt is missing.')

        resolution = int(resolution_s)

        if not output_model_name:
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            output_model_name = f'tune-a-video-{timestamp}'
        output_model_name = slugify.slugify(output_model_name)

        repo_dir = pathlib.Path(__file__).parent
        output_dir = repo_dir / 'experiments' / output_model_name
        if overwrite_existing_model or upload_to_hub:
            shutil.rmtree(output_dir, ignore_errors=True)
        output_dir.mkdir(parents=True)

        if upload_to_hub:
            self.join_model_library_org(
                self.hf_token if self.hf_token else input_token)

        config = OmegaConf.load('Tune-A-Video/configs/man-surfing.yaml')
        config.pretrained_model_path = self.download_base_model(base_model)
        config.output_dir = output_dir.as_posix()
        config.train_data.video_path = training_video.name  # type: ignore
        config.train_data.prompt = training_prompt
        config.train_data.n_sample_frames = 8
        config.train_data.width = resolution
        config.train_data.height = resolution
        config.train_data.sample_start_idx = 0
        config.train_data.sample_frame_rate = 1
        config.validation_data.prompts = [validation_prompt]
        config.validation_data.video_length = 8
        config.validation_data.width = resolution
        config.validation_data.height = resolution
        config.validation_data.num_inference_steps = 50
        config.validation_data.guidance_scale = 7.5
        config.learning_rate = learning_rate
        config.gradient_accumulation_steps = gradient_accumulation
        config.train_batch_size = 1
        config.max_train_steps = n_steps
        config.checkpointing_steps = checkpointing_steps
        config.validation_steps = validation_epochs
        config.seed = seed
        config.mixed_precision = 'fp16' if fp16 else ''
        config.use_8bit_adam = use_8bit_adam

        config_path = output_dir / 'config.yaml'
        with open(config_path, 'w') as f:
            OmegaConf.save(config, f)

        command = f'accelerate launch Tune-A-Video/train_tuneavideo.py --config {config_path}'
        subprocess.run(shlex.split(command))
        save_model_card(save_dir=output_dir,
                        base_model=base_model,
                        training_prompt=training_prompt,
                        test_prompt=validation_prompt,
                        test_image_dir='samples')

        message = 'Training completed!'
        print(message)

        if upload_to_hub:
            upload_message = self.model_uploader.upload_model(
                folder_path=output_dir.as_posix(),
                repo_name=output_model_name,
                upload_to=upload_to,
                private=use_private_repo,
                delete_existing_repo=delete_existing_repo,
                input_token=input_token)
            print(upload_message)
            message = message + '\n' + upload_message

        if remove_gpu_after_training:
            space_id = os.getenv('SPACE_ID')
            if space_id:
                api = HfApi(
                    token=self.hf_token if self.hf_token else input_token)
                api.request_space_hardware(repo_id=space_id,
                                           hardware='cpu-basic')

        return message
