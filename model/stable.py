import numpy as np
from tqdm.auto import tqdm

import torch
import torch.nn as nn

from utils.utils import retrieve_timesteps

from diffusers.utils.torch_utils import randn_tensor


class StableDiff1d(nn.Module):
    def __init__(self, autoencoder, textencoder, unet, scheduler, is_inpaint, cfg, args):
        super().__init__()
        self.autoencoder = autoencoder # autoencoder
        self.textencoder = textencoder # clip
        self.unet = unet # denosing unet
        self.scheduler = scheduler
        
        self.is_inpaint = is_inpaint
        
        self.cfg = cfg
        self.args = args
        self.device = self.args.device
        
        # idk
        self.interrupt = None
        
    def progress_bar(self, iterable=None, total=None):
        if not hasattr(self, "_progress_bar_config"):
            self._progress_bar_config = {}
        elif not isinstance(self._progress_bar_config, dict):
            raise ValueError(
                f"`self._progress_bar_config` should be of type `dict`, but is {type(self._progress_bar_config)}."
            )

        if iterable is not None:
            return tqdm(iterable, **self._progress_bar_config)
        elif total is not None:
            return tqdm(total=total, **self._progress_bar_config)
        else:
            raise ValueError("Either `total` or `iterable` has to be defined.")
    
    # use clip    
    def encode_metric(self, metrics):
        return self.textencoder.embed_text_with_l2norm(metrics) # b, dim, timestep
    
    @property
    def do_classifier_free_guidance(self, ):
        return self._guidance_scale > 1 and self.unet.config.time_cond_proj_dim is None
    
    # Copied from diffusers.pipelines.latent_consistency_models.pipeline_latent_consistency_text2img.LatentConsistencyModelPipeline.get_guidance_scale_embedding
    def get_guidance_scale_embedding(self, w, embedding_dim=512, dtype=torch.float32):
        """
        See https://github.com/google-research/vdm/blob/dc27b98a554f65cdc654b800da5aa1846545d41b/model_vdm.py#L298

        Args:
            timesteps (`torch.Tensor`):
                generate embedding vectors at these timesteps
            embedding_dim (`int`, *optional*, defaults to 512):
                dimension of the embeddings to generate
            dtype:
                data type of the generated embeddings

        Returns:
            `torch.FloatTensor`: Embedding vectors with shape `(len(timesteps), embedding_dim)`
        """
        assert len(w.shape) == 1
        w = w * 1000.0

        half_dim = embedding_dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb)
        emb = w.to(dtype)[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:  # zero pad
            emb = torch.nn.functional.pad(emb, (0, 1))
        assert emb.shape == (w.shape[0], embedding_dim)
        return emb
        
        
        
    def forward(self, 
                 metrics,
                 events = None,
                 mask_condition = None,
                 guidance_scale=1, 
                 latents=None,
                 timesteps = None,
                 num_inference_steps = 1000
                 ):
        
        if self.is_inpaint and (events == None or mask_condition == None):
            print("if want to use inpaint task, events ")
            print(f"events : {events is not None}, mask_condition : {mask_condition is not None}")
            exit()
        
        
        self._guidance_scale = guidance_scale
        # self._guidance_rescale = guidance_rescale
        # self._clip_skip = clip_skip
        # self._cross_attention_kwargs = cross_attention_kwargs
        # self._interrupt = False
        callback_on_step_end = None
        callback = None
        
        # parameters
        batch_size = metrics.shape[0]
        device = self.device
        scale_factor = self.cfg.scale_factor
        length = self.cfg.length

        # _, events_embeds = self.encode_metric(metrics)
        events_embeds, _ = self.encode_metric(metrics)
        events_embeds = events_embeds[:, np.newaxis, :]
        negative_prompt_embeds = None
        
        # The prompt or prompts not to guide the image generation. If not defined, one has to pass
        # `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
        # less than `1`).
        if self.do_classifier_free_guidance: # temp
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
            _ = torch.cat([events_embeds, events_embeds])
            # prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
        
        # timesteps : the list of time steps
        # num_inference_steps : the number of steps to infer
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
        
        # Prepare latent variables
        num_channels_latents = self.cfg.latent_channels
        shape = (batch_size, num_channels_latents, length // scale_factor)
        if latents is None:
            latents = randn_tensor(shape, generator=None, device=device, dtype=events_embeds.dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        if self.is_inpaint:
            noise = latents # difference is that latents is multiplyed with init_noise_sigma
        latents = latents * self.scheduler.init_noise_sigma
        
        # if inpaint, handle mask component
        if self.is_inpaint:
            b, c, w = events.shape
            events = self.autoencoder.normalize_negone_to_one(events)
            masked_event = events * (mask_condition < 0.5)
            latents_masked_event = self.autoencoder.encode(masked_event).latent_dist.sample()
            latents_masked_event = self.autoencoder.config.scaling_factor * latents_masked_event # my mistake
            mask = torch.nn.functional.interpolate(mask_condition, size=(w // self.cfg.scale_factor))
            
        
        # Optionally get Guidance Scale Embedding
        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size)
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)
            
        
        # Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)
        with torch.no_grad():
            with self.progress_bar(total=num_inference_steps) as progress_bar:
                for i, t in enumerate(timesteps):
                    if self.interrupt:
                        continue

                    # expand the latents if we are doing classifier free guidance
                    latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                    
                    if self.is_inpaint:
                        latent_model_input = torch.cat([latent_model_input, mask, latents_masked_event], dim=1)
                    
                    # predict the noise residual
                    noise_pred = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=events_embeds,
                        timestep_cond=timestep_cond,
                        # cross_attention_kwargs=self.cross_attention_kwargs,
                        # added_cond_kwargs=added_cond_kwargs,
                        return_dict=False,
                    )[0]

                    # perform guidance
                    if self.do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                    # if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                    #     # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    #     noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=self.guidance_rescale)

                    # compute the previous noisy sample x_t -> x_t-1
                    # latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
                    latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                    # if callback_on_step_end is not None:
                    #     callback_kwargs = {}
                    #     for k in callback_on_step_end_tensor_inputs:
                    #         callback_kwargs[k] = locals()[k]
                    #     callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    #     latents = callback_outputs.pop("latents", latents)
                    #     prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    #     negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                    # call the callback, if provided
                    if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                        progress_bar.update()
                        # if callback is not None and i % callback_steps == 0:
                        #     step_idx = i // getattr(self.scheduler, "order", 1)
                        #     callback(step_idx, t, latents)
        
        
        decoded = self.autoencoder.decode(latents / self.autoencoder.config.scaling_factor, return_dict=False, generator=None)[0]
        
        # have to do denorm

        return (decoded, )
        
        
        
        
        
        