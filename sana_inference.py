from diffusers import SanaSprintPipeline
import torch
import torchvision
import peft
from peft.tuners.lora.layer import Linear as LoraLinear
import memsave_torch
import types
from safetensors.torch import load_file
from torchvision.utils import make_grid
from pytorch_lightning import seed_everything
import PIL
from huggingface_hub import login
login()

seed_everything(100)

adapter_name = "hypernoise_adapter"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pipe = SanaSprintPipeline.from_pretrained(
    "Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers",
    torch_dtype=torch.bfloat16,
).to(device, torch.bfloat16)

pipe.transformer = peft.PeftModel.from_pretrained(
    pipe.transformer,
    "lucaeyring/HyperNoise_Sana_Sprint_0.6B",
    adapter_name=adapter_name,
).to(device, torch.bfloat16)
pipe.transformer.enable_adapter_layers()

def scaled_base_lora_forward(self, x, *args, **kwargs):
    base_result = self.base_layer(x, *args, **kwargs)
    if self.disable_adapters:
        return base_result
    else:
        lora_out = self.lora_B[adapter_name](self.lora_A[adapter_name](x))
        return lora_out * self.scaling[adapter_name]

# Patching logic
patched_count = 0
for name, module in pipe.transformer.base_model.model.named_modules():
    if name == "proj_out" and isinstance(module, LoraLinear):
        print(f"Found target layer '{name}'. Applying custom forward method.")
        module.forward = types.MethodType(scaled_base_lora_forward, module)
        patched_count += 1

if patched_count > 0:
    print(f"✅ Behavior patched for {patched_count} layer(s).")
else:
    print(f"Error: Could not find layer 'proj_out' to patch.")

# load prompts
fo = open("assets/example_prompts.txt", "r")
prompts = fo.readlines()
fo.close()

batch_size = 1

height, width = 1024, 1024
latent_shape = [
    batch_size,
    pipe.transformer.config.in_channels,
    height // pipe.vae_scale_factor,
    width // pipe.vae_scale_factor
]

hypernoise_images = []
base_images = []
with torch.inference_mode():
    guidance = torch.full([1], 4.5, device=device, dtype=torch.bfloat16)
    guidance = guidance.expand(batch_size).to(torch.bfloat16)
    guidance = guidance * pipe.transformer.config.guidance_embeds_scale

    generator = torch.Generator(device=device).manual_seed(1)
    for index, prompt in enumerate(prompts):
        prompt_embeds, prompt_attention_mask = pipe.encode_prompt(
            prompt=[prompt],
            device=device,
        )
        noise_prompt_embeds = prompt_embeds
        noise_prompt_attention_mask = prompt_attention_mask
        batch_init_latents = torch.randn(
            latent_shape,
            device=device,
            dtype=torch.bfloat16,
            generator=generator,
        )
        pipe.transformer.enable_adapter_layers()
        batch_latents = pipe.transformer(
            hidden_states=batch_init_latents,
            encoder_hidden_states=noise_prompt_embeds, 
            encoder_attention_mask=noise_prompt_attention_mask,
            guidance=guidance,
            timestep=torch.tensor([1.0], device=device, dtype=torch.bfloat16),
        ).sample
        batch_latents = batch_latents + batch_init_latents

        pipe.transformer.disable_adapter_layers()
        batch_images = pipe(
            latents=batch_latents,
            prompt_embeds=prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            num_inference_steps=4,
            generator=generator,
            intermediate_timesteps=None,
            return_dict=False,
            output_type="pt",
        )[0]

        batch_images_old = pipe(
            latents=batch_init_latents,
            prompt_embeds=prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            num_inference_steps=4,
            generator=generator,
            intermediate_timesteps=None,
            return_dict=False,
            output_type="pt",
        )[0]
        hypernoise_images.append(batch_images)
        base_images.append(batch_images_old)

hypernoise_images = torch.cat(hypernoise_images, dim=0)
hypernoise_images = hypernoise_images.detach().cpu().float()
hypernoise_images = torch.clamp(hypernoise_images, 0, 1)
grid = make_grid(hypernoise_images, nrow=5, normalize=False, padding=2)
grid_image = torchvision.transforms.ToPILImage()(grid)
grid_image.save(f"assets/grid_hypernoise.png")

base_images = torch.cat(base_images, dim=0)        
base_images = base_images.detach().cpu().float()
base_images = torch.clamp(base_images, 0, 1)
grid = make_grid(base_images, nrow=5, normalize=False, padding=2)
grid_image = torchvision.transforms.ToPILImage()(grid)
grid_image.save(f"testing_base.png")