from diffusers import StableDiffusionPipeline
from pathlib import Path
import torch


def main():
    model_id = "runwayml/stable-diffusion-v1-5"
    cache_dir = "/root/autodl-tmp/cache/hub"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        local_files_only=True,
        torch_dtype=dtype,
    ).to(device)

    output_dir = Path("/root/autodl-tmp/fyp/object_insert_workflow/inputs/generated_eval")
    output_dir.mkdir(parents=True, exist_ok=True)

    cases = [
        # ("spatial_01", "A red cube on the left and a blue sphere on the right, white background"),
        # ("spatial_02", "A green book above a yellow cup on a wooden table"),
        # ("spatial_03", "A black bicycle in front of a white wall, a tree behind the bicycle"),
        # ("spatial_04", "A small dog under a chair, studio lighting"),
        # ("attr_01", "A shiny metallic red sports car, side view, high detail"),
        # ("attr_02", "A large striped orange cat with blue eyes sitting on a sofa"),
        # ("attr_03", "A transparent glass teapot filled with purple tea, close-up"),
        # ("attr_04", "A tiny matte black drone hovering over a city street at dusk"),
        ("simple_02", "A basketball on right side of a cat"),
    ]

    for idx, (name, prompt) in enumerate(cases, start=1):
        seed = 1000 + idx
        generator = torch.Generator(device=device).manual_seed(seed)
        image = pipe(
            prompt=prompt,
            num_inference_steps=30,
            guidance_scale=7.5,
            generator=generator,
        ).images[0]
        image.save(output_dir / f"{name}.png")
        print(f"Saved: {output_dir / f'{name}.png'}")


if __name__ == "__main__":
    main()
