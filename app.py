import torch
import spaces
import gradio as gr
from diffusers import DiffusionPipeline
import diffusers
import numpy as np
import random

# =========================================================
# MODEL CONFIGURATION
# =========================================================
MAX_SEED = np.iinfo(np.int32).max

# =========================================================
# PROMPT EXAMPLES
# =========================================================
prompt_examples = [
    "The shy college girl, with glasses and a tight plaid skirt, nervously approaches her professor",
    "Her skirt rose a little higher with each gentle push, a soft blush of blush spreading across her cheeks as she felt the satisfying warmth of his breath on her cheek.",
    "a girl in a school uniform having her skirt pulled up by a boy, and then being fucked",
    "Moody mature anime scene of two lovers fuck under neon rain, sensual atmosphere",
    "Moody mature anime scene of two lovers kissing under neon rain, sensual atmosphere",
    "The girl sits on the boy's lap by the window, his hands resting on her waist. She is unbuttoning his shirt, her expression focused and intense.",
    "A girl with long, black hair is sleeping on her desk in the classroom. Her skirt has ridden up, revealing her thighs, and a trail of drool escapes her slightly parted lips.",
    "The waves rolled gently, a slow, sweet kiss of the lip, a slow, slow build of anticipation as their toes bumped gently – a slow, sweet kiss of the lip, a promise of more to come.",
    "Her elegant silk gown swayed gracefully as she approached him, the delicate fabric brushing against her legs. A warm blush spread across her cheeks as she felt his breath on her face.",
    "Her white blouse and light cotton skirt rose a little higher with each gentle push, a soft blush spreading across her cheeks as she felt the satisfying warmth of his breath on her cheek.",
    "A woman in a business suit having her skirt lifted by a man, and then being sexually assaulted.",
    "The older woman sits on the man's lap by the fireplace, his hands resting on her hips. She is unbuttoning his vest, her expression focused and intense. He takes control of the situation as she finishes unbuttoning his shirt, pushing her onto her back and begins to have sex with her.",
    "There is a woman with long black hair. Her face features alluring eyes and full lips, with a slender figure adorned in black lace lingerie. She lies on the bed, loosening her lingerie strap with one hand while seductively glancing downward.",
    "In a dimly lit room, the same woman teases with her dark, flowing hair, now covering her voluptuous breasts, while a black garter belt accentuates her thighs. She sits on the sofa, leaning back, lifting one leg to expose her most private areas through the sheer lingerie.",
    "A woman with glasses, lying on the bed in just her bra, spreads her legs wide, revealing all! She wears a sultry expression, gazing directly at the viewer with her brown eyes, her short black hair cascading over the pillow. Her slim figure, accentuated by the lacy lingerie, exudes a seductive aura.",
    "A soft focus on the girl's face, eyes closed, biting her lip, as her roommate performs oral pleasure, the experienced woman's hair cascading between her thighs.",
    "A woman in a blue hanbok sits on a wooden floor, her legs folded beneath her, gazing out of a window, the sunlight highlighting the graceful lines of her clothing.",
    "The couple, immersed in a wooden outdoor bath, share an intimate moment, her wet kimono clinging to her curves, his hands exploring her body beneath the water's surface.",
    "A steamy shower scene, the twins embrace under the warm water, their soapy hands gliding over each other's curves, their passion intensifying as they explore uncharted territories.",
    "The teacher, with a firm grip, pins the student against the blackboard, her skirt hiked up, exposing her delicate lace panties. Their heavy breathing echoes in the quiet room as they share an intense, intimate moment.",
    "After hours, the girl sits on top of the teacher's lap, riding him on the classroom floor, her hair cascading over her face as she moves with increasing intensity, their bodies glistening with sweat.",
    "In the dimly lit dorm room, the roommates lay entangled in a passionate embrace, their naked bodies glistening with sweat, as the experienced woman teaches her lover the art of kissing and touching.",
    "The once-innocent student, now confident, takes charge, straddling her lover on the couch, their bare skin illuminated by the warm glow of the sunset through the window.",
    "A close-up of the secretary's hand unzipping her boss's dress shirt, her fingers gently caressing his chest, their eyes locked in a heated embrace in the supply closet.",
    "The secretary, in a tight pencil skirt and silk blouse, leans back on the boss's desk, her legs wrapped around his waist, her blouse unbuttoned, revealing her lace bra, as he passionately kisses her, his hands exploring her body.",
    "On the living room couch, one twin sits astride her sister's lap, their lips locked in a passionate kiss, their hands tangled in each other's hair, unraveling a new level of intimacy.",
    "In a dimly lit chamber, the dominant woman, dressed in a leather corset and thigh-high boots, stands tall, her hand gripping her submissive partner's hair, his eyes closed in submission as she instructs him to please her.",
    "The dominant, in a sheer lace bodysuit, sits on a throne-like chair, her legs spread, as the submissive, on his knees, worships her with his tongue, his hands bound behind his back.",
    "A traditional Japanese onsen, with steam rising, a young woman in a colorful kimono kneels on a tatami mat, her back to the viewer, as her male partner, also in a kimono, gently unties her obi, revealing her bare back.",
    "In a serene outdoor setting, the woman, in a vibrant summer kimono, sits on a bench, her legs slightly spread, her partner kneeling before her, his hands gently caressing her exposed thigh.",
]

# =========================================================
# LOAD PIPELINE
# =========================================================
print("Loading Z-Image-Turbo pipeline...")
diffusers.utils.logging.set_verbosity_info()

pipe = DiffusionPipeline.from_pretrained(
    "Tongyi-MAI/Z-Image-Turbo",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=False,
    attn_implementation="kernels-community/vllm-flash-attn3",
)
pipe.to("cuda")

# =========================================================
# RANDOM PROMPT FUNCTION
# =========================================================
def get_random_prompt():
    return random.choice(prompt_examples)

# =========================================================
# IMAGE GENERATOR
# =========================================================
@spaces.GPU
def generate_image(prompt, height, width, num_inference_steps, seed, randomize_seed, num_images):
    if not prompt:
        raise gr.Error("Please enter a prompt.")

    if randomize_seed:
        seed = torch.randint(0, 2**32 - 1, (1,)).item()

    num_images = min(max(1, int(num_images)), 4)

    generator = torch.Generator("cuda").manual_seed(int(seed))

    result = pipe(
        prompt=prompt,
        height=int(height),
        width=int(width),
        num_inference_steps=int(num_inference_steps),
        guidance_scale=0.0,
        generator=generator,
        max_sequence_length=1024,
        num_images_per_prompt=num_images,
        output_type="pil",
    )

    return result.images, seed

# =========================================================
# GRADIO UI
# =========================================================
with gr.Blocks() as demo:
    
    gr.HTML("""
        <style>
            .gradio-container {
                background: linear-gradient(135deg, #fef9f3 0%, #f0e6fa 50%, #e6f0fa 100%) !important;
            }
            footer {display: none !important;}
        </style>
        <div style="text-align: center; margin-bottom: 20px;">
            <h1 style="color: #6b5b7a; font-size: 2.2rem; font-weight: 700; margin-bottom: 0.3rem;">
                🖼️ NSFW Uncensored Adult "Text to Image"
            </h1>
            <p style="color: #8b7b9b; font-size: 1rem;">
                Powered by Z-Image-Turbo Model
            </p>
            <div style="margin-top: 12px; display: flex; justify-content: center; gap: 12px; flex-wrap: wrap;">
                <a href="https://huggingface.co/spaces/Heartsync/FREE-NSFW-HUB" target="_blank">
                    <img src="https://img.shields.io/static/v1?label=FREE&message=NSFW%20HUB&color=%230000ff&labelColor=%23800080&logo=huggingface&logoColor=white&style=for-the-badge" alt="badge">
                </a>
                <a href="https://huggingface.co/spaces/Heartsync/Prompt-Dump" target="_blank">
                    <img src="https://img.shields.io/static/v1?label=100%25%20FREE&message=AI%20Playground&color=%230000ff&labelColor=%23800080&logo=huggingface&logoColor=%23ffa500&style=for-the-badge" alt="badge">
                </a>
                <a href="https://huggingface.co/spaces/Heartsync/NSFW-Uncensored-photo" target="_blank">
                    <img src="https://img.shields.io/static/v1?label=Text%20to%20Image%28Photo%29&message=NSFW%20Uncensored&color=%230000ff&labelColor=%23800080&logo=Huggingface&logoColor=%23ffa500&style=for-the-badge" alt="badge">
                </a>
                <a href="https://huggingface.co/spaces/Heartsync/NSFW-Uncensored-video2" target="_blank">
                    <img src="https://img.shields.io/static/v1?label=Image%20to%20Video%282%29&message=NSFW%20Uncensored&color=%230000ff&labelColor=%23800080&logo=Huggingface&logoColor=%23ffa500&style=for-the-badge" alt="badge">
                </a>
                <a href="https://huggingface.co/spaces/Heartsync/adult" target="_blank">
                    <img src="https://img.shields.io/static/v1?label=Text%20to%20Image%20to%20Video&message=ADULT&color=%23ff00ff&labelColor=%23000080&logo=Huggingface&logoColor=%23ffa500&style=for-the-badge" alt="badge">
                </a>
            </div>
        </div>
    """)


    with gr.Row():
        with gr.Column(scale=1):
            prompt_input = gr.Textbox(
                label="✏️ Prompt",
                placeholder="Describe the image you want to create...",
                lines=3
            )
            
            random_button = gr.Button("🎲 Random Prompt", variant="secondary")
            
            with gr.Row():
                height_input = gr.Slider(512, 2048, 1024, step=64, label="Height")
                width_input = gr.Slider(512, 2048, 1024, step=64, label="Width")
            
            num_images_input = gr.Slider(1, 4, 2, step=1, label="🖼️ Number of Images")

            with gr.Accordion("⚙️ Options", open=False):
                steps_slider = gr.Slider(
                    minimum=1, 
                    maximum=30, 
                    step=1, 
                    value=18, 
                    label="Inference Steps"
                )
                seed_input = gr.Slider(
                    label="Seed", 
                    minimum=0, 
                    maximum=MAX_SEED, 
                    step=1, 
                    value=42
                )
                randomize_seed_checkbox = gr.Checkbox(
                    label="Randomize Seed", 
                    value=True
                )

            generate_button = gr.Button(
                "✨ Generate Image", 
                variant="primary"
            )

        with gr.Column(scale=1):
            output_gallery = gr.Gallery(
                label="🎨 Generated Images",
                height=450,
                columns=2
            )
            used_seed_output = gr.Number(label="Seed Used", interactive=False)

    random_button.click(
        fn=get_random_prompt,
        outputs=[prompt_input]
    )

    generate_button.click(
        fn=generate_image,
        inputs=[prompt_input, height_input, width_input, steps_slider, seed_input, randomize_seed_checkbox, num_images_input],
        outputs=[output_gallery, used_seed_output],
    )

if __name__ == "__main__":
    demo.queue().launch()
