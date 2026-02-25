from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import get_message_profile
from mlx_vlm.utils import load_config

# Load a quantized version of Qwen2-VL for your M1 Mac
model_path = "mlx-community/Qwen2-VL-7B-Instruct-4Bit"
model, processor = load(model_path)
config = load_config(model_path)

# Your prompt requesting JSON format
prompt = """
You are a highly accurate data extraction assistant. Look at these images of the front and back of an Egyptian National ID card. 
Extract the following information in Arabic and output ONLY a valid JSON object with the following keys:
"first_name", "last_name", "address", "national_id", "profession", "gender", "religion", "marital_status", "expiration_date".
Do not include any markdown formatting, just the raw JSON.
"""

# Path to the downloaded images
images = ["ID_Front.jpeg", "ID_Back.jpeg"]

# Generate the output
output = generate(
    model, 
    processor, 
    prompt=prompt, 
    image=images, 
    verbose=True,
    max_tokens=500
)

print(output)