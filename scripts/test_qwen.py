import time
import ollama

MODEL = "qwen2.5vl:3b"

# Đổi đường dẫn này thành đường dẫn ảnh có thật trên máy bạn
IMAGE_PATH = r"data/sample/plastic/plastic_01.png"

prompt = """
You are a waste image classifier.
Classify the main waste item into exactly one of these labels:
cardboard, glass, metal, paper, plastic, trash.

Return only the label, without explanations.
If the image is unclear, return uncertain.
"""

start = time.perf_counter()

response = ollama.chat(
    model=MODEL,
    messages=[
        {
            "role": "user",
            "content": prompt,
            "images": [IMAGE_PATH],
        }
    ],
    options={"temperature": 0},
)

elapsed = time.perf_counter() - start

print("Qwen prediction:", response.message.content)
print(f"Inference time: {elapsed:.2f} seconds")
