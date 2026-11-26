"""Manually send one sample to the optional, local Ollama model."""
import time
from pathlib import Path

MODEL = "qwen2.5vl:3b"
IMAGE_PATH = Path(__file__).resolve().parents[1] / "data/sample/plastic/plastic_01.png"

PROMPT = """
You are a waste image classifier.
Classify the main waste item into exactly one of these labels:
cardboard, glass, metal, paper, plastic, trash.

Return only the label, without explanations.
If the image is unclear, return uncertain.
"""


def main():
    try:
        import ollama
    except ImportError as exc:
        raise SystemExit("This optional CLI requires the ollama Python package.") from exc
    client = ollama.Client(host="http://127.0.0.1:11434", timeout=60)
    start = time.perf_counter()
    response = client.chat(
        model=MODEL,
        messages=[{"role": "user", "content": PROMPT, "images": [str(IMAGE_PATH)]}],
        options={"temperature": 0},
    )
    print("Qwen prediction:", response.message.content)
    print(f"Inference time: {time.perf_counter() - start:.2f} seconds")


if __name__ == "__main__":
    main()
