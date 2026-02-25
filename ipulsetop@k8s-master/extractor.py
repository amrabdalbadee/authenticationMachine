"""
Egyptian National ID Card Data Extractor
=========================================
Extracts structured data from Egyptian National ID cards using Vision LLMs.
Designed to run on 4GB RAM + CPU with multiple backend options.

Backends (in priority order):
  1. Qwen2-VL-2B-Instruct  — Best Arabic OCR, ~3.5GB RAM (recommended)
  2. moondream2             — Lightest, ~2GB RAM, limited Arabic
  3. Ollama                 — If installed locally (moondream / llava)
  4. Claude API             — Cloud fallback (requires ANTHROPIC_API_KEY)
"""

import json
import os
import re
import sys
import base64
import argparse
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional


# ─── Data Schema ─────────────────────────────────────────────────────────────

@dataclass
class EgyptianIDData:
    """Structured output of an Egyptian National ID card."""

    # Front side fields
    full_name_arabic: Optional[str] = None       # e.g., عمرو محمد عبدالبديع اسماعيل ابراهيم
    address: Optional[str] = None                # e.g., عمارة مهندس الرى والصرف بالناصرية
    district: Optional[str] = None               # e.g., ثان العامرية
    governorate: Optional[str] = None            # e.g., الاسكندرية
    national_id_number: Optional[str] = None     # 14-digit number e.g. 29803120201713

    # Back side fields
    issue_date: Optional[str] = None             # e.g., 2022/10
    occupation: Optional[str] = None             # e.g., مهندس كهرباء
    gender: Optional[str] = None                 # ذكر / أنثى
    religion: Optional[str] = None               # مسلم / مسيحي
    marital_status: Optional[str] = None         # أعزب / متزوج / مطلق / أرمل
    expiry_date: Optional[str] = None            # e.g., 2029/10/08

    # Derived from national ID number (first digit = century, next 6 = DOB)
    date_of_birth: Optional[str] = None          # e.g., 1998-03-12
    birth_governorate_code: Optional[str] = None # 2-digit code

    # Confidence / metadata
    confidence: str = "medium"
    backend_used: str = "unknown"
    raw_text_front: Optional[str] = None
    raw_text_back: Optional[str] = None

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=indent)

    def parse_national_id(self) -> None:
        """Derive DOB and governorate from the 14-digit National ID."""
        nid = re.sub(r'\D', '', self.national_id_number or '')
        if len(nid) == 14:
            century_code = nid[0]
            yy = nid[1:3]
            mm = nid[3:5]
            dd = nid[5:7]
            gov_code = nid[7:9]

            century = '19' if century_code == '2' else '20'
            self.date_of_birth = f"{century}{yy}-{mm}-{dd}"
            self.birth_governorate_code = gov_code
            self.national_id_number = nid  # normalized


# ─── Prompt ──────────────────────────────────────────────────────────────────

EXTRACTION_PROMPT = """You are an expert OCR system for Egyptian National ID cards (بطاقة تحقيق الشخصية).
Extract ALL text you can read from this image and return it as a JSON object.

Return ONLY valid JSON with these exact keys (use null for missing/unreadable fields):
{
  "full_name_arabic": "full name in Arabic",
  "address": "address line in Arabic",
  "district": "district/neighborhood",
  "governorate": "governorate name",
  "national_id_number": "14 digit number",
  "issue_date": "YYYY/MM format",
  "occupation": "job title in Arabic",
  "gender": "ذكر or أنثى",
  "religion": "religion in Arabic",
  "marital_status": "marital status in Arabic",
  "expiry_date": "YYYY/MM/DD format"
}

Important:
- The national ID number is always 14 digits
- Read Arabic text carefully, preserving original Arabic script
- Return ONLY the JSON object, no explanation"""


# ─── Backend 1: Qwen2-VL-2B ──────────────────────────────────────────────────

def extract_with_qwen2vl(image_path: str) -> dict:
    """
    Qwen2-VL-2B-Instruct: Best Arabic OCR, ~3.5GB RAM.
    Install: pip install transformers qwen-vl-utils pillow torch
    """
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info
    import torch

    print("[Qwen2-VL] Loading model (first run downloads ~4GB)...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct",
        torch_dtype=torch.float32,   # float32 for CPU compatibility
        device_map="cpu",
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{os.path.abspath(image_path)}"},
                {"type": "text", "text": EXTRACTION_PROMPT},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                       padding=True, return_tensors="pt")

    print("[Qwen2-VL] Running inference on CPU (may take 30-120 seconds)...")
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=512)

    generated = processor.batch_decode(
        [out[len(inp):] for inp, out in zip(inputs.input_ids, output_ids)],
        skip_special_tokens=True
    )[0]

    return parse_json_response(generated), generated


# ─── Backend 2: moondream2 ────────────────────────────────────────────────────

def extract_with_moondream(image_path: str) -> dict:
    """
    moondream2: Lightest (~2GB RAM). Limited Arabic but fast on CPU.
    Install: pip install transformers pillow torch einops
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from PIL import Image
    import torch

    print("[moondream2] Loading model...")
    model_id = "vikhyatk/moondream2"
    revision = "2025-01-09"

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        revision=revision,
        torch_dtype=torch.float32,
    )
    model.eval()

    image = Image.open(image_path).convert("RGB")
    enc_image = model.encode_image(image)

    print("[moondream2] Running inference...")
    response = model.answer_question(enc_image, EXTRACTION_PROMPT, tokenizer)
    return parse_json_response(response), response


# ─── Backend 3: Ollama ────────────────────────────────────────────────────────

def extract_with_ollama(image_path: str, model_name: str = "moondream") -> dict:
    """
    Ollama backend — uses locally installed Ollama server.
    Install: https://ollama.com  then: ollama pull moondream
    Supports: moondream, llava, llava-phi3, bakllava
    """
    import urllib.request
    import urllib.error

    print(f"[Ollama] Using model: {model_name}")

    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()

    payload = json.dumps({
        "model": model_name,
        "prompt": EXTRACTION_PROMPT,
        "images": [img_b64],
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 512}
    }).encode()

    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read())
            response = result.get("response", "")
            return parse_json_response(response), response
    except urllib.error.URLError as e:
        raise RuntimeError(f"Ollama not available: {e}. Start it with: ollama serve")


# ─── Backend 4: Claude API ────────────────────────────────────────────────────

def extract_with_claude_api(image_path: str) -> dict:
    """
    Cloud fallback using Anthropic Claude API.
    Requires: pip install anthropic
    Set: ANTHROPIC_API_KEY environment variable
    """
    import anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")

    client = anthropic.Anthropic(api_key=api_key)

    with open(image_path, "rb") as f:
        img_data = base64.standard_b64encode(f.read()).decode("utf-8")

    ext = Path(image_path).suffix.lower().lstrip(".")
    media_type_map = {"jpg": "image/jpeg", "jpeg": "image/jpeg",
                      "png": "image/png", "webp": "image/webp"}
    media_type = media_type_map.get(ext, "image/jpeg")

    print("[Claude API] Sending to cloud API...")
    message = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64",
                                              "media_type": media_type,
                                              "data": img_data}},
                {"type": "text", "text": EXTRACTION_PROMPT}
            ]
        }]
    )
    response = message.content[0].text
    return parse_json_response(response), response


# ─── JSON Parser ─────────────────────────────────────────────────────────────

def parse_json_response(text: str) -> dict:
    """Extract JSON from model response, handling markdown code blocks."""
    # Strip markdown fences
    text = re.sub(r"```(?:json)?\s*", "", text).strip()
    text = text.rstrip("`").strip()

    # Find first { ... } block
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Fallback: try to parse the whole string
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


# ─── Main Extractor ───────────────────────────────────────────────────────────

class EgyptianIDExtractor:
    """
    High-level extractor that handles one or two ID card images.

    Usage:
        extractor = EgyptianIDExtractor(backend="qwen2vl")
        result = extractor.extract(front_image="front.jpg", back_image="back.jpg")
        print(result.to_json())
    """

    BACKENDS = {
        "qwen2vl":  (extract_with_qwen2vl,  "Qwen2-VL-2B-Instruct"),
        "moondream":(extract_with_moondream, "moondream2"),
        "ollama":   (extract_with_ollama,    "Ollama (local server)"),
        "claude":   (extract_with_claude_api,"Claude API (cloud)"),
    }

    def __init__(self, backend: str = "qwen2vl", ollama_model: str = "moondream"):
        if backend not in self.BACKENDS:
            raise ValueError(f"Unknown backend '{backend}'. Choose from: {list(self.BACKENDS)}")
        self.backend = backend
        self.ollama_model = ollama_model

    def _run_backend(self, image_path: str) -> tuple[dict, str]:
        fn, _ = self.BACKENDS[self.backend]
        if self.backend == "ollama":
            return fn(image_path, self.ollama_model)
        return fn(image_path)

    def extract(self,
                front_image: Optional[str] = None,
                back_image: Optional[str] = None) -> EgyptianIDData:
        """
        Extract data from front and/or back ID images.
        Merges both results into a single EgyptianIDData object.
        """
        if not front_image and not back_image:
            raise ValueError("Provide at least one image (front or back)")

        result = EgyptianIDData(backend_used=self.BACKENDS[self.backend][1])
        merged: dict = {}

        if front_image:
            print(f"\n── Processing FRONT image: {front_image}")
            data, raw = self._run_backend(front_image)
            result.raw_text_front = raw
            merged.update({k: v for k, v in data.items() if v})

        if back_image:
            print(f"\n── Processing BACK image: {back_image}")
            data, raw = self._run_backend(back_image)
            result.raw_text_back = raw
            merged.update({k: v for k, v in data.items() if v})

        # Populate structured fields
        for field_name in [
            "full_name_arabic", "address", "district", "governorate",
            "national_id_number", "issue_date", "occupation",
            "gender", "religion", "marital_status", "expiry_date"
        ]:
            val = merged.get(field_name)
            if val:
                setattr(result, field_name, val)

        # Derive DOB from national ID number
        result.parse_national_id()

        result.confidence = "high" if merged else "low"
        return result


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Extract data from Egyptian National ID cards using Vision LLMs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Both sides with Qwen2-VL (best accuracy, ~3.5GB RAM):
  python extractor.py --front front.jpg --back back.jpg --backend qwen2vl

  # Front only with moondream (lightest, ~2GB RAM):
  python extractor.py --front front.jpg --backend moondream

  # Using Ollama (requires: ollama serve && ollama pull llava):
  python extractor.py --front front.jpg --back back.jpg --backend ollama --ollama-model llava

  # Cloud fallback (requires ANTHROPIC_API_KEY):
  python extractor.py --front front.jpg --back back.jpg --backend claude

Backend RAM requirements:
  qwen2vl   ~3.5GB  (best Arabic OCR, recommended)
  moondream ~2.0GB  (fastest, limited Arabic)
  ollama    varies  (depends on model)
  claude    0GB     (cloud, requires API key)
        """
    )
    parser.add_argument("--front", help="Path to front side image")
    parser.add_argument("--back", help="Path to back side image")
    parser.add_argument("--backend", default="qwen2vl",
                        choices=["qwen2vl", "moondream", "ollama", "claude"],
                        help="Vision model backend to use (default: qwen2vl)")
    parser.add_argument("--ollama-model", default="moondream",
                        help="Ollama model name (default: moondream)")
    parser.add_argument("--output", help="Save JSON output to file")
    parser.add_argument("--raw", action="store_true",
                        help="Include raw OCR text in output")

    args = parser.parse_args()

    if not args.front and not args.back:
        parser.error("Provide --front and/or --back image paths")

    extractor = EgyptianIDExtractor(backend=args.backend,
                                    ollama_model=args.ollama_model)
    result = extractor.extract(front_image=args.front, back_image=args.back)

    if not args.raw:
        result.raw_text_front = None
        result.raw_text_back = None

    output_json = result.to_json()
    print("\n" + "═" * 60)
    print("EXTRACTED ID DATA")
    print("═" * 60)
    print(output_json)

    if args.output:
        Path(args.output).write_text(output_json, encoding="utf-8")
        print(f"\n✓ Saved to: {args.output}")


if __name__ == "__main__":
    main()
