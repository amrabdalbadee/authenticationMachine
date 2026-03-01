"""
Egyptian National ID Card Data Extractor  v2
=============================================
Two-pass approach:
  Pass 1 — Raw OCR:  "What Arabic text do you see?" (no JSON, no structure)
  Pass 2 — Parse:    Extract fields from the raw text using targeted questions
  Pass 3 — Regex:    Validate / recover critical fields (ID number, dates)

Backends (in priority order):
  1. Qwen2-VL-2B-Instruct  — Best Arabic OCR, ~3.5 GB RAM (recommended)
  2. moondream2             — Lightest, ~2 GB RAM, limited Arabic
  3. Ollama                 — If installed locally (moondream / llava)
  4. Claude API             — Cloud fallback (requires ANTHROPIC_API_KEY)
"""

import json
import os
import re
import base64
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional


# ─── Data Schema ─────────────────────────────────────────────────────────────

@dataclass
class EgyptianIDData:
    full_name_arabic: Optional[str] = None
    address: Optional[str] = None
    district: Optional[str] = None
    governorate: Optional[str] = None
    national_id_number: Optional[str] = None
    issue_date: Optional[str] = None
    occupation: Optional[str] = None
    gender: Optional[str] = None
    religion: Optional[str] = None
    marital_status: Optional[str] = None
    expiry_date: Optional[str] = None
    date_of_birth: Optional[str] = None
    birth_governorate_code: Optional[str] = None
    confidence: str = "medium"
    backend_used: str = "unknown"
    raw_text_front: Optional[str] = None
    raw_text_back: Optional[str] = None

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=indent)

    def parse_national_id(self) -> None:
        nid = re.sub(r'\D', '', self.national_id_number or '')
        if len(nid) == 14:
            century_code = nid[0]
            yy, mm, dd = nid[1:3], nid[3:5], nid[5:7]
            century = '19' if century_code == '2' else '20'
            self.date_of_birth = f"{century}{yy}-{mm}-{dd}"
            self.birth_governorate_code = nid[7:9]
            self.national_id_number = nid


# ─── Prompts ─────────────────────────────────────────────────────────────────

# Pass 1: no JSON, no field names — just transcribe what you see
RAW_OCR_PROMPT = (
    "This is an Egyptian National ID card (بطاقة تحقيق الشخصية). "
    "Transcribe every Arabic word and every number you can read, exactly as printed. "
    "Do not translate. Do not explain. Output ONLY the raw text."
)


def build_parse_prompt(raw_text: str, side: str) -> str:
    side_hint = (
        "FRONT of the ID card — typically contains: full name (first name on one line, "
        "father/grandfather names below), home address, and the 14-digit national ID number."
        if side == "front" else
        "BACK of the ID card — typically contains: issue date, occupation/job title, "
        "gender (ذكر/أنثى), religion (مسلم/مسيحي), marital status (أعزب/متزوج/مطلق/أرمل), "
        "and card expiry date (البطاقة سارية حتى ...)."
    )
    return f"""You are a data extraction assistant for Egyptian National ID cards.
You have already obtained the raw OCR text from this card image. Your job is to
map that text to structured fields — do NOT add any text that isn't in the raw text below.

CARD SIDE: {side_hint}

RAW OCR TEXT:
\"\"\"
{raw_text}
\"\"\"

Instructions:
- Copy values VERBATIM from the raw text. Never invent or guess.
- national_id_number → exactly 14 digits (digits only, no spaces).
- Dates: convert Eastern Arabic numerals (٠١٢٣...) to Western (0123...) if needed.
  issue_date format: YYYY/MM   |   expiry_date format: YYYY/MM/DD
- If a field is not present in the raw text, output null — never output a description.

Return ONLY this JSON object (no markdown fences, no extra text):
{{
  "full_name_arabic": null,
  "address": null,
  "district": null,
  "governorate": null,
  "national_id_number": null,
  "issue_date": null,
  "occupation": null,
  "gender": null,
  "religion": null,
  "marital_status": null,
  "expiry_date": null
}}"""


# ─── Validation helpers ───────────────────────────────────────────────────────

_PLACEHOLDER_RE = re.compile(
    r'(14\s*digit|YYYY|MM/DD|format|arabic|job\s*title|religion\s*in|marital|your\s|number\s*here)',
    re.IGNORECASE
)

_ARABIC_GENDERS   = {'ذكر', 'أنثى', 'انثى'}
_ARABIC_RELIGIONS = {'مسلم', 'مسيحي', 'مسيحى', 'يهودي'}
_ARABIC_MARITAL   = {'أعزب', 'اعزب', 'متزوج', 'مطلق', 'أرمل', 'ارمل'}


def _clean(val) -> Optional[str]:
    if not val:
        return None
    val = str(val).strip()
    if not val or val.lower() in ('null', 'none', ''):
        return None
    if _PLACEHOLDER_RE.search(val):
        return None
    # Must contain at least some Arabic or meaningful content
    return val


def _validate_id(val) -> Optional[str]:
    if not val:
        return None
    digits = re.sub(r'\D', '', str(val))
    return digits if len(digits) == 14 else None


def _to_western(text: str) -> str:
    for i, ch in enumerate('٠١٢٣٤٥٦٧٨٩'):
        text = text.replace(ch, str(i))
    return text


def _validate_date(val, fmt: str = "any") -> Optional[str]:
    if not val:
        return None
    val = _to_western(str(val).strip())
    if fmt == "YYYY/MM" and re.match(r'^\d{4}/\d{2}$', val):
        return val
    if fmt == "YYYY/MM/DD" and re.match(r'^\d{4}/\d{2}/\d{2}$', val):
        return val
    if fmt == "any" and re.match(r'^\d{4}/\d{2}(/\d{2})?$', val):
        return val
    return None


# ─── Regex fallback ───────────────────────────────────────────────────────────

def _regex_extract(raw: str) -> dict:
    """Extract key fields from raw text using patterns — used when model fails."""
    result: dict = {}
    text = _to_western(raw)

    # 14-digit National ID — digits possibly separated by spaces
    for m in re.finditer(r'\b(\d[\d ]{12,26}\d)\b', text):
        digits = m.group(1).replace(' ', '')
        if len(digits) == 14:
            result['national_id_number'] = digits
            break

    # Expiry date: "حتى YYYY/MM/DD"
    exp = re.search(r'(?:حتى|حتي)\s*(\d{4}/\d{2}/\d{2})', text)
    if exp:
        result['expiry_date'] = exp.group(1)

    # Issue date: standalone YYYY/MM that's NOT the expiry
    for m in re.finditer(r'\b(\d{4}/\d{2})\b', text):
        d = m.group(1)
        if d != result.get('expiry_date'):
            result.setdefault('issue_date', d)

    # Gender / religion / marital — exact token match
    for token in _ARABIC_GENDERS:
        if token in raw:
            result['gender'] = 'ذكر' if token == 'ذكر' else 'أنثى'
            break
    for token in _ARABIC_RELIGIONS:
        if token in raw:
            result['religion'] = token
            break
    for token in _ARABIC_MARITAL:
        if token in raw:
            result['marital_status'] = token
            break

    return result


# ─── Model backends ──────────────────────────────────────────────────────────

class _ModelCache:
    _model = None
    _processor = None
    _backend = None

_cache = _ModelCache()


def _run_qwen2vl(image_path: str, prompt: str, model_name: str = "Qwen/Qwen2-VL-2B-Instruct") -> str:
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info
    import torch

    if _cache._backend != ('qwen2vl', model_name):
        print(f"[Qwen2-VL] Loading model {model_name}...")
        _cache._model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map="cpu",
        )
        _cache._processor = AutoProcessor.from_pretrained(model_name)
        _cache._backend = ('qwen2vl', model_name)

    model, proc = _cache._model, _cache._processor
    messages = [{"role": "user", "content": [
        {"type": "image", "image": f"file://{os.path.abspath(image_path)}"},
        {"type": "text",  "text": prompt},
    ]}]
    text_in = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    img_in, vid_in = process_vision_info(messages)
    inputs = proc(text=[text_in], images=img_in, videos=vid_in,
                  padding=True, return_tensors="pt")
    with __import__('torch').no_grad():
        out = model.generate(**inputs, max_new_tokens=700,
                             temperature=0.05, do_sample=False)
    return proc.batch_decode(
        [o[len(i):] for i, o in zip(inputs.input_ids, out)],
        skip_special_tokens=True
    )[0].strip()


def _run_moondream(image_path: str, prompt: str) -> str:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from PIL import Image
    import torch

    if _cache._backend != 'moondream':
        print("[moondream2] Loading model (~2 GB)...")
        _cache._model = AutoModelForCausalLM.from_pretrained(
            "vikhyatk/moondream2", revision="2025-01-09",
            trust_remote_code=True, torch_dtype=torch.float32)
        _cache._processor = AutoTokenizer.from_pretrained(
            "vikhyatk/moondream2", revision="2025-01-09")
        _cache._model.eval()
        _cache._backend = 'moondream'

    image = Image.open(image_path).convert("RGB")
    enc = _cache._model.encode_image(image)
    return _cache._model.answer_question(enc, prompt, _cache._processor).strip()


def _run_ollama(image_path: str, prompt: str, model_name: str = "llava") -> str:
    import urllib.request, urllib.error
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()
    payload = json.dumps({
        "model": model_name,
        "prompt": prompt,
        "images": [img_b64],
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 700}
    }).encode()
    req = urllib.request.Request(
        "http://localhost:11434/api/generate", data=payload,
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read()).get("response", "").strip()
    except urllib.error.URLError as e:
        raise RuntimeError(f"Ollama unavailable: {e}. Run: ollama serve")


def _run_claude(image_path: str, prompt: str) -> str:
    import anthropic
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")
    client = anthropic.Anthropic(api_key=api_key)
    ext = Path(image_path).suffix.lower().lstrip(".")
    mt = {"jpg": "image/jpeg", "jpeg": "image/jpeg",
          "png": "image/png", "webp": "image/webp"}.get(ext, "image/jpeg")
    with open(image_path, "rb") as f:
        img_b64 = base64.standard_b64encode(f.read()).decode()
    msg = client.messages.create(
        model="claude-opus-4-5", max_tokens=1024,
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": mt, "data": img_b64}},
            {"type": "text",  "text": prompt}
        ]}]
    )
    return msg.content[0].text.strip()


# ─── JSON parser ─────────────────────────────────────────────────────────────

def _parse_json(text: str) -> dict:
    text = re.sub(r'```(?:json)?\s*', '', text).strip().rstrip('`').strip()
    m = re.search(r'\{.*\}', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


# ─── Two-pass processor ───────────────────────────────────────────────────────

def _process_image(image_path: str, side: str, run_fn, verbose: bool = False):
    """Returns (raw_text, validated_dict)."""

    # Pass 1 — raw OCR
    print(f"  [Pass 1] Raw OCR transcription...")
    raw = run_fn(image_path, RAW_OCR_PROMPT)
    if verbose:
        print(f"\n  RAW TEXT:\n{'─'*40}\n{raw}\n{'─'*40}\n")

    # Pass 2 — model parses its own raw text
    print(f"  [Pass 2] Structured field extraction...")
    parse_response = run_fn(image_path, build_parse_prompt(raw, side))
    if verbose:
        print(f"\n  PARSE RESPONSE:\n{'─'*40}\n{parse_response}\n{'─'*40}\n")

    parsed = _parse_json(parse_response)

    # Validate every field
    cleaned = {
        'full_name_arabic':    _clean(parsed.get('full_name_arabic')),
        'address':             _clean(parsed.get('address')),
        'district':            _clean(parsed.get('district')),
        'governorate':         _clean(parsed.get('governorate')),
        'national_id_number':  _validate_id(parsed.get('national_id_number')),
        'issue_date':          _validate_date(parsed.get('issue_date'), "YYYY/MM"),
        'occupation':          _clean(parsed.get('occupation')),
        'expiry_date':         _validate_date(parsed.get('expiry_date'), "YYYY/MM/DD"),
        'gender':              parsed.get('gender') if parsed.get('gender') in _ARABIC_GENDERS else None,
        'religion':            parsed.get('religion') if parsed.get('religion') in _ARABIC_RELIGIONS else None,
        'marital_status':      parsed.get('marital_status') if parsed.get('marital_status') in _ARABIC_MARITAL else None,
    }

    # Pass 3 — regex fallback from raw text for anything still None
    print(f"  [Pass 3] Regex fallback recovery...")
    regex_result = _regex_extract(raw)
    recovered = []
    for k, v in regex_result.items():
        if v and not cleaned.get(k):
            cleaned[k] = v
            recovered.append(k)
    if recovered and verbose:
        print(f"  Recovered via regex: {recovered}")
    elif recovered:
        print(f"  Recovered via regex: {recovered}")

    return raw, {k: v for k, v in cleaned.items() if v is not None}


# ─── Main Extractor ───────────────────────────────────────────────────────────

class EgyptianIDExtractor:
    """
    High-level extractor for Egyptian National ID cards.

    Usage:
        extractor = EgyptianIDExtractor(backend="qwen2vl")
        result = extractor.extract(front_image="front.jpg", back_image="back.jpg")
        print(result.to_json())
    """

    BACKEND_NAMES = {
        "qwen2vl":  "Qwen2-VL-2B-Instruct",
        "moondream":"moondream2",
        "ollama":   "Ollama",
        "claude":   "Claude API",
    }

    def __init__(self, backend: str = "qwen2vl",
                 qwen_model: str = "Qwen/Qwen2-VL-2B-Instruct",
                 ollama_model: str = "llava",
                 verbose: bool = False):
        if backend not in self.BACKEND_NAMES:
            raise ValueError(f"Unknown backend: {backend!r}. Options: {list(self.BACKEND_NAMES)}")
        self.backend = backend
        self.qwen_model = qwen_model
        self.ollama_model = ollama_model
        self.verbose = verbose

    def _make_run_fn(self):
        if self.backend == "qwen2vl":
            m = self.qwen_model
            return lambda img, prompt: _run_qwen2vl(img, prompt, m)
        if self.backend == "moondream":
            return _run_moondream
        if self.backend == "ollama":
            m = self.ollama_model
            return lambda img, prompt: _run_ollama(img, prompt, m)
        if self.backend == "claude":
            return _run_claude

    def extract(self,
                front_image: Optional[str] = None,
                back_image:  Optional[str] = None) -> EgyptianIDData:

        if not front_image and not back_image:
            raise ValueError("Provide at least one image (front or back)")

        run_fn = self._make_run_fn()
        result = EgyptianIDData(backend_used=self.BACKEND_NAMES[self.backend])
        merged: dict = {}

        if front_image:
            print(f"\n── FRONT: {front_image}")
            raw, data = _process_image(front_image, "front", run_fn, self.verbose)
            result.raw_text_front = raw
            merged.update(data)

        if back_image:
            print(f"\n── BACK: {back_image}")
            raw, data = _process_image(back_image, "back", run_fn, self.verbose)
            result.raw_text_back = raw
            # Only fill in fields not already found on front side
            for k, v in data.items():
                if v and not merged.get(k):
                    merged[k] = v

        for f in ["full_name_arabic", "address", "district", "governorate",
                   "national_id_number", "issue_date", "occupation",
                   "gender", "religion", "marital_status", "expiry_date"]:
            if merged.get(f):
                setattr(result, f, merged[f])

        result.parse_national_id()

        key_fields = ["full_name_arabic", "national_id_number",
                      "expiry_date", "gender", "occupation"]
        filled = sum(1 for f in key_fields if getattr(result, f))
        result.confidence = "high" if filled >= 4 else "medium" if filled >= 2 else "low"
        return result


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Extract data from Egyptian National ID cards (two-pass OCR)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Both sides (recommended):
  python extractor.py --front front.jpg --back back.jpg

  # Lighter model:
  python extractor.py --front front.jpg --backend moondream

  # Ollama (install ollama + pull llava first):
  python extractor.py --front front.jpg --backend ollama --ollama-model llava

  # Cloud:
  ANTHROPIC_API_KEY=sk-... python extractor.py --front front.jpg --backend claude

  # Debug — see raw OCR and parse responses:
  python extractor.py --front front.jpg --back back.jpg --verbose
        """
    )
    parser.add_argument("--front",   help="Front side image path")
    parser.add_argument("--back",    help="Back side image path")
    parser.add_argument("--backend", default="qwen2vl",
                        choices=["qwen2vl", "moondream", "ollama", "claude"])
    parser.add_argument("--qwen-model", default="Qwen/Qwen2-VL-2B-Instruct",
                        help="Qwen2-VL model name (default: Qwen2-VL-2B-Instruct)")
    parser.add_argument("--ollama-model", default="llava")
    parser.add_argument("--output",  help="Save JSON output to file")
    parser.add_argument("--verbose", action="store_true",
                        help="Show raw OCR text and intermediate parse responses")
    args = parser.parse_args()

    if not args.front and not args.back:
        parser.error("Provide --front and/or --back")

    extractor = EgyptianIDExtractor(
        backend=args.backend,
        qwen_model=args.qwen_model,
        ollama_model=args.ollama_model,
        verbose=args.verbose
    )
    result = extractor.extract(front_image=args.front, back_image=args.back)

    if not args.verbose:
        result.raw_text_front = None
        result.raw_text_back = None

    print("\n" + "═" * 55)
    print("EXTRACTED ID DATA")
    print("═" * 55)
    print(result.to_json())

    if args.output:
        Path(args.output).write_text(result.to_json(), encoding="utf-8")
        print(f"\n✓ Saved to: {args.output}")


if __name__ == "__main__":
    main()