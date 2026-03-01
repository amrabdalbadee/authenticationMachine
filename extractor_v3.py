"""
Egyptian National ID Card Data Extractor  v3
=============================================
Two-pass pipeline:
  Pass 1 — Raw OCR  : "Transcribe everything you see" (no JSON, no structure)
  Pass 2 — Parse    : Map raw text → structured fields (model sees its own OCR output)
  Pass 3 — Regex    : Rule-based recovery for ID number, dates, gender, religion

Supported backends
──────────────────
  ID                   Model / HuggingFace repo                       RAM (CPU)
  ─────────────────    ─────────────────────────────────────────────  ─────────
  qwen2vl-2b           Qwen/Qwen2-VL-2B-Instruct                      ~3.5 GB  ← default
  qwen2vl-7b           Qwen/Qwen2-VL-7B-Instruct                      ~8  GB
  qwen25vl-3b          Qwen/Qwen2.5-VL-3B-Instruct                    ~4  GB
  qwen25vl-7b          Qwen/Qwen2.5-VL-7B-Instruct                    ~8  GB
  arabic-qwen          AhmedSSabir/ArabicOCR-Qwen2.5-VL-7B            ~8  GB
  donut                naver-clova-ix/donut-base                       ~1.5GB
  qari                 arbml/Qari                                      ~5  GB
  arabic-glm           THUDM/glm-4v-9b  (Arabic-GLM-OCR-v1 weights)   ~10 GB
  baseer               Abdulmohsen/baseer                              ~5  GB
  deepseek-ocr         deepseek-ai/deepseek-vl2-small                 ~5  GB
"""

import json
import os
import re
import argparse
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Callable


# ══════════════════════════════════════════════════════════════════════════════
# 1.  DATA SCHEMA
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class EgyptianIDData:
    # ── Front side ────────────────────────────────────────────────────────────
    full_name_arabic:       Optional[str] = None   # عمرو محمد عبدالبديع ...
    address:                Optional[str] = None   # عمارة مهندس الرى ...
    district:               Optional[str] = None   # ثان العامرية
    governorate:            Optional[str] = None   # الاسكندرية
    national_id_number:     Optional[str] = None   # 14 digits

    # ── Back side ─────────────────────────────────────────────────────────────
    issue_date:             Optional[str] = None   # YYYY/MM
    occupation:             Optional[str] = None   # مهندس كهرباء
    gender:                 Optional[str] = None   # ذكر / أنثى
    religion:               Optional[str] = None   # مسلم / مسيحي
    marital_status:         Optional[str] = None   # أعزب / متزوج / ...
    expiry_date:            Optional[str] = None   # YYYY/MM/DD

    # ── Derived ───────────────────────────────────────────────────────────────
    date_of_birth:          Optional[str] = None   # YYYY-MM-DD (from NID)
    birth_governorate_code: Optional[str] = None   # 2-digit code

    # ── Meta ──────────────────────────────────────────────────────────────────
    confidence:   str           = "medium"
    backend_used: str           = "unknown"
    raw_text_front: Optional[str] = None
    raw_text_back:  Optional[str] = None

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=indent)

    def parse_national_id(self) -> None:
        """Derive date-of-birth and governorate code from the 14-digit NID."""
        nid = re.sub(r'\D', '', self.national_id_number or '')
        if len(nid) == 14:
            century = '19' if nid[0] == '2' else '20'
            self.date_of_birth          = f"{century}{nid[1:3]}-{nid[3:5]}-{nid[5:7]}"
            self.birth_governorate_code = nid[7:9]
            self.national_id_number     = nid


# ══════════════════════════════════════════════════════════════════════════════
# 2.  PROMPTS
# ══════════════════════════════════════════════════════════════════════════════

RAW_OCR_PROMPT = (
    "هذه بطاقة تحقيق شخصية مصرية. "
    "انسخ كل كلمة عربية وكل رقم تراه تماماً كما هو مطبوع، بدون ترجمة أو شرح. "
    "اكتب النص الخام فقط.\n\n"
    "--- ENGLISH ---\n"
    "This is an Egyptian National ID card. "
    "Transcribe every Arabic word and every number exactly as printed. "
    "No translation, no explanation — output RAW TEXT ONLY."
)


def build_parse_prompt(raw_text: str, side: str) -> str:
    side_hint = (
        "FRONT side — contains: full name (given name on first line, "
        "father/grandfather names below), home address, 14-digit national ID number."
        if side == "front" else
        "BACK side — contains: issue date, occupation/job title, "
        "gender (ذكر / أنثى), religion (مسلم / مسيحي), "
        "marital status (أعزب / متزوج / مطلق / أرمل), "
        "and card expiry date after the phrase (البطاقة سارية حتى)."
    )
    return f"""You are a data-extraction assistant for Egyptian National ID cards.
The raw OCR text below was captured from the {side_hint}

RAW OCR TEXT:
\"\"\"
{raw_text}
\"\"\"

Rules:
1. Copy values VERBATIM from the raw text — never invent or translate.
2. national_id_number → exactly 14 digits (strip spaces, keep only digits).
3. Convert Eastern-Arabic numerals (٠١٢٣٤٥٦٧٨٩) to Western (0-9) in dates.
   issue_date  → YYYY/MM        expiry_date → YYYY/MM/DD
4. If a field is absent from the raw text output null — never write a description.

Return ONLY this JSON object with no markdown fences and no extra text:
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


# ══════════════════════════════════════════════════════════════════════════════
# 3.  VALIDATION & REGEX RECOVERY
# ══════════════════════════════════════════════════════════════════════════════

_PLACEHOLDER_RE = re.compile(
    r'(14\s*digit|YYYY|MM/DD|format|arabic|job\s*title|marital|your\s|number\s*here)',
    re.IGNORECASE,
)
_ARABIC_GENDERS   = {'ذكر', 'أنثى', 'انثى'}
_ARABIC_RELIGIONS = {'مسلم', 'مسيحي', 'مسيحى', 'يهودي'}
_ARABIC_MARITAL   = {'أعزب', 'اعزب', 'متزوج', 'مطلق', 'أرمل', 'ارمل'}


def _to_western(text: str) -> str:
    for i, ch in enumerate('٠١٢٣٤٥٦٧٨٩'):
        text = text.replace(ch, str(i))
    return text


def _clean(val) -> Optional[str]:
    if not val:
        return None
    val = str(val).strip()
    if not val or val.lower() in ('null', 'none'):
        return None
    if _PLACEHOLDER_RE.search(val):
        return None
    return val


def _validate_id(val) -> Optional[str]:
    if not val:
        return None
    digits = re.sub(r'\D', '', _to_western(str(val)))
    return digits if len(digits) == 14 else None


def _validate_date(val, fmt: str = "any") -> Optional[str]:
    if not val:
        return None
    val = _to_western(str(val).strip())
    patterns = {
        "YYYY/MM":    r'^\d{4}/\d{2}$',
        "YYYY/MM/DD": r'^\d{4}/\d{2}/\d{2}$',
        "any":        r'^\d{4}/\d{2}(/\d{2})?$',
    }
    return val if re.match(patterns[fmt], val) else None


def _regex_extract(raw: str) -> dict:
    """Best-effort rule-based extraction — supplements model output."""
    result: dict = {}
    text = _to_western(raw)

    # 14-digit National ID (may have spaces between groups)
    for m in re.finditer(r'\b(\d[\d ]{12,26}\d)\b', text):
        digits = m.group(1).replace(' ', '')
        if len(digits) == 14:
            result['national_id_number'] = digits
            break

    # Expiry date: after "حتى"
    exp = re.search(r'(?:حتى|حتي)\s*(\d{4}/\d{2}/\d{2})', text)
    if exp:
        result['expiry_date'] = exp.group(1)

    # Issue date: standalone YYYY/MM not already used as expiry
    for m in re.finditer(r'\b(\d{4}/\d{2})\b', text):
        d = m.group(1)
        if d != result.get('expiry_date'):
            result.setdefault('issue_date', d)
            break

    # Exact token matches
    for t in _ARABIC_GENDERS:
        if t in raw:
            result['gender'] = 'ذكر' if t == 'ذكر' else 'أنثى'
            break
    for t in _ARABIC_RELIGIONS:
        if t in raw:
            result['religion'] = t
            break
    for t in _ARABIC_MARITAL:
        if t in raw:
            result['marital_status'] = t
            break

    return result


# ══════════════════════════════════════════════════════════════════════════════
# 4.  MODEL BACKENDS
#     Each backend exposes a single callable:  run_fn(image_path, prompt) -> str
# ══════════════════════════════════════════════════════════════════════════════

# ── Shared model cache (keeps weights in RAM across front/back calls) ─────────
class _Cache:
    model     = None
    processor = None
    backend   = None

_CACHE = _Cache()


# ── 4a.  Qwen2-VL family (2B / 7B) and Qwen2.5-VL family (3B / 7B) ──────────
def _make_qwen_runner(model_id: str) -> Callable:
    """
    Works for any Qwen2-VL or Qwen2.5-VL checkpoint.
    Qwen2.5-VL uses Qwen2_5_VL* classes; Qwen2-VL uses Qwen2VL*.
    We detect the family from the model ID and import accordingly.
    """
    def run(image_path: str, prompt: str) -> str:
        import torch

        if _CACHE.backend != model_id:
            print(f"[{model_id}] Loading model (first run: download may take a while)...")

            if "2.5" in model_id or "2_5" in model_id.lower():
                from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
                ModelClass = Qwen2_5_VLForConditionalGeneration
            else:
                from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
                ModelClass = Qwen2VLForConditionalGeneration

            _CACHE.model = ModelClass.from_pretrained(
                model_id,
                torch_dtype=torch.float32,
                device_map="cpu",
            )
            _CACHE.processor = AutoProcessor.from_pretrained(model_id)
            _CACHE.backend = model_id

        try:
            from qwen_vl_utils import process_vision_info
        except ImportError:
            raise ImportError("pip install qwen-vl-utils")

        messages = [{"role": "user", "content": [
            {"type": "image", "image": f"file://{os.path.abspath(image_path)}"},
            {"type": "text",  "text": prompt},
        ]}]

        text_in = _CACHE.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        img_in, vid_in = process_vision_info(messages)
        inputs = _CACHE.processor(
            text=[text_in], images=img_in, videos=vid_in,
            padding=True, return_tensors="pt")

        with torch.no_grad():
            out = _CACHE.model.generate(
                **inputs, max_new_tokens=700,
                temperature=0.05, do_sample=False)

        return _CACHE.processor.batch_decode(
            [o[len(i):] for i, o in zip(inputs.input_ids, out)],
            skip_special_tokens=True,
        )[0].strip()

    return run


# ── 4b.  Donut ────────────────────────────────────────────────────────────────
def _run_donut(image_path: str, prompt: str) -> str:
    """
    Donut (Document Understanding Transformer) by Naver CLOVA.
    It is an encoder-decoder model that reads document images.
    We use the base checkpoint with a free-form generation task.
    Note: Donut was pretrained on synthetic English/Korean docs;
          Arabic accuracy is limited — best treated as a lightweight fallback.
    Install: pip install transformers pillow torch
    """
    from transformers import DonutProcessor, VisionEncoderDecoderModel
    from PIL import Image
    import torch

    MODEL_ID = "naver-clova-ix/donut-base"

    if _CACHE.backend != MODEL_ID:
        print(f"[Donut] Loading model {MODEL_ID}...")
        _CACHE.processor = DonutProcessor.from_pretrained(MODEL_ID)
        _CACHE.model     = VisionEncoderDecoderModel.from_pretrained(MODEL_ID)
        _CACHE.model.eval()
        _CACHE.backend   = MODEL_ID

    proc  = _CACHE.processor
    model = _CACHE.model

    image = Image.open(image_path).convert("RGB")

    # Donut uses special task tokens; we use a document-parse task
    task_prompt = "<s_docvqa><s_question>Transcribe all text from this ID card.</s_question><s_answer>"
    decoder_ids = proc.tokenizer(
        task_prompt, add_special_tokens=False, return_tensors="pt"
    ).input_ids

    pixel_values = proc(image, return_tensors="pt").pixel_values

    with torch.no_grad():
        outputs = model.generate(
            pixel_values,
            decoder_input_ids=decoder_ids,
            max_length=model.decoder.config.max_position_embeddings,
            pad_token_id=proc.tokenizer.pad_token_id,
            eos_token_id=proc.tokenizer.eos_token_id,
            use_cache=True,
            bad_words_ids=[[proc.tokenizer.unk_token_id]],
            return_dict_in_generate=True,
        )

    seq = proc.batch_decode(outputs.sequences)[0]
    # Strip Donut special tokens
    seq = seq.replace(proc.tokenizer.eos_token, "").replace(proc.tokenizer.pad_token, "")
    seq = re.sub(r"<[^>]+>", " ", seq).strip()
    return seq


# ── 4c.  ArabicOCR-Qwen2.5-VL-7B ─────────────────────────────────────────────
def _run_arabic_qwen(image_path: str, prompt: str) -> str:
    """
    ArabicOCR-Qwen2.5-VL-7B — Qwen2.5-VL fine-tuned on Arabic document OCR.
    HuggingFace: AhmedSSabir/ArabicOCR-Qwen2.5-VL-7B
    RAM: ~8 GB (fp32 CPU) | ~4 GB (bf16 GPU)
    Install: pip install transformers qwen-vl-utils torch
    """
    MODEL_ID = "AhmedSSabir/ArabicOCR-Qwen2.5-VL-7B"
    runner = _make_qwen_runner(MODEL_ID)
    return runner(image_path, prompt)


# ── 4d.  Qari OCR ─────────────────────────────────────────────────────────────
def _run_qari(image_path: str, prompt: str) -> str:
    """
    Qari — Arabic OCR model from ARBML.
    HuggingFace: arbml/Qari
    Uses a standard CausalLM + AutoTokenizer interface with image support.
    Install: pip install transformers pillow torch
    """
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    from PIL import Image
    import torch

    MODEL_ID = "NAMAA-Space/Qari-OCR-v0.3-VL-2B-Instruct"

    if _CACHE.backend != MODEL_ID:
        print(f"[Qari] Loading model {MODEL_ID}...")
        _CACHE.processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
        _CACHE.model = Qwen2VLForConditionalGeneration.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
            torch_dtype=torch.float32,
            device_map="cpu",
        )
        _CACHE.model.eval()
        _CACHE.backend = MODEL_ID

    image = Image.open(image_path).convert("RGB")

    # Qari follows a chat-style interface similar to Qwen-VL
    messages = [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": prompt},
    ]}]

    text_in = _CACHE.processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    inputs = _CACHE.processor(text=text_in, images=[image], return_tensors="pt")

    with torch.no_grad():
        out = _CACHE.model.generate(**inputs, max_new_tokens=700, do_sample=False)

    return _CACHE.processor.decode(out[0][inputs.input_ids.shape[-1]:],
                                   skip_special_tokens=True).strip()


# ── 4e.  Arabic-GLM-OCR-v1 ───────────────────────────────────────────────────
def _run_arabic_glm(image_path: str, prompt: str) -> str:
    """
    Arabic-GLM-OCR-v1 — GLM-4V fine-tuned for Arabic document understanding.
    HuggingFace: THUDM/glm-4v-9b  (use Arabic-GLM-OCR-v1 weights when available)
    Checkpoint alias: Arabic-Clinic/Arabic-GLM-OCR-v1
    RAM: ~10 GB fp32 CPU | ~5 GB bf16 GPU
    Install: pip install transformers pillow torch
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from PIL import Image
    import torch

    # Try the Arabic fine-tune first, fall back to base GLM-4V
    for MODEL_ID in ("Arabic-Clinic/Arabic-GLM-OCR-v1", "THUDM/glm-4v-9b"):
        try:
            if _CACHE.backend != MODEL_ID:
                print(f"[Arabic-GLM] Loading {MODEL_ID}...")
                _CACHE.processor = AutoTokenizer.from_pretrained(
                    MODEL_ID, trust_remote_code=True)
                _CACHE.model = AutoModelForCausalLM.from_pretrained(
                    MODEL_ID,
                    trust_remote_code=True,
                    torch_dtype=torch.float32,
                    device_map="cpu",
                )
                _CACHE.model.eval()
                _CACHE.backend = MODEL_ID
            break
        except Exception as e:
            print(f"  [Arabic-GLM] Could not load {MODEL_ID}: {e}")
            continue

    image = Image.open(image_path).convert("RGB")
    inputs = _CACHE.processor.apply_chat_template(
        [{"role": "user", "image": image, "content": prompt}],
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
    )

    with torch.no_grad():
        out = _CACHE.model.generate(**inputs, max_new_tokens=700, do_sample=False)

    return _CACHE.processor.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


# ── 4f.  Baseer ───────────────────────────────────────────────────────────────
def _run_baseer(image_path: str, prompt: str) -> str:
    """
    Baseer — Arabic multimodal VLM from the Abdulmohsen group.
    HuggingFace: Abdulmohsen/baseer
    Follows a standard VLM interface with image + text input.
    RAM: ~5 GB fp32 CPU
    Install: pip install transformers pillow torch
    """
    from transformers import AutoModelForCausalLM, AutoProcessor
    from PIL import Image
    import torch

    MODEL_ID = "Abdulmohsen/baseer"

    if _CACHE.backend != MODEL_ID:
        print(f"[Baseer] Loading model {MODEL_ID}...")
        _CACHE.processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
        _CACHE.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
            torch_dtype=torch.float32,
            device_map="cpu",
        )
        _CACHE.model.eval()
        _CACHE.backend = MODEL_ID

    image = Image.open(image_path).convert("RGB")
    inputs = _CACHE.processor(text=prompt, images=image, return_tensors="pt")

    with torch.no_grad():
        out = _CACHE.model.generate(**inputs, max_new_tokens=700, do_sample=False)

    return _CACHE.processor.decode(out[0], skip_special_tokens=True).strip()


# ── 4g.  DeepSeek OCR ─────────────────────────────────────────────────────────
def _run_deepseek_ocr(image_path: str, prompt: str) -> str:
    """
    DeepSeek-VL2-Small — DeepSeek's vision-language model, strong on document OCR.
    HuggingFace: deepseek-ai/deepseek-vl2-small  (~4.5 GB fp32 CPU)
    For larger accuracy: deepseek-ai/deepseek-vl2  (~20 GB — GPU recommended)
    Install: pip install transformers pillow torch deepseek-vl2
    """
    from transformers import AutoModelForCausalLM, AutoProcessor
    from PIL import Image
    import torch

    MODEL_ID = "deepseek-ai/deepseek-vl2-small"

    if _CACHE.backend != MODEL_ID:
        print(f"[DeepSeek-OCR] Loading {MODEL_ID}...")
        _CACHE.processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
        _CACHE.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
            torch_dtype=torch.float32,
            device_map="cpu",
        )
        _CACHE.model.eval()
        _CACHE.backend = MODEL_ID

    image = Image.open(image_path).convert("RGB")

    conversation = [{"role": "User", "content": f"<image_placeholder>{prompt}"}]
    preamble = _CACHE.processor.apply_sft_template_for_multi_turn_prompts(
        conversations=conversation,
        sft_format=_CACHE.processor.sft_format,
        system_prompt="",
    )
    inputs = _CACHE.processor(preamble, [image], return_tensors="pt").to("cpu")

    with torch.no_grad():
        out = _CACHE.model.generate(
            **inputs,
            max_new_tokens=700,
            do_sample=False,
            pad_token_id=_CACHE.processor.tokenizer.eos_token_id,
        )

    answer = _CACHE.processor.tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return answer.strip()


# ══════════════════════════════════════════════════════════════════════════════
# 5.  BACKEND REGISTRY
# ══════════════════════════════════════════════════════════════════════════════

BACKENDS: dict[str, dict] = {
    # key          display_name                                   run_fn
    "qwen2vl-2b": {
        "name":   "Qwen2-VL-2B-Instruct",
        "ram":    "~3.5 GB",
        "arabic": "★★★★★",
        "fn":     _make_qwen_runner("Qwen/Qwen2-VL-2B-Instruct"),
    },
    "qwen2vl-7b": {
        "name":   "Qwen2-VL-7B-Instruct",
        "ram":    "~8 GB",
        "arabic": "★★★★★",
        "fn":     _make_qwen_runner("Qwen/Qwen2-VL-7B-Instruct"),
    },
    "qwen25vl-3b": {
        "name":   "Qwen2.5-VL-3B-Instruct",
        "ram":    "~4 GB",
        "arabic": "★★★★★",
        "fn":     _make_qwen_runner("Qwen/Qwen2.5-VL-3B-Instruct"),
    },
    "qwen25vl-7b": {
        "name":   "Qwen2.5-VL-7B-Instruct",
        "ram":    "~8 GB",
        "arabic": "★★★★★",
        "fn":     _make_qwen_runner("Qwen/Qwen2.5-VL-7B-Instruct"),
    },
    "arabic-qwen": {
        "name":   "ArabicOCR-Qwen2.5-VL-7B",
        "ram":    "~8 GB",
        "arabic": "★★★★★",
        "fn":     _run_arabic_qwen,
        "note":   "Fine-tuned specifically on Arabic document OCR",
    },
    "donut": {
        "name":   "Donut (naver-clova-ix/donut-base)",
        "ram":    "~1.5 GB",
        "arabic": "★★☆☆☆",
        "fn":     _run_donut,
        "note":   "Lightest option; limited Arabic support",
    },
    "qari": {
        "name":   "Qari OCR (arbml/Qari)",
        "ram":    "~5 GB",
        "arabic": "★★★★☆",
        "fn":     _run_qari,
        "note":   "Arabic-focused OCR from ARBML",
    },
    "arabic-glm": {
        "name":   "Arabic-GLM-OCR-v1",
        "ram":    "~10 GB",
        "arabic": "★★★★☆",
        "fn":     _run_arabic_glm,
        "note":   "GLM-4V fine-tuned for Arabic documents",
    },
    "baseer": {
        "name":   "Baseer",
        "ram":    "~5 GB",
        "arabic": "★★★★☆",
        "fn":     _run_baseer,
        "note":   "Arabic multimodal VLM",
    },
    "deepseek-ocr": {
        "name":   "DeepSeek-VL2-Small",
        "ram":    "~5 GB",
        "arabic": "★★★★☆",
        "fn":     _run_deepseek_ocr,
        "note":   "Strong document OCR; good Arabic reading",
    },
}


def list_backends() -> None:
    print(f"\n{'ID':<15}  {'Model':<40}  {'RAM':<9}  {'Arabic':<12}  Notes")
    print("─" * 95)
    for key, info in BACKENDS.items():
        note = info.get("note", "")
        print(f"{key:<15}  {info['name']:<40}  {info['ram']:<9}  {info['arabic']:<12}  {note}")
    print()


# ══════════════════════════════════════════════════════════════════════════════
# 6.  JSON PARSER
# ══════════════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════════════
# 7.  TWO-PASS IMAGE PROCESSOR
# ══════════════════════════════════════════════════════════════════════════════

def _process_image(image_path: str, side: str,
                   run_fn: Callable, verbose: bool = False) -> tuple[str, dict]:
    """
    Returns (raw_ocr_text, validated_field_dict).

    Pass 1 — model transcribes the image as raw text (no JSON bias)
    Pass 2 — model maps its own raw text to structured JSON fields
    Pass 3 — regex recovery fills any remaining gaps
    """

    # ── Pass 1: raw OCR ───────────────────────────────────────────────────────
    print("  [Pass 1] Raw OCR transcription...")
    raw = run_fn(image_path, RAW_OCR_PROMPT)
    if verbose:
        print(f"\n  {'─'*42}\n  RAW OCR:\n{raw}\n  {'─'*42}\n")

    # ── Pass 2: structured extraction from raw text ───────────────────────────
    print("  [Pass 2] Structured field extraction...")
    parse_resp = run_fn(image_path, build_parse_prompt(raw, side))
    if verbose:
        print(f"\n  {'─'*42}\n  PARSE RESPONSE:\n{parse_resp}\n  {'─'*42}\n")

    parsed = _parse_json(parse_resp)

    cleaned: dict = {
        "full_name_arabic":   _clean(parsed.get("full_name_arabic")),
        "address":            _clean(parsed.get("address")),
        "district":           _clean(parsed.get("district")),
        "governorate":        _clean(parsed.get("governorate")),
        "national_id_number": _validate_id(parsed.get("national_id_number")),
        "issue_date":         _validate_date(parsed.get("issue_date"),   "YYYY/MM"),
        "occupation":         _clean(parsed.get("occupation")),
        "expiry_date":        _validate_date(parsed.get("expiry_date"),  "YYYY/MM/DD"),
        "gender":       parsed.get("gender")        if parsed.get("gender")        in _ARABIC_GENDERS   else None,
        "religion":     parsed.get("religion")      if parsed.get("religion")      in _ARABIC_RELIGIONS else None,
        "marital_status": parsed.get("marital_status") if parsed.get("marital_status") in _ARABIC_MARITAL else None,
    }

    # ── Pass 3: regex recovery ────────────────────────────────────────────────
    print("  [Pass 3] Regex fallback recovery...")
    recovered = []
    for k, v in _regex_extract(raw).items():
        if v and not cleaned.get(k):
            cleaned[k] = v
            recovered.append(k)
    if recovered:
        print(f"  ✓ Recovered via regex: {recovered}")

    return raw, {k: v for k, v in cleaned.items() if v is not None}


# ══════════════════════════════════════════════════════════════════════════════
# 8.  HIGH-LEVEL EXTRACTOR
# ══════════════════════════════════════════════════════════════════════════════

class EgyptianIDExtractor:
    """
    Extract structured data from Egyptian National ID card images.

    Usage:
        extractor = EgyptianIDExtractor(backend="qwen2vl-2b")
        result = extractor.extract(front_image="front.jpg", back_image="back.jpg")
        print(result.to_json())
    """

    def __init__(self,
                 backend: str  = "qwen2vl-2b",
                 verbose: bool = False):
        if backend not in BACKENDS:
            raise ValueError(
                f"Unknown backend: {backend!r}\n"
                f"Available: {list(BACKENDS)}\n"
                f"Run with --list-backends to see all options."
            )
        self.backend_key = backend
        self.backend_cfg = BACKENDS[backend]
        self.verbose     = verbose

    def extract(self,
                front_image: Optional[str] = None,
                back_image:  Optional[str] = None) -> EgyptianIDData:

        if not front_image and not back_image:
            raise ValueError("Provide at least one image (front and/or back)")

        run_fn = self.backend_cfg["fn"]
        result = EgyptianIDData(backend_used=self.backend_cfg["name"])
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
            # Back-side fields only fill gaps — don't overwrite good front data
            for k, v in data.items():
                if v and not merged.get(k):
                    merged[k] = v

        _FIELDS = [
            "full_name_arabic", "address", "district", "governorate",
            "national_id_number", "issue_date", "occupation",
            "gender", "religion", "marital_status", "expiry_date",
        ]
        for f in _FIELDS:
            if merged.get(f):
                setattr(result, f, merged[f])

        result.parse_national_id()

        # Confidence score
        key_fields = ["full_name_arabic", "national_id_number",
                      "expiry_date", "gender", "occupation"]
        filled = sum(1 for f in key_fields if getattr(result, f))
        result.confidence = "high" if filled >= 4 else "medium" if filled >= 2 else "low"

        return result


# ══════════════════════════════════════════════════════════════════════════════
# 9.  CLI
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="extractor",
        description="Egyptian National ID card OCR — two-pass Vision LLM extractor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available backends (use --list-backends for full table):
  qwen2vl-2b   ← default, 4 GB RAM, best Arabic, recommended for CPU
  qwen25vl-3b  — newer model, slightly more accurate, 4 GB RAM
  arabic-qwen  — fine-tuned on Arabic OCR, 8 GB RAM
  deepseek-ocr — strong document reader, 5 GB RAM
  donut        — lightest at 1.5 GB, limited Arabic
  (and more — run --list-backends)

Examples:
  # Both sides (recommended):
  python extractor.py --front front.jpg --back back.jpg

  # Specific backend:
  python extractor.py --front front.jpg --back back.jpg --backend qwen25vl-3b

  # Save to JSON file:
  python extractor.py --front front.jpg --back back.jpg --output result.json

  # Debug — see raw OCR text at every pass:
  python extractor.py --front front.jpg --back back.jpg --verbose
        """
    )
    parser.add_argument("--front",   metavar="IMAGE", help="Front side image path")
    parser.add_argument("--back",    metavar="IMAGE", help="Back side image path")
    parser.add_argument("--backend", default="qwen2vl-2b",
                        choices=list(BACKENDS),
                        help="Vision model backend (default: qwen2vl-2b)")
    parser.add_argument("--output",  metavar="FILE",  help="Save JSON output to file")
    parser.add_argument("--verbose", action="store_true",
                        help="Print raw OCR and intermediate parse responses")
    parser.add_argument("--list-backends", action="store_true",
                        help="Print backend table and exit")

    args = parser.parse_args()

    start_time = time.perf_counter()

    if args.list_backends:
        list_backends()
        return

    if not args.front and not args.back:
        parser.error("Provide --front and/or --back image path(s)")

    extractor = EgyptianIDExtractor(backend=args.backend, verbose=args.verbose)
    result    = extractor.extract(front_image=args.front, back_image=args.back)

    # Hide raw text from output unless --verbose
    if not args.verbose:
        result.raw_text_front = None
        result.raw_text_back  = None

    print("\n" + "═" * 58)
    print("  EXTRACTED ID DATA")
    print("═" * 58)
    print(result.to_json())

    if args.output:
        Path(args.output).write_text(result.to_json(), encoding="utf-8")
        print(f"\n✓ Saved to: {args.output}")

    elapsed = time.perf_counter() - start_time
    print(f"\n✨ Done in {elapsed:.2f} seconds.")
if __name__ == "__main__":
    main()