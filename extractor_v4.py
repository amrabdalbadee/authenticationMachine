"""
Egyptian Document Data Extractor  v4
======================================
Supports three document types:
  • national_id     — Egyptian National ID Card (بطاقة الرقم القومي)
  • driver_license  — Egyptian Driver's License (رخصة القيادة)
  • passport        — Egyptian Passport (جواز السفر)

Three-pass pipeline (all document types):
  Pass 1 — Raw OCR  : "Transcribe everything you see" (no JSON, no structure)
  Pass 2 — Parse    : Map raw text → structured fields (model sees its own OCR output)
  Pass 3 — Regex    : Rule-based recovery for IDs, dates, MRZ, and categorical fields

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

Document type usage
───────────────────
  # National ID (default):
  python extractor.py --doc-type national_id --front front.jpg --back back.jpg

  # Driver's License:
  python extractor.py --doc-type driver_license --image front.jpg 

  # Passport (single image — data page):
  python extractor.py --doc-type passport --image passport_data_page.jpg
"""

import json
import os
import re
import argparse
import time
import tempfile
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Callable


# ══════════════════════════════════════════════════════════════════════════════
# 0.  IMAGE ORIENTATION CORRECTION
# ══════════════════════════════════════════════════════════════════════════════

def _correct_orientation(image_path: str, verbose: bool = False) -> str:
    """
    Fix image rotation / skew BEFORE it reaches the OCR pipeline.

    1. **EXIF transpose** (Pillow) — corrects rotation metadata written by
       phone cameras (90°, 180°, 270° and mirror flips).
    2. **Deskew** (OpenCV) — detects and corrects slight document tilt
       (angles up to ±45°) using the minimum-area bounding rectangle of
       edge contours.

    Returns the path to the corrected image (a temp file).  If no
    correction is needed, returns the original *image_path* unchanged.
    """
    from PIL import Image, ImageOps
    corrected = False

    # ── Step 1: EXIF-based rotation ───────────────────────────────────────
    img = Image.open(image_path)
    exif_transposed = ImageOps.exif_transpose(img)
    if exif_transposed is not img:          # orientation tag was present
        img = exif_transposed
        corrected = True
        if verbose:
            print("  [orient] Applied EXIF transpose.")

    # ── Step 2: OpenCV deskew ─────────────────────────────────────────────
    try:
        import cv2
        import numpy as np

        # Convert current PIL image → OpenCV BGR array
        rgb = img.convert("RGB")
        arr = np.array(rgb)[:, :, ::-1].copy()   # RGB → BGR

        gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        # Binarise + find edges
        gray = cv2.bitwise_not(gray)
        thresh = cv2.threshold(gray, 0, 255,
                               cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        coords = np.column_stack(np.where(thresh > 0))

        if coords.shape[0] > 50:                  # enough pixels to estimate
            angle = cv2.minAreaRect(coords)[-1]   # range (-90, 0]

            # Normalise into (-45, 45] — the actual document skew
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle

            if 0.5 < abs(angle) <= 45:            # skip trivial corrections
                (h, w) = arr.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated = cv2.warpAffine(
                    arr, M, (w, h),
                    flags=cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_REPLICATE,
                )
                # Convert back to PIL
                img = Image.fromarray(cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB))
                corrected = True
                if verbose:
                    print(f"  [orient] Deskewed by {angle:.2f}°")
    except ImportError:
        # OpenCV not installed — EXIF-only correction is still valuable.
        if verbose:
            print("  [orient] OpenCV not available; skipping deskew.")

    if not corrected:
        return image_path                          # nothing to do

    # Save to a temp file that persists for the lifetime of the process.
    suffix = Path(image_path).suffix or ".jpg"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    img.convert("RGB").save(tmp, quality=95)
    tmp.close()
    if verbose:
        print(f"  [orient] Saved corrected image → {tmp.name}")
    return tmp.name


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
    serial_number:          Optional[str] = None   # J12345678 (Front)

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


@dataclass
class DriverLicenseData:
    """Structured data extracted from an Egyptian Driver's License."""

    full_name_arabic:    Optional[str] = None   # الاسم كاملاً بالعربية
    full_name_latin:     Optional[str] = None   # Latin-script name
    national_id_number:  Optional[str] = None   # 14 digits
    nationality:         Optional[str] = None   # الجنسية
    occupation:          Optional[str] = None   # المهنة / الوظيفة
    address:             Optional[str] = None   # العنوان
    issuing_authority:   Optional[str] = None   # وحدة المرور / جهة الإصدار
    traffic_department:  Optional[str] = None   # إدارة المرور
    license_type:        Optional[str] = None   # نوع الرخصة
    license_categories:  Optional[str] = None   # e.g. "B" or "A, B"
    issue_date:          Optional[str] = None   # YYYY/MM/DD
    expiry_date:         Optional[str] = None   # YYYY/MM/DD
    condition:           Optional[str] = None   # any restriction note

    # ── Meta ──────────────────────────────────────────────────────────────────
    confidence:     str            = "medium"
    backend_used:   str            = "unknown"
    raw_text_front: Optional[str]  = None
    raw_text_back:  Optional[str]  = None

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=indent)


@dataclass
class PassportData:
    """Structured data extracted from an Egyptian Passport."""

    # ── Bio-data page ─────────────────────────────────────────────────────────
    full_name_arabic:    Optional[str] = None   # الاسم كاملاً بالعربية
    full_name_latin:     Optional[str] = None   # Full name in Latin
    surname:             Optional[str] = None   # SURNAME (Latin)
    given_names:         Optional[str] = None   # GIVEN NAMES (Latin)
    nationality:         Optional[str] = None   # EGYPTIAN / مصري
    national_id_number:  Optional[str] = None   # 14 digits (الرقم القومي)
    passport_number:     Optional[str] = None   # e.g. A26171466
    date_of_birth:       Optional[str] = None   # YYYY/MM/DD
    place_of_birth:      Optional[str] = None   # محل الميلاد
    sex:                 Optional[str] = None   # M / F
    issue_date:          Optional[str] = None   # YYYY/MM/DD
    expiry_date:         Optional[str] = None   # YYYY/MM/DD
    issuing_authority:   Optional[str] = None   # جهة إصدار الجواز
    profession:          Optional[str] = None   # الوظيفة والمهنة
    address:             Optional[str] = None   # العنوان
    civil_status:        Optional[str] = None   # الموقف التجنيدي

    # ── MRZ (Machine Readable Zone) ───────────────────────────────────────────
    mrz_line1:           Optional[str] = None   # P<EGY... (44 chars)
    mrz_line2:           Optional[str] = None   # digits and < (44 chars)

    # ── Meta ──────────────────────────────────────────────────────────────────
    confidence:     str            = "medium"
    backend_used:   str            = "unknown"
    raw_text_front: Optional[str]  = None   # data page raw text

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=indent)

    def parse_mrz(self) -> None:
        """
        Extract structured fields from MRZ line 2 if present.
        MRZ line 2 format (TD3):  PPPPPPPPP<CNNN YYMMDD C YYMMDD C PPPPPPPPPPPPPP C
        Positions (1-indexed):
          1–9   passport number
          10    check digit
          11–13 nationality
          14–19 DOB YYMMDD
          20    check digit
          21    sex (M/F/<)
          22–27 expiry YYMMDD
          28    check digit
          29–42 personal number
        """
        line = re.sub(r'\s+', '', self.mrz_line2 or '')
        if len(line) < 44:
            return
        # Passport number (strip trailing <)
        pn = line[0:9].rstrip('<')
        if pn and not self.passport_number:
            self.passport_number = pn
        # DOB
        dob = line[13:19]
        if re.match(r'^\d{6}$', dob) and not self.date_of_birth:
            yy = int(dob[:2])
            century = '19' if yy >= 24 else '20'
            self.date_of_birth = f"{century}{dob[:2]}/{dob[2:4]}/{dob[4:6]}"
        # Sex
        sex_char = line[20]
        if sex_char in ('M', 'F') and not self.sex:
            self.sex = sex_char
        # Expiry
        exp = line[21:27]
        if re.match(r'^\d{6}$', exp) and not self.expiry_date:
            yy = int(exp[:2])
            century = '20' if yy <= 50 else '19'
            self.expiry_date = f"{century}{exp[:2]}/{exp[2:4]}/{exp[4:6]}"
        # Personal number (national ID) — pos 29-42, strip < and non-digits
        personal_raw = line[28:42].rstrip("<").strip()
        personal_digits = re.sub(r"\D", "", personal_raw)
        if len(personal_digits) == 14 and not self.national_id_number:
            self.national_id_number = personal_digits


# ══════════════════════════════════════════════════════════════════════════════
# 2.  PROMPTS
# ══════════════════════════════════════════════════════════════════════════════

# ── 2a. National ID prompts ────────────────────────────────────────────────────

RAW_OCR_PROMPT = (
    "هذه بطاقة تحقيق شخصية مصرية. "
    "انسخ كل كلمة عربية وكل رقم تراه تماماً كما هو مطبوع، بدون ترجمة أو شرح. "
    "اكتب النص الخام فقط.\n\n"
    "--- ENGLISH ---\n"
    "This is an Egyptian National ID card. "
    "Transcribe every Arabic word and every number exactly as printed. "
    "No translation, no explanation — output RAW TEXT ONLY."
)


# def build_parse_prompt(raw_text: str, side: str) -> str:
#     side_hint = (
#         "FRONT side — contains: full name (given name on first line, "
#         "father/grandfather names below), home address, 14-digit national ID number."
#         if side == "front" else
#         "BACK side — contains: issue date, occupation/job title, "
#         "gender (ذكر / أنثى), religion (مسلم / مسيحي), "
#         "marital status (أعزب / متزوج / مطلق / أرمل), "
#         "and card expiry date after the phrase (البطاقة سارية حتى)."
#     )
#     return f"""You are a data-extraction assistant for Egyptian National ID cards.
# The raw OCR text below was captured from the {side_hint}

# RAW OCR TEXT:
# \"\"\"
# {raw_text}
# \"\"\"

# Rules:
# 1. Copy values VERBATIM from the raw text — never invent or translate.
# 2. national_id_number → exactly 14 digits (strip spaces, keep only digits).
# 3. Convert Eastern-Arabic numerals (٠١٢٣٤٥٦٧٨٩) to Western (0-9) in dates.
#    issue_date  → YYYY/MM        expiry_date → YYYY/MM/DD
# 4. If a field is absent from the raw text output null — never write a description.

# Return ONLY this JSON object with no markdown fences and no extra text:
# {{
#   "full_name_arabic": null,
#   "address": null,
#   "district": null,
#   "governorate": null,
#   "national_id_number": null,
#   "issue_date": null,
#   "occupation": null,
#   "gender": null,
#   "religion": null,
#   "marital_status": null,
#   "expiry_date": null
# }}"""

def build_parse_prompt(raw_text: str, side: str) -> str:
    if side == "front":
        spatial_hint = (
            "1. Full Name: Look at the 4 lines of bold Arabic text to the right of the photo. "
            "Line 1 is the Given Name, Lines 2-4 are father/grandfather names. "
            "2. Address: Look at the 2 lines of smaller Arabic text directly below the name. "
            "3. National ID: Extract the 14-digit number at the bottom center (starts with 2 or 3). "
            "4. Serial Number: The alpha-numeric code (e.g., J07966517) is at the very bottom left. "
            "IMPORTANT: occupation, issue_date, gender, religion, marital_status, and expiry_date "
            "do NOT appear on the front side — always return null for these fields."
        )
    else:
        spatial_hint = (
            "1. Occupation: The top line of text (e.g., مهندس كهرباء). "
            "2. Issue Date: The YYYY/MM numbers at the very top left (e.g., ٢٠٢٢/١٠). "
            "3. Gender/Religion/Marital Status: The three distinct words in the middle row. "
            "4. Expiry Date: The full date (YYYY/MM/DD) following the phrase 'البطاقة سارية حتى' at the bottom."
        )

    return f"""
Act as an Egyptian Document OCR Expert. You are analyzing the {side.upper()} of an Egyptian National ID.

The raw OCR text below was captured from the card:
RAW OCR TEXT:
\"\"\"
{raw_text}
\"\"\"

### SPATIAL GUIDES:
{spatial_hint}

### EXTRACTION RULES:
1. **Verbatim Arabic:** Extract names and addresses exactly as written.
2. **Digit Conversion:** Convert all Eastern Arabic numerals (٠١٢٣٤٥٦٧٨٩) to Western digits (0-9).
3. **National ID:** Ensure the 14-digit number is captured as a continuous string with no spaces.
4. **Dates:** - Issue Date (Back top-left): Format as YYYY/MM/01.
   - Expiry Date (Back bottom): Format as YYYY/MM/DD.
5. **Null Values:** If a field is not present on this side, return null.

### OUTPUT:
Return ONLY a valid JSON object.

{{
  "full_name_arabic": null,
  "address": null,
  "district": null,
  "governorate": null,
  "national_id_number": null,
  "serial_number": null,
  "issue_date": null,
  "occupation": null,
  "gender": null,
  "religion": null,
  "marital_status": null,
  "expiry_date": null
}}
"""


# ── 2b. Driver's License prompts ──────────────────────────────────────────────

RAW_OCR_PROMPT_DL = (
    "هذه رخصة قيادة مصرية. "
    "انسخ كل كلمة عربية وكل رقم وكل حرف لاتيني تراه تماماً كما هو مطبوع، بدون ترجمة أو شرح. "
    "اكتب النص الخام فقط.\n\n"
    "--- ENGLISH ---\n"
    "This is an Egyptian Driver's License. "
    "Transcribe every Arabic word, every number, and every Latin character exactly as printed. "
    "No translation, no explanation — output RAW TEXT ONLY."
)


def build_parse_prompt_dl(raw_text: str, side: str = "front") -> str:
    spatial_hint = (
        "1. Traffic Department → Look for 'ادارة مرور' or 'إدارة مرور' followed by the governorate (e.g., ادارة مرور الاسكندرية).\n"
        "2. Issuing Authority → Look for 'وحده مرور' or 'وحدة مرور' followed by the area (e.g., وحده مرور برج العرب).\n"
        "3. License Type → Look for 'رخصه قياده' or 'رخصة قيادة' followed by the type (e.g., خاصه, مهنيه).\n"
        "4. National ID Number → Exactly 14 continuous digits starting with 2 or 3.\n"
        "5. Full Name (Arabic) → The longest continuous Arabic name string (e.g., عمرو محمد عبدالبديع اسماعيل ابراهيم).\n"
        "6. Full Name (Latin) → The full name printed in English/Latin characters.\n"
        "7. Address → The physical address block, usually printed under the Latin name (e.g., عمارة مهندس الرى والصرف بالناصرية).\n"
        "8. Nationality → Look for 'مصرى', 'مصري', or 'Egyptian'.\n"
        "9. Occupation → A job title printed near the nationality (e.g., طالب, مهندس, موظف).\n"
        "10. Issue Date → Look for 'تاريخ التحرير'. Extract the date and strictly format as YYYY/MM/DD.\n"
        "11. Expiry Date → Look for 'نهاية الترخيص'. Extract the date and strictly format as YYYY/MM/DD.\n"
        "12. Condition → Any printed restriction text like 'يرتدى نظارة' (wears glasses).\n"
        "13. License Categories → An isolated Latin letter representing the vehicle class (e.g., B, A).\n"
    )

    return f"""
Act as an Expert Data Extractor specialized in Egyptian Driver's Licenses.
Analyze the raw OCR text below. Some text may be fragmented or lack traditional labels.

RAW OCR TEXT:
\"\"\"
{raw_text}
\"\"\"

### FIELD GUIDE:
{spatial_hint}

### EXTRACTION RULES:
1. **Verbatim Arabic**: Extract values exactly as printed (do not correct spelling like 'رخصه' to 'رخصة' if printed with a 'ه').
2. **Digit Conversion**: Replace all Eastern Arabic digits (٠١٢٣٤٥٦٧٨٩) with Western digits (0-9).
3. **National ID**: Ensure it is exactly 14 digits with no spaces or dashes.
4. **Dates**: You MUST format dates as YYYY/MM/DD regardless of how they are printed (e.g., 15/07/2018 becomes 2018/07/15).
5. **No Hallucinations**: Return null if a value is genuinely missing from the raw text.

### OUTPUT:
Return ONLY a valid JSON object. No markdown formatting, no explanations.

{{
  "full_name_arabic": null,
  "full_name_latin": null,
  "national_id_number": null,
  "nationality": null,
  "occupation": null,
  "address": null,
  "issuing_authority": null,
  "traffic_department": null,
  "license_type": null,
  "license_categories": null,
  "issue_date": null,
  "expiry_date": null,
  "condition": null
}}
"""


# ── 2c. Passport prompts ──────────────────────────────────────────────────────

RAW_OCR_PROMPT_PASSPORT = (
    "استخرج جميع النصوص المكتوبة باللغة العربية والإنجليزية من جواز السفر. "
    "من المهم جداً استخراج: الرقم القومي (14 رقم)، العنوان، المهنة، الموقف التجنيدي، والاسم بالكامل. "
    "انسخ كل كلمة ورقم كما هو في الصورة تماماً، بما في ذلك سطور MRZ في الأسفل.\n\n"
    "--- ENGLISH ---\n"
    "This is an Egyptian Passport. "
    "Transcribe ALL text, focusing on both Arabic and English fields. "
    "It is CRITICAL to extract the Arabic fields: National ID (الرقم القومي), Address (العنوان), "
    "Profession (المهنة), Civil Status (الموقف التجنيدي), and Full Arabic Name. "
    "Transcribe accurately, including the MRZ lines. Output RAW TEXT ONLY."
)


def build_parse_prompt_passport(raw_text: str) -> str:
    return f"""
Act as an Egyptian Document OCR Expert specialised in Egyptian passports.
You are analysing the BIO-DATA PAGE of an Egyptian Passport.

IMPORTANT: The data is printed in both Arabic (on the right) and English (on the left). 
You must scan the entire page for specific labels.

RAW OCR TEXT (for reference):
\"\"\"
{raw_text}
\"\"\"

### SPATIAL & LABEL GUIDE:
1.  **Passport Number**: Top right. Label: "رقم الجواز / Passport No". (e.g., A37484706).
2.  **Full Name Arabic**: Right side, bold. Label: "الاسم / Full Name".
3.  **National ID Number**: Look for "الرقم القومي" followed by 14 digits. This is often printed in a smaller font near the middle or bottom of the Arabic text block.
4.  **Profession (المهنة)**: Look for "المهنة" or "الوظيفة". (e.g., ELECTRICAL ENGINEER / مهندس كهرباء).
5.  **Address (العنوان)**: Look for "العنوان". (e.g., الجمهورية العربية مصرية).
6.  **Civil Status (الموقف التجنيدي)**: Look for the label "الموقف التجنيدي" followed by a code or description (e.g., 18).
7.  **MRZ Lines**: The two lines at the very bottom (44 characters each).

### EXTRACTION RULES:
1. **Digit Conversion**: Replace all ٠١٢٣٤٥٦٧٨٩ with 0-9.
2. **National ID**: Ensure it is 14 digits. If the "الرقم القومي" label isn't found, try to extract it from the MRZ Line 2 (positions 29–42).
3. **Dates**: Use YYYY/MM/DD format.
4. **Verbatim**: Copy names and addresses exactly as they appear.

### OUTPUT — return ONLY valid JSON:
{{
  "full_name_arabic": null,
  "full_name_latin": null,
  "surname": null,
  "given_names": null,
  "nationality": null,
  "national_id_number": null,
  "passport_number": null,
  "date_of_birth": null,
  "place_of_birth": null,
  "sex": null,
  "issue_date": null,
  "expiry_date": null,
  "issuing_authority": null,
  "profession": null,
  "address": null,
  "civil_status": null,
  "mrz_line1": null,
  "mrz_line2": null
}}
"""


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


def _regex_extract_dl(raw: str) -> dict:
    """Best-effort rule-based extraction for Egyptian Driver's License."""
    result: dict = {}
    text = _to_western(raw)

    # 1. 14-digit National ID (Starts with 2 or 3)
    for m in re.finditer(r'(?<!\d)(\d[\d ]{12,26}\d)(?!\d)', text):
        digits = m.group(1).replace(' ', '')
        if len(digits) == 14 and digits[0] in ('2', '3'):
            result['national_id_number'] = digits
            break

    # 2. Traffic Department (e.g., ادارة مرور الاسكندرية)
    td_match = re.search(r'((?:إدارة|ادارة)\s*مرور\s+[^\n\d]+)', text)
    if td_match:
        result['traffic_department'] = td_match.group(1).strip()

    # 3. Issuing Authority (e.g., وحده مرور برج العرب)
    ia_match = re.search(r'((?:وحدة|وحده)\s*مرور\s+[^\n\d]+)', text)
    if ia_match:
        result['issuing_authority'] = ia_match.group(1).strip()

    # 4. License Type (e.g., رخصه قياده خاصه)
    lt_match = re.search(r'((?:رخصة|رخصه)\s*(?:قيادة|قياده)\s*(?:خاصة|خاصه|مهنية|مهنيه|ثالثة|ثالثه|ثانية|ثانيه|أولى|اولى|دراجة|دراجه)[^\n\d]*)', text)
    if lt_match:
        result['license_type'] = lt_match.group(1).strip()

    # 5. Condition
    cond_match = re.search(r'(يرتدي نظارة|يرتدى نظارة)', text)
    if cond_match:
        result['condition'] = cond_match.group(1).strip()

    # 6. Nationality
    nat_match = re.search(r'\b(مصرى|مصري|Egyptian)\b', text, re.IGNORECASE)
    if nat_match:
        result['nationality'] = nat_match.group(1).strip()

    # 7. Date parsing helper (Handles DD/MM/YYYY -> YYYY/MM/DD flip)
    def parse_date(date_str):
        date_str = re.sub(r'\s+', '', date_str)
        m = re.search(r'(\d{2,4})[/\\\-](\d{2})[/\\\-](\d{2,4})', date_str)
        if m:
            p1, p2, p3 = m.groups()
            if len(p3) == 4:     # Was DD/MM/YYYY
                return f"{p3}/{p2}/{p1}"
            elif len(p1) == 4:   # Was YYYY/MM/DD
                return f"{p1}/{p2}/{p3}"
        return None

    # 8. Issue Date
    issue_match = re.search(r'(?:تاريخ التحرير|تاريخ الإصدار|تاريخ الاصدار)\s*[:\-]?\s*([\d\s/\\-]+)', text)
    if issue_match:
        d = parse_date(issue_match.group(1))
        if d: result['issue_date'] = d

    # 9. Expiry Date
    exp_match = re.search(r'(?:نهاية الترخيص|تاريخ الانتهاء|صالحة حتى|صالحه حتي)\s*[:\-]?\s*([\d\s/\\-]+)', text)
    if exp_match:
        d = parse_date(exp_match.group(1))
        if d: result['expiry_date'] = d

    # 10. License Categories
    cats = re.findall(r'\b([A-E])\b', text)
    if cats:
        result.setdefault('license_categories', ', '.join(sorted(set(cats))))

    return result


def _regex_extract_passport(raw: str) -> dict:
    """Enhanced rule-based extraction for Egyptian Passport."""
    result: dict = {}
    # Convert numbers for processing
    text_western = _to_western(raw)

    # 1. National ID (14 digits) - Look for label or raw 14-digit string
    nid_match = re.search(r'(?:الرقم القومي|القومي)\s*[:\-]?\s*(\d{14})', text_western)
    if nid_match:
        result['national_id_number'] = nid_match.group(1)
    else:
        # Fallback: any standalone 14 digits
        standalone_nid = re.search(r'(?<!\d)(\d{14})(?!\d)', text_western)
        if standalone_nid:
            result['national_id_number'] = standalone_nid.group(1)

    # 2. Passport Number (Letter + 8 digits)
    pn = re.search(r'\b([A-Z]\d{8})\b', text_western)
    if pn:
        result['passport_number'] = pn.group(1)

    # 3. Profession / المهنة
    prof_match = re.search(r'(?:المهنة|المهنه|الوظيفة)\s*[:\-]?\s*([^\n]+)', raw)
    if prof_match:
        result['profession'] = prof_match.group(1).strip()

    # 4. Civil Status / الموقف التجنيدي
    civil_match = re.search(r'(?:الموقف التجنيدي|تجنيد)\s*[:\-]?\s*(\d+|[^\n]+)', raw)
    if civil_match:
        result['civil_status'] = civil_match.group(1).strip()

    # 5. Dates (YYYY/MM/DD)
    # Passports usually have Issue and Expiry dates. 
    # Issue is generally the earlier date, Expiry is the later.
    found_dates = re.findall(r'(\d{4}/\d{2}/\d{2})', text_western)
    if len(found_dates) >= 2:
        sorted_dates = sorted(found_dates)
        result['issue_date'] = sorted_dates[0]
        result['expiry_date'] = sorted_dates[-1]

    # 6. MRZ Recovery
    mrz_lines = re.findall(r'[A-Z0-9<]{44}', raw.replace(' ', ''))
    if len(mrz_lines) >= 1: result['mrz_line1'] = mrz_lines[0]
    if len(mrz_lines) >= 2: result['mrz_line2'] = mrz_lines[1]

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


# ── 4e.  LightOnOCR-2-1B ──────────────────────────────────────────────────────
def _run_lighton_ocr(image_path: str, prompt: str) -> str:
    """
    LightOnOCR-2-1B — High-quality OCR model from LightOn AI.
    HuggingFace: lightonai/LightOnOCR-2-1B
    RAM: ~4.5 GB fp32 CPU
    Install: pip install "transformers>=5.0.0" pypdfium2 pillow torch
    """
    from transformers import LightOnOcrForConditionalGeneration, LightOnOcrProcessor
    from PIL import Image
    import torch

    MODEL_ID = "lightonai/LightOnOCR-2-1B-bbox"

    if _CACHE.backend != MODEL_ID:
        print(f"[LightOnOCR] Loading model {MODEL_ID}...")
        _CACHE.processor = LightOnOcrProcessor.from_pretrained(MODEL_ID)
        _CACHE.model = LightOnOcrForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float32,
            device_map="cpu",
        )
        _CACHE.model.eval()
        _CACHE.backend = MODEL_ID

    image = Image.open(image_path).convert("RGB")

    # Follow the suggested chat template structure
    messages = [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": prompt}
    ]}]
    # Note: LightOnOCR is often used for full page transcription.
    # We use the processor's chat template which handles the image token placement.
    
    # Use the processor's chat template to get the formatted text (untokenized)
    text_in = _CACHE.processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    
    # Let the processor handle everything (input_ids, pixel_values, image_sizes)
    inputs = _CACHE.processor(text=text_in, images=image, return_tensors="pt")
    
    # Ensure float32 for CPU
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.float32)

    with torch.no_grad():
        out = _CACHE.model.generate(**inputs, max_new_tokens=1024, do_sample=False)

    generated_ids = out[0, inputs["input_ids"].shape[1]:]
    return _CACHE.processor.decode(generated_ids, skip_special_tokens=True).strip()



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
    "lighton-ocr": {
        "name":   "LightOnOCR-2-1B",
        "ram":    "~4.5 GB",
        "arabic": "★★★★☆",
        "fn":     _run_lighton_ocr,
        "note":   "High-quality document OCR; optimized for transcription",
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
                   run_fn: Callable, verbose: bool = False,
                   doc_type: str = "national_id") -> tuple[str, dict]:
    """
    Returns (raw_ocr_text, validated_field_dict).

    Pass 1 — model transcribes the image as raw text (no JSON bias)
    Pass 2 — model maps its own raw text to structured JSON fields
    Pass 3 — regex recovery fills any remaining gaps

    doc_type: "national_id" | "driver_license" | "passport"
    """

    # ── Step 0: Fix rotation / skew ────────────────────────────────────────
    image_path = _correct_orientation(image_path, verbose=verbose)

    # ── Select OCR prompt ─────────────────────────────────────────────────────
    if doc_type == "driver_license":
        ocr_prompt = RAW_OCR_PROMPT_DL
    elif doc_type == "passport":
        ocr_prompt = RAW_OCR_PROMPT_PASSPORT
    else:
        ocr_prompt = RAW_OCR_PROMPT

    # ── Pass 1: raw OCR ───────────────────────────────────────────────────────
    print("  [Pass 1] Raw OCR transcription...")
    raw = run_fn(image_path, ocr_prompt)

    # Some models parrot the prompt back; strip it if so.
    if raw.startswith(ocr_prompt):
        raw = raw[len(ocr_prompt):].strip()

    if verbose:
        print(f"\n  {'─'*42}\n  RAW OCR:\n{raw}\n  {'─'*42}\n")

    # ── Pass 2: structured extraction from raw text ───────────────────────────
    print("  [Pass 2] Structured field extraction...")
    if doc_type == "driver_license":
        parse_prompt = build_parse_prompt_dl(raw, side)
    elif doc_type == "passport":
        parse_prompt = build_parse_prompt_passport(raw)
    else:
        parse_prompt = build_parse_prompt(raw, side)

    parse_resp = run_fn(image_path, parse_prompt)
    if verbose:
        print(f"\n  {'─'*42}\n  PARSE RESPONSE:\n{parse_resp}\n  {'─'*42}\n")

    parsed = _parse_json(parse_resp)

    # ── Pass 2b: validate & clean fields ─────────────────────────────────────
    print("  [Pass 2b] Validating and cleaning fields...")

    if doc_type == "driver_license":
        cleaned: dict = {
            "full_name_arabic":   _clean(parsed.get("full_name_arabic")),
            "full_name_latin":    _clean(parsed.get("full_name_latin")),
            "national_id_number": _validate_id(parsed.get("national_id_number")),
            "nationality":        _clean(parsed.get("nationality")),
            "occupation":         _clean(parsed.get("occupation")),
            "address":            _clean(parsed.get("address")),
            "issuing_authority":  _clean(parsed.get("issuing_authority")),
            "traffic_department": _clean(parsed.get("traffic_department")),
            "license_type":       _clean(parsed.get("license_type")),
            "license_categories": _clean(parsed.get("license_categories")),
            "issue_date":         _validate_date(parsed.get("issue_date"),   "YYYY/MM/DD"),
            "expiry_date":        _validate_date(parsed.get("expiry_date"),  "YYYY/MM/DD"),
            "condition":          _clean(parsed.get("condition")),
        }
        regex_fn = _regex_extract_dl

    elif doc_type == "passport":
        cleaned = {
            "full_name_arabic":   _clean(parsed.get("full_name_arabic")),
            "full_name_latin":    _clean(parsed.get("full_name_latin")),
            "surname":            _clean(parsed.get("surname")),
            "given_names":        _clean(parsed.get("given_names")),
            "nationality":        _clean(parsed.get("nationality")),
            "national_id_number": _validate_id(parsed.get("national_id_number")),
            "passport_number":    _clean(parsed.get("passport_number")),
            "date_of_birth":      _validate_date(parsed.get("date_of_birth"),   "YYYY/MM/DD"),
            "place_of_birth":     _clean(parsed.get("place_of_birth")),
            "sex":                parsed.get("sex") if parsed.get("sex") in ("M", "F") else None,
            "issue_date":         _validate_date(parsed.get("issue_date"),       "YYYY/MM/DD"),
            "expiry_date":        _validate_date(parsed.get("expiry_date"),      "YYYY/MM/DD"),
            "issuing_authority":  _clean(parsed.get("issuing_authority")),
            "profession":         _clean(parsed.get("profession")),
            "address":            _clean(parsed.get("address")),
            "civil_status":       _clean(parsed.get("civil_status")),
            "mrz_line1":          _clean(parsed.get("mrz_line1")),
            "mrz_line2":          _clean(parsed.get("mrz_line2")),
        }
        regex_fn = _regex_extract_passport

    else:  # national_id (original behaviour)
        cleaned = {
            "full_name_arabic":   _clean(parsed.get("full_name_arabic")),
            "address":            _clean(parsed.get("address")),
            "district":           _clean(parsed.get("district")),
            "governorate":        _clean(parsed.get("governorate")),
            "national_id_number": _validate_id(parsed.get("national_id_number")),
            "serial_number":      _clean(parsed.get("serial_number")),
            "issue_date":         _validate_date(parsed.get("issue_date"),   "YYYY/MM"),
            "occupation":         _clean(parsed.get("occupation")),
            "expiry_date":        _validate_date(parsed.get("expiry_date"),  "YYYY/MM/DD"),
            "gender":       parsed.get("gender")        if parsed.get("gender")        in _ARABIC_GENDERS   else None,
            "religion":     parsed.get("religion")      if parsed.get("religion")      in _ARABIC_RELIGIONS else None,
            "marital_status": parsed.get("marital_status") if parsed.get("marital_status") in _ARABIC_MARITAL else None,
        }
        regex_fn = _regex_extract

    # ── Pass 3: regex recovery ────────────────────────────────────────────────
    print("  [Pass 3] Regex fallback recovery...")
    recovered = []
    for k, v in regex_fn(raw).items():
        if v and not cleaned.get(k):
            cleaned[k] = v
            recovered.append(k)
    if recovered:
        print(f"  ✓ Recovered via regex: {recovered}")

    return raw, {k: v for k, v in cleaned.items() if v is not None}



def _ocr_image(image_path: str, run_fn: Callable,
               doc_type: str = "national_id",
               verbose: bool = False) -> str:
    """Pass 1 only — returns raw OCR text for an image without parsing."""
    image_path = _correct_orientation(image_path, verbose=verbose)
    if doc_type == "driver_license":
        ocr_prompt = RAW_OCR_PROMPT_DL
    elif doc_type == "passport":
        ocr_prompt = RAW_OCR_PROMPT_PASSPORT
    else:
        ocr_prompt = RAW_OCR_PROMPT
    raw = run_fn(image_path, ocr_prompt)
    if raw.startswith(ocr_prompt):
        raw = raw[len(ocr_prompt):].strip()
    if verbose:
        print(f"\n  {'─'*42}\n  RAW OCR ({image_path}):\n{raw}\n  {'─'*42}\n")
    return raw

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
            # Back-side exclusive fields always win; front-side fields only fill gaps.
            _BACK_ONLY = {"occupation", "issue_date", "gender", "religion",
                          "marital_status", "expiry_date"}
            for k, v in data.items():
                if v and (k in _BACK_ONLY or not merged.get(k)):
                    merged[k] = v

        _FIELDS = [
            "full_name_arabic", "address", "district", "governorate",
            "national_id_number", "serial_number", "issue_date", "occupation",
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


# ──────────────────────────────────────────────────────────────────────────────

class DriverLicenseExtractor:
    """
    Extract structured data from Egyptian Driver's License images.

    Usage:
        extractor = DriverLicenseExtractor(backend="qwen2vl-2b")
        result = extractor.extract(image="license.jpg")
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

    def extract(self, image: str) -> DriverLicenseData:
        """Single-image extraction — driver licenses have one scannable side."""
        if not image:
            raise ValueError("Provide an image path")

        run_fn = self.backend_cfg["fn"]
        result = DriverLicenseData(backend_used=self.backend_cfg["name"])

        # ── Pass 1: raw OCR ───────────────────────────────────────────────────
        print(f"\n── IMAGE: {image}")
        raw = _ocr_image(image, run_fn, doc_type="driver_license", verbose=self.verbose)
        result.raw_text_front = raw

        # ── Pass 2: structured parse ──────────────────────────────────────────
        print("  [Pass 2] Structured extraction...")
        parse_prompt = build_parse_prompt_dl(raw)
        parse_resp   = run_fn(image, parse_prompt)
        if self.verbose:
            print(f"  PARSE RESPONSE:\n{parse_resp}")
        parsed = _parse_json(parse_resp)

        # ── Pass 2b: validate & clean ─────────────────────────────────────────
        print("  [Pass 2b] Validating and cleaning fields...")
        cleaned: dict = {
            "full_name_arabic":   _clean(parsed.get("full_name_arabic")),
            "full_name_latin":    _clean(parsed.get("full_name_latin")),
            "national_id_number": _validate_id(parsed.get("national_id_number")),
            "nationality":        _clean(parsed.get("nationality")),
            "occupation":         _clean(parsed.get("occupation")),
            "address":            _clean(parsed.get("address")),
            "issuing_authority":  _clean(parsed.get("issuing_authority")),
            "traffic_department": _clean(parsed.get("traffic_department")),
            "license_type":       _clean(parsed.get("license_type")),
            "license_categories": _clean(parsed.get("license_categories")),
            "issue_date":         _validate_date(parsed.get("issue_date"),  "YYYY/MM/DD"),
            "expiry_date":        _validate_date(parsed.get("expiry_date"), "YYYY/MM/DD"),
            "condition":          _clean(parsed.get("condition")),
        }

        # ── Pass 3: regex recovery ────────────────────────────────────────────
        print("  [Pass 3] Regex fallback recovery...")
        recovered = []
        for k, v in _regex_extract_dl(raw).items():
            if v and not cleaned.get(k):
                cleaned[k] = v
                recovered.append(k)
        if recovered:
            print(f"  ✓ Recovered via regex: {recovered}")

        # ── Map to dataclass ──────────────────────────────────────────────────
        _FIELDS = [
            "full_name_arabic", "full_name_latin", "national_id_number",
            "nationality", "occupation", "address", "issuing_authority",
            "traffic_department", "license_type", "license_categories",
            "issue_date", "expiry_date", "condition",
        ]
        for f in _FIELDS:
            if cleaned.get(f):
                setattr(result, f, cleaned[f])

        # ── Confidence ────────────────────────────────────────────────────────
        key_fields = ["full_name_arabic", "national_id_number",
                      "license_categories", "expiry_date"]
        filled = sum(1 for f in key_fields if getattr(result, f))
        result.confidence = "high" if filled >= 3 else "medium" if filled >= 2 else "low"

        return result


# ──────────────────────────────────────────────────────────────────────────────

class PassportExtractor:
    """
    Extract structured data from an Egyptian Passport bio-data page.

    Usage:
        extractor = PassportExtractor(backend="qwen2vl-2b")
        result = extractor.extract(image="passport.jpg")
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

    def extract(self, image: str) -> PassportData:
        """Single-image extraction — scan the passport bio-data page."""
        if not image:
            raise ValueError("Provide an image path")

        run_fn = self.backend_cfg["fn"]
        result = PassportData(backend_used=self.backend_cfg["name"])

        print(f"\n── IMAGE: {image}")
        raw, data = _process_image(image, "front", run_fn, self.verbose,
                                   doc_type="passport")
        result.raw_text_front = raw

        _FIELDS = [
            "full_name_arabic", "full_name_latin", "surname", "given_names",
            "nationality", "national_id_number", "passport_number",
            "date_of_birth", "place_of_birth", "sex",
            "issue_date", "expiry_date", "issuing_authority",
            "profession", "address", "civil_status",
            "mrz_line1", "mrz_line2",
        ]
        for f in _FIELDS:
            if data.get(f):
                setattr(result, f, data[f])

        # Use MRZ to fill any remaining gaps
        result.parse_mrz()

        # Confidence
        key_fields = ["full_name_latin", "passport_number",
                      "date_of_birth", "expiry_date", "sex"]
        filled = sum(1 for f in key_fields if getattr(result, f))
        result.confidence = "high" if filled >= 4 else "medium" if filled >= 2 else "low"

        return result


# ══════════════════════════════════════════════════════════════════════════════
# 9.  CLI
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="extractor",
        description="Egyptian Document OCR — three-pass Vision LLM extractor (v4)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Document types:
  national_id      Egyptian National ID card (front + back)   ← default
  driver_license   Egyptian Driver's License (single image)
  passport         Egyptian Passport bio-data page (single image)

Available backends (use --list-backends for full table):
  qwen2vl-2b   ← default, ~3.5 GB RAM, best Arabic, recommended for CPU
  qwen25vl-3b  — newer model, slightly more accurate, ~4 GB RAM
  arabic-qwen  — fine-tuned on Arabic OCR, ~8 GB RAM
  donut        — lightest at ~1.5 GB, limited Arabic
  (and more — run --list-backends)

Examples:
  # National ID — both sides:
  python extractor.py --doc-type national_id --front front.jpg --back back.jpg

  # Driver's License — single image:
  python extractor.py --doc-type driver_license --image license.jpg

  # Passport — data page:
  python extractor.py --doc-type passport --image passport.jpg

  # Save JSON output:
  python extractor.py --doc-type passport --image passport.jpg --output result.json

  # Debug — see raw OCR text at every pass:
  python extractor.py --doc-type driver_license --image license.jpg --verbose
        """
    )
    parser.add_argument("--doc-type", default="national_id",
                        choices=["national_id", "driver_license", "passport"],
                        help="Document type to extract (default: national_id)")
    parser.add_argument("--front",   metavar="IMAGE", help="Front side image (National ID)")
    parser.add_argument("--back",    metavar="IMAGE", help="Back side image (National ID)")
    parser.add_argument("--image",   metavar="IMAGE", help="Single image (Driver's License or Passport)")
    parser.add_argument("--backend", default="qwen2vl-2b",
                        choices=list(BACKENDS),
                        help="Vision model backend (default: qwen2vl-2b)")
    parser.add_argument("--output",  metavar="FILE",  help="Save JSON output to file")
    parser.add_argument("--save-raw", metavar="FILE", help="Save raw OCR text to JSON file")
    parser.add_argument("--verbose", action="store_true",
                        help="Print raw OCR and intermediate parse responses")
    parser.add_argument("--list-backends", action="store_true",
                        help="Print backend table and exit")

    args = parser.parse_args()

    start_time = time.perf_counter()

    if args.list_backends:
        list_backends()
        return

    if not args.front and not args.back and not args.image:
        parser.error("Provide --image (for driver_license/passport) or --front/--back (for national_id)")

    doc_type = args.doc_type

    # ── Select extractor ──────────────────────────────────────────────────────
    if doc_type == "driver_license":
        if not args.image:
            parser.error("--image is required for driver_license")
        extractor = DriverLicenseExtractor(backend=args.backend, verbose=args.verbose)
        result    = extractor.extract(image=args.image)
        raw_data  = {"raw_text_front": result.raw_text_front}
    elif doc_type == "passport":
        if not args.image:
            parser.error("--image is required for passport")
        extractor = PassportExtractor(backend=args.backend, verbose=args.verbose)
        result    = extractor.extract(image=args.image)
        raw_data  = {"raw_text_front": result.raw_text_front}
    else:
        extractor = EgyptianIDExtractor(backend=args.backend, verbose=args.verbose)
        result    = extractor.extract(front_image=args.front, back_image=args.back)
        raw_data  = {
            "raw_text_front": result.raw_text_front,
            "raw_text_back":  result.raw_text_back,
        }

    # Base directory for results
    results_dir = Path("results") / doc_type / args.backend
    if args.output or args.save_raw:
        results_dir.mkdir(parents=True, exist_ok=True)

    # Save raw OCR if requested before potentially clearing it
    if args.save_raw:
        save_path = results_dir / Path(args.save_raw).name
        save_path.write_text(json.dumps(raw_data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n✓ Raw OCR saved to: {save_path}")

    # Intentionally retaining raw_text_front and raw_text_back even without --verbose
    # so that the user can inspect the OCR output in the parsed JSON.

    print("\n" + "═" * 58)
    print(f"  EXTRACTED {doc_type.upper().replace('_', ' ')} DATA")
    print("═" * 58)
    print(result.to_json())

    if args.output:
        save_path = results_dir / Path(args.output).name
        save_path.write_text(result.to_json(), encoding="utf-8")
        print(f"\n✓ Saved to: {save_path}")

    elapsed = time.perf_counter() - start_time
    print(f"\n✨ Done in {elapsed:.2f} seconds.")

    if args.output or args.save_raw:
        time_path = results_dir / "time.json"
        time_data = {"execution_time_seconds": round(elapsed, 3)}
        time_path.write_text(json.dumps(time_data, indent=2), encoding="utf-8")
        print(f"✓ Execution time saved to: {time_path}")
if __name__ == "__main__":
    main()