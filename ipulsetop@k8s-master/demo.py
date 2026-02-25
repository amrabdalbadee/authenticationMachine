"""
demo.py — Quick demo of the Egyptian ID Extractor
Run: python demo.py --front path/to/front.jpg --back path/to/back.jpg
"""

import sys
import argparse
sys.path.insert(0, ".")

from src.extractor import EgyptianIDExtractor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--front", help="Front side image path")
    parser.add_argument("--back", help="Back side image path")
    parser.add_argument("--backend", default="qwen2vl",
                        choices=["qwen2vl", "moondream", "ollama", "claude"])
    args = parser.parse_args()

    print(f"🔍 Egyptian ID Extractor — Backend: {args.backend}")
    print("=" * 55)

    extractor = EgyptianIDExtractor(backend=args.backend)
    result = extractor.extract(
        front_image=args.front,
        back_image=args.back
    )

    # Pretty-print key fields
    fields = [
        ("الاسم الكامل",         "full_name_arabic"),
        ("الرقم القومي",          "national_id_number"),
        ("تاريخ الميلاد",         "date_of_birth"),
        ("العنوان",               "address"),
        ("الحي",                  "district"),
        ("المحافظة",              "governorate"),
        ("المهنة",                "occupation"),
        ("النوع",                 "gender"),
        ("الديانة",               "religion"),
        ("الحالة الاجتماعية",     "marital_status"),
        ("تاريخ الإصدار",         "issue_date"),
        ("تاريخ الانتهاء",        "expiry_date"),
    ]

    print("\n📋 Extracted Information:\n")
    for label, attr in fields:
        val = getattr(result, attr, None)
        status = "✅" if val else "❌"
        print(f"  {status}  {label:<22} {val or 'not found'}")

    print(f"\n⚙️  Backend: {result.backend_used}")
    print(f"📊 Confidence: {result.confidence}")
    print("\n📄 Full JSON:\n")
    print(result.to_json())


if __name__ == "__main__":
    main()
