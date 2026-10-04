"""
ماژول پیش‌پردازش متن فارسی
شامل: نرمال‌سازی، حذف لینک/منشن/ایموجی، توکنایز کردن، حذف استاپ‌وردها
"""
import re #Regular Expressions
from hazm import Normalizer, WordTokenizer, stopwords_list

# نکته‌ی مهم: تابع سراسری word_tokenize در hazm هر بار یک WordTokenizer جدید
# می‌سازه که خیلی کند می‌شه (~0.4 ثانیه به ازای هر فراخوانی!). به همین خاطر
# اینجا یک نمونه‌ی سراسری می‌سازیم و برای همه‌ی فراخوانی‌ها از همون استفاده می‌کنیم.
_normalizer = Normalizer()
_tokenizer = WordTokenizer()
_stopwords = set(stopwords_list())

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
MENTION_PATTERN = re.compile(r"@\w+")
HASHTAG_PATTERN = re.compile(r"#(\w+)")
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002700-\U000027BF"
    "\U0001F900-\U0001F9FF"
    "]+",
    flags=re.UNICODE,
)
REPEATED_CHAR_PATTERN = re.compile(r"(.)\1{2,}")  # مثلا "خییییلی" -> "خیلی"
NON_PERSIAN_PATTERN = re.compile(r"[^\u0600-\u06FF\s]")  # فقط حروف فارسی/عربی و فاصله


def clean_text(text: str, remove_non_persian: bool = False) -> str:
    """پاک‌سازی اولیه‌ی متن قبل از نرمال‌سازی hazm"""
    text = URL_PATTERN.sub(" ", text)
    text = MENTION_PATTERN.sub(" ", text)
    text = HASHTAG_PATTERN.sub(r"\1", text)  # فقط # رو حذف کن، کلمه بمونه
    text = EMOJI_PATTERN.sub(" ", text)
    text = REPEATED_CHAR_PATTERN.sub(r"\1\1", text)  # تکرار زیاد حروف رو کم کن
    if remove_non_persian:
        text = NON_PERSIAN_PATTERN.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_text(text: str) -> str:
    """نرمال‌سازی حروف عربی/فارسی، نیم‌فاصله و اعداد با hazm"""
    return _normalizer.normalize(text)


def tokenize(text: str, remove_stopwords: bool = True) -> list:
    tokens = _tokenizer.tokenize(text)
    if remove_stopwords:
        tokens = [t for t in tokens if t not in _stopwords]
    return tokens


def preprocess(text: str, remove_stopwords: bool = True, join: bool = True):
    """پایپلاین کامل: پاک‌سازی -> نرمال‌سازی -> توکنایز -> (اختیاری) حذف استاپ‌ورد"""
    text = clean_text(text)
    text = normalize_text(text)
    tokens = tokenize(text, remove_stopwords=remove_stopwords)
    return " ".join(tokens) if join else tokens


if __name__ == "__main__":
    samples = [
        "خیلییییی خوشحالم که این محصولو خریدم 😍😍 @user1 https://example.com #عالیه",
        "این وضع ب طرز خنده داري گريه داره ...",
    ]
    for s in samples:
        print("قبل :", s)
        print("بعد :", preprocess(s))
        print("---")
