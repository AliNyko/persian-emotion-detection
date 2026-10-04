"""
ماژول بارگذاری دیتاست ArmanEmo
دیتاست از فایل‌های train.tsv و test.tsv خونده می‌شه (فرمت: متن TAB لیبل)
"""
import pandas as pd
import numpy as np
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "arman_raw", "dataset")

LABEL_MAP_FA = {
    "ANGRY": "خشم",
    "FEAR": "ترس",
    "HAPPY": "شادی",
    "HATE": "نفرت",
    "SAD": "غم",
    "SURPRISE": "تعجب",
    "OTHER": "سایر",
}


def load_split(filename: str) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, filename)
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["text", "label"],
        quoting=3,          # نادیده گرفتن نقل‌قول‌های داخل متن (چون دیتاست تمیز tsv نیست)
        on_bad_lines="skip",
        engine="python",
    )
    df = df.dropna(subset=["text", "label"]).reset_index(drop=True)
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 0].reset_index(drop=True)
    df["label_fa"] = df["label"].map(LABEL_MAP_FA)
    return df


def load_dataset():
    train_df = load_split("train.tsv")
    test_df = load_split("test.tsv")
    return train_df, test_df


# --- EmoPars: دیتاست مکمل برای افزایش حجم داده آموزش ---
EMOPARS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "emopars_raw", "dataset.csv")
EMOPARS_TO_ARMAN = {
    "Anger": "ANGRY",
    "Fear": "FEAR",
    "Happiness": "HAPPY",
    "Hatred": "HATE",
    "Sadness": "SAD",
    "Wonder": "SURPRISE",
}


def load_emopars(min_max_vote: int = 3, min_margin: int = 1) -> pd.DataFrame:
    """
    EmoPars برای هر جمله ۶ تا عدد رأی (۰ تا ۵) برای هر احساس داره، نه یک لیبل قطعی.
    اینجا فقط نمونه‌هایی رو نگه می‌داریم که یک احساس به‌وضوح غالب باشه:
    - حداقل رأی برای احساس غالب: min_max_vote
    - فاصله‌ی رأی احساس غالب با دومین احساس پررأی: min_margin
    این فیلتر باعث می‌شه نمونه‌های مبهم و چندبرچسبی (که در EmoPars زیادن) حذف بشن.
    """
    df = pd.read_csv(EMOPARS_PATH)
    emo_cols = list(EMOPARS_TO_ARMAN.keys())
    values = df[emo_cols].values

    sorted_idx = values.argsort(axis=1)
    max_idx = sorted_idx[:, -1]
    max_val = values[np.arange(len(df)), max_idx]
    second_val = values[np.arange(len(df)), sorted_idx[:, -2]]
    margin = max_val - second_val

    mask = (max_val >= min_max_vote) & (margin >= min_margin)
    filtered = df[mask].copy()
    filtered["label"] = [EMOPARS_TO_ARMAN[emo_cols[i]] for i in max_idx[mask]]
    filtered["text"] = filtered["text"].astype(str).str.strip()
    filtered = filtered[filtered["text"].str.len() > 0]
    filtered["label_fa"] = filtered["label"].map(LABEL_MAP_FA)

    return filtered[["text", "label", "label_fa"]].reset_index(drop=True)


def load_combined_dataset(use_emopars: bool = True):
    """
    دیتاست ترکیبی: کل ArmanEmo (train+test با split خودش) + زیرمجموعه‌ی
    باکیفیت EmoPars (فقط اضافه‌شده به train، نه test، تا ارزیابی روی
    استاندارد اصلی ArmanEmo باقی بمونه و قابل مقایسه‌ست).
    """
    train_df, test_df = load_dataset()
    if use_emopars:
        emopars_df = load_emopars()
        print(f"تعداد نمونه‌های باکیفیت انتخاب‌شده از EmoPars: {len(emopars_df)}")
        train_df = pd.concat([train_df, emopars_df], ignore_index=True)
        train_df = train_df.drop_duplicates(subset="text").reset_index(drop=True)
    return train_df, test_df


if __name__ == "__main__":
    train_df, test_df = load_dataset()
    print("تعداد نمونه‌های آموزش:", len(train_df))
    print("تعداد نمونه‌های تست:", len(test_df))
    print("\nتوزیع کلاس‌ها در داده آموزش:")
    print(train_df["label_fa"].value_counts())
