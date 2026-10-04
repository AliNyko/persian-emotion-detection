"""
مرحله ۳: مدل‌های پایه ماشین لرنینگ کلاسیک
TF-IDF به‌عنوان استخراج ویژگی + چند کلاسیفایر مختلف برای مقایسه
"""
import os
import time
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import clone

from data_loader import load_combined_dataset
from preprocessing import preprocess

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def prepare_data(use_emopars: bool = False):
    # نکته‌ی مهم (نتیجه‌ی آزمایش): اضافه کردن EmoPars به داده‌ی آموزش
    # F1-macro مدل‌های کلاسیک رو کم کرد (SVM از 0.475 به 0.420)، چون
    # لیبل‌های EmoPars به‌صورت خودکار و از روی رأی‌گیری ساخته شدن و
    # نویز بیشتری نسبت به لیبل‌های انسانی ArmanEmo دارن. به همین دلیل
    # پیش‌فرض این تابع الان فقط ArmanEmo هست؛ برای آزمایش می‌تونی
    # use_emopars=True رو صدا بزنی.
    train_df, test_df = load_combined_dataset(use_emopars=use_emopars)
    print("در حال پیش‌پردازش متن‌ها (ممکنه چند دقیقه طول بکشه)...")
    train_df["clean_text"] = train_df["text"].apply(lambda t: preprocess(t))
    test_df["clean_text"] = test_df["text"].apply(lambda t: preprocess(t))
    # حذف نمونه‌هایی که بعد از پیش‌پردازش خالی موندن
    train_df = train_df[train_df["clean_text"].str.len() > 0].reset_index(drop=True)
    test_df = test_df[test_df["clean_text"].str.len() > 0].reset_index(drop=True)
    return train_df, test_df


def train_and_evaluate(train_df, test_df, save_prefix=""):
    # نکته (نتیجه‌ی آزمایش چهارم): اضافه کردن ویژگی‌های سطح کاراکتر
    # (char n-gram) در کنار ویژگی‌های سطح کلمه، F1-macro مدل SVM رو از
    # ۰.۴۷۵ به ۰.۴۹۳ رسوند. دلیلش اینه که متن‌های محاوره‌ای و غیررسمی
    # (که در استفاده‌ی واقعی خیلی رایجن) تغییرات املایی زیادی دارن که
    # سطح کلمه به‌تنهایی نمی‌تونه خوب مدلش کنه؛ n-gram های کاراکتری به
    # تکه‌های زیرکلمه‌ای (مثل ریشه‌ی فعل) حساس‌ترن و در برابر این
    # تغییرات مقاوم‌ترن.
    word_vectorizer = TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2),   # هم تک‌کلمه هم دوکلمه‌ای (بایگرام) برای گرفتن کمی context
        min_df=2,
        sublinear_tf=True,
    )
    char_vectorizer = TfidfVectorizer(
        max_features=20000,
        ngram_range=(2, 4),
        min_df=2,
        analyzer="char_wb",   # n-gram کاراکتری در محدوده‌ی هر کلمه
        sublinear_tf=True,
    )
    vectorizer = FeatureUnion([("word", word_vectorizer), ("char", char_vectorizer)])

    X_train = vectorizer.fit_transform(train_df["clean_text"])
    X_test = vectorizer.transform(test_df["clean_text"])
    y_train = train_df["label"]
    y_test = test_df["label"]

    models = {
        "naive_bayes": MultinomialNB(),
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "linear_svm": LinearSVC(class_weight="balanced", max_iter=5000),
    }

    results = []
    trained_models = {}

    for name, model in models.items():
        start = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - start

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1_macro = f1_score(y_test, preds, average="macro")

        print(f"\n=== {name} (زمان آموزش: {elapsed:.1f}s) ===")
        print(f"Accuracy: {acc:.4f} | F1-macro: {f1_macro:.4f}")
        print(classification_report(y_test, preds, zero_division=0))

        results.append({"model": name, "accuracy": acc, "f1_macro": f1_macro, "train_time_s": elapsed})
        trained_models[name] = model

    results_df = pd.DataFrame(results).sort_values("f1_macro", ascending=False)
    best_name = results_df.iloc[0]["model"]
    best_model = trained_models[best_name]

    print("\n================ خلاصه مقایسه مدل‌های کلاسیک ================")
    print(results_df.to_string(index=False))
    print(f"\nبهترین مدل کلاسیک: {best_name}")

    # LinearSVC و بعضی مدل‌ها خروجی احتمالاتی (predict_proba) ندارن.
    # برای این‌که در UI بشه درصد اطمینان مدل رو نشون داد، یک نسخه‌ی
    # کالیبره‌شده از بهترین مدل رو دوباره آموزش می‌دیم.
    print("\nدر حال کالیبره کردن بهترین مدل برای خروجی احتمالاتی (برای UI)...")
    calibrated_model = CalibratedClassifierCV(clone(models[best_name]), method="sigmoid", cv=3)
    calibrated_model.fit(X_train, y_train)

    # ذخیره‌ی وکتورایزر و بهترین مدل برای استفاده در اپ نهایی
    joblib.dump(vectorizer, os.path.join(MODELS_DIR, f"{save_prefix}tfidf_vectorizer.joblib"))
    joblib.dump(calibrated_model, os.path.join(MODELS_DIR, f"{save_prefix}best_classic_model.joblib"))
    results_df.to_csv(os.path.join(MODELS_DIR, f"{save_prefix}classic_results.csv"), index=False)

    return results_df


if __name__ == "__main__":
    train_df, test_df = prepare_data()
    train_and_evaluate(train_df, test_df)
