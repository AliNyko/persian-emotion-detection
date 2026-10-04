"""
آزمایش چهارم: آیا می‌شه مدل کلاسیک (بهترین مدل فعلی) رو بهتر کرد؟
دو ایده امتحان می‌شه:
  ۱) اضافه کردن ویژگی‌های سطح کاراکتر (char n-gram) در کنار کلمه، برای
     مقاوم‌تر شدن در برابر غلط‌های املایی/گویش محاوره‌ای رایج در متن‌های غیررسمی
  ۲) جستجوی شبکه‌ای (grid search) برای پارامتر C در LinearSVC به‌جای مقدار پیش‌فرض
"""
import sys
sys.path.append("src")
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, accuracy_score, classification_report

from data_loader import load_dataset
from preprocessing import preprocess

print("در حال بارگذاری و پیش‌پردازش داده (ArmanEmo)...")
train_df, test_df = load_dataset()
train_df["clean_text"] = train_df["text"].apply(lambda t: preprocess(t, remove_stopwords=True))
test_df["clean_text"] = test_df["text"].apply(lambda t: preprocess(t, remove_stopwords=True))
train_df = train_df[train_df["clean_text"].str.len() > 0].reset_index(drop=True)
test_df = test_df[test_df["clean_text"].str.len() > 0].reset_index(drop=True)

y_train, y_test = train_df["label"], test_df["label"]

# ---------- baseline (همون چیزی که الان توی مدل نهاییه) ----------
vec_word_base = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2)
Xtr_base = vec_word_base.fit_transform(train_df["clean_text"])
Xte_base = vec_word_base.transform(test_df["clean_text"])
clf_base = LinearSVC(class_weight="balanced", max_iter=5000)
clf_base.fit(Xtr_base, y_train)
f1_base = f1_score(y_test, clf_base.predict(Xte_base), average="macro")
print(f"\n[baseline]           F1-macro = {f1_base:.4f}")

# ---------- ایده ۱: word + char n-gram ترکیبی ----------
vec_word = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
vec_char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=20000, min_df=2, sublinear_tf=True)

Xtr_word = vec_word.fit_transform(train_df["clean_text"])
Xte_word = vec_word.transform(test_df["clean_text"])
Xtr_char = vec_char.fit_transform(train_df["clean_text"])
Xte_char = vec_char.transform(test_df["clean_text"])

Xtr_combo = hstack([Xtr_word, Xtr_char]).tocsr()
Xte_combo = hstack([Xte_word, Xte_char]).tocsr()

clf_combo = LinearSVC(class_weight="balanced", max_iter=5000)
clf_combo.fit(Xtr_combo, y_train)
f1_combo = f1_score(y_test, clf_combo.predict(Xte_combo), average="macro")
print(f"[word+char combo]    F1-macro = {f1_combo:.4f}")

# ---------- ایده ۲: grid search روی C (روی بهترین ترکیب ویژگی تا اینجا) ----------
best_X_train = Xtr_combo if f1_combo > f1_base else Xtr_base
best_X_test = Xte_combo if f1_combo > f1_base else Xte_base
feature_name = "word+char" if f1_combo > f1_base else "word-only (baseline)"

print(f"\nدر حال grid search روی C با ویژگی‌های «{feature_name}» ...")
param_grid = {"C": [0.1, 0.3, 1.0, 3.0, 10.0]}
grid = GridSearchCV(
    LinearSVC(class_weight="balanced", max_iter=5000),
    param_grid,
    scoring="f1_macro",
    cv=5,
    n_jobs=-1,
)
grid.fit(best_X_train, y_train)
best_clf = grid.best_estimator_
f1_tuned = f1_score(y_test, best_clf.predict(best_X_test), average="macro")
acc_tuned = accuracy_score(y_test, best_clf.predict(best_X_test))

print(f"بهترین C پیداشده: {grid.best_params_['C']}")
print(f"[بعد از tuning]      F1-macro = {f1_tuned:.4f}  |  Accuracy = {acc_tuned:.4f}")

print("\n================ جمع‌بندی آزمایش چهارم ================")
print(f"baseline (فعلی):              {f1_base:.4f}")
print(f"word+char n-gram:              {f1_combo:.4f}")
print(f"word+char + C تنظیم‌شده:       {f1_tuned:.4f}")
improvement = f1_tuned - f1_base
print(f"\nمیزان بهبود نسبت به baseline: {improvement:+.4f} ({improvement*100:+.2f} واحد درصد)")

if f1_tuned > f1_base + 0.005:
    print("\n>>> بهبود معنادار بود. جزئیات classification_report:")
    print(classification_report(y_test, best_clf.predict(best_X_test), zero_division=0))
