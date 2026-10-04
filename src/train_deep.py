"""
مرحله ۴: مدل‌های یادگیری عمیق
یک لایه‌ی Embedding قابل‌آموزش + LSTM یا GRU دولایه (Bidirectional) برای مقایسه
"""
import os
import json
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, GRU, Bidirectional, Dense, Dropout, SpatialDropout1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.utils.class_weight import compute_class_weight

from data_loader import load_combined_dataset
from preprocessing import preprocess

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# ثابت کردن seed برای تکرارپذیری نتایج - بدون این، به‌خاطر مقداردهی اولیه
# تصادفی وزن‌ها و dropout، هر بار اجرای دوباره F1-macro تا حدی (~۰.۰۵) نوسان می‌کنه
np.random.seed(42)
tf.random.set_seed(42)

MAX_VOCAB = 15000
MAX_LEN = 40          # حداکثر تعداد کلمه در هر جمله (بقیه یا padding میشن یا truncate)
EMBED_DIM = 100        # کاهش از ۱۲۸ به ۱۰۰ برای کم‌کردن ظرفیت مدل (ضد overfitting)


def prepare_data(use_emopars: bool = False):
    # نکته‌ی مهم (نتیجه‌ی آزمایش): با این‌که هدف اولیه از اضافه کردن EmoPars
    # کمک به overfitting مدل عمیق بود، در عمل چون لیبل‌های EmoPars خودکار
    # (بر اساس رأی‌گیری) و نویزی‌تر از ArmanEmo هستن، نتیجه بدتر شد
    # (GRU از F1=0.415 روی ArmanEmo تنها به 0.288 با داده‌ی ترکیبی رسید).
    # پس پیش‌فرض این تابع الان ArmanEmo تنهاست؛ use_emopars=True برای
    # تکرار آزمایش augmentation در دسترسه.
    train_df, test_df = load_combined_dataset(use_emopars=use_emopars)
    print("در حال پیش‌پردازش متن‌ها...")
    train_df["clean_text"] = train_df["text"].apply(lambda t: preprocess(t, remove_stopwords=False))
    test_df["clean_text"] = test_df["text"].apply(lambda t: preprocess(t, remove_stopwords=False))
    train_df = train_df[train_df["clean_text"].str.len() > 0].reset_index(drop=True)
    test_df = test_df[test_df["clean_text"].str.len() > 0].reset_index(drop=True)

    # نکته‌ی مهم: چون train_df از چسبوندن ArmanEmo و بعدش EmoPars ساخته شده،
    # قبل از هر کاری باید shuffle بشه. وگرنه Keras با validation_split=0.1
    # فقط از انتهای آرایه (که کاملاً از EmoPars هست) validation set می‌سازه؛
    # یعنی یک validation set کاملاً غیرنماینده که early stopping رو خیلی
    # زود و اشتباه فعال می‌کنه.
    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
    return train_df, test_df


def build_model(vocab_size, num_classes, cell_type="lstm"):
    """
    نسخه‌ی ساده‌تر نسبت به تلاش اول: یک لایه Bidirectional RNN به‌جای دوتا،
    به‌همراه SpatialDropout1D (حذف کل بردار یک کلمه به‌جای تک‌تک عددها -
    برای امبدینگ متن مؤثرتر از Dropout معموله)، recurrent_dropout داخل خود
    سلول RNN، و L2 regularization روی لایه‌ی Dense. هدف: کاهش ظرفیت مدل
    و overfitting که در تلاش اول دیده شد.
    """
    RNNLayer = LSTM if cell_type == "lstm" else GRU
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=EMBED_DIM, mask_zero=True),
        SpatialDropout1D(0.3),
        Bidirectional(RNNLayer(48)),
        Dropout(0.4),
        Dense(32, activation="relu", kernel_regularizer=l2(1e-4)),
        Dropout(0.3),
        Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer=Adam(learning_rate=1e-3),   # لرنینگ ریت پیش‌فرض خوب برای Adam
        loss="sparse_categorical_crossentropy",  # چون لیبل‌ها عدد صحیح‌اند نه one-hot
        metrics=["accuracy"],
    )
    return model


def train_and_evaluate(train_df, test_df, cell_types=("lstm", "gru")):
    # --- توکنایز کردن و ساخت دنباله‌های عددی ---
    tokenizer = Tokenizer(num_words=MAX_VOCAB, oov_token="<OOV>")
    tokenizer.fit_on_texts(train_df["clean_text"])

    X_train_seq = tokenizer.texts_to_sequences(train_df["clean_text"])
    X_test_seq = tokenizer.texts_to_sequences(test_df["clean_text"])
    X_train = pad_sequences(X_train_seq, maxlen=MAX_LEN, padding="post", truncating="post")
    X_test = pad_sequences(X_test_seq, maxlen=MAX_LEN, padding="post", truncating="post")

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_df["label"])
    y_test = label_encoder.transform(test_df["label"])
    num_classes = len(label_encoder.classes_)
    vocab_size = min(MAX_VOCAB, len(tokenizer.word_index) + 1)

    class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(enumerate(class_weights))

    results = []
    trained_models = {}

    for cell_type in cell_types:
        print(f"\n================ آموزش مدل {cell_type.upper()} ================")
        model = build_model(vocab_size, num_classes, cell_type=cell_type)
        model.summary()

        early_stop = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

        history = model.fit(
            X_train, y_train,
            validation_split=0.1,
            epochs=12,
            batch_size=64,
            class_weight=class_weight_dict,
            callbacks=[early_stop],
            verbose=2,
        )

        preds_proba = model.predict(X_test, verbose=0)
        preds = np.argmax(preds_proba, axis=1)
        acc = accuracy_score(y_test, preds)
        f1_macro = f1_score(y_test, preds, average="macro")

        print(f"\n--- نتیجه {cell_type.upper()} روی داده تست ---")
        print(f"Accuracy: {acc:.4f} | F1-macro: {f1_macro:.4f}")
        print(classification_report(y_test, preds, target_names=label_encoder.classes_, zero_division=0))

        results.append({"model": f"deep_{cell_type}", "accuracy": acc, "f1_macro": f1_macro,
                         "epochs_trained": len(history.history["loss"])})
        trained_models[cell_type] = model

        # ذخیره‌ی این مدل به‌صورت مستقل (صرف‌نظر از این‌که در نهایت بهترین باشه یا نه)
        model.save(os.path.join(MODELS_DIR, f"deep_{cell_type}.keras"))

    # ذخیره‌ی آرتیفکت‌های مشترک (توکنایزر و لیبل‌انکودر) که بین LSTM و GRU مشترکن
    with open(os.path.join(MODELS_DIR, "tokenizer.pkl"), "wb") as f:
        pickle.dump(tokenizer, f)
    with open(os.path.join(MODELS_DIR, "label_encoder.pkl"), "wb") as f:
        pickle.dump(label_encoder, f)

    # اضافه کردن نتایج این اجرا به فایل نتایج (به‌جای بازنویسی کامل، append می‌کنیم
    # تا بشه LSTM و GRU رو در دو اجرای جدا هم آموزش داد)
    results_path = os.path.join(MODELS_DIR, "deep_results.csv")
    new_results_df = pd.DataFrame(results)
    if os.path.exists(results_path):
        old_df = pd.read_csv(results_path)
        old_df = old_df[~old_df["model"].isin(new_results_df["model"])]  # جایگزینی نتایج تکراری
        combined_df = pd.concat([old_df, new_results_df], ignore_index=True)
    else:
        combined_df = new_results_df
    combined_df.to_csv(results_path, index=False)

    print("\n================ خلاصه‌ی تا این لحظه‌ی مدل‌های عمیق ================")
    print(combined_df.sort_values("f1_macro", ascending=False).to_string(index=False))

    return combined_df


def finalize_best_model():
    """بعد از آموزش LSTM و GRU (در یک یا چند اجرای جدا)، بهترین مدل رو
    بر اساس F1-macro انتخاب و به‌عنوان best_deep_model.keras ذخیره می‌کنه."""
    results_path = os.path.join(MODELS_DIR, "deep_results.csv")
    results_df = pd.read_csv(results_path)
    best_row = results_df.sort_values("f1_macro", ascending=False).iloc[0]
    best_cell = best_row["model"].replace("deep_", "")

    src_path = os.path.join(MODELS_DIR, f"deep_{best_cell}.keras")
    dst_path = os.path.join(MODELS_DIR, "best_deep_model.keras")
    import shutil
    shutil.copyfile(src_path, dst_path)

    with open(os.path.join(MODELS_DIR, "deep_config.json"), "w", encoding="utf-8") as f:
        json.dump({"max_len": MAX_LEN, "best_cell_type": best_cell}, f, ensure_ascii=False)

    print(f"بهترین مدل عمیق نهایی: {best_cell.upper()} (F1-macro={best_row['f1_macro']:.4f}) -> ذخیره شد در best_deep_model.keras")


if __name__ == "__main__":
    import sys
    cell_types = sys.argv[1:] if len(sys.argv) > 1 else ["lstm", "gru"]
    train_df, test_df = prepare_data()
    train_and_evaluate(train_df, test_df, cell_types=cell_types)
