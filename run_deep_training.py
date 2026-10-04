import sys
sys.path.append('src')
from train_deep import prepare_data, train_and_evaluate, finalize_best_model
train_df, test_df = prepare_data()
train_and_evaluate(train_df, test_df)
finalize_best_model()