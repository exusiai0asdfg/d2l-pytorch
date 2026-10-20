
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold

# ========= 1. 基本设置 =========

ROOT = Path(__file__).resolve().parent

train_data = pd.read_csv(ROOT / "train.csv")
test_data = pd.read_csv(ROOT / "test.csv")

n_splits = 5
seed = 42

# ========= 2. 数据预处理 =========

# 保留原始特征，不使用 One-hot
train_x = train_data.drop(columns=["Id", "SalePrice"]).copy()
test_x = test_data.drop(columns=["Id"]).copy()

# 预测 log(1 + 房价)
train_y = np.log1p(
    train_data["SalePrice"].to_numpy(dtype=np.float64)
)

# 自动寻找类别型列
combined = pd.concat(
    [train_x, test_x],
    ignore_index=True
)

cat_cols = combined.select_dtypes(
    include=["object", "string", "category"]
).columns.tolist()

# MSSubClass 虽然是数字，实际上代表建筑类型
if "MSSubClass" not in cat_cols:
    cat_cols.append("MSSubClass")

# CatBoost 的类别型特征需要字符串或整数类别
# 类别缺失值统一用特殊字符串表示
for df in (train_x, test_x):
    for col in cat_cols:
        df[col] = (
            df[col]
            .astype("string")
            .fillna("__MISSING__")
            .astype(str)
        )

print("训练集:", train_x.shape)
print("测试集:", test_x.shape)
print("类别型特征数:", len(cat_cols))

# ========= 3. 五折交叉验证 =========

kf = KFold(
    n_splits=n_splits,
    shuffle=True,
    random_state=seed
)

# 保存每一个样本在其验证折上的预测
oof_pred = np.zeros(len(train_x))

# 保存 5 个模型对测试集预测的平均值
test_pred = np.zeros(len(test_x))

# 累计各模型的特征重要性
feature_importance = np.zeros(train_x.shape[1])

for fold, (train_idx, val_idx) in enumerate(
    kf.split(train_x), start=1
):

    x_tr = train_x.iloc[train_idx]
    x_val = train_x.iloc[val_idx]

    y_tr = train_y[train_idx]
    y_val = train_y[val_idx]

    # 每一折创建新模型
    model = CatBoostRegressor(
        iterations=3000,
        depth=5,
        learning_rate=0.03,
        loss_function="RMSE",
        eval_metric="RMSE",
        l2_leaf_reg=4,
        random_strength=0.5,
        bagging_temperature=0.5,
        random_seed=seed + fold,
        thread_count=4,
        allow_writing_files=False,
        verbose=False
    )

    model.fit(
        x_tr,
        y_tr,
        cat_features=cat_cols,
        eval_set=(x_val, y_val),
        early_stopping_rounds=150,
        use_best_model=True,
        verbose=False
    )

    # 当前折验证集预测
    val_pred = model.predict(x_val)
    oof_pred[val_idx] = val_pred

    # Kaggle 测试集预测，五折模型取平均
    test_pred += model.predict(test_x) / n_splits

    # 特征重要性
    feature_importance += (
        model.get_feature_importance() / n_splits
    )

    fold_score = np.sqrt(
        np.mean((val_pred - y_val) ** 2)
    )

    print(
        f"Fold {fold}: "
        f"LogRMSE = {fold_score:.5f}, "
        f"Best Trees = {model.best_iteration_ + 1}"
    )

# ========= 4. 总体交叉验证成绩 =========

cv_logrmse = np.sqrt(
    np.mean((oof_pred - train_y) ** 2)
)

print("\n===== Cross Validation =====")
print(f"CatBoost CV LogRMSE: {cv_logrmse:.5f}")

# ========= 5. 创建 Kaggle 提交文件 =========

# 从 log(1 + price) 转换回真实房价
prices = np.expm1(test_pred)
prices = np.maximum(prices, 0)

submission = pd.DataFrame({
    "Id": test_data["Id"].to_numpy(),
    "SalePrice": prices
})

assert submission.shape == (len(test_data), 2)
assert np.isfinite(submission["SalePrice"]).all()

output_path = ROOT / "submission_catboost.csv"

submission.to_csv(output_path, index=False)

print("\n已生成:", output_path)
print(submission.head())

# ========= 6. 输出重要特征 =========

importance = pd.Series(
    feature_importance,
    index=train_x.columns
).sort_values(ascending=False)

print("\n===== Top 15 Important Features =====")
print(importance.head(15))
