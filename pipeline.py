import os
import warnings
from itertools import combinations

import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore")


# =========================================================
# 1) 경로 설정 (로컬/서버 환경 모두 동작하도록 상대경로 기본)
# =========================================================
SEED = 42
BASE_PATH = os.getenv("BASE_PATH", "train")
SAVE_PATH = os.getenv("SAVE_PATH", "output")
os.makedirs(SAVE_PATH, exist_ok=True)

CUSTOMER_PATH = os.path.join(BASE_PATH, "train_customer_info.csv")
FINANCE_PATH = os.path.join(BASE_PATH, "train_finance_profile.csv")
TARGET_PATH = os.path.join(BASE_PATH, "train_targets.csv")
TXN_PATH = os.path.join(BASE_PATH, "train_transaction_history.csv")


# =========================================================
# 2) 유틸 함수
# =========================================================
def safe_divide(a, b):
    return a / (b.replace(0, np.nan) if isinstance(b, pd.Series) else (b if b != 0 else np.nan))


def calculate_entropy(series: pd.Series) -> float:
    value_counts = series.value_counts(normalize=True)
    if len(value_counts) <= 1:
        return 0.0
    return float(-(value_counts * np.log(value_counts + 1e-12)).sum())


# =========================================================
# 3) 데이터 로드
# =========================================================
customer = pd.read_csv(CUSTOMER_PATH)
finance = pd.read_csv(FINANCE_PATH)
targets = pd.read_csv(TARGET_PATH)
txn = pd.read_csv(TXN_PATH)

print("customer shape :", customer.shape)
print("finance shape  :", finance.shape)
print("targets shape  :", targets.shape)
print("txn shape      :", txn.shape)

customer["join_date"] = pd.to_datetime(customer["join_date"])
txn["trans_date"] = pd.to_datetime(txn["trans_date"])


# =========================================================
# 4) Transaction Feature Engineering
# =========================================================
def build_transaction_features(txn_df: pd.DataFrame) -> pd.DataFrame:
    df = txn_df.copy()
    ref_date = df["trans_date"].max() + pd.Timedelta(days=1)

    df = df.sort_values(["customer_id", "trans_date"]) 
    df["trans_year_month"] = df["trans_date"].dt.to_period("M").astype(str)
    df["trans_dayofweek"] = df["trans_date"].dt.dayofweek
    df["is_weekend"] = (df["trans_dayofweek"] >= 5).astype(int)

    basic_agg = df.groupby("customer_id").agg(
        txn_count=("trans_id", "count"),
        txn_amount_sum=("trans_amount", "sum"),
        txn_amount_mean=("trans_amount", "mean"),
        txn_amount_std=("trans_amount", "std"),
        txn_amount_min=("trans_amount", "min"),
        txn_amount_max=("trans_amount", "max"),
        installment_ratio=("is_installment", "mean"),
        weekend_txn_ratio=("is_weekend", "mean"),
        last_txn_date=("trans_date", "max"),
        first_txn_date=("trans_date", "min"),
        biz_type_nunique=("biz_type", "nunique"),
        item_category_nunique=("item_category", "nunique"),
    ).reset_index()

    basic_agg["recency_days"] = (ref_date - basic_agg["last_txn_date"]).dt.days
    basic_agg["customer_active_days"] = (
        basic_agg["last_txn_date"] - basic_agg["first_txn_date"]
    ).dt.days.clip(lower=1)

    basic_agg["txn_per_active_day"] = safe_divide(basic_agg["txn_count"], basic_agg["customer_active_days"])
    basic_agg["amount_per_active_day"] = safe_divide(basic_agg["txn_amount_sum"], basic_agg["customer_active_days"])
    basic_agg["txn_amount_cv"] = safe_divide(basic_agg["txn_amount_std"], basic_agg["txn_amount_mean"].abs() + 1)

    df["prev_trans_date"] = df.groupby("customer_id")["trans_date"].shift(1)
    df["gap_days"] = (df["trans_date"] - df["prev_trans_date"]).dt.days

    gap_agg = df.groupby("customer_id").agg(
        avg_gap_days=("gap_days", "mean"),
        std_gap_days=("gap_days", "std"),
        max_gap_days=("gap_days", "max"),
    ).reset_index()

    diversity_agg = df.groupby("customer_id").agg(
        biz_entropy=("biz_type", calculate_entropy),
        item_entropy=("item_category", calculate_entropy),
    ).reset_index()

    biz_pivot = pd.crosstab(df["customer_id"], df["biz_type"], normalize="index").add_prefix("biz_ratio_").reset_index()
    cat_pivot = pd.crosstab(df["customer_id"], df["item_category"], normalize="index").add_prefix("cat_ratio_").reset_index()

    monthly = df.groupby(["customer_id", "trans_year_month"]).agg(
        monthly_amount=("trans_amount", "sum"),
        monthly_count=("trans_id", "count"),
    ).reset_index()

    monthly_agg = monthly.groupby("customer_id").agg(
        monthly_amount_mean=("monthly_amount", "mean"),
        monthly_amount_std=("monthly_amount", "std"),
        monthly_count_mean=("monthly_count", "mean"),
        monthly_count_std=("monthly_count", "std"),
        active_months=("trans_year_month", "nunique"),
    ).reset_index()

    max_month = pd.Period(df["trans_date"].max(), freq="M")
    recent_3m = {(max_month - i).strftime("%Y-%m") for i in range(3)}
    monthly["is_recent_3m"] = monthly["trans_year_month"].isin(recent_3m).astype(int)

    recent_agg = monthly.groupby("customer_id").apply(
        lambda x: pd.Series(
            {
                "recent_3m_amount_sum": x.loc[x["is_recent_3m"] == 1, "monthly_amount"].sum(),
                "recent_3m_count_sum": x.loc[x["is_recent_3m"] == 1, "monthly_count"].sum(),
                "all_month_amount_sum": x["monthly_amount"].sum(),
                "all_month_count_sum": x["monthly_count"].sum(),
            }
        )
    ).reset_index()

    recent_agg["recent_3m_amount_ratio"] = safe_divide(
        recent_agg["recent_3m_amount_sum"], recent_agg["all_month_amount_sum"]
    )
    recent_agg["recent_3m_count_ratio"] = safe_divide(
        recent_agg["recent_3m_count_sum"], recent_agg["all_month_count_sum"]
    )

    def calc_amount_trend(g):
        g = g.sort_values("trans_date")
        if len(g) < 2:
            return 0.0
        x = np.arange(len(g))
        y = g["trans_amount"].values
        return np.polyfit(x, y, 1)[0]

    trend_df = df.groupby("customer_id").apply(
        lambda g: pd.Series({"txn_amount_trend": calc_amount_trend(g)})
    ).reset_index()

    feat = basic_agg.merge(gap_agg, on="customer_id", how="left")
    feat = feat.merge(diversity_agg, on="customer_id", how="left")
    feat = feat.merge(biz_pivot, on="customer_id", how="left")
    feat = feat.merge(cat_pivot, on="customer_id", how="left")
    feat = feat.merge(monthly_agg, on="customer_id", how="left")
    feat = feat.merge(recent_agg, on="customer_id", how="left")
    feat = feat.merge(trend_df, on="customer_id", how="left")

    feat = feat.drop(columns=["last_txn_date", "first_txn_date"])
    return feat


txn_features = build_transaction_features(txn)
print("txn_features shape:", txn_features.shape)


# =========================================================
# 5) 메인 데이터셋 구성
# =========================================================
df = customer.merge(finance, on="customer_id", how="inner")
df = df.merge(txn_features, on="customer_id", how="left")
df = df.merge(targets, on="customer_id", how="inner")

print("shape before FE:", df.shape)


# =========================================================
# 6) 메인 Feature Engineering
# =========================================================
base_cat_cols = ["gender", "region_code", "prefer_category", "income_group"]
bigram_cols = []

# 범주 조합 (CatBoost에서 유효, XGB OHE에서도 interaction 근사)
for c1, c2 in combinations(base_cat_cols, 2):
    col = f"BG_{c1}_{c2}"
    df[col] = df[c1].astype(str) + "_" + df[c2].astype(str)
    bigram_cols.append(col)

# 비율/상호작용
ratio_pairs = {
    "deposit_loan_ratio": ("total_deposit_balance", "total_loan_balance"),
    "deposit_per_card": ("total_deposit_balance", "num_active_cards"),
    "loan_per_card": ("total_loan_balance", "num_active_cards"),
    "txn_per_month": ("txn_count", "active_months"),
    "amount_per_txn": ("txn_amount_sum", "txn_count"),
    "recent_vs_total": ("recent_3m_amount_sum", "all_month_amount_sum"),
    "recent_count_ratio2": ("recent_3m_count_sum", "all_month_count_sum"),
    "avg_monthly_amount_per_active_month": ("all_month_amount_sum", "active_months"),
    "avg_monthly_count_per_active_month": ("all_month_count_sum", "active_months"),
    "recent_amount_per_count": ("recent_3m_amount_sum", "recent_3m_count_sum"),
}

for new_col, (a, b) in ratio_pairs.items():
    df[new_col] = safe_divide(df[a], df[b] + 1)

# 금융 강도/구조
interaction_cols = {
    "credit_per_card": df["credit_score"] * df["num_active_cards"],
    "deposit_credit": df["total_deposit_balance"] * df["credit_score"],
    "loan_credit": df["total_loan_balance"] * df["credit_score"],
    "deposit_minus_loan": df["total_deposit_balance"] - df["total_loan_balance"],
    "asset_total": df["total_deposit_balance"] + df["total_loan_balance"],
    "asset_diff": df["total_deposit_balance"] - df["total_loan_balance"],
    "asset_abs_diff": (df["total_deposit_balance"] - df["total_loan_balance"]).abs(),
    "credit_asset": df["credit_score"] * df["total_deposit_balance"],
    "card_asset": df["num_active_cards"] * df["total_deposit_balance"],
    "card_loan_interaction": df["num_active_cards"] * df["total_loan_balance"],
    "recency_amount_interaction": df["recency_days"] * df["txn_amount_mean"],
    "recency_credit_interaction": df["recency_days"] * df["credit_score"],
}

for col, val in interaction_cols.items():
    df[col] = val

reference_date = txn["trans_date"].max() + pd.Timedelta(days=1)
df["days_since_join"] = (reference_date - df["join_date"]).dt.days
df = df.drop(columns=["join_date"])

# 로그 스케일
log_cols = [
    "total_deposit_balance",
    "total_loan_balance",
    "card_cash_service_amt",
    "card_loan_amt",
    "txn_amount_sum",
    "recent_3m_amount_sum",
    "all_month_amount_sum",
]
for c in log_cols:
    df[f"log_{c}"] = np.log1p(df[c].clip(lower=0))

# 카테고리 집중도 (카테고리 구성 바뀌어도 동적으로 동작)
cat_ratio_cols = [c for c in df.columns if c.startswith("cat_ratio_")]
biz_ratio_cols = [c for c in df.columns if c.startswith("biz_ratio_")]
if cat_ratio_cols:
    df["max_cat_ratio"] = df[cat_ratio_cols].max(axis=1)
if biz_ratio_cols:
    df["max_biz_ratio"] = df[biz_ratio_cols].max(axis=1)

# 빈도 인코딩 (타깃 미사용이므로 leakage는 작지만, fold 추정 편향을 줄이기 위해 최소 세트만 유지)
freq_cols = base_cat_cols
for col in freq_cols:
    freq = df[col].value_counts(normalize=True)
    df[col + "_freq"] = df[col].map(freq)

print("shape after FE:", df.shape)
print("Created bigram cols:", bigram_cols)


# =========================================================
# 7) 특성 / 타겟 분리
# =========================================================
target_churn = "target_churn"
target_ltv = "target_ltv"

feature_cols = [c for c in df.columns if c not in ["customer_id", target_churn, target_ltv]]
X = df[feature_cols].copy()
y_churn = df[target_churn].copy()
y_ltv = df[target_ltv].copy()

cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
num_cols = [c for c in X.columns if c not in cat_cols]

print("\nNumber of categorical columns:", len(cat_cols))
print("Number of numeric columns:", len(num_cols))

# churn + ltv 분포를 같이 반영한 stratify 키
ltv_bins = pd.qcut(y_ltv.rank(method="first"), q=5, labels=False)
stratify_key = y_churn.astype(str) + "_" + ltv_bins.astype(str)


# =========================================================
# 8) 5-Fold Stratified CV
# =========================================================
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

# OOF 저장용
oof_cat_churn = np.zeros(len(X))
oof_xgb_churn = np.zeros(len(X))
oof_blend_churn = np.zeros(len(X))

oof_cat_ltv = np.zeros(len(X))
oof_xgb_ltv = np.zeros(len(X))
oof_blend_ltv = np.zeros(len(X))

fold_scores = []

for fold, (train_idx, valid_idx) in enumerate(skf.split(X, stratify_key), 1):
    print("\n" + "=" * 70)
    print(f"FOLD {fold}/{n_splits}")
    print("=" * 70)

    X_train, X_valid = X.iloc[train_idx].copy(), X.iloc[valid_idx].copy()
    y_churn_train, y_churn_valid = y_churn.iloc[train_idx].copy(), y_churn.iloc[valid_idx].copy()
    y_ltv_train, y_ltv_valid = y_ltv.iloc[train_idx].copy(), y_ltv.iloc[valid_idx].copy()

    clip_upper = y_ltv_train.quantile(0.995)
    y_ltv_train_clip = y_ltv_train.clip(upper=clip_upper)
    y_ltv_train_log = np.log1p(y_ltv_train_clip)
    y_ltv_valid_log = np.log1p(y_ltv_valid.clip(lower=0))

    fold_cat_cols = X_train.select_dtypes(include=["object"]).columns.tolist()
    fold_num_cols = [c for c in X_train.columns if c not in fold_cat_cols]

    xgb_preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), fold_num_cols),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("ohe", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                fold_cat_cols,
            ),
        ]
    )

    # -------- Churn: CatBoost --------
    pos_weight = (y_churn_train == 0).sum() / max((y_churn_train == 1).sum(), 1)

    churn_cat_model = CatBoostClassifier(
        iterations=2500,
        learning_rate=0.03,
        depth=6,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=SEED,
        verbose=250,
        auto_class_weights="Balanced",
    )
    churn_cat_model.fit(
        X_train,
        y_churn_train,
        cat_features=fold_cat_cols,
        eval_set=(X_valid, y_churn_valid),
        early_stopping_rounds=250,
        use_best_model=True,
    )

    # -------- Churn: XGBoost --------
    churn_xgb_model = Pipeline(
        steps=[
            ("preprocess", xgb_preprocessor),
            (
                "model",
                xgb.XGBClassifier(
                    n_estimators=2500,
                    learning_rate=0.03,
                    max_depth=6,
                    min_child_weight=3,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    reg_alpha=0.2,
                    reg_lambda=2.5,
                    eval_metric="auc",
                    random_state=SEED,
                    scale_pos_weight=pos_weight,
                    tree_method="hist",
                ),
            ),
        ]
    )
    churn_xgb_model.fit(X_train, y_churn_train)

    cat_churn_pred = churn_cat_model.predict_proba(X_valid)[:, 1]
    xgb_churn_pred = churn_xgb_model.predict_proba(X_valid)[:, 1]

    cat_auc = roc_auc_score(y_churn_valid, cat_churn_pred)
    xgb_auc = roc_auc_score(y_churn_valid, xgb_churn_pred)
    churn_weight_cat = cat_auc / (cat_auc + xgb_auc + 1e-9)
    blend_churn_pred = churn_weight_cat * cat_churn_pred + (1 - churn_weight_cat) * xgb_churn_pred

    # -------- LTV: CatBoost --------
    ltv_cat_model = CatBoostRegressor(
        iterations=3500,
        learning_rate=0.025,
        depth=6,
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=SEED,
        verbose=250,
    )
    ltv_cat_model.fit(
        X_train,
        y_ltv_train_log,
        cat_features=fold_cat_cols,
        eval_set=(X_valid, y_ltv_valid_log),
        early_stopping_rounds=250,
        use_best_model=True,
    )

    # -------- LTV: XGBoost --------
    ltv_xgb_model = Pipeline(
        steps=[
            ("preprocess", xgb_preprocessor),
            (
                "model",
                xgb.XGBRegressor(
                    n_estimators=3500,
                    learning_rate=0.025,
                    max_depth=5,
                    min_child_weight=4,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    reg_alpha=0.5,
                    reg_lambda=2.0,
                    random_state=SEED,
                    objective="reg:squarederror",
                    tree_method="hist",
                ),
            ),
        ]
    )
    ltv_xgb_model.fit(X_train, y_ltv_train_log)

    cat_ltv_pred = np.expm1(ltv_cat_model.predict(X_valid)).clip(0, clip_upper)
    xgb_ltv_pred = np.expm1(ltv_xgb_model.predict(X_valid)).clip(0, clip_upper)

    cat_rmse = np.sqrt(mean_squared_error(y_ltv_valid, cat_ltv_pred))
    xgb_rmse = np.sqrt(mean_squared_error(y_ltv_valid, xgb_ltv_pred))
    ltv_weight_cat = xgb_rmse / (cat_rmse + xgb_rmse + 1e-9)
    blend_ltv_pred = ltv_weight_cat * cat_ltv_pred + (1 - ltv_weight_cat) * xgb_ltv_pred

    # OOF 저장
    oof_cat_churn[valid_idx] = cat_churn_pred
    oof_xgb_churn[valid_idx] = xgb_churn_pred
    oof_blend_churn[valid_idx] = blend_churn_pred

    oof_cat_ltv[valid_idx] = cat_ltv_pred
    oof_xgb_ltv[valid_idx] = xgb_ltv_pred
    oof_blend_ltv[valid_idx] = blend_ltv_pred

    # Fold 평가
    fold_auc = roc_auc_score(y_churn_valid, blend_churn_pred)
    fold_pr_auc = average_precision_score(y_churn_valid, blend_churn_pred)
    fold_rmse = np.sqrt(mean_squared_error(y_ltv_valid, blend_ltv_pred))
    fold_mae = mean_absolute_error(y_ltv_valid, blend_ltv_pred)
    fold_r2 = r2_score(y_ltv_valid, blend_ltv_pred)
    fold_se = np.mean((y_ltv_valid - blend_ltv_pred) ** 2)
    fold_score = 0.5 * fold_auc + 0.5 * (1 / (1 + np.log(fold_se)))

    fold_scores.append(
        {
            "fold": fold,
            "auc": fold_auc,
            "pr_auc": fold_pr_auc,
            "rmse": fold_rmse,
            "mae": fold_mae,
            "r2": fold_r2,
            "se": fold_se,
            "competition_score": fold_score,
            "churn_weight_cat": churn_weight_cat,
            "ltv_weight_cat": ltv_weight_cat,
        }
    )

    print(f"Dynamic Churn Weight (CatBoost): {churn_weight_cat:.4f}")
    print(f"Dynamic LTV Weight (CatBoost)  : {ltv_weight_cat:.4f}")
    print(f"Fold AUC              : {fold_auc:.6f}")
    print(f"Fold PR-AUC           : {fold_pr_auc:.6f}")
    print(f"Fold RMSE             : {fold_rmse:,.4f}")
    print(f"Fold MAE              : {fold_mae:,.4f}")
    print(f"Fold R2               : {fold_r2:.6f}")
    print(f"Fold SE               : {fold_se:,.6f}")
    print(f"Fold CompetitionScore : {fold_score:.6f}")


# =========================================================
# 9) 전체 OOF 평가
# =========================================================
print("\n" + "=" * 70)
print("OOF FINAL METRICS")
print("=" * 70)

oof_auc = roc_auc_score(y_churn, oof_blend_churn)
oof_pr_auc = average_precision_score(y_churn, oof_blend_churn)
oof_rmse = np.sqrt(mean_squared_error(y_ltv, oof_blend_ltv))
oof_mae = mean_absolute_error(y_ltv, oof_blend_ltv)
oof_r2 = r2_score(y_ltv, oof_blend_ltv)
oof_se = np.mean((y_ltv - oof_blend_ltv) ** 2)
oof_score = 0.5 * oof_auc + 0.5 * (1 / (1 + np.log(oof_se)))

print(f"OOF AUC              : {oof_auc:.6f}")
print(f"OOF PR-AUC           : {oof_pr_auc:.6f}")
print(f"OOF RMSE             : {oof_rmse:,.4f}")
print(f"OOF MAE              : {oof_mae:,.4f}")
print(f"OOF R2               : {oof_r2:.6f}")
print(f"OOF SE               : {oof_se:,.6f}")
print(f"OOF CompetitionScore : {oof_score:.6f}")


# =========================================================
# 10) 결과 저장
# =========================================================
fold_result_df = pd.DataFrame(fold_scores)
fold_result_path = os.path.join(SAVE_PATH, "cv_fold_scores.csv")
fold_result_df.to_csv(fold_result_path, index=False, encoding="utf-8-sig")
print(f"\nSaved fold scores: {fold_result_path}")

oof_result = X.copy()
oof_result["actual_churn"] = y_churn.values
oof_result["cat_churn_pred"] = oof_cat_churn
oof_result["xgb_churn_pred"] = oof_xgb_churn
oof_result["blend_churn_pred"] = oof_blend_churn
oof_result["actual_ltv"] = y_ltv.values
oof_result["cat_ltv_pred"] = oof_cat_ltv
oof_result["xgb_ltv_pred"] = oof_xgb_ltv
oof_result["blend_ltv_pred"] = oof_blend_ltv

oof_result_path = os.path.join(SAVE_PATH, "oof_predictions_blend.csv")
oof_result.to_csv(oof_result_path, index=False, encoding="utf-8-sig")
print(f"Saved OOF predictions: {oof_result_path}")
print("\nDone.")
