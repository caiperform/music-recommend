# -*- coding: utf-8 -*-
"""
LightGBM LambdaRank 排序模型：对多路召回的候选集做特征工程并排序，输出 NDCG 指标。

运行（先执行 preprocess.py 生成 models/preprocessed_data.pkl）：
    python train_ranker.py
"""
import json
import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score
from collections import defaultdict

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

# 加载数据
with open(MODEL_DIR / "preprocessed_data.pkl", "rb") as f:
    data = pickle.load(f)

train_matrix = data["train_matrix"]
interaction_matrix = data["interaction_matrix"]
idx2user = data["idx2user"]
idx2artist = data["idx2artist"]
train_indices = data["train_indices"]
val_indices = data["val_indices"]
test_indices = data["test_indices"]
user_top_tags = data["user_top_tags"]
artist_top_tags = data["artist_top_tags"]


# ========== 1. 构建训练样本 ==========
def build_samples(interaction_pairs, num_negatives=3):
    """
    为每个正样本采样负样本，构建训练集
    interaction_pairs: [(user_idx, artist_idx), ...]
    """
    # 用户已交互的艺术家集合
    user_pos_items = defaultdict(set)
    for u, a in interaction_pairs:
        user_pos_items[u].add(a)

    samples = []
    all_artist_set = set(range(len(idx2artist)))

    for u, a in interaction_pairs:
        # 正样本
        samples.append((u, a, 1))

        # 负样本：随机采样
        neg_candidates = list(all_artist_set - user_pos_items[u])
        if len(neg_candidates) >= num_negatives:
            neg_samples = np.random.choice(neg_candidates, num_negatives, replace=False)
        else:
            neg_samples = neg_candidates

        for neg_a in neg_samples:
            samples.append((u, neg_a, 0))

    return pd.DataFrame(samples, columns=["user_idx", "artist_idx", "label"])


print("构建训练样本...")
train_df = build_samples(train_indices, num_negatives=3)
val_df = build_samples(val_indices, num_negatives=3)
test_df = build_samples(test_indices, num_negatives=3)

print(f"训练样本数: {len(train_df)} (正样本: {train_df['label'].sum()})")
print(f"验证样本数: {len(val_df)}")
print(f"测试样本数: {len(test_df)}")

# ========== 2. 特征工程 ==========
# 全局热度
artist_global_popularity = np.array(train_matrix.sum(axis=0)).flatten()
user_total_plays = np.array(train_matrix.sum(axis=1)).flatten()


def add_features(df):
    """为样本添加特征"""

    # 确保索引列为整数，防止稀疏矩阵索引报错 (Inexact indices)
    df["user_idx"] = df["user_idx"].astype(int)
    df["artist_idx"] = df["artist_idx"].astype(int)

    # ================= 基础特征 =================
    # 1. 用户总播放量
    df["user_total_plays"] = df["user_idx"].apply(lambda x: user_total_plays[x])

    # 2. 艺术家全局热度
    df["artist_popularity"] = df["artist_idx"].apply(lambda x: artist_global_popularity[x])
    df["artist_popularity_log"] = np.log1p(df["artist_popularity"])

    # 3. 标签 Jaccard 相似度
    def tag_sim(row):
        u_tags = set(user_top_tags.get(row["user_idx"], []))
        a_tags = set(artist_top_tags.get(row["artist_idx"], []))
        if not u_tags or not a_tags:
            return 0.0
        return len(u_tags & a_tags) / len(u_tags | a_tags)

    df["tag_jaccard"] = df.apply(tag_sim, axis=1)

    # ================= 扩展特征 =================

    # 1. 用户活跃度分桶 (5个等级)
    df["user_active_bucket"] = pd.cut(df["user_total_plays"], bins=5, labels=False).fillna(0).astype(int)

    # 2. 艺术家被多少不同用户听过 (区分大众/小众)
    artist_user_count = np.array((train_matrix > 0).sum(axis=0)).flatten()
    df["artist_user_count_log"] = np.log1p(df["artist_idx"].apply(lambda x: artist_user_count[x]))

    # 3. 用户对该艺术家的历史交互权重 (训练矩阵中的实际值)
    df["user_artist_weight"] = df.apply(
        lambda row: train_matrix[int(row["user_idx"]), int(row["artist_idx"])],
        axis=1
    )

    # 4. 艺术家平均被同一用户播放的次数 (深度偏好 vs 浅尝辄止)
    artist_avg_weight = np.zeros(len(idx2artist))
    for aid in range(len(idx2artist)):
        col_data = train_matrix[:, aid].data
        if len(col_data) > 0:
            artist_avg_weight[aid] = col_data.mean()
        else:
            artist_avg_weight[aid] = 0.0
    df["artist_avg_weight"] = df["artist_idx"].apply(lambda x: artist_avg_weight[x])

    # 5. 艺术家流行度的排名分位 (归一化到 0-1)
    popularity_ranks = np.argsort(np.argsort(artist_global_popularity))
    max_rank = len(artist_global_popularity) - 1
    popularity_percentiles = popularity_ranks / max_rank if max_rank > 0 else 0
    df["artist_popularity_percentile"] = df["artist_idx"].apply(lambda x: popularity_percentiles[x])

    return df


print("添加特征...")
train_df = add_features(train_df)
val_df = add_features(val_df)
test_df = add_features(test_df)

# 特征列
feature_cols = ["user_total_plays", "artist_popularity_log", "tag_jaccard",
                "user_active_bucket", "artist_user_count_log", "user_artist_weight",
                "artist_popularity_percentile"]

# ========== 3. 训练LightGBM模型 ==========
print("训练LightGBM Ranker...")
train_dataset = lgb.Dataset(
    train_df[feature_cols],
    label=train_df["label"],
    group=train_df.groupby("user_idx").size().values  # 每个用户的样本数
)

val_dataset = lgb.Dataset(
    val_df[feature_cols],
    label=val_df["label"],
    group=val_df.groupby("user_idx").size().values,
    reference=train_dataset
)

params = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "ndcg_eval_at": [10, 20],
    "boosting_type": "gbdt",
    "num_leaves": 64,
    "learning_rate": 0.05,
    "verbose": -1
}

ranker = lgb.train(
    params,
    train_dataset,
    valid_sets=[val_dataset],
    num_boost_round=200,
    callbacks=[lgb.early_stopping(stopping_rounds=20)]
)

print("模型训练完成！")

# ========== 4. 模型评估 ==========
# 预测
test_preds = ranker.predict(test_df[feature_cols])
test_df["pred"] = test_preds


# 按用户分组计算NDCG@K
def calculate_ndcg_at_k(df, k=10):
    ndcg_scores = []
    for user_idx, group in df.groupby("user_idx"):
        if len(group) < k:
            continue
        true_relevance = group["label"].values
        pred_scores = group["pred"].values
        # 排序并取top-k
        sorted_indices = np.argsort(pred_scores)[::-1][:k]
        if len(sorted_indices) == k:
            try:
                ndcg = ndcg_score([true_relevance[sorted_indices]], [pred_scores[sorted_indices]])
                ndcg_scores.append(ndcg)
            except Exception:
                pass
    return np.mean(ndcg_scores)


test_ndcg10 = calculate_ndcg_at_k(test_df, k=10)
test_ndcg20 = calculate_ndcg_at_k(test_df, k=20)
print(f"测试集 NDCG@10: {test_ndcg10:.4f}")
print(f"测试集 NDCG@20: {test_ndcg20:.4f}")

# 特征重要性
importance = ranker.feature_importance(importance_type="gain")
feature_importance = pd.DataFrame({
    "feature": feature_cols,
    "importance": importance
}).sort_values("importance", ascending=False)
print("\n特征重要性:")
print(feature_importance)

ranker.save_model(str(MODEL_DIR / "ranker_model.txt"))
print("模型已保存到 models/ranker_model.txt")

# 保存特征列名
with open(MODEL_DIR / "feature_cols.json", "w", encoding="utf-8") as f:
    json.dump(feature_cols, f, ensure_ascii=False)
print("特征列名已保存到 models/feature_cols.json")
