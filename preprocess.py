# -*- coding: utf-8 -*-
"""
数据预处理：读取 Last.fm HetRec 2011 原始数据，清洗、构建稀疏交互矩阵、
按用户划分训练/验证/测试集，并抽出用户/艺术家的 Top-20 标签。

输出：models/preprocessed_data.pkl
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

# ========== 1. 加载数据 ==========
# 用户-艺术家交互数据
user_artists = pd.read_csv(DATA_DIR / "user_artists.dat", sep="\t", header=0)
user_artists.columns = ["userID", "artistID", "weight"]

# 标签映射表
tags = pd.read_csv(DATA_DIR / "tags.dat", sep="\t", header=0, encoding="latin-1")
tags.columns = ["tagID", "tagValue"]

# 用户打标签数据
user_tagged = pd.read_csv(DATA_DIR / "user_taggedartists.dat", sep="\t", header=0)
user_tagged.columns = ["userID", "artistID", "tagID", "day", "month", "year"]

# 用户好友关系
user_friends = pd.read_csv(DATA_DIR / "user_friends.dat", sep="\t", header=0)
user_friends.columns = ["userID", "friendID"]

print(f"原始交互: {len(user_artists)} 条, 用户 {user_artists['userID'].nunique()}, "
      f"艺术家 {user_artists['artistID'].nunique()}")

# ========== 2. 数据清洗 ==========
# 过滤冷启动用户（收听记录>=10）
user_counts = user_artists.groupby("userID").size()
active_users = user_counts[user_counts >= 10].index
user_artists = user_artists[user_artists["userID"].isin(active_users)]

# 过滤冷启动艺术家（被收听>=5）
artist_counts = user_artists.groupby("artistID").size()
popular_artists = artist_counts[artist_counts >= 5].index
user_artists = user_artists[user_artists["artistID"].isin(popular_artists)]

print(f"清洗后: {user_artists['userID'].nunique()} 用户, {user_artists['artistID'].nunique()} 艺术家")

# ========== 3. 构建用户/艺术家映射 ==========
user_ids = user_artists["userID"].unique()
artist_ids = user_artists["artistID"].unique()

user2idx = {uid: i for i, uid in enumerate(user_ids)}
artist2idx = {aid: i for i, aid in enumerate(artist_ids)}
idx2user = {i: uid for uid, i in user2idx.items()}
idx2artist = {i: aid for aid, i in artist2idx.items()}

user_artists["user_idx"] = user_artists["userID"].map(user2idx)
user_artists["artist_idx"] = user_artists["artistID"].map(artist2idx)

num_users = len(user2idx)
num_artists = len(artist2idx)

# ========== 4. 构建稀疏交互矩阵 ==========
interaction_matrix = csr_matrix(
    (user_artists["weight"],
     (user_artists["user_idx"], user_artists["artist_idx"])),
    shape=(num_users, num_artists)
)

print(f"交互矩阵形状: {interaction_matrix.shape}")
print(f"稀疏度: {interaction_matrix.nnz / (num_users * num_artists):.4%}")

# ========== 5. 划分训练/验证/测试集 ==========
# 对每个用户，随机留出交互记录（8:1:1）
train_indices = []
val_indices = []
test_indices = []

for user_idx in range(num_users):
    # 获取该用户的所有交互
    _, artist_indices = interaction_matrix[user_idx].nonzero()
    if len(artist_indices) < 3:
        continue  # 交互太少跳过

    # 随机打乱并切分
    np.random.seed(42)
    np.random.shuffle(artist_indices)
    n = len(artist_indices)
    train_idx = artist_indices[:int(n * 0.8)]
    val_idx = artist_indices[int(n * 0.8):int(n * 0.9)]
    test_idx = artist_indices[int(n * 0.9):]

    for aid in train_idx:
        train_indices.append((user_idx, aid))
    for aid in val_idx:
        val_indices.append((user_idx, aid))
    for aid in test_idx:
        test_indices.append((user_idx, aid))

# 构建训练矩阵（只保留训练集中的交互）
train_matrix = csr_matrix(
    (np.ones(len(train_indices)),
     (np.array([t[0] for t in train_indices]),
      np.array([t[1] for t in train_indices]))),
    shape=(num_users, num_artists)
)

print(f"训练交互数: {len(train_indices)}")
print(f"验证交互数: {len(val_indices)}")
print(f"测试交互数: {len(test_indices)}")

# ========== 6. 构建标签特征 ==========
# 合并标签名
user_tagged = user_tagged.merge(tags, on="tagID", how="left")

# 构建用户标签偏好向量（每个用户最常用的Top-20标签）
user_tag_counts = user_tagged.groupby(["userID", "tagValue"]).size().reset_index(name="count")
user_tag_counts = user_tag_counts.sort_values(["userID", "count"], ascending=[True, False])

user_top_tags = {}
for uid, group in user_tag_counts.groupby("userID"):
    top_tags = group.head(20)["tagValue"].tolist()
    if uid in user2idx:
        user_top_tags[user2idx[uid]] = top_tags

# 构建艺术家标签向量
artist_tag_counts = user_tagged.groupby(["artistID", "tagValue"]).size().reset_index(name="count")
artist_tag_counts = artist_tag_counts.sort_values(["artistID", "count"], ascending=[True, False])

artist_top_tags = {}
for aid, group in artist_tag_counts.groupby("artistID"):
    top_tags = group.head(20)["tagValue"].tolist()
    if aid in artist2idx:
        artist_top_tags[artist2idx[aid]] = top_tags

print(f"带标签的用户: {len(user_top_tags)}, 带标签的艺术家: {len(artist_top_tags)}")

# 保存预处理结果
MODEL_DIR.mkdir(parents=True, exist_ok=True)
with open(MODEL_DIR / "preprocessed_data.pkl", "wb") as f:
    pickle.dump({
        "train_matrix": train_matrix,
        "interaction_matrix": interaction_matrix,
        "user2idx": user2idx,
        "artist2idx": artist2idx,
        "idx2user": idx2user,
        "idx2artist": idx2artist,
        "train_indices": train_indices,
        "val_indices": val_indices,
        "test_indices": test_indices,
        "user_top_tags": user_top_tags,
        "artist_top_tags": artist_top_tags,
        "user_friends": user_friends,
        "num_users": num_users,
        "num_artists": num_artists
    }, f)

print("预处理完成，数据已保存到 models/preprocessed_data.pkl")
