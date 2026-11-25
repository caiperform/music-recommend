# -*- coding: utf-8 -*-
"""
多路召回：标签内容召回 + ALS 协同过滤召回 + 热度兜底召回，合并去重后作为排序模型的候选集。

运行：
    python recall.py            # 打印召回统计并画出三路召回韦恩图
"""
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 无界面环境下也能出图
import matplotlib.pyplot as plt
import numpy as np
from implicit.als import AlternatingLeastSquares

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"

# 加载预处理数据
with open(MODEL_DIR / "preprocessed_data.pkl", "rb") as f:
    data = pickle.load(f)

train_matrix = data["train_matrix"]
user2idx = data["user2idx"]
artist2idx = data["artist2idx"]
idx2artist = data["idx2artist"]
idx2user = data["idx2user"]
user_top_tags = data["user_top_tags"]
artist_top_tags = data["artist_top_tags"]
num_users = data["num_users"]
num_artists = data["num_artists"]
user_friends = data["user_friends"]


# ========== 召回通路A：标签内容召回 ==========
def jaccard_similarity(set1, set2):
    """计算两个集合的Jaccard相似度"""
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


def tag_based_recall(user_idx, top_k=200, filter_interacted=False):
    """
    基于标签的召回
    filter_interacted: 是否过滤已交互的艺术家
    """
    user_tags = set(user_top_tags.get(user_idx, []))
    if not user_tags:
        return []

    scores = []
    for artist_idx in range(num_artists):
        artist_tags = set(artist_top_tags.get(artist_idx, []))
        sim = jaccard_similarity(user_tags, artist_tags)
        if sim > 0:
            scores.append((artist_idx, sim))

    # 排序取top_k
    scores.sort(key=lambda x: x[1], reverse=True)
    candidates = [artist_idx for artist_idx, _ in scores[:top_k]]

    # 可选：过滤已交互
    if filter_interacted:
        interacted = set(train_matrix[user_idx].nonzero()[1])
        candidates = [aid for aid in candidates if aid not in interacted]

    return candidates


# ========== 召回通路B：ALS协同过滤召回 ==========
print("训练ALS模型...")
als_model = AlternatingLeastSquares(factors=50, regularization=0.1, iterations=30)
als_model.fit(train_matrix.T.tocsr())
print("ALS模型训练完成。")


def als_recall(user_idx, top_k=200, filter_interacted=True):
    """
    基于ALS的召回
    filter_interacted: ALS的recommend方法自带过滤，这里保留参数保持一致性
    """
    try:
        recommended = als_model.recommend(
            user_idx,
            train_matrix[user_idx],
            N=top_k,
            filter_already_liked_items=filter_interacted
        )
        return list(recommended[0])
    except Exception:
        return []


# ========== 召回通路C：热度兜底召回 ==========
artist_popularity = np.array(train_matrix.sum(axis=0)).flatten()
popular_artists = np.argsort(artist_popularity)[::-1][:200].tolist()


def popularity_recall(top_k=100, filter_interacted=False, user_idx=None):
    """
    热度兜底召回
    filter_interacted: 是否过滤已交互的艺术家
    user_idx: 如果过滤，需要知道哪个用户
    """
    candidates = popular_artists[:top_k]

    if filter_interacted and user_idx is not None:
        interacted = set(train_matrix[user_idx].nonzero()[1])
        candidates = [aid for aid in candidates if aid not in interacted]

    return candidates


# ========== 合并多路召回 ==========
def multi_channel_recall(user_idx, tag_k=200, als_k=200, pop_k=100, filter_interacted=True):
    """
    多路召回合并、去重
    filter_interacted: 是否过滤已交互的艺术家
    """
    # 各路召回
    tag_candidates = tag_based_recall(user_idx, top_k=tag_k, filter_interacted=filter_interacted)
    als_candidates = als_recall(user_idx, top_k=als_k)
    pop_candidates = popularity_recall(top_k=pop_k, filter_interacted=filter_interacted, user_idx=user_idx)

    # 合并去重
    all_candidates = list(set(tag_candidates + als_candidates + pop_candidates))

    # 如果各路召回没有单独过滤，这里统一过滤
    if not filter_interacted:
        interacted = set(train_matrix[user_idx].nonzero()[1])
        all_candidates = [aid for aid in all_candidates if aid not in interacted]

    return all_candidates


# ========== 召回效果分析函数 ==========
def analyze_recall(user_idx, tag_k=200, als_k=200, pop_k=100):
    """
    分析召回效果，返回原始和过滤后的数据
    """
    # 获取原始召回（不过滤）
    tag_raw = set(tag_based_recall(user_idx, top_k=tag_k, filter_interacted=False))
    als_raw = set(als_recall(user_idx, top_k=als_k))
    pop_raw = set(popularity_recall(top_k=pop_k, filter_interacted=False))

    # 获取已交互
    interacted = set(train_matrix[user_idx].nonzero()[1])

    # 过滤后
    tag_filtered = tag_raw - interacted
    als_filtered = als_raw - interacted
    pop_filtered = pop_raw - interacted

    return {
        "raw": {
            "tag": tag_raw,
            "als": als_raw,
            "pop": pop_raw,
            "interacted": interacted
        },
        "filtered": {
            "tag": tag_filtered,
            "als": als_filtered,
            "pop": pop_filtered
        }
    }


def print_recall_stats(user_idx, tag_k=200, als_k=200, pop_k=100):
    """打印详细的召回统计"""
    data = analyze_recall(user_idx, tag_k, als_k, pop_k)

    raw = data["raw"]
    filtered = data["filtered"]
    interacted = raw["interacted"]

    print(f"\n{'=' * 50}")
    print(f"用户 {idx2user[user_idx]} 召回效果分析")
    print(f"{'=' * 50}")

    # 各路召回统计
    print("\n【各路召回统计】")
    print(f"  标签召回: {len(raw['tag'])} 个")
    print(f"  ALS召回: {len(raw['als'])} 个")
    print(f"  热度召回: {len(raw['pop'])} 个")
    print(f"  已交互艺术家: {len(interacted)} 个")

    # 过滤统计
    print("\n【过滤统计】")
    print(f"  标签召回过滤后: {len(filtered['tag'])} 个 (过滤了 {len(raw['tag']) - len(filtered['tag'])} 个)")
    print(f"  ALS召回过滤后: {len(filtered['als'])} 个 (过滤了 {len(raw['als']) - len(filtered['als'])} 个)")
    print(f"  热度召回过滤后: {len(filtered['pop'])} 个 (过滤了 {len(raw['pop']) - len(filtered['pop'])} 个)")

    # 韦恩图数据（过滤后）
    tag_f, als_f, pop_f = filtered["tag"], filtered["als"], filtered["pop"]

    n_a_only = len(tag_f - als_f - pop_f)
    n_b_only = len(als_f - tag_f - pop_f)
    n_c_only = len(pop_f - tag_f - als_f)
    n_ab_only = len((tag_f & als_f) - pop_f)
    n_ac_only = len((tag_f & pop_f) - als_f)
    n_bc_only = len((als_f & pop_f) - tag_f)
    n_abc = len(tag_f & als_f & pop_f)

    print("\n【韦恩图数据（过滤后）】")
    print(f"  纯标签独有: {n_a_only}")
    print(f"  纯ALS独有: {n_b_only}")
    print(f"  纯热度独有: {n_c_only}")
    print(f"  标签∩ALS (不含热度): {n_ab_only}")
    print(f"  标签∩热度 (不含ALS): {n_ac_only}")
    print(f"  ALS∩热度 (不含标签): {n_bc_only}")
    print(f"  三路交集: {n_abc}")
    print(f"  并集大小: {n_a_only + n_b_only + n_c_only + n_ab_only + n_ac_only + n_bc_only + n_abc}")

    # 合并后的候选数
    candidates = multi_channel_recall(user_idx, tag_k, als_k, pop_k, filter_interacted=True)
    print("\n【最终召回结果】")
    print(f"  多路合并去重后候选数: {len(candidates)}")

    return {
        "raw": raw,
        "filtered": filtered,
        "venn_data": (n_a_only, n_b_only, n_c_only, n_ab_only, n_ac_only, n_bc_only, n_abc)
    }


def draw_venn(venn_data, user_id, out_path):
    """把三路召回的候选集重叠情况画成韦恩图"""
    from matplotlib_venn import venn3

    n_a_only, n_b_only, n_c_only, n_ab_only, n_ac_only, n_bc_only, n_abc = venn_data

    plt.figure(figsize=(10, 8))
    venn3(subsets=(n_a_only, n_b_only, n_ab_only, n_c_only, n_ac_only, n_bc_only, n_abc),
          set_labels=("标签召回", "ALS召回", "热度召回"))
    plt.title(f"三路召回候选集韦恩图 (用户{user_id} - 已过滤)")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"韦恩图已保存: {out_path}")


# ========== 测试代码 ==========
if __name__ == "__main__":
    # 测试单个用户
    test_user_idx = 2
    stats = print_recall_stats(test_user_idx)
    draw_venn(stats["venn_data"], idx2user[test_user_idx], OUTPUT_DIR / "venn3.png")

    # 测试多个用户
    print(f"\n{'=' * 50}")
    print("多用户召回统计")
    print(f"{'=' * 50}")
    for user_idx in [0, 1, 2, 3, 4]:
        candidates = multi_channel_recall(user_idx, filter_interacted=True)
        print(f"用户 {idx2user[user_idx]}: {len(candidates)} 个候选")
