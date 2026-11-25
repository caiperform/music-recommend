import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from collections import Counter


class TagBasedArtistRecommender:
    """
    基于标签的音乐艺术家推荐系统
    用户输入自定义标签，系统推荐匹配的艺术家
    """

    def __init__(self):
        """初始化系统，加载艺术家标签数据"""
        self.artists_df = None
        self.tag_matrix = None
        self.vectorizer = None
        self._load_data()
        self._build_tag_matrix()

    def _load_data(self):
        """加载艺术家和标签数据"""
        # 内置示例数据（包含多种流派的知名艺术家）
        # 实际使用时，可以替换为 Last.fm 数据集
        sample_data = [
            # 摇滚/另类
            {"artist": "Radiohead", "tags": "alternative rock experimental art rock electronic british melancholic"},
            {"artist": "Nirvana", "tags": "grunge rock alternative 90s seattle heavy"},
            {"artist": "Arctic Monkeys", "tags": "indie rock british alternative garage rock 2000s"},
            {"artist": "The Beatles", "tags": "classic rock pop rock british 60s legendary psychedelic"},
            {"artist": "Pink Floyd", "tags": "progressive rock psychedelic classic rock space atmospheric"},
            {"artist": "Led Zeppelin", "tags": "classic rock hard rock blues rock 70s heavy"},
            {"artist": "Queen", "tags": "classic rock arena rock pop rock 70s 80s theatrical"},
            {"artist": "Foo Fighters", "tags": "alternative rock post-grunge hard rock 90s 2000s"},
            {"artist": "Muse", "tags": "alternative rock progressive space rock british epic"},
            {"artist": "Tame Impala", "tags": "psychedelic rock indie electronic neo-psychedelia dreamy"},

            # 电子
            {"artist": "Daft Punk", "tags": "electronic house french disco dance funk"},
            {"artist": "Aphex Twin", "tags": "electronic ambient experimental idm techno"},
            {"artist": "Deadmau5", "tags": "electronic house progressive house electro dance"},
            {"artist": "Kraftwerk", "tags": "electronic krautrock synthpop german pioneering"},
            {"artist": "Skrillex", "tags": "electronic dubstep edm bass heavy"},
            {"artist": "Calvin Harris", "tags": "electronic edm pop dance house mainstream"},

            # 流行
            {"artist": "Taylor Swift", "tags": "pop country pop singer-songwriter female mainstream"},
            {"artist": "Michael Jackson", "tags": "pop king of pop r&b dance 80s legendary"},
            {"artist": "Billie Eilish", "tags": "pop electropop alternative female dark moody"},
            {"artist": "Adele", "tags": "pop soul female ballad british emotional"},
            {"artist": "Lorde", "tags": "pop indie pop female alternative new zealand"},
            {"artist": "The Weeknd", "tags": "pop r&b alternative r&b dark synthwave"},

            # 嘻哈/说唱
            {"artist": "Kendrick Lamar", "tags": "hip-hop rap conscious west coast poetic"},
            {"artist": "Kanye West", "tags": "hip-hop rap experimental producer innovative"},
            {"artist": "Drake", "tags": "hip-hop rap pop rap mainstream toronto"},
            {"artist": "Eminem", "tags": "hip-hop rap hardcore detroit lyrical"},
            {"artist": "Tyler the Creator", "tags": "hip-hop rap alternative experimental creative"},
            {"artist": "J. Cole", "tags": "hip-hop rap conscious storytelling soulful"},

            # R&B/灵魂
            {"artist": "Frank Ocean", "tags": "r&b alternative r&b soul experimental emotional"},
            {"artist": "SZA", "tags": "r&b neo-soul alternative female chill"},
            {"artist": "Erykah Badu", "tags": "neo-soul r&b soul female jazz-influenced"},

            # 爵士
            {"artist": "Miles Davis", "tags": "jazz trumpet bebop cool jazz fusion legendary"},
            {"artist": "John Coltrane", "tags": "jazz saxophone spiritual jazz avant-garde legendary"},
            {"artist": "Kamasi Washington", "tags": "jazz contemporary jazz saxophone spiritual epic"},

            # 古典/器乐
            {"artist": "Ludovico Einaudi", "tags": "classical contemporary classical piano minimalist emotional"},
            {"artist": "Max Richter", "tags": "classical contemporary classical ambient minimalist cinematic"},
            {"artist": "Ólafur Arnalds", "tags": "classical neo-classical ambient icelandic piano strings"},

            # 金属
            {"artist": "Metallica", "tags": "metal thrash metal heavy metal hard rock legendary"},
            {"artist": "Tool", "tags": "metal progressive metal alternative metal complex artistic"},
            {"artist": "Black Sabbath", "tags": "metal heavy metal doom metal classic 70s legendary"},

            # 民谣/独立
            {"artist": "Bon Iver", "tags": "indie folk folktronica atmospheric emotional experimental"},
            {"artist": "Fleet Foxes", "tags": "indie folk baroque pop harmonies acoustic"},
            {"artist": "Sufjan Stevens", "tags": "indie folk singer-songwriter experimental orchestral"},
            {"artist": "Phoebe Bridgers", "tags": "indie folk singer-songwriter female sad emotional"},

            # 拉丁/世界音乐
            {"artist": "Bad Bunny", "tags": "latin reggaeton trap latin urban puerto rico"},
            {"artist": "Rosalía", "tags": "latin flamenco pop experimental spanish female"},
            {"artist": "Burna Boy", "tags": "afrobeats nigerian world music dance african"},
        ]

        self.artists_df = pd.DataFrame(sample_data)
        print(f"已加载 {len(self.artists_df)} 位艺术家的标签数据")

    def _build_tag_matrix(self):
        """构建标签TF-IDF矩阵"""
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words='english',  # 过滤英语停用词
            ngram_range=(1, 2),  # 使用1-2个词的组合，捕捉"classic rock"这样的短语
            max_features=1000
        )

        self.tag_matrix = self.vectorizer.fit_transform(self.artists_df['tags'])
        print(f"标签矩阵已构建，特征维度: {self.tag_matrix.shape}")

    def _preprocess_tags(self, user_tags):
        """预处理用户输入的标签"""
        if isinstance(user_tags, str):
            # 按逗号、空格、分号分割
            tags = re.split(r'[,;，；\s]+', user_tags.lower().strip())
        elif isinstance(user_tags, list):
            tags = [t.lower().strip() for t in user_tags]
        else:
            raise ValueError("标签必须是字符串或列表")

        # 过滤空标签
        tags = [t for t in tags if t]
        return tags

    def _expand_tags(self, tags):
        """
        标签扩展：添加同义词和相关词
        这能让推荐更智能，比如输入"sad"也能匹配到"melancholic"
        """
        synonyms = {
            'sad': ['melancholic', 'emotional', 'dark', 'moody'],
            'happy': ['upbeat', 'energetic', 'dance', 'feel-good'],
            'chill': ['ambient', 'relaxing', 'calm', 'atmospheric', 'chillout'],
            'heavy': ['metal', 'hard rock', 'loud', 'aggressive'],
            'dance': ['electronic', 'edm', 'house', 'disco', 'upbeat'],
            'instrumental': ['classical', 'ambient', 'jazz', 'electronic'],
            'old': ['classic', '60s', '70s', '80s', 'legendary', 'vintage'],
            'new': ['contemporary', 'modern', '2020s', 'current'],
            'female': ['female vocalist', 'woman'],
            'guitar': ['rock', 'indie', 'folk', 'acoustic'],
            'piano': ['classical', 'contemporary classical', 'minimalist'],
            'experimental': ['avant-garde', 'innovative', 'creative'],
            'energy': ['energetic', 'upbeat', 'dance', 'heavy', 'hype'],
            'relax': ['ambient', 'chill', 'calm', 'atmospheric', 'minimalist'],
            'workout': ['energetic', 'upbeat', 'heavy', 'dance', 'edm', 'hype'],
            'study': ['ambient', 'classical', 'instrumental', 'minimalist', 'focus'],
            'sleep': ['ambient', 'calm', 'minimalist', 'atmospheric', 'soft'],
            'road trip': ['rock', 'indie', 'pop', 'singalong', 'energetic'],
        }

        expanded = list(tags)
        for tag in tags:
            if tag in synonyms:
                expanded.extend(synonyms[tag][:3])  # 最多添加3个同义词

        return ' '.join(expanded)

    def recommend(self, user_input, top_n=10, use_expansion=True):
        """
        根据用户输入的标签推荐艺术家

        参数:
            user_input: str 或 list，用户输入的标签
            top_n: int，返回的推荐数量
            use_expansion: bool，是否使用标签扩展

        返回:
            DataFrame，包含艺术家名、匹配分数和标签
        """
        # 预处理用户输入
        tags = self._preprocess_tags(user_input)

        if not tags:
            return pd.DataFrame(columns=['artist', 'score', 'tags'])

        # 可选：扩展标签
        if use_expansion:
            query_text = self._expand_tags(tags)
        else:
            query_text = ' '.join(tags)

        # 将查询转换为TF-IDF向量
        query_vector = self.vectorizer.transform([query_text])

        # 计算余弦相似度
        similarities = cosine_similarity(query_vector, self.tag_matrix).flatten()

        # 获取top N
        top_indices = similarities.argsort()[::-1][:top_n]

        # 构建结果
        results = []
        for idx in top_indices:
            if similarities[idx] > 0:  # 只返回有相关性的结果
                results.append({
                    'artist': self.artists_df.iloc[idx]['artist'],
                    'score': round(similarities[idx] * 100, 1),
                    'tags': self.artists_df.iloc[idx]['tags']
                })

        return pd.DataFrame(results)

    def get_tag_suggestions(self):
        """获取系统中所有可用的标签，帮助用户探索"""
        all_tags = []
        for tags_str in self.artists_df['tags']:
            all_tags.extend(tags_str.split())

        # 统计频率
        tag_counts = Counter(all_tags)
        return tag_counts.most_common(30)

    def find_similar_artists(self, artist_name, top_n=5):
        """根据艺术家名字找相似的艺术家"""
        # 找到目标艺术家
        mask = self.artists_df['artist'].str.lower() == artist_name.lower()
        if not mask.any():
            print(f"未找到艺术家: {artist_name}")
            return pd.DataFrame()

        artist_idx = mask.idxmax()
        artist_vector = self.tag_matrix[artist_idx]

        # 计算相似度
        similarities = cosine_similarity(artist_vector, self.tag_matrix).flatten()

        # 排除自己
        similarities[artist_idx] = -1

        top_indices = similarities.argsort()[::-1][:top_n]

        results = []
        for idx in top_indices:
            results.append({
                'artist': self.artists_df.iloc[idx]['artist'],
                'score': round(similarities[idx] * 100, 1),
                'tags': self.artists_df.iloc[idx]['tags']
            })

        return pd.DataFrame(results)


# ============ 交互式命令行界面 ============
def interactive_mode():
    """交互式推荐模式"""
    recommender = TagBasedArtistRecommender()

    print("\n" + "=" * 40)
    print("音乐艺术家标签推荐系统")
    print("=" * 40)
    print("输入标签（用逗号或空格分隔），系统为你推荐艺术家")
    print("示例：sad emotional piano")
    print("输入 ':tags' 查看热门标签")
    print("输入 ':similar 艺术家名' 查找相似艺术家")
    print("输入 ':quit' 退出")
    print("=" * 60 + "\n")

    while True:
        user_input = input("\n请输入标签 > ").strip()

        if not user_input:
            continue

        if user_input.lower() == ':quit':
            print("再见！")
            break

        if user_input.lower() == ':tags':
            print("\n系统中的热门标签:")
            popular_tags = recommender.get_tag_suggestions()
            for tag, count in popular_tags:
                print(f"  • {tag} ({count})")
            continue

        if user_input.lower().startswith(':similar'):
            artist_name = user_input[8:].strip()
            if artist_name:
                print(f"\n与 '{artist_name}' 相似的艺术家:")
                similar = recommender.find_similar_artists(artist_name, top_n=5)
                if not similar.empty:
                    for _, row in similar.iterrows():
                        print(f"  {row['artist']} (匹配度: {row['score']}%)")
                        print(f"   标签: {row['tags']}\n")
            else:
                print("请指定艺术家名字，例如: :similar Radiohead")
            continue

        # 正常推荐
        results = recommender.recommend(user_input, top_n=10)

        if results.empty:
            print("没有找到匹配的艺术家，试试其他标签？")
        else:
            print(f"\n根据 '{user_input}' 的推荐结果:\n")
            for i, (_, row) in enumerate(results.iterrows(), 1):
                print(f"{i}. {row['artist']} (匹配度: {row['score']}%)")
                print(f"   标签: {row['tags']}\n")


# ============ API 接口（方便集成到网页/应用） ============
def get_recommendations_by_tags(tags, top_n=10):
    """
    函数式接口，方便外部调用

    参数:
        tags: str, 用户输入的标签
        top_n: int, 返回数量

    返回:
        list of dict, 推荐结果
    """
    recommender = TagBasedArtistRecommender()
    results_df = recommender.recommend(tags, top_n=top_n)
    return results_df.to_dict('records')


# ============ 运行 ============
if __name__ == "__main__":
    # 方式1: 交互式命令行
    interactive_mode()

    # 方式2: 直接调用（取消注释使用）
    # results = get_recommendations_by_tags("chill electronic ambient", top_n=5)
    # for r in results:
    #     print(f"{r['artist']}: {r['score']}%")