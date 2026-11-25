# -*- coding: utf-8 -*-
"""
Flask 网页 Demo：输入音乐标签，返回匹配的艺术家推荐结果。

启动：
    python app.py
然后访问 http://127.0.0.1:5000
"""
from flask import Flask, render_template, request, jsonify
import os
import traceback

# ========== 配置 ==========
BACKGROUND_URL = "https://images.unsplash.com/photo-1493225255756-d9584f8606e9?q=80&w=2070&auto=format&fit=crop"

app = Flask(__name__)

# ========== 导入推荐函数 ==========
print("=" * 60)
print("正在启动音乐推荐系统...")
print(f"当前工作目录：{os.getcwd()}")
print("=" * 60)

# 尝试导入推荐函数
try:
    from recommender import get_recommendations_by_tags

    print("✅ 成功导入 get_recommendations_by_tags")

    # 测试调用，验证推荐器是否正常工作
    print("正在测试推荐器...")
    test_result = get_recommendations_by_tags("rock", top_n=3)
    print(f"✅ 推荐器测试成功！返回 {len(test_result)} 条结果")
    if test_result:
        print(f"   示例结果：{test_result[0].get('artist', 'N/A')}")

except ImportError as e:
    print("❌ 导入失败：找不到 'recommender.py' 文件或模块")
    print(f"   错误详情：{e}")
    print("   请确保 recommender.py 文件存在于当前目录")

    # 定义一个备用函数，防止程序崩溃
    def get_recommendations_by_tags(tags, top_n=10):
        return [{"artist": "导入失败，请检查文件", "score": 0, "tags": "错误"}]

except Exception as e:
    print(f"❌ 推荐器初始化失败：{e}")
    traceback.print_exc()

    def get_recommendations_by_tags(tags, top_n=10):
        return [{"artist": f"初始化错误：{str(e)[:30]}", "score": 0, "tags": "请检查终端输出"}]

print("=" * 60)
print("服务启动中，请访问 http://127.0.0.1:5000")
print("=" * 60)


# ========== 路由：首页 ==========
@app.route("/")
def index():
    """首页"""
    return render_template("music.html", background_url=BACKGROUND_URL)


# ========== 路由：推荐接口（POST，供前端调用） ==========
@app.route("/recommend", methods=["POST"])
def recommend():
    """
    推荐API接口
    接收标签，返回推荐结果
    """
    print("\n" + "-" * 50)
    print("收到推荐请求")

    try:
        # 1. 获取参数
        tags = request.form.get("tags", "").strip()
        print(f"  用户输入标签：'{tags}'")

        # 2. 参数校验
        if not tags:
            print("  ❌ 标签为空")
            return jsonify({
                "success": False,
                "error": "请输入音乐标签",
                "results": []
            })

        # 3. 调用推荐函数
        print("  正在调用推荐引擎...")
        results = get_recommendations_by_tags(tags, top_n=10)
        print(f"  ✅ 推荐引擎返回 {len(results)} 条结果")

        # 4. 格式化结果
        formatted_results = []
        for i, item in enumerate(results, 1):
            formatted_results.append({
                "rank": i,
                "artist": item.get("artist", "未知艺术家"),
                "score": item.get("score", 0),
                "tags": item.get("tags", "")
            })

        # 5. 返回成功响应
        response = {
            "success": True,
            "query": tags,
            "count": len(formatted_results),
            "results": formatted_results,
            "error": None
        }
        print("  ✅ 响应构建成功，返回前端")
        return jsonify(response)

    except Exception as e:
        print(f"  ❌ 处理请求时发生异常：{e}")
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": f"服务器错误：{str(e)}",
            "results": []
        })


# ========== 路由：RESTful API（GET，供外部调用） ==========
@app.route("/api/recommend", methods=["GET"])
def recommend_api():
    """
    RESTful API接口（GET方式）
    示例：/api/recommend?tags=rock+electronic&top_n=5
    """
    try:
        tags = request.args.get("tags", "").strip()
        top_n = request.args.get("top_n", 10, type=int)

        if not tags:
            return jsonify({
                "success": False,
                "error": "请提供 tags 参数",
                "results": []
            })

        results = get_recommendations_by_tags(tags, top_n=top_n)

        return jsonify({
            "success": True,
            "query": tags,
            "count": len(results),
            "results": results
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "results": []
        })


# ========== 路由：健康检查 ==========
@app.route("/health", methods=["GET"])
def health_check():
    """健康检查接口"""
    return jsonify({
        "status": "running",
        "message": "音乐推荐系统服务正常"
    })


# ========== 启动服务 ==========
if __name__ == "__main__":
    try:
        app.run(
            debug=True,
            host="127.0.0.1",
            port=int(os.environ.get("PORT", 5000)),
            threaded=True
        )
    except KeyboardInterrupt:
        print("\n服务已停止")
    except Exception as e:
        print(f"启动失败：{e}")
        traceback.print_exc()
