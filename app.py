from flask import Flask, request, jsonify
from recommend import recommend, build_index
import os

app = Flask(__name__)

image_paths = [os.path.join("data/fashion/train", cls, f)
               for cls in os.listdir("data/fashion/train")
               for f in os.listdir(os.path.join("data/fashion/train", cls))[:3]]
index, _ = build_index(image_paths)

@app.route("/recommend", methods=["POST"])
def recommend_api():
    file = request.files["file"]
    img_path = "temp.jpg"
    file.save(img_path)
    recs = recommend(img_path, image_paths, index)
    return jsonify({"recommendations": recs})

if __name__ == "__main__":
    app.run(debug=True)
