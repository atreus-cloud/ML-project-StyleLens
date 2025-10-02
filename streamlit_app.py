import streamlit as st
from recommend import recommend, build_index
import os

st.title("StyleLens - Fashion Recommender")

image_paths = [os.path.join("data/fashion/train", cls, f)
               for cls in os.listdir("data/fashion/train")
               for f in os.listdir(os.path.join("data/fashion/train", cls))[:3]]
index, _ = build_index(image_paths)

uploaded_file = st.file_uploader("Upload a clothing image", type=["jpg", "png"])
if uploaded_file:
    img_path = "temp.jpg"
    with open(img_path, "wb") as f:
        f.write(uploaded_file.read())
    st.image(img_path, caption="Uploaded Image", use_column_width=True)
    recs = recommend(img_path, image_paths, index)
    st.subheader("Recommended Similar Styles:")
    for r in recs:
        st.image(r, width=150)
