import os
import cv2
import faiss
import torch
import numpy as np
from torchvision import models, transforms

model_path = "models/stylelens_resnet.pth"
data_dir = "data/fashion/train"
img_size = 224

transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=True)
model.fc = torch.nn.Identity()
state_dict = torch.load(model_path, map_location=device)
model.load_state_dict(state_dict, strict=False)
model = model.to(device).eval()

def extract_features(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size))
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model(tensor).cpu().numpy()
    return feat

def build_index(image_paths):
    feats = [extract_features(p) for p in image_paths]
    feats = np.vstack(feats)
    dim = feats.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(feats)
    return index, feats

def recommend(query_image, image_paths, index, k=5):
    q_feat = extract_features(query_image)
    D, I = index.search(q_feat, k)
    return [image_paths[i] for i in I[0]]

if __name__ == "__main__":
    image_paths = [os.path.join(data_dir, cls, f)
                   for cls in os.listdir(data_dir)
                   for f in os.listdir(os.path.join(data_dir, cls))[:3]]
    index, feats = build_index(image_paths)
    results = recommend(image_paths[0], image_paths, index)
    print("Recommendations:", results)
