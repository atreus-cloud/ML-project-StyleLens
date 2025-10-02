import torch
import onnx
import tensorflow as tf
import onnx_tf
from torchvision import models

img_size = 224
num_classes = 5

model = models.resnet18(pretrained=True)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load("models/stylelens_resnet.pth", map_location="cpu"))

dummy_input = torch.randn(1, 3, img_size, img_size)
torch.onnx.export(model, dummy_input, "stylelens.onnx", export_params=True)

onnx_model = onnx.load("stylelens.onnx")
tf_rep = onnx_tf.backend.prepare(onnx_model)
tf_rep.export_graph("stylelens_tf")

converter = tf.lite.TFLiteConverter.from_saved_model("stylelens_tf")
tflite_model = converter.convert()
with open("stylelens.tflite", "wb") as f:
    f.write(tflite_model)
