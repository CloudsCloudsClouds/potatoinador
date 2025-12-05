import torch

# This script is just in case the model is already trained in a pth, but doesn't have a onnx
# It's just legacy, all train.py now also export a onnx
model = torch.load("potato_rock_classifier.pth")

onnx_prom = torch.onnx.export(model, (torch.randn(1, 3, 224, 224),), dynamo=True)

onnx_prom.save("potato_rock_classifier.onnx")
