import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

DEVICE = torch.device("cpu")

classes = ['A+', 'A-', 'AB+', 'AB-', 'B+', 'B-', 'O+', 'O-']

model = models.densenet201(weights=None)

num_ftrs = model.classifier.in_features

model.classifier = nn.Sequential(
    nn.Linear(num_ftrs, 256),
    nn.BatchNorm1d(256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, 8)
)

model.load_state_dict(torch.load("best_academic_model.pth", map_location=DEVICE))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])


def predict(image_path):

    image = Image.open(image_path).convert("RGB")

    image = transform(image)

    image = image.unsqueeze(0)

    with torch.no_grad():

        outputs = model(image)

        probabilities = torch.softmax(outputs, dim=1)

        confidence, predicted = torch.max(probabilities,1)

    predicted_class = classes[predicted.item()]

    confidence_score = round(confidence.item()*100,2)

    return predicted_class, confidence_score