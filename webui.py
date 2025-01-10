import torch
import torch.nn.functional as F
import gradio as gr
import os
import cv2
import numpy as np

# Параметры модели и пути
MODEL_PATH = 'best.pth'
CLASS_NAMES = ['Cat', 'Dog', 'Panda']
RESIZE_W = 128
RESIZE_H = 128
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Определение модели
class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=7, stride=2, padding=0)
        self.conv2 = torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=7, stride=2, padding=0)
        self.conv3 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7, stride=2, padding=0)
        self.pool = torch.nn.MaxPool2d(kernel_size=3, stride=3)
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, RESIZE_W, RESIZE_H)
            out = self.conv1(dummy_input)
            out = self.conv2(out)
            out = self.conv3(out)
            out = self.pool(out)
            self.flatten_size = out.numel()
        self.fc = torch.nn.Linear(self.flatten_size, 3)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Загрузка модели
def load_model():
    model = Net()
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    return model.to(DEVICE)

model = torch.load(MODEL_PATH)

# Функция для обработки изображения
def process_image(image):
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (RESIZE_W, RESIZE_H))
    img = img.astype(np.float32) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    return img.to(DEVICE)

# Функция для предсказания
def predict_image(image):
    img = process_image(image)
    output = model(img)
    probabilities = F.softmax(output, dim=1).squeeze().tolist()  # Вычисляем вероятности
    result = {CLASS_NAMES[i]: prob for i, prob in enumerate(probabilities)}  # Формируем словарь
    return result

# Создание интерфейса Gradio
interface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=3),  # Отображаем вероятности для всех классов
    title="Image Classifier - Cat, Dog, Panda",
    description="Upload an image of a Cat, Dog, or Panda for classification. It will display probabilities for all classes.",
)

# Запуск интерфейса
if __name__ == "__main__":
    interface.launch()
