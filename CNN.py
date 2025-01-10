import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomCNN(nn.Module):
    def __init__(self, input_height, input_width):
        super(CustomCNN, self).__init__()
        
        # Сверточные слои
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=7, stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=7, stride=2, padding=0)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7, stride=2, padding=0)
        
        # MaxPooling
        self.pool = nn.MaxPool2d(kernel_size=3, stride=3)
        
        # Вычисляем размерность выхода после сверток и пула
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, input_height, input_width)
            out = self.conv1(dummy_input)
            out = self.conv2(out)
            out = self.conv3(out)
            out = self.pool(out)
            self.flatten_size = out.numel()
        
        # Полносвязный слой
        self.fc = nn.Linear(self.flatten_size, 3)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Разворачиваем тензор
        x = self.fc(x)
        return x

# Пример использования
input_height = 64
input_width = 64

model = CustomCNN(input_height, input_width)
print(model)

# Проверка с фейковым входом
dummy_input = torch.randn(1, 3, input_height, input_width)
output = model(dummy_input)
print("Output shape:", output.shape)
