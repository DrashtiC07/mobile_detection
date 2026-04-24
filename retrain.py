import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b0
from torch.utils.data import DataLoader, WeightedRandomSampler
import os

# Same transforms as before (simple version)
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224, scale=(0.7,1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.3,0.3,0.3),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

train_data = datasets.ImageFolder("dataset/train", transform=transform)

# Calculate class weights (more weight to iPhone & Samsung)
class_counts = [len(os.listdir(f"dataset/train/{cls}")) for cls in ['iphone','oppo','samsung','vivo']]
weights = [1.0 / count for count in class_counts]
sample_weights = [weights[label] for label in train_data.targets]

sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

train_loader = DataLoader(train_data, batch_size=32, sampler=sampler)

# Load best model and continue training
model = efficientnet_b0(weights=None)
model.classifier = nn.Sequential(
    nn.Dropout(0.5), nn.Linear(1280, 512), nn.ReLU(), nn.Dropout(0.4), nn.Linear(512, 4)
)
model.load_state_dict(torch.load("models/efficientnet_best.pth", map_location='cpu'))
model.train()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

for epoch in range(10):   # 10 more epochs
    model.train()
    correct = total = 0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        _, pred = outputs.max(1)
        total += labels.size(0)
        correct += pred.eq(labels).sum().item()
    
    print(f"Epoch {epoch+1}/10 | Train Acc: {100*correct/total:.2f}%")

torch.save(model.state_dict(), "models/efficientnet_final_fixed.pth")
print("✅ Retraining done with class balancing!")