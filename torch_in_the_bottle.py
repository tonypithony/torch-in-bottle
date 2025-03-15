# pip install tornado torch torchvision torchaudio Pillow

# https://youtu.be/ZA28l34Jih0?si=vaRKhG-wHu7dwGNv
# https://programmersought.com/article/14957873228/

from bottle import run, route, request, json_dumps
import torch
from torchvision import models, transforms
from PIL import Image
import io
import base64

# Загрузка предобученной модели
model = models.resnet50(pretrained=True)
model.eval()

# Определение преобразования для предварительной обработки входного изображения
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Список меток классов ImageNet
with open('imagenet-classes.txt', 'r') as f:
    imagenet_classes = f.read().splitlines()

@route('/predict/', method=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Получаем строку Base64 из запроса
        image_data = request.json.get('image')
        
        if not image_data:
            return json_dumps({"error": "No image data provided."}), 400
        
        # Декодируем строку Base64
        try:
            image_bytes = base64.b64decode(image_data)
        except Exception as e:
            return json_dumps({"error": "Invalid Base64 string."}), 400
        
        # Открываем изображение
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as e:
            return json_dumps({"error": "Unable to open image."}), 400
        
        image_tensor = preprocess(image).unsqueeze(0)
        
        with torch.no_grad():
            output = model(image_tensor)
        
        # Получаем индекс предсказанного класса
        predicted_class_index = output.argmax(dim=1).item()
        
        # Получаем имя класса из списка
        predicted_class_name = imagenet_classes[predicted_class_index]
        
        # Возвращаем предсказанный класс в формате JSON
        return json_dumps({
            "predicted_class_index": predicted_class_index, 
            "predicted_class_name": predicted_class_name
        })
    
    else:
        return json_dumps({"error": "Invalid request method."}), 400

if __name__ == "__main__":
    run(host='0.0.0.0', port=8042, server='tornado', reload=True)

# Bottle v0.14-dev server starting up (using TornadoServer())...
# Listening on http://0.0.0.0:8042/
# Hit Ctrl-C to quit.
