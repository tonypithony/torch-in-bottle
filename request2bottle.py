# Чтобы отправить изображение в формате Base64 с помощью библиотеки requests в Python, вам нужно сначала преобразовать изображение в строку Base64, а затем отправить его в POST-запросе. Вот пример, как это сделать:

# Преобразуйте изображение в строку Base64.
# Отправьте POST-запрос с этой строкой.

import requests
import base64
from PIL import Image

def resize_image(image_path, output_path, max_size=(400, 400)):
    with Image.open(image_path) as img:
        img.thumbnail(max_size) # thumbnail() позволяет создать миниатюру, сохранив пропорции. 
        img.save(output_path)

# Путь к вашему изображению
image_path = "cover.jpg"
output_path = "resized.png"

# Уменьшаем изображение
resize_image(image_path, output_path)

# Преобразуем уменьшенное изображение в Base64
def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Преобразуем уменьшенное изображение в Base64
image_base64 = image_to_base64(output_path)


# Функция для преобразования изображения в строку Base64
def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Путь к вашему изображению
image_path = output_path #"ai.png"

# Преобразуем изображение в Base64
image_base64 = image_to_base64(image_path)

# URL вашего FastAPI сервиса
url = "http://192.168.100.149:8042/predict/"

# Отправляем POST-запрос
response = requests.post(url, json={"image": image_base64})

# Выводим статус-код и текст ответа
print("Status Code:", response.status_code)
print("Response Text:", response.text, '\n')

# Если статус-код 200, то пробуем декодировать JSON
if response.status_code == 200:
    decoded_text = response.json()
    # Выводим результат
    print(decoded_text['predicted_class_name'])
else:
    print("Error: Unable to decode JSON response.")


# Status Code: 200
# Response Text: {"predicted_class_index": 708, "predicted_class_name": "pedestal, plinth, footstall"} 

# pedestal, plinth, footstall