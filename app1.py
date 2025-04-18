import os
import torch
import timm
from flask import Flask, request, jsonify
from torchvision import transforms
from PIL import Image


# Flask setup
app = Flask(__name__)
UPLOAD_FOLDER = 'uploaded'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model
def load_model(model_path, num_classes):
    model = timm.create_model("rexnet_150", pretrained=False, num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))  # for CPU
    model.eval()
    return model

model_path = "disease_best_model.pth"
num_classes = 72 # Update with your actual number
model = load_model(model_path, num_classes)

# Class names list
class_names = ["Apple___alternaria_leaf_spot","Apple___black_rot","Apple___brown_spot","Apple___gray_spot","Apple___healthy","Apple___rust",
                "Apple___scab","Bell_pepper___bacterial_spot","Bell_pepper___healthy","Blueberry___healthy","Cassava___bacterial_blight",
                "Cassava___brown_streak_disease","Cassava___green_mottle","Cassava___healthy","Cassava___mosaic_disease","Cherry___healthy",
                "Cherry___powdery_mildew","Coffee___healthy","Coffee___red_spider_mite","Coffee___rust","Corn___common_rust","Corn___gray_leaf_spot",
                "Corn___healthy","Corn___northern_leaf_blight","Grape___Leaf_blight","Grape___black_measles","Grape___black_rot","Grape___healthy",
                "Orange___citrus_greening","Peach___bacterial_spot","Peach___healthy","Potato___bacterial_wilt","Potato___early_blight","Potato___healthy",
                "Potato___late_blight","Potato___leafroll_virus","Potato___mosaic_virus","Potato___nematode","Potato___pests","Potato___phytophthora",
                "Raspberry___healthy","Rice___bacterial_blight","Rice___blast","Rice___brown_spot","Rice___tungro","Rose___healthy","Rose___rust",
                "Rose___slug_sawfly","Soybean___healthy","Squash___powdery_mildew","Strawberry___healthy","Strawberry___leaf_scorch","Sugercane___healthy",
                "Sugercane___mosaic","Sugercane___red_rot","Sugercane___rust","Sugercane___yellow_leaf","Tomato___bacterial_spot","Tomato___early_blight",
                "Tomato___healthy","Tomato___late_blight","Tomato___leaf_curl","Tomato___leaf_mold","Tomato___mosaic_virus","Tomato___septoria_leaf_spot",
                "Tomato___spider_mites","Tomato___target_spot","Watermelon___anthracnose","Watermelon___downy_mildew","Watermelon___healthy","Watermelon___mosaic_virus"]  

# Preprocessing
transformations = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

# Predict function
def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transformations(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image)
        _, pred = torch.max(output, 1)
    
    return class_names[pred.item()]

# Routes
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'imagefile' not in request.files:
            return 'No file part'

        file = request.files['imagefile']
        if file.filename == '':
            return 'No selected file'

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        prediction = predict_image(filepath)
        return jsonify({"result":prediction})
    


if __name__ == '__main__':
    app.run(port=3000, debug=True)
