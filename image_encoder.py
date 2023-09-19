import torch
import clip
import os
from PIL import Image
import json

def get_image_features(model, preprocess, folder_path):
    image_features_dict = {}

    for subdir, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):
                # 获取完整的文件路径
                full_path = os.path.join(subdir, file)

                # 使用CLIP预处理图片并计算feature
                image_tensor = preprocess(Image.open(full_path)).unsqueeze(0).to(device)
                with torch.no_grad():
                    image_features = model.encode_image(image_tensor)
                image_features = image_features.cpu().numpy().tolist()

                # 保存image feature到字典
                image_features_dict[full_path] = image_features[0]

    return image_features_dict

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    folders = {

        "D:\\8715_project\\japanese-handwriting-analysis\\bin_seg": "D:\\8715_project\\japanese-handwriting-analysis\\bin_img_json"
    }

    for input_folder, output_folder in folders.items():
        image_features_dict = get_image_features(model, preprocess, input_folder)
        # 保存结果到JSON文件中
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        output_file = os.path.join(output_folder, os.path.basename(input_folder) + ".json")
        with open(output_file, 'w') as f:
            json.dump(image_features_dict, f, indent=4)
