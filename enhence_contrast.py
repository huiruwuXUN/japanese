import cv2
import numpy as np

def enhance_contrast_image(img_path):
    # 读取图像
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"Error: Unable to read image from {img_path}")
        return

    # 将图像转换为浮点数，方便后续计算
    img_float = img.astype('float32')

    # 找到图像的最小和最大值
    min_val = np.min(img_float)
    max_val = np.max(img_float)

    # 执行线性对比度拉伸
    enhanced_img = 255.0 * (img_float - min_val) / (max_val - min_val)
    enhanced_img = np.clip(enhanced_img, 0, 255).astype('uint8')  # 保证值在0-255范围内

    # 保存增强的图像
    #cv2.imwrite(output_path, enhanced_img)

    # 显示增强的图像
    cv2.imshow("Enhanced Contrast Linearly", enhanced_img)
    cv2.imshow("original ",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    image_path = "D:\8715_project\japanese-handwriting-analysis\seg_letter\RC05117_03\\5.jpg"
    #output_path = "path_to_save_enhanced_image.jpg"
    enhance_contrast_image(image_path)
