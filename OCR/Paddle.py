import paddle
from paddleocr import PaddleOCR
from PIL import Image

def test_paddle_ocr(img_path):
    try:
        # 检查是否使用GPU
        print("是否使用CUDA:", paddle.device.is_compiled_with_cuda())

        # 打印PaddlePaddle版本
        print("PaddlePaddle 版本:", paddle.__version__)

        # 检查设备
        print("使用设备:", paddle.device.get_device())

        # 初始化PaddleOCR
        ocr = PaddleOCR(use_angle_cls=True, use_gpu=True, show_log=False)

        # 执行OCR识别
        result = ocr.ocr(img_path, cls=True)

        # 打印结果
        for line in result:
            print(line)

    except Exception as e:
        print("运行时出错:", e)

if __name__ == "__main__":
    img_path = 'testocr/1.jpg'  # 替换为你想识别的图片路径
    test_paddle_ocr(img_path)
