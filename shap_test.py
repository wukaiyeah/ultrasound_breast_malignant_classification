import shap
import numpy as np
from keras.applications.mobilenet import preprocess_input
import keras
import cv2
import os
import glob
from matplotlib import pyplot as plt
from PIL import Image
model_weight = '/media/wukai/project/breast_diff_resolustion_project/MobileNet_224_checkpoints/2021913101.h5'
model = keras.models.load_model(model_weight)
# 定义模型预测函数（示例）
def predict(images):
    # 替换为你的模型推理逻辑
    # 注意：输入images应为uint8格式，需在此函数内完成预处理（归一化等）
    processed_images = preprocess_input(images.astype(np.float32))  # 假设preprocess_input是模型预处理
    return model.predict(processed_images)

# 参数设置
topk = 4
batch_size = 50
n_evals = 10000
class_names = ['benign', 'malignant']
for img_dir in glob.glob('/media/wukai/project/breast_diff_resolustion_project/tumor_samples/*'):
    img_name = os.path.basename(img_dir).split('.')[0]
    # 加载图像（使用SHAP工具确保格式正确）
    img = Image.open(img_dir)  # 形状 (224,224,3)，uint8类型
    img = cv2.resize(np.array(img), (224, 224))
    img = np.expand_dims(img, axis=0)   # 转换为 (1,224,224,3)

    # 创建masker（指定dtype和模糊核）
    masker = shap.maskers.Image(
        "blur(127,127)", 
        shape=(224,224,3)
    )

    # 创建解释器
    explainer = shap.Explainer(
        predict, 
        masker,
        output_names=class_names
    )

    # 计算SHAP值
    shap_values = explainer(
        img,
        max_evals=n_evals,
        batch_size=batch_size,
        outputs=shap.Explanation.argsort.flip[:topk]
    )

    # 可视化
    shap.image_plot(shap_values, img)
    plt.savefig(f'tumor_sample_cam/{img_name}_shap.jpg')