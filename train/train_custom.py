from __future__ import print_function

import os
import sys
from glob import glob
import cv2
import numpy as np

from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau
from keras.applications.resnet50 import ResNet50
from keras.applications.imagenet_utils import preprocess_input
from keras.models import Model
from keras.layers import Dense, Flatten
from keras.optimizers import SGD

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import angle_error, RotNetDataGenerator

def preprocess_image(image_path, max_width=4000, max_height=3000):
    """按比例缩放图像，确保不超过最大尺寸"""
    img = cv2.imread(image_path)
    if img is None:
        print(f"警告: 无法读取图像 {image_path}")
        return None
        
    height, width = img.shape[:2]
    
    # 计算缩放比例
    scale_w = max_width / width if width > max_width else 1
    scale_h = max_height / height if height > max_height else 1
    scale = min(scale_w, scale_h)  # 选择较小的缩放比例以确保两个维度都不超过限制
    
    # 如果图像小于最大尺寸，则不进行缩放
    if scale >= 1:
        return image_path
        
    # 计算新的尺寸
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    # 创建处理后的图像目录
    processed_dir = os.path.join(os.path.dirname(image_path), 'processed')
    os.makedirs(processed_dir, exist_ok=True)
    
    # 生成新的文件名
    filename = os.path.basename(image_path)
    new_path = os.path.join(processed_dir, f'processed_{filename}')
    
    # 如果处理后的文件已存在，直接返��路径
    if os.path.exists(new_path):
        return new_path
        
    # 调整图像大小
    resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    # 保存处理后的图像
    cv2.imwrite(new_path, resized)
    print(f"已处理图像 {filename}: {width}x{height} -> {new_width}x{new_height}")
    
    return new_path

def get_custom_filenames(data_path, valid_extensions=('.jpg', '.jpeg', '.png')):
    """获取指定目录下的所有图片文件并进行预处理"""
    all_files = []
    for ext in valid_extensions:
        all_files.extend(glob(os.path.join(data_path, f'*{ext}')))
    
    # 预处理所有图像
    processed_files = []
    for file_path in all_files:
        processed_path = preprocess_image(file_path)
        if processed_path:
            processed_files.append(processed_path)
    
    # 随机分割训练集和测试集（80%训练，20%测试）
    import random
    random.shuffle(processed_files)
    split_idx = int(len(processed_files) * 0.8)
    return processed_files[:split_idx], processed_files[split_idx:]

# 设置数据路径并获取训练和测试文件名
data_path = 'data/custom_images'  # 修改为您的图片目录路径
train_filenames, test_filenames = get_custom_filenames(data_path)

print(len(train_filenames), 'train samples')
print(len(test_filenames), 'test samples')

model_name = 'rotnet_custom_resnet50'

# 模型配置参数
nb_classes = 36000  # 输出类别数(角度0-359.99, 步长0.01)
input_shape = (224, 224, 3)  # 输入图像尺寸

# 加载预训练的ResNet50模型作为基础模型
base_model = ResNet50(weights='imagenet', include_top=False,
                      input_shape=input_shape)

# 构建分类层
x = base_model.output
x = Flatten()(x)
x = Dense(512, activation='relu')(x)
final_output = Dense(nb_classes, activation='softmax', name='fc36000')(x)

# 创建完整模型
model = Model(inputs=base_model.input, outputs=final_output)

model.summary()

# 配置模型训练参数
model.compile(loss='categorical_crossentropy',
              optimizer=SGD(lr=0.01, momentum=0.9),
              metrics=[angle_error])

# 训练超参数
batch_size = 64
nb_epoch = 50

# 创建模型保存目录
output_folder = 'models'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 配置回调函数
monitor = 'val_angle_error'
checkpointer = ModelCheckpoint(
    filepath=os.path.join(output_folder, model_name + '.keras'),
    monitor=monitor,
    save_best_only=True
)
reduce_lr = ReduceLROnPlateau(monitor=monitor, patience=3)
early_stopping = EarlyStopping(monitor=monitor, patience=5)
tensorboard = TensorBoard()

# 开始训练
model.fit(
    RotNetDataGenerator(
        train_filenames,
        input_shape=input_shape,
        batch_size=batch_size,
        preprocess_func=preprocess_input,
        crop_center=True,
        crop_largest_rect=True,
        shuffle=True
    ),
    steps_per_epoch=len(train_filenames) // batch_size,
    epochs=nb_epoch,
    validation_data=RotNetDataGenerator(
        test_filenames,
        input_shape=input_shape,
        batch_size=batch_size,
        preprocess_func=preprocess_input,
        crop_center=True,
        crop_largest_rect=True
    ),
    validation_steps=len(test_filenames) // batch_size,
    callbacks=[checkpointer, reduce_lr, early_stopping, tensorboard],
    workers=10
) 