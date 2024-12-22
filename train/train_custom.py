from __future__ import print_function

import os
import sys
from glob import glob
import cv2
import numpy as np
import shutil

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import SGD

from concurrent.futures import ThreadPoolExecutor
from functools import partial
import threading

from tensorflow.keras import mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# 添加一个线程锁来保护打印操作
print_lock = threading.Lock()

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import angle_error, RotNetDataGenerator

import tensorflow as tf

def preprocess_image(image_path, max_width=4000, max_height=3000):
    """按比例缩放图像，确保不超过最大尺寸"""
    # 检查是否已经处理过
    processed_dir = os.path.join(os.path.dirname(image_path), 'processed')
    filename = os.path.basename(image_path)
    new_path = os.path.join(processed_dir, f'processed_{filename}')
    
    # 如果处理后的文件已存在，直接返回路径
    if os.path.exists(new_path):
        return new_path
    
    img = cv2.imread(image_path)
    if img is None:
        with print_lock:
            print(f"警告: 无法读取图像 {image_path}")
        return None
        
    height, width = img.shape[:2]
    
    # 计算缩放比例
    scale_w = max_width / width if width > max_width else 1
    scale_h = max_height / height if height > max_height else 1
    scale = min(scale_w, scale_h)
    
    # 如果图像小于最大尺寸，复制到processed目录
    if scale >= 1:
        os.makedirs(processed_dir, exist_ok=True)
        shutil.copy2(image_path, new_path)
        return new_path
        
    # 计算新的尺寸
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    # 创建处理后的图像目录
    os.makedirs(processed_dir, exist_ok=True)
    
    # 整图像大小
    resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    # 保存处理后的像
    cv2.imwrite(new_path, resized)
    with print_lock:
        print(f"已处理图像 {filename}: {width}x{height} -> {new_width}x{new_height}")
    
    return new_path

def get_custom_filenames(data_path, valid_extensions=('.JPG', '.jpg', '.jpeg', '.png')):
    """Get and process all images from the data directory"""
    # First, get all original images
    original_files = []
    for ext in valid_extensions:
        original_files.extend(glob(os.path.join(data_path, f'*{ext}')))
    
    if not original_files:
        raise ValueError(f"No original images found in {data_path}")
    
    print(f"Found {len(original_files)} original images")
    
    # Process all images using ThreadPoolExecutor
    processed_files = []
    with ThreadPoolExecutor() as executor:
        processed_files = list(filter(None, executor.map(preprocess_image, original_files)))
    
    if not processed_files:
        raise ValueError(f"No images were successfully processed")
    
    print(f"Successfully processed {len(processed_files)} images")
    
    # Randomly split into train and test sets
    import random
    random.shuffle(processed_files)
    split_idx = int(len(processed_files) * 0.8)
    return processed_files[:split_idx], processed_files[split_idx:]

# Change the relative path to absolute path
data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'custom_images')
print(f"Looking for images in: {data_path}")
train_filenames, test_filenames = get_custom_filenames(data_path)

if len(train_filenames) == 0 or len(test_filenames) == 0:
    raise ValueError(f"No images found in {data_path}. Please ensure the directory contains valid image files.")

print(len(train_filenames), 'train samples')
print(len(test_filenames), 'test samples')

model_name = 'rotnet_custom_resnet50'

# 修改模型配置参数
nb_classes = 36000
input_shape = (224, 224, 3)

# 加载预训练的ResNet50模型，冻结部分层
base_model = ResNet50(weights='imagenet', 
                     include_top=False,
                     input_shape=input_shape,
                     pooling='avg')  # 使用全局平均池化

# 冻结前面的层，只训练后面的层
for layer in base_model.layers[:-50]:  # 只训练最后50层
    layer.trainable = False

# 简化分类层结构，添加dropout防止过拟合
x = base_model.output
x = Dense(1024, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
x = tf.keras.layers.Dropout(0.5)(x)
final_output = Dense(nb_classes, activation='softmax', name='fc36000')(x)

# 创建完整模型
model = Model(inputs=base_model.input, outputs=final_output)

# 使用更小的初始学习率和更好的优化器
optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.0001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07
)

# 配置模型训练参数
model.compile(
    loss='categorical_crossentropy',
    optimizer=optimizer,
    metrics=[angle_error]
)

# 减小批量大小以提高泛化能力
batch_size = 16
nb_epoch = 100

# 创建模型保存目录
output_folder = 'models'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Add this: Create logs directory for TensorBoard
log_dir = os.path.join('logs', model_name)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# 配置回调函数
monitor = 'val_angle_error'
checkpointer = ModelCheckpoint(
    filepath=os.path.join(output_folder, model_name + '.keras'),
    monitor=monitor,
    save_best_only=True
)
reduce_lr = ReduceLROnPlateau(monitor=monitor, patience=3, mode='min')
early_stopping = EarlyStopping(monitor=monitor, patience=5, mode='min')
tensorboard = TensorBoard(
    log_dir=log_dir,
    histogram_freq=1,
    write_graph=True,
    write_images=True,
    update_freq='batch',
    profile_batch=0
)

# Create a custom callback to ensure metrics are logged properly
class MetricsLogger(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        # Ensure angle error is logged with the correct name
        if 'angle_error' in logs:
            logs['angle_error'] = float(logs['angle_error'])
        if 'val_angle_error' in logs:
            logs['val_angle_error'] = float(logs['val_angle_error'])

# Update the callbacks list to include the MetricsLogger
callbacks = [
    checkpointer,
    reduce_lr,
    early_stopping,
    tensorboard,
    MetricsLogger()
]

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
    callbacks=callbacks
) 
