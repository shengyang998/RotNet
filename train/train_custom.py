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
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

from concurrent.futures import ThreadPoolExecutor
import threading

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import angle_error_regression, RotNetDataGenerator

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

model_name = 'rotnet_custom_resnet50_regression'

# 修改模型配置参数 - 回归模型只需要预测一个角度值
input_shape = (224, 224, 3)

# 加载预训练的ResNet50模型
base_model = ResNet50(weights='imagenet', 
                     include_top=False,
                     input_shape=input_shape,
                     pooling='avg')

# 冻结部分层
for layer in base_model.layers[:-50]:
    layer.trainable = False

# 简化回归模型结构
x = base_model.output
x = Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
x = tf.keras.layers.Dropout(0.5)(x)
# 最后一层使用线性激活函数，直接输出角度值
final_output = Dense(1, activation='linear', name='angle_output')(x)

# 创建完整模型
model = Model(inputs=base_model.input, outputs=final_output)

# 使用Adam优化器
optimizer = Adam(learning_rate=0.0001)

# 配置模型训练参数 - 使用MSE或自定义损失函数
def angle_loss(y_true, y_pred):
    """Custom loss function for angle regression"""
    # Convert angles to radians
    diff = tf.abs(y_true - y_pred)
    # Handle wrap-around cases (e.g., 359 vs 1 degree)
    return tf.reduce_mean(tf.minimum(diff, 360.0 - diff))

model.compile(
    loss=angle_loss,
    optimizer=optimizer,
    metrics=[angle_error_regression]
)

# 训练参数
batch_size = 16
nb_epoch = 100

# 创建目录
output_folder = 'models'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

log_dir = os.path.join('logs', model_name)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# 配置回调函数
monitor = 'val_angle_error_regression'
checkpointer = ModelCheckpoint(
    filepath=os.path.join(output_folder, model_name + '.keras'),
    monitor=monitor,
    mode='min',
    save_best_only=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor=monitor,
    factor=0.5,
    patience=3,
    min_lr=1e-6,
    mode='min',
    verbose=1
)

early_stopping = EarlyStopping(
    monitor=monitor,
    patience=10,
    mode='min',
    restore_best_weights=True,
    verbose=1
)

tensorboard = TensorBoard(
    log_dir=log_dir,
    histogram_freq=1,
    write_graph=True,
    update_freq='epoch'
)

# 创建数据生成器 - 使用回归模式
train_generator = RotNetDataGenerator(
    train_filenames,
    input_shape=input_shape,
    batch_size=batch_size,
    preprocess_func=preprocess_input,
    crop_center=True,
    crop_largest_rect=True,
    shuffle=True,
    one_hot=False  # 不使用one-hot编码，直接输出角度值
)

validation_generator = RotNetDataGenerator(
    test_filenames,
    input_shape=input_shape,
    batch_size=batch_size,
    preprocess_func=preprocess_input,
    crop_center=True,
    crop_largest_rect=True,
    one_hot=False  # 不使用one-hot编码，直接输出角度值
)

# 开始训练
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_filenames) // batch_size,
    epochs=nb_epoch,
    validation_data=validation_generator,
    validation_steps=len(test_filenames) // batch_size,
    callbacks=[checkpointer, reduce_lr, early_stopping, tensorboard],
    workers=4,
    use_multiprocessing=False
)

# 添加训练过程可视化
plt.figure(figsize=(12, 4))

# 绘制损失曲线
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# 绘制角度误差曲线
plt.subplot(1, 2, 2)
plt.plot(history.history['angle_error_regression'], label='Training Angle Error')
plt.plot(history.history['val_angle_error_regression'], label='Validation Angle Error')
plt.title('Model Angle Error')
plt.xlabel('Epoch')
plt.ylabel('Angle Error (degrees)')
plt.legend()

plt.tight_layout()
plt.savefig('training_history.png')
plt.close()
 
