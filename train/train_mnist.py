from __future__ import print_function

import os
import sys

from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.datasets import mnist
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Model

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import angle_error, RotNetDataGenerator, binarize_images

# 加载MNIST数据集,只需要图像不需要标签
(X_train, _), (X_test, _) = mnist.load_data()

model_name = 'rotnet_mnist'

# 模型配置参数
nb_filters = 64  # 卷积核数量
pool_size = (2, 2)  # 最大池化窗口大小
kernel_size = (3, 3)  # 卷积核大小
nb_classes = 36000  # 输出类别数(角度0-359.99, 步长0.01)

# 获取数据维度信息
nb_train_samples, img_rows, img_cols = X_train.shape
img_channels = 1  # MNIST为灰度图像,通道数为1
input_shape = (img_rows, img_cols, img_channels)
nb_test_samples = X_test.shape[0]

print('Input shape:', input_shape)
print(nb_train_samples, 'train samples')
print(nb_test_samples, 'test samples')

# 构建CNN模型
input = Input(shape=(img_rows, img_cols, img_channels))
x = Conv2D(nb_filters, kernel_size, activation='relu')(input)  # 第一个卷积层
x = Conv2D(nb_filters, kernel_size, activation='relu')(x)  # 第二个卷积层
x = MaxPooling2D(pool_size=(2, 2))(x)  # 最大池化层
x = Dropout(0.25)(x)  # dropout防止过拟合
x = Flatten()(x)  # 展平层
x = Dense(128, activation='relu')(x)  # 全连接层
x = Dropout(0.25)(x)  # dropout防止过拟合
x = Dense(512, activation='relu')(x)  # 增加神经元数量
x = Dropout(0.25)(x)
x = Dense(nb_classes, activation='softmax')(x)  # 现在输出36000个类别

model = Model(inputs=input, outputs=x)

model.summary()

# 配置模型训练参数
model.compile(loss='categorical_crossentropy',  # 分类交叉熵损失函数
              optimizer='adam',  # Adam优化器
              metrics=[angle_error])  # 使用角度误差作为评估指标

# 训练超参数
batch_size = 128
nb_epoch = 50

# 创建模型保存目录
output_folder = 'models'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 配置回调函数
checkpointer = ModelCheckpoint(  # 保存最佳模型
    filepath=os.path.join(output_folder, model_name + '.keras'),
    save_best_only=True
)
early_stopping = EarlyStopping(patience=2)  # 早停策略
tensorboard = TensorBoard()  # TensorBoard可视化

# 开始训练
model.fit(
    # 训练数据生成器
    RotNetDataGenerator(
        X_train,
        batch_size=batch_size,
        preprocess_func=binarize_images,
        shuffle=True
    ),
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=nb_epoch,
    validation_data=RotNetDataGenerator(
        X_test,
        batch_size=batch_size,
        preprocess_func=binarize_images
    ),
    validation_steps=nb_test_samples // batch_size,
    verbose=1,
    callbacks=[checkpointer, early_stopping, tensorboard]
)
