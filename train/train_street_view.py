from __future__ import print_function

import os
import sys

from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau
from keras.applications.resnet50 import ResNet50
from keras.applications.imagenet_utils import preprocess_input
from keras.models import Model
from keras.layers import Dense, Flatten
from keras.optimizers import SGD

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import angle_error, RotNetDataGenerator
from data.street_view import get_filenames as get_street_view_filenames


# 设置数据路径并获取训练和测试文件名
data_path = os.path.join('data', 'street_view')
train_filenames, test_filenames = get_street_view_filenames(data_path)

print(len(train_filenames), 'train samples')
print(len(test_filenames), 'test samples')

model_name = 'rotnet_street_view_resnet50'

# 模型配置参数
nb_classes = 360  # 输出类别数(角度0-359)
input_shape = (224, 224, 3)  # 输入图像尺寸

# 加载预训练的ResNet50模型作为基础模型
base_model = ResNet50(weights='imagenet', include_top=False,
                      input_shape=input_shape)

# 构建分类层
x = base_model.output
x = Flatten()(x)
final_output = Dense(nb_classes, activation='softmax', name='fc360')(x)

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
monitor = 'val_angle_error'  # 监控验证集角度误差
checkpointer = ModelCheckpoint(  # 保存最佳模型
    filepath=os.path.join(output_folder, model_name + '.keras'),
    monitor=monitor,
    save_best_only=True
)
reduce_lr = ReduceLROnPlateau(monitor=monitor, patience=3)  # 学习率自动调整
early_stopping = EarlyStopping(monitor=monitor, patience=5)  # 早停策略
tensorboard = TensorBoard()  # TensorBoard可视化

# 开始训练
model.fit_generator(
    # 训练数据生成器
    RotNetDataGenerator(
        train_filenames,
        input_shape=input_shape,
        batch_size=batch_size,
        preprocess_func=preprocess_input,
        crop_center=True,
        crop_largest_rect=True,
        shuffle=True
    ),
    steps_per_epoch=len(train_filenames) / batch_size,
    epochs=nb_epoch,
    # 验证数据生成器
    validation_data=RotNetDataGenerator(
        test_filenames,
        input_shape=input_shape,
        batch_size=batch_size,
        preprocess_func=preprocess_input,
        crop_center=True,
        crop_largest_rect=True
    ),
    validation_steps=len(test_filenames) / batch_size,
    callbacks=[checkpointer, reduce_lr, early_stopping, tensorboard],
    workers=10  # 多进程数据加载
)
