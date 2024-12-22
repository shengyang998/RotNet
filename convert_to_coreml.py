import os
import sys
import coremltools
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import argparse

def normalize_angle(angle):
    """Normalize angle to [-1, 1] range"""
    return (angle / 180.0) - 1.0

def denormalize_angle(normalized):
    """Convert normalized value back to angle"""
    return (normalized + 1.0) * 180.0

def angle_error_normalized(y_true, y_pred):
    """
    Calculate the angle error for normalized angles
    Input angles are in normalized form [-1, 1]
    Returns error in degrees
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    # Convert normalized values back to angles
    y_true_angle = denormalize_angle(y_true)
    y_pred_angle = denormalize_angle(y_pred)
    
    # Calculate absolute difference
    diff = tf.abs(y_true_angle - y_pred_angle)
    return tf.minimum(360.0 - diff, diff)

def get_sample_data(input_shape, num_samples=100):
    """Generate sample data for calibration"""
    return np.random.rand(num_samples, *input_shape).astype(np.float32)

def convert_keras_to_coreml(keras_model_path, output_path, quantize_mode='none', num_calibration_samples=100):
    """
    Convert Keras model to Core ML format with quantization options
    
    Args:
        keras_model_path: Path to the Keras model file
        output_path: Path to save the Core ML model
        quantize_mode: Quantization mode ('none', 'weights', 'full')
        num_calibration_samples: Number of samples to use for calibration in full quantization
    """
    print("Loading Keras model...")
    
    # 加载自定义的损失函数和指标
    custom_objects = {
        'angle_error_normalized': angle_error_normalized
    }
    
    # 加载模型
    model = load_model(keras_model_path, custom_objects=custom_objects)
    
    # 定义输入格式
    input_shape = model.input_shape[1:]  # 去掉batch维度
    print(f"Model input shape: {input_shape}")
    
    # 创建输入描述
    image_input_description = coremltools.ImageType(
        name="image",
        shape=input_shape,
        scale=1/255.0,  # 归一化系数
        color_layout='RGB'
    )
    
    # 创建输出描述
    output_description = coremltools.TensorType(
        name="angle",
        shape=(1,)  # 输出一个角度值
    )
    
    # 配置量化选项
    config = None
    if quantize_mode != 'none':
        print(f"Applying {quantize_mode} quantization...")
        weights = coremltools.optimize.coreml.quantization.QuantizedWeights()
        
        if quantize_mode == 'weights':
            # 仅量化权重为8位整数
            config = coremltools.ComputeUnit.CPU_AND_NE
            weights.quantize_weights(nbits=8)
        elif quantize_mode == 'full':
            # 全量化模式，包括权重和激活值
            config = coremltools.ComputeUnit.CPU_AND_NE
            weights.quantize_weights(nbits=8)
            
            # 生成校准数据
            print("Generating calibration data...")
            calibration_data = get_sample_data(input_shape, num_calibration_samples)
            
            # 配置全量化参数
            config = coremltools.optimize.coreml.quantization.QuantizationConfig(
                mode="linear",
                nbits=8,
                symmetric=True,
                weight_threshold=0.9,
                compute_units=coremltools.ComputeUnit.CPU_AND_NE
            )
    
    # 转换为 Core ML 模型
    print("Converting to Core ML format...")
    coreml_model = coremltools.convert(
        model,
        inputs=[image_input_description],
        outputs=[output_description],
        minimum_deployment_target=coremltools.target.iOS15,
        source='tensorflow',
        compute_units=config
    )
    
    # 如果是全量化模式，进行校准
    if quantize_mode == 'full':
        print("Calibrating quantized model...")
        coreml_model = coremltools.optimize.coreml.quantization.quantize_weights(
            coreml_model,
            config,
            calibration_data
        )
    
    # 添加模型元数据
    coreml_model.author = "Your Name"
    coreml_model.license = "Your License"
    coreml_model.short_description = f"Image rotation angle prediction model ({quantize_mode} quantization)"
    coreml_model.version = "1.0"
    
    # 添加输入输出描述
    coreml_model.input_description["image"] = "Input image to predict rotation angle"
    coreml_model.output_description["angle"] = "Predicted rotation angle in degrees"
    
    # 添加后处理代码来转换归一化的输出到实际角度
    output_name = coreml_model.output_names[0]
    coreml_model = coremltools.models.neural_network.NeuralNetworkBuilder(
        spec=coreml_model.get_spec(),
        nn_spec=coreml_model.get_spec().neuralNetwork
    )
    
    # 添加后处理层来转换输出范围从[-1,1]到[0,360]
    coreml_model.add_scale(
        name="denormalize",
        input_name=output_name,
        output_name="rotation_angle",
        W=180.0,  # 缩放因子
        b=180.0   # 偏置
    )
    
    # 保存模型
    print(f"Saving Core ML model to {output_path}")
    coreml_model.save(output_path)
    
    # 打印模型大小信息
    original_size = os.path.getsize(keras_model_path) / (1024 * 1024)
    converted_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\nModel size comparison:")
    print(f"Original Keras model: {original_size:.2f} MB")
    print(f"Converted CoreML model: {converted_size:.2f} MB")
    print(f"Size reduction: {((original_size - converted_size) / original_size * 100):.1f}%")
    
    print("\nConversion completed successfully!")

def main():
    parser = argparse.ArgumentParser(description='Convert Keras model to CoreML format with quantization options')
    parser.add_argument('model_path', help='Path to the Keras model file')
    parser.add_argument('--quantize', choices=['none', 'weights', 'full'], default='none',
                      help='Quantization mode: none, weights, or full (default: none)')
    parser.add_argument('--num-samples', type=int, default=100,
                      help='Number of calibration samples for full quantization (default: 100)')
    parser.add_argument('--output', help='Output path for the CoreML model (optional)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        sys.exit(1)
    
    # 设置输出路径
    if args.output:
        output_path = args.output
    else:
        base_path = os.path.splitext(args.model_path)[0]
        if args.quantize != 'none':
            output_path = f"{base_path}_{args.quantize}_quantized.mlmodel"
        else:
            output_path = f"{base_path}.mlmodel"
    
    try:
        convert_keras_to_coreml(
            args.model_path,
            output_path,
            quantize_mode=args.quantize,
            num_calibration_samples=args.num_samples
        )
    except Exception as e:
        print(f"Error during conversion: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 