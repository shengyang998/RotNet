import os
import shutil
from pathlib import Path
import filecmp

def copy_images(source_dir, target_dir):
    # 确保目标目录存在
    os.makedirs(target_dir, exist_ok=True)
    
    # 获取源目录中的所有jpg文件
    source_path = Path(source_dir)
    jpg_files = list(source_path.rglob("*.jpg")) + list(source_path.rglob("*.JPG"))
    
    print(f"找到 {len(jpg_files)} 个jpg文件")
    
    copied_count = 0
    skipped_count = 0
    
    # 复制文件
    for i, jpg_file in enumerate(jpg_files, 1):
        target_file = os.path.join(target_dir, jpg_file.name)
        
        # 检查文件是否已存在
        if os.path.exists(target_file):
            # 如果文件内容相同，跳过复制
            if filecmp.cmp(str(jpg_file), target_file, shallow=False):
                skipped_count += 1
                if i % 100 == 0:
                    print(f"处理进度: {i}/{len(jpg_files)} (已复制: {copied_count}, 已跳过: {skipped_count})")
                continue
            
            # 如果内容不同，使用新文件名
            base, ext = os.path.splitext(jpg_file.name)
            counter = 1
            while os.path.exists(target_file):
                target_file = os.path.join(target_dir, f"{base}_{counter}{ext}")
                counter += 1
        
        shutil.copy2(jpg_file, target_file)
        copied_count += 1
        
        if i % 100 == 0:
            print(f"处理进度: {i}/{len(jpg_files)} (已复制: {copied_count}, 已跳过: {skipped_count})")
    
    print(f"\n复制完成！")
    print(f"总文件数: {len(jpg_files)}")
    print(f"复制文件数: {copied_count}")
    print(f"跳过文件数: {skipped_count}")

if __name__ == "__main__":
    source_directory = r"E:\LightroomImport"
    target_directory = "data/custom_images"
    
    copy_images(source_directory, target_directory)