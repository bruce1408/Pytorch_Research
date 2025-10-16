import os
import glob
import pandas as pd

def convert_xls_to_xlsx(file_path, output_dir):
    """将 xls 文件转换为 xlsx 格式"""
    try:
        print(f"正在将 {os.path.basename(file_path)} 转换为 xlsx 格式...")
        # 读取 xls 文件
        df = pd.read_excel(file_path, engine='xlrd')
        
        # 构建新的 xlsx 文件路径，保存在原始文件的同一目录
        original_dir = os.path.dirname(file_path)
        new_filename = os.path.splitext(os.path.basename(file_path))[0] + '.xlsx'
        new_file_path = os.path.join(original_dir, new_filename)
        
        # 保存为 xlsx 格式
        df.to_excel(new_file_path, index=False, engine='openpyxl')
        print(f"文件已转换并保存为: {new_file_path}")
        return new_file_path
    except Exception as e:
        print(f"转换文件格式时发生错误: {str(e)}")
        return None
    

def create_label(row):
    """创建标签：将指定列的值用下划线连接"""
    try:
        # 获取需要连接的列的值，并转换为字符串
        values = [
            str(row['指标名']),
            str(row['车位类型']),
            str(row['距离范围']),
            str(row['车位状态'])
        ]
        # 使用下划线连接所有值
        return '_'.join(values)
    except Exception as e:
        print(f"创建标签时出错: {str(e)}")
        return "error"
    
    
def process_excel_file(file_path, output_dir):
    """处理单个Excel文件"""
    try:
        print(f"\n正在处理文件: {os.path.basename(file_path)}")
        
        # 首先检查文件类型
        # file_type = check_file_type(file_path)
        
        # 尝试不同的引擎读取文件
        df = None
        errors = []
        
        # 尝试使用不同的引擎读取文件
        engines = ['xlrd', 'openpyxl', 'odf']
        for engine in engines:
            try:
                df = pd.read_excel(file_path, engine=engine)
                print(f"成功使用{engine}引擎读取文件")
                break
            except Exception as e:
                errors.append(f"{engine}错误: {str(e)}")
                continue
        
        if df is None:
            print(f"无法读取文件 {os.path.basename(file_path)}，尝试了所有可用引擎")
            print("错误信息：")
            for error in errors:
                print(error)
            return False
        
        # 创建标签列
        print("正在创建标签...")
        df['label'] = df.apply(create_label, axis=1)
        
        # 保存处理后的文件，强制使用 .xlsx 格式
        output_path = os.path.join(
            os.path.dirname(file_path),
            f"labeled_{os.path.splitext(os.path.basename(file_path))[0]}.xlsx"  # 修改这里，强制使用 .xlsx 扩展名
        )
        
        output_path = os.path.join(
            output_dir,
            f"labeled_{os.path.splitext(os.path.basename(file_path))[0]}.xlsx"
        )
        
        # 使用 openpyxl 引擎保存为 xlsx 格式
        df.to_excel(output_path, index=False, engine='openpyxl')
        print(f"已保存标签文件到: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"处理文件时发生未预期的错误: {str(e)}")
        return False
    
    
def process_directory(directory_path, output_dir):
    """处理指定目录下的所有Excel文件"""
    # 确保目录路径存在
    if not os.path.exists(directory_path):
        print(f"错误：目录 '{directory_path}' 不存在！")
        return
    
    
    if not os.path.exists(output_dir):
        print(f"目录 '{output_dir}' 不存在，正在创建...")
        os.makedirs(output_dir)
        print(f"已成功创建目录 '{output_dir}'")
    
    # 获取所有Excel文件（支持.xls和.xlsx）
    excel_files = []
    excel_files.extend(glob.glob(os.path.join(directory_path, "*.xls")))
    excel_files.extend(glob.glob(os.path.join(directory_path, "*.xlsx")))
    excel_files.extend(glob.glob(os.path.join(directory_path, "*.XLS")))
    excel_files.extend(glob.glob(os.path.join(directory_path, "*.XLSX")))
    
    if not excel_files:
        print(f"在目录 '{directory_path}' 中没有找到Excel文件！")
        return
    
    print(f"找到 {len(excel_files)} 个Excel文件")
    
    # 处理统计
    success_count = 0
    failed_count = 0
    
    # 处理每个文件
    for file_path in excel_files:
        print(f"\n{'='*50}")
        
        # 检查文件是否为 xls 格式
        if file_path.lower().endswith('.xls'):
            # 转换为 xlsx 格式
            converted_path = convert_xls_to_xlsx(file_path, output_dir)
            if converted_path:
                # 使用转换后的文件路径继续处理
                if process_excel_file(converted_path, output_dir):
                    success_count += 1
                else:
                    failed_count += 1
            else:
                failed_count += 1
        else:
            # 直接处理 xlsx 文件
            if process_excel_file(file_path, output_dir):
                success_count += 1
            else:
                failed_count += 1
    
    # 打印处理结果统计
    print("\n处理完成！")
    print(f"成功处理: {success_count} 个文件")
    print(f"处理失败: {failed_count} 个文件")

def main():
    """主函数"""
    print("=== Excel文件批量处理工具 ===")
    
    # 指定要处理的目录路径
    target_directory = "/mnt/share_disk/bruce_trie/workspace/outputs/psd_validation_mobius_results"
    output_dir = os.path.join(target_directory, "label_results")

    print(f"处理目录: {target_directory}")
    
    try:
        process_directory(target_directory, output_dir)
    except Exception as e:
        print(f"程序执行过程中发生错误: {str(e)}")

if __name__ == "__main__":
    main()