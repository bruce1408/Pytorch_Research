import os
import numpy as np
import argparse
import glob

def convert_npz_to_raw(input_dir, output_dir, file_ext='.npz'):
    """
    将多级目录下的所有 NPZ 文件转换为 RAW 文件，保持相同的目录结构
    
    参数:
        input_dir (str): 输入顶级目录路径，包含 NPZ 文件的目录
        output_dir (str): 输出顶级目录路径，将保存 RAW 文件
        file_ext (str): 要处理的文件扩展名，默认为 '.npz'
    """
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")
    
    # 使用 glob 查找所有子目录中的 NPZ 文件
    pattern = os.path.join(input_dir, f"**/*{file_ext}")
    file_list = glob.glob(pattern, recursive=True)
    print(file_list)
    
    if not file_list:
        print(f"在 {input_dir} 及其子目录中未找到 {file_ext} 文件")
        return
    
    print(f"找到 {len(file_list)} 个 {file_ext} 文件")
    
    # 处理每个文件
    for input_path in file_list:
        # 计算相对路径，保持目录结构
        rel_path = os.path.relpath(input_path, input_dir)
        output_rel_path = os.path.splitext(rel_path)[0] + '.raw'
        output_path = os.path.join(output_dir, output_rel_path)
        
        # 确保输出文件的目录存在
        output_dir_path = os.path.dirname(output_path)
        if not os.path.exists(output_dir_path):
            os.makedirs(output_dir_path)
        
        # 读取 NPZ 文件
        try:
            data = np.load(input_path)
            # NPZ 文件包含多个数组，需要确定要导出的数组
            if isinstance(data, np.lib.npyio.NpzFile):
                # 获取 NPZ 文件中的第一个数组（假设这是我们需要的）
                array_name = list(data.keys())[0]
                array_data = data[array_name]
                
                # 将数据写入 RAW 文件
                array_data.tofile(output_path)
                print(f"已转换: {input_path} -> {output_path}")
                print(f"形状: {array_data.shape}, 数据类型: {array_data.dtype}")
            else:
                print(f"无法处理文件: {input_path}，不是有效的 NPZ 文件")
        except Exception as e:
            print(f"处理文件 {input_path} 时出错: {e}")
        finally:
            # 如果是 NpzFile 对象，需要关闭
            if 'data' in locals() and isinstance(data, np.lib.npyio.NpzFile):
                data.close()

def main():
    parser = argparse.ArgumentParser(description='将 NPZ 文件转换为 RAW 文件，保持目录结构')
    parser.add_argument('--input_dir', type=str, default="/share/cdd/V71_npz",  help='包含 NPZ 文件的顶级输入目录')
    parser.add_argument('--output_dir', type=str, default= "/share/cdd/V71_raw/", help='保存 RAW 文件的顶级输出目录')
    parser.add_argument('--ext', type=str, default='.npz', help='要处理的文件扩展名 (默认: .npz)')
    
    args = parser.parse_args()
    
    convert_npz_to_raw(args.input_dir, args.output_dir, args.ext)
    
if __name__ == "__main__":
    main()