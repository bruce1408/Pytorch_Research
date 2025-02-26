import pandas as pd
import os
import glob
import gc  # 添加垃圾回收模块

# 设置要读取的Excel文件所在的目录和输出文件路径
excel_dir = "/Users/brucetrie/Downloads/8620项目/370_parking_obstacle/25_0219_model/excel_file_merge"
output_file = "./合并后的数据.xlsx"

# 获取目录中所有的Excel文件
excel_files = glob.glob(os.path.join(excel_dir, "*.xls")) + glob.glob(os.path.join(excel_dir, "*.xlsx"))

if not excel_files:
    print("目录中没有找到Excel文件！")
    exit()

print(f"找到以下Excel文件：")
for file in excel_files:
    print(f" - {os.path.basename(file)}")

# 读取第一个文件作为基础表
print(f"使用 {os.path.basename(excel_files[0])} 作为基础表")
df_base = pd.read_excel(excel_files[0])

# 依次合并其他文件
for file in excel_files[1:]:
    try:
        print(f"正在合并: {os.path.basename(file)}")
        # 使用chunksize参数分块读取大文件
        df_current = pd.read_excel(file)
        
        # 找出在两个表中重复的列（除了合并键'label'）
        common_cols = set(df_base.columns) & set(df_current.columns)
        if 'label' in common_cols:
            common_cols.remove('label')
        
        # 如果存在重复列，对df_current中的重复列进行重命名
        if common_cols:
            # 使用文件名（去掉扩展名）作为后缀
            file_suffix = os.path.splitext(os.path.basename(file))[0]
            rename_dict = {col: f"{col}_{file_suffix}" for col in common_cols}
            df_current = df_current.rename(columns=rename_dict)
        
        # 按'label'列进行左连接合并
        df_base = pd.merge(df_base, df_current, on='label', how='left')
        
        # 强制垃圾回收
        del df_current
        gc.collect()
        
        # 每合并一个文件就保存一次中间结果，避免全部失败
        temp_output = f"./temp_merged_{len(df_base.columns)}.xlsx"
        df_base.to_excel(temp_output, index=False)
        print(f"已保存中间结果至 {temp_output}")
        
    except Exception as e:
        print(f"合并文件 {os.path.basename(file)} 时出错: {str(e)}")
        continue

# 保存合并后的数据到新的Excel文件
try:
    df_base.to_excel(output_file, index=False)
    print(f"合并完成！结果已保存至 {output_file}")
    
    # 删除临时文件
    for temp_file in glob.glob("./temp_merged_*.xlsx"):
        os.remove(temp_file)
        
except Exception as e:
    print(f"保存最终结果时出错: {str(e)}")
    print("请查看最新的临时文件作为结果")