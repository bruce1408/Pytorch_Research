# import pandas as pd
# import os
# import glob
# import sys

# # 检查必要的依赖
# try:
#     import xlrd
#     print(f"xlrd版本: {xlrd.__version__}")
# except ImportError:
#     print("缺少必要的依赖 'xlrd'。请使用以下命令安装：")
#     print("pip install xlrd==1.2.0")  # 使用1.2.0版本，它支持.xls和.xlsx
#     sys.exit(1)

# try:
#     import openpyxl
#     print(f"openpyxl版本: {openpyxl.__version__}")
# except ImportError:
#     print("缺少依赖 'openpyxl'。如果需要处理.xlsx文件，请安装：")
#     print("pip install openpyxl")

# # 设置要读取的Excel文件所在的目录和输出文件路径
# excel_dir = "/home/bruce_ultra/workspace/excel_file_merge"
# output_file = "./merge_multi_file.xlsx"

# # 定义要筛选的列名关键词
# target_keywords = ["实验版本", "对照版本"]

# # 获取目录中所有的Excel文件
# excel_files = glob.glob(os.path.join(excel_dir, "*.xls")) + glob.glob(os.path.join(excel_dir, "*.xlsx"))

# if not excel_files:
#     print("目录中没有找到Excel文件！")
#     exit()

# print(f"找到以下Excel文件：")
# for file in excel_files:
#     print(f" - {os.path.basename(file)}")

# # 首先收集所有文件中的label值，以便计算交集
# all_labels_sets = []

# # 读取所有文件的label列
# for file in excel_files:
#     try:
#         print(f"读取文件 {os.path.basename(file)} 的label列")
        
#         # 尝试使用适当的引擎读取文件
#         try:
#             if file.endswith('.xls'):
#                 df = pd.read_excel(file, engine='xlrd')
#             else:
#                 df = pd.read_excel(file, engine='openpyxl')
#         except Exception as e:
#             print(f"  使用第一个引擎读取失败，尝试其他引擎: {str(e)}")
#             if file.endswith('.xls'):
#                 df = pd.read_excel(file, engine='openpyxl')
#             else:
#                 df = pd.read_excel(file, engine='xlrd')
        
#         # 确保文件有label列
#         if 'label' not in df.columns:
#             print(f"  警告：文件 {os.path.basename(file)} 没有'label'列，将跳过")
#             continue
        
#         # 收集label值
#         labels = set(df['label'].dropna().tolist())
#         all_labels_sets.append(labels)
#         print(f"  找到 {len(labels)} 个唯一label值")
        
#     except Exception as e:
#         print(f"读取文件 {os.path.basename(file)} 时出错: {str(e)}")
#         continue

# # 如果没有成功读取任何文件的label列
# if not all_labels_sets:
#     print("没有找到任何有效的label列，无法继续")
#     sys.exit(1)

# # 计算所有label值的交集
# common_labels = set.intersection(*all_labels_sets)
# print(f"所有文件共有 {len(common_labels)} 个相同的label值")

# if not common_labels:
#     print("没有找到共同的label值，无法合并文件")
#     sys.exit(1)

# # 读取第一个文件作为基础表
# print(f"使用 {os.path.basename(excel_files[0])} 作为基础表")
# try:
#     # 尝试使用xlrd引擎
#     df_base_full = pd.read_excel(excel_files[0], engine='xlrd')
# except Exception as e:
#     print(f"使用xlrd引擎读取失败: {str(e)}")
#     try:
#         # 尝试使用openpyxl引擎
#         df_base_full = pd.read_excel(excel_files[0], engine='openpyxl')
#     except Exception as e2:
#         print(f"使用openpyxl引擎也失败: {str(e2)}")
#         print("请确保已安装正确的依赖库并且文件格式正确")
#         sys.exit(1)

# # 筛选包含关键词的列和label列
# base_columns = ['label']  # 始终保留label列
# for col in df_base_full.columns:
#     if any(keyword in col for keyword in target_keywords):
#         base_columns.append(col)

# # 如果没有找到匹配的列，提示用户
# if len(base_columns) <= 1:  # 只有label列
#     print(f"警告：在基础表中没有找到包含关键词 {target_keywords} 的列！")
#     print("将使用所有列作为基础表。")
#     base_columns = df_base_full.columns.tolist()

# # 创建只包含目标列的基础表，并只保留共同的label值
# df_base = df_base_full[base_columns].copy()
# df_base = df_base[df_base['label'].isin(common_labels)]
# print(f"基础表筛选后的列: {base_columns}")
# print(f"基础表筛选后的行数: {len(df_base)}")

# # 依次合并其他文件
# for file in excel_files[1:]:
#     try:
#         print(f"正在合并: {os.path.basename(file)}")
        
#         # 尝试使用适当的引擎读取文件
#         try:
#             if file.endswith('.xls'):
#                 df_current_full = pd.read_excel(file, engine='xlrd')
#             else:
#                 df_current_full = pd.read_excel(file, engine='openpyxl')
#         except Exception as e:
#             print(f"  读取失败，尝试其他引擎: {str(e)}")
#             if file.endswith('.xls'):
#                 df_current_full = pd.read_excel(file, engine='openpyxl')
#             else:
#                 df_current_full = pd.read_excel(file, engine='xlrd')
        
#         # 筛选当前文件中包含关键词的列和label列
#         current_columns = ['label']  # 始终保留label列
#         for col in df_current_full.columns:
#             if any(keyword in col for keyword in target_keywords):
#                 current_columns.append(col)
        
#         # 如果没有找到匹配的列，跳过此文件
#         if len(current_columns) <= 1:  # 只有label列
#             print(f"  跳过此文件：没有找到包含关键词 {target_keywords} 的列")
#             continue
        
#         # 创建只包含目标列的当前表，并只保留共同的label值
#         df_current = df_current_full[current_columns].copy()
#         df_current = df_current[df_current['label'].isin(common_labels)]
#         print(f"  当前文件筛选后的列: {current_columns}")
#         print(f"  当前文件筛选后的行数: {len(df_current)}")
        
#         # 找出在两个表中重复的列（除了合并键'label'）
#         common_cols = set(df_base.columns) & set(df_current.columns)
#         if 'label' in common_cols:
#             common_cols.remove('label')
        
#         # 如果存在重复列，对df_current中的重复列进行重命名
#         if common_cols:
#             # 使用文件名（去掉扩展名）作为后缀
#             file_suffix = os.path.splitext(os.path.basename(file))[0]
#             rename_dict = {col: f"{col}_{file_suffix}" for col in common_cols}
#             df_current = df_current.rename(columns=rename_dict)
        
#         # 按'label'列进行内连接合并，确保只保留共同的label值
#         df_base = pd.merge(df_base, df_current, on='label', how='inner')
#         print(f"  合并后，列数: {len(df_base.columns)}, 行数: {len(df_base)}")
        
#     except Exception as e:
#         print(f"合并文件 {os.path.basename(file)} 时出错: {str(e)}")
#         continue

# # 保存合并后的数据到新的Excel文件
# try:
#     # 确保使用openpyxl引擎保存为Excel格式
#     df_base.to_excel(output_file, index=False, engine='openpyxl')
#     print(f"合并完成！结果已保存至 {output_file}")
#     print(f"最终表格行数: {len(df_base)}, 列数: {len(df_base.columns)}")
# except Exception as e:
#     print(f"保存Excel文件失败: {str(e)}")
#     # 如果保存Excel失败，尝试保存为CSV作为备选
#     csv_output = output_file.replace('.xlsx', '.csv')
#     df_base.to_csv(csv_output, index=False)
#     print(f"已将结果保存为CSV格式: {csv_output}")


import pandas as pd
import os
import glob
import sys

# 检查必要的依赖
try:
    import xlrd
    print(f"xlrd版本: {xlrd.__version__}")
except ImportError:
    print("缺少必要的依赖 'xlrd'。请使用以下命令安装：")
    print("pip install xlrd==1.2.0")  # 使用1.2.0版本，它支持.xls和.xlsx
    sys.exit(1)

try:
    import openpyxl
    print(f"openpyxl版本: {openpyxl.__version__}")
except ImportError:
    print("缺少依赖 'openpyxl'。如果需要处理.xlsx文件，请安装：")
    print("pip install openpyxl")

# 设置要读取的Excel文件所在的目录和输出文件路径
excel_dir = "/home/bruce_ultra/workspace/excel_file_merge"
output_file = "./merge_multi_file.xlsx"

# 定义要筛选的列名关键词
target_keywords = ["实验版本", "对照版本"]

# 定义基础表中要保留的额外列
additional_columns = ["指标名", "样本量", "障碍物类型"]

# 获取目录中所有的Excel文件
excel_files = glob.glob(os.path.join(excel_dir, "*.xls")) + glob.glob(os.path.join(excel_dir, "*.xlsx"))

if not excel_files:
    print("目录中没有找到Excel文件！")
    exit()

print(f"找到以下Excel文件：")
for file in excel_files:
    print(f" - {os.path.basename(file)}")

# 首先收集所有文件中的label值，以便计算交集
all_labels_sets = []

# 读取所有文件的label列
for file in excel_files:
    try:
        print(f"读取文件 {os.path.basename(file)} 的label列")
        
        # 尝试使用适当的引擎读取文件
        try:
            if file.endswith('.xls'):
                df = pd.read_excel(file, engine='xlrd')
            else:
                df = pd.read_excel(file, engine='openpyxl')
        except Exception as e:
            print(f"  使用第一个引擎读取失败，尝试其他引擎: {str(e)}")
            if file.endswith('.xls'):
                df = pd.read_excel(file, engine='openpyxl')
            else:
                df = pd.read_excel(file, engine='xlrd')
        
        # 确保文件有label列
        if 'label' not in df.columns:
            print(f"  警告：文件 {os.path.basename(file)} 没有'label'列，将跳过")
            continue
        
        # 收集label值
        labels = set(df['label'].dropna().tolist())
        all_labels_sets.append(labels)
        print(f"  找到 {len(labels)} 个唯一label值")
        
    except Exception as e:
        print(f"读取文件 {os.path.basename(file)} 时出错: {str(e)}")
        continue

# 如果没有成功读取任何文件的label列
if not all_labels_sets:
    print("没有找到任何有效的label列，无法继续")
    sys.exit(1)

# 计算所有label值的交集
common_labels = set.intersection(*all_labels_sets)
print(f"所有文件共有 {len(common_labels)} 个相同的label值")

if not common_labels:
    print("没有找到共同的label值，无法合并文件")
    sys.exit(1)

# 读取第一个文件作为基础表
print(f"使用 {os.path.basename(excel_files[0])} 作为基础表")
try:
    # 尝试使用xlrd引擎
    df_base_full = pd.read_excel(excel_files[0], engine='xlrd')
except Exception as e:
    print(f"使用xlrd引擎读取失败: {str(e)}")
    try:
        # 尝试使用openpyxl引擎
        df_base_full = pd.read_excel(excel_files[0], engine='openpyxl')
    except Exception as e2:
        print(f"使用openpyxl引擎也失败: {str(e2)}")
        print("请确保已安装正确的依赖库并且文件格式正确")
        sys.exit(1)

# 筛选基础表中的列：包含关键词的列、label列和额外指定的列
base_columns = ['label']  # 始终保留label列

# 添加额外指定的列（如果存在）
for col in additional_columns:
    if col in df_base_full.columns:
        base_columns.append(col)
    else:
        print(f"警告：基础表中没有找到列 '{col}'")

# 添加包含关键词的列
for col in df_base_full.columns:
    if col not in base_columns and any(keyword in col for keyword in target_keywords):
        base_columns.append(col)

# 如果没有找到匹配的列，提示用户
if len(base_columns) <= 1 + len([col for col in additional_columns if col in df_base_full.columns]):
    # 只有label列和额外指定的列
    print(f"警告：在基础表中没有找到包含关键词 {target_keywords} 的列！")

# 创建只包含目标列的基础表，并只保留共同的label值
df_base = df_base_full[base_columns].copy()
df_base = df_base[df_base['label'].isin(common_labels)]
print(f"基础表筛选后的列: {base_columns}")
print(f"基础表筛选后的行数: {len(df_base)}")

# 依次合并其他文件
for file in excel_files[1:]:
    try:
        print(f"正在合并: {os.path.basename(file)}")
        
        # 尝试使用适当的引擎读取文件
        try:
            if file.endswith('.xls'):
                df_current_full = pd.read_excel(file, engine='xlrd')
            else:
                df_current_full = pd.read_excel(file, engine='openpyxl')
        except Exception as e:
            print(f"  读取失败，尝试其他引擎: {str(e)}")
            if file.endswith('.xls'):
                df_current_full = pd.read_excel(file, engine='openpyxl')
            else:
                df_current_full = pd.read_excel(file, engine='xlrd')
        
        # 筛选当前文件中包含关键词的列和label列
        current_columns = ['label']  # 始终保留label列
        for col in df_current_full.columns:
            if any(keyword in col for keyword in target_keywords):
                current_columns.append(col)
        
        # 如果没有找到匹配的列，跳过此文件
        if len(current_columns) <= 1:  # 只有label列
            print(f"  跳过此文件：没有找到包含关键词 {target_keywords} 的列")
            continue
        
        # 创建只包含目标列的当前表，并只保留共同的label值
        df_current = df_current_full[current_columns].copy()
        df_current = df_current[df_current['label'].isin(common_labels)]
        print(f"  当前文件筛选后的列: {current_columns}")
        print(f"  当前文件筛选后的行数: {len(df_current)}")
        
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
        
        # 按'label'列进行内连接合并，确保只保留共同的label值
        df_base = pd.merge(df_base, df_current, on='label', how='inner')
        print(f"  合并后，列数: {len(df_base.columns)}, 行数: {len(df_base)}")
        
    except Exception as e:
        print(f"合并文件 {os.path.basename(file)} 时出错: {str(e)}")
        continue

# 保存合并后的数据到新的Excel文件
try:
    # 确保使用openpyxl引擎保存为Excel格式
    df_base.to_excel(output_file, index=False, engine='openpyxl')
    print(f"合并完成！结果已保存至 {output_file}")
    print(f"最终表格行数: {len(df_base)}, 列数: {len(df_base.columns)}")
except Exception as e:
    print(f"保存Excel文件失败: {str(e)}")
    # 如果保存Excel失败，尝试保存为CSV作为备选
    csv_output = output_file.replace('.xlsx', '.csv')
    df_base.to_csv(csv_output, index=False)
    print(f"已将结果保存为CSV格式: {csv_output}")