import os
import os
import inspect

def enter_workspace():
    """
    将当前工作目录更改为调用此函数的脚本所在的目录。
    """
    # inspect.stack()[1] 获取的是调用者的堆栈帧信息
    caller_frame = inspect.stack()[1]

    # 从堆栈帧中拿到调用者脚本的文件路径
    caller_filepath = caller_frame.filename
    
    # 获取该文件所在目录的绝对路径
    caller_dir = os.path.dirname(os.path.abspath(caller_filepath))
    
    # 将当前工作目录切换到该目录
    os.chdir(caller_dir)
    
    # print(f"成功进入工作目录: {os.getcwd()}")
