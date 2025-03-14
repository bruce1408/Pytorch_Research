import logging
import queue
import unicodedata
import datetime
import sys
import os
import atexit
import threading
from logging.handlers import QueueHandler, QueueListener
from termcolor import colored

def get_display_width(text):
    """
    计算字符串在终端中的显示宽度。
    全角字符计为宽度2，半角字符计为宽度1。
    """
    width = 0
    for char in text:
        if unicodedata.east_asian_width(char) in ('F', 'W', 'A'):
            width += 2
        else:
            width += 1
    return width

class AsyncLoggerManager:
    """异步日志管理器（单例模式）
    
    使用队列实现异步日志记录，避免日志记录影响主程序性能
    """
    
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        """实现单例模式"""
        with cls._lock:
            if not cls._instance:
                cls._instance = super(AsyncLoggerManager, cls).__new__(cls)
                cls._instance.init_logger(*args, **kwargs)
        return cls._instance

    def init_logger(self, name="QnnHelperLogger", work_dir=None, log_file=None, level=logging.INFO):
        """初始化日志记录器
        
        Args:
            name: 日志记录器名称
            work_dir: 日志文件保存目录
            log_file: 日志文件名称
            level: 日志级别
        """
        
        if not hasattr(self, "initialized"):
            self.logger = logging.getLogger(name)
            self.logger.setLevel(level)
            
            # 创建消息队列
            self.log_queue = queue.Queue(-1)  # -1表示队列大小无限制

            if log_file is None:
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                log_file = f"qnnhelper_{timestamp}.log"
                if work_dir is not None:
                    log_file = os.path.join(work_dir, log_file)

            # 定义日志格式
            fmt = '%(message)s'

            # 创建文件处理器，将日志写入文件
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)
            file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))

            # 创建控制台处理器，将日志输出到控制台，使用彩色格式
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(level)
            console_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))

            # 创建日志队列和队列处理器
            queue_handler = QueueHandler(self.log_queue)
            self.logger.addHandler(queue_handler)

            # 创建队列监听器，监听日志队列并将日志发送到文件和控制台
            self.listener = QueueListener(self.log_queue, file_handler, console_handler)
            self.listener.start()
            self.initialized = True

            # 确保在程序退出时停止队列监听器
            atexit.register(self.stop_listener)

    def stop_listener(self):
        """停止队列监听器"""
        if self.listener:
            self.listener.stop()
            self.listener = None


# 获取异步日志记录器实例
logger_manager = AsyncLoggerManager()

# 使用锁，确保在多线程情况下，日志打印不会错乱
print_lock = threading.Lock()

def print_colored_box(text, bbox_width=40, text_color='white', box_color='green', background_color='on_white', attrs=['bold'], text_background=False, align='left'):
    with print_lock:  # 确保每次打印时，只有一个线程/进程在打印
        if isinstance(text, list):
            content_width = max(get_display_width(item) for item in text)
        else:
            content_width = get_display_width(text)

        # 确定总宽度，考虑到边框的宽度(+2)
        total_width = max(bbox_width, content_width + 4)
        
        # 生成顶部和底部的边框
        top_bottom_border = '+' + '-' * (total_width - 2) + '+'
        logger_manager.logger.info(colored(top_bottom_border, box_color, attrs=attrs))

        if isinstance(text, list):
            for item in text:
                space_padding = total_width - 2 - get_display_width(item) - 2  # 减去边框和文本两侧的空格
                if align == 'left':
                    line = f" {item} " + " " * space_padding
                elif align == 'right':
                    line = " " * space_padding + f" {item} "
                elif align == 'center':
                    left_padding = space_padding // 2
                    right_padding = space_padding - left_padding
                    line = " " * left_padding + f" {item} " + " " * right_padding

                logger_manager.logger.info(colored("|", box_color, attrs=attrs) + 
                            colored(line, text_color, attrs=attrs, on_color=background_color if text_background else None) + 
                            colored("|", box_color, attrs=attrs))
        else:
            space_padding = total_width - 2 - get_display_width(text) - 2
            if align == 'left':
                line = f" {text} " + " " * space_padding
            elif align == 'right':
                line = " " * space_padding + f" {text} "
            elif align == 'center':
                left_padding = space_padding // 2
                right_padding = space_padding - left_padding
                line = " " * left_padding + f" {text} " + " " * right_padding
            logger_manager.logger.info(colored("|", box_color, attrs=attrs) + 
                        colored(line, text_color, attrs=attrs, on_color=background_color if text_background else None) + 
                        colored("|", box_color, attrs=attrs))

        logger_manager.logger.info(colored(top_bottom_border, box_color, attrs=attrs))


def print_colored_box_line(title, message, attrs=['bold'], text_color='white', box_color='yellow', box_width=80):
    with print_lock:  # 确保每次打印时，只有一个线程/进程在打印
        # 使用 get_display_width 计算实际显示宽度
        title_width = get_display_width(title)
        message_width = get_display_width(message)
        
        # 计算需要的填充空格
        title_padding = box_width - 4 - title_width
        message_padding = box_width - 4 - message_width
        
        # 创建边框
        horizontal_border = '+' + '-' * (box_width - 2) + '+'
        colored_horizontal_border = colored(horizontal_border, box_color, attrs=attrs)
        
        # 创建标题和消息文本，手动计算居中位置
        left_title_pad = title_padding // 2
        right_title_pad = title_padding - left_title_pad
        left_message_pad = message_padding // 2
        right_message_pad = message_padding - left_message_pad
        
        title_text = f"| {' ' * left_title_pad}{title}{' ' * right_title_pad} |"
        message_text = f"| {' ' * left_message_pad}{message}{' ' * right_message_pad} |"
        
        # 添加颜色
        colored_title = colored(title_text, text_color, 'on_' + box_color, attrs=attrs)
        colored_message = colored(message_text, text_color, 'on_' + box_color, attrs=attrs)
        
        # 打印方框
        logger_manager.logger.info(colored_horizontal_border)
        logger_manager.logger.info(colored_title)
        logger_manager.logger.info(colored_horizontal_border)
        logger_manager.logger.info(colored_message)
        logger_manager.logger.info(colored_horizontal_border)
