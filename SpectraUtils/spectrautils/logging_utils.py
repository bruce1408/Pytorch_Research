import os
import sys
import queue
import atexit
import datetime
import logging
from colorama import Fore, Style, init
from logging.handlers import QueueHandler, QueueListener

init(autoreset=True)

class ColoredFormatter(logging.Formatter):
    """自定义的日志格式化器，用于在控制台输出彩色日志"""
    
    LEVEL_COLOR_MAPPING = {
        logging.DEBUG: Fore.CYAN,      # 调试信息：青色
        logging.INFO: Fore.GREEN,      # 普通信息：绿色
        logging.WARNING: Fore.YELLOW,  # 警告信息：黄色
        logging.ERROR: Fore.RED,       # 错误信息：红色
        logging.CRITICAL: Fore.MAGENTA # 严重错误：紫色
    }
    
    def format(self, record):
        """格式化日志记录"""
        level_color = self.LEVEL_COLOR_MAPPING.get(record.levelno, '')
        prefix = f"[{record.asctime} {record.name}] ({record.filename} {record.lineno}): {record.levelname} "
        colored_prefix = f"{level_color}{prefix}{Style.RESET_ALL}"
        return f"{colored_prefix}{record.msg}"


class AsyncLoggerManager:
    """异步日志管理器（单例模式）"""
    
    _instance = None

    def __new__(cls, *args, **kwargs):
        """实现单例模式"""
        if not cls._instance:
            cls._instance = super(AsyncLoggerManager, cls).__new__(cls)
            cls._instance.init_logger(*args, **kwargs)
        return cls._instance

    def init_logger(self, work_dir=None, log_file=None, level=logging.INFO, name_prefix=None):
        """初始化日志记录器"""
        if not hasattr(self, "initialized"):
            self.logger = logging.getLogger()
            
            # 清除已有的处理器，避免重复添加
            self.logger.handlers.clear()
            
            self.logger.setLevel(level)
            
            # 创建消息队列
            self.log_queue = queue.Queue(-1)

            if log_file is None:
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                log_file = f"{timestamp}.log"
                if name_prefix:
                    log_file = f"{name_prefix}_{log_file}"
                if work_dir is not None:
                    os.makedirs(work_dir, exist_ok=True)
                    log_file = os.path.join(work_dir, log_file)

            # 设置日志格式
            fmt = '[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s'

            # 创建文件处理器
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)
            file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))

            # 创建控制台处理器
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(level)
            console_handler.setFormatter(ColoredFormatter(fmt, datefmt='%Y-%m-%d %H:%M:%S'))
            
            # 创建日志队列处理器
            queue_handler = QueueHandler(self.log_queue)
            self.logger.addHandler(queue_handler)

            # 创建队列监听器，监听日志队列并将日志发送到文件和控制台
            if not hasattr(self, 'listener'):
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



if __name__ == "__main__":
    """测试日志记录器"""
    # 禁止重复记录stderr
    # filter_stderr()

    # 指定日志目录
    log_directory = "./logs"
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)
    
    # 获取异步日志记录器实例
    async_logger_manager = AsyncLoggerManager(work_dir=log_directory)
    logger = async_logger_manager.logger
    
    # 记录一些日志消息
    logger.info("This is an INFO message.")
    logger.warning("This is a WARNING message.")
    logger.error("This is an ERROR message.")