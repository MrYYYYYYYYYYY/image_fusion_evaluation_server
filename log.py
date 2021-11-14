import logging
from flask.logging import default_handler
import os

from logging.handlers import RotatingFileHandler
from logging import StreamHandler
import logging
from logging import handlers

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

LOG_PATH = os.path.join(BASE_DIR, 'logs')

LOG_PATH_ERROR = os.path.join(LOG_PATH, 'error.log')
LOG_PATH_INFO = os.path.join(LOG_PATH, 'info.log')
LOG_PATH_ALL = os.path.join(LOG_PATH, 'all.log')

LOG_FILE_MAX_BYTES = 100 * 1024 * 1024
LOG_FILE_BACKUP_COUNT = 10


class LoggerF(object):
    level_relations = {
        'debug':logging.DEBUG,
        'info':logging.INFO,
        'warning':logging.WARNING,
        'error':logging.ERROR,
        'crit':logging.CRITICAL
    }#日志级别关系映射
 
    def __init__(self,filename,level='info',when='D',backCount=3,fmt='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'):
        self.logger = logging.getLogger(filename)
        format_str = logging.Formatter(fmt)#设置日志格式
        self.logger.setLevel(self.level_relations.get(level))#设置日志级别
        sh = logging.StreamHandler()#往屏幕上输出
        sh.setFormatter(format_str) #设置屏幕上显示的格式
        th = handlers.TimedRotatingFileHandler(filename=filename,when=when,backupCount=backCount,encoding='utf-8')#往文件里写入#指定间隔时间自动生成文件的处理器
        th.setFormatter(format_str)#设置文件里写入的格式
        self.logger.addHandler(sh) #把对象加到logger里
        self.logger.addHandler(th)

class Logger(object):

    def init_app(self, app):
        app.logger.removeHandler(default_handler)

        formatter = logging.Formatter(
            '%(asctime)s [%(thread)d:%(threadName)s] [%(filename)s:%(module)s:%(funcName)s] '
            '[%(levelname)s]: %(message)s'
        )

        file_handler_e = RotatingFileHandler(
            filename=LOG_PATH_ERROR,
            mode='a',
            maxBytes=LOG_FILE_MAX_BYTES,
            backupCount=LOG_FILE_BACKUP_COUNT,
            encoding='utf-8'
        )
        file_handler = RotatingFileHandler(
            filename=LOG_PATH_ALL,
            mode='a',
            maxBytes=LOG_FILE_MAX_BYTES,
            backupCount=LOG_FILE_BACKUP_COUNT,
            encoding='utf-8'
        )

        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)
        file_handler_e.setFormatter(formatter)
        file_handler_e.setLevel(logging.DEBUG)


        stream_handler = StreamHandler()
        stream_handler.setFormatter(formatter)
        stream_handler.setLevel(logging.DEBUG)

        for logger in (
                app.logger,
                logging.getLogger('sqlalchemy'),
                logging.getLogger('werkzeug')

        ):
            logger.addHandler(file_handler)
            logger.addHandler(file_handler_e)

