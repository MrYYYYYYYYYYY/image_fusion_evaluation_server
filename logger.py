
#:werkzeug.exceptions.NotFound: 404 Not Found: The requested URL was not found on the server. If you entered the URL manually please check your spelling and try again.
import logging
import logging.handlers as handlers


class InfoLog(object):
    def __init__(self, file_name):
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logHandler = handlers.TimedRotatingFileHandler(file_name, when='D', interval=1, backupCount=10)
        self.logHandler.setLevel(logging.INFO)
        self.logHandler.setFormatter(formatter)
        self.file_name = file_name
        self.cons = logging.basicConfig(format='%(asctime)s: %(message)s', level=logging.INFO)

    def send_msg(self, msg):
        logger = logging.getLogger(self.file_name)
        logger.setLevel(logging.INFO)
        logger.addHandler(self.logHandler)
        logger.info(msg)
