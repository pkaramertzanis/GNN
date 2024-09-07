import logging
import sys
APP_LOGGER_NAME = 'GNN_muta'



def setup_applevel_logger(logger_name = APP_LOGGER_NAME, file_name=None, level_stream=logging.WARNING, level_file=logging.DEBUG):

    # remove all handlers associated with the root logger object (needed in Google Colab)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    sh.setLevel(level_stream)
    logger.handlers.clear()
    logger.addHandler(sh)
    if file_name:
        fh = logging.FileHandler(file_name, encoding='utf-8')
        fh.setFormatter(formatter)
        fh.setLevel(level_file)
        logger.addHandler(fh)
    return logger

def get_logger(module_name):
   return logging.getLogger(APP_LOGGER_NAME).getChild(module_name)