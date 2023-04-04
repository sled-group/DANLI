import torch
import logging
import os


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class DataParallel(torch.nn.DataParallel):
    """
    Allow nn.DataParallel to call model's attributes.
    """

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


def identity(x):
    """
    pickable equivalent of lambda x: x
    """
    return x


def create_logger(
    logger_name,
    debug=False,
    log_dir="log/",
    log_name="log.txt",
    output=["stream", "file"],
):

    logger = logging.getLogger(logger_name)

    LOG_FORMAT = "%(asctime)s %(name)s %(levelname)s: %(message)s"
    DATE_FORMAT = "%m/%d/%Y %H:%M:%S"
    formatter = logging.Formatter(fmt=LOG_FORMAT, datefmt=DATE_FORMAT)

    log_level = logging.INFO if not debug else logging.DEBUG

    handlers = []
    if "stream" in output:
        std_handler = logging.StreamHandler()
        handlers.append(std_handler)

    if "file" in output:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        log_fn = os.path.join(log_dir, log_name)
        file_handler = logging.FileHandler(log_fn)
        handlers.append(file_handler)

    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        logger.addHandler(handler)

    logger.propagate = False

    return logger
