import logging
import logging.config
import yaml

from pkg_resources import resource_filename


def construct_logger(module_name: str):
    """
    reads a logging config file and returns a logger.

    Parameters
    ----------
    module_name : str
        the name to be used for the logger

    Returns
    -------
    logging.logger
        a logger instance to use for logging. log log log log log
    """
    log_config_file = resource_filename("config", "log_config.yaml")
    with open(log_config_file) as fp:
        log_config = yaml.safe_load(fp.read())
    logging.config.dictConfig(log_config)
    logger = logging.getLogger(module_name)
    return logger
