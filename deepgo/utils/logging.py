""" logging.py
"""

import logging
import logging.config
import pathlib
import datetime
import os

import sys
from logging import DEBUG, INFO, WARNING, ERROR


class Formatter(logging.Formatter):
    """ Custom log message formatter
    """

    def __init__(self, *args, fancy_formatting=False, **kwargs):
        self.fancy = fancy_formatting
        super().__init__(*args, **kwargs)

    def format(self, record):
        return ''.join(filter(lambda x: x is not None, (
            # pylint: disable=line-too-long
            f'[{self.formatTime(record)}]',
            ' ',
            '\033[1m'  if self.fancy and record.levelno >= logging.INFO                                       else None,
            '\033[91m' if self.fancy and record.levelno >= logging.ERROR                                      else None,
            '\033[93m' if self.fancy and record.levelno >= logging.WARNING and record.levelno < logging.ERROR else None,
            record.levelname.lower(),
            '\033[0m'  if self.fancy else None,
            (
                f' ({record.filename}:{record.lineno})'
                if record.levelno != INFO else None
            ),
            ': ',
            record.getMessage(),
        )))


HANDLER = logging.StreamHandler(sys.stderr)
HANDLER.setFormatter(Formatter(fancy_formatting=sys.stderr.isatty()))

logging.basicConfig(
    handlers=[HANDLER],
)

LOGGER = logging.getLogger(__name__)

log = LOGGER.log  # pylint: disable=invalid-name


def set_level(level: int):
    """
    Parameters
    ----------
    level : int
        log level

    Side effects
    ------------
    Sets the log level to `level`

    Returns
    -------
    None
    """
    LOGGER.setLevel(level)

# ludvig's above
# bryan's below

# Based on https://mail.python.org/pipermail/python-list/2010-November/591474.html
class MultilineFormatter(logging.Formatter):
    def format(self, record):
        str = logging.Formatter.format(self, record)
        header, footer = str.split(record.message)
        str = str.replace('\n', '\n' + ' '*len(header))
        return str

def setup_logging(logfile=None, loglevel=logging.DEBUG):
    if logfile is None:
        logfile = "log/" + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')

    pathlib.Path(os.path.dirname(logfile)).mkdir(parents=True, exist_ok=True)

    cfg = dict(
          version=1,
          formatters={
              "f": {"()":
                        "deepgo.utils.logging.MultilineFormatter",
                    "format":
                        "%(levelname)-8s [%(asctime)s] %(message)s",
                    "datefmt":
                        "%m/%d %H:%M:%S"}
              },
          handlers={
              "s": {"class": "logging.StreamHandler",
                    "formatter": "f",
                    "level": loglevel},
              "f": {"class": "logging.FileHandler",
                    "formatter": "f",
                    # "level": logging.DEBUG,
                    "level": loglevel,
                    "filename": logfile}
              },
          root={
              "handlers": ["s", "f"],
              "level": logging.NOTSET
              },
          disable_existing_loggers=False,
      )
    logging.config.dictConfig(cfg)
