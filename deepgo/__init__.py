# WARNING: parser must be called very early on for argcomplete.
# argcomplete evaluates the package until the parser is constructed before it
# can generate completions. Because of this, the parser must be constructed
# before the full package is imported to behave in a usable way. Note that
# running
# > python -m histonet
# will actually import the entire package (along with dependencies like
# pytorch, numpy, and pandas), before running __main__.py, which takes
# about 0.5-1 seconds.
# See Performance section of https://argcomplete.readthedocs.io/en/latest/

from .parser import parser
parser()

from deepgo.__version__ import __version__
from deepgo.config import config
from deepgo.main import main
from deepgo.train import train
from deepgo.prepare import prepare

import deepgo.datasets as datasets
import deepgo.models as models
import deepgo.utils as utils
import deepgo.transforms as transforms
import deepgo.models as models
