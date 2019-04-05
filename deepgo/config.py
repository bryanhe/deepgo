import os
import configparser
import types

FILENAME = None
param = {}
for filename in ["deepgo.cfg",
                 ".deepgo.cfg",
                 os.path.expanduser("~/deepgo.cfg"),
                 os.path.expanduser("~/.deepgo.cfg")]:
    if os.path.isfile(filename):
        FILENAME = filename
        config = configparser.ConfigParser()
        with open(filename, "r") as f:
            config.read_string("[config]\n" + f.read())
            param = config["config"]
        break

config = types.SimpleNamespace(FILENAME = FILENAME,
                               KGS_ROOT = param.get("kgs_root", "data/kgs/"),
                               KGS_PROCESSED_ROOT = param.get("kgs_processed_root", "data/kgs-processed/"))
