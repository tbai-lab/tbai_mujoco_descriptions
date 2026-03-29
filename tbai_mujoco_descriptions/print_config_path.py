import argparse

from . import get_config_path

parser = argparse.ArgumentParser()
parser.add_argument("robot")
print(get_config_path(parser.parse_args().robot))
