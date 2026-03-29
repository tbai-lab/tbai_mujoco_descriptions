import argparse

from . import get_mjcf_path

parser = argparse.ArgumentParser()
parser.add_argument("robot")
print(get_mjcf_path(parser.parse_args().robot))
