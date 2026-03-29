import argparse

from . import get_scene_path

parser = argparse.ArgumentParser()
parser.add_argument("robot")
print(get_scene_path(parser.parse_args().robot))
