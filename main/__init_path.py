from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp
import sys


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


this_dir = osp.dirname(__file__)

lib_path = osp.join(this_dir, '..', 'lib')
add_path(lib_path)

data_path = osp.join(this_dir, '..', 'data')
add_path(data_path)

smpl_path = osp.join(this_dir, '..', 'smplpytorch')
add_path(smpl_path)

mano_path = osp.join(this_dir, '..', 'manopth')
add_path(mano_path)