"""
This file belongs to the MultiBodySync code repository.
Author: Erik Wijmans, Jiahui Huang <huang-jh18@mails.tsinghua.edu.cn>
"""


import glob
from pathlib import Path
from torch.utils.cpp_extension import load


_ext_src_root = str(Path(__file__).parent)
_ext_sources = glob.glob("{}/src/*.cpp".format(_ext_src_root)) + glob.glob(
    "{}/src/*.cu".format(_ext_src_root)
)
_ext_headers = glob.glob("{}/include/*".format(_ext_src_root))

pointnet2_ext = load(name='pointnet2_ext',
                     sources=_ext_sources,
                     extra_include_paths=["{}/include".format(_ext_src_root)])
