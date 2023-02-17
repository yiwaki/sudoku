# python setup.py build_ext --inplace
# python setup.py build_ext --inplace --compiler=mingw32 (for mingw/windows builds)
from distutils.core import Extension, setup

from numpy.distutils.misc_util import get_numpy_include_dirs

setup(
    package_dir={'': ''},
    packages=[],
    ext_modules=[
        Extension(
            'sudoku',
            sources=['src/wrap_bruteforce.c', 'src/bruteforce.c', 'src/bitmap.c', 'src/matrix.c'],
            include_dirs=[] + get_numpy_include_dirs(),
            library_dirs=[],
            libraries=[],
            extra_compile_args=[],
            extra_link_args=[],
        )
    ],
)
