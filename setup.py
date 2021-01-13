from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import os
import sys
import numpy
from Cython.Build import cythonize

class build_numpy(build_ext):
    def finalize_options(self):
        build_ext.finalize_options(self)
        __builtins__.__NUMPY_SETUP__ = False
        import numpy
        self.include_dirs.append(numpy.get_include())

def install(gmp):
    description = 'FairCORELS, a modified version of CORELS to build fair and interpretable models'
    long_description = description
    with open('faircorels/README.md') as f:
        long_description = f.read()

    version = '0.92'

    pyx_file = 'faircorels/_corels.pyx'

    source_dir = 'faircorels/src/corels/src/'
    sources = ['utils.cpp', 'rulelib.cpp', 'run.cpp', 'pmap.cpp', 
               'corels.cpp', 'cache.cpp',
               # files under this line are for improved filtering only
               'statistical_parity_improved_pruning.cpp', 'mistral_backtrack.cpp', 'mistral_constraint.cpp',
               'mistral_global.cpp', 'mistral_sat.cpp', 'mistral_search.cpp',
               'mistral_solver.cpp', 'mistral_structure.cpp', 'mistral_variable.cpp'
               ]
    
    for i in range(len(sources)):
        sources[i] = source_dir + sources[i]
    
    sources.append(pyx_file)

    sources.append('faircorels/src/utils.cpp')

    cpp_args = ['-Wall', '-O3', '-std=c++11']
    libraries = []

    if os.name == 'posix':
        libraries.append('m')

    if gmp:
        libraries.append('gmp')
        cpp_args.append('-DGMP')

    if os.name == 'nt':
        cpp_args.append('-D_hypot=hypot')
        if sys.version_info[0] < 3:
            raise Exception("Python 3.x is required on Windows")

    extension = Extension("faircorels._corels", 
                sources = sources,
                libraries = libraries,
                include_dirs = ['faircorels/src/', 'faircorels/src/corels/src', numpy.get_include()],
                language = "c++",
                extra_compile_args = cpp_args)

    extensions = [extension]
    extensions = cythonize(extensions)

    numpy_version = 'numpy'

    if sys.version_info[0] < 3 or sys.version_info[1] < 5:
        numpy_version = 'numpy<=1.16'

    setup(
        name = 'faircorels',
        packages = ['faircorels'],
        ext_modules = extensions,
        version = version,
        author = 'Ulrich Aivodji, Julien Ferry, Sebastien Gambs, Marie-Jose Huguet, Mohamed Siala',
        author_email = 'a.u.matchi@gmail.com, julienferry12@gmail.com',
        description = description,
        long_description = long_description,
        long_description_content_type='text/markdown',
        setup_requires = [numpy_version],
        install_requires = [numpy_version],
        python_requires = '>=2.7',
        url = 'https://github.com/ferryjul/fairCORELS',
        download_url = 'https://github.com/ferryjul/fairCORELS/archive/0.7.tar.gz',
        cmdclass = {'build_ext': build_numpy},
        license = "GNU General Public License v3 (GPLv3)",
        classifiers = [
            "Programming Language :: C++",
            "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
            "Operating System :: OS Independent"
        ]
    )

if __name__ == "__main__":
    try:
        install(True)
    except:
        install(False)
