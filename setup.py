from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import os
import sys

#from Cython.Build import cythonize


class build_numpy(build_ext):
    def finalize_options(self):
        build_ext.finalize_options(self)
        __builtins__.__NUMPY_SETUP__ = False
        import numpy
        self.include_dirs.append(numpy.get_include())

def install(gmp):
    description = 'FairCORELS, a modified version of CORELS to build fair and interpretable models'
    long_description = description
    with open('fairules/README.txt') as f:
        long_description = f.read()

    version = '1.6'

    pyx_file = 'fairules/_corels.cpp'

    source_dir = 'fairules/src/corels/src/'
    sources = ['utils.cpp', 'rulelib.cpp', 'run.cpp', 'pmap.cpp', 
               'corels.cpp', 'cache.cpp']
    
    for i in range(len(sources)):
        sources[i] = source_dir + sources[i]
    
    sources.append(pyx_file)

    sources.append('fairules/src/utils.cpp')

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

    extension = Extension("fairules._corels", 
                sources = sources,
                libraries = libraries,
                include_dirs = ['fairules/src/', 'fairules/src/corels/src'],
                language = "c++",
                extra_compile_args = cpp_args)

    extensions = [extension]
    #extensions = cythonize(extensions)

    numpy_version = 'numpy'

    if sys.version_info[0] < 3 or sys.version_info[1] < 5:
        numpy_version = 'numpy<=1.16'

    setup(
        name = 'fairules',
        packages = ['fairules'],
        ext_modules = extensions,
        version = version,
        author = 'Elaine Angelino, Nicholas Larus-Stone, Hongyu Yang, Cythnia Rudin, Vassilios Kaxiras, Margo Seltzer, Ulrich Aïvodji, Julien Ferry, Sébastien Gambs, Marie-José Huguet, Mohamed Siala',
        author_email = 'a.u.matchi@gmail.com',
        description = description,
        long_description = long_description,
        setup_requires = [numpy_version],
        install_requires = [numpy_version],
        python_requires = '>=2.7',
        url = 'https://github.com/aivodji/fairules',
        download_url = 'https://github.com/aivodji/fairules/archive/v1.6.tar.gz',
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
