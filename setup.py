from setuptools import setup, Extension
import numpy

setup_args = dict(
    ext_modules = [
        Extension(
            "cut_finder",
            ["mercury/explainability/explainers/_tree_splitters/cut_finder.pyx"],
            extra_compile_args=['-fopenmp'],
            extra_link_args=['-fopenmp'],
            include_dirs=[numpy.get_include()]
        )
    ]
)
setup(**setup_args)