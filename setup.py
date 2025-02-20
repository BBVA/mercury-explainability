import os, shutil


# Move tutorials inside mercury.explainability before packaging
if os.path.exists('tutorials'):
	shutil.move('tutorials', 'mercury/explainability/tutorials')


from setuptools import setup, find_packages, Extension

import numpy


setup_args = dict(
	name				 = 'mercury-explainability',
	packages			 = find_packages(include = ['mercury*', 'tutorials*']),
	include_package_data = True,
	package_data		 = {'mypackage': ['tutorials/*', 'tutorials/data/*']},
	ext_modules			 = [Extension('cut_finder', ['mercury/explainability/explainers/_tree_splitters/cut_finder.pyx'],
									  extra_compile_args = ['-fopenmp', '-O2'],
									  extra_link_args = ['-fopenmp'],
									  include_dirs = [numpy.get_include()]
		)
	]

)

setup(**setup_args)
