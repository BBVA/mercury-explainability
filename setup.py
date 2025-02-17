import os, shutil


# Move tutorials inside mercury.explainability before packaging
if os.path.exists('tutorials'):
    shutil.move('tutorials', 'mercury/explainability/tutorials')


from setuptools import setup, find_packages


setup_args = dict(
	name				 = 'mercury-explainability',
	packages			 = find_packages(include = ['mercury*', 'tutorials*']),
	include_package_data = True,
	package_data		 = {'mypackage': ['tutorials/*', 'tutorials/data/*']}
)

setup(**setup_args)
