from os.path import dirname, realpath
from setuptools import setup, find_packages
from opencood.version import __version__


def _read_requirements_file():
    """Return the elements in requirements_1.txt."""
    req_file_path = '%s/requirements.txt' % dirname(realpath(__file__))
    with open(req_file_path) as f:
        return [line.strip() for line in f]

setup(
    name='V2XGen',
    version=__version__,
    packages=find_packages(),
    url='https://github.com/ucla-mobility/V2V4Real',
    license='MIT',
    author='JiaKai Liu',
    author_email='jacky_ljk@163.com',
    description='v2xgen codebase',
    long_description=open("README.md").read(),
    install_requires=_read_requirements_file(),
)
