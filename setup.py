from setuptools import setup
from version import __version__

setup(
    name='lip-link',
    author='Vlad Zinca',
    author_email='vlad.zinca@protonmail.com',
    url='https://github.com/vladzinca/lip-link',
    version=__version__,
    install_requires=[
        'Levenshtein>=0.25.1',
        'numpy>=1.24.4',
        'opencv-python>=4.8.1',
        'pre-commit>=3.2.1',
        'requests>=2.22.0',
        'textblob>=0.18.0',
        'torch>=2.1.2',
        'torchvision>=0.16.2',
    ],
    description='Lip Link',
    keywords=['lip', 'link', 'machine', 'learning']
    )
