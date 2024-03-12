from setuptools import setup
from version import __version__

setup(
    name='lip-link',
    author='Vlad Zinca',
    author_email='vlad.zinca@protonmail.com',
    url='https://github.com/vladzinca/lip-link',
    version=__version__,
    install_requires=[
        'pre-commit>=3.2.1',
    ],
    description='Lip Link',
    keywords=['lip', 'link']
    )
