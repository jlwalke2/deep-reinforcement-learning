from setuptools import setup, find_packages

setup(
    name='deeprl',
    version='1.0',
    description='Library for performing deep reinforcement learning with Keras.',
    author='Jon Walker',
    author_email='jlwalker@inbox.com',
    packages=find_packages(),
    install_requires=['keras'],

)