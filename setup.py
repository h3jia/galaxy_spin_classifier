from setuptools import setup


setup(
    name='galaxy_spin_classifier',
    version='0.1.0.dev1',
    description='CNN classifiers for galaxy spin research.',
    url='https://github.com/HerculesJack/galaxy_spin_classifier',
    author='He Jia and Hongming Zhu',
    maintainer='He Jia',
    maintainer_email='he.jia.phy@gmail.com',
    license='Apache License, Version 2.0',
    python_requires=">=3",
    install_requires=['astropy', 'numpy', 'scipy', 'torch']
)
