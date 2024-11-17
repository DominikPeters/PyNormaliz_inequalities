from setuptools import setup, find_packages

setup(
    name='PyNormaliz_inequalities',
    version='0.1.0',
    description='A Python package for interacting with PyNormaliz using inequalities.',
    author='Dominik Peters',
    author_email='mail@dominik-peters.de',
    license='MIT',
    packages=find_packages(),
    install_requires=[
        'PyNormaliz',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
    ],
)
