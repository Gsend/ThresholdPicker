from setuptools import setup, find_packages


setup(
    name='ThresholdPicker',
    version='0.0.0',
    license='MIT',
    author="Gilad Senderovich",
    author_email='giladsnd@gmail.com',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    url='_____ update',
    keywords='Threshold moving optimization',
    install_requires=[
          'numpy', 'pandas',  'matplotlib'
      ],

)