import setuptools

with open('./README.md', 'r') as f:
    long_description = f.read()

setuptools.setup(name='mltools',
                 version='0.0.1',
                 author='Riccardo Finotello',
                 author_email='riccardo.finotello@gmail.com',
                 description='Tools for machine learning and data science',
                 long_description=long_description,
                 long_description_content_type='text/markdown',
                 url='https://github.com/thesfinox/ml-dev',
                 packages=setuptools.find_packages(),
                 classifiers=['Programming Language :: Python :: 3',
                              'License :: OSI Approved :: MIT License',
                              'Operating System :: OS Independent',
                              'Topic :: Scientific/Engineering :: Mathematics',
                              'Topic :: Scientific/Engineering :: Physics',
                              'Topic :: Scientific/Engineering :: Statistics'
                             ],
                 install_requires=['numpy',
                                   'pandas',
                                   'tensorflow>=2',
                                   'seaborn>=0.11',
                                   'matplotlib',
                                   'statsmodels',
                                   'scikit-learn>0.22',
                                   'scikit-optimize',
                                   'shap'
                                  ],
                 python_requires='>=3.6, <3.9'
                ) 
