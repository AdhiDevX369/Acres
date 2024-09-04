from setuptools import setup, find_packages

setup(
    name='yield_per_acre_analysis',
    version='0.1.0',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
        'matplotlib',
        'seaborn',
        'jupyter',
        'joblib',
        'category_encoders',
        'flask'
    ],
    entry_points={
        'console_scripts': [
            'run-analysis=src.investment_analysis:main',  
        ],
    },
    description='A project to analyze yield per acre based on various investments.',
    author='Adithya Bandara',
)
