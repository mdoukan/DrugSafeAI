from setuptools import setup, find_packages

setup(
    name="drugsafe_ai",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'flask',
        'xgboost',
        'pandas',
        'numpy',
        'scikit-learn',
        'requests',
        'joblib'
    ]
) 