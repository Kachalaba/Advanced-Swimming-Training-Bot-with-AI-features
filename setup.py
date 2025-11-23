from setuptools import setup, find_packages

setup(
    name="sprint-bot",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "opencv-python>=4.8.0",
        "ultralytics>=8.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "fpdf2>=2.7.0",
        "numpy>=1.24.0",
    ],
)
