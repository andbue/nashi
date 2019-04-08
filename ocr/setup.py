from setuptools import setup

with open("../README.md") as f:
    readme = f.read()

setup(
    name='nashi',
    version='0.0.1',
    license='GPL_v3.0',
    author='Andreas Büttner',
    author_email='andreas.buettner@uni-wuerzburg.de',
    description='An OCR client complementing webapp for the transcription of scanned pages',
    url='https://github.com/andbue/nashi',
    long_description=readme,
    long_description_content_type='text/markdown',
    packages=['nashi_ocr'],
    include_package_data=True,
    zip_safe=False,
    python_requires='>=3',
    install_requires=[
        "requests",
        "zipfile",
        "json",
        "gzip",
        "h5py",
        "numpy", 
        "lxml",
        "getpass", 
        'pillow',
        'scikit-image',
        'calamari_ocr'
    ],
    classifiers=(
        'Programming Language :: Python :: 3',
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
    ),
)