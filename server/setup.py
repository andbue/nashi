from setuptools import setup

with open("./README.md") as f:
    readme = f.read()

setup(
    name='nashi',
    version='0.0.45',
    license='GPL_v3.0',
    author='Andreas Büttner',
    author_email='andreas.buettner@uni-wuerzburg.de',
    description='A webapp for the transcription of scanned pages',
    url='https://github.com/andbue/nashi',
    long_description=readme,
    long_description_content_type='text/markdown',
    packages=['nashi'],
    include_package_data=True,
    zip_safe=False,
    entry_points={
        'console_scripts': [
            'nashi-import=nashi.import_from_pagexml:main',
            'nashi-export=nashi.import_from_pagexml:bookexport',
            'nashi-delete=nashi.import_from_pagexml:bookdelete'
        ],
    },
    python_requires='>=3',
    install_requires=[
        'sqlalchemy',
        'flask_security',
        'bcrypt',
        'flask_bootstrap',
        'flask_nav',
        'celery',
        'redis',
        'lxml',
        'pillow',
        'krakensegment',
        'scikit-image',
        'Flask',
        'email_validator'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: JavaScript',
        'Development Status :: 3 - Alpha',
        'Framework :: Flask',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
    ],
)
