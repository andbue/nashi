from setuptools import setup

setup(
    name='nashi',
    packages=['nashi'],
    include_package_data=True,
    install_requires=[
        'Flask',
        'sqlalchemy',
        'flask_security',
        'bcrypt',
        'flask_bootstrap',
        'flask_nav',
        'celery',
        'redis',
        'lxml',
        'pillow',
        'kraken',
        'scikit-image'
    ],
)
