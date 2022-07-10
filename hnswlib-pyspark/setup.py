from setuptools import setup, find_packages
try:
    from setupext_janitor import janitor
    CleanCommand = janitor.CleanCommand
except ImportError:
    CleanCommand = None

cmd_classes = {}
if CleanCommand is not None:
    cmd_classes['clean'] = CleanCommand

setup(
    name="pyspark_hnsw",
    url="https://github.com/jelmerk/hnswlib/tree/master/hnswlib-pyspark",
    version="1.1.0",
    zip_safe=True,
    packages=find_packages(exclude=['tests']),
    extras_require={
        'dev': ['findspark', 'pytest'],
        'test': ['findspark', 'pytest'],
    },
    setup_requires=['setupext_janitor'],
    cmdclass=cmd_classes,
    entry_points={
        # normal parameters, ie. console_scripts[]
        'distutils.commands': [
            ' clean = setupext_janitor.janitor:CleanCommand']
    }
)