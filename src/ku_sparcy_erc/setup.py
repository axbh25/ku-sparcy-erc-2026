from glob import glob
import os
from setuptools import find_packages, setup

package_name = 'ku_sparcy_erc'

setup(
    name=package_name,
    version='0.3.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
         glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'),
         glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='KU SPARCy',
    maintainer_email='noreply@users.noreply.github.com',
    description=(
        'KU SPARCy autonomous library-assistant solution for ERC 2026.'),
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # Frozen Day 1 regression executable.
            'opening_sequence = ku_sparcy_erc.opening_sequence:main',
            # Frozen Day 2 perception/search executable.
            'mission_start = ku_sparcy_erc.mission_start:main',
            # Day 3 competition entry executable.
            'day3_mission = ku_sparcy_erc.day3_mission:main',
        ],
    },
)
