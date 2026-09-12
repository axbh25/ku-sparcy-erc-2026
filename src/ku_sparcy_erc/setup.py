from glob import glob
import os
from setuptools import find_packages, setup

package_name = 'ku_sparcy_erc'

setup(
    name=package_name,
    version='0.5.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='KU SPARCy',
    maintainer_email='noreply@users.noreply.github.com',
    description='KU SPARCy autonomous library-assistant solution for ERC 2026.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'opening_sequence = ku_sparcy_erc.opening_sequence:main',
            'mission_start = ku_sparcy_erc.mission_start:main',
            'day3_mission = ku_sparcy_erc.day3_mission:main',
            'day4_mission = ku_sparcy_erc.day4_mission:main',
            'staged_grasp_experiment = ku_sparcy_erc.staged_grasp_experiment:main',
            'home_pose_recorder = ku_sparcy_erc.home_pose_recorder:main',
            'day5_autonomous_pick = ku_sparcy_erc.day5_autonomous_pick:main',
        ],
    },
)
