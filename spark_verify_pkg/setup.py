from setuptools import find_packages, setup

package_name = 'spark_verify_nodes'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jywilson',
    maintainer_email='jywilson@ieee.org',
    description='Phase 1 MyCobot mock ecosystem verification nodes.',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'mock_camera_publisher = spark_verify_nodes.mock_camera_publisher:main',
            'joint_command_dispatcher = spark_verify_nodes.joint_command_dispatcher:main',
        ],
    },
)
