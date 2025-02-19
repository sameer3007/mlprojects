from setuptools import find_packages,setup

def get_requirements(file_path):
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace('\n','') for req in requirements]
        if '-e .' in requirements:
            requirements.remove('-e .')
    return requirements


setup(name='mlprojects',version='0.0.1',author='sameer',author_email='sheikhsameer30072000@gmail.com'
      ,packages=find_packages(),install_requires=get_requirements('requirements.txt'))