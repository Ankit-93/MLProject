from setuptools import find_packages, setup
from typing import List

def get_requirements(file_path:str)->List[str]:
    """
    this will return the requirement list
    """
    HYPHEN_DOT_E = '-e .'
    required=[]
    with open(file_path) as f:
        required = f.readlines()

    required =[req.replace('\n','') for req in required]
    if HYPHEN_DOT_E in required:
        required.remove(HYPHEN_DOT_E)
    
    return required




setup(
    name = 'mlproject', author ='Ankit',author_email = 'ankit.dutta0@gmail.com',
    packages=find_packages(),install_requires=get_requirements('requirements.txt')
    )
