from setuptools import setup, find_packages
def get_requirements(file_path="requirements.txt"):
    """
    This function returns a list of requirements from the requirements.txt file.
    """
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]
        if "-e ." in requirements:
            requirements.remove("-e .")
    return requirements
setup(
    name="mlproject",
    version="0.0.1",
    author = "Huan",
    author_email="thuhiendinhhuan@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements(),

)