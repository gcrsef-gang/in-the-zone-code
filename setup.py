from setuptools import setup


with open("README.md", "r") as f:
    long_description = f.read()


setup(
    name="itz",
    version="0.0.1",
    description="Code for *In the Zone: The effects of zoning regulation changes on urban life*",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/gcrsef-gang/in-the-zone-code",
    install_requires=[
        "folium==0.12.1.post1",
        "matplotlib==3.5.1", 
        "pandas==1.4.1",
        "semopy==2.3.9"
    ]
)