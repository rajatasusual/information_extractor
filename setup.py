from setuptools import setup, find_packages

with open("requirements.txt") as f:
    required = f.read().splitlines()[1:]
with open("README.md", "r") as fh:
    long_description = fh.read()
setup(
    name='spacy_coref_rel',
    version='0.1.0',
    author='Rajatasusual',
    description='spacy_coref_rel is a tool that leverages spaCy for coreference resolution and SpanBERT for relation extraction. This project integrates named entity recognition (NER) with relation extraction to identify and analyze relationships between entities in text.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=required,
     classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    entry_points={
        'console_scripts': [
            'example_relations=example_relations:main',
        ],
    },
)
