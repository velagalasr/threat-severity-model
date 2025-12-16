from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="threat-severity-model",
    version="1.0.0",
    author="Senior ML Engineering Team",
    author_email="ml-team@company.com",
    description="Production-grade threat severity & risk scoring ML system using NSL-KDD dataset",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/company/threat-severity-model",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Topic :: Security",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "black==23.9.1",
            "flake8==6.1.0",
            "mypy==1.5.1",
            "isort==5.12.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "threat-train=scripts.train:main",
            "threat-evaluate=scripts.evaluate:main",
            "threat-serve=scripts.serve:main",
            "threat-monitor=scripts.monitor:main",
        ],
    },
)
