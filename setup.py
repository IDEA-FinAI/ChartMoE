from setuptools import setup, find_namespace_packages
import platform

DEPENDENCY_LINKS = []
if platform.system() == "Windows":
    DEPENDENCY_LINKS.append("https://download.pytorch.org/whl/torch_stable.html")


def fetch_requirements(filename):
    with open(filename) as f:
        return [ln.strip() for ln in f.read().split("\n")]


setup(
    name="chartmoe",
    version="0.1.0",
    author="Coobiw",
    description="ChartMoE: Mixture of Expert Connector for Better Chart Understanding",
    keywords="Multimodal Large Language Model (MLLM), Chart Understanding, Mixture of Expert (MoE)",
    license="3-Clause BSD",
    packages=find_namespace_packages(include="chartmoe.*"),
    install_requires=fetch_requirements("requirements.txt"),
    python_requires=">=3.9",
    include_package_data=True,
    dependency_links=DEPENDENCY_LINKS,
    zip_safe=False,
)