"""
Fallback setup.py for older pip versions that don't support pyproject.toml
"""

from setuptools import setup, find_packages
import pathlib

# Read the README file
here = pathlib.Path(__file__).parent.resolve()
long_description = (here / "README.md").read_text(encoding="utf-8")

# Read requirements from pyproject.toml equivalent
install_requires = [
    "numpy>=1.21.0",
    "pandas>=1.3.0", 
    "pyyaml>=6.0",
    "python-dotenv>=0.19.0",
    "requests>=2.25.0",
    "aiohttp>=3.8.0",
    "typing-extensions>=4.0.0",
]

extras_require = {
    "dev": [
        "pytest>=7.0.0",
        "pytest-asyncio>=0.21.0",
        "pytest-cov>=4.0.0",
        "black>=22.0.0",
        "flake8>=5.0.0",
        "mypy>=1.0.0",
        "pre-commit>=2.20.0",
    ],
    "llm-providers": [
        "openai>=1.0.0",
        "anthropic>=0.3.0",
        "transformers>=4.20.0",
        "torch>=1.12.0",
        "ollama>=0.1.0",
    ],
    "visualization": [
        "matplotlib>=3.5.0",
        "plotly>=5.0.0",
        "seaborn>=0.11.0",
        "streamlit>=1.25.0",
    ],
    "jupyter": [
        "jupyter>=1.0.0",
        "notebook>=6.4.0",
        "ipywidgets>=7.6.0",
    ],
}

# Add 'full' extra that includes everything
extras_require["full"] = list(set().union(*extras_require.values()))

setup(
    name="aegis-ai",
    version="1.0.0",
    description="AI Evaluation and Guard Intelligence System - A comprehensive framework for evaluating AI alignment risks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/aegis-ai/aegis",
    author="AEGIS Development Team",
    author_email="contact@aegis-ai.org",
    license="MIT",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Security",
        "Topic :: Software Development :: Testing",
    ],
    keywords="ai-safety red-teaming alignment evaluation llm security adversarial-testing",
    packages=find_packages(exclude=["tests*", "examples*"]),
    python_requires=">=3.8",
    install_requires=install_requires,
    extras_require=extras_require,
    package_data={
        "aegis": [
            "config/*.yaml",
            "config/*.yml",
            "evaluation/targets/*.py",
            "providers/*.py",
        ],
    },
    entry_points={
        "console_scripts": [
            "aegis=aegis.cli:main",
        ],
    },
    project_urls={
        "Homepage": "https://github.com/aegis-ai/aegis",
        "Documentation": "https://aegis-ai.readthedocs.io",
        "Repository": "https://github.com/aegis-ai/aegis",
        "Bug Tracker": "https://github.com/aegis-ai/aegis/issues",
        "Changelog": "https://github.com/aegis-ai/aegis/blob/main/CHANGELOG.md",
    },
)