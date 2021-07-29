from setuptools import setup

setup(
    name="pytorch-bezier-mnist",
    py_modules=["pytorch_bezier_mnist"],
    install_requires=["torch", "torchvision"],
)