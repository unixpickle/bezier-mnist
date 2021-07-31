# bezier-mnist

This is a work-in-progress vector version of the MNIST dataset.

# Samples

Here are some samples from the training set. Note that, while these are rasterized, the underlying images can be rendered at any resolution because they are smooth vector graphics.

<img src="samples.png" alt="A grid of sixteen digit images" width="300" height="300">

# Usage

I have already converted all of MNIST to Bezier curves. This dataset can be downloaded at [this page](https://data.aqnichol.com/bezier-mnist/). There are two files: `train.zip` and `test.zip`, each containing a separate `json` file for each digit image.

To load this dataset (and automatically download it), you can use [pytorch-bezier-mnist](pytorch-bezier-mnist) included in this repo.

# Examples

The [examples](examples) directory contains some machine learning examples that use the dataset.
