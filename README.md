
# DifferNet on DAGM Dataset

This repository contains the implementation for the project "Same Same But DifferNet: Semi-Supervised Defect Detection with Normalizing Flows on DAGM Dataset" by Zhaleh Keshavarzi as part of the Deep Learning and Machine Learning course.

## Project Overview

This project involves training the Same Same But DifferNet model for semi-supervised defect detection using normalizing flows on the DAGM dataset. The DAGM dataset, originally created for a competition, contains synthetic images designed for defect detection on textured surfaces.

## Getting Started

### Prerequisites

You will need Python 3.6 and the packages specified in `requirements.txt`. It is recommended to set up a virtual environment using pip and install the packages there.

To install the required packages, run:

```bash
$ pip install -r requirements.txt
```

### Configuration and Running

Configurations related to data, model, training, and visualization can be modified in `config.py`. The default settings will run training using parameters specified for this project on the DAGM dataset.

To start the training, run:

```bash
$ python main.py
```

If training on the DAGM data does not lead to the expected results, such as an AUROC of 1.0 on simpler datasets, there might be an issue. The loss function may produce negative values, which is normal since it reflects the negative log likelihood.

## Results

The results of our experiments with the DAGM dataset show varying performance across different classes, with AUROC scores ranging from 0.6415 to 0.9897. Detailed results and visualizations of anomaly localization can be found in the report.

## Credits

This project builds on the code of the FrEIA framework for the implementation of normalizing flows. Refer to their documentation for more details.

## Citation

If this work contributes to your research, please cite:

```
@inproceedings{RudWan2021,
  author = {Marco Rudolph and Bastian Wandt and Bodo Rosenhahn},
  title = {Same Same But DifferNet: Semi-Supervised Defect Detection with Normalizing Flows},
  booktitle = {Winter Conference on Applications of Computer Vision (WACV)},
  year = {2021},
  month = jan
}
```

## License

This project is licensed under the MIT License.

