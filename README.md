<br/>
<p align="center">
  <h3 align="center">Allergic Rhinitis</h3>

  <p align="center">
    DL Model for Allergic Rhinitis Prediction
    <br/>
    <br/>
    <a href="https://github.com/bluesaiyancodes/Allergic-Rhinitis/issues">Report Bug</a>
    .
    <a href="https://github.com/bluesaiyancodes/Allergic-Rhinitis/issues">Request Feature</a>
  </p>
</p>

<div align="center">

![Downloads](https://img.shields.io/github/downloads/bluesaiyancodes/Allergic-Rhinitis/total) ![Contributors](https://img.shields.io/github/contributors/bluesaiyancodes/Allergic-Rhinitis?color=dark-green) ![Issues](https://img.shields.io/github/issues/bluesaiyancodes/Allergic-Rhinitis) ![License](https://img.shields.io/github/license/bluesaiyancodes/Allergic-Rhinitis)  ![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white)

</div>

## Table Of Contents

* [About the Project](#about-the-project)
* [Built With](#built-with)
* [Getting Started](#getting-started)
  * [Prerequisites](#prerequisites)
  * [Installation](#installation)
* [Usage](#usage)
* [Roadmap](#roadmap)
* [Contributing](#contributing)
* [License](#license)
* [Authors](#authors)
* [Acknowledgements](#acknowledgements)

## About The Project

![Screen Shot](images/screenshot.png)


This is a Deep Learning Model to predict Allergic Rhinitis based on nose endoscopy images.

The data for this experiment was obtained from a hospital in Seoul, Korea.

One of the major problems faced in this experiment is the lack of data. We only received <b>90</b> images from the hospital. Morover, many of these images were contaminated withnoise and uneven illumination.

Performed various image preprocessing to address the issues which resulted in a dataset that we would work with.


## Built With

We used <b>transfer learning</b> as the amount of images were very short. We tried various base models and among which <i>InceptionNet </i> and <i>XceptionNet</i> seemed to work well.

<b>Data Augmentation</b> was used with <i>zoom, pixel shift, shear and image flips</i> to generate additional data for training the model.

We found that <b>Focal loss</b> worked really well with our model dataset.
We selected <b>Siren</b> as the activation for our model.

We used <b>weightedLoss</b> and <b>learningDecay</b> along with <b>earlyStop</b> for the model callbacks.

This project consists <b>5Fold Cross Validation</b> and <b>LeaveOneOut Cross Validation</b> methods. Additionally <b>Voting</b> can be used for testing to further analyse the model performance. 


<b><i>Various Models, Loss Funtions, Activation Functions are already present int he codebase to play and tinker with.</i></b>

* []()
* []()

## Getting Started

This Project is still under development. We will keep updating it regularly.

### Prerequisites

* Import and install the packages manually. We will add a list of packages later.

* Run the python file named [pyFile[automation].py](pyFile[automation].py) for automated run. <b>Go through the code first</b>

* The main file contraining various methods are present in the [pyFile.py](pyFile.py)<b>Changes in the code can be performed here</b>

### Installation

To be updated. ^^

## Usage

Examples and our work results

## Roadmap

See the [open issues](https://github.com/bluesaiyancodes/Allergic-Rhinitis/issues) for a list of proposed features (and known issues).

## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.
* If you have suggestions for adding or removing projects, feel free to [open an issue](https://github.com/bluesaiyancodes/Allergic-Rhinitis/issues/new) to discuss it, or directly create a pull request after you edit the *README.md* file with necessary changes.
* Please make sure you check your spelling and grammar.
* Create individual PR for each suggestion.
* Please also read through the [Code Of Conduct](https://github.com/bluesaiyancodes/Allergic-Rhinitis/blob/main/CODE_OF_CONDUCT.md) before posting your first idea as well.

### Creating A Pull Request

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

Distributed under the MIT License. See [LICENSE](https://github.com/bluesaiyancodes/Allergic-Rhinitis/blob/master/LICENSE.md) for more information.

## Authors

* [**Bishal Ranjan Swain**](https://bluesaiyancodes.github.io/) - *PhD Candidate at Kumoh National Insitute of Technology* - *Project Lead*

## Acknowledgements

* [Prof. Jaepil Ko](http://cvpr.kumoh.ac.kr/#team)
* [Minhae Kang](http://cvpr.kumoh.ac.kr/#team)

