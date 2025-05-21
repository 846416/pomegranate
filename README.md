**PL-YOLO: a lightweight method for real-time detection of pomegranates**

This repository contains the essential components of our PL-YOLO model, which serves as the backbone of our research on pomegranate fruit detection in complex backgrounds. These resources are integral to our study and are detailed as follows:

 

**Table of Contents**

1. Source Code
2. Dataset
3. Model Architecture
4. Key Improvements
5. Repository Contents

 

**Source Code**

The source code provided offers a comprehensive implementation of our proposed PL-YOLO model, covering the architecture, training, and testing processes. It includes the core code for conducting comparative experiments and ablation studies. This enables other researchers to reproduce our experimental results and verify the model's performance and improvements. The source code also reflects the innovative modifications we have made during the model optimization process.

 

**Dataset**

The dataset is a crucial resource that allows other researchers to validate the model's performance on the same data. This helps in evaluating the reliability and stability of our model in practical applications. However, the link to the dataset is currently unavailable due to possible network issues or problems with the link itself. If you need access to the dataset, please check the validity of the link and try again later. The dataset link is: https://doi.org/10.1016/j.dib.2023.109468. If you encounter any issues with the link, it might be due to network problems or an invalid link. Please ensure the link is correct and try accessing it again.

 

**Model Architecture**

PL-YOLO is an optimized lightweight model specifically designed for the challenging task of pomegranate fruit detection in complex backgrounds. To address these challenges, PL-YOLO incorporates several key improvements into its architecture:

Key Improvements

1. IEFE Module: Accurately captures edge information through a Sobel Conv branch.
2. FPSF Module: Extracts feature at multiple scales using convolutional layers with different dilation rates. This captures information from various sizes and backgrounds within the image.
3. CGAFPN Model: The CGAF effectively captures and utilizes crucial context information to enhance feature representation. Wavelet pooling improves computational efficiency in the sampling process, providing robust support for real-time image processing.
4. DESH Model: Through reparameterization techniques, DEConv is made equivalent to standard convolutions. This significantly reduces the number of parameters, making the model lightweight.

 

**Repository Contents**

This repository provides the complete implementation of PL-YOLO, including YAML files for training the model, training scripts, and testing scripts. These resources are designed to help users reproduce and build upon our results. We hope this repository will facilitate further research and development in the field of pomegranate fruit detection and computer vision in general.

**Academic Translation and Polishing**

This project was completed by the first author under the supervision of the supervisor. All code and models are solely intended for academic exchange and research purposes. Any commercial use (including but not limited to code, weight files, and derivative works) is strictly prohibited. If you use related results in academic research, you can acknowledge the source in your paper and comply with open-source licenses and academic norms.
