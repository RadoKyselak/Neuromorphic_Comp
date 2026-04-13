# Neuromorphic_Computing
## Overview
Neuromorphic computing is a hardware and software paradigm inspired by the structure of the human brain. This project benchmarks two neural network approaches on the MNIST handwritten-digit dataset:
- **SNN (Spiking Neural Network)** — a biologically inspired model that communicates via discrete spikes, implemented with `torch`, `torchvision`, and `spikingjelly`.
- **CNN (Convolutional Neural Network)** — a standard deep-learning model, implemented with `keras` (TensorFlow backend), `numpy`, and `scikit-learn`.
The goal is to compare the accuracy and run-time efficiency of SNNs vs CNNs for empirical research purposes.
## Requirements
- Python 3.8 or higher
## Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/RadoKyselak/Neuromorphic_Comp.git
   cd Neuromorphic_Comp
   ```
2. Create a virtual environment and activate it:
   ```sh
   python -m venv venv
   venv\Scripts\activate  # On Mac use `source venv/bin/activate`
   ```
3. Install the required packages:
   ```sh
   pip install -r requirements.txt
   ```
## Usage
To run the SNN benchmark:
```sh
python "SNN benchmark.py"
```
To run the CNN benchmark:
```sh
python "CNN benchmark.py"
```
Both scripts automatically download the MNIST dataset on first run. The SNN script prints the start/end time and training loss, then reports test accuracy. The CNN script trains for 4 epochs and reports final accuracy.
## License
This project is licensed under the Apache License. See the [LICENSE](LICENSE) file for details.
## Contact
If you have any questions or feedback, please open an [issue](https://github.com/RadoKyselak/Neuromorphic_Comp/issues).
