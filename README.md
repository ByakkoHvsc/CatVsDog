<div align="center">
  <img src="Cat_Vs_Dog.jpg" alt="Dogs vs Cats" style="width: 250px;"/>
    <h1>Dogs v/s Cats Classification
</h1>
</div>

This repository contains a Convolutional Neural Network (CNN) model for classifying images of dogs and cats using TensorFlow and Keras. The dataset used is from [Kaggle](https://www.kaggle.com/datasets/salader/dogs-vs-cats).

## Dataset

The dataset consists of 25,000 images of dogs and cats, with 20,000 images for training and 5,000 images for validation.

## Setup

1. **Clone the repository**:
    ```bash
    git clone https://github.com/ByakkoHvsc/dogs-vs-cats.git
    cd dogs-vs-cats
    ```

2. **Install required libraries**:
    ```bash
    pip install tensorflow keras matplotlib kaggle
    ```

3. **Download the dataset from Kaggle**:
    - Ensure you have the `kaggle.json` API key file ( included in the repo ).
    - Place `kaggle.json` in `~/.kaggle/`:
      ```bash
      mkdir -p ~/.kaggle
      cp kaggle.json ~/.kaggle/
      chmod 600 ~/.kaggle/kaggle.json
      ```
    - Download the dataset:
      ```bash
      kaggle datasets download -d salader/dogs-vs-cats
      unzip dogs-vs-cats.zip -d /content
      ```

## Model Architecture

The CNN model is built using Keras' Sequential API and consists of the following layers:
- Convolutional Layers
- Batch Normalization Layers
- Max Pooling Layers
- Flatten Layer
- Dense Layers
- Dropout Layers

## Training

The model is trained for 10 epochs with a batch size of 16. The data augmentation techniques used include rescaling, shearing, zooming, and horizontal flipping.

## Results

The training and validation accuracy and loss are plotted using Matplotlib.

## Using Google Colab for GPU-Based Training

Google Colab provides free access to GPUs, which can significantly speed up the training process.

### Steps to Enable GPU in Google Colab:

1. **Open Google Colab**: Go to [Google Colab](https://colab.research.google.com/).
2. **Open a new notebook**: Click on "File" > "New Notebook".
3. **Enable GPU**:
    - Click on "Runtime" in the menu.
    - Select "Change runtime type".
    - In the pop-up window, select "GPU" from the "Hardware accelerator" drop-down menu.
    - Click "Save".

### Limitations of Google Colab:

- **Session Time Limit**: Colab has a maximum session length of 12 hours, after which the session will reset.
- **Idle Timeout**: If your Colab notebook remains idle for too long, the session may disconnect.
- **Resource Sharing**: The GPU resources are shared among users, so the performance might vary.

### Best Practices:

- **Save your progress regularly**: Use Google Drive to save your model checkpoints and data periodically.
- **Use smaller batch sizes**: This can help fit your training process within the session limits.

### Example of Using Google Colab:

1. **Mount Google Drive**: 
    ```python
    from google.colab import drive
    drive.mount('/content/drive')
    ```

2. **Set up Kaggle API**:
    ```python
    !mkdir -p ~/.kaggle
    !cp /content/drive/MyDrive/kaggle.json ~/.kaggle/
    !chmod 600 ~/.kaggle/kaggle.json
    ```

3. **Download and unzip the dataset**:
    ```python
    !kaggle datasets download -d salader/dogs-vs-cats
    !unzip dogs-vs-cats.zip -d /content
    ```

For more detailed instructions, refer to the [TensorFlow GPU guide](https://www.tensorflow.org/install/gpu).

## GPU vs. CPU

Training deep learning models on a GPU can significantly reduce training time compared to a CPU. To enable GPU support in TensorFlow, you can use the following setup:

### macOS Users

1. **Install Homebrew**: If you don't have Homebrew installed, you can install it by running:
    ```bash
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    ```

2. **Install Miniforge**: Miniforge is a minimal conda installer that can be used to install TensorFlow with GPU support on macOS.
    ```bash
    brew install miniforge
    ```

3. **Create a new conda environment**:
    ```bash
    conda create -n tf-gpu tensorflow-macos tensorflow-metal
    conda activate tf-gpu
    ```

4. **Verify GPU availability**:
    ```python
    import tensorflow as tf
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    ```

### Windows Users

1. **Install CUDA and cuDNN**: Download and install the appropriate versions of [CUDA](https://developer.nvidia.com/cuda-downloads) and [cuDNN](https://developer.nvidia.com/cudnn).

2. **Set Environment Variables**:
    - Add the CUDA and cuDNN bin directories to your system's PATH variable.
    - Example paths:
      ```plaintext
      C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin
      C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\libnvvp
      C:\tools\cuda\bin
      ```

3. **Install TensorFlow with GPU support**:
    ```bash
    pip install tensorflow-gpu
    ```

4. **Verify GPU availability**:
    ```python
    import tensorflow as tf
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    ```

For more detailed instructions, refer to the [TensorFlow GPU guide](https://www.tensorflow.org/install/gpu).
