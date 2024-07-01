# LipLink

As Machine Learning has been expanding its scope more and more in the last few decades, new kinds of everyday problems begin to benefit from it. This is especially relevant to areas concerning human accesibility, with these new techniques allowing for great improvement.

One such case is lip reading, which represents the ability to understand what is being said by simply looking at one's movement of the mouth during speech.

We propose LipLink, a program which uses machine learning techniques to lip read, as well as the theoretical notions that were required for its creation. LipLink is a complex neural network model built with PyTorch that succeeds in generalizing the mimic of the lips present in videos from the GRID dataset with an accuracy of `88.04%`.

The goal of this project is to set the basis of a more complex system which, in time, may extend its portability to virtual reality and smart devices.

## How to use

The recommended way to run the program is through a Python virtual environment.

I am using Python 3.8.10, but any newer version will do fine.

1.  To create a Python virtual environment, run:
```bash
vlad_@kalpapadapa:$ python -m venv .venv
```

2.  Activate the Python virtual environment:
```bash
vlad_@kalpapadapa:$ source .venv/bin/activate
(.venv) vlad_@kalpapadapa:$
```

You can deactivate it using:
```bash
(.venv) vlad_@kalpapadapa:$ deactivate
vlad_@kalpapadapa:$
```

3.  Install the required packages:
```bash
(.venv) vlad_@kalpapadapa:$ pip install -r requirements.txt
```

4.  Following the installation, a `pip freeze` should look something like this:
```bash
(.venv) vlad_@kalpapadapa:$ pip freeze
certifi==2024.6.2
cfgv==3.4.0
charset-normalizer==3.3.2
click==8.1.7
distlib==0.3.8
exceptiongroup==1.2.1
filelock==3.15.4
fsspec==2024.6.1
identify==2.5.36
idna==3.7
iniconfig==2.0.0
jinja2==3.1.4
joblib==1.4.2
Levenshtein==0.25.1
-e git+git@github.com:vladzinca/lip-link.git@94d4fdc10b64431374d37f1149c9e9f29111620c#egg=lip_link
MarkupSafe==2.1.5
mpmath==1.3.0
networkx==3.1
nltk==3.8.1
nodeenv==1.9.1
numpy==1.24.4
nvidia-cublas-cu12==12.1.3.1
nvidia-cuda-cupti-cu12==12.1.105
nvidia-cuda-nvrtc-cu12==12.1.105
nvidia-cuda-runtime-cu12==12.1.105
nvidia-cudnn-cu12==8.9.2.26
nvidia-cufft-cu12==11.0.2.54
nvidia-curand-cu12==10.3.2.106
nvidia-cusolver-cu12==11.4.5.107
nvidia-cusparse-cu12==12.1.0.106
nvidia-nccl-cu12==2.20.5
nvidia-nvjitlink-cu12==12.5.40
nvidia-nvtx-cu12==12.1.105
opencv-python==4.10.0.84
packaging==24.1
pillow==10.4.0
platformdirs==4.2.2
pluggy==1.5.0
pre-commit==3.5.0
pytest==8.2.2
PyYAML==6.0.1
rapidfuzz==3.9.3
regex==2024.5.15
requests==2.32.3
sympy==1.12.1
textblob==0.18.0.post0
tomli==2.0.1
torch==2.3.1
torchvision==0.18.1
tqdm==4.66.4
triton==2.3.1
typing-extensions==4.12.2
urllib3==2.2.2
virtualenv==20.26.3
```

5.  Afterwards, one can run the different program functionalities using the following command:
```bash
(.venv) vlad_@kalpapadapa:$ python src/main.py [--fetch / --train / --test]
```

One can give the command with any of the three arguments or all at once.

If given:

-  `--fetch` will download and extract the GRID dataset at the current working directory
-  `--train` will train the model using the data obtained with `--fetch`
-  `--test` will test the model trained with `--train`

If running the epoch 100 checkpoint of experiment 2 using the training dataset that was not seen by the model during its training, one would get:

```bash
(.venv) vlad_@kalpapadapa:$ python src/main.py --test
From lip-link-kernel: Initializing testing...
From lip-link-kernel: Process started at "2024-07-01 16:31:48.234489".
From lip-link-kernel: Device set to "cuda".
From lip-link-kernel: The testing will use the GPU.
Predicting for data point 1/100:
Target: "set white in h seven soon".
Prediction: "set white in t seven soon".
================================================================
Predicting for data point 2/100:
Target: "set red by u nine again".
Prediction: "set reed by u nine again".
================================================================

[...]

Predicting for data point 99/100:
Target: "bin blue at l seven soon".
Prediction: "bin blue at v seven soon".
================================================================
Predicting for data point 100/100:
Target: "place green in j nine soon".
Prediction: "place green in k nine soon".
================================================================
Results:
Average word count, per sentence: 6.
Average character count, per sentence: 24.92.
Word error rate, total: 20.5%.
Word error rate, per sentence: 1.23.
Levenshtein distance, total: 11.96%.
Levenshtein distance, characters per sentence: 2.98.
Accuracy, total: 88.04%.
================================================================
From lip-link-kernel: Process ended at "2024-07-01 16:32:15.923646".
From lip-link-kernel: Whole process took 00:00:27.
```

As we can observe, the accuracy is computed to be `88.04%`, as already discussed.
