# upscaler-flask-api

Python Flask API for Restoration/Upscaling
powered by [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN).

## Model

The following models are available by default:

* RealESRGAN_x2plus
* RealESRGAN_x4plus
* RealESRNet_x4plus
* RealESRGAN_x4plus_anime_6B

## Installation

### Clone this repository

```bash
git clone https://github.com/ashleykleynhans/upscaler-flask-api.git
cd upscaler-flask-api
```

### Install the required Python dependencies

#### Linux and Mac

```
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
python3 setup.py develop
```

#### Windows

```
python3 -m venv venv
venv\Scripts\activate
pip3 install -r requirements.txt
python3 setup.py develop
```

## Download Real-ESRGAN Models

```bash
mkdir -p ESRGAN/models
cd ESRGAN/models
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth
```

## Download GFPGAN model

```bash
cd ../..
mkdir -p GFPGAN/models
wget https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth
```

## Examples

Refer to the [examples](./examples) provided for getting started
with making calls to the API.
