#!/usr/bin/env python3
import sys
import os
import io
import argparse
import uuid
import base64
import cv2
import logging
import time
import json
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from PIL import Image
from flask import Flask, request, jsonify, make_response
from waitress import serve


script_dir = os.path.dirname(os.path.abspath(__file__))
LOG_LEVEL = logging.DEBUG
TMP_PATH = '/tmp/upscaler'
GPU_ID = 0
MODELS_PATH = f'{script_dir}/ESRGAN/models'
GFPGAN_MODEL_PATH = f'{script_dir}/GFPGAN/models/GFPGANv1.3.pth'
log_path = ''

# Mac does not have permission to /var/log for example
if sys.platform == 'linux':
    log_path = '/var/log/'

logging.basicConfig(
    filename=f'{log_path}upscaler.log',
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=LOG_LEVEL
)

logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))


class Timer:
    def __init__(self):
        self.start = time.time()

    def restart(self):
        self.start = time.time()

    def get_elapsed_time(self):
        end = time.time()
        return round(end - self.start, 1)


def get_args():
    parser = argparse.ArgumentParser(
        description='Upscaler REST API'
    )

    parser.add_argument(
        '-p', '--port',
        help='Port to listen on',
        type=int,
        default=8090
    )

    parser.add_argument(
        '-H', '--host',
        help='Host to bind to',
        default='0.0.0.0'
    )

    return parser.parse_args()


def upscale(
        source_image_path,
        image_extension,
        model_name='RealESRGAN_x4plus',
        outscale=4,
        face_enhance=False,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        denoise_strength=0.5,
        fp32=False,
):
    """
    model_name options:
        - RealESRGAN_x4plus
        - RealESRNet_x4plus
        - RealESRGAN_x4plus_anime_6B
        - RealESRGAN_x2plus
        - realesr-animevideov3
        - realesr-general-x4v3

    image_extension: .jpg or .png

    outscale: The final upsampling scale of the image

    face_enhance: Whether or not to enhance the face

    tile: Tile size, 0 for no tile during testing

    tile_pad: Tile padding (default = 10)

    pre_pad: Pre padding size at each border

    denoise_strength: 0 for weak denoise (keep noise)
                      1 for strong denoise ability
                      Only used for the realesr-general-x4v3 model
    """

    # determine models according to model names
    model_name = model_name.split('.')[0]

    if image_extension == '.jpg':
        image_format = 'JPEG'
    elif image_extension == '.png':
        image_format = 'PNG'
    else:
        raise ValueError(f'Unsupported image type, must be either JPEG or PNG')

    if model_name == 'RealESRGAN_x4plus':  # x4 RRDBNet model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
    elif model_name == 'RealESRNet_x4plus':  # x4 RRDBNet model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
    elif model_name == 'RealESRGAN_x4plus_anime_6B':  # x4 RRDBNet model with 6 blocks
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
        netscale = 4
    elif model_name == 'RealESRGAN_x2plus':  # x2 RRDBNet model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        netscale = 2
    # TODO: Implement these
    # elif model_name == 'realesr-animevideov3':  # x4 VGG-style model (XS size)
    #     model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu')
    #     netscale = 4
    #     file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth']
    # elif model_name == 'realesr-general-x4v3':  # x4 VGG-style model (S size)
    #     model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
    #     netscale = 4
    #     file_url = [
    #         'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth',
    #         'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth'
    #     ]
    elif model_name == '4x-UltraSharp':  # x4 RRDBNet model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
    elif model_name == 'lollypop':  # x4 RRDBNet model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
    else:
        raise ValueError(f'Unsupported model: {model_name}')

    # determine model paths
    model_path = os.path.join(MODELS_PATH, model_name + '.pth')

    if not os.path.isfile(model_path):
        raise Exception(f'Could not find model: {model_path}')

    # use dni to control the denoise strength
    dni_weight = None
    # if model_name == 'realesr-general-x4v3' and denoise_strength != 1:
    #     wdn_model_path = model_path.replace('realesr-general-x4v3', 'realesr-general-wdn-x4v3')
    #     model_path = [model_path, wdn_model_path]
    #     dni_weight = [denoise_strength, 1 - denoise_strength]

    upsampler = RealESRGANer(
        scale=netscale,
        model_path=model_path,
        dni_weight=dni_weight,
        model=model,
        tile=tile,
        tile_pad=tile_pad,
        pre_pad=pre_pad,
        half=not fp32,
        gpu_id=GPU_ID
    )

    if face_enhance:  # Use GFPGAN for face enhancement
        from gfpgan import GFPGANer
        face_enhancer = GFPGANer(
            model_path=GFPGAN_MODEL_PATH,
            upscale=outscale,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=upsampler
        )

    img = cv2.imread(source_image_path, cv2.IMREAD_UNCHANGED)

    try:
        if face_enhance:
            _, _, output = face_enhancer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
        else:
            output, _ = upsampler.enhance(img, outscale=outscale)
    except RuntimeError as e:
        raise RuntimeError(e)
    else:
        result_image = Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
        output_buffer = io.BytesIO()
        result_image.save(output_buffer, format=image_format)
        image_data = output_buffer.getvalue()
        return base64.b64encode(image_data).decode('utf-8')


def determine_file_extension(image_data):
    try:
        if image_data.startswith('/9j/'):
            image_extension = '.jpg'
        elif image_data.startswith('iVBORw0Kg'):
            image_extension = '.png'
        else:
            # Default to png if we can't figure out the extension
            image_extension = '.png'
    except Exception as e:
        image_extension = '.png'

    return image_extension


app = Flask(__name__)


@app.errorhandler(400)
def not_found(error):
    return make_response(jsonify(
        {
            'status': 'error',
            'msg': f'Bad Request',
            'detail': str(error)
        }
    ), 400)


@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify(
        {
            'status': 'error',
            'msg': f'{request.url} not found',
            'detail': str(error)
        }
    ), 404)


@app.errorhandler(500)
def internal_server_error(error):
    return make_response(jsonify(
        {
            'status': 'error',
            'msg': 'Internal Server Error',
            'detail': str(error)
        }
    ), 500)


@app.route('/', methods=['GET'])
def ping():
    return make_response(jsonify(
        {
            'status': 'ok'
        }
    ), 200)


@app.route('/upscale', methods=['POST'])
def upscaling_api():
    total_timer = Timer()
    logging.debug('Received upscaling API request')
    payload = request.get_json()

    if not os.path.exists(TMP_PATH):
        logging.debug(f'Creating temporary directory: {TMP_PATH}')
        os.makedirs(TMP_PATH)

    unique_id = uuid.uuid4()
    source_image_data = payload['source_image']

    # Decode the source image data
    source_image = base64.b64decode(source_image_data)
    source_file_extension = determine_file_extension(source_image_data)
    source_image_path = f'{TMP_PATH}/source_{unique_id}{source_file_extension}'

    # Save the source image to disk
    with open(source_image_path, 'wb') as source_file:
        source_file.write(source_image)

    # Set defaults if they are not specified in the payload
    if 'model' not in payload:
        payload['model'] = 'RealESRGAN_x4plus'

    if 'scale' not in payload:
        payload['scale'] = 2

    if 'face_enhance' not in payload:
        payload['face_enhance'] = False

    if 'tile' not in payload:
        payload['tile'] = 0

    if 'tile_pad' not in payload:
        payload['tile_pad'] = 10

    if 'pre_pad' not in payload:
        payload['pre_pad'] = 0

    try:
        logging.debug(f'Model: {payload["model"]}')
        logging.debug(f'Scale: {payload["scale"]}')
        logging.debug(f'Face enhance: {payload["face_enhance"]}')
        logging.debug(f'Tile: {payload["tile"]}')
        logging.debug(f'Tile Pad: {payload["tile_pad"]}')
        logging.debug(f'Pre Pad: {payload["pre_pad"]}')

        result_image = upscale(
            source_image_path,
            source_file_extension,
            payload['model'],
            payload['scale'],
            payload['face_enhance'],
            payload['tile'],
            payload['tile_pad'],
            payload['pre_pad']
        )

        status_code = 200

        response = {
            'status': 'ok',
            'image': result_image
        }
    except Exception as e:
        logging.error(e)

        response = {
            'status': 'error',
            'msg': 'Upscaling failed',
            'detail': str(e)
        }

        status_code = 500

    # Clean up temporary images
    os.remove(source_image_path)

    total_time = total_timer.get_elapsed_time()
    logging.info(f'Total time taken for upscaling API call {total_time} seconds')

    return make_response(jsonify(response), status_code)


if __name__ == '__main__':
    args = get_args()

    serve(
        app,
        host=args.host,
        port=args.port
    )
