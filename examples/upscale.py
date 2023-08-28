#!/usr/bin/env python3
import io
import json
import uuid
import requests
import base64
from PIL import Image

URL = 'http://127.0.0.1:8090/upscale'

SOURCE_IMAGE = '../data/src_upscale.jpeg'


def encode_image_to_base64(image_path):
    with open(image_path, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def save_result_image(resp_json):
    img = Image.open(io.BytesIO(base64.b64decode(resp_json['output']['image'])))
    output_file = f'{uuid.uuid4()}.jpg'

    with open(output_file, 'wb') as f:
        print(f'Saving image: {output_file}')
        img.save(f, format='JPEG')


if __name__ == '__main__':
    payload = {
        "source_image": encode_image_to_base64(SOURCE_IMAGE),
        "model": "RealESRGAN_x2plus",
        "scale": 2,
        "face_enhance": True
    }

    r = requests.post(
        URL,
        json=payload
    )

    print(f'HTTP status code: {r.status_code}')

    resp_json = r.json()

    if r.status_code == 200:
        img = Image.open(io.BytesIO(base64.b64decode(resp_json['image'])))
        output_file = f'{uuid.uuid4()}.jpg'

        with open(output_file, 'wb') as f:
            print(f'Saving image: {output_file}')
            img.save(f, format='JPEG')
    else:
        print(json.dumps(resp_json, indent=4, default=str))
