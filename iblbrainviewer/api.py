# ---------------------------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------------------------

import base64
from datetime import datetime
import gzip
from io import BytesIO
import json
from pathlib import Path
import random
import requests
import uuid

import numpy as np

# import ast
# import struct
# from numpy.lib.format import (
#     header_data_from_array_1_0,
#     _write_array_header,
#     read_magic,
#     _check_version,
#     _read_bytes,
# )

from iblatlas.atlas import AllenAtlas
from iblbrainviewer.mappings import RegionMapper


# ---------------------------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------------------------

FEATURES_BASE_URL = "https://atlas.internationalbrainlab.org/"
FEATURES_API_BASE_URL = "https://features.internationalbrainlab.org/api/"

# DEBUG
DEBUG = False
if DEBUG:
    FEATURES_BASE_URL = 'https://localhost:8456/'
    FEATURES_API_BASE_URL = 'https://localhost:5000/api/'

DEFAULT_RES_UM = 25
DEFAULT_VOLUME_SHAPE = (528, 320, 456)


# ---------------------------------------------------------------------------------------------
# Util functions
# ---------------------------------------------------------------------------------------------

def now():
    return datetime.now().isoformat()


def new_token(max_length=None):
    token = str(uuid.UUID(int=random.getrandbits(128)))
    if max_length:
        token = token[:max_length]
    return token


def create_bucket_metadata(
        bucket_uuid, alias=None, short_desc=None, long_desc=None, url=None, tree=None):
    return {
        'uuid': bucket_uuid,
        'alias': alias,
        'url': url,
        'tree': tree,
        'short_desc': short_desc,
        'long_desc': long_desc,
        'token': new_token(),
        'last_access_date': now(),
    }


def make_features(acronyms, values, hemisphere=None, map_nodes=False):
    mapper = RegionMapper(acronyms, values, hemisphere=hemisphere, map_nodes=map_nodes)
    return mapper.map_regions()


def feature_dict(aids, values):

    return {
        'data': {int(aid): {'mean': float(value)} for aid, value in zip(aids, values)},
        'statistics': {
            'mean': {
                'min': values.min(),
                'max': values.max(),
                'mean': values.mean(),
                'median': np.median(values)
            }
        },
    }


def list_buckets():
    param_path = Path.home() / '.ibl' / 'custom_features.json'
    with open(param_path, 'r') as f:
        info = json.load(f)
    return list(info['buckets'].keys())


def renormalize_array(arr, min_max=None):
    # Ensure that the input array has exactly three dimensions
    if arr.ndim != 3:
        raise ValueError("Input array must have exactly 3 dimensions.")

    # Compute the min and max values for the entire array
    min_max = min_max if min_max is not None else (arr.min(), arr.max())
    min_value, max_value = min_max

    # Check if the array is constant (min and max values are the same)
    if min_value == max_value:
        return (np.ones_like(arr) * 127).astype(np.uint8)

    # Normalize the entire array to [0, 255]
    normalized_array = ((arr - min_value) / (max_value - min_value) * 255).astype(np.uint8)

    return (min_value, max_value), normalized_array


def clamp(value, min_value, max_value):
    return min(max(value, min_value), max_value)


def base64_encode(input_string):
    encoded_bytes = base64.b64encode(input_string)
    encoded_string = encoded_bytes.decode('utf-8')
    return encoded_string


def base64_decode(encoded_string):
    decoded_bytes = base64.b64decode(encoded_string.encode('utf-8'))
    return decoded_bytes


def to_npy_gz_bytes(arr, extra=None):
    # Buffer with the NPY format bytes.
    buffer = BytesIO()
    np.save(buffer, arr)
    buffer.seek(0)

    # Buffer with the NPY gzip compressed bytes.
    compressed_data_io = BytesIO()

    # with gzip.open(path, 'wb') as gzip_file:
    with gzip.GzipFile(fileobj=compressed_data_io, mode='wb') as gzip_file:
        gzip_file.write(buffer.read())

        # Adding extra metadata at the end of the gzipped byte buffer.
        # additional_data = np.array([min_value, max_value], dtype=np.float32).tobytes()
        if extra is not None:
            extra = np.array(extra)
            additional_data = extra.astype(np.float32).tobytes()
            assert len(additional_data) == (4 * extra.size)
            gzip_file.write(additional_data)

    # The compressed data is now in compressed_data_io
    return compressed_data_io.getvalue()


def load_npy_gz(path):
    path = Path(path)
    assert '.npy.gz' in str(path)

    # Decompress.
    with gzip.open(path, 'rb') as gzip_file:
        bytes = gzip_file.read()

    # Read the header to get the extra metadata with the min and max value.
    buf = BytesIO(bytes)
    buf.seek(0)

    # NOTE: below is a tentative of adding extra metadata fields in the npy header, but it doesn't
    # work because the standard numpy npy loader checks that there are no extra metadata fields.
    # We want generated npy to be readable b the standard npy loader.

    # _header_size_info = {
    #     (1, 0): ('<H', 'latin1'),
    #     (2, 0): ('<I', 'latin1'),
    #     (3, 0): ('<I', 'utf8'),
    # }
    # version = read_magic(buf)
    # _check_version(version)
    # hinfo = _header_size_info.get(version)
    # if hinfo is None:
    #     raise ValueError("Invalid version {!r}".format(version))
    # hlength_type, encoding = hinfo
    # hlength_str = _read_bytes(buf, struct.calcsize(hlength_type), "array header length")
    # header_length = struct.unpack(hlength_type, hlength_str)[0]
    # header = _read_bytes(buf, header_length, "array header")
    # header = header.decode(encoding)
    # d = ast.literal_eval(header)

    # Load the array normally.
    # buf.seek(0)

    arr = np.load(buf)

    return arr


def encode_array(arr, dtype=np.float32):
    return base64_encode(to_npy_gz_bytes(arr.astype(dtype)))


def decode_array(s):
    bytes = base64_decode(s)
    compressed_data_io = BytesIO(bytes)
    with gzip.GzipFile(fileobj=compressed_data_io, mode='rb') as gzip_file:
        uncompressed = gzip_file.read()
    buf = BytesIO(uncompressed)
    buf.seek(0)
    arr = np.load(buf)
    return arr


# ---------------------------------------------------------------------------------------------
# Feature uploader
# ---------------------------------------------------------------------------------------------

class FeatureUploader:
    def __init__(self, bucket_uuid, short_desc=None, long_desc=None, tree=None, token=None):
        # Go in user dir and search bucket UUID and token
        # If nothing create new ones and save on disk, and create on the server
        # with post request

        assert bucket_uuid

        self.param_path = Path.home() / '.ibl' / 'custom_features.json'
        self.param_path.parent.mkdir(exist_ok=True, parents=True)
        self.bucket_uuid = bucket_uuid

        # Create the param file if it doesn't exist.
        if not self.param_path.exists():
            self._create_empty_params()
        assert self.param_path.exists()

        # Load the param file.
        self.params = self._load_params()

        # Try loading the token associated to the bucket.
        saved_token = self._load_bucket_token(bucket_uuid) or token

        # The token can also be passed in the constructor.
        self.token = saved_token or new_token()

        # If there is no saved token, we assume the bucket does not exist and we create it.
        if not saved_token:
            print(f"Creating new bucket {bucket_uuid}.")

            # Create the bucket metadata.
            metadata = create_bucket_metadata(
                bucket_uuid, short_desc=short_desc, long_desc=long_desc, tree=tree)

            # Create a new bucket on the server.
            self._create_new_bucket(bucket_uuid, metadata=metadata)

            # Save the token in the param file.
            self._save_bucket_token(bucket_uuid, self.token)

        # Update the bucket metadata.
        elif short_desc or long_desc or tree:
            metadata = {}
            if short_desc:
                metadata['short_desc'] = short_desc
            if long_desc:
                metadata['long_desc'] = long_desc
            if tree:
                metadata['tree'] = tree
            try:
                self._patch_bucket(metadata)
            except RuntimeError:
                # HACK: if the patching failed whereas there is a saved token, it means the
                # bucket has been destroyed on the server. We receate it here.
                print(f"Recreating new bucket {bucket_uuid}.")

                # Create the bucket metadata.
                metadata = create_bucket_metadata(
                    bucket_uuid, short_desc=short_desc, long_desc=long_desc, tree=tree)

                # Create a new bucket on the server.
                self._create_new_bucket(bucket_uuid, metadata=metadata)

        assert self.token

    # Internal methods
    # ---------------------------------------------------------------------------------------------

    def _headers(self, token=None):
        return {
            'Authorization': f'Bearer {token or self.token}',
            'Content-Type': 'application/json'
        }

    def _url(self, endpoint):
        if endpoint.startswith('/'):
            endpoint = endpoint[1:]
        return FEATURES_API_BASE_URL + endpoint

    def _post(self, endpoint, data):
        url = self._url(endpoint)
        response = requests.post(url, headers=self._headers(), json=data, verify=not DEBUG)
        if response.status_code != 200:
            raise RuntimeError(response.text)
        return response

    def _patch(self, endpoint, data):
        url = self._url(endpoint)
        response = requests.patch(url, headers=self._headers(), json=data, verify=not DEBUG)
        if response.status_code != 200:
            raise RuntimeError(response.text)
        return response

    def _get(self, endpoint):
        url = self._url(endpoint)
        response = requests.get(url, verify=not DEBUG)
        if response.status_code != 200:
            raise RuntimeError(response.text)
        return response

    def _delete(self, endpoint):
        url = self._url(endpoint)
        response = requests.delete(url, headers=self._headers(), verify=not DEBUG)
        if response.status_code != 200:
            raise RuntimeError(response.text)
        return response

    # Params
    # ---------------------------------------------------------------------------------------------

    def _create_empty_params(self):
        with open(self.param_path, 'w') as f:
            json.dump({'buckets': {}}, f, indent=1)

    def _load_params(self):
        with open(self.param_path, 'r') as f:
            return json.load(f)

    def _save_params(self, params):
        with open(self.param_path, 'w') as f:
            json.dump(params, f, indent=1)

    # Bucket token
    # ---------------------------------------------------------------------------------------------

    def _load_bucket_token(self, bucket_uuid):
        assert self.params
        return self.params.get('buckets', {}).get(
            bucket_uuid, {}).get('token', None)

    def _save_bucket_token(self, bucket_uuid, token):
        params = self.params
        if bucket_uuid not in params['buckets']:
            params['buckets'][bucket_uuid] = {}
        params['buckets'][bucket_uuid]['token'] = token
        self._save_params(params)

    def _delete_bucket_token(self, bucket_uuid):
        params = self.params
        _ = params['buckets'].pop(bucket_uuid)
        self._save_params(params)

    # Global key
    # ---------------------------------------------------------------------------------------------

    def _load_global_key(self):
        assert self.params
        return self.params.get('global_key', None)

    def _save_global_key(self, gk):
        assert self.params
        params = self.params
        params['global_key'] = gk
        self._save_params(params)

    def _prompt_global_key(self):
        return input(
            "Plase copy-paste the global key from the documentation webpage:\n")

    def _get_global_key(self):
        """Global authentication to create new buckets.

        1. If the global key is saved in ~/.ibl/custom_features.json, use it.
        2. Otherwise, prompt it and save it.

        """
        gk = self._load_global_key()
        if not gk:
            gk = self._prompt_global_key()
            self._save_global_key(gk)
        assert gk
        return gk

    # Bucket creation
    # ---------------------------------------------------------------------------------------------

    def _create_new_bucket(self, bucket_uuid, metadata=None):
        # Make a POST request to /api/buckets to create the new bucket.
        # NOTE: need for global key authentication to create a new bucket.
        metadata = metadata or {}
        metadata['token'] = self.token
        data = {'uuid': bucket_uuid, 'metadata': metadata}
        endpoint = '/buckets'
        url = self._url(endpoint)
        gk = self._get_global_key()
        response = requests.post(url, json=data, headers=self._headers(gk), verify=not DEBUG)
        if response.status_code != 200:
            raise RuntimeError(response.text)

    def _patch_bucket(self, metadata):
        # Make a PATCH request to /api/buckets/<uuid> to update the bucket metadata.
        metadata = metadata or {}
        data = {'metadata': metadata}
        endpoint = f'/buckets/{self.bucket_uuid}'
        response = self._patch(endpoint, data)
        if response.status_code != 200:
            raise RuntimeError(response.text)

    def _post_or_patch_features(self, method, fname, acronyms, values,
                                short_desc=None, hemisphere=None, map_nodes=False):

        assert method in ('post', 'patch')
        assert fname
        assert acronyms is not None
        assert values is not None
        assert len(acronyms) == len(values)

        # Prepare the JSON payload.
        data = make_features(acronyms, values, hemisphere=hemisphere, map_nodes=map_nodes)
        payload = {
            'fname': fname,
            'short_desc': short_desc,
            'feature_data': {
                'mappings': {
                    'allen': feature_dict(data['allen']['index'], data['allen']['values']),
                    'beryl': feature_dict(data['beryl']['index'], data['beryl']['values']),
                    'cosmos': feature_dict(data['cosmos']['index'], data['cosmos']['values']),
                }
            }
        }

        # Make a POST request to /api/buckets/<uuid>.
        if method == 'post':
            _ = self._post(f'buckets/{self.bucket_uuid}', payload)
        elif method == 'patch':
            _ = self._patch(f'buckets/{self.bucket_uuid}/{fname}', payload)

    def _post_or_patch_volume(
            self, method, fname, volume,
            xyz=None, values=None,
            short_desc=None, min_max=None):

        assert method in ('post', 'patch')
        assert fname

        assert volume.ndim == 3
        assert volume.shape == DEFAULT_VOLUME_SHAPE

        # Renormalize the volume array if it is not already in uint8
        if volume.dtype == np.uint8:
            volume_8 = volume
            min_max = volume_8.min(), volume_8.max()
        else:
            min_max, volume_8 = renormalize_array(volume, min_max=min_max)

        # Convert the uint8 volume into a npy.gz buffer.
        bytes = to_npy_gz_bytes(volume_8, extra=min_max)
        assert bytes is not None

        volume_b64 = base64_encode(bytes)

        # Prepare the JSON payload.
        payload = {
            'fname': fname,
            'short_desc': short_desc,
            'feature_data': {
                'volume': volume_b64,
            }
        }

        # Optional xyz/values data when using create_dots()
        if xyz is not None and values is not None:
            payload['feature_data']['xyz'] = encode_array(xyz)
            payload['feature_data']['values'] = encode_array(values)

        # Make a POST request to /api/buckets/<uuid>.
        if method == 'post':
            _ = self._post(f'buckets/{self.bucket_uuid}', payload)
        elif method == 'patch':
            _ = self._patch(f'buckets/{self.bucket_uuid}/{fname}', payload)

    def _delete_bucket(self):
        # Make a DELETE request to /api/buckets/<uuid> to delete the bucket
        return self._delete(f'/buckets/{self.bucket_uuid}')

    # Public methods
    # ---------------------------------------------------------------------------------------------

    def get_buckets_url(self, uuids):
        assert uuids
        assert isinstance(uuids, list)
        # NOTE: %2C is a comma encoded
        return f'{FEATURES_BASE_URL}?buckets={"%2C".join(uuids)}&bucket={uuids[0]}'

    def patch_bucket(self, **metadata):
        self._patch_bucket(metadata)

    def delete_bucket(self):
        self._delete_bucket()
        self._delete_bucket_token(self.bucket_uuid)

    def create_features(self, fname, acronyms, values, short_desc=None,
                        hemisphere=None, map_nodes=False):
        """Create new features in the bucket."""
        self._post_or_patch_features(
            'post',
            fname,
            acronyms,
            values,
            short_desc=short_desc,
            hemisphere=hemisphere,
            map_nodes=map_nodes)

    def get_bucket_metadata(self):
        response = self._get(f'buckets/{self.bucket_uuid}')
        return response.json()

    def list_features(self):
        """Return the list of fnames in the bucket."""
        return self.get_bucket_metadata()['features']

    def get_features(self, fname):
        """Retrieve features in the bucket."""
        assert fname
        response = self._get(f'/buckets/{self.bucket_uuid}/{fname}')
        features = response.json()
        return features

    def features_exist(self, fname):
        """Determine weather features exist in the bucket."""
        try:
            self.get_features(fname)
        except RuntimeError:
            return False
        return True

    def patch_features(self, fname, acronyms, values, short_desc=None,
                       hemisphere=None, map_nodes=False):
        """Update existing features in the bucket."""
        self._post_or_patch_features(
            'patch',
            fname,
            acronyms,
            values,
            short_desc=short_desc,
            hemisphere=hemisphere,
            map_nodes=map_nodes)

    def delete_features(self, fname):
        """Delete existing features in the bucket."""
        self._delete(f'/buckets/{self.bucket_uuid}/{fname}')

    def upload_volume(self, fname, volume, min_max=None, short_desc=None, patch=False, **kwargs):
        """Create a new volume in the bucket."""
        self._post_or_patch_volume(
            'post' if not patch else 'patch',
            fname, volume, min_max=min_max, short_desc=short_desc, **kwargs)

    def upload_dots(self, fname, xyz, values, dot_size=3, min_max=None,
                    patch=False, short_desc=None):
        """Create a new volume in the bucket, starting from points."""
        assert fname
        assert xyz.ndim == 2
        assert xyz.shape[0] > 0
        assert xyz.shape[1] == 3
        assert dot_size >= 1

        n = xyz.shape[0]
        assert values.ndim == 1
        assert values.shape == (n,)

        a = AllenAtlas()

        shape = a.bc.nxyz
        # shape is (456, 528, 320)

        shape = (shape[1], shape[2], shape[0])
        # shape is now (528, 320, 456)

        assert shape == DEFAULT_VOLUME_SHAPE

        # Create an empty volume.
        volume = np.zeros(shape, dtype=np.float32)

        # Get the voxel coordinates of the dots.
        i, j, k = a.bc.xyz2i(xyz, mode='clip').T

        a, b, c = shape

        s = dot_size // 2
        for u in (-s, 0, +s):
            for v in (-s, 0, +s):
                for w in (-s, 0, +s):
                    # NOTE: j, k, i because of transposition
                    volume[
                        np.clip(j + u, 0, a - 1),
                        np.clip(k + v, 0, b - 1),
                        np.clip(i + w, 0, c - 1)] = values

        return self.upload_volume(
            fname, volume, min_max=min_max, short_desc=short_desc, patch=patch,
            xyz=xyz, values=values)

    def delete_volume(self, fname):
        """Delete existing volume in the bucket."""
        self._delete(f'/buckets/{self.bucket_uuid}/{fname}')
