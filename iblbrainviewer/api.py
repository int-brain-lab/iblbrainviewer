from datetime import datetime
from pathlib import Path

import json
import numpy as np
import random
import requests
import uuid

from iblbrainviewer.mappings import RegionMapper


FEATURES_BASE_URL = "https://atlas2.internationalbrainlab.org/"
FEATURES_API_BASE_URL = "https://features.internationalbrainlab.org/api/"

# DEBUG
DEBUG = False
if DEBUG:
    FEATURES_BASE_URL = 'https://localhost:8456/'
    FEATURES_API_BASE_URL = 'https://localhost:5000/api/'


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

    def _delete_bucket(self):
        # Make a DELETE request to /api/buckets/<uuid> to delete the bucket
        return self._delete(f'/buckets/{self.bucket_uuid}')

    # Public methods
    # ---------------------------------------------------------------------------------------------

    def _post_or_patch_features(
            self, method, fname, acronyms, values, short_desc=None, hemisphere=None, map_nodes=False):

        assert method in ('post', 'patch')
        assert fname
        # assert mapping
        assert acronyms is not None
        assert values is not None
        assert len(acronyms) == len(values)

        # Prepare the JSON payload.
        data = make_features(acronyms, values, hemisphere=hemisphere, map_nodes=map_nodes)
        # assert 'data' in data
        # assert 'statistics' in data
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

    def create_features(self, fname, acronyms, values, desc=None, hemisphere=None, map_nodes=False):
        """Create new features in the bucket."""
        self._post_or_patch_features(
            'post', fname, acronyms, values, short_desc=desc, hemisphere=hemisphere, map_nodes=map_nodes)

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
        try:
            self.get_features(fname)
        except RuntimeError:
            return False
        return True

    def patch_features(self, fname, acronyms, values, desc=None, hemisphere=None, map_nodes=False):
        """Update existing features in the bucket."""
        self._post_or_patch_features(
            'patch', fname, acronyms, values, short_desc=desc, hemisphere=hemisphere, map_nodes=map_nodes)

    def delete_features(self, fname):
        self._delete(f'/buckets/{self.bucket_uuid}/{fname}')
