from pathlib import Path
import json
import os
import shutil
import tempfile
import unittest

import numpy as np

from iblatlas.regions import BrainRegions
from iblbrainviewer.api import DEFAULT_VOLUME_SHAPE, FeatureUploader, decode_array


def mock_volume(radius):
    # Create a ball volume.
    shape = DEFAULT_VOLUME_SHAPE
    arr = np.zeros(shape, dtype=np.float32)
    center = (shape[0] // 2, shape[1] // 2, shape[2] // 2)
    i, j, k = np.ogrid[:shape[0], :shape[1], :shape[2]]
    distance = np.sqrt((i - center[0])**2 + (j - center[1])**2 + (k - center[2])**2)
    arr[distance <= radius] = 1.0
    return arr, distance


# @unittest.SkipTest
class TestApp(unittest.TestCase):

    def setUp(self):
        # Bucket authentication token for tests.
        self.token = 'bb77d7eb-509b-4ed2-9df6-9907c3cd6ab9'

        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_client(self):
        bucket_uuid = f'my_bucket_{self.token}'
        fname = 'newfeatures'

        acronyms = ['CP', 'SUB']
        values = [42, 420]
        tree = {'dir': {'my custom features': fname}}

        # Create or load the bucket.
        up = FeatureUploader(bucket_uuid, tree=tree, token=self.token)

        # Create the features.
        if not up.features_exist(fname):
            up.create_features(fname, acronyms, values, hemisphere='left')

        # Patch the bucket metadata.
        tree['duplicate features'] = fname
        up.patch_bucket(tree=tree)

        # List all features in the bucket.
        print(up.list_features())

        # Retrieve one feature.
        features = up.get_features(fname)
        print(features)

        # Patch the features.
        values[1] = 10
        up.patch_features(fname, acronyms, values, hemisphere='left')

        # up.download_features(fname, 'downloaded_feature_file.json')

        # Delete the bucket
        # up.delete_bucket()

    def test_client_2(self):

        br = BrainRegions()

        n = 300
        mapping = 'beryl'
        fname1 = 'fet1'
        fname2 = 'fet2'
        bucket = 'test_bucket'
        tree = {'dir': {'custom features 1': fname1}, 'custom features 2': fname2}

        # Beryl regions.
        acronyms = np.unique(br.acronym[br.mappings[mapping.title()]])[:n]
        n = len(acronyms)
        values1 = np.random.randn(n)
        values2 = np.random.randn(n)
        assert len(acronyms) == len(values1)
        assert len(acronyms) == len(values2)

        # Create or load the bucket.
        up = FeatureUploader(bucket, tree=tree, token=self.token)

        # Create the features.
        if not up.features_exist(fname1):
            up.create_features(fname1, acronyms, values1, hemisphere='left')
        if not up.features_exist(fname2):
            up.create_features(fname2, acronyms, values2, hemisphere='left')

        url = up.get_buckets_url([bucket])
        print(url)

    def test_client_volume(self):

        fname = 'myvolume'
        bucket = 'test_bucket'
        tree = {'my test volume': fname}
        radius = 100

        # Create or load the bucket.
        up = FeatureUploader(bucket, tree=tree, token=self.token)

        # Create a ball volume.
        arr, distance = mock_volume(radius)

        # Create the features.
        desc = "this is my volume"
        if not up.features_exist(fname):
            up.upload_volume(fname, arr, short_desc=desc)

        # Retrieve one feature.
        features = up.get_features(fname)
        self.assertTrue(features['feature_data']['volume'])
        # self.assertEqual(features['short_desc'], desc)

        # Patch the features.
        arr[distance <= radius] = 2.0
        up.upload_volume(fname, arr, patch=True)

    def test_client_dots(self):

        bucket = 'test_bucket'
        fname = 'mydots'
        # tree = {'my test dots': fname}

        # Create or load the bucket.
        up = FeatureUploader(bucket, token=self.token)

        # Create the features.
        short_desc = "these are my dots"
        n = 10000
        xyz = np.random.normal(scale=1e-3, size=(n, 3)).astype(np.float32)
        values = np.random.uniform(low=0, high=1, size=(n,)).astype(np.float32)
        up.upload_dots(fname, xyz, values, short_desc=short_desc, patch=up.features_exist(fname))

        # Retrieve one feature.
        features = up.get_features(fname)
        self.assertTrue(features['feature_data']['volume'])
        # self.assertEqual(features['short_desc'], short_desc)

        xyz = decode_array(features['feature_data']['xyz'])
        values = decode_array(features['feature_data']['values'])

        self.assertEqual(xyz.shape, (n, 3))
        self.assertEqual(values.shape, (n,))

    def test_local(self):
        # Local feature uploader
        up = FeatureUploader()

        # Features.
        fname = 'fet'
        acronyms = ['CP', 'SUB']
        values = [42, 420]
        up.local_features(fname, acronyms, values, hemisphere='left', output_dir=self.temp_dir)

        with open(self.temp_dir / f'{fname}.json', 'r') as f:
            fet = json.load(f)
            self.assertEqual(fet['feature_data']['mappings']['allen']['data']['1863']['mean'], 420)

        # Volume.
        fname = 'vol'
        radius = 100
        arr, _ = mock_volume(radius)
        up.local_volume(fname, arr, output_dir=self.temp_dir)

        with open(self.temp_dir / f'{fname}.json', 'r') as f:
            fet = json.load(f)
            self.assertTrue(len(fet['feature_data']['volume']) > 65536)

        # Dots.
        fname = 'dots'
        n = 10000
        xyz = np.random.normal(scale=1e-3, size=(n, 3)).astype(np.float32)
        values = np.random.uniform(low=0, high=1, size=(n,)).astype(np.float32)
        up.local_dots(fname, xyz, values, output_dir=self.temp_dir)

        with open(self.temp_dir / f'{fname}.json', 'r') as f:
            fet = json.load(f)
            self.assertTrue(len(fet['feature_data']['volume']) > 65536)
            self.assertTrue(len(fet['feature_data']['xyz']) > 1024)
            self.assertTrue(len(fet['feature_data']['values']) > 1024)
