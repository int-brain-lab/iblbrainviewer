import unittest

import numpy as np

from iblatlas.regions import BrainRegions
from iblbrainviewer.api import FeatureUploader


@unittest.SkipTest
class TestApp(unittest.TestCase):

    def setUp(self):
        # Bucket authentication token for tests.
        self.token = 'bb77d7eb-509b-4ed2-9df6-9907c3cd6ab9'

    def test_client(self):
        bucket_uuid = f'my{self.token}'
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

        # Delete the bucket
        # up.delete_bucket()

    def test_client_2(self):

        br = BrainRegions()

        n = 300
        mapping = 'beryl'
        fname1 = 'fet1'
        fname2 = 'fet2'
        bucket = 'mybucket'
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
            up.create_features(fname1, acronyms, values1, mapping=mapping)
        if not up.features_exist(fname2):
            up.create_features(fname2, acronyms, values2, mapping=mapping)

        url = up.get_buckets_url([bucket])
        print(url)
