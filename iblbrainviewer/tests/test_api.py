from iblbrainviewer.api import FeatureUploader
import unittest


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
