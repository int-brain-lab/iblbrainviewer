import numpy as np
import unittest

from iblatlas.regions import BrainRegions
from iblutil.numerical import ismember
from iblbrainviewer.mappings import RegionMapper


class TestNavigateRegions(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.br = BrainRegions()

    def test_allen(self):
        # Test 1
        acronyms = np.array(['MOp1', 'MOs5'])
        values = np.array([1, 2])
        mapper = RegionMapper(acronyms, values, hemisphere='left')
        json_all, json_final = mapper.navigate_regions(acronyms, values, mapping='Allen')
        # 1. json_all to be the same as json_final as both regions are in volume
        assert json_all == json_final
        # 2. json_final and json_all to contain two regions
        assert len(json_final) == len(json_all) == 2
        # 3. MOp1 and MOs5 with the values assigned
        assert json_final['MOp1'] == 1
        assert json_final['MOs5'] == 2

        # Test 2
        acronyms = np.array(['MO', 'MOp5', 'MOp1'])
        values = np.array([1, 2, 3])
        mapper = RegionMapper(acronyms, values, hemisphere='left')
        json_all, json_final = mapper.navigate_regions(acronyms, values, mapping='Allen')
        # 1. json_all to contain additional keys MO, MO1, MO2/3, MO5, MO6a and MO6b compared to json_final as these
        #    regions aren't in the volume
        compare = np.array(['MO', 'MO1', 'MO2/3', 'MO5', 'MO6a', 'MO6b'])
        isin, _ = ismember(np.array(list(json_all.keys())), compare)
        assert np.sum(isin) == len(compare)
        isin, _ = ismember(np.array(list(json_final.keys())), compare)
        assert np.sum(isin) == 0
        # 2. Value of MO to propagate down to MOs and MO1, MO2/3, MO5, MO6a and MO6b
        compare = np.r_[compare, self.br.descendants(self.br.acronym2id('MOs'))['acronym']]
        assert all([json_all[c] == 1 for c in compare])
        # 3. MOp to be assigned np.mean(MOp5, MOp1) and this to be propagated down to MOp2/3, MOp6a, MOp6b
        assert json_all['MOp'] == np.mean(values[1:])
        compare = np.array(['MOp2/3', 'MOp6a', 'MOp6b'])
        assert all([json_final[c] == np.mean(values[1:]) for c in compare])
        # 4. MOp5 and MOp1 will be given their assigned values
        assert json_final['MOp5'] == 2
        assert json_final['MOp1'] == 3

        # Test 3
        acronyms = np.array(['HY', 'PVZ', 'ADP', 'AHA'])
        values = np.array([1, 2, 3, 4])
        mapper = RegionMapper(acronyms, values, hemisphere='left')
        json_all, json_final = mapper.navigate_regions(acronyms, values, mapping='allen')
        # 1. All PVZ and all it's children are given value of PVZ
        acr = self.br.descendants(self.br.acronym2id('PVZ'))['acronym']
        assert all(json_all[a] == 2 for a in acr)
        # 2. Check that NC which is not in the volume is in json_all but not json_final
        assert json_all['NC'] == 2
        assert json_final.get('NC', None) is None
        # 3. Check that PVR is the mean value of ADP and AHA
        pvr_val = np.mean(values[-2:])
        assert json_all['PVR'] == pvr_val
        # 4. Check that children of PVR except ADP and AHA have the same value as PVR
        acr = self.br.descendants(self.br.acronym2id('PVR'))['acronym']
        acr = np.delete(acr, np.where(np.isin(acr, ['ADP', 'AHA']))[0])
        assert all([json_all[a] == pvr_val for a in acr])
        # 5. Check ADP and AHA have the expected value
        assert json_all['ADP'] == 3
        assert json_all['AHA'] == 4
        # 6. But AHA is not in the final json as not in the volume
        assert json_final.get('AHA', None) is None

        # Test 4
        acronyms = np.array(['FF', 'A13', 'PSTN', 'HY'])
        values = np.array([1, 2, 3, 4])
        mapper = RegionMapper(acronyms, values, hemisphere='left')
        json_all, json_final = mapper.navigate_regions(acronyms, values, mapping='Allen')
        # 1. Check that ZI has been given the average value of FF and A13
        assert json_all['ZI'] == np.mean(values[:2])
        # 2. Check that LZ has been given the mean value of ZI and PSTN, note not of PSTN, FF and A13
        assert json_all['LZ'] == np.mean([3, json_all['ZI']])
        assert json_all['LZ'] != np.mean(values[:3])
        # 3. Check that other children of LZ have the same value of LZ, but PSTN has its own value
        assert json_all['LPO'] == json_all['LZ']
        assert json_all['PSTN'] == 3
        # 4. Check that other children of HY have same value as HY
        acr = self.br.descendants(self.br.acronym2id('PVR'))['acronym']
        assert all([json_all[a] == 4 for a in acr])

        # Test 5
        # Now we need to look at the case where we have HY-lf and this behaviour
        acronyms = np.array(['HY', 'HY-lf', 'ME'])
        values = np.array([1, 2, 3])
        mapper = RegionMapper(acronyms, values, hemisphere='left')
        json_all, json_final = mapper.navigate_regions(acronyms, values, mapping='Allen')
        # 1. In json_all, we expect each to have its own value with same name
        assert json_all['HY'] == 1
        assert json_all['HY-lf'] == 2
        assert json_all['ME'] == 3
        # 2. In json_final we expect HY to be replaced by the value for HY-lf
        assert json_final['HY'] == json_all['HY-lf'] == 2
        # 3. We expect all the other children in the tree to take the value for HY
        acr = self.br.descendants(self.br.acronym2id('PVR'))['acronym']
        assert all([json_all[a] == 1 for a in acr])

        # Test 6
        acronyms = np.array(['HY-lf', 'ZI-lf'])
        values = np.array([1, 2])
        mapper = RegionMapper(acronyms, values, hemisphere='left')
        json_all, json_final = mapper.navigate_regions(acronyms, values, mapping='Allen')
        # 1. In json_all they are just given the values
        assert json_all['HY-lf'] == 1
        assert json_all['ZI-lf'] == 2
        # 2. In json_final they are moved to HY and ZI respectively
        assert json_final.get('HY-lf', None) is None
        assert json_final.get('ZI-lf', None) is None
        assert json_final['HY'] == json_all['HY-lf'] == 1
        assert json_final['ZI'] == json_all['ZI-lf'] == 2

        # Test 7
        acronyms = np.array(['FF', 'A13', 'ZI-lf', 'HY', 'PSTN'])
        values = np.array([1, 2, 3, 4, 5])
        mapper = RegionMapper(acronyms, values, hemisphere='left')
        json_all, json_final = mapper.navigate_regions(acronyms, values, mapping='Allen')
        # 1. In json_all ZI should have mean value of FF, A13 and ZI-lf
        assert json_all['ZI'] == np.mean(values[:3])
        # 2. In json_all LZ should have mean value of ZI and PSTN
        assert json_all['LZ'] == np.mean(np.r_[np.mean(values[:3]), 5])
        # 3. In json_final ZI should have value of ZI-lf, FF should have value of FF
        assert json_final['ZI'] == 3
        assert json_final['FF'] == 1
        # 4. In json_final LHA should have value of LZ in json_all
        assert json_final['LHA'] == json_all['LZ']
        # 5. Other children of HY should have this value
        assert json_final['HY'] == 4
        assert json_final['MPO'] == 4

        # Test 8. These ones are at the end of the road
        acronyms = np.array(['PVHap', 'PVHpv', 'PVHm', 'SO', 'HY'])
        values = np.array([1, 2, 3, 4, 5])
        mapper = RegionMapper(acronyms, values, hemisphere='left')
        json_all, json_final = mapper.navigate_regions(acronyms, values, mapping='Allen')
        # 1. The parent PVHp of PVHap and PVHpv should take the average value
        assert json_all['PVHp'] == np.mean(values[:2])
        # 2. The parent PVH of PVHm and PVHp should take their average value
        pvh_val = np.mean(np.r_[np.mean(values[:2]), 3])
        assert json_all['PVH'] == pvh_val
        # 3. The parent PVZ of PVH and SO should be their average values
        assert json_all['PVZ'] == np.mean(np.r_[pvh_val, 4])
        # 3. Only PVH and SO are in json_final
        assert json_final.get('PVH', None) is not None
        assert json_final.get('PVHm', None) is None

    def test_swanson(self):
        # Test 1
        acronyms = np.array(['MOp1', 'MOs5'])
        values = np.array([1, 2])
        mapper = RegionMapper(acronyms, values, hemisphere='left')
        json_all, json_final = mapper.navigate_regions(acronyms, values, mapping='Swanson')
        # 1. Value of MOp1 has been assigned to MOp and value of MOs5 to MOs
        assert json_final['MOp'] == 1
        assert json_final['MOs'] == 2

        # Test 2
        acronyms = np.array(['MO', 'MOp5', 'MOp1'])
        values = np.array([1, 2, 3])
        mapper = RegionMapper(acronyms, values, hemisphere='left')
        json_all, json_final = mapper.navigate_regions(acronyms, values, mapping='Swanson')
        # 1. json_all to contain additional keys MO, MO1, MO2/3, MO5, MO6a and MO6b compared to json_final as these
        #    regions aren't in the volume
        compare = np.array(['MO', 'MO1', 'MO2/3', 'MO5', 'MO6a', 'MO6b', 'MOp5', 'MOp1'])
        isin, _ = ismember(np.array(list(json_all.keys())), compare)
        assert np.sum(isin) == len(compare)
        isin, _ = ismember(np.array(list(json_final.keys())), compare)
        assert np.sum(isin) == 0
        # 2. Value of MO to propagate down to MOs
        assert json_final['MOs'] == values[0]
        # 3. MOp to be assigned np.mean(MOp5, MOp1)
        assert json_final['MOp'] == np.mean(values[1:])

        # Test 3
        acronyms = np.array(['HY', 'PVZ', 'ADP', 'AHA'])
        values = np.array([1, 2, 3, 4])
        mapper = RegionMapper(acronyms, values, hemisphere='left')
        json_all, json_final = mapper.navigate_regions(acronyms, values, mapping='Swanson')
        # 1. All PVZ and all it's children are given value of PVZ
        acr = self.br.descendants(self.br.acronym2id('PVZ'))['acronym']
        assert all(json_all[a] == 2 for a in acr)
        # 2. Check that NC which is not in the volume is in json_all but not json_final
        assert json_all['NC'] == 2
        assert json_final.get('NC', None) is None
        # 3. Check that PVR is the mean value of ADP and AHA
        pvr_val = np.mean(values[-2:])
        assert json_all['PVR'] == pvr_val
        # 4. Check that children of PVR except ADP and AHA have the same value as PVR
        acr = self.br.descendants(self.br.acronym2id('PVR'))['acronym']
        acr = np.delete(acr, np.where(np.isin(acr, ['ADP', 'AHA']))[0])
        assert all([json_all[a] == pvr_val for a in acr])
        # 5. Check ADP and AHA have the expected value
        assert json_all['ADP'] == 3
        assert json_all['AHA'] == 4
        # 6. AHA is included in swanson
        assert json_final['AHA'] == 4

        # Test 4
        acronyms = np.array(['FF', 'A13', 'PSTN', 'HY'])
        values = np.array([1, 2, 3, 4])
        mapper = RegionMapper(acronyms, values, hemisphere='left')
        json_all, json_final = mapper.navigate_regions(acronyms, values, mapping='Swanson')
        # 1. Check that ZI has been given the average value of FF and A13
        assert json_all['ZI'] == np.mean(values[:2])
        # 2. Check that LZ has been given the mean value of ZI and PSTN, note not of PSTN, FF and A13
        assert json_all['LZ'] == np.mean([3, json_all['ZI']])
        assert json_all['LZ'] != np.mean(values[:3])
        # 3. Check that other children of LZ have the same value of LZ, but PSTN has its own value
        assert json_all['LPO'] == json_all['LZ']
        assert json_all['PSTN'] == 3
        # 4. Check that other children of HY have same value as HY
        acr = self.br.descendants(self.br.acronym2id('PVR'))['acronym']
        assert all([json_all[a] == 4 for a in acr])
        # 5. Check that A13 is in Swanson mapping
        assert json_final['A13'] == 2

        # Test 5
        # Now we need to look at the case where we have HY-lf and this behaviour
        acronyms = np.array(['HY', 'HY-lf', 'ME'])
        values = np.array([1, 2, 3])
        mapper = RegionMapper(acronyms, values, hemisphere='left')
        json_all, json_final = mapper.navigate_regions(acronyms, values, mapping='Swanson')
        # 1. In json_all, we expect each to have its own value with same name
        assert json_all['HY'] == 1
        assert json_all['HY-lf'] == 2
        assert json_all['ME'] == 3
        # 2. In json_final we expect there to be no HY
        assert json_final.get('HY', None) is None
        # 3. We expect all the other children in the tree to take the value for HY
        acr = self.br.descendants(self.br.acronym2id('PVR'))['acronym']
        assert all([json_all[a] == 1 for a in acr])

        # Test 6
        acronyms = np.array(['HY-lf', 'ZI-lf'])
        values = np.array([1, 2])
        mapper = RegionMapper(acronyms, values, hemisphere='left')
        json_all, json_final = mapper.navigate_regions(acronyms, values, mapping='Swanson')
        # 1. In json_all they are just given the values
        assert json_all['HY-lf'] == 1
        assert json_all['ZI-lf'] == 2
        # 2. In json_final they are moved to HY and ZI respectively
        assert json_final.get('HY-lf', None) is None
        assert json_final.get('ZI-lf', None) is None
        assert json_final.get('HY', None) is None
        assert json_final['ZI'] == json_all['ZI-lf'] == 2


class TestMapValues(unittest.TestCase):
    def test_map_to_nodes_acronyms(self):
        # Case where there is no mapping
        acronyms = np.array(['HY', 'ZI', 'CB', 'MOs', 'VPM', 'AVPV'])
        values = np.arange(acronyms.size)
        mapper = RegionMapper(acronyms, values, hemisphere='left')
        new_acronyms = mapper.map_nodes_to_leaves()
        assert all(new_acronyms == np.array(['HY-lf', 'ZI-lf', 'CB-lf', 'MOs', 'VPM', 'AVPV']))
        mapper = RegionMapper(acronyms, values, hemisphere='right')
        new_acronyms = mapper.map_nodes_to_leaves()
        assert all(new_acronyms == np.array(['HY-lf', 'ZI-lf', 'CB-lf', 'MOs', 'VPM', 'AVPV']))

        # Case where mapping already exists
        acronyms = np.array(['HY', 'ZI-lf', 'CB-lf', 'MOs', 'VPM', 'AVPV'])
        values = np.arange(acronyms.size)
        mapper = RegionMapper(acronyms, values, hemisphere='left')
        new_acronyms = mapper.map_nodes_to_leaves()
        # new acronyms is unchanged
        assert all(new_acronyms == acronyms)
        mapper = RegionMapper(acronyms, values, hemisphere='right')
        new_acronyms = mapper.map_nodes_to_leaves()
        assert all(new_acronyms == acronyms)

    def test_map_to_nodes_ids(self):
        # Case where there is no mapping
        acronyms = np.array([1097,  797,  512,  993,  733,  272])
        values = np.arange(acronyms.size)
        mapper = RegionMapper(acronyms, values)
        new_acronyms = mapper.map_nodes_to_leaves()
        assert all(new_acronyms == np.array([5003,  5019,  5000,  993,  733,  272]))
        mapper = RegionMapper(-1 * acronyms, values)
        new_acronyms = mapper.map_nodes_to_leaves()
        assert all(new_acronyms == -1 * np.array([5003,  5019,  5000,  993,  733,  272]))

        # Case where mapping already exists
        acronyms = np.array([1097,  5019,  5000,  993,  733,  272])
        values = np.arange(acronyms.size)
        mapper = RegionMapper(acronyms, values)
        new_acronyms = mapper.map_nodes_to_leaves()
        # new acronyms is unchanged
        assert all(new_acronyms == acronyms)
        mapper = RegionMapper(-1 * acronyms, values)
        new_acronyms = mapper.map_nodes_to_leaves()
        assert all(new_acronyms == -1 * acronyms)

    def test_hemisphere_requirement(self):
        # When passing acronyms, must pass in a hemisphere argument
        acronyms = np.array(['HY', 'ZI-lf', 'CB-lf', 'MOs', 'VPM', 'AVPV'])
        values = np.arange(acronyms.size)
        with self.assertRaises(AssertionError) as context:
            RegionMapper(acronyms, values)

        self.assertTrue('hemisphere' in str(context.exception))

        # For ids, no need
        acronyms = np.array([5003, 5019, 5000, 993, 733, 272])
        _ = RegionMapper(acronyms, values)

    def test_validate_regions(self):
        # If we pass in acronyms
        acronyms = np.array(['HY', 'ZI-lf', 'CB-lf', 'MOs', 'VPM', 'AVPV'])
        values = np.arange(acronyms.size)
        mapper = RegionMapper(acronyms, values, hemisphere='left')
        assert mapper.is_acronym

        # If we pass in a dud acronyms
        acronyms = np.array(['lala', 'ZI-lf', 'CB-lf', 'MOs', 'VPM', 'AVPV'])
        values = np.arange(acronyms.size)
        with self.assertRaises(AssertionError) as context:
            RegionMapper(acronyms, values, hemisphere='left')
        self.assertTrue('The acronyms: lala' in str(context.exception))

        # Now check with ids
        acronyms = np.array([5003,  5019,  5000,  993,  733,  272])
        values = np.arange(acronyms.size)
        mapper = RegionMapper(acronyms, values)
        assert not mapper.is_acronym

        acronyms = np.array([50003, 5019, 5000, 993, 733, 272])
        values = np.arange(acronyms.size)
        with self.assertRaises(AssertionError) as context:
            RegionMapper(acronyms, values, hemisphere='left')
        self.assertTrue('The atlas ids: 50003' in str(context.exception))


class TestAllenMappings(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.br = BrainRegions()

    def test_acronyms(self):
        acronyms = np.array(['MOs', 'MOp5', 'MOp1'])
        values = np.array([1, 2, 3])
        # Left hemisphere
        mapper = RegionMapper(acronyms, values, hemisphere='left')
        index, vals = mapper.map_acronyms_to_allen()
        assert all(index > self.br.n_lr)
        assert all(['MO' in b for b in self.br.acronym[index]])

        # Right hemisphere
        mapper = RegionMapper(acronyms, values, hemisphere='right')
        index, vals = mapper.map_acronyms_to_allen()
        assert all(index <= self.br.n_lr)
        assert all(['MO' in b for b in self.br.acronym[index]])

        # Left hemisphere
        acronyms = np.array(['TH-lf', 'HY-lf', 'MOp1'])
        values = np.array([1, 2, 3])
        mapper = RegionMapper(acronyms, values, hemisphere='left')
        index, vals = mapper.map_acronyms_to_allen()
        assert all(index > self.br.n_lr)
        assert all([b in ['TH', 'HY', 'MOp1', 'MOp'] for b in self.br.acronym[index]])

        # Right hemisphere
        acronyms = np.array(['TH-lf', 'HY-lf', 'MOp1'])
        values = np.array([1, 2, 3])
        mapper = RegionMapper(acronyms, values, hemisphere='right')
        index, vals = mapper.map_acronyms_to_allen()
        assert all(index <= self.br.n_lr)
        assert all([b in ['TH', 'HY', 'MOp1', 'MOp'] for b in self.br.acronym[index]])

    def test_atlas_ids(self):
        # Left hemisphere
        acronyms = np.array([-993, -648, -320])
        values = np.array([1, 2, 3])
        mapper = RegionMapper(acronyms, values)
        index, vals = mapper.map_ids_to_allen()
        assert all(index > self.br.n_lr)
        assert all(['MO' in b for b in self.br.acronym[index]])

        # Right hemisphere
        acronyms = np.array([993, 648, 320])
        values = np.array([1, 2, 3])
        mapper = RegionMapper(acronyms, values)
        index, vals = mapper.map_ids_to_allen()
        assert all(index <= self.br.n_lr)
        assert all(['MO' in b for b in self.br.acronym[index]])

        # Left hemisphere
        acronyms = np.array([-5015, -5003, -320])
        values = np.array([1, 2, 3])
        mapper = RegionMapper(acronyms, values)
        index, vals = mapper.map_ids_to_allen()
        assert all(index > self.br.n_lr)
        assert all([b in ['TH', 'HY', 'MOp1', 'MOp'] for b in self.br.acronym[index]])

        # Right hemisphere
        acronyms = np.array([5015, 5003, 320])
        values = np.array([1, 2, 3])
        mapper = RegionMapper(acronyms, values)
        index, vals = mapper.map_ids_to_allen()
        assert all(index <= self.br.n_lr)
        assert all([b in ['TH', 'HY', 'MOp1', 'MOp'] for b in self.br.acronym[index]])

    def test_atlas_ids_lateralised(self):
        acronyms = np.array([-993, -648, -320, 5015, 5003, 320])
        values = np.array([1, 2, 3, 4, 5, 6])
        mapper = RegionMapper(acronyms, values)
        index, vals = mapper.map_ids_to_allen()
        assert all([a in index for a in self.br.id2index(320)[1][0]])
        assert self.br.acronym2index('HY', hemisphere='right')[1][0] in index
        assert self.br.id2index(-993, mapping='Allen-lr')[1][0] in index


class TestBerylMappings(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.br = BrainRegions()

    def test_acronyms(self):
        # Left hemisphere
        acronyms = np.array(['MOs', 'MOp5', 'MOp1'])
        values = np.array([1, 2, 3])
        mapper = RegionMapper(acronyms, values, hemisphere='left')
        index, vals = mapper.map_to_beryl()
        assert all(index > self.br.n_lr)
        assert all([b in ['MOs', 'MOp'] for b in self.br.acronym[index]])
        assert vals[self.br.acronym[index] == 'MOp'] == np.mean(values[1:])
        assert vals[self.br.acronym[index] == 'MOs'] == 1

        # Right hemisphere
        acronyms = np.array(['MOs', 'MOp5', 'MOp1'])
        values = np.array([1, 2, 3])
        mapper = RegionMapper(acronyms, values, hemisphere='right')
        index, vals = mapper.map_to_beryl()
        assert all(index <= self.br.n_lr)
        assert all([b in ['MOs', 'MOp'] for b in self.br.acronym[index]])
        assert vals[self.br.acronym[index] == 'MOp'] == np.mean(values[1:])
        assert vals[self.br.acronym[index] == 'MOs'] == 1

        # Left hemisphere
        acronyms = np.array(['TH-lf', 'HY-lf', 'MOp1'])
        values = np.array([1, 2, 3])
        mapper = RegionMapper(acronyms, values, hemisphere='left')
        index, vals = mapper.map_to_beryl()
        # root is weird
        assert all(index[index != 1] > self.br.n_lr)
        assert all([b in ['root', 'MOp'] for b in self.br.acronym[index]])

        # Right hemisphere
        acronyms = np.array(['TH-lf', 'HY-lf', 'MOp1'])
        values = np.array([1, 2, 3])
        mapper = RegionMapper(acronyms, values, hemisphere='right')
        index, vals = mapper.map_to_beryl()
        # root is weird
        assert all(index[index != 1] <= self.br.n_lr)
        assert all([b in ['root', 'MOp'] for b in self.br.acronym[index]])

    def test_atlas_ids(self):
        # Left hemisphere
        acronyms = np.array([-993, -648, -320])
        values = np.array([1, 2, 3])
        mapper = RegionMapper(acronyms, values)
        index, vals = mapper.map_to_beryl()
        assert all(index > self.br.n_lr)
        assert all([b in ['MOs', 'MOp'] for b in self.br.acronym[index]])
        assert vals[self.br.acronym[index] == 'MOp'] == np.mean(values[1:])
        assert vals[self.br.acronym[index] == 'MOs'] == 1

        # Right hemisphere
        acronyms = np.array([993, 648, 320])
        values = np.array([1, 2, 3])
        mapper = RegionMapper(acronyms, values)
        index, vals = mapper.map_to_beryl()
        assert all(index <= self.br.n_lr)
        assert all([b in ['MOs', 'MOp'] for b in self.br.acronym[index]])
        assert vals[self.br.acronym[index] == 'MOp'] == np.mean(values[1:])
        assert vals[self.br.acronym[index] == 'MOs'] == 1

        # Left hemisphere
        acronyms = np.array([-5015, -5003, -320])
        values = np.array([1, 2, 3])
        mapper = RegionMapper(acronyms, values)
        index, vals = mapper.map_to_beryl()
        # root is weird
        assert all(index[index != 1] > self.br.n_lr)
        assert all([b in ['root', 'MOp'] for b in self.br.acronym[index]])

        # Right hemisphere
        acronyms = np.array([5015, 5003, 320])
        values = np.array([1, 2, 3])
        mapper = RegionMapper(acronyms, values)
        index, vals = mapper.map_to_beryl()
        # root is weird
        assert all(index[index != 1] <= self.br.n_lr)
        assert all([b in ['root', 'MOp'] for b in self.br.acronym[index]])

    def test_atlas_ids_lateralised(self):
        acronyms = np.array([-993, -648, -320, 5015, 5003, 320])
        values = np.array([1, 2, 3, 4, 5, 6])
        mapper = RegionMapper(acronyms, values)
        index, vals = mapper.map_to_beryl()
        assert all([b in ['root', 'MOp', 'MOs'] for b in self.br.acronym[index]])
        # We have MOp for left and right hemisphere
        assert len(np.where(self.br.acronym[index] == 'MOp')[0]) == 2


class TestCosmosMappings(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.br = BrainRegions()

    def test_acronyms(self):
        # Left hemisphere
        acronyms = np.array(['MOs', 'MOp5', 'MOp1'])
        values = np.array([1, 2, 3])
        mapper = RegionMapper(acronyms, values, hemisphere='left')
        index, vals = mapper.map_to_cosmos()
        assert all(index > self.br.n_lr)
        assert all([b in ['Isocortex'] for b in self.br.acronym[index]])
        assert vals[0] == np.mean(values)

        # Right hemisphere
        acronyms = np.array(['MOs', 'MOp5', 'MOp1'])
        values = np.array([1, 2, 3])
        mapper = RegionMapper(acronyms, values, hemisphere='right')
        index, vals = mapper.map_to_cosmos()
        assert all(index <= self.br.n_lr)
        assert all([b in ['Isocortex'] for b in self.br.acronym[index]])
        assert vals[0] == np.mean(values)

        # Left hemisphere
        acronyms = np.array(['TH-lf', 'HY-lf', 'MOp1'])
        values = np.array([1, 2, 3])
        mapper = RegionMapper(acronyms, values, hemisphere='left')
        index, vals = mapper.map_to_cosmos()
        assert all(index > self.br.n_lr)
        assert all([b in ['Isocortex', 'TH', 'HY'] for b in self.br.acronym[index]])

        # Right hemisphere
        acronyms = np.array(['TH-lf', 'HY-lf', 'MOp1'])
        values = np.array([1, 2, 3])
        mapper = RegionMapper(acronyms, values, hemisphere='right')
        index, vals = mapper.map_to_cosmos()
        assert all(index <= self.br.n_lr)
        assert all([b in ['Isocortex', 'TH', 'HY'] for b in self.br.acronym[index]])

    def test_atlas_ids(self):
        # Left hemisphere
        acronyms = np.array([-993, -648, -320])
        values = np.array([1, 2, 3])
        mapper = RegionMapper(acronyms, values)
        index, vals = mapper.map_to_cosmos()
        assert all(index > self.br.n_lr)
        assert all([b in ['Isocortex'] for b in self.br.acronym[index]])
        assert vals[0] == np.mean(values)

        # Right hemisphere
        acronyms = np.array([993, 648, 320])
        values = np.array([1, 2, 3])
        mapper = RegionMapper(acronyms, values)
        index, vals = mapper.map_to_cosmos()
        assert all(index <= self.br.n_lr)
        assert all([b in ['Isocortex'] for b in self.br.acronym[index]])
        assert vals[0] == np.mean(values)

        # Left hemisphere
        acronyms = np.array([-5015, -5003, -320])
        values = np.array([1, 2, 3])
        mapper = RegionMapper(acronyms, values)
        index, vals = mapper.map_to_cosmos()
        assert all(index > self.br.n_lr)
        assert all([b in ['Isocortex', 'TH', 'HY'] for b in self.br.acronym[index]])

        # Right hemisphere
        acronyms = np.array([5015, 5003, 320])
        values = np.array([1, 2, 3])
        mapper = RegionMapper(acronyms, values)
        index, vals = mapper.map_to_cosmos()
        assert all(index <= self.br.n_lr)
        assert all([b in ['Isocortex', 'TH', 'HY'] for b in self.br.acronym[index]])

    def test_atlas_ids_lateralised(self):
        acronyms = np.array([-993, -648, -320, 5015, 5003, 320])
        values = np.array([1, 2, 3, 4, 5, 6])
        mapper = RegionMapper(acronyms, values)
        index, vals = mapper.map_to_cosmos()
        assert all([b in ['Isocortex', 'TH', 'HY'] for b in self.br.acronym[index]])
        # We have Isocortex for left and right hemisphere
        assert len(np.where(self.br.acronym[index] == 'Isocortex')[0]) == 2


class TestFullProcess(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.br = BrainRegions()

    def test_process_regions(self):
        # Left hemisphere
        acronyms = np.array(['MOs', 'MOp5', 'MOp1'])
        values = np.array([1, 2, 3])
        mapper = RegionMapper(acronyms, values, hemisphere='left')
        data = mapper.map_regions()

        assert ['allen', 'beryl', 'cosmos'] == list(data.keys())