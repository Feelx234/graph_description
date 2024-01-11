import unittest
import hashlib

from graph_description.torch_port.torch_datasets import Planetoid, CitationFull
from graph_description.utils import get_dataset_folder
import numpy as np

folder = get_dataset_folder()

def my_hash(arr):
    tmp_writeable=arr.flags.writeable
    arr.flags.writeable=False
    the_hash = hashlib.sha256(arr.tobytes()).hexdigest()
    arr.flags.writeable = tmp_writeable
    return the_hash

# Hashes to check for agreement in future iterations
d_x_target, d_y_target, d_e_target, d_s_train_target, d_s_val_target, d_s_test_target = (
{'Cora': '3aecfdb5014544cc954dc24647b0f11359f706bfea9e4fe2bf31c7d684d53c73',
 'citeseer': 'e3353af7815344afc908fdeb103a4b12614a4890e8032a298648177243bfa1d6',
 'pubmed': 'bda1705d915f2c08508b655b990f073e11127c79b16069174e81d9e0c3c88af3'}
,
{'Cora': 'd0018d7ffd314eb25a4904973f2c7eb1b7be135d668de42d453714724f8e9d03',
 'citeseer': '448f8c04e02516b2784cfd3c5d82049f84ea77d6ef94c65ee19f8ea8b792b0fa',
 'pubmed': '973a30d7c0567fe67fb9026058f1105859f87d44981b7b88992d648877d174c7'}
,
{'Cora': '724254bf4ed0fc905b0a27cdf6cc42373465372c9aa277de5625bff0b4c57c7b',
 'citeseer': 'ac01cf6f7b3fd88c3245191cc13b962d5173da7b78a3487b7aaa3cc290d11900',
 'pubmed': 'e1c78d6230b975b4263e3670fe7269b0b621e4b455e6e8503cae09e0990bf5ef'}
,
{('Cora', 'full'): '33ecbbd095773f24e8959f7bfe7246efc467c5911147c2e6b98cf078a1327346',
 ('Cora', 'geom-gcn'): 'd161fc0188790e212522d4899404f2c42b057ebff0c6995591b135aee4885e8a',
 ('Cora', 'public'): 'b211325c561d6f73f978a4b0f3d44571c58eb70ca55a2aa27b9d4988344476c1',
 ('Cora', 'random'): '8f4f222f891d1349a5a9d7a228a437e934e3ccc2c74d4219c4e9bab6b76fc0a0',
 ('citeseer', 'full'): '7af341d90e92d0f51e173d7132b17d617344880dad8ea062747a7ec18a1ac666',
 ('citeseer', 'geom-gcn'): '046e0d6aa79604a2d39fd9ee11e12748849941ee772cb7004c0ee8a5e0e09f7c',
 ('citeseer', 'public'): '8efce52f9800be63f1ac0858b52958bbdde930d20ce5a0b4e5431c13a7d16eab',
 ('citeseer', 'random'): 'c745ef10828ce9bd7891ae504fa10a08d16f12bc99a29a981185cd1ca8ee536a',
 ('pubmed', 'full'): '0d692b33a33fd1347f131541f5432fb606894408783096023647aed51a75b415',
 ('pubmed', 'geom-gcn'): 'ff047d7ae0a1118eabec8ac69e8d054b312660a9a87210361373251984f429a6',
 ('pubmed', 'public'): 'c212c436a88f7ca1f236f4e98a2ce1070e4a7e1bdf1d9f8e3899ee8d0aa8b429',
 ('pubmed', 'random'): '6692b00ea1e4a8226139b060ca4a49e4bc84895d3df5bebf40cf961542aa8e47'}
,
{('Cora', 'full'): '96c8f5dc197fc3c014b3d655f1c28d0d276221e867ca01d64f74edb39cfb7173',
 ('Cora', 'geom-gcn'): 'f34080500f704c9cd4d02b396e2230ae96f25de2f9b53e5e13a841e9bd410ae8',
 ('Cora', 'public'): '96c8f5dc197fc3c014b3d655f1c28d0d276221e867ca01d64f74edb39cfb7173',
 ('Cora', 'random'): '23fb2e791fb7bf950b8636a0875e360c2ba017d17cc21208b339a357aa533881',
 ('citeseer', 'full'): 'b11b25e3892ac2656e35555f2ea1ddcbccba18f2f1233baffbfdaacf3a7fda63',
 ('citeseer', 'geom-gcn'): '19335b4b9128ea8b5286d8401a67284478df855b53080794a32d29139366ad41',
 ('citeseer', 'public'): 'b11b25e3892ac2656e35555f2ea1ddcbccba18f2f1233baffbfdaacf3a7fda63',
 ('citeseer', 'random'): 'e0d8bccaf4ff999fcac61f2185c1fab4253c8509f05e3f6a18607392f19a2bc7',
 ('pubmed', 'full'): '71f6cee77418b9ccab03e3c6a04a71f1fe0c07007ba3f006589bdda856de0b8b',
 ('pubmed', 'geom-gcn'): 'c0a18a6b235f82da65193b13141f04102ffc620dea6aac69796da0d65471728c',
 ('pubmed', 'public'): '71f6cee77418b9ccab03e3c6a04a71f1fe0c07007ba3f006589bdda856de0b8b',
 ('pubmed', 'random'): '36bed1a859d5825fe072ea2575198d3b1d9114266c8c7681605d12a4b4533fa7'}
,
{('Cora', 'full'): 'ec05c123cc125ff38ad3963cba5d85415a2e7b125dc68064bbf65fb0661644e6',
 ('Cora', 'geom-gcn'): '7afcd568d18d64762597dbed0b71b736049cd329f8ce83a2120d844bf6e43f5f',
 ('Cora', 'public'): 'ec05c123cc125ff38ad3963cba5d85415a2e7b125dc68064bbf65fb0661644e6',
 ('Cora', 'random'): '817fd8a0ac50c6fcfdfd317f70b65eaa2c9f9c24ae49b55e9ac6357cd63bd801',
 ('citeseer', 'full'): 'e5f2dd697aa940453faf3ca12682f7ddb7da9a7b011e141672be2b230889b30a',
 ('citeseer', 'geom-gcn'): '2999e9c3d40e3c5a03fb65e9a1f2c1bfa652920027b6f121737a77741b715614',
 ('citeseer', 'public'): 'e5f2dd697aa940453faf3ca12682f7ddb7da9a7b011e141672be2b230889b30a',
 ('citeseer', 'random'): 'aac4836d86ff3394c31ea5c44951af225b23db6350eaed6b77cba0d9af4374ab',
 ('pubmed', 'full'): 'fd659d892af41768c740746a0ecccdb3984e5277939e9264588b5f6efb4d5385',
 ('pubmed', 'geom-gcn'): '1689a8e042a431fd997a65cc5a462c6c4291845bdda0518e1a3ff05aa8fa71b4',
 ('pubmed', 'public'): 'fd659d892af41768c740746a0ecccdb3984e5277939e9264588b5f6efb4d5385',
 ('pubmed', 'random'): '90a150cca817694c511fcaebb74930eddd95babf6edd44ff83c68ea2564944aa'}
)



targets = {
    "x" : d_x_target,
    "y" : d_y_target,
    "edge_index" : d_e_target,
    "train_mask" : d_s_train_target,
    "val_mask"   : d_s_val_target,
    "test_mask"  : d_s_test_target,
}

class HashTracker:
    def __init__(self, tag):
        self.d_x = {}
        self.d_y = {}
        self.d_e = {}
        self.tag=tag


    def add_hashes(self, dataset, data):
        self.d_x[(dataset)]=my_hash(data._data.x)
        self.d_y[(dataset)]=my_hash(data._data.y)
        self.d_e[(dataset)]=my_hash(data._data.edge_index)

    def pprint(self):
        import pprint
        pp = pprint.PrettyPrinter(depth=4)
        tag = self.tag
        print(f"d_x_{tag}target, d_y_{tag}target, d_e_{tag}target = (")
        pp.pprint(self.d_x)
        print(",")
        pp.pprint(self.d_y)
        print(",")
        pp.pprint(self.d_e)
        print(")")

class PlanetoidHashTracker:
    def __init__(self):
        self.d_x = {}
        self.d_y = {}
        self.d_e = {}
        self.d_s_train = {}
        self.d_s_val = {}
        self.d_s_test = {}

    def add_hashes(self, dataset, split, data):
        self.d_x[(dataset)]=my_hash(data._data.x)
        self.d_y[(dataset)]=my_hash(data._data.y)
        self.d_e[(dataset)]=my_hash(data._data.edge_index)
        self.d_s_train[(dataset, split)]=my_hash(data._data.train_mask)
        self.d_s_val[(dataset, split)]=my_hash(data._data.val_mask)
        self.d_s_test[(dataset, split)]=my_hash(data._data.test_mask)

    def pprint(self):
        import pprint
        pp = pprint.PrettyPrinter(depth=4)
        print("d_x_target, d_y_target, d_e_target, d_s_train_target, d_s_val_target, d_s_test_target = (")
        pp.pprint(self.d_x)
        print(",")
        pp.pprint(self.d_y)
        print(",")
        pp.pprint(self.d_e)
        print(",")
        pp.pprint(self.d_s_train)
        print(",")
        pp.pprint(self.d_s_val)
        print(",")
        pp.pprint(self.d_s_test)
        print(")")





class TestPlanetoidDatasetLoading(unittest.TestCase):
    def check_hashes(self, dataset, split, data):
        for key, d_target in targets.items():
            arr = getattr(data._data, key)
            the_hash = my_hash(arr)
            if key in ["x", "y", "edge_index"]:
                self.check_hash(dataset, d_target, the_hash, key)
            else:
                self.check_hash((dataset, split), d_target, the_hash, key)

    def check_hash(self, index, d_target, hash_value, key):
        #print(d_target)
        self.assertEqual(d_target[index], hash_value, f"found_inconsistency for key '{key}' "+ str(index))

    def test_planetoid_loading_no_reload_consistency(self):
        """Smoke test to run the different datasets"""
        force_reload = False
        hash_tracker = PlanetoidHashTracker()
        for dataset in ["citeseer", "pubmed", "Cora"]:
            for split in "public", "full", "geom-gcn", "random":
                np.random.seed(0)
                data = Planetoid(folder, dataset, force_reload=force_reload, split=split, log=False)
                hash_tracker.add_hashes(dataset, split, data)
                self.check_hashes(dataset, split, data)

    def test_planetoid_loading_force_reload_consistency(self):
        """Smoke test to run the different datasets"""
        force_reload = True
        hash_tracker = PlanetoidHashTracker()
        for dataset in ["citeseer", "pubmed", "Cora"]:
            for split in "public", "full", "geom-gcn", "random":
                np.random.seed(0)
                data = Planetoid(folder, dataset, force_reload=force_reload, split=split, log=False)
                hash_tracker.add_hashes(dataset, split, data)
                self.check_hashes(dataset, split, data)

        # hash_tracker.pprint()





d_x_citation_target, d_y_citation_target, d_e_citation_target = (
{'CiteSeer': 'bfd9b9b1d9c17e4c616f3a60b409a1299ba365f78353444d18c76ba5cada8b60',
 'Cora': '710f556e2a1320533c41d573bf32de32fbc8409e8ff641ea655771a9d7a1c543',
 'Cora_ML': '86b1053a1f0defea0cbfc5c5e6543c32415c6aba8ecf037672a062a74ec4e3e5',
 'DBLP': 'dc135347e63952a401218347861fe2292978b23028cbacfd13befe3749e8b1b6',
 'PubMed': 'e1d3b8f812556e585dba6b1b8375ed37f603852bd443307231709a35dfc3787b'}
,
{'CiteSeer': '50444478c7b13f99701b4cfb01c4c984a5138e7469df7f4ebdb30f6beb7dd14c',
 'Cora': 'c0337b253b331a78e7613a8b8038867d9fae42d6c6cec44b3ec55b05b82e4991',
 'Cora_ML': 'a9f950be5894ac3a1dd69108dfc843dda81157569b0a44a515e744bc1b6e8eb7',
 'DBLP': '74fcbdea13195b1d05d5339471d71d96b41e15cc18b2927bf01eecce7c7f2519',
 'PubMed': '973a30d7c0567fe67fb9026058f1105859f87d44981b7b88992d648877d174c7'}
,
{'CiteSeer': '220ba2d2850103e52dbe16e7a82420e14e9dd15e7350c8f11fb8d2535766a749',
 'Cora': 'c88426defea354deb76c98008dfc3a6eb3e1c73d533ffd7ceff166dced090267',
 'Cora_ML': '6fb8f7acfeec68ee98358467afd4a53d24c5b2c882bbbdddca1b7174e27362b4',
 'DBLP': '4dd8acfc55414fb27b09fead2357aa4b1acadae1da5ec3e497d1a4863c5b4baa',
 'PubMed': 'cba1d66eae6ed39f2f1c46f97e6a52d4e7c3ce9b00d3a7a1c432f73e3eadfc67'}
)

citation_targets = {
    "x" : d_x_citation_target,
    "y" : d_y_citation_target,
    "edge_index" : d_e_citation_target,
}

class TestCitationDatasetLoading(unittest.TestCase):
    def check_hashes(self, dataset, data):
        for key, d_target in citation_targets.items():
            arr = getattr(data._data, key)
            the_hash = my_hash(arr)
            self.check_hash(dataset, d_target, the_hash, key)

    def check_hash(self, index, d_target, hash_value, key):
        #print(d_target)
        self.assertEqual(d_target[index], hash_value, f"found hash inconsistency for key '{key}' "+ str(index))


    def test_planetoid_loading(self):
        """Smoke test to run the different datasets"""
        force_reload = True
        hash_tracker = HashTracker(tag="citation_")
        for dataset in ["Cora", "Cora_ML", "CiteSeer", "DBLP", "PubMed"]:
            for force_reload in [True, False]:
                with self.subTest(f"reload_test", dataset=dataset, force_reload=force_reload):
                    np.random.seed(0)
                    data = CitationFull(folder, dataset, force_reload=force_reload, log=False)
                    hash_tracker.add_hashes(dataset, data)
                    #self.check_hashes(dataset, data)
        #hash_tracker.pprint()



if __name__ == "__main__":
    unittest.main()