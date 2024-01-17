import unittest
import hashlib

from graph_description.torch_port.torch_datasets import Planetoid, CitationFull, WikiCS
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
{'WikiCS': '2c652e33b3433e171435605094ad78fa968385241d924840be5f8811fc483898',
 'citeseer': 'e3353af7815344afc908fdeb103a4b12614a4890e8032a298648177243bfa1d6',
 'pubmed': 'bda1705d915f2c08508b655b990f073e11127c79b16069174e81d9e0c3c88af3'}
,
{'WikiCS': '3158ba2c99e0789577507b5feab0da1a53922a41e68345b08a8bc6bed2f2192d',
 'citeseer': '448f8c04e02516b2784cfd3c5d82049f84ea77d6ef94c65ee19f8ea8b792b0fa',
 'pubmed': '973a30d7c0567fe67fb9026058f1105859f87d44981b7b88992d648877d174c7'}
,
{'WikiCS': '41009cf2069adf98e98c165886c77515471015822736d041560a4d26a6cbc6eb',
 'citeseer': 'ac01cf6f7b3fd88c3245191cc13b962d5173da7b78a3487b7aaa3cc290d11900',
 'pubmed': 'e1c78d6230b975b4263e3670fe7269b0b621e4b455e6e8503cae09e0990bf5ef'}
,
{('WikiCS', 'public'): 'bae72b6c94ccc6225757c3d81e65e9fabacc041304b5f09b198b13de87e57f6a',
 ('citeseer', 'full'): '7af341d90e92d0f51e173d7132b17d617344880dad8ea062747a7ec18a1ac666',
 ('citeseer', 'geom-gcn'): '046e0d6aa79604a2d39fd9ee11e12748849941ee772cb7004c0ee8a5e0e09f7c',
 ('citeseer', 'public'): '8efce52f9800be63f1ac0858b52958bbdde930d20ce5a0b4e5431c13a7d16eab',
 ('citeseer', 'random'): 'c745ef10828ce9bd7891ae504fa10a08d16f12bc99a29a981185cd1ca8ee536a',
 ('pubmed', 'geom-gcn'): 'ff047d7ae0a1118eabec8ac69e8d054b312660a9a87210361373251984f429a6',
 ('pubmed', 'full'): '0d692b33a33fd1347f131541f5432fb606894408783096023647aed51a75b415',
 ('pubmed', 'public'): 'c212c436a88f7ca1f236f4e98a2ce1070e4a7e1bdf1d9f8e3899ee8d0aa8b429',
 ('pubmed', 'random'): '6692b00ea1e4a8226139b060ca4a49e4bc84895d3df5bebf40cf961542aa8e47'}
,
{('WikiCS', 'public'): '2fbfacb7de0b7b9758001db87d3e9396ae6f684e6d7f444961267aaf9a449870',
 ('citeseer', 'full'): 'b11b25e3892ac2656e35555f2ea1ddcbccba18f2f1233baffbfdaacf3a7fda63',
 ('citeseer', 'geom-gcn'): '19335b4b9128ea8b5286d8401a67284478df855b53080794a32d29139366ad41',
 ('citeseer', 'public'): 'b11b25e3892ac2656e35555f2ea1ddcbccba18f2f1233baffbfdaacf3a7fda63',
 ('citeseer', 'random'): 'e0d8bccaf4ff999fcac61f2185c1fab4253c8509f05e3f6a18607392f19a2bc7',
 ('pubmed', 'full'): '71f6cee77418b9ccab03e3c6a04a71f1fe0c07007ba3f006589bdda856de0b8b',
 ('pubmed', 'geom-gcn'): 'c0a18a6b235f82da65193b13141f04102ffc620dea6aac69796da0d65471728c',
 ('pubmed', 'public'): '71f6cee77418b9ccab03e3c6a04a71f1fe0c07007ba3f006589bdda856de0b8b',
 ('pubmed', 'random'): '36bed1a859d5825fe072ea2575198d3b1d9114266c8c7681605d12a4b4533fa7'}
,
{('WikiCS', 'public'): '9c23fea2990b0c7c6f166a8b37aba080462025b0632bf9b8f547a12e4862a09e',
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
    def __init__(self, tag, lookup=None):
        self.d_x = {}
        self.d_y = {}
        self.d_e = {}
        self.tag=tag
        self.lookup = lookup


    def add_hashes(self, dataset, split, data):
        self.d_x[(dataset)]=my_hash(data._data.x)
        self.d_y[(dataset)]=my_hash(data._data.y)
        self.d_e[(dataset)]=my_hash(data._data.edge_index)

    def get_hashes(self, dataset, split):
        return {
             "x" : self.d_x[(dataset)],
             "y" : self.d_y[(dataset)],
             "edge_index" : self.d_e[(dataset)]
        }

    def get_static_hashes(self, dataset, split):
        out = {}
        for key in ["x", "y", "edge_index"]:
            out[key] = self.lookup[key][dataset]

        return out

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

class SplitHashTracker:
    def __init__(self, lookup=None):
        self.d_x = {}
        self.d_y = {}
        self.d_e = {}
        self.d_s_train = {}
        self.d_s_val = {}
        self.d_s_test = {}
        self.lookup = lookup

    def add_hashes(self, dataset, split, data):
        self.d_x[(dataset)]=my_hash(data._data.x)
        self.d_y[(dataset)]=my_hash(data._data.y)
        self.d_e[(dataset)]=my_hash(data._data.edge_index)
        self.d_s_train[(dataset, split)]=my_hash(data._data.train_mask)
        self.d_s_val[(dataset, split)]=my_hash(data._data.val_mask)
        self.d_s_test[(dataset, split)]=my_hash(data._data.test_mask)


    def get_hashes(self, dataset, split):
        return {
             "x" : self.d_x[(dataset)],
             "y" : self.d_y[(dataset)],
             "edge_index" : self.d_e[(dataset)],
             "train_mask" : self.d_s_train[(dataset, split)],
             "val_mask" : self.d_s_val[(dataset, split)],
             "test_mask" : self.d_s_test[(dataset, split)]
        }

    def get_static_hashes(self, dataset, split):
        out = {}
        for key in ["x", "y", "edge_index"]:
            out[key] = self.lookup[key][dataset]

        for key in ["train_mask", "val_mask", "test_mask"]:
            assert key in self.lookup, key
            assert (dataset, split) in self.lookup[key], str((key, dataset, split))
            out[key] = self.lookup[key][(dataset, split)]

        return out

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







d_x_citation_target, d_y_citation_target, d_e_citation_target = (
{'CiteSeer': 'bfd9b9b1d9c17e4c616f3a60b409a1299ba365f78353444d18c76ba5cada8b60',
 'Cora': '710f556e2a1320533c41d573bf32de32fbc8409e8ff641ea655771a9d7a1c543',
 'Cora_ML': '86b1053a1f0defea0cbfc5c5e6543c32415c6aba8ecf037672a062a74ec4e3e5',
 'DBLP': 'dc135347e63952a401218347861fe2292978b23028cbacfd13befe3749e8b1b6',
 'PubMed': 'e1d3b8f812556e585dba6b1b8375ed37f603852bd443307231709a35dfc3787b',}
,
{'CiteSeer': '50444478c7b13f99701b4cfb01c4c984a5138e7469df7f4ebdb30f6beb7dd14c',
 'Cora': 'c0337b253b331a78e7613a8b8038867d9fae42d6c6cec44b3ec55b05b82e4991',
 'Cora_ML': 'a9f950be5894ac3a1dd69108dfc843dda81157569b0a44a515e744bc1b6e8eb7',
 'DBLP': '74fcbdea13195b1d05d5339471d71d96b41e15cc18b2927bf01eecce7c7f2519',
 'PubMed': '973a30d7c0567fe67fb9026058f1105859f87d44981b7b88992d648877d174c7',}
,
{'CiteSeer': '220ba2d2850103e52dbe16e7a82420e14e9dd15e7350c8f11fb8d2535766a749',
 'Cora': 'c88426defea354deb76c98008dfc3a6eb3e1c73d533ffd7ceff166dced090267',
 'Cora_ML': '6fb8f7acfeec68ee98358467afd4a53d24c5b2c882bbbdddca1b7174e27362b4',
 'DBLP': '4dd8acfc55414fb27b09fead2357aa4b1acadae1da5ec3e497d1a4863c5b4baa',
 'PubMed': 'cba1d66eae6ed39f2f1c46f97e6a52d4e7c3ce9b00d3a7a1c432f73e3eadfc67',}
)

citation_targets = {
    "x" : d_x_citation_target,
    "y" : d_y_citation_target,
    "edge_index" : d_e_citation_target,
}

class TestDirectDatasetLoading(unittest.TestCase):
    #def check_hashes(self, dataset, data):
    #    for key, d_target in citation_targets.items():
    #        arr = getattr(data._data, key)
    #        the_hash = my_hash(arr)
    #        self.check_hash(dataset, d_target, the_hash, key)

    def check_hashes(self, tracker, dataset, split):
        static_hashes = tracker.get_static_hashes(dataset, split)
        current_hashes = tracker.get_hashes(dataset, split)
        for key, expected_hash in static_hashes.items():
            current_hash = current_hashes[key]
            self.assertEqual(current_hash, expected_hash, f"found hash inconsistency for key '{key}' {dataset} {split}")


    def atest_loading_with_splits(self):
        """Testing consistency in model loading across multiple runs"""
        force_reload = True
        hash_tracker = HashTracker(tag="citation_", lookup=citation_targets)
        split_tracker = SplitHashTracker(lookup=targets)
        citation_datasets = ["Cora", "Cora_ML", "CiteSeer", "DBLP", "PubMed"]
        planetoid_datasets = ["citeseer", "pubmed", "Cora"]
        wikics_dataset = ["WikiCS",]
        all_datasets = citation_datasets+planetoid_datasets+wikics_dataset

        for dataset in all_datasets:
            if dataset in wikics_dataset:
                dataset_cls = WikiCS
                tracker = split_tracker
                splits = ["public"]
            elif dataset in citation_datasets:
                dataset_cls = CitationFull
                tracker = hash_tracker
                splits = ["public"]
            elif dataset in planetoid_datasets:
                dataset_cls = Planetoid
                tracker = split_tracker
                splits = ["public", "full", "geom-gcn", "random"]
            for force_reload in [False]:
                for split in splits:
                    with self.subTest(f"reload_test", dataset=dataset, force_reload=force_reload, split=split):
                        np.random.seed(0)
                        data = dataset_cls(folder, dataset, force_reload=force_reload, log=False, split=split)
                        tracker.add_hashes(dataset, split, data)
                        self.check_hashes(tracker, dataset, split)
        # hash_tracker.pprint()
        # split_tracker.pprint()

from graph_description.datasets import edges_read_attributed_graph
from collections import Counter
class TestIntendedLoading(unittest.TestCase):

    def setUp(self):
        self.datasets= [
            ("cora", "planetoid"),
            ("citeseer", "planetoid"),
            ("pubmed", "planetoid"),
            ("cora", "citationfull"),
            ("cora_ml", "citationfull"),
            ("citeseer", "citationfull"),
            ("dblp", "citationfull"),
            ("pubmed", "citationfull"),
            ("wikics", "wikics"),
        ]


    def test_loading_full(self):
        """Smoke test that test whether all datasets can be loaded by full path"""
        for name, group in self.datasets:
            edges, df = edges_read_attributed_graph(name, dataset_path=folder, group=group)


    def test_loading_auto(self):
        """Smoke test that test whether all datasets can be loaded by specifying name only
        This also makes sure errors are raised for non unique names
        """
        for name, group in self.datasets:
            edges, df = edges_read_attributed_graph(name, dataset_path=folder, group=group)
        non_unique_datasets = set(key for key,value in Counter(name for name, group in self.datasets).items() if value>1)

        for name, _ in self.datasets:
            if name in non_unique_datasets:
                with self.assertRaises(ValueError):
                    edges, df = edges_read_attributed_graph(name, dataset_path=folder, group=None)
            else:
                edges, df = edges_read_attributed_graph(name, dataset_path=folder, group=None)



if __name__ == "__main__":
    unittest.main()