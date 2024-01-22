import unittest


 #import Planetoid, CitationFull, WikiCS, Coauthor
from graph_description.utils import get_dataset_folder
import numpy as np
from graph_description.datasets import edges_read_attributed_graph
from collections import Counter


import torch_geometric.datasets as real_datasets
import graph_description.torch_port.torch_datasets as my_datasets


from graph_description.utils import get_dataset_folder
import numpy as np

folder = get_dataset_folder()
pytorch_folder = get_dataset_folder().parent/"pytorch_datasets"

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
            ("pubmed", "citationfull")]
        #     ("wikics", "wikics"),
        #     ("physics", "coauthor"),
        #     ("cs", "coauthor")
        # ]


    def test_agreeswith_original(self):
        """Smoke test that test whether all datasets can be loaded by full path"""
        lower_case_to_real = {
            "planetoid" : "Planetoid",
            "citationfull" : "CitationFull",
        }

        for name, group in self.datasets:
            cls = getattr(my_datasets, lower_case_to_real[group])
            # print(cls)
            should_reload = False
            our_dataset = cls(folder, name, force_reload=should_reload)
            # print(our_dataset._data.x)

            real_cls = getattr(real_datasets, lower_case_to_real[group])
            # print(real_cls)
            real_dataset = real_cls(pytorch_folder, name)
            # print(real_dataset.x)

            # edge_index agrees
            real_edge_index = real_dataset.edge_index.cpu().detach().numpy()
            np.testing.assert_array_equal(our_dataset._data.edge_index, real_edge_index)
            
            # y agrees
            real_y = real_dataset.y.cpu().detach().numpy()
            np.testing.assert_array_equal(our_dataset._data.y, real_y)

            real_x = real_dataset.x.cpu().detach().numpy()
            rows,cols = np.nonzero(real_x)
            np.testing.assert_array_equal(our_dataset._data.x.row, rows)
            np.testing.assert_array_equal(our_dataset._data.x.col, cols)
            np.testing.assert_array_equal(our_dataset._data.x.data, real_x[(rows,cols)])

            #self.assertEqual(edges.shape[1], 2, msg=f"Make sure the edges are returned in correct format {name} {group}")



if __name__ == "__main__":
    unittest.main()