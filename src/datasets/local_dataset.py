import os
import glob
import torch
from torch_geometric.data import InMemoryDataset, Data
from utils.dataset_utils import cnf_parse_pyg
import utils.cnf_utils as cnf_utils

class LocalDataset(InMemoryDataset):
    def __init__(self, args, root, transform=None, pre_transform=None, pre_filter=None):
        self.name = args.dataset
        self.args = args
        self.raw_dir = self.rawdata_dir
        print('total # of problems: ', self.args.n_pairs)
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        # This will list all CNF files in the raw directory
        return [f for f in os.listdir(self.raw_dir) if os.path.isfile(os.path.join(self.raw_dir, f))]

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        # As data is local, no need to download anything.
        pass

    def process(self):
        data_list = []
        file_path_pattern = os.path.join(self.raw_dir, '*.txt')  # Assuming all files have .cnf extensiona
        files = glob.glob(file_path_pattern)

        for file_name in files[:100]:  # Limit to loading 100 instances
            cnf, n_vars = cnf_utils.read_cnf(file_name)
            graph_data = cnf_parse_pyg(cnf, is_sat=None, num_vars=n_vars, num_clauses=len(cnf))
            data_list.append(graph_data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        print(len(data_list))

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, len(self))


# def read_cnf_file(filepath: str) -> List[List[int]]:
#     with open(filepath, 'r') as file:
#         lines = file.readlines()
#         clauses = []
#         for line in lines:
#             if line.startswith('p cnf'):
#                 continue
#             # Convert line to list of integers and remove the trailing zero
#             clause = list(map(int, line.strip().split()))[:-1]
#             if clause:
#                 clauses.append(clause)
#     return clauses

# def load_cnf_files(directory: str) -> List[List[List[int]]]:
#     all_data = []
#     for filename in os.listdir(directory):
#         if filename.endswith('.cnf'):
#             filepath = os.path.join(directory, filename)
#             cnf_data = read_cnf_file(filepath)
#             all_data.append(cnf_data)
#     return all_data

# if __name__ == '__main__':
#     # Example usage
#     directory_path = '/path/to/cnf_files'
#     cnf_data_list = load_cnf_files(directory_path)
#     dataset = LocalDataset(root='/path/to/save/dataset', data_list=cnf_data_list)
#     dataset.process()
    
#     # Now dataset can be used for training and testing
