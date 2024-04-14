import os
from torch_geometric.data import InMemoryDataset
from typing import List

class LocalDataset(InMemoryDataset):
    def __init__(self, root, data_list):
        super(RandomCktDataset, self).__init__(root)
        self.data_list = data_list

    def process(self):
        # Example of processing; adapt based on actual dataset class requirements
        self.data, self.slices = self.collate(self.data_list)

def read_cnf_file(filepath: str) -> List[List[int]]:
    with open(filepath, 'r') as file:
        lines = file.readlines()
        clauses = []
        for line in lines:
            if line.startswith('p cnf'):
                continue
            # Convert line to list of integers and remove the trailing zero
            clause = list(map(int, line.strip().split()))[:-1]
            if clause:
                clauses.append(clause)
    return clauses

def load_cnf_files(directory: str) -> List[List[List[int]]]:
    all_data = []
    for filename in os.listdir(directory):
        if filename.endswith('.cnf'):
            filepath = os.path.join(directory, filename)
            cnf_data = read_cnf_file(filepath)
            all_data.append(cnf_data)
    return all_data

if __name__ == '__main__':
    # Example usage
    directory_path = '/path/to/cnf_files'
    cnf_data_list = load_cnf_files(directory_path)
    dataset = LocalDataset(root='/path/to/save/dataset', data_list=cnf_data_list)
    dataset.process()
    
    # Now dataset can be used for training and testing
