import torch
import os
import sys
import utils
from network import Semilabel


seq_file = sys.argv[1]
files_path = os.path.split(seq_file)[0]
model_path = 'C:\\Workspace\\PE\\UI\\saved_models\\'
device = torch.device('cpu')
pretrained_model = torch.load(os.path.join(model_path, '128_Linear.pth'), map_location=device)
pretrained_model.eval()
pretrained_model.is_training = False
downstream_model_names = ['56_Linear.pth', '66_Linear.pth', '68_Linear.pth', '71_Linear.pth', '69_Linear.pth']
downstream_models = [torch.load(os.path.join(model_path, model_name), map_location=device)
                     for model_name in downstream_model_names]

pdb_file = 'C:\\Workspace\\PE\\UI\\saved_models\\1emm_mod.pdb'
atom_lines = utils.process_pdb(pdb_file, atoms_type=['N', 'CA', 'C', 'O'])

coord_array_ca, acid_array, coord_array = utils.extract_coord(atom_lines, atoms_type=['N', 'CA', 'C', 'O'])
utils.compare_len(coord_array, acid_array, ['N', 'CA', 'C', 'O'])
seq_dict = utils.read_fasta(seq_file)

with open(os.path.join(files_path, 'results.txt'), 'a') as f:
    f.write('sequence_name\tpredicted_value\n')


for seq_name, seq in seq_dict.items():
    seq_dict[seq_name] = utils.seq2array(seq)
    array = utils.get_knn_135(coord_array, seq)
    input_ = torch.tensor(array.reshape(-1, 135)).float()
    hidden = pretrained_model(input_).squeeze(1)
    
    output = 0
    for model in downstream_models:
        model.eval()
        model.is_training = False
        output += model(hidden)
    output /= 5
    
    with open(os.path.join(files_path, 'results.txt'), 'a') as f:
        f.write(f'{seq_name}\t{float(output)}\n')
    print(seq_name)

print('Completed')
