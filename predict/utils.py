import numpy as np
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from math import ceil

AA_ALPHABET = {'A': 'ALA', 'F': 'PHE', 'C': 'CYS', 'D': 'ASP', 'N': 'ASN',
               'E': 'GLU', 'Q': 'GLN', 'G': 'GLY', 'H': 'HIS', 'L': 'LEU',
               'I': 'ILE', 'K': 'LYS', 'M': 'MET', 'P': 'PRO', 'R': 'ARG',
               'S': 'SER', 'T': 'THR', 'V': 'VAL', 'W': 'TRP', 'Y': 'TYR'}
AA_ALPHABET_REV = {'ALA': 'A', 'PHE': 'F', 'CYS': 'C', 'ASP': 'D', 'ASN': 'N',
                   'GLU': 'E', 'GLN': 'Q', 'GLY': 'G', 'HIS': 'H', 'LEU': 'L',
                   'ILE': 'I', 'LYS': 'K', 'MET': 'M', 'PRO': 'P', 'ARG': 'R',
                   'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'}
AA_NUM = {'A': 0, 'F': 1, 'C': 2, 'D': 3, 'N': 4,
          'E': 5, 'Q': 6, 'G': 7, 'H': 8, 'L': 9,
          'I': 10, 'K': 11, 'M': 12, 'P': 13, 'R': 14,
          'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19}
AA_HYDROPATHICITY_INDEX = {'R': -4.5, 'K': -3.9, 'N': -3.5, 'D': -3.5, 'Q': -3.5,
                           'E': -3.5, 'H': -3.2, 'P': -1.6, 'Y': -1.3, 'W': -0.9,
                           'S': -0.8, 'T': -0.7, 'G': -0.4, 'A': 1.8, 'M': 1.9,
                           'C': 2.5, 'F': 2.8, 'L': 3.8, 'V': 4.2, 'I': 4.5}
AA_BULKINESS_INDEX = {'R': 14.28, 'K': 15.71, 'N': 12.82, 'D': 11.68, 'Q': 14.45,
                      'E': 13.57, 'H': 13.69, 'P': 17.43, 'Y': 18.03, 'W': 21.67,
                      'S': 9.47, 'T': 15.77, 'G': 3.4, 'A': 11.5, 'M': 16.25,
                      'C': 13.46, 'F': 19.8, 'L': 21.4, 'V': 21.57, 'I': 21.4}
AA_FLEXIBILITY_INDEX = {'R': 2.6, 'K': 1.9, 'N': 14., 'D': 12., 'Q': 4.8,
                        'E': 5.4, 'H': 4., 'P': 0.05, 'Y': 0.05, 'W': 0.05,
                        'S': 19., 'T': 9.3, 'G': 23., 'A': 14., 'M': 0.05,
                        'C': 0.05, 'F': 7.5, 'L': 5.1, 'V': 2.6, 'I': 1.6}
AA_PROPERTY = {}


for aa in AA_HYDROPATHICITY_INDEX.keys():
    AA_PROPERTY.update({aa: [(5.5 - AA_HYDROPATHICITY_INDEX[aa]) / 10,
                             AA_BULKINESS_INDEX[aa] / 21.67,
                             (25. - AA_FLEXIBILITY_INDEX[aa]) / 25.]})
    aa_long = AA_ALPHABET[aa]
    AA_PROPERTY.update({aa_long: [(5.5 - AA_HYDROPATHICITY_INDEX[aa]) / 10,
                                  AA_BULKINESS_INDEX[aa] / 21.67,
                                  (25. - AA_FLEXIBILITY_INDEX[aa]) / 25.]})


def read_fasta(file_path):
    seq_dict = {}  # {seq_name -> fasta_seq}
    seq_file = open(file_path, 'r')
    seq_data = seq_file.readlines()
    # 读取fasta文件

    for i in range(len(seq_data)):
        if seq_data[i][0] == '>':
            seq_name = seq_data[i][1:-1]
            seq_dict[seq_name] = ''
            j = 1
            while True:
                if i + j >= len(seq_data) or seq_data[i + j][0] == '>':
                    break
                else:
                    seq_dict[seq_name] += ''.join(seq_data[i + j].split())
                j += 1
    return seq_dict


def process_pdb(file_name, atoms_type):
    atom_lines = []  # (chain, model) -> atom_lines

    file = open(file_name, 'r')
    pdb_lines = file.readlines()

    for line in pdb_lines:
        line_split = line.split()

        if line_split[0] == 'ATOM' and line_split[2] in atoms_type:
            if len(line_split[4]) > 1:
                line_split.insert(4, line_split[4][0])
                line_split[5] = line_split[5][1:]
            try:
                atom_lines.append(line_split)
            except KeyError:
                atom_lines = [line_split]

    return atom_lines


def compare_len(coord_array, aa_array, atoms_type):
    atoms = len(atoms_type)
    print(coord_array.shape[0] / atoms, aa_array.shape[0])
    if coord_array.shape[0] / atoms > aa_array.shape[0]:
        raise("Seq too short!")
    elif coord_array.shape[0] / atoms < aa_array.shape[0]:
        raise("Seq too long!")


def seq2array(aa_seq: str) -> np.ndarray:
    aa_seq = list(aa_seq)
    for i in range(len(aa_seq)):
        aa_seq[i] = AA_ALPHABET[aa_seq[i]]
    aa_seq_array = np.array(aa_seq)
    return aa_seq_array


def extract_coord(atoms_data, atoms_type):
    coord_array_ca = np.zeros((ceil(len(atoms_data) / len(atoms_type)), 3))  # CA坐标, shape: L * 3
    coord_array_all = np.zeros((len(atoms_data), 3))  # 所有backbone原子(或atom_types中的原子)坐标, shape: 4L * 3
    aa_names = []
    for i in range(len(atoms_data)):
        coord_array_all[i] = [float(atoms_data[i][j]) for j in range(6, 9)]
        # 写法可能不合适,未考虑氨基酸内部原子顺序不一致的情况
        if i % len(atoms_type) == atoms_type.index('CA'):
            coord_array_ca[i // len(atoms_type)] = [float(atoms_data[i][j]) for j in range(6, 9)]
            aa_names.append(atoms_data[i][3][-3::])
    aa_names_array = np.array(aa_names)  # shape: L * 1
    return coord_array_ca, aa_names_array, coord_array_all


def MapDis(coo):
    return squareform(pdist(coo, metric='euclidean')).astype('float32')


def get_len(vec):
    return np.linalg.norm(vec, axis=-1)


def norm(vec):
    return vec / get_len(vec).reshape(-1, 1)


def batch_cos(vecs1, vecs2):
    cos = np.diag(np.matmul(norm(vecs1), norm(vecs2).T))
    cos = np.clip(cos, -1, 1)
    return cos


def get_torsion(vec1, vec2, axis):
    n = np.cross(axis, vec2)
    n2 = np.cross(vec1, axis)
    sign = np.sign(batch_cos(vec1, n))
    angle = np.arccos(batch_cos(n2, n))
    torsion = sign * angle
    if len(torsion) == 1:
        return torsion[0]
    else:
        return torsion


def coo2tor(coo):
    ca_c = (coo[2::4] - coo[1::4])[:-1]
    ca_n = (coo[::4] - coo[1::4])[1:]
    ca_ca = (coo[1::4][1:] - coo[1::4][:-1])

    tor_c = get_torsion(ca_ca[1:], ca_c[:-1], ca_ca[:-1])
    tor_n = get_torsion(ca_ca[1:], ca_n[:-1], ca_ca[:-1])

    tor_last_c = get_torsion(-ca_ca[-2], ca_c[-1], ca_ca[-1])
    tor_last_n = get_torsion(-ca_ca[-2], ca_n[-1], ca_ca[-1])

    tor_c = np.concatenate([tor_c, [tor_last_c]])
    tor_n = np.concatenate([tor_n, [tor_last_n]])

    tor = [tor_c, tor_n]
    return np.array(tor, dtype='float32').T


def tor2sincos(tor):
    sin = np.sin(tor)
    cos = np.cos(tor)
    sincos = np.array((sin[:, 0], cos[:, 0], sin[:, 1], cos[:, 1]))
    return sincos.astype('float32')


def KNNStructRepRelative(ca, seq, k=15, index_norm=200):
    dismap = MapDis(ca)
    nn_indexs = np.argsort(dismap, axis=1)[:, :k]
    relative_indexs = nn_indexs.reshape(-1, k, 1) - \
                      nn_indexs[:, 0].reshape(-1, 1, 1).astype('float32')
    relative_indexs /= index_norm
    seq_embeded = []
    for aa in seq:
        seq_embeded.append(AA_PROPERTY[aa])
    knn_feature = np.array(seq_embeded)[nn_indexs]
    knn_distance = [dismap[i][nn_indexs[i]] for i in range(len(nn_indexs))]
    knn_distance = np.array(knn_distance).reshape(-1, k, 1)
    print(knn_distance.shape, relative_indexs.shape)
    knn_rep = np.concatenate((knn_distance, relative_indexs, knn_feature), -1)
    return knn_rep.astype('float32')


def get_knn_135(coo, aa, k=15):
    if len(coo) % 4 != 0:
        raise Exception('Absence of certain atoms!')
    arrays = coo
    tor_arrays = tor2sincos(coo2tor(coo))
    tor_arrays = tor_arrays.transpose()
    print(tor_arrays.shape, coo.shape)
    tor_arrays = np.concatenate((tor_arrays, np.zeros((1, 4))), axis=0)
    ca_coo = []
    for i in range(len(arrays)):
        if i % 4 == 1:
            ca_coo.append(arrays[i])
    ca_coo = np.array(ca_coo)
    arrays = np.concatenate((ca_coo, tor_arrays), axis=1)
    structure_feature = []
    for i, array in enumerate(arrays):
        dic = {}
        list_coord = []
        for j, arr in enumerate(arrays):
            x = arrays[j][:3] - arrays[i][:3]
            dis = np.linalg.norm(x, axis=0, keepdims=True)[0]
            a, b, c, d = arrays[j][3:]
            tor_c = np.arctan2(a, b)
            tor_n = np.arctan2(c, d)
            tor_avg = (tor_c + tor_n) / 2
            dic[dis] = list(x) + [tor_avg]
        dic1 = sorted(dic.items(), key=lambda x: x[0], reverse=False)[:15]
        for key in dic1:
            list_coord.append(key[1])
        structure_feature.append(list_coord)

    arrays_orgin = KNNStructRepRelative(ca_coo, aa, k=k)
    print(arrays_orgin.shape, np.array(structure_feature).shape)
    new_feature = np.concatenate((arrays_orgin, np.array(structure_feature)), axis=2)
    return new_feature
