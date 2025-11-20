import numpy as np
from torch.utils.data import Dataset
import pickle
import pandas as pd
from scipy.sparse import load_npz
from glob import glob
from math import ceil
from typing import Any, Dict  # put near the top of the file with other imports


# Label mapping: 0 = Null, 1 = Acceptor, 2 = Donor, 3 = Mask (e.g., r==1)
OUT_MAP = np.asarray([[1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1],
                      [0, 0, 0]])


def ceil_div(x, y):
    return int(ceil(float(x) / y))


def getData_multispecies(data_dir, setType, species_list):
    all_annotations = []
    transcriptToLabel_all = {}
    seqData_all = {}

    def _parse_name(name):
        # name is "species---gene---transcript_id"
        parts = name.split('---')
        if len(parts) < 3:
            raise ValueError(
                f"Unexpected name format (expected 'species---gene---transcript'): {name}"
            )
        species = parts[0]
        gene_id = parts[1]
        tx_id = parts[2]
        tx_key = f"{species}---{tx_id}"
        return species, gene_id, tx_key

    for species_name in species_list:
        # 1) transcriptToLabel
        label_path = f"{data_dir}/sparse_discrete_label_data_{species_name}_{setType}.pickle"
        with open(label_path, "rb") as handle:
            tx2lab = pickle.load(handle)
        # Keys are already "{species}---{transcript_id}"
        transcriptToLabel_all.update(tx2lab)

        # 2) annotations
        ann_path = f"{data_dir}/annotations/annotation_{species_name}_{setType}.txt"
        # Columns: name, chrom, strand, tx_start, tx_end, jn_start, jn_end
        ann = pd.read_csv(
            ann_path,
            sep="\t",
            header=None,
            usecols=[0, 1, 2, 3, 4],
        )
        ann.columns = ["name", "chrom", "strand", "tx_start", "tx_end"]

        species_col, gene_col, transcript_col = zip(
            *[_parse_name(n) for n in ann["name"].values]
        )
        ann["species"] = species_col
        ann["gene"] = gene_col
        ann["transcript"] = transcript_col

        ann["chrom"] = ann["chrom"].apply(
            lambda c, s=species_name: f"{s}|{c}"
        )

        ann = ann[["transcript", "gene", "chrom", "strand", "tx_start", "tx_end"]]
        all_annotations.append(ann)

        pattern = f"{data_dir}/sparse_sequence_data/{species_name}_*_{setType}.npz"
        for path in glob(pattern):
            fname = path.split("/")[-1]
            base = fname[:-4] if fname.endswith(".npz") else fname
            suffix = f"_{setType}"
            if not base.endswith(suffix):
                raise ValueError(f"Unexpected npz filename (suffix): {fname}")
            mid = base[: -len(suffix)]  # "{species_name}_{chrom}"

            prefix = f"{species_name}_"
            if not mid.startswith(prefix):
                raise ValueError(f"Unexpected npz filename (prefix): {fname}")

            chrom_name = mid[len(prefix):]  # whatever is after species_
            chrom_key = f"{species_name}|{chrom_name}"
            seqData_all[chrom_key] = load_npz(path).tocsr()

    annotation_all = pd.concat(all_annotations, axis=0, ignore_index=True)
    return annotation_all, transcriptToLabel_all, seqData_all



class DataPointFull:
    def __init__(
        self,
        transcript,
        gene,
        chrom,
        strand,
        start,
        end,
        tx_start,
        tx_end,
        splice_loc,
        splice_type,
        SL,
        CL_max,
        shift,
        mask_l,
        mask_r,
        include_pos=False,
    ):
        self.transcript = transcript
        self.gene = gene
        self.chrom = chrom
        self.strand = strand
        self.start = start
        self.end = end
        self.tx_start = tx_start
        self.tx_end = tx_end
        self.splice_loc = splice_loc
        self.splice_type = splice_type
        self.SL = SL
        self.CL_max = CL_max
        self.shift = shift
        self.mask_l = mask_l
        self.mask_r = mask_r
        self.include_pos = include_pos

    def getData(self, seqData):
        X = np.zeros((4, self.SL + self.CL_max))
        r = np.zeros((self.SL + self.CL_max))
        Y = np.zeros((3, self.SL + self.CL_max))

        if self.include_pos:
            pos = np.arange(self.start - 1, self.end)
            chrm = np.repeat(self.chrom, self.SL)
            transcript = np.repeat(self.transcript, self.SL)

        if self.strand == '+':
            X[:, self.mask_l:self.SL + self.CL_max - self.mask_r] = (
                seqData[self.chrom][
                    self.start - self.CL_max // 2 - 1 + self.mask_l:
                    self.end + self.CL_max // 2 - self.mask_r,
                    :4,
                ]
                .toarray()
                .T
            )
            r[self.mask_l:self.SL + self.CL_max - self.mask_r] = (
                seqData[self.chrom][
                    self.start - self.CL_max // 2 - 1 + self.mask_l:
                    self.end + self.CL_max // 2 - self.mask_r,
                    4,
                ]
                .toarray()[:, 0]
            )
        else:
            X[:, self.mask_r:self.SL + self.CL_max - self.mask_l] = (
                seqData[self.chrom][
                    self.start - self.CL_max // 2 - 1 + self.mask_r:
                    self.end + self.CL_max // 2 - self.mask_l,
                    :4,
                ]
                .toarray()
                .T
            )
            r[self.mask_r:self.SL + self.CL_max - self.mask_l] = (
                seqData[self.chrom][
                    self.start - self.CL_max // 2 - 1 + self.mask_r:
                    self.end + self.CL_max // 2 - self.mask_l,
                    4,
                ]
                .toarray()[:, 0]
            )
            # Reverse for negative strand
            X = X[:, ::-1]
            r = r[::-1]
            X = X[::-1, :]

        Y[0, self.mask_l:(self.SL + self.CL_max - self.mask_r)] = np.ones(
            self.SL + self.CL_max - self.mask_r - self.mask_l
        )
        # Place splice labels (acceptor/donor)
        Y[:, self.splice_loc - self.shift + self.CL_max // 2] = OUT_MAP[
            np.array(self.splice_type, dtype=np.int8)
        ].T
        r_sum = np.sum(r)
        if r_sum > 0:
            Y[:, r == 1] = OUT_MAP[3 * np.ones(int(r_sum), dtype=np.int8)].T

        if self.include_pos:
            return X.copy(), Y.copy(), pos, chrm, transcript
        else:
            return X.copy(), Y.copy()


def getDataPointListFull(annotation, transcriptToLabel, SL, CL_max, shift, include_pos=False):
    data = []
    for idx in range(annotation.shape[0]):
        transcript = annotation['transcript'].values[idx]
        gene = annotation['gene'].values[idx]
        chrom = annotation['chrom'].values[idx]
        strand = annotation['strand'].values[idx]
        tx_start = annotation['tx_start'].values[idx]
        tx_end = annotation['tx_end'].values[idx]

        length = tx_end - tx_start + 1
        num_points = ceil_div(length, shift)

        Y_type, Y_idx = transcriptToLabel[transcript]
        label = [np.array(Y_type), np.array(Y_idx)]

        for i in range(num_points):
            if strand == '+':
                start, end = tx_start + shift * i, tx_start + SL + shift * i - 1
                if i == 0:
                    start_point = start
                inRange = [
                    l >= start - start_point - CL_max // 2
                    and l <= end - start_point + CL_max // 2
                    for l in label[1]
                ]
                mask_l = tx_start - np.min([start - CL_max // 2, tx_start])
                mask_r = np.max([end + CL_max // 2, tx_end]) - tx_end
                data.append(
                    DataPointFull(
                        transcript,
                        gene,
                        chrom,
                        strand,
                        start,
                        end,
                        tx_start,
                        tx_end,
                        label[1][inRange],
                        label[0][inRange],
                        SL,
                        CL_max,
                        start - start_point,
                        mask_l,
                        mask_r,
                        include_pos,
                    )
                )
            else:
                start, end = tx_end - SL - shift * i + 1, tx_end - shift * i
                if i == 0:
                    start_point = end
                inRange = [
                    l >= start_point - end - CL_max // 2
                    and l <= start_point - start + CL_max // 2
                    for l in label[1]
                ]
                mask_l = np.max([end + CL_max // 2, tx_end]) - tx_end
                mask_r = tx_start - np.min([start - CL_max // 2, tx_start])
                data.append(
                    DataPointFull(
                        transcript,
                        gene,
                        chrom,
                        strand,
                        start,
                        end,
                        tx_start,
                        tx_end,
                        label[1][inRange],
                        label[0][inRange],
                        SL,
                        CL_max,
                        start_point - end,
                        mask_l,
                        mask_r,
                        include_pos,
                    )
                )
    return data


class spliceDataset(Dataset):
    def __init__(self, annotation, transform=None, target_transform=None):
        self.annotation = annotation
        self.transform = transform
        self.target_transform = target_transform
        self.seqData: Dict[str, Any] | None = None


    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, idx):
        X, Y = self.annotation[idx].getData(self.seqData)
        if self.transform:
            X = self.transform(X)
        if self.target_transform:
            Y = self.target_transform(Y)
        return X, Y