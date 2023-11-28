from kipoi.data import Dataset
import six
from pyfaidx import Fasta
from pybedtools import Interval
import pandas as pd
import numpy as np
import warnings
import sequence

a_tissues = ['Retina - Eye', 'RPE/Choroid/Sclera - Eye', 'Adipose - Subcutaneous',
           'Adipose - Visceral (Omentum)', 'Adrenal Gland', 'Artery - Aorta',
           'Artery - Coronary', 'Artery - Tibial', 'Bladder', 'Brain - Amygdala',
           'Brain - Anterior cingulate', 'Brain - Caudate nucleus',
           'Brain - Cerebellar Hemisphere', 'Brain - Cerebellum', 'Brain - Cortex',
           'Brain - Frontal Cortex', 'Brain - Hippocampus', 'Brain - Hypothalamus ',
           'Brain - Nucleus accumbens', 'Brain - Putamen',
           'Brain - Spinal cord (C1)', 'Substantia nigra - Brain',
           'Mammary Tissue - Breast', 'Cells - EBV-xform lymphocytes',
           'Cells - Leukemia (CML)', 'Cells - Xform. fibroblasts',
           'Cervix - Ectocervix', 'Cervix - Endocervix', 'Colon - Sigmoid',
           'Colon - Transverse', 'Esophagus - Gastroesoph. Junc.',
           'Esophagus - Mucosa', 'Esophagus - Muscularis', 'Fallopian Tube',
           'Heart - Atrial Appendage', 'Heart - Left Ventricle', 'Kidney - Cortex',
           'Liver', 'Lung', 'Minor Salivary Gland', 'Muscle - Skeletal',
           'Nerve - Tibial', 'Ovary', 'Pancreas', 'Pituitary', 'Prostate',
           'Skin - Not Sun Exposed', 'Skin - Sun Exposed (Lower leg)',
           'Small Intestine - Ileum', 'Spleen', 'Stomach', 'Testis', 'Thyroid',
           'Uterus', 'Vagina', 'Whole Blood']


tissues = ['Retina - Eye', 'RPE/Choroid/Sclera - Eye', 'Subcutaneous - Adipose',
           'Visceral (Omentum) - Adipose', 'Adrenal Gland', 'Aorta - Artery',
           'Coronary - Artery', 'Tibial - Artery', 'Bladder', 'Amygdala - Brain',
           'Anterior cingulate - Brain', 'Caudate nucleus - Brain',
           'Cerebellar Hemisphere - Brain', 'Cerebellum - Brain', 'Cortex - Brain',
           'Frontal Cortex - Brain', 'Hippocampus - Brain', 'Hypothalamus - Brain',
           'Nucleus accumbens - Brain', 'Putamen - Brain',
           'Spinal cord (C1) - Brain', 'Substantia nigra - Brain',
           'Mammary Tissue - Breast', 'EBV-xform lymphocytes - Cells',
           'Leukemia (CML) - Cells', 'Xform. fibroblasts - Cells',
           'Ectocervix - Cervix', 'Endocervix - Cervix', 'Sigmoid - Colon',
           'Transverse - Colon', 'Gastroesoph. Junc. - Esophagus',
           'Mucosa - Esophagus', 'Muscularis - Esophagus', 'Fallopian Tube',
           'Atrial Appendage - Heart', 'Left Ventricle - Heart', 'Cortex - Kidney',
           'Liver', 'Lung', 'Minor Salivary Gland', 'Skeletal - Muscle',
           'Tibial - Nerve', 'Ovary', 'Pancreas', 'Pituitary', 'Prostate',
           'Not Sun Exposed - Skin', 'Sun Exposed (Lower leg) - Skin',
           'Ileum - Small Intestine', 'Spleen', 'Stomach', 'Testis', 'Thyroid',
           'Uterus', 'Vagina', 'Whole Blood']

bases = ['A', 'C', 'G', 'T']

class ExonInterval(Interval):
    ''' Encode exon logic
    '''

    def __init__(self,
                 length=600,
                 intron_start=None,
                 intron_end=None,
                 fasta=None,
                 **kwargs):
        ''' intron_start, intron_end are the start and end of the flanking introns of the given exon.
            intron_start is the start position on the left, intron_end is the end position on the right
        '''
        super().__init__(**kwargs)
        self.exon_start = self.start
        self.exon_end = self.end
        self.intron_start = intron_start
        self.intron_end = intron_end
        self.l = length
        self.fasta = fasta

    def getseq(self, start, end):
        seq = self.fasta.get_seq(self.chrom, start, end, self.strand == '-')
        ## TODO: pad or crop
        seq = seq.seq.upper()
        return seq

    def sequence(self):
        ''' Return padded or croped sequence with the same length
        '''
        exon_length = self.exon_end - self.exon_start + 1
        exon_intron_length = self.intron_end - self.intron_start + 1

        if exon_length > self.l:
            # start croping
            # + 63 because MMSplice takes 50 and 13 base in the intron
            crop_length = exon_length - self.l
            cutting_point = int((self.exon_end + self.exon_start) / 2)
            crop_left = int(crop_length / 2)
            crop_right = crop_length - crop_left
            if self.strand == "+":
                # add the required intron length
                crop_left += 50
                crop_right += 13
                seq_l = self.getseq(self.exon_start - 50, cutting_point - crop_left - 1)
                seq_r = self.getseq(cutting_point + crop_right, self.exon_end + 13)
                seq = seq_l + seq_r
            else:
                crop_left += 13
                crop_right += 50
                seq_l = self.getseq(self.exon_start - 13, cutting_point - crop_left - 1)
                seq_r = self.getseq(cutting_point + crop_right, self.exon_end + 50)
                seq = seq_r + seq_l
        elif exon_intron_length > self.l:
            crop_length = exon_intron_length - self.l
            # -2 to preserve dinucleotides
            if self.exon_start - self.intron_start < self.intron_end - self.exon_end:
                crop_left = min(int(crop_length / 2), self.exon_start - self.intron_start - 2)
                crop_right = crop_length - crop_left
                #assert crop_right > 0
            else:
                crop_right = min(int(crop_length / 2), self.intron_end - self.exon_end - 2)
                crop_left = crop_length - crop_right
                #assert crop_left > 0
            seq = self.getseq(self.intron_start + crop_left, self.intron_end - crop_right)

            # test
            _seq = self.getseq(self.exon_start, self.exon_end)
            if _seq in seq:
                warnings.warn("Seq does not contain the whole exon")

        else:
            pad_length = self.l - exon_intron_length
            pad_left = int(pad_length / 2)
            pad_right = pad_length - pad_left
            seq = self.getseq(self.intron_start, self.intron_end)
            seq = pad_left * "N" + seq + pad_right * "N"

            # test
            _seq = self.getseq(self.exon_start, self.exon_end)
            assert _seq in seq

        assert len(seq) == self.l
        return seq

def onehot(seq):
    X = np.zeros((len(seq), len(bases)))
    for i, char in enumerate(seq):
        if char == "N":
            pass
        else:
            X[i, bases.index(char.upper())] = 1
    return X

def logit(x):
    x = clip(x)
    return np.log(x) - np.log(1 - x)

def clip(x):
    return np.clip(x, 1e-5, 1-1e-5)

class Ascot(Dataset):
    ''' Load Acsot exons and PSI across tissues
        * Take the exon and flanking introns
        * Pad length to L.
        * If exon longer than L, cut out from the middle
        * If exon + flanking introns longer than L, cut the flanking intron
    '''

    def __init__(self,
                 ascot,
                 fasta_file,
                 length=900,
                 tissues=tissues,
                 encode=True,
                 pad_trim_same_l=True,
                 flanking=300,
                 flanking_exons=False,
                 region_anno=False,
                 seq_align='start',
                 mean_inpute=True,
                 use_logit=False):
        '''Args:
            - ascot: ascot psi file
            - fasta_file: fasta format file path
            - pad_trim_same_l: when True, trim sequence from middle of exon.
                           If false, return exon + flanking intron sequence of both sides
                           maxlen = flanking*2 + 300
            - flanking: length of intron to take, only effective when pad_trim_same_l=False
            - encode: whether encode sequence
            - region_anno: return binary indicator on region annotation
            - seq_align: if "both", return two sequence, one align from start, one from end
            - use_logit: if True, return target PSI in logits
            - flanking_exons: if True, return flanking exon sequence as well. e.g 100bp exon and 300bp intron
        '''

        if isinstance(fasta_file, six.string_types):
            fasta = Fasta(fasta_file, as_raw=False)
        self.fasta = fasta
        self.L = length
        self.tissues = tissues
        self.pad_trim_same_l = pad_trim_same_l
        self.encode = encode
        self.flanking = flanking
        if isinstance(flanking, tuple):
            self.flanking_l = flanking[0]
            self.flanking_r = flanking[0]
        self.exons, self.PSI, self.mean = self.read_exon(ascot)
        self.region_anno = region_anno
        self.seq_align = seq_align
        self.mean_inpute = mean_inpute
        if seq_align == 'both' and not encode:
            assert "When not encode sequence, only return one input sequence string"
        self.use_logit = use_logit
        self.flanking_exons = flanking_exons

    def __len__(self):
        return len(self.exons)

    def read_exon(self, ascot):
        exons = pd.read_csv(ascot, index_col=0)
        PSI = exons[self.tissues].values
        exons = exons[['chrom',
                       'exon_start',
                       'exon_end',
                       'intron_start',
                       'intron_end',
                       'strand',
                       'exon_id',
                       'gene_id']]
        PSI[PSI == -1] = np.nan
        m = np.nanmean(PSI, axis=1)
        m = m[:, np.newaxis]
        if np.mean(m) > 1:
            PSI = PSI / 100.
            m = m / 100.
        return exons, PSI, m

    def get_seq(self, exon):
        exon = ExonInterval(chrom=exon.chrom,
                            length=self.L,
                            start=exon.exon_start,
                            end=exon.exon_end,
                            strand=exon.strand,
                            intron_start=exon.intron_start,
                            intron_end=exon.intron_end,
                            fasta=self.fasta)
        return exon.sequence()

    def __getitem__(self, idx):
        exon = self.exons.iloc[idx]
        from copy import deepcopy
        psi = deepcopy(self.PSI[idx])
        m = self.mean[idx]

        # about sample weight:
        # (np.nanvar(psi) / np.squeeze(m)) * 100, (X)
        # std = (np.var(psi_copy) / np.squeeze(m)) * 100, psi_copy: mean inputed psi
        # np.var(psi_copy) * 100: OK, cor(sum_PSI_pred, NA frac) high

        # copy to compute var or std for sample weight
        psi_copy = deepcopy(psi)
        assert np.sum(psi_copy == -1.) == 0

        psi_copy[np.isnan(psi_copy)] = m
        std = min(np.var(psi_copy) * 100, 4)
        if self.mean_inpute:
            psi = psi_copy  # mean inpute

        out = {}
        if self.use_logit:
            psi = logit(psi)
        # convert back to -1
        #psi[np.isnan(psi)] = -1.
        out["targets"] = psi
        out["inputs"] = {}
        if self.pad_trim_same_l:
            seq = "N" * 50 + self.get_seq(exon) + "N" * 13
            if self.encode:
                seq = onehot(seq)
            out["inputs"]["seq"] = seq
        else:
            seq = self.fasta.get_seq(exon.chrom,
                                     exon.exon_start - self.flanking,
                                     exon.exon_end + self.flanking,
                                     exon.strand == '-')
            seq = seq.seq.upper()
            
            out['inputs']['fasta'] = seq
            if self.flanking_exons:
                if exon.strand == "+":
                    exon_up = self.fasta.get_seq(exon.chrom,
                                         exon.intron_start - self.L + self.flanking,
                                         exon.intron_start + self.flanking - 1,
                                         exon.strand == '-')
                    exon_up = exon_up.seq.upper()
                    exon_dw = self.fasta.get_seq(exon.chrom,
                                         exon.intron_end - self.flanking + 1,
                                         exon.intron_end + self.L - self.flanking,
                                         exon.strand == '-')
                    exon_dw = exon_dw.seq.upper()
                else:
                    exon_up = self.fasta.get_seq(exon.chrom,
                                         exon.intron_end - self.flanking + 1,
                                         exon.intron_end + self.L - self.flanking,
                                         exon.strand == '-')
                    exon_up = exon_up.seq.upper()
                    exon_dw = self.fasta.get_seq(exon.chrom,
                                         exon.intron_start - self.L + self.flanking,
                                         exon.intron_start + self.flanking - 1,
                                         exon.strand == '-')
                    exon_dw = exon_dw.seq.upper()

            if self.encode:
                # from mtsplice.utils.utils import HiddenPrints
                # with HiddenPrints():
                if self.seq_align == 'both':
                    seql = sequence.encodeDNA([seq], maxlen=self.L, seq_align='start')[0]
                    seqr = sequence.encodeDNA([seq], maxlen=self.L, seq_align='end')[0]
                    out["inputs"]["seql"] = seql
                    out["inputs"]["seqr"] = seqr
                    if self.flanking_exons:
                        out["inputs"]["exon_up"] = sequence.encodeDNA([exon_up])[0]
                        out["inputs"]["exon_dw"] = sequence.encodeDNA([exon_dw])[0]
                else:
                    seq = sequence.encodeDNA([seq], maxlen=self.L, seq_align=self.seq_align)[0]
                    out["inputs"]["seq"] = seq
            else:
                out["inputs"]["seq"] = seq
            if self.region_anno:
                anno = anno_region(self.flanking,
                                   exon.exon_end - exon.exon_start + 1,
                                   self.L, align=self.seq_align)
                out["inputs"]["anno"] = anno
        out["inputs"]["mean"] = np.repeat(logit(m), 56)
        out["inputs"]["std"] = std
        out['metadata'] = {}
        out['metadata']['chrom'] = exon.chrom
        out['metadata']['exon_id'] = exon.exon_id
        out['metadata']['exon_start'] = exon.exon_start
        out['metadata']['exon_end'] = exon.exon_end
        out['metadata']['intron_start'] = exon.intron_start
        out['metadata']['intron_end'] = exon.intron_end
        out['metadata']['strand'] = exon.strand
        return (out['inputs'],out['targets'])