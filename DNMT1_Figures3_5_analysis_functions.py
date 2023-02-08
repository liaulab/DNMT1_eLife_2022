# -*- coding: utf-8 -*-
"""
@author: kevin

Functions to perform data processing and analysis
"""
#%% import packages

import os, re, sys, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
import seaborn as sns

import scipy.stats as stats
import scipy.spatial.distance as dist
import scipy.interpolate as interp
import scipy.optimize as sp_opt
import scipy.cluster as sp_cl

import statsmodels.api as sm
import statsmodels.stats.multitest as smm
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.sandbox.regression.predstd import wls_prediction_std

from Bio.Data import IUPACData
from Bio import Align
from Bio import SeqIO
from Bio.Align import substitution_matrices as sm
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

#%% internal functions

def col_check(df, list_reqcols, df_name):
    list_dfcols = df.columns.tolist()
    # check that the df has the required columns
    if not all(col in list_dfcols for col in list_reqcols):
        list_miss = [col for col in list_reqcols if col not in list_dfcols]
        raise Exception(df_name + ' is missing required column(s): ' + str(list_miss))
    return

def convert_type(in_var, type1, type2, var_name):
    if isinstance(in_var, type1):
        if type2 == list:
            in_var = [in_var]
        else:
            in_var = type2(in_var)
    elif isinstance(in_var, type2):
        pass
    else:
        t1, t2 = str(type1.__name__), str(type2.__name__)
        raise Exception('Invalid value for ' + var_name + '. Must be ' + t1 + ' or ' + t2)
    return in_var

def save_figs(fig, out_fig, out_folder, save_as, savefig_kws=dict(bbox_inches='tight', dpi=300)):
    # create the output directory if doesn't exist
    outpath = Path.cwd() / out_folder
    Path.mkdir(outpath, exist_ok=True)
    # export files; if save_as is None, skip export
    if save_as is None:
        print('The value of save_as is None. No figures were saved.')
        pass
    else:
        # convert to list if save_as is str, then save for all ftypes in save_as
        save_as = convert_type(save_as, str, list, 'save_as')
        for ftype in save_as:
            out_plot = outpath / (out_fig + '.' + ftype)
            fig.savefig(out_plot, format=ftype, **savefig_kws)
        print ('All figures saved as ' + str(save_as))
    return

def map_positions(sequence, positions, regex='[ACGTN]'):
    """
    Inputs: sequence (str) and positions (list or iterable)
    Given sequence and desired positions, returns a tuple of new positions
    that correspond to their new positions post alignment
    """
    alignment = [m.start() for m in re.finditer(regex, sequence, flags=re.IGNORECASE)]
    list_newpositions = []
    for position in positions:
        new_position = alignment[position]
        list_newpositions.append(new_position)
    return list_newpositions

def dna(sequence, rc=False):
    """
    gets really tedious to write Seq(x, IUPAC.ambiguous_dna) every time I need
    to convert a string, so just making a function to abbreviate
    (is this called a wrapper?)
    rc == reverse complement. if true, return reverse complement
    """
    if rc:
        dna_sequence = Seq(sequence).reverse_complement()
    else:
        dna_sequence = Seq(sequence)
    return dna_sequence

def tln(sequence, avoid=None, as_str=True):
    """
    making a wrapper for the biopython translate to ignore non-sequences
    avoid = string to avoid for translation (e.g. "no_tln" or "mismatch")
    if sequence == avoid, return sequence without translating
    """
    if isinstance(sequence, str):
        if avoid is not None and sequence == avoid:
            seq = sequence
        else:
            seq = dna(sequence).translate(gap='-')
            if as_str:
                seq = str(seq)
            return seq
    elif np.isnan(sequence):
        seq = sequence
        return seq
    else:
        raise Exception('Sequence was not str or NaN')

def check_seqs(sequence, short_error='too_short', frame_error='not_in_frame',
               N_error='has_N', pass_check='tln_ok'):
    """
    making a wrapper to check a DNA sequence string for codon translation
    if seq has Ns, returns 'has_N', if seq is not divisible by 3, returns 'not_in_frame'
    if seq has partial codons (e.g. codon with dash), returns 'partial_cdn'
    otherwise, if seq passes all checks, returns 'tln_ok'
    """
    # check if sequence is greater than 3 nt, if not, return short_error
    if len(sequence) < 3:
        return short_error
    # check if sequence is multiple of 3, if not, return frame_error
    if len(sequence) % 3 != 0:
        return frame_error
    # check if sequence contains Ns, as these are not suitable for codon alignment
    if 'N' in sequence:
        return N_error
    # if sequence passes all the checks, return pass_check
    return pass_check

def get_align_objs(aligner, seqA, seqB, codons=False, return_last_aln_only=True):
    """
    wrapper function to perform biopython pairwise alignments because it's really
    hard to use biopython pairwise aligners in a pd.apply() and retrieve multiple
    things (e.g. alignment, n_alignments, score, seqs) without repeating pd.apply()
    """
    alignments = aligner.align(seqA, seqB) # align the two sequences
    n_alignments = len(alignments) # get the number of total alignments
    # get the first alignment and aligned/reference seqs
    aln1 = alignments[0]
    aln1_seqA = aln1.__str__().split('\n')[0]
    aln1_seqB = aln1.__str__().split('\n')[2]
    if codons: # remove whitespace if codons
        aln1_seqA = aln1_seqA.replace(' ', '')
        aln1_seqB = aln1_seqB.replace(' ', '')
    # if n_alignments > 1, also get last alignment (tends to be left-aligned/have right-most gaps)
    if n_alignments > 1:
        aln2 = alignments[n_alignments - 1]
        aln2_seqA = aln2.__str__().split('\n')[0]
        aln2_seqB = aln2.__str__().split('\n')[2]
        if codons: # remove whitespace if codons
            aln2_seqA = aln2_seqA.replace(' ', '')
            aln2_seqB = aln2_seqB.replace(' ', '')
    else: # otherwise, just set aln2_seqA/aln2_seqB to NaN
        aln2_seqA, aln2_seqB = np.nan, np.nan
    # if return last aln only == True, then return a tuple of (n_alignments, last aln seqA/seqB)
    # otherwise, re
    # often use this b/c last alignment is usually left-aligned (desired), reduces amt of crap returned
    if return_last_aln_only:
        if n_alignments == 1: # if only 1 alignment, return the only alignment
            return (n_alignments, aln1_seqA, aln1_seqB)
        else: # otherwise, returns the last alignment
            return (n_alignments, aln2_seqA, aln2_seqB)
    # otherwise, return a tuple of 5 (first/last alignment seqs)
    else:
        return (n_alignments, aln1_seqA, aln1_seqB, aln2_seqA, aln2_seqB)

def codon_submat(nt_matrix='NUC.4.4'):
    """
    Generates a codon substitution matrix for pairwise alignment scoring during
    codon alignment. The default uses the nucleotide scores from NUC.4.4 (EDNAFULL).
    """
    dna_align = Align.PairwiseAligner()
    # use the specified substitution matrix for scoring
    dna_align.substitution_matrix = sm.load(nt_matrix)
    # set gap penalty high to prevent gaps; we want only substitutions
    dna_align.open_gap_score = -100
    # get codon alphabet from existing SCHNEIDER matrix
    cdn_alphabet = sm.load('SCHNEIDER').alphabet
    # generate new Array with 2 dimensions and codon alphabet
    cdn_submat = sm.Array(alphabet=cdn_alphabet, dims=2)
    # perform a nested for loop to align all codons with dna aligner and write scores to list
    for cdn1 in cdn_alphabet:
        for cdn2 in cdn_alphabet:
            cdn_submat[cdn1, cdn2] = dna_align.align(cdn1, cdn2).score
    return cdn_submat

def fill_gaps(seqA, seqB, regex='[^-]', sub='-', codon=False):
    """
    wrapper function to map gaps from seqA (e.g. aln_seq after codon alignment)
    to seqB (e.g. aln_wdw seq with no gaps, adjusted to length of aln_nogap/acw_nogap)
    """
    # seqA = alnmap (aln_cdn mapped) and seqB = wdw_nogap (window with no gaps, adj to size of aln_nogap)
    # if seqB is not a list of codons, convert str to list
    if not codon:
        seqB = list(seqB)
    # find positions of all non-gaps in seqA (post-alignment sequence with gaps)
    seqA_idxs = [m.start() for m in re.finditer(regex, seqA)]
    # if no gaps in seqA, then just return seqB without modification
    if len(seqA_idxs) == 0:
        seqB = ''.join(seqB)
        return seqB
    # make a dict of seqA idx with seqB chars
    idx_map = {idx:char for idx,char in zip(seqA_idxs, seqB)}
    # fill gaps in idxs using list comprehension
    seqB_gapped = [idx_map[idx] if idx in idx_map.keys() else sub for idx in range(len(seqA))]
    seqB_gapped = ''.join(seqB_gapped)
    # return seqB with gaps added
    return seqB_gapped

def find_gaps(seqA, seqB, replace=True):
    """
    wrapper function to map the gap positionss from seqA (e.g. aln_cds seq) to
    seqB (e.g. aln_wdw seq adjusted to length of aln_cds), and replaces any Xs
    with gaps so that stripping the wdw_seq gaps will work correctly
    """
    # seqA is aln_cds seq, seqB is the wdw_adj seq
    gapsA = [m.start() for m in re.finditer('-', seqA)]
    gapsB = [m.start() for m in re.finditer('-', seqB)]
    gap_mismatches = [idx for idx in gapsA if idx not in gapsB]
    # if no mismatched gaps, then set gap_int to None
    if len(gap_mismatches) == 0:
        gap_int = None
    else:
        gap_int = (min(gap_mismatches), max(gap_mismatches))
    # if replace is True, replace all mismatches (i.e. 'X') with gaps and return new seq
    if replace:
        # if no mismatched gaps, just return seqB without changes
        if gap_int is None:
            return seqB
        else:
            seqB_st = seqB[:gap_int[0]] # seqB until first gap mismatch
            seqB_end = seqB[gap_int[1] + 1:] # seqB after last gap mismatch
            # replace all 'X' in seqB from first gap to last gap
            seqB_gaps = seqB[gap_int[0]:gap_int[1] + 1].replace('X','-')
            # rejoin seqB and return
            seqB_adj = seqB_st + seqB_gaps + seqB_end
            return seqB_adj
    # if replace is not True, then just return the gap interval
    else:
        return gap_int

#%% analyze_alleles() - translate CRISPResso alleles into protein variants

def analyze_alleles(sample, in_ref, in_batch, cwd=Path.cwd(), window=None, reverse=False, return_rev=False,
                    save=True, save_intermediate=False, out_folder='', out_file=None, return_df=None):
    """
    Translate alleles from CRISPResso2 output into protein variants and merge.

    Takes the CRISPResso2 raw reads table and translates all in-frame mutations
    into protein variants. The translated alleles are then merged with the
    CRISPResso2 allele frequency around sgRNA table and exported to csv.

    Parameters
    ----------
    sample : str
        The name of the sample to analyze. Must be present in the 'sample_name'
        and 'name' columns of the in_ref and in_batch file, respectively.
    in_ref : str, Path object, or dict
        Str or path to the reference file, which must have the columns 'sample_name'
        and 'CDS_frame'. Alternatively, a dict can be passed with the columns as
        keys. The 'sample_name' column must correspond
        to the 'name' column of the in_batch file. The 'CDS_frame' column specifies
        the # of nt in the coding_seq before the first complete codon in the amplicon.
        Example: For a CDS_seq 'CAATGTTTTCA...' with frame (CA/ATG/TTT/TCA/...),
        the CDS_frame == 2 because there are 2 nt before the first full codon (ATG).
    in_batch : str, Path object, or dict
        String or path to the batch file that was used for the CRISPResso2 analysis.
        The in_batch file must have columns for the sample name ('name'), the name
        of the amplicon ('amplicon_name'), the amplicon sequence ('amplicon_seq'),
        the gRNA sequence ('guide_seq'), and the CDS within the amplicon ('coding_seq').
        Alternatively, a dict can be passed with the columns as keys.
    cwd : str or Path object, default Path.cwd()
        String or path to the directory where the CRISPResso2 output files are stored.
        The default is the current working directory (cwd). If the CRISPResso2 output
        directory is not the cwd, then this parameter must be specified. If a string
        is passed, it is assumed to be a subfolder of the cwd. If a Path object is
        passed, it is assumed to be a direct path.
    window : int, default None
        The plot window size used for the CRISPResso2 analysis, corresponding
        to the --plot_window_size parameter. If None, the function will infer
        the window size from the length of the WT reference sequence in the
        CRISPResso allele frequency table around the cut site.
    reverse : bool, default False
        Whether the sample was sequenced in the reverse orientation of the CDS.
        Reversed amplicons are processed by CRISPResso2 in that orientation, which
        requires different processing in order to return the correct translations.
        If reverse is True, then the CDS frame should be given as the # of extra nt
        in the most C-terminal codon (i.e. the extra nt in the 5' of the read amplicon)
    return_rev : bool, default False
        Whether to return the reverse complement of the processed alleles (df_final).
        Only applies when reverse is True (samples sequenced in the reverse orientation).
    save : bool, default True
        Whether to save the final translated allele df (df_final) as a csv file.
        The default is True.
    save_intermediate : bool, default False
        Whether to save all of the intermediate dataframes as an .xlsx file with
        each dataframe as a separate sheet. The intermediate dataframes are df_stats,
        df_agg, df_merge, df_mismatch, df_raw.
    out_folder : str or Path object, default ''
        Name of the subfolder to save the output file. The default is the current
        working directory. If a string is passed, it is assumed to be a subfolder
        of the cwd. If a Path object is passed, it is assumed to be the direct
        path to the output folder
    out_file : str, default None
        Name of the output csv file. If specified, the string must include the
        .csv extension. The default is None, which is sample + '_alleles_tln_freq.csv'.
    return_df : {None, 'final', 'in-frame', 'raw', 'all'}, default None
        Whether to return a dataframe at function end. 'final' will return the
        final aggregated alleles df (df_final), 'in-frame' will return the in-frame
        translations df (df_if), and 'raw' will return the raw reads with positions
        and translations df (df_raw). The default is None, which returns stats,
        which are the # of alleles in the CRISPResso output, # of alleles aggregated
        by the function, # of alleles after merging the two dataframes, and the
        # of mismatched alleles (present in one df but not the other). If a list
        or 'all' is passed, then it will return multiple dataframes in a dictionary
        with the key as the return_df str and the df as the value.
    """
    # !!!
    t_start = time.perf_counter()

    ### SET UP BIOPYTHON PAIRWISE ALIGNERS W/ EMBOSS NEEDLE DEFAULT PARAMS

    # dna alignment uses the NUC.4.4 (EDNAFULL) substitution matrix
    # only used to align the cds to the amplicon, so don't penalize open end gap
    dna_align = Align.PairwiseAligner()
    dna_align.substitution_matrix = sm.load('NUC.4.4')
    dna_align.open_gap_score = -10 # penalty for opening a gap
    dna_align.end_open_gap_score = 0
    dna_align.extend_gap_score = -0.5 # penalty for extending gap

    # protein alignment uses the BLOSUM62 substitution matrix
    # gap penalties based on EMBOSS Needle defaults (except we penalize end gaps)
    prot_align = Align.PairwiseAligner()
    prot_align.substitution_matrix = sm.load('BLOSUM62')
    prot_align.open_gap_score = -10 # penalty for opening a gap (including end)
    prot_align.extend_gap_score = -0.5 # penalty for extending gap (including end)

    # codon alignment with the NUC.4.4 (EDNAFULL) substitution matrix
    # gap penalties based on CRISPResso defaults
    cdn_align = Align.PairwiseAligner()
    cdn_align.substitution_matrix = codon_submat()
    cdn_align.open_gap_score = -60 # penalty for opening a gap (including end)
    cdn_align.extend_gap_score = -6 # penalty for extending a gap (including end)

    ### IMPORT REFERENCE AND BATCH FILES AND CLEAN UP

    # import reference and batch files, perform column header check
    # if in_ref and/or in_batch are dicts, then convert to dataframe first
    if isinstance(in_ref, dict):
        df_ref = pd.DataFrame()
        df_ref = df_ref.append(in_ref, ignore_index=True)
    else:
        df_ref = pd.read_csv(in_ref)
    if isinstance(in_batch, dict):
        df_batch = pd.DataFrame()
        df_batch = df_batch.append(in_batch, ignore_index=True)
    else:
        df_batch = pd.read_csv(in_batch, sep='\t')
    # check in_ref and in_batch for the required columns
    list_refcols = ['sample_name', 'CDS_frame']
    list_batchcols = ['name', 'amplicon_name', 'amplicon_seq', 'guide_seq', 'coding_seq']
    col_check(df_ref, list_refcols, 'in_ref')
    col_check(df_batch, list_batchcols, 'in_batch')
    # ensure df_ref/df_batch dtypes are correct (appending dict can cause errors)
    df_ref = df_ref.astype({k:v for k,v in zip(list_refcols,['str','int'])})
    df_batch = df_batch.astype({k:'str' for k in list_batchcols})
    # clean up and merge df_batch information into df_ref
    df_batch = df_batch.rename(columns={'name': 'sample_name'})
    df_ref = df_ref.merge(df_batch, on='sample_name')

    # CRISPResso column names are too long; convert to shorthand for ease
    dict_cols = {'Aligned_Sequence': 'aln_seq', 'Reference_Sequence': 'ref_seq', 'n_deleted': 'n_del',
                 'n_inserted': 'n_ins', 'n_mutated': 'n_mut', '#Reads': 'n_reads', '%Reads': 'pct_reads'}
    # define path, import raw allele frequencies table (df_raw) and isolate modified reads
    if isinstance(cwd, str):
        inpath = Path.cwd() / (cwd + '/CRISPResso_on_' + sample)
    elif isinstance(cwd, Path):
        inpath = cwd / ('CRISPResso_on_' + sample)
    df_raw = pd.read_csv(inpath / 'Alleles_frequency_table.zip', sep='\t').rename(columns=dict_cols)
    df_raw = df_raw.loc[df_raw['Read_Status'] == 'MODIFIED'].drop(columns=['Read_Status', 'Reference_Name']).copy()
    df_raw['UMI'] = df_raw['aln_seq'].str[:4]

    ### DEFINE VARIABLES

    # define reference amplicon name/seq, cds seq, gRNA name/seq, CRISPResso alleles output filename
    amp_name = df_ref.loc[df_ref['sample_name'] == sample]['amplicon_name'].values[0]
    seq_amp = df_ref.loc[df_ref['sample_name'] == sample]['amplicon_seq'].values[0]
    seq_cds = df_ref.loc[df_ref['sample_name'] == sample]['coding_seq'].values[0]
    seq_grna = df_ref.loc[df_ref['sample_name'] == sample]['guide_seq'].values[0]
    in_alleles = amp_name + '.Alleles_frequency_table_around_sgRNA_' + seq_grna + '.txt'
    # new version of crispresso changed output file name to guide_name if specified
    if 'guide_name' in df_ref.columns.tolist():
        # ensure dtype is correct (appending dict can cause errors)
        df_ref = df_ref.astype({'guide_name':'str'})
        guide_name = df_ref.loc[df_ref['sample_name'] == sample]['guide_name'].values[0]
        if not (inpath / in_alleles).exists():
            in_alleles = amp_name + '.Alleles_frequency_table_around_' + guide_name + '.txt'
        if not (inpath / in_alleles).exists():
            raise Exception('Could not find the CRISPResso alleles output .txt file')

    # adjust grna seq based on amplicon orientation and find cut site position (assumes SpCas9 +3)
    if seq_grna in seq_amp:
        idx_cut = (seq_amp.find(seq_grna) + len(seq_grna) - 3) - 1
    elif str(Seq(seq_grna).reverse_complement()) in seq_amp:
        seq_grna = str(Seq(seq_grna).reverse_complement())
        idx_cut = (seq_amp.find(seq_grna) + 3) - 1
    else:
        raise Exception('gRNA sequence not found in amplicon sequence')

    # define cds_l (# of nt until 1st full codon) and cds_r (# of nt in last codon) from CDS_frame ref info
    # note that it does not support 2 exons or non-contiguous exon seq
    cds_l = df_ref.loc[df_ref['sample_name'] == sample]['CDS_frame'].values[0]
    if not reverse and cds_l in (0,1,2):
        cds_r = len(seq_cds[cds_l:]) % 3 * -1 # convert # of nt in last codon to neg for slicing str
        if cds_r == 0: # can't slice str on right with 0, so replace 0 with None
            cds_r = None
    # if reverse is True, convert cds_l to cds_r and calculate cds_l
    elif reverse and cds_l in (0,1,2):
        cds_r = cds_l * -1
        if cds_r == 0: # can't slice str on right with 0, so replace 0 with None
            cds_r = None
        cds_l = len(seq_cds[:cds_r]) % 3
    else:
        raise Exception('CDS_frame is not 0, 1, or 2. Please check values.')

    # define intron/exon boundaries ('edge') by aligning the amplicon and cds sequences
    seq_cds_aln = get_align_objs(dna_align, dna(seq_cds), dna(seq_amp))[1]
    # if seq_cds_aln starts with gaps, assume left side of seq_cds is intron/exon boundary
    if seq_cds_aln[0] == '-':
        splice_l = True
    else:
        splice_l = False
    # if seq_cds_aln ends with gaps, assume right side of seq_cds is intron/exon boundary
    if seq_cds_aln[-1] == '-':
        splice_r = True
    else:
        splice_r = False

    # define the quantification/plot window size (if None, infer from ref seq length in in_alleles)
    if window == None:
        window = int((pd.read_csv(inpath / in_alleles, sep='\t')['Reference_Sequence'].str.len().values[0]) / 2)

    ### FIND IMPORTANT INDEX POSITIONS IN REFERENCE SEQUENCES

    # find the CDS start/end positions in ref amplicon
    idxs_cds = (seq_amp.find(seq_cds), seq_amp.find(seq_cds) + len(seq_cds) - 1)
    # find the window start/end positions (cut site position +/- window size)
    idxs_wdw = (idx_cut - window + 1, idx_cut + window)
    # find the start/end of the CDS seq within the quantification window
    idxs_cds_wdw = (max(idxs_cds[0], idxs_wdw[0]), min(idxs_cds[1], idxs_wdw[1]))
    # define the sequence in the cds/window overlap
    seq_cds_wdw = seq_amp[idxs_cds_wdw[0]: idxs_cds_wdw[1] + 1]

    ### CHECK FOR ANY CDS FRAME CHANGES AFTER SLICING TO SIZE OF THE WINDOW
    ### if slice to window truncates CDS, re-check for partial left/right codons

    # if left idx of cds_wdw = cds, then wdw_l = cds_l (wdw_l = # of nt until 1st full codon)
    if idxs_cds_wdw[0] == idxs_cds[0]:
        wdw_l = cds_l
    else: # if left idx is different, then find new offset for left window
        if (idxs_cds_wdw[0] - idxs_cds[0] - cds_l) % 3 == 0:
            wdw_l = 0
        else:
            wdw_l = 3 - ((idxs_wdw[0] - idxs_cds[0] - cds_l) % 3)
    # if right idx of cds_wdw = cds, then wdw_r = cds_r (wdw_r = # nt in last codon)
    if idxs_cds_wdw[1] == idxs_cds[1]:
        wdw_r = cds_r
    else: # if right idx is different, then find new offset for right window
        if (len(seq_cds_wdw) - wdw_l) % 3 == 0:
            wdw_r = None # can't slice str on right with 0, so use None
        else:
            wdw_r = ((len(seq_cds_wdw) - wdw_l) % 3) * -1

    ### DEFINE THE REFERENCE PROTEIN/CODON SEQUENCES

    # get the reference protein seq for the entire CDS seq and the CDS seq within the window
    # also convert seq_cds and seq_cds_wdw to list of codons
    seq_cds_prot = str(Seq(seq_cds[cds_l: cds_r]).translate())
    seq_wdw_prot = str(Seq(seq_cds_wdw[wdw_l: wdw_r]).translate())
    seq_cds_cdn = re.findall('...', seq_cds[cds_l: cds_r])
    seq_cds_wdw_cdn = re.findall('...', seq_cds_wdw[wdw_l: wdw_r])

    ### MAP POST-ALIGNMENT POSITIONS FOR MODIFIED READS IN DF_RAW
    ### this is how CRISPResso calls quant window, frameshift vs. in-frame muts

    # first make list of desired post-alignment reference positions to find
    # cut site, cds start, cds end, cds_wdw start, cds_wdw end
    list_pos = [idx_cut, idxs_cds[0], idxs_cds[1], idxs_cds_wdw[0], idxs_cds_wdw[1]]
    # find post-alignment positions with map_positions() and convert to dataframe
    df_pos = df_raw['ref_seq'].apply(lambda x: map_positions(x, list_pos)).tolist()
    df_pos = pd.DataFrame(data=df_pos, columns=['cut_idx', 'cds_start','cds_end','cds_wdw_start','cds_wdw_end'], index=df_raw.index)
    # get new wdw st/end from cut_idx; can't use idxs_wdw bc enlarges if insertions present
    df_pos['wdw_start'] = df_pos['cut_idx'] - window + 1
    df_pos['wdw_end'] = df_pos['cut_idx'] + window

    ### SLICE ALIGNED/REFERENCE SEQUENCES W/ DF_POS IDXS TO GET SUBSEQUENCES

    # aln/ref amplicon seqs corresponding to the coding sequence
    df_raw['aln_cds'] = [seq[st:end] for seq,st,end in zip(df_raw['aln_seq'], df_pos['cds_start'], df_pos['cds_end'] + 1)]
    df_raw['ref_cds'] = [seq[st:end] for seq,st,end in zip(df_raw['ref_seq'], df_pos['cds_start'], df_pos['cds_end'] + 1)]
    # aln/ref amplicon seqs falling within the cds/window overlap (WILL EXPAND IF INSERTIONS PRESENT)
    df_raw['aln_cds_wdw'] = [seq[st:end] for seq,st,end in zip(df_raw['aln_seq'], df_pos['cds_wdw_start'], df_pos['cds_wdw_end'] + 1)]
    df_raw['ref_cds_wdw'] = [seq[st:end] for seq,st,end in zip(df_raw['ref_seq'], df_pos['cds_wdw_start'], df_pos['cds_wdw_end'] + 1)]
    # aln/ref amplicon seqs falling within the quantification window (FIXED LENGTH)
    df_raw['aln_wdw'] = [seq[st:end] for seq,st,end in zip(df_raw['aln_seq'], df_pos['wdw_start'], df_pos['wdw_end'] + 1)]
    df_raw['ref_wdw'] = [seq[st:end] for seq,st,end in zip(df_raw['ref_seq'], df_pos['wdw_start'], df_pos['wdw_end'] + 1)]

    ### CALCULATE INDEL SIZES FOR SLICED SUBSEQS AND ASSIGN MUTATION TYPES (E.G. IN-FRAME/FRAMESHIFT/SPLICE)

    # amp_indel = n_ins - n_del (the indel size as calculated by crispresso)
    df_raw['amp_indel'] = df_raw['n_ins'] - df_raw['n_del']
    # cds indel = gaps in ref_cds - gaps in aln_cds (variable length cds_seq)
    df_raw['cds_indel'] = df_raw['ref_cds'].str.count('-') - df_raw['aln_cds'].str.count('-')
    # cds_wdw_indel = gaps in ref_cds_wdw - gaps in aln_cds_wdw (variable length cds_seq in window)
    df_raw['cds_wdw_indel'] = df_raw['ref_cds_wdw'].str.count('-') - df_raw['aln_cds_wdw'].str.count('-')

    # first deal with alleles where amp_indel == cds_wdw_indel (entire indel is within cds_wdw)
    df_muts = df_raw.loc[df_raw['amp_indel'] == df_raw['cds_wdw_indel']].iloc[:, 7:].copy()
    # call mutation as in-frame if the indel % 3 == 0; else frameshift
    df_muts['mut_type'] = np.where(df_muts['amp_indel'] % 3 == 0, 'in-frame', 'frameshift')
    # add mut_types to df_raw for these alleles (may not be fully accurate)
    df_raw = df_raw.join(df_muts['mut_type'])

    # now deal with alleles where amp_indel != cds_wdw_indel (skip if none)
    df_muts2 = df_raw.loc[df_raw['amp_indel'] != df_raw['cds_wdw_indel']].iloc[:, 7:].copy()
    if df_muts2.shape[0] > 0:
        # check for gaps at the edges of aln_cds
        df_muts2['cds_gap_l'] = np.where(df_muts2['aln_cds'].str.startswith('-'), True, False)
        df_muts2['cds_gap_r'] = np.where(df_muts2['aln_cds'].str.endswith('-'), True, False)
        # for alleles where amp_indel == cds_indel, call in-frame if indel % 3 == 0, else frameshift
        df_muts2['mut_type'] = np.where(df_muts2['amp_indel'] == df_muts2['cds_indel'],
                                        np.where(df_muts2['cds_indel'] % 3 == 0, 'in-frame', 'frameshift'), df_muts2['mut_type'])
        # now deal with alleles where amp_indel != cds_indel; if amp_indel is del larger than cds_indel and edge gap exists, call indel exceeds cds
        df_muts2['mut_type'] = np.where((df_muts2['amp_indel'] < df_muts2['cds_indel']) & (df_muts2[['cds_gap_l', 'cds_gap_r']].any(axis=1)),
                                        'indel exceeds cds', df_muts2['mut_type'])
        # if splice_l/splice_r == True, then find the 2 splice nts at each site
        # if gap in the splice nts, or if mut_type is indel exceeds cds and splice_l/cds_gap_l or splice_r/cds_gap_r are both True, call splice site mut
        if splice_l:
            df_muts2['splice_l'] = [seq[idx-2:idx] for seq,idx in zip(df_raw.loc[df_muts2.index, 'aln_seq'], df_pos.loc[df_muts2.index, 'cds_start'])]
            df_muts2['mut_type'] = np.where((df_muts2['mut_type'] == 'indel exceeds cds') & (df_muts2['cds_gap_l']), 'splice site', df_muts2['mut_type'])
            df_muts2['mut_type'] = np.where(df_muts2['splice_l'].str.contains('-'), 'splice site', df_muts2['mut_type'])
        if splice_r:
            df_muts2['splice_r'] = [seq[idx+1:idx+3] for seq,idx in zip(df_raw.loc[df_muts2.index, 'aln_seq'], df_pos.loc[df_muts2.index, 'cds_end'])]
            df_muts2['mut_type'] = np.where((df_muts2['mut_type'] == 'indel exceeds cds') & (df_muts2['cds_gap_r']), 'splice site', df_muts2['mut_type'])
            df_muts2['mut_type'] = np.where(df_muts2['splice_r'].str.contains('-'), 'splice site', df_muts2['mut_type'])
        # fill missing mut_types in df_raw with df_muts2 values
        df_raw['mut_type'].fillna(value=df_muts2['mut_type'], inplace=True)

    # if unclassified alleles (mut_type=NaN) remain, isolate and call muts (df_muts3)
    if df_muts2.loc[df_muts2['mut_type'].isna()].shape[0] > 0:
        df_muts3 = df_muts2.loc[df_muts2['mut_type'].isna()].copy()
        # if cds_indel == cds_wdw_indel (and neither == amp_indel), then call mut based on cds/cds_wdw indel
        df_muts3['mut_type'] = np.where(df_muts3['cds_indel'] == df_muts3['cds_wdw_indel'],
                                        np.where(df_muts3['cds_indel'] % 3 == 0, 'likely in-frame', 'likely frameshift'), df_muts3['mut_type'])
        # otherwise, call mut based on amp indel size
        df_muts3['mut_type'] = np.where(df_muts3['mut_type'].isna(), np.where(df_muts3['amp_indel'] % 3 == 0, 'likely in-frame', 'likely frameshift'), df_muts3['mut_type'])
        # fill remainder of df_raw and df_muts2 mut_types with df_muts3
        df_muts2['mut_type'].fillna(value=df_muts3['mut_type'], inplace=True)
        df_raw['mut_type'].fillna(value=df_muts3['mut_type'], inplace=True)

    # call consensus indel sizes; if 'amp_indel == cds_indel/cds_wdw indel, or if mut == splice site/indel exceeds cds, use amp indel
    df_raw['indel'] = np.nan
    df_raw['indel'] = np.where(df_raw['amp_indel'] == df_raw['cds_wdw_indel'], df_raw['cds_wdw_indel'], df_raw['indel'])
    df_raw['indel'] = np.where(df_raw['amp_indel'] == df_raw['cds_indel'], df_raw['cds_indel'], df_raw['indel'])
    df_raw['indel'] = np.where(df_raw['mut_type'].str.contains('splice site|indel exceeds cds'), df_raw['amp_indel'], df_raw['indel'])
    # otherwise, if cds_indel == cds_wdw indel, then use cds/cds_wdw_indel, otherwise use amp indel
    df_raw['indel'] = np.where(df_raw['indel'].isna(), np.where(df_raw['cds_indel'] == df_raw['cds_wdw_indel'], df_raw['cds_indel'], df_raw['amp_indel']), df_raw['indel'])

    ### PREPARE FOR TRANSLATION AND ALIGNMENT OF IN-FRAME ALLELES IN DF_MUTS (DF_TLN)

    # isolate in-frame alleles from df_muts (indels should be contained within aln_cds_wdw)
    df_tln = df_muts.loc[df_muts['mut_type'] == 'in-frame'].copy()
    df_tln = df_tln.iloc[:, :6].drop_duplicates().copy() # drop duplicates and unnecessary cols
    # trim the aln_cds_wdw and remove gaps --> acw_nogap
    df_tln['acw_nogap'] = df_tln['aln_cds_wdw'].str.replace('-', '').str[wdw_l: wdw_r]
    # check for Ns or seqs with < 3 nt in the acw_nogap seq; these cannot be codon aligned
    df_tln['check'] = np.where(df_tln['acw_nogap'].str.len() < 3, 'too_short', np.where(df_tln['acw_nogap'].str.contains('N'), 'has_N', 'tln_ok'))

    ### PERFORM CODON ALIGNMENT AND TRANSLATION ON IN-FRAME ALLELES WITHOUT Ns

    # isolate all alleles from df_tln without Ns and perform codon alignment
    df_caln = df_tln.loc[df_tln['check'] == 'tln_ok'].copy()
    # get_align_objs() returns the # of alignments and aln/ref seqs for the top alignment
    # if multiple equal alnments, only returns the last one bc this is usually the left-aligned indel
    df_caln['aln'] = df_caln['acw_nogap'].apply(lambda x: get_align_objs(cdn_align, re.findall('...', x), seq_cds_wdw_cdn, codons=True)).tolist()
    # expand the column of tuples to individual columns and drop the tuples
    df_caln = df_caln.join(pd.DataFrame(data=df_caln['aln'].tolist(), columns=['n_aln', 'aln_cdn', 'ref_cdn'], index=df_caln.index))
    df_caln.drop(columns='aln', inplace=True)
    # translate the codon-aligned aln/ref sequences --> aln_tln/ref_tln
    if reverse: # if seq orientation is reversed, translate the rev complement
        df_caln = df_caln.assign(aln_tln=df_caln['aln_cdn'].apply(lambda x: tln(str(dna(x,rc=True)))), ref_tln=df_caln['ref_cdn'].apply(lambda x: tln(str(dna(x,rc=True)))))
    else:
        df_caln = df_caln.assign(aln_tln=df_caln['aln_cdn'].apply(lambda x: tln(x)), ref_tln=df_caln['ref_cdn'].apply(lambda x: tln(x)))

    # identify any nt offsets between window start/end and cds_wdw start/end
    df_caln['wdw_l'] = df_pos.loc[df_caln.index]['wdw_start'].sub(df_pos.loc[df_caln.index]['cds_wdw_start'])
    df_caln['wdw_r'] = df_pos.loc[df_caln.index]['cds_wdw_end'].sub(df_pos.loc[df_caln.index]['wdw_end'])
    # isolate alleles where cds_wdw_seq starts or ends outside of wdw (wdw_l/wdw_r > 0)
    df_ctrim = df_caln.loc[(df_caln['wdw_l'] > 0) | (df_caln['wdw_r'] > 0)].copy()
    # if alleles require trimming, continue with trimming workflow
    if df_ctrim.shape[0] > 0:
        # add placeholder 'X' to the ends of aln_wdw to make it same length as cds_wdw_seq
        df_ctrim['wdw_adj'] = df_ctrim.apply(lambda x: ('X' * x['wdw_l']) + x['aln_wdw'] + ('X' * x['wdw_r']), axis=1)
        # trim seq if wdw_l/wdw_r offset is negative
        df_ctrim['wdw_nogap'] = df_ctrim['wdw_adj']
        df_ctrim.loc[df_ctrim['wdw_l'] < 0, 'wdw_nogap'] = df_ctrim.loc[df_ctrim['wdw_l'] < 0].apply(lambda x: x['wdw_nogap'][x['wdw_l']*-1:], axis=1)
        df_ctrim.loc[df_ctrim['wdw_r'] < 0, 'wdw_nogap'] = df_ctrim.loc[df_ctrim['wdw_r'] < 0].apply(lambda x: x['wdw_nogap'][:x['wdw_r']], axis=1)
        # now remove gaps and trim with the global wdw_l/wdw_r offsets like acw_nogap
        df_ctrim['wdw_nogap'] = df_ctrim['wdw_nogap'].str.replace('-','').str[wdw_l:wdw_r]
        # this places the wdw seq into the correct frame so we can trim partial codons on ends
        # convert wdw_nogap to cdn and remove cdns with 'X' to get all full cdns in wdw
        df_ctrim['wdw_cdn'] = df_ctrim['wdw_nogap'].apply(lambda x: re.findall('...', x))
        df_ctrim['wdw_cdn'] = df_ctrim['wdw_cdn'].apply(lambda x: str().join([cdn for cdn in x if 'X' not in cdn]))
        # realign wdw cdns to acw_nogap to get the st/end of aln_cdns in wdw
        df_ctrim['realn'] = df_ctrim.apply(lambda x: get_align_objs(cdn_align, re.findall('...', x['wdw_cdn']), re.findall('...', x['acw_nogap']))[1], axis=1).str.replace(' ','')
        # re-split into cdns and map '-' for gaps, 'C' for full cdn
        df_ctrim['cdnmap'] = df_ctrim['realn'].apply(lambda x: re.findall('...', x))
        df_ctrim['cdnmap'] = df_ctrim['cdnmap'].apply(lambda x: str().join(['-' if '---' in cdn else 'C' for cdn in x]))
        # get idxs for st/end of wdw cdns (from first to last C; usually contiguous but sometimes not)
        df_ctrim['wdw_idxs'] = df_ctrim['cdnmap'].apply(lambda x: re.match('(^[-]*)([C][C-]*[C])([-]*$)', x))
        # sometimes funky stuff happens here, so only trim if the span of the wdw_idxs are not longer than aln_tln
        # otherwise, just use the length of aln_tln (no trim)
        df_ctrim['trim_ok'] = [x.end(2) - x.start(2) for x in df_ctrim['wdw_idxs']]
        df_ctrim['trim_ok'] = np.where(df_ctrim['trim_ok'] <= df_ctrim['aln_tln'].str.len(), True, False)
        # use match.end() - 1 here because we are going to map position -- m.end() is always end idx+1 (assumes slicing logic)
        df_ctrim['wdw_idxs'] = [(x.start(2), x.end(2) - 1) for x in df_ctrim['wdw_idxs']]
        # 01NOV22: add len check due to aln errors; if idx_end > aln_tln, set trim_ok to false
        df_ctrim['len_qc'] = df_ctrim.apply(lambda x: x['wdw_idxs'][1] - len(x['aln_tln'].replace('-','')), axis=1)
        df_ctrim['trim_ok'] = np.where(df_ctrim['len_qc'] >= 0, False, df_ctrim['trim_ok'])
        df_ctrim['wdw_idxs'] = np.where(df_ctrim['trim_ok'], df_ctrim['wdw_idxs'], pd.Series([(0, x-1) for x in df_ctrim['aln_tln'].str.replace('-','').str.len()], index=df_ctrim.index))
        # now map the st/end idxs to the post-cdn alignment aln_tln (skip gaps!) to get the final idxs
        df_ctrim['map_idxs'] = df_ctrim.apply(lambda x: map_positions(x['aln_tln'], x['wdw_idxs'], regex='[^-]'), axis=1)
        # slice the aln_tln and ref_tln using the post-alignment idxs to get the wdw tln (use end idx+1 since slicing)
        df_ctrim['aln_trim'] = [seq[idxs[0]:idxs[1] + 1] for seq,idxs in zip(df_ctrim['aln_tln'], df_ctrim['map_idxs'])]
        df_ctrim['ref_trim'] = [seq[idxs[0]:idxs[1] + 1] for seq,idxs in zip(df_ctrim['ref_tln'], df_ctrim['map_idxs'])]
        # join trimmed tlns to df_caln; for final tlns, use trim tln if not nan
        df_caln = df_caln.join(df_ctrim[['aln_trim', 'ref_trim']])
        df_caln['aln_final'] = np.where(df_caln['aln_trim'].isna(), df_caln['aln_tln'], df_caln['aln_trim'])
        df_caln['ref_final'] = np.where(df_caln['ref_trim'].isna(), df_caln['ref_tln'], df_caln['ref_trim'])
    # otherwise, if no alleles to trim, set aln/ref_final in df_paln to aln/ref_tln
    else:
        df_caln[['aln_final', 'ref_final']] = df_caln[['aln_tln', 'ref_tln']]

    # join the aln/ref final translations to df_tln and rename columns
    df_tln = df_tln.join(df_caln[['aln_final', 'ref_final']]).rename(columns={'aln_final': 'prot_aln', 'ref_final': 'prot_ref'})

    ### TRANSLATE AND PERFORM PROTEIN ALIGNMENT ON IN-FRAME ALLELES WITH Ns

    # now deal with in-frame alleles that contain Ns (cannot be codon aligned)
    df_paln = df_tln.loc[df_tln['check'] == 'has_N'].copy()
    # translate acw_nogap seq and perform protein alignment against seq_wdw_prot
    # retrieve n_alignments and aln/ref seqs for top alignment with get_align_objs
    # if seq orientation is reversed, translate and align w/ rev complement
    if reverse:
        df_paln['acw_nogap_tln'] = df_paln['acw_nogap'].apply(lambda x: tln(str(dna(x, rc=True))))
        seq_wdw_prot_rev = tln(str(dna(seq_cds_wdw[wdw_l:wdw_r], rc=True)))
        df_paln['aln'] = df_paln['acw_nogap_tln'].apply(lambda x: get_align_objs(prot_align, x, seq_wdw_prot_rev)).tolist()
    else:
        df_paln['acw_nogap_tln'] = df_paln['acw_nogap'].apply(lambda x: tln(x))
        df_paln['aln'] = df_paln['acw_nogap_tln'].apply(lambda x: get_align_objs(prot_align, x, seq_wdw_prot)).tolist()
    # expand the column of tuples to individual columns and drop the tuples
    df_paln = df_paln.join(pd.DataFrame(data=df_paln['aln'].tolist(), columns=['n_aln', 'aln_tln', 'ref_tln'], index=df_paln.index))
    df_paln.drop(columns='aln', inplace=True)

    # identify any nt offsets between window start/end and cds_wdw start/end
    df_paln['wdw_l'] = df_pos.loc[df_paln.index]['wdw_start'].sub(df_pos.loc[df_paln.index]['cds_wdw_start'])
    df_paln['wdw_r'] = df_pos.loc[df_paln.index]['cds_wdw_end'].sub(df_pos.loc[df_paln.index]['wdw_end'])
    # isolate alleles where cds_wdw_seq starts or ends outside of wdw (wdw_l/wdw_r > 0)
    df_ptrim = df_paln.loc[(df_paln['wdw_l'] > 0) | (df_paln['wdw_r'] > 0)].copy()
    # if alleles require trimming, continue with trimming workflow
    if df_ptrim.shape[0] > 0:
        # add placeholder 'X' to the ends of aln_wdw to make it same length as cds_wdw_seq
        df_ptrim['wdw_adj'] = df_ptrim.apply(lambda x: ('X' * x['wdw_l']) + x['aln_wdw'] + ('X' * x['wdw_r']), axis=1)
        # trim seq if wdw_l/wdw_r offset is negative
        df_ptrim['wdw_nogap'] = df_ptrim['wdw_adj']
        df_ptrim.loc[df_ptrim['wdw_l'] < 0, 'wdw_nogap'] = df_ptrim.loc[df_ptrim['wdw_l'] < 0].apply(lambda x: x['wdw_nogap'][x['wdw_l']*-1:], axis=1)
        df_ptrim.loc[df_ptrim['wdw_r'] < 0, 'wdw_nogap'] = df_ptrim.loc[df_ptrim['wdw_r'] < 0].apply(lambda x: x['wdw_nogap'][:x['wdw_r']], axis=1)
        # now remove gaps and trim with the global wdw_l/wdw_r offsets like acw_nogap
        # this places the wdw seq into the correct frame so we can trim partial codons on ends
        df_ptrim['wdw_nogap'] = df_ptrim['wdw_nogap'].str.replace('-','').str[wdw_l:wdw_r]
        # convert wdw_nogap to cdn, remove cdns with 'X' to get all full cdns in wdw, and translate
        df_ptrim['wdw_tln'] = df_ptrim['wdw_nogap'].apply(lambda x: re.findall('...', x))
        df_ptrim['wdw_tln'] = df_ptrim['wdw_tln'].apply(lambda x: str().join([cdn for cdn in x if 'X' not in cdn]))
        if reverse:
            df_ptrim['wdw_tln'] = df_ptrim['wdw_tln'].apply(lambda x: tln(str(dna(x, rc=True))))
        else:
            df_ptrim['wdw_tln'] = df_ptrim['wdw_tln'].apply(lambda x: tln(x))
        # realign wdw tln to acw_nogap_tln to get the st/end of aln_tln in wdw
        df_ptrim['realn'] = df_ptrim.apply(lambda x: get_align_objs(prot_align, x['wdw_tln'], x['acw_nogap_tln'])[1], axis=1)
        # replace all non-gaps in tln with 'C' to make regex easier
        df_ptrim['tlnmap'] = df_ptrim['realn'].str.replace('[^-]', 'C', regex=True)
        # get idxs for st/end of wdw tln (from first to last C; usually contiguous but sometimes not)
        df_ptrim['wdw_idxs'] = df_ptrim['tlnmap'].apply(lambda x: re.match('(^[-]*)([C][C-]*[C])([-]*$)', x))
        # use match.end() - 1 here because we are going to map position -- m.end() is always end idx+1 (assumes slicing logic)
        df_ptrim['wdw_idxs'] = [(x.start(2), x.end(2) - 1) for x in df_ptrim['wdw_idxs']]
        # now map the st/end idxs to the post-protein alignment aln_tln (skip gaps!) to get the final idxs
        df_ptrim['map_idxs'] = df_ptrim.apply(lambda x: map_positions(x['aln_tln'], x['wdw_idxs'], regex='[^-]'), axis=1)
        # slice the aln_tln and ref_tln using the post-alignment idxs to get the wdw tln (use end idx+1 since slicing)
        df_ptrim['aln_trim'] = [seq[idxs[0]:idxs[1] + 1] for seq,idxs in zip(df_ptrim['aln_tln'], df_ptrim['map_idxs'])]
        df_ptrim['ref_trim'] = [seq[idxs[0]:idxs[1] + 1] for seq,idxs in zip(df_ptrim['ref_tln'], df_ptrim['map_idxs'])]
        # join trimmed tlns to df_paln; for final tlns, use trim tln if not nan
        df_paln = df_paln.join(df_ptrim[['aln_trim', 'ref_trim']])
        df_paln['aln_final'] = np.where(df_paln['aln_trim'].isna(), df_paln['aln_tln'], df_paln['aln_trim'])
        df_paln['ref_final'] = np.where(df_paln['ref_trim'].isna(), df_paln['ref_tln'], df_paln['ref_trim'])
    # otherwise, if no alleles to trim, set aln/ref_final in df_paln to aln/ref_tln
    else:
        df_paln[['aln_final', 'ref_final']] = df_paln[['aln_tln', 'ref_tln']]

    # now fill the remaining aln/ref translations in df_tln with the final aligned aln/ref prots
    df_tln['prot_aln'].fillna(value=df_paln['aln_final'], inplace=True)
    df_tln['prot_ref'].fillna(value=df_paln['ref_final'], inplace=True)
    # if there are alleles where acw_nogap == too_short, call no_tln
    df_tln['prot_aln'] = np.where(df_tln['check'] == 'too_short', 'no_tln', df_tln['prot_aln'])
    df_tln['prot_ref'] = np.where(df_tln['check'] == 'too_short', 'no_tln', df_tln['prot_ref'])

    ### TRANSLATION AND ALIGNMENT OF IN-FRAME ALLELES IN DF_MUTS2 (DF_TLN2)

    # isolate in-frame (including likely) alleles from df_muts2 (all other indels)
    df_tln2 = df_muts2.loc[df_muts2['mut_type'].str.contains('in-frame|likely in-frame')].drop_duplicates().copy()
    # trim the aln_cds with cds offsets and remove gaps --> aln_nogap
    df_tln2['aln_nogap'] = df_tln2['aln_cds'].str.replace('-', '').str[cds_l: cds_r]
    # check if aln_nogap seqs < 3 nt (too short), % 3 != 0 (not in frame), or has Ns (cannot codon aln)
    df_tln2['check'] = df_tln2['aln_nogap'].apply(lambda x: check_seqs(x))

    ### PERFORM CODON ALIGNMENT AND TRANSLATION ON IN-FRAME ALLELES WITHOUT Ns

    # isolate all alleles from df_tln2 without Ns and perform codon alignment
    df_caln2 = df_tln2.loc[df_tln2['check'] == 'tln_ok'].copy()
    # retrieve n_alignments and aln/ref seqs for top alignment with get_align_objs
    df_caln2['aln'] = df_caln2['aln_nogap'].str.findall('...').apply(lambda x: get_align_objs(cdn_align, x, seq_cds_cdn, codons=True)).tolist()
    # expand the column of tuples to individual columns and drop the tuples
    df_caln2 = df_caln2.join(pd.DataFrame(data=df_caln2['aln'].tolist(), columns=['n_aln', 'aln_cdn', 'ref_cdn'], index=df_caln2.index))
    df_caln2.drop(columns='aln', inplace=True)
    # translate the codon-aligned aln/ref sequences --> aln_tln/ref_tln
    if reverse: # if seq orientation is reversed, translate the rev complement
        df_caln2 = df_caln2.assign(aln_tln=df_caln2['aln_cdn'].apply(lambda x: tln(str(dna(x,rc=True)))), ref_tln=df_caln2['ref_cdn'].apply(lambda x: tln(str(dna(x,rc=True)))))
    else:
        df_caln2 = df_caln2.assign(aln_tln=df_caln2['aln_cdn'].apply(lambda x: tln(x)), ref_tln=df_caln2['ref_cdn'].apply(lambda x: tln(x)))

    # identify any nt offsets between window start/end and cds start/end
    df_caln2['wdw_l'] = df_pos.loc[df_caln2.index]['wdw_start'].sub(df_pos.loc[df_caln2.index]['cds_start'])
    df_caln2['wdw_r'] = df_pos.loc[df_caln2.index]['cds_end'].sub(df_pos.loc[df_caln2.index]['wdw_end'])
    # isolate alleles where cds_seq starts or ends outside of wdw (wdw_l/wdw_r > 0)
    df_ctrim2 = df_caln2.loc[(df_caln2['wdw_l'] > 0) | (df_caln2['wdw_r'] > 0)].copy()
    # if alleles require trimming, continue with trimming workflow
    if df_ctrim2.shape[0] > 0:
        # add placeholder 'X' to the ends of aln_wdw to make it same length as cds_seq
        df_ctrim2['wdw_adj'] = df_ctrim2.apply(lambda x: ('X' * x['wdw_l']) + x['aln_wdw'] + ('X' * x['wdw_r']), axis=1)
        # trim seq if wdw_l/wdw_r offset is negative
        df_ctrim2['wdw_nogap'] = df_ctrim2['wdw_adj']
        df_ctrim2.loc[df_ctrim2['wdw_l'] < 0, 'wdw_nogap'] = df_ctrim2.loc[df_ctrim2['wdw_l'] < 0].apply(lambda x: x['wdw_nogap'][x['wdw_l']*-1:], axis=1)
        df_ctrim2.loc[df_ctrim2['wdw_r'] < 0, 'wdw_nogap'] = df_ctrim2.loc[df_ctrim2['wdw_r'] < 0].apply(lambda x: x['wdw_nogap'][:x['wdw_r']], axis=1)
        # remove gaps from wdw_adj using gap positions from aln_cds and trim with the global cds_l/cds_r offsets
        # this places the wdw seq into the correct frame so we can trim partial codons on ends
        df_ctrim2['wdw_nogap'] = df_ctrim2.apply(lambda x: find_gaps(x['aln_cds'], x['wdw_nogap']), axis=1)
        df_ctrim2['wdw_nogap'] = df_ctrim2['wdw_nogap'].str.replace('-','').str[cds_l:cds_r]
        # convert wdw_nogap to cdn and add gaps at the same positions as aln_cdn
        # use the aln_nogap > aln_tln position change in alignment to insert gaps correctly
        df_ctrim2['wdw_cdn'] = df_ctrim2['wdw_nogap'].apply(lambda x: re.findall('...', x))
        df_ctrim2['wdw_cdngaps'] = df_ctrim2.apply(lambda x: fill_gaps(x['aln_tln'], x['wdw_cdn'], sub='---', codon=True), axis=1)
        # find the codon positions in wdw_adj that are fully within the wdw (no placeholder X) and get idxs
        df_ctrim2['wdw_map'] = df_ctrim2['wdw_adj'].apply(lambda x: str().join(['J' if 'X' in cdn else 'C' for cdn in re.findall('...', x)]))
        df_ctrim2['wdw_idxs'] = df_ctrim2['wdw_map'].apply(lambda x: re.match('(^[J]*)([C]*)([J]*$)', x).span(2))
        # slice wdw_cdngaps using the wdw idxs and translate to get the trimmed aln tln (wdw_cdngaps is essentially a mask of aln_cdn)
        df_ctrim2['aln_trim'] = [seq[(idxs[0] * 3):(idxs[1] * 3)] for seq,idxs in zip(df_ctrim2['wdw_cdngaps'], df_ctrim2['wdw_idxs'])]
        df_ctrim2['aln_trim'] = df_ctrim2['aln_trim'].apply(lambda x: str().join(['-' if 'X' in cdn else '-' if '-' in cdn else tln(cdn) for cdn in re.findall('...', x)]))
        # slice the ref tln to get the trimmed ref tln
        df_ctrim2['ref_trim'] = [seq[idxs[0]:idxs[1]] for seq,idxs in zip(df_ctrim2['ref_tln'], df_ctrim2['wdw_idxs'])]
        # join trimmed tlns to df_caln2; for final tlns, use trim tln if not nan
        df_caln2 = df_caln2.join(df_ctrim2[['aln_trim', 'ref_trim']])
        df_caln2['aln_final'] = np.where(df_caln2['aln_trim'].isna(), df_caln2['aln_tln'], df_caln2['aln_trim'])
        df_caln2['ref_final'] = np.where(df_caln2['ref_trim'].isna(), df_caln2['ref_tln'], df_caln2['ref_trim'])
    # otherwise, if no alleles to trim, set aln/ref_final in df_paln to aln/ref_tln
    else:
        df_caln2[['aln_final', 'ref_final']] = df_caln2[['aln_tln', 'ref_tln']]

    # join the aln/ref final translations to df_tln and rename columns
    df_tln2 = df_tln2.join(df_caln2[['aln_final', 'ref_final']]).rename(columns={'aln_final': 'prot_aln', 'ref_final': 'prot_ref'})

    ### TRANSLATE AND PERFORM PROTEIN ALIGNMENT ON IN-FRAME ALLELES WITH Ns

    # now deal with in-frame alleles that contain Ns (cannot be codon aligned)
    df_paln2 = df_tln2.loc[df_tln2['check'] == 'has_N'].copy()
    # translate aln_nogap seq and perform protein alignment against seq_cds_prot
    # retrieve n_alignments and aln/ref seqs for top alignment with get_align_objs
    # if seq orientation is reversed, translate and align w/ rev complement
    if reverse:
        df_paln2['aln_nogap_tln'] = df_paln2['aln_nogap'].apply(lambda x: tln(str(dna(x, rc=True))))
        seq_cds_prot_rev = tln(str(dna(seq_cds[cds_l:cds_r], rc=True)))
        df_paln2['aln'] = df_paln2['aln_nogap_tln'].apply(lambda x: get_align_objs(prot_align, x, seq_cds_prot_rev)).tolist()
    else:
        df_paln2['aln_nogap_tln'] = df_paln2['aln_nogap'].apply(lambda x: tln(x))
        df_paln2['aln'] = df_paln2['aln_nogap_tln'].apply(lambda x: get_align_objs(prot_align, x, seq_cds_prot)).tolist()
    # expand the column of tuples to individual columns and drop the tuples
    df_paln2 = df_paln2.join(pd.DataFrame(data=df_paln2['aln'].tolist(), columns=['n_aln', 'aln_tln', 'ref_tln'], index=df_paln2.index))
    df_paln2.drop(columns='aln', inplace=True)

    # identify any nt offsets between window start/end and cds start/end
    df_paln2['wdw_l'] = df_pos.loc[df_paln2.index]['wdw_start'].sub(df_pos.loc[df_paln2.index]['cds_start'])
    df_paln2['wdw_r'] = df_pos.loc[df_paln2.index]['cds_end'].sub(df_pos.loc[df_paln2.index]['wdw_end'])
    # isolate alleles where cds_seq starts or ends outside of wdw (wdw_l/wdw_r > 0)
    df_ptrim2 = df_paln2.loc[(df_paln2['wdw_l'] > 0) | (df_paln2['wdw_r'] > 0)].copy()
    # if alleles require trimming, continue with trimming workflow
    if df_ptrim2.shape[0] > 0:
        # add placeholder 'X' to the ends of aln_wdw to make it same length as cds_seq
        df_ptrim2['wdw_adj'] = df_ptrim2.apply(lambda x: ('X' * x['wdw_l']) + x['aln_wdw'] + ('X' * x['wdw_r']), axis=1)
        # trim seq if wdw_l/wdw_r offset is negative
        df_ptrim2.loc[df_ptrim2['wdw_l'] < 0, 'wdw_adj'] = df_ptrim2.loc[df_ptrim2['wdw_l'] < 0].apply(lambda x: x['wdw_adj'][x['wdw_l']*-1:], axis=1)
        df_ptrim2.loc[df_ptrim2['wdw_r'] < 0, 'wdw_adj'] = df_ptrim2.loc[df_ptrim2['wdw_r'] < 0].apply(lambda x: x['wdw_adj'][:x['wdw_r']], axis=1)
        # remove gaps from wdw_adj using gap positions from aln_cds and trim with the global cds_l/cds_r offsets
        # this places the wdw seq into the correct frame so we can trim partial codons on ends
        df_ptrim2['wdw_nogap'] = df_ptrim2.apply(lambda x: find_gaps(x['aln_cds'], x['wdw_adj']), axis=1)
        df_ptrim2['wdw_nogap'] = df_ptrim2['wdw_nogap'].str.replace('-','').str[cds_l:cds_r]
        # convert wdw_nogap to protein and add gaps at the same positions as aln_nogap_tln
        # use the aln_nogap_tln > aln_tln post_alignment change to insert gaps correctly
        if reverse:
            df_ptrim2['wdw_tln'] = df_ptrim2['wdw_nogap'].apply(lambda x: re.findall('...', str(dna(x, rc=True))))
        else:
            df_ptrim2['wdw_tln'] = df_ptrim2['wdw_nogap'].apply(lambda x: re.findall('...', x))
        df_ptrim2['wdw_tln'] = df_ptrim2['wdw_tln'].apply(lambda x: str().join(['J' if 'X' in cdn else tln(cdn) for cdn in x]))
        df_ptrim2['wdw_tlngaps'] = df_ptrim2.apply(lambda x: fill_gaps(x['aln_tln'], x['wdw_tln']), axis=1)
        # find the codon positions in wdw_adj that are fully within the wdw (no placeholder X) and get idxs
        df_ptrim2['wdw_map'] = df_ptrim2['wdw_adj'].apply(lambda x: str().join(['J' if 'X' in cdn else 'C' for cdn in re.findall('...', x)]))
        df_ptrim2['wdw_idxs'] = df_ptrim2['wdw_map'].apply(lambda x: re.match('(^[J]*)([C]*)([J]*$)', x).span(2))
        # slice wdw_tlngaps and ref_tln with the wdw idxs to get the trimmed tlns (wdw_tlngaps is essentially a mask of aln_tln)
        df_ptrim2['aln_trim'] = [seq[idxs[0]:idxs[1]] for seq,idxs in zip(df_ptrim2['wdw_tlngaps'], df_ptrim2['wdw_idxs'])]
        # replace any 'J's with '-' gaps b/c these arise from partial codons with X
        df_ptrim2['aln_trim'] = df_ptrim2['aln_trim'].str.replace('J', '-')
        df_ptrim2['ref_trim'] = [seq[idxs[0]:idxs[1]] for seq,idxs in zip(df_ptrim2['ref_tln'], df_ptrim2['wdw_idxs'])]
        # join trimmed tlns to df_caln2; for final tlns, use trim tln if not nan
        df_paln2 = df_paln2.join(df_ptrim2[['aln_trim', 'ref_trim']])
        df_paln2['aln_final'] = np.where(df_paln2['aln_trim'].isna(), df_paln2['aln_tln'], df_paln2['aln_trim'])
        df_paln2['ref_final'] = np.where(df_paln2['ref_trim'].isna(), df_paln2['ref_tln'], df_paln2['ref_trim'])
    # otherwise, if no alleles to trim, set aln/ref_final in df_paln to aln/ref_tln
    else:
        df_paln2[['aln_final', 'ref_final']] = df_paln2[['aln_tln', 'ref_tln']]

    # fill the remaining aln/ref translations in df_tln2 with the final aligned aln/ref prots
    df_tln2['prot_aln'].fillna(value=df_paln2['aln_final'], inplace=True)
    df_tln2['prot_ref'].fillna(value=df_paln2['ref_final'], inplace=True)
    # if there are alleles where acw_nogap == too_short, call no_tln
    df_tln2['prot_aln'] = np.where(df_tln2['check'].str.contains('too_short|not_in_frame'), 'no_tln', df_tln2['prot_aln'])
    df_tln2['prot_ref'] = np.where(df_tln2['check'].str.contains('too_short|not_in_frame'), 'no_tln', df_tln2['prot_ref'])

    ### MERGE ALIGNMENTS WITH CRISPRESSO ALLELES

    # first take the translations/alignments and map back to df_muts/df_muts2
    list_keys = df_muts.iloc[:, :6].columns.tolist()
    df_muts['idx'] = df_muts.index.tolist()
    df_muts = df_muts.merge(df_tln, how='left', on=list_keys, validate='m:1')
    df_muts = df_muts[['idx', 'mut_type', 'prot_aln', 'prot_ref']]
    list_keys2 = df_muts2.columns.tolist()
    df_muts2['idx'] = df_muts2.index.tolist()
    df_muts2 = df_muts2.merge(df_tln2, how='left', on=list_keys2, validate='m:1')
    df_muts2 = df_muts2[['idx', 'mut_type', 'prot_aln', 'prot_ref']]
    # concatenate dfs and drop cols except original idxs and aln/ref prots
    df_muts = pd.concat([df_muts, df_muts2]).set_index(keys='idx')
    # fill the prot_aln/prot_ref NaN values for non-in-frame muts (e.g. frameshift, splice site)
    df_muts = df_muts.fillna(value={'prot_aln': df_muts['mut_type'], 'prot_ref': df_muts['mut_type']})
    # join the prot aln/prot ref cols to df_raw
    df_raw = df_raw.join(df_muts[['prot_aln', 'prot_ref']])

    # make a column for column indexes to track raw alleles through merging
    df_raw['raw_idx'] = df_raw.index.tolist()
    # groupby all columns (crispresso cols + my cols (e.g. indel, mut, prot aln/ref))
    # sum the # and pct of reads, group raw allele indexes to list
    list_groupcols = ['aln_wdw', 'ref_wdw', 'n_del', 'n_ins', 'n_mut',
                      'amp_indel', 'indel', 'mut_type', 'prot_aln', 'prot_ref']
    df_agg = df_raw.groupby(list_groupcols, as_index=False).agg({'n_reads': 'sum', 'pct_reads': 'sum', 'raw_idx': list})
    # sort by % reads, re-index, rename cols
    df_agg = df_agg.sort_values(by='pct_reads', ascending=False).reset_index(drop=True)
    df_agg = df_agg.rename(columns={'aln_wdw': 'aln_seq','ref_wdw': 'ref_seq'})
    # make a new column to hold the df_agg indexes (to map mismatches after merging with crispresso)
    df_agg['agg_idx'] = df_agg.index.tolist()
    # also make a new column to denote the # of raw alleles per agg allele
    df_agg['n_raw_alleles'] = df_agg['raw_idx'].apply(lambda x: len(x))
    # groupby crispresso columns to make output that will merge 1:1 with df_crispresso
    # groups with >1 allele are "mismatches", take the info from the allele w/ most reads, which will be first by sorting
    list_groupcols2 = ['aln_seq', 'ref_seq', 'n_del', 'n_ins', 'n_mut']
    dict_agg = {'n_reads':'sum','pct_reads':'sum'}
    dict_agg.update({col:'first' for col in list_groupcols[5:]})
    dict_agg.update({'raw_idx': 'sum', 'agg_idx': list})
    df_merge = df_agg.groupby(list_groupcols2, as_index=False).agg(dict_agg)
    df_merge['pct_reads'] = df_merge['pct_reads'].round(decimals=5)
    # count the # of mismatched alleles after aggregating on just the crispresso cols
    df_merge['n_alleles'] = df_merge['agg_idx'].apply(lambda x: len(x))

    # import crispresso output alleles, rename cols, round % reads
    df_alleles = pd.read_csv(inpath / in_alleles, sep='\t').rename(columns=dict_cols)
    df_alleles['pct_reads'] = df_alleles['pct_reads'].round(decimals=5)
    # merge the two dataframes and drop cols to generate final output
    df_merge = pd.merge(df_alleles, df_merge, how='outer', on=list(dict_cols.values()), indicator=True)
    df_final = df_merge.drop(columns=['raw_idx', 'agg_idx', '_merge']).copy()
    # finally, fill in the wild-type allele (unedited == True) row information
    if reverse: # if sequencing orientation reversed, use rev complement
        wt_allele = [0, 0, 'wild-type', str(seq_wdw_prot_rev), str(seq_wdw_prot_rev), 1]
    else:
        wt_allele = [0, 0, 'wild-type', str(seq_wdw_prot), str(seq_wdw_prot), 1]
    df_final.loc[df_final['Unedited'] == True, ['amp_indel', 'indel', 'mut_type', 'prot_aln', 'prot_ref', 'n_alleles']] = wt_allele
    # if reverse and return_rev, then return the reverse complement of aln_seq/ref_seq
    if reverse and return_rev:
        df_final['ref_seq'] = df_final['ref_seq'].apply(lambda x: str(dna(x, rc=True)))
        df_final['aln_seq'] = df_final['aln_seq'].apply(lambda x: str(dna(x, rc=True)))

    # isolate the mismatched alleles (for output/stats purposes)
    df_mismatch = df_merge.loc[df_merge['n_alleles'] > 1].copy()
    df_mismatch = df_agg.loc[df_mismatch['agg_idx'].explode()].assign(merge_idx=df_mismatch['agg_idx'].explode().index.tolist())

    ### EXPORT FILES, CLEAN-UP, RETURN STATS AND DATAFRAMES

    # remove alignment tools
    del dna_align
    del prot_align
    del cdn_align
    # define save parameters
    if isinstance(out_folder, str):
        outpath = Path.cwd() / out_folder
    elif isinstance(out_folder, Path):
        outpath = out_folder
    Path.mkdir(outpath, exist_ok=True)
    if out_file is None:
        out_file = sample + '_alleles_tln_freq.csv'
    if save:
        df_final.to_csv((outpath / out_file))

    # calculate final stats
    stats = (df_alleles.shape[0] - 1, # n_alleles (non-WT) in the crispresso output file
             df_agg.shape[0], # n_alleles after aggregating alleles in function
             df_mismatch.shape[0], # n_alleles in df_agg not merging correctly (mismatches)
             df_merge.loc[df_merge['n_alleles'] > 1].shape[0], # n_alleles expected after proper mismatch merging
             df_final.loc[df_final['prot_aln'] == 'no_tln'].shape[0]) # n_alleles untranslated
    # print a header and the final stats
    print(sample + ' allele analysis statistics:' + '\n',
          '# of non-WT alleles in CRISPResso output = ' + str(stats[0]) + '\n',
          '# of alleles aggregated in script = ' + str(stats[1]) + '\n',
          '# of mismatched alleles in script = ' + str(stats[2]) + '\n',
          '# of expected alleles from mismatches = ' + str(stats[3]) + '\n',
          '# of untranslated alleles = ' + str(stats[4]) + '\n')

    # make dict to map kws to dfs and return dfs if desired; else return stats
    dict_df = {'final': df_final, 'mismatch': df_mismatch, 'merge': df_merge, 'agg': df_agg, 'raw': df_raw}
    # if saving other intermediate dfs is desired, then export an excel with each df as its own sheet
    if save_intermediate:
        out_intermediate = sample + '_intermediates.xlsx'
        with pd.ExcelWriter(outpath / out_intermediate) as outfile:
            # write stats to first sheet, then the rest of the dfs
            df_stats = pd.DataFrame(data=stats, columns=[sample], index=['n_crispresso', 'n_agg', 'n_mismatches', 'n_untranslated']).transpose()
            df_stats.to_excel(outfile, sheet_name='stats')
            for df_name, df in zip(['df_agg','df_merge','df_mismatch','df_raw'], [df_agg, df_merge, df_mismatch, df_raw]):
                df.to_excel(outfile, sheet_name=df_name)
    t_end = time.perf_counter()
    print(sample + ' allele frequency analysis completed in %.2f sec \n' % (t_end - t_start))

    # return stats if no dataframe outputs desired
    if return_df is None:
        return stats
    # if return_df == all, set return_df to list of all dfs
    elif return_df == 'all':
        return_df = list(dict_df.keys())
    # if return_df is a single dataframe as a string, return that df directly
    elif isinstance(return_df, str) and return_df in list(dict_df.keys()):
        return dict_df[return_df]
    # at this point, return_df should be a list of all dfs desired; return dfs in dict format, otherwise stats if invalid
    if isinstance(return_df, list) and all(key in list(dict_df.keys()) for key in return_df):
        dict_return = {key: dict_df[key] for key in return_df}
        return dict_return
    else:
        print('Value of return_df was invalid. Returning stats.')
        return stats

#%% batch_analyze() - perform analyze_alleles() on a batch of samples

def batch_analyze(list_samples, in_ref, in_batch, out_folder='', list_outfiles=None,
                  save_stats=True, out_stats='batch_analysis_stats.csv', dict_kws=dict()):
    """
    Batch analysis of samples using analyze_alleles() to translate alleles.

    Parameters
    ----------
    list_samples : list of strings
        List of sample names to translate alleles for. Must be found in both the
        'sample_name' and 'name' columns of in_ref and in_batch, respectively.
    in_ref: str or path
        String or path to the reference file. Must have column headers.
        in_ref must have a 'sample_name' col which should correspond to the 'name'
        column in the batch file, and a 'CDS_frame' col, which is the # of nt
        in the coding_seq until the 1st codon for that sample amplicon.
        Example: For a CDS_seq (CA/ATG/TTT/TCA..), the CDS_frame = 2 as the
        first full codon is ATG
    in_batch: str or path
        String or path to the CRISPResso2 batch file used to analyze the NGS data.
        files. Must have columns for the sample name ('name'), the name of the
        amplicon ('amplicon_name'), the sequence of the amplicon ('amplicon_seq'),
        the sequence of the sgRNA ('guide_seq'), and the sequence of the CDS within
        the amplicon ('coding_seq').
    out_folder : str, default ''
        Name of the subfolder to save the output file. The default is the current
        working directory.
    list_outfiles : list of str, default None
        A list of file names for the output csv files. If specified, the file names
        must include the .csv extension and must be the same length as list_samples.
        The default is None, which uses the sample name + '_alleles_tln_freq.csv'.
    save_stats : bool, default True
        Whether to save the analysis stats for the samples as a csv file.
        The default is True.
    out_stats : str, default 'batch_analysis_stats.csv'
        Name of the output csv file for the analysis statistics. The default
        is 'batch_analysis_stats.csv'.
    dict_kws : dict of key, value mappings in format sample: {k:v}
        Dictionary of sample, kwargs mappings to be passed to analyze_alleles(),
        with sample names as keys (must be found in list_samples) and a dictionary
        of keyword arguments as values. NOTE: DO NOT PASS KWARGS IF UNFAMILIAR
        WITH FUNCTION AS IT MAY NOT WORK PROPERLY.
    """

    batch_start = time.perf_counter()
    # generate output csv file names from sample names if none specified
    if list_outfiles is None:
        list_outfiles = [sample + '_alleles_tln_freq.csv' for sample in list_samples]
    # check that list_samples and list_outfiles are same length
    if len(list_samples) != len(list_outfiles):
        raise Exception('list_samples and list_outfiles are not equal length.')
    # make placeholder list to store stats after each sample analysis
    list_stats = []
    # perform analyze_alleles() on the list of samples
    for sample, outfile in zip(list_samples, list_outfiles):
        # check if kwargs passed for sample, if not pass empty dict
        if sample in dict_kws:
            kwargs = dict_kws[sample]
        else:
            kwargs = {}
        temp = analyze_alleles(sample=sample, in_ref=in_ref, in_batch=in_batch,
                               out_folder=out_folder, out_file=outfile, **kwargs)
        list_stats.append(temp)
    # convert the stats into a dataframe (df_stats) and export if desired
    list_dfcols = ['crispresso_alleles', 'aggregated_alleles', 'mismatched_alleles', 'expected_alleles', 'untranslated_alleles']
    df_stats = pd.DataFrame(columns=['sample_name'] + list_dfcols)
    df_stats['sample_name'] = list_samples
    df_stats[list_dfcols] = list_stats
    # make sure dtype is correct for each column
    df_stats = df_stats.astype(dtype={k: v for k,v in zip(df_stats.columns.tolist(), ['str','int','int','int','int','int'])})
    # write df_stats to csv if desired
    if save_stats:
        path = Path.cwd()
        outpath = path / out_folder
        Path.mkdir(outpath, exist_ok=True)
        df_stats.to_csv((outpath / out_stats))

    batch_end = time.perf_counter()
    print('Batch analysis completed in %.2f min' % ((batch_end - batch_start) / 60))
    return df_stats

#%% compare_alleles() - compare translated alleles between samples and merge with CRISPRessoCompare

def compare_alleles(sample1, sample2, in_ref, in_batch, dir_crispresso=None, dir_tlns=Path.cwd(),
                    in_s1=None, in_s2=None, reverse=False, save=True, out_folder='',
                    out_file=None, return_df=None):
    """
    Consolidate allele translations with the CRISPRessoCompare output table.

    For a specified pairwise sample comparison (sample1 vs sample2), this function
    merges the samples' translated allele frequencies generated by analyze_alleles().
    Then, it consolidates the translated alleles with the output from CRISPRessoCompare,
    which contains log2 fold change values (%Reads in sample 1 vs. %Reads in sample2)
    for each allele in the comparison and exports the final table as a csv. If
    any mismatched alleles are found between the merged translated alleles and
    the CRISPRessoCompare output table, then the function will end and return the
    mismatched alleles as a dataframe.

    Parameters
    ----------
    sample1, sample2 : str
        The names of the samples for comparison. The samples will be processed
        as sample1 normalized to sample2 (e.g. sample1 = drug, sample2 = vehicle).
        The samples passed for sample1 and sample2 must match the samples passed
        to the -n1 and -n2 parameters in CRISPRessoCompare, respectively. The
        sample names must also be present in the 'sample_name' and 'name' columns
        of in_ref and in_batch.
    in_ref: str or path
        String or path to the reference file. Must have column headers.
        in_ref must have a 'sample_name' col which should correspond to the 'name'
        column in the batch file, and a 'CDS_frame' col, which is the # of nt
        in the coding_seq until the 1st codon for that sample amplicon.
        Example: For a CDS_seq (CA/ATG/TTT/TCA..), the CDS_frame = 2 as the
        first full codon is ATG.
    in_batch: str or path
        String or path to the CRISPResso2 batch file used to analyze the NGS data.
        files. Must have columns for the sample name ('name'), the name of the
        amplicon ('amplicon_name'), the sequence of the amplicon ('amplicon_seq'),
        the sequence of the sgRNA ('guide_seq'), and the sequence of the CDS within
        the amplicon ('coding_seq').
    dir_crispresso : str or Path object, default None
        The name of the subfolder containing the CRISPRessoCompare output files.
        The default is None, which will use the sample1 and sample2 names to generate
        the default CRISPRessoCompare output name (CRISPRessoCompare_on_sample1_vs_sample2).
        If the -n parameter was specified in CRISPRessoCompare, use that string
        for dir_crispresso. If a string is passed, it is assumed to be a subfolder
        of the cwd. If a Path object is passed, it is assumed to be a direct path.
    dir_tlns : str or Path object, default Path.cwd()
        The name of the subfolder containing the translated allele frequency csv
        files (generated by analyze_alleles()) for the specified samples. The
        default is the current working directory. If a string is passed, it is assumed
        to be a subfolder of the cwd. If a Path object is passed, it is assumed to be a direct path.
    in_s1, in_s2 : str, default None
        The name of the translated allele frequency csv files for sample1 and sample2.
        The default is None for both in_s1 and in_s2, which will generate the default
        out_file names from analyze_alleles (sample_alleles_tln_freq.csv) using the
        sample1/sample2 inputs. For example, if sample1 = 'A1_DAC' and sample2 = 'A1_DMSO',
        then in_s1 = 'A1_DAC_alleles_tln_freq.csv' and in_s2 = 'A1_DMSO_alleles_tln_freq.csv'
    reverse : bool, default False
        Whether the sample was sequenced in the reverse orientation of the CDS.
        Reversed amplicons are processed by CRISPResso2 in that orientation, which
        requires different processing in order to return the correct translations.
    save : bool, default True
        Whether to save the final consolidated alleles df (df_final) as a csv file.
        The default is True.
    out_folder : str, default ''
        Name of the subfolder to save the output file. The default is the current
        working directory.
    out_file : str, default None
        Name of the output csv file. If specified, the string must include the
        .csv extension. The default is None, which uses the latter part of dir_crispresso
        (everything after 'CRISPRessoCompare_on_') and '_comparison_tln.csv'.
        For example, if dir_crispresso = 'CRISPRessoCompare_on_A1', it takes 'A1'
        and the out_file name is 'A1_comparison_tln.csv'.
    return_df : {None, 'final', 'mismatch', 'all'}, default None
        Whether to return dataframes at function end. The default is None, which
        will return the comparison statistics tuple. Calling 'final' returns the
        final df (df_final) and 'mismatch' returns the mismatched alleles df (df_mm).
        If a list or 'all' is passed, then it will return multiple dfs as a dict
        with the key as the return_df str and the df as the value.
    """

    ### IMPORT REFERENCE AND BATCH FILES AND CLEAN UP

    # import reference and batch files
    df_ref = pd.read_csv(in_ref)
    df_batch = pd.read_csv(in_batch, sep='\t')
    # column header check for df_ref and df_batch
    list_refcols = ['sample_name', 'CDS_frame']
    list_batchcols = ['name', 'amplicon_name', 'amplicon_seq', 'guide_seq', 'coding_seq']
    col_check(df_ref, list_refcols, 'in_ref')
    col_check(df_batch, list_batchcols, 'in_batch')
    # clean up and merge information into df_ref
    df_batch.rename(columns={'name': 'sample_name'}, inplace=True)
    df_ref = df_ref.merge(df_batch, on='sample_name')

    ### DEFINE FILE PATHS/NAMES AND IMPORT DATA

    # define path(s) to the CRISPRessoCompare output files; default is the CRISPRessoCompare name default
    if dir_crispresso is None:
        dir_crispresso = 'CRISPRessoCompare_on_' + sample1 + '_vs_' + sample2
    # assume Path obj is direct path to dir, str is subdir of the cwd
    if isinstance(dir_crispresso, str):
        inpath = Path.cwd() / dir_crispresso
    elif isinstance(dir_crispresso, Path):
        inpath = dir_crispresso
    # define path(s) to the translated allele output files from analyze_alleles(); default is cwd
    if isinstance(dir_tlns, str):
        tln_path = Path.cwd() / dir_tlns
    elif isinstance(dir_tlns, Path):
        tln_path = dir_tlns
    # define the translated allele output file names; default is sample + '_alleles_tln_freq.csv'
    if in_s1 is None:
        in_s1 = sample1 + '_alleles_tln_freq.csv'
    if in_s2 is None:
        in_s2 = sample2 + '_alleles_tln_freq.csv'

    # list of columns to merge on (e.g. allele characteristics; not reads)
    list_mergecols = ['aln_seq', 'ref_seq', 'Unedited', 'n_del', 'n_ins', 'n_mut', 'amp_indel', 'indel', 'mut_type', 'prot_aln', 'prot_ref']
    # dict of columns corresponding to n_reads and pct_reads for each sample
    dict_readcols = {prefix1 + sample: prefix2 + sample for prefix1,prefix2 in zip(['#Reads_', '%Reads_'], ['n_reads_', 'pct_reads_']) for sample in [sample1, sample2]}
    # dict to rename the CRISPRessoCompare columns to shorthand for ease
    dict_cols = {'Reference_Sequence': 'ref_seq', 'n_deleted': 'n_del', 'n_inserted': 'n_ins', 'n_mutated': 'n_mut'}

    # define amp_name and grna in preparation for import
    amp_name = df_ref.loc[df_ref['sample_name'] == sample1]['amplicon_name'].values[0]
    grna = df_ref.loc[df_ref['sample_name'] == sample1]['guide_seq'].values[0]
    # check that sgRNA seq and amplicon name are identical for both samples
    if df_ref.loc[df_ref['sample_name'] == sample2]['amplicon_name'].values[0] != amp_name:
        raise Exception('The amplicon names for ' + sample1 + ' and ' + sample2 + ' do not match')
    if df_ref.loc[df_ref['sample_name'] == sample2]['guide_seq'].values[0] != grna:
        raise Exception('The guide seqs for ' + sample1 + ' and ' + sample2 + ' do not match')

    # define the CRISPRessoCompare allele frequency filename
    infile = amp_name + '.Alleles_frequency_table_around_sgRNA_' + grna + '.txt'
    # new version of crispresso changed output file name to guide_name if specified
    if 'guide_name' in df_ref.columns.tolist():
        guide_name = df_ref.loc[df_ref['sample_name'] == sample1]['guide_name'].values[0]
        if df_ref.loc[df_ref['sample_name'] == sample2]['guide_name'].values[0] != guide_name:
            raise Exception('The guide names for ' + sample1 + ' and ' + sample2 + ' do not match')
        if not (inpath / infile).exists():
            infile = amp_name + '.Alleles_frequency_table_around_' + guide_name + '.txt'
        if not (inpath / infile).exists():
            raise Exception('Could not find the CRISPRessoCompare alleles output .txt file')
    # import the CRISPRessoCompare allele frequency around sgRNA cut site and rename cols
    df_crispresso = pd.read_csv(inpath / infile, sep='\t', index_col=0).rename(columns=dict_cols).rename(columns=dict_readcols)

    # import the translated allele frequency csv files and merge
    df_s1 = pd.read_csv(tln_path / in_s1, index_col=0).drop(columns='n_alleles')
    df_s2 = pd.read_csv(tln_path / in_s2, index_col=0).drop(columns='n_alleles')
    df_tlns = pd.merge(df_s1, df_s2, how='outer', suffixes=('_' + sample1, '_' + sample2), on=list_mergecols, indicator=True)

    # if reverse is True, then check to see if rev complement necessary for merging
    if reverse:
        ref_seq = df_crispresso.loc[df_crispresso['Unedited'] == True]['ref_seq'].values[0]
        s1_seq = df_s1.loc[df_s1['Unedited'] == True]['ref_seq'].values[0]
        s2_seq = df_s2.loc[df_s2['Unedited'] == True]['ref_seq'].values[0]
        if s1_seq == s2_seq:
            if ref_seq == s1_seq:
                pass
            elif str(dna(ref_seq, rc=True)) == s1_seq:
                df_crispresso['ref_seq'] = df_crispresso['ref_seq'].apply(lambda x: str(dna(x, rc=True)))
            else:
                raise Exception('The CRISPRessoCompare ref_seq does not match the samples.')
        else:
            raise Exception('The ref_seqs for sample1 and sample2 do not match.')

    ### PREPARE DATAFRAMES FOR MERGING AND CORRECT MISMATCHES IF PRESENT

    # check that df_tlns and df_crispresso are the same size; if not, check for mismatches
    # these mismatches likely arise from differences in indel/mut_type/prot_aln/prot_ref
    if df_tlns.shape[0] != df_crispresso.shape[0]:
        # get idxs for s1/s2 alleles, then merge (df_tlns2) without indel/mut_type/prot_aln/prot_ref
        df_s1['idx'] = df_s1.index.tolist()
        df_s2['idx'] = df_s2.index.tolist()
        df_tlns2 = pd.merge(df_s1, df_s2, how='outer', suffixes=('_s1', '_s2'), on=list_mergecols[:-4], indicator=True)
        # if df_tlns2 is still not the same size as df_crispresso, raise an exception
        if df_tlns2.shape[0] != df_crispresso.shape[0]:
            raise Exception('The merged alleles and CRISPResso output tables are not equal length.')
        # isolate alleles in both samples and find mismatches in mut_type, prot_aln/ref
        df_tlns2 = df_tlns2.loc[df_tlns2['_merge'] == 'both'].copy()
        list_mm1 = [col + '_s1' for col in list_mergecols[-4:]]
        list_mm2 = [col + '_s2' for col in list_mergecols[-4:]]
        df_mm1 = df_tlns2[list_mm1].rename(columns={k:v for k,v in zip(list_mm1, list_mergecols[-4:])}).copy()
        df_mm2 = df_tlns2[list_mm2].rename(columns={k:v for k,v in zip(list_mm2, list_mergecols[-4:])}).copy()
        df_mm = df_tlns2.loc[(df_mm1 != df_mm2).any(axis=1)].copy()
        df_mm['allele'] = np.where(df_mm['n_reads_s1'] > df_mm['n_reads_s2'], 's1', np.where(df_mm['n_reads_s2'] > df_mm['n_reads_s1'], 's2', 'equal'))
        df_mm['allele'] = np.where(df_mm['allele'] == 'equal', np.where(df_mm['pct_reads_s1'] >= df_mm['pct_reads_s2'], 's1', 's2'), df_mm['allele'])
        # call mut_type/prot_aln/ref on sample with more reads, if equal, use pct_reads
        df_adj = pd.DataFrame(columns=list_mergecols[-4:], index=df_mm.index)
        df_adj[['idx_s1', 'idx_s2', 'allele']] = df_mm[['idx_s1', 'idx_s2', 'allele']]
        df_adj.loc[df_adj['allele'] == 's1', list_mergecols[-4:]] =  df_mm.loc[df_mm['allele'] == 's1', list_mm1].values
        df_adj.loc[df_adj['allele'] == 's2', list_mergecols[-4:]] =  df_mm.loc[df_mm['allele'] == 's2', list_mm2].values
        # replace the alleles in sample1/sample2 with the consensus call in df_adj
        df_s1.loc[df_adj['idx_s1'], list_mergecols[-4:]] = df_adj[list_mergecols[-4:]].values
        df_s2.loc[df_adj['idx_s2'], list_mergecols[-4:]] = df_adj[list_mergecols[-4:]].values
        # drop the idx cols for df_s1/s2 and redo the df_tlns merge with new values
        df_s1.drop(columns='idx', inplace=True)
        df_s2.drop(columns='idx', inplace=True)
        df_tlns = pd.merge(df_s1, df_s2, how='outer', suffixes=('_' + sample1, '_' + sample2), on=list_mergecols, indicator=True)
        # if merge still does not lead to same size, raise an exception
        if df_tlns.shape[0] != df_crispresso.shape[0]:
            print('The # of alleles in df_tlns (merged) and df_crispresso are not equal.\n',
                  'The merged alleles df (df_tlns) has ' + str(df_tlns.shape[0]) + ' rows.\n',
                  'The CRISPRessoCompare df (df_crispresso) has ' + str(df_crispresso.shape[0]) + ' rows.')
            raise Exception('The merged alleles and CRISPResso output are not equal length after mismatch correction.')
        # clean up df_mm (if mismatches exist)
        df_mm = df_mm.reindex(columns=list_mergecols[:-4] + ['n_reads_s1', 'n_reads_s2', 'pct_reads_s1', 'pct_reads_s2'] + list_mm1 + list_mm2 + ['idx_s1', 'idx_s2', 'allele'])
    # otherwise, if there are no mismatches, generate an empty dataframe
    else:
        list_mm1 = [col + '_s1' for col in list_mergecols[-4:]]
        list_mm2 = [col + '_s2' for col in list_mergecols[-4:]]
        df_mm = pd.DataFrame(columns=list_mergecols[:-4] + ['n_reads_s1', 'n_reads_s2', 'pct_reads_s1', 'pct_reads_s2'] + list_mm1 + list_mm2 + ['idx_s1', 'idx_s2', 'allele'])

    # now fill in NaNs in df_tlns and round decimals in all dfs for proper merging
    df_tlns[list(dict_readcols.values())] = df_tlns[list(dict_readcols.values())].fillna(value=0)
    df_tlns[list(dict_readcols.values())] = df_tlns[list(dict_readcols.values())].round(decimals=5)
    df_crispresso[list(dict_readcols.values())] = df_crispresso[list(dict_readcols.values())].round(decimals=5)
    # get CRISPResso columns except for the LFC column for cross-checking dfs
    list_checkcols = df_crispresso.columns[:-1].tolist()

    # the CRISPRessoCompare alleles are not unique (missing info), so we can't use pd.merge
    # therefore, check that df_tlns and df_crispresso are identical; if so, add 'each_LFC' col to df_tlns
    if (df_tlns[list_checkcols] == df_crispresso[list_checkcols]).all(axis=1).all():
        df_tlns['each_LFC'] = df_crispresso['each_LFC']
    # otherwise, identify mismatches despite equal # of alleles and return
    else:
        df_mismatch = df_tlns.loc[~(df_tlns[list_checkcols] == df_crispresso[list_checkcols]).all(axis=1)]
        list_mmcols2 = (df_tlns[list_checkcols] == df_crispresso[list_checkcols]).all(axis=0)
        list_mmcols2 = list_mmcols2.loc[list_mmcols2 == False].index.tolist()
        list_mmrows = df_mismatch.index.tolist()
        print('The following columns do not contain identical values: ' + str(list_mmcols2) + '\n',
              'The following rows (alleles) do not contain identical values: ' + str(list_mmrows))
        print('The merged alleles and CRISPResso output tables are not identical.\n',
              'Returning mismatches for ' + sample1 + ' vs ' + sample2)
        return df_mismatch

    ### CLEAN UP DATAFRAMES, EXPORT FILES, RETURN STATS/DATAFRAMES

    # rearrange columns into a nice coherent order
    list_finalcols = list_mergecols + list(dict_readcols.values())
    df_final = pd.DataFrame(columns=list_finalcols)
    df_final[list_finalcols] = df_tlns[list_finalcols]
    # round the LFC values for convenience
    df_final['LFC'] = df_tlns['each_LFC'].round(decimals=3)

    # save and export files
    if out_file is None:
        out_file = dir_crispresso.replace('CRISPRessoCompare_on_', '') + '_comparison.csv'
    if save:
        outpath = Path.cwd() / out_folder
        Path.mkdir(outpath, exist_ok=True)
        df_final.to_csv((outpath / out_file))
    # calculate and print output stats
    stats = (df_final.shape[0], # n_alleles total
             df_final.loc[df_final['pct_reads_%s' % sample2] == 0].shape[0], # n_alleles only in sample1
             df_final.loc[df_final['pct_reads_%s' % sample1] == 0].shape[0], # n_alleles only in sample2
             df_final['LFC'].min(), # min LFC value for sample1 vs sample2
             df_final['LFC'].max(), # max LFC value for sample1 vs sample2
             df_final.loc[df_final['mut_type'] == 'in-frame']['LFC'].max(), # max LFC for in-frame alleles
             df_final.loc[df_final['mut_type'].str.contains('frameshift|splice site|indel exceeds cds')]['LFC'].max(), # max LFC for non-in-frame alleles
             df_final.loc[df_final['mut_type'] == 'wild-type']['LFC'].values[0]) # LFC for the WT allele
    print('# of total alleles = ' + str(stats[0]) + '\n',
          '# of unique alleles in ' + sample1 + ' = ' + str(stats[1]) + '\n',
          '# of unique alleles in ' + sample2 + ' = ' + str(stats[2]) + '\n',
          'Min overall LFC = ' + str(stats[3]) + '\n',
          'Max overall LFC = ' + str(stats[4]) + '\n',
          'Max in-frame LFC = ' + str(stats[5]) + '\n',
          'Max non in-frame LFC = ' + str(stats[6]) + '\n',
          'WT LFC = ' + str(stats[7]) + '\n')
    print(sample1 + ' vs ' + sample2 + ' allele comparison completed \n')

    # make dict to map kws to dfs and return dfs if desired; else return stats
    dict_df = {'final': df_final, 'mismatch': df_mm, 'stats': stats}
    # return stats if no dataframe outputs desired
    if return_df is None:
        return stats
    # if return_df == all, set return_df to list of all dfs
    elif return_df == 'all':
        return_df = list(dict_df.keys())
    # if return_df is a single dataframe as a string, return that df directly
    elif isinstance(return_df, str) and return_df in list(dict_df.keys()):
        return dict_df[return_df]
    # at this point, return_df should be a list of all desired dfs; return dfs in dict, otherwise stats if invalid
    if isinstance(return_df, list) and all(key in list(dict_df.keys()) for key in return_df):
        dict_return = {key: dict_df[key] for key in return_df}
        return dict_return
    else:
        print('Value of return_df was invalid. Returning stats.')
        return stats

#%% batch_compare() - perform compare_alleles() on a batch of sample comparisons

def batch_compare(list_comparisons, in_ref, in_batch, list_crispresso_dirs=None,
                  dir_tlns='', list_in_tlns=None, out_folder='', list_outfiles=None,
                  save_stats=True, out_stats='batch_comparison_stats.csv', save_mismatches=True,
                  out_mismatches='batch_comparison_mismatches.csv', dict_kws=dict()):
    """
    Batch pairwise comparison of samples using compare_alleles().

    This function takes a list of sample comparisons and uses compare_alleles()
    to merge the translated allele frequencies (from analyze_alleles()) for each
    pairwise comparison and consolidate them with the CRISPRessoCompare output.
    For more information, see the compare_alleles() documentation.

    Parameters
    ----------
    list_comparisons : list of tuples in format (sample1, sample2)
        A list of tuples of (sample1, sample2) for performing pairwise comparisons.
        The comparison will be processed as sample1 vs. sample2 (e.g. drug - vehicle).
        The samples passed in each tuple must be present in the 'sample_name' and
        'name' columns of in_ref and in_batch. sample1 and sample2 must match the
        -n1 and -n2 parameters of CRISPRessoCompare.
    in_ref: str or path
        String or path to the reference file. Must have column headers.
        in_ref must have a 'sample_name' col which should correspond to the 'name'
        column in the batch file, and a 'CDS_frame' col, which is the # of nt
        in the coding_seq until the 1st codon for that sample amplicon.
        Example: For a CDS_seq (CA/ATG/TTT/TCA...), the CDS_frame is 2 as the
        first full codon is ATG.
    in_batch: str or path
        String or path to the CRISPResso2 batch file used to analyze the NGS data.
        files. Must have columns for the sample name ('name'), the name of the
        amplicon ('amplicon_name'), the sequence of the amplicon ('amplicon_seq'),
        the sequence of the sgRNA ('guide_seq'), and the sequence of the CDS within
        the amplicon ('coding_seq').
    list_crispresso_dirs : list of str, default None
        A list of strings corresponding to the CRISPRessoCompare output subfolders.
        Must be in the same order and length as list_comparisons. The default is
        None, which uses the sample names passed in list_comparisons to generate
        the CRISPRessoCompare folder names (CRISPRessoCompare_on_sample1_vs_sample2).
        The directories should match the -n parameter of CRISPRessoCompare
    dir_tlns : str, default ''
        The name of the subfolder with the translated allele frequency files
        (from analyze_alleles()). The default is the current working directory.
    list_in_tlns : list of tuples in format (sample1_file, sample2_file), default None
        A list of tuples of (sample1_file, sample2_file) corresponding
        to the translated allele frequency csv file names for each comparison.
        Must be in the same order and length as list_comparisons. The default is
        None, which uses the sample names passed in list_comparisons to generate
        file names (sample1_alleles_tln_freq.csv, sample2_alleles_tln_freq.csv).
    out_folder : str, default ''
        Name of the subfolder to save the output files. The default is the current
        working directory.
    list_outfiles : list of str, default None
        A list of file names for the output csv files. If specified, the file names
        must include the .csv extension and must be the same length as list_comparisons.
        The default is None, which uses the latter part of the CRISPRessoCompare folder
        name for each comparison in list_crispresso_dirs (everything after
        'CRISPRessoCompare_on_') and '_comparison_tln.csv' to generate an out_file
        name. For example, if the CRISPResso folder name is 'CRISPRessoCompare_on_A1',
        the out_file name is 'A1_comparison_tln.csv'.
    save_stats : bool, default True
        Whether to save the comparison statistics as a csv file. The default is True.
    out_stats : str, default 'batch_comparison_stats.csv'
        Name of the output csv file for the comparison statistics. The default is
        'batch_comparison_stats.csv'.
    save_mismatches : bool, default True
        Whether to save the aggregate mismatched alleles in the comparisons as
        a csv file. The default is True.
    out_stats : str, default 'batch_comparison_mismatches.csv'
        Name of the output csv file for the aggregate mismatched alleles. The
        default is 'batch_comparison_mismatches.csv'.
    **kwargs : key, value mappings in format x=y
        All other keyword arguments to be passed to compare_alleles(). See the
        compare_alleles() documentation for more information. NOTE: DO NOT PASS
        KWARGS IF UNFAMILIAR WITH FUNCTION AS IT MAY NOT WORK PROPERLY.
    dict_kws : dict of key, value mappings in format comparison: {k:v}
        Dictionary of comparison, kwargs mappings to be passed to compare_alleles(),
        with comparison tuples as keys (must be found in list_comparisons) and a
        dictionary of keyword arguments as values. NOTE: DO NOT PASS KWARGS IF
        UNFAMILIAR WITH FUNCTION AS IT MAY NOT WORK PROPERLY.
    """

    batch_start = time.perf_counter()

    # generate CRISPRessoCompare folder names using list_comparisons if None specified
    if list_crispresso_dirs is None:
        list_crispresso_dirs = ['CRISPRessoCompare_on_' + s1 + '_vs_' + s2 for s1,s2 in list_comparisons]
    # generate sample input file names using list_comparisons if None specified
    if list_in_tlns is None:
        list_in_tlns = [(s1 + '_alleles_tln_freq.csv', s2 + '_alleles_tln_freq.csv') for s1,s2 in list_comparisons]
    # generate output file names using list_comparisons if None specified
    if list_outfiles is None:
        list_outfiles = [x.replace('CRISPRessoCompare_on_', '') + '_comparison_tlns.csv' for x in list_crispresso_dirs]
    # check that list_comparisons, list_crispresso_dirs, list_in_tlns, list_outfiles are same length
    if any(len(x) != len(list_comparisons) for x in [list_comparisons, list_crispresso_dirs, list_in_tlns, list_outfiles]):
        raise Exception('list_comparisons, list_crispresso_dirs, list_in_tlns, list_outfiles are not all equal length')
    # make placeholder list to store stats and df to store mismatches after each comparison
    list_stats = []
    df_mismatches = pd.DataFrame()
    # make placeholder list to report any comparisons that failed
    list_failed = []
    # perform compare_alleles() on the list of comparisons
    for comparison, dir_crispresso, in_tlns, outfile in zip(list_comparisons, list_crispresso_dirs, list_in_tlns, list_outfiles):
        # check if kwargs passed for sample, if not pass empty dict
        if comparison in dict_kws:
            kwargs = dict_kws[comparison]
        else:
            kwargs = {}
        temp = compare_alleles(sample1=comparison[0], in_s1=in_tlns[0], sample2=comparison[1], in_s2=in_tlns[1],
                               in_ref=in_ref, in_batch=in_batch, dir_crispresso=dir_crispresso, dir_tlns=dir_tlns,
                               out_folder=out_folder, out_file=outfile, return_df=['mismatch', 'stats'], **kwargs)
        # if compare_alleles was successful, append the stats to list_stats and mismatches to list_mismatches
        if isinstance(temp, dict):
            list_stats.append(temp['stats'])
            temp_mm = temp['mismatch']
            temp_mm['comparison'] = str(comparison)
            df_mismatches = df_mismatches.append(temp_mm, ignore_index=True)
        # if the comparison was not successful (df_mismatch returned); append tuple w/ # of mismatches, then None x7
        else:
            list_stats.append(tuple([temp.shape[0]] + [None for x in range(0,7)]))
            list_failed.append(comparison)
    # now convert the stats into a dataframe (df_stats) and export if desired
    list_dfcols = ['n_alleles_total_or_mismatch', 'n_unique_sample1', 'n_unique_sample2',
                   'min_LFC_all', 'max_LFC_all', 'max_LFC_in-frame', 'max_LFC_not_in-frame', 'WT_LFC']
    df_list = pd.DataFrame(data=list_stats, columns=list_dfcols)
    df_stats = pd.DataFrame(data=list_comparisons, columns=['sample1', 'sample2'])
    df_stats = df_stats.join(df_list)
    # make sure df is correct type for each column
    dict_dtypes = {k:v for k,v in zip(df_stats.columns.tolist(), ['str','str','int','int','int'] + ['float' for x in range(0,5)])}
    df_stats = df_stats.astype(dtype=dict_dtypes)
    # write df_stats and/or df_mismatches to csv if desired
    if save_stats:
        outpath = Path.cwd() / out_folder
        Path.mkdir(outpath, exist_ok=True)
        df_stats.to_csv((outpath / out_stats))
    if save_mismatches:
        outpath = Path.cwd() / out_folder
        Path.mkdir(outpath, exist_ok=True)
        df_mismatches.to_csv((outpath / out_mismatches))

    if len(list_failed) > 0:
        print('These comparisons were not completed due to mismatches: ' + str(list_failed))
    batch_end = time.perf_counter()
    print('Batch comparison completed in %.2f sec' % (batch_end - batch_start))
    return {'stats': df_stats, 'mismatches': df_mismatches}

#%% internal fx for processing sgRNA comparisons data

### group_comp(), agg_comps == fx for aggregating alleles with 1-2 nt point mutations
def group_comp(group, s1_col, s2_col, out_col):
    s1_pct, s2_pct = group[s1_col].max(), group[s2_col].max()
    if s1_pct > s2_pct:
        val = group.loc[group[s1_col].idxmax(), out_col]
    elif s2_pct > s1_pct:
        val = group.loc[group[s2_col].idxmax(), out_col]
    else:
        val = group.loc[group[s1_col].idxmax(), out_col]
    return val

def agg_comps(infile, clean_anno=True, keep_col_names=False):
    if isinstance(infile, str):
        df_input = pd.read_csv(Path.cwd() / infile, index_col=0)
    elif isinstance(infile, Path):
        df_input = pd.read_csv(infile, index_col=0)
    elif isinstance(infile, pd.DataFrame):
        df_input = infile.copy()
    if not all(df_input.iloc[:,-5:-1].columns.str.contains('reads')):
        raise Exception('Cannot find read columns')
    dict_readcols = {k:v for k,v in zip(df_input.iloc[:,-5:-1].columns.tolist(),['n_reads_s1','n_reads_s2','pct_reads_s1','pct_reads_s2'])}
    df_comp = df_input.copy().rename(columns=dict_readcols).drop(columns=['Unedited','LFC'])
    # make new cols aln/ref seq (replace nt w/ placeholder Ns to identify indel position)
    # also make new cols for n_alleles and the dominant sample (greater pct_reads)
    df_comp['n_alleles'] = 1
    df_comp['aln_pos'] = df_comp['aln_seq'].str.replace('[ACGT]', 'N', regex=True)
    df_comp['ref_pos'] = df_comp['ref_seq'].str.replace('[ACGT]', 'N', regex=True)
    # clean up annotations (remove likely, identify nonsense, etc.)
    if clean_anno:
        df_comp['mut_type'] = df_comp['mut_type'].str.replace('likely ', '')
        df_comp['prot_aln'] = df_comp['prot_aln'].str.replace('likely ', '')
        df_comp['prot_ref'] = df_comp['prot_ref'].str.replace('likely ', '')
        df_comp['mut_type'] = np.where(df_comp['prot_aln'].str.contains('\*', regex=True), 'nonsense', df_comp['mut_type'])
        seq_wt = df_comp[df_comp['mut_type'] == 'wild-type']['prot_ref'].values[0]
        df_comp['mut_type'] = np.where(df_comp['prot_aln'] == seq_wt, 'wild-type', df_comp['mut_type'])

    # make dict of output column names and groupby transformations (via pd.agg)
    # for allele info columns, take the allele info using the sample with the most pct reads (via pd.apply)
    # sum the num/pct reads for sample1/sample2
    dict_agg = {x:pd.NamedAgg(x, 'sum') for x in dict_readcols.values()}
    # n_major/pct_major == reads contributed by the primary/major allele before aggregation
    dict_agg.update({y:pd.NamedAgg(x, 'max') for x,y in zip(dict_readcols.values(),['n_major_s1','n_major_s2','pct_major_s1','pct_major_s2'])})
    dict_agg.update({'n_alleles': pd.NamedAgg('n_alleles', 'sum')}) # number of alleles aggregated into the single allele
    # make list of columns for groupby
    list_groupcols = ['n_del','n_ins','amp_indel','indel','aln_pos','ref_pos']
    # make list of allele info columns
    list_infocols = ['aln_seq','ref_seq','n_del','n_ins','n_mut','amp_indel','indel','mut_type','prot_aln','prot_ref']

    # filter alleles w/ >2 point mutations and any +1 insertions (the +1 nt may vary, so don't merge)
    df_subs = df_comp.loc[(df_comp['n_mut'] > 2) | ((df_comp['n_ins'] == 1) & (df_comp['amp_indel'] == 1))].copy().drop(columns=['aln_pos','ref_pos'])
    df_subs = df_subs.assign(**{k:df_comp[v] for k,v in zip(['n_major_s1','n_major_s2','pct_major_s1','pct_major_s2'], dict_readcols.values())})
    # now isolate the rest of the alleles as a groupby object and make 2 dfs -- info and agg
    grouped = df_comp.loc[~df_comp.index.isin(df_subs.index)].groupby(list_groupcols, as_index=False)
    df_agg = grouped.agg(**dict_agg).drop(columns=['aln_pos','ref_pos'])
    df_info = pd.DataFrame(columns=list_infocols)
    for col in list_infocols:
        df_info[col] = grouped.apply(lambda x: group_comp(x, 'pct_reads_s1', 'pct_reads_s2', col)).reset_index(drop=True).iloc[:,-1]
    if not (df_agg[['n_del', 'n_ins', 'amp_indel', 'indel']] == df_info[['n_del', 'n_ins', 'amp_indel', 'indel']]).all().all():
        print('Warning! df_agg does not equal df_info')
    # merge info and agg dfs; reindex cols just to be safe
    df_merge = df_info.join(df_agg.drop(columns=['n_del', 'n_ins', 'amp_indel', 'indel']))
    df_merge = df_merge.reindex(columns=list_infocols + list(dict_agg.keys()))
    # now merge df_subs with df_merge to finalize the dataframe; re-calculate LFC
    df_final = df_merge.append(df_subs, ignore_index=True).sort_values(by='pct_reads_s1', ascending=False).reset_index(drop=True)
    df_final['LFC'] = np.log2((df_final['pct_reads_s1'] + 0.1) / (df_final['pct_reads_s2'] + 0.1))
    # add new mut_type annotation column (wt/if/lof)
    df_final['mut_type2'] = np.where(df_final['mut_type'].str.contains('wild-type|in-frame'), df_final['mut_type'], 'loss-of-function')
    # calculate adjusted allele frequencies as % of non-edited (i.e. non-WT) reads
    edited_pcts = (df_final[df_final['mut_type'] != 'wild-type']['pct_reads_s1'].sum(), df_final[df_final['mut_type'] != 'wild-type']['pct_reads_s2'].sum())
    df_final['adj_freq_s1'] = df_final['pct_reads_s1'] / edited_pcts[0] * 100
    df_final['adj_freq_s2'] = df_final['pct_reads_s2'] / edited_pcts[1] * 100
    # set adj_freq_s1/s2 to np.nan for wild-type alleles
    df_final['adj_freq_s1'] = np.where(df_final['mut_type'] == 'wild-type', np.nan, df_final['adj_freq_s1'])
    df_final['adj_freq_s2'] = np.where(df_final['mut_type'] == 'wild-type', np.nan, df_final['adj_freq_s2'])
    # if desired, rename columns back to original
    if keep_col_names:
        df_final = df_final.rename(columns={v:k for k,v in dict_readcols.items()})
    return df_final

#%% indelphi_process() -- for processing the indelphi data w/ CDS awareness

def indelphi_process(sample, in_indelphi, in_ref, reverse=False):

    ### SET UP ALIGNMENT TOOLS ###

    # dna alignment uses the NUC.4.4 (EDNAFULL) substitution matrix
    dna_align = Align.PairwiseAligner()
    dna_align.substitution_matrix = sm.load('NUC.4.4')
    dna_align.open_gap_score = -10 # penalty for opening a gap
    dna_align.extend_gap_score = -0.5 # penalty for extending gap

    # protein alignment uses the BLOSUM62 substitution matrix
    # gap penalties based on EMBOSS Needle defaults (except we penalize end gaps)
    prot_align = Align.PairwiseAligner()
    prot_align.substitution_matrix = sm.load('BLOSUM62')
    prot_align.open_gap_score = -10 # penalty for opening a gap (including end)
    prot_align.extend_gap_score = -0.5 # penalty for extending gap (including end)

    # codon alignment with the NUC.4.4 (EDNAFULL) substitution matrix
    # gap penalties based on CRISPResso defaults
    cdn_align = Align.PairwiseAligner()
    cdn_align.substitution_matrix = codon_submat()
    cdn_align.open_gap_score = -60 # penalty for opening a gap (including end)
    cdn_align.extend_gap_score = -6 # penalty for extending a gap (including end)

    ### IMPORT REFERENCE AND BATCH FILES AND CLEAN UP ###

    # clean up the inDelphi input df/csv real quick
    df_pred = in_indelphi.copy().rename(columns={'Category':'mut_cat','Predicted frequency':'pred_freq', 'Genotype': 'edit_seq'})
    df_pred = df_pred.drop(columns=['Genotype position', 'Inserted Bases','Microhomology length'])
    df_pred['indel'] = np.where(df_pred['mut_cat'] == 'del', df_pred['Length'] * -1, df_pred['Length'])
    list_dphicols = ['Name', 'indel', 'pred_freq', 'edit_seq', 'mut_cat', 'Length']
    df_pred = df_pred.reindex(columns=list_dphicols)

    # import the necessary info from in_ref
    df_ref = in_ref.copy()
    seq_amp = df_ref.loc[df_ref['sample'] == sample, 'dphi_wdw_seq'].values[0] # +/- 60 nt around cut_site
    seq_cds = df_ref.loc[df_ref['sample'] == sample]['dphi_cds_seq'].values[0]
    seq_grna = df_ref.loc[df_ref['sample'] == sample]['sgRNA_seq'].values[0]
    seq_crispresso = df_ref.loc[df_ref['sample'] == sample]['crispresso_wdw_seq'].values[0]
    # if the seq orientation is reversed w.r.t. to CDS sense (due to gRNA), get RC
    if reverse:
        seq_amp = str(Seq(seq_amp).reverse_complement())
        seq_cds = str(Seq(seq_cds).reverse_complement())

    # adjust grna seq based on seq orientation and find cut site position (assumes SpCas9 +3)
    if seq_grna in seq_amp:
        idx_cut = (seq_amp.find(seq_grna) + len(seq_grna) - 3) - 1
    elif str(Seq(seq_grna).reverse_complement()) in seq_amp:
        seq_grna = str(Seq(seq_grna).reverse_complement())
        idx_cut = (seq_amp.find(seq_grna) + 3) - 1
    else:
        raise Exception('gRNA sequence not found in the inDelphi sequence context')

    # define the CDS frame (trim partial codons) using the CDS_frame ref info
    # if the seq direction is reversed, then the CDS_frame in df_ref should refer to the 3' end of the CDS
    # that is, the df_ref CDS_frame is the # of nt until the 1st full codon at the 5' of the full_window_seq
    cds_l = int(df_ref.loc[df_ref['sample'] == sample]['CDS_frame_dphi'].values[0])
    if not reverse and cds_l in (0,1,2):
        cds_r = len(seq_cds[cds_l:]) % 3 * -1 # convert # of nt in last codon to neg for slicing str
        if cds_r == 0: # can't slice str on right with 0, so replace 0 with None
            cds_r = None
    # if the sequence direction is in the reverse orientation, convert cds_l to cds_r and calculate cds_l
    elif reverse and cds_l in (0,1,2):
        cds_r = cds_l * -1
        if cds_r == 0: # can't slice str on right with 0, so replace 0 with None
            cds_r = None
        cds_l = len(seq_cds[:cds_r]) % 3
    else:
        raise Exception('CDS_frame is not 0, 1, or 2. Please check values.')

    # look for intron/exon boundaries ('edge') by aligning the CDS seq to the window seq
    seq_cds_aln = get_align_objs(dna_align, Seq(seq_cds), Seq(seq_amp))[1]
    # if seq_cds_aln starts with gaps, assume left side of seq_cds is intron/exon boundary
    if seq_cds_aln[0] == '-':
        splice_l = True
    else:
        splice_l = False
    # if seq_cds_aln ends with gaps, assume right side of seq_cds is intron/exon boundary
    if seq_cds_aln[-1] == '-':
        splice_r = True
    else:
        splice_r = False

    ### FIND IMPORTANT INDEX POSITIONS IN REFERENCE SEQUENCES ###

    # find the CDS start/end positions in ref amplicon
    idxs_cds = (seq_amp.find(seq_cds), seq_amp.find(seq_cds) + len(seq_cds) - 1)
    # define the crispresso window size (assumed as 1/2 of the length of seq_crispresso)
    window = int((len(seq_crispresso)/2))
    # find the crispresso window start/end positions (cut site position +/- window size)
    idxs_wdw = (idx_cut - window + 1, idx_cut + window)
    # find the start/end of the CDS seq within the quantification window
    idxs_cds_wdw = (max(idxs_cds[0], idxs_wdw[0]), min(idxs_cds[1], idxs_wdw[1]))
    # define the sequence in the cds/crispresso window overlap
    seq_cds_wdw = seq_amp[idxs_cds_wdw[0]: idxs_cds_wdw[1] + 1]

    ### CHECK FOR ANY CDS FRAME CHANGES AFTER SLICING TO SIZE OF THE WINDOW
    ### if slice to window truncates CDS, re-check for partial left/right codons

    # if left idx of cds_wdw = cds, then wdw_l = cds_l (wdw_l = # of nt until 1st full codon)
    if idxs_cds_wdw[0] == idxs_cds[0]:
        wdw_l = cds_l
    else: # if left idx is different, then find new offset for left window
        if (idxs_cds_wdw[0] - idxs_cds[0] - cds_l) % 3 == 0:
            wdw_l = 0
        else:
            wdw_l = 3 - ((idxs_wdw[0] - idxs_cds[0] - cds_l) % 3)
    # if right idx of cds_wdw = cds, then wdw_r = cds_r (wdw_r = # nt in last codon)
    if idxs_cds_wdw[1] == idxs_cds[1]:
        wdw_r = cds_r
    else: # if right idx is different, then find new offset for right window
        if (len(seq_cds_wdw) - wdw_l) % 3 == 0:
            wdw_r = None # can't slice str on right with 0, so use None
        else:
            wdw_r = ((len(seq_cds_wdw) - wdw_l) % 3) * -1

    ### DEFINE THE REFERENCE PROTEIN/CODON SEQUENCES ###

    # get the reference protein seq for the entire CDS seq and the CDS seq within the window
    # also convert seq_cds and seq_cds_wdw to list of codons
    seq_cds_prot = str(Seq(seq_cds[cds_l: cds_r]).translate())
    seq_wdw_prot = str(Seq(seq_cds_wdw[wdw_l: wdw_r]).translate())
    seq_cds_cdn = re.findall('...', seq_cds[cds_l: cds_r])
    seq_cds_wdw_cdn = re.findall('...', seq_cds_wdw[wdw_l: wdw_r])

    ### ALIGN INDELPHI PREDICTIONS TO REF SEQ AND MAP POSITIONS ###

    # if the sequence direction is reversed, convert edit_seq to reverse complement
    if reverse:
        df_pred['edit_seq'] = df_pred['edit_seq'].apply(lambda x: str(Seq(x).reverse_complement()))
    # align the edit_seq to seq_amp and get # of alignments and post-alignment seqs
    df_pred[['aln_seq','ref_seq']] = df_pred['edit_seq'].apply(lambda x: get_align_objs(dna_align, Seq(x), Seq(seq_amp))).apply(pd.Series)[[1,2]]
    # now map the post-alignment positions back to see how to slice
    # first make list of post-alignment reference positions to find
    # cut site, cds start, cds end, cds_wdw start, cds_wdw end
    list_pos = [idx_cut, idxs_cds[0], idxs_cds[1], idxs_cds_wdw[0], idxs_cds_wdw[1]]
    # find post-alignment positions with map_positions() and convert to dataframe
    df_pos = pd.DataFrame(data=df_pred['ref_seq'].apply(lambda x: map_positions(x, list_pos)).tolist(),
                          columns=['cut_idx','cds_start','cds_end','cds_wdw_start','cds_wdw_end'], index=df_pred.index)
    # get new wdw st/end from cut_idx; can't use idxs_wdw bc enlarges if insertions present
    df_pos['wdw_start'] = df_pos['cut_idx'] - window + 1
    df_pos['wdw_end'] = df_pos['cut_idx'] + window

    ### SLICE ALIGNED/REFERENCE SEQUENCES W/ DF_POS IDXS TO GET SUBSEQUENCES ###

    # post-alignment seqs corresponding to the coding sequence
    df_pred['aln_cds'] = [seq[st:end] for seq,st,end in zip(df_pred['aln_seq'], df_pos['cds_start'], df_pos['cds_end'] + 1)]
    df_pred['ref_cds'] = [seq[st:end] for seq,st,end in zip(df_pred['ref_seq'], df_pos['cds_start'], df_pos['cds_end'] + 1)]
    # post-alignment seqs corresponding to the cds/window overlap (WILL EXPAND IF INSERTIONS PRESENT)
    df_pred['aln_cds_wdw'] = [seq[st:end] for seq,st,end in zip(df_pred['aln_seq'], df_pos['cds_wdw_start'], df_pos['cds_wdw_end'] + 1)]
    df_pred['ref_cds_wdw'] = [seq[st:end] for seq,st,end in zip(df_pred['ref_seq'], df_pos['cds_wdw_start'], df_pos['cds_wdw_end'] + 1)]
    # post-alignment seqs falling within the crispresso window (FIXED LENGTH)
    df_pred['aln_wdw'] = [seq[st:end] for seq,st,end in zip(df_pred['aln_seq'], df_pos['wdw_start'], df_pos['wdw_end'] + 1)]
    df_pred['ref_wdw'] = [seq[st:end] for seq,st,end in zip(df_pred['ref_seq'], df_pos['wdw_start'], df_pos['wdw_end'] + 1)]

    ### CALCULATE INDEL SIZES FOR SUBSEQS AND ASSIGN MUTATION TYPES (E.G. IN-FRAME/FRAMESHIFT/SPLICE) ###

    # calculate the indel size in the post-aln CDS (cds_indel) as # gaps ref_cds - # gaps aln_cds
    df_pred['cds_indel'] = df_pred['ref_cds'].str.count('-') - df_pred['aln_cds'].str.count('-')
    # first deal w/ genotypes where indel == cds_indel (i.e. entire indel is located within CDS)
    df_muts = df_pred.loc[df_pred['indel'] == df_pred['cds_indel']][list_dphicols + ['aln_seq','ref_seq','aln_cds','ref_cds', 'cds_indel']].copy()
    # call mutation as in-frame if indel == cds_indel and indel % 3 == 0; else frameshift
    df_muts['mut_type'] = np.where(df_muts['indel'] % 3 == 0, 'in-frame', 'frameshift')
    # join mut_types to df_pred for these alleles
    df_pred = df_pred.join(df_muts['mut_type'])

    # now deal with alleles where indel != cds_indel (skip if none)
    df_muts2 = df_pred.loc[df_pred['indel'] != df_pred['cds_indel']][list_dphicols + ['aln_seq','ref_seq','aln_cds','ref_cds', 'cds_indel', 'mut_type']].copy()
    if df_muts2.shape[0] > 0:
        # check for gaps at the edges of aln_cds
        df_muts2['cds_gap_l'] = np.where(df_muts2['aln_cds'].str.startswith('-'), True, False)
        df_muts2['cds_gap_r'] = np.where(df_muts2['aln_cds'].str.endswith('-'), True, False)
        # if indel is del larger than cds_indel and edge gap exists, call 'indel exceeds cds'
        df_muts2['mut_type'] = np.where((df_muts2['indel'] < df_muts2['cds_indel']) & (df_muts2[['cds_gap_l', 'cds_gap_r']].any(axis=1)),
                                        'indel exceeds cds', df_muts2['mut_type'])
        # if splice_l/splice_r == True, then find the 2 splice nts at each site
        # if gap in the splice nts, or if mut_type is indel exceeds cds and splice_l/cds_gap_l or splice_r/cds_gap_r are both True, call splice site mut
        if splice_l:
            df_muts2['splice_l'] = [seq[idx-2:idx] for seq,idx in zip(df_pred.loc[df_muts2.index, 'aln_seq'], df_pos.loc[df_muts2.index, 'cds_start'])]
            df_muts2['mut_type'] = np.where((df_muts2['mut_type'] == 'indel exceeds cds') & (df_muts2['cds_gap_l']), 'splice site', df_muts2['mut_type'])
            df_muts2['mut_type'] = np.where(df_muts2['splice_l'].str.contains('-'), 'splice site', df_muts2['mut_type'])
        if splice_r:
            df_muts2['splice_r'] = [seq[idx+1:idx+3] for seq,idx in zip(df_pred.loc[df_muts2.index, 'aln_seq'], df_pos.loc[df_muts2.index, 'cds_end'])]
            df_muts2['mut_type'] = np.where((df_muts2['mut_type'] == 'indel exceeds cds') & (df_muts2['cds_gap_r']), 'splice site', df_muts2['mut_type'])
            df_muts2['mut_type'] = np.where(df_muts2['splice_r'].str.contains('-'), 'splice site', df_muts2['mut_type'])
        # fill missing mut_types in df_raw with df_muts2 values
        df_pred['mut_type'].fillna(value=df_muts2['mut_type'], inplace=True)

    ### CODON ALIGNMENT AND TRANSLATION OF IN-FRAME ALLELES IN DF_MUTS (DF_TLN/DF_CALN) ###

    # isolate in-frame alleles from df_pred (indel == cds_indel)
    df_tln = df_pred.loc[df_pred['mut_type'] == 'in-frame'][['indel','cds_indel','aln_cds','ref_cds','aln_cds_wdw','ref_cds_wdw']].copy()
    df_tln = df_tln.iloc[:, :6].drop_duplicates().copy() # drop duplicates and unnecessary cols
    # trim aln_cds and aln_cds_wdw and remove gaps --> aln_nogap, acw_nogap
    df_tln['cds_nogap'] = df_tln['aln_cds'].str.replace('-', '').str[wdw_l: wdw_r]
    df_tln['acw_nogap'] = df_tln['aln_cds_wdw'].str.replace('-', '').str[wdw_l: wdw_r]
    # check for seqs with < 3 nt >> these cannot be codon aligned
    df_tln['check_cds'] = np.where(df_tln['cds_nogap'].str.len() < 3, 'too_short', 'tln_ok')
    df_tln['check_acw'] = np.where(df_tln['acw_nogap'].str.len() < 3, 'too_short', 'tln_ok')

    # perform codon alignment on full length cds (df_caln) and cds_wdw (df_caln2) seqs
    df_caln = df_tln.loc[df_tln['check_cds'] == 'tln_ok'].copy()
    df_caln[['n_aln','aln_cdn','ref_cdn']] = df_caln['cds_nogap'].apply(lambda x: get_align_objs(cdn_align, re.findall('...', x), seq_cds_cdn, codons=True)).apply(pd.Series)
    df_caln = df_caln.assign(aln_tln=df_caln['aln_cdn'].apply(lambda x: tln(x)), ref_tln=df_caln['ref_cdn'].apply(lambda x: tln(x)))
    df_caln2 = df_tln.loc[df_tln['check_acw'] == 'tln_ok'].copy()
    df_caln2[['n_aln','aln_cdn','ref_cdn']] = df_caln2['acw_nogap'].apply(lambda x: get_align_objs(cdn_align, re.findall('...', x), seq_cds_wdw_cdn, codons=True)).apply(pd.Series)
    df_caln2 = df_caln2.assign(aln_tln=df_caln2['aln_cdn'].apply(lambda x: tln(x)), ref_tln=df_caln2['ref_cdn'].apply(lambda x: tln(x)))
    # join the aln/ref translations to df_tln and rename columns
    df_tln = df_tln.join(df_caln[['aln_tln','ref_tln']]).rename(columns={'aln_tln': 'prot_aln_cds','ref_tln': 'prot_ref_cds'})
    df_tln = df_tln.join(df_caln2[['aln_tln','ref_tln']]).rename(columns={'aln_tln': 'prot_aln_wdw','ref_tln': 'prot_ref_wdw'})
    df_tln.fillna(value='no_tln', inplace=True)

    ### MERGE WITH INDELPHI, CLEAN, AND EXPORT ###

    # join the prot aln/prot ref cols to df_pred
    df_pred = df_pred.join(df_tln[['prot_aln_cds', 'prot_ref_cds', 'prot_aln_wdw', 'prot_ref_wdw']])
    # fill the prot_aln/prot_ref NaN values for non-in-frame muts (e.g. frameshift, splice site)
    df_pred = df_pred.fillna(value={k: df_muts['mut_type'] for k in ['prot_aln_cds', 'prot_ref_cds', 'prot_aln_wdw', 'prot_ref_wdw']})
    list_finalcols = ['aln_wdw', 'ref_wdw', 'indel', 'cds_indel', 'mut_type', 'prot_aln_wdw',
                      'prot_ref_wdw', 'pred_freq', 'edit_seq', 'prot_aln_cds', 'prot_ref_cds']
    df_final = df_pred[list_finalcols].copy()
    # remove alignment tools
    return df_final

#%% internal fx for pca calculations

# fx for allele cdf calculations for gini coefficients
def get_allele_cumsum_v2(df_input, col_pct, out_col='cdf'):
    # make dataframe to hold allele cdfs
    df_cdf = pd.DataFrame(columns=['allele_id'])
    df_cdf.loc[0] = 0 # set 1st row to 0
    # sort by col_pct from high to low freq first and assign allele IDs, calculate cumsum, merge to df_cdf
    df_input = df_input.sort_values([col_pct], ascending=False).reset_index(drop=True)
    df_input['allele_id'] = df_input.index + 1
    df_input[out_col] = df_input[col_pct].cumsum()
    df_cdf = df_cdf.merge(df_input[['allele_id', out_col]], on='allele_id', how='outer')
    # set the 0 allele to 0%, fill nan with 100%
    df_cdf.loc[0] = 0
    df_cdf['allele_id'] = df_cdf['allele_id'].astype(int)
    return df_cdf