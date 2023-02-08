# -*- coding: utf-8 -*-
"""
@author: kevin

Functions for screen analysis
"""
#%% import packages

from collections import Counter
from pathlib import Path
import time
import warnings

import numpy as np
import pandas as pd

from Bio import SeqIO

#%% count_reads() - count sgRNA reads in FASTQ (adapted from count_spacers.py)

def count_reads(in_fastq, in_ref, KEY_INTERVAL=(10,80), DIR='FWD',
                KEY='CGAAACACCG', KEY_REV='GTTTTAGA', out_counts='counts.csv',
                out_np='np_counts.csv', out_stats='stats.txt'):
    """
    Count the reads in a FASTQ file and assign them to a reference sgRNA set.

    Given a set of sgRNA sequences and a FASTQ file, count the reads in the
    FASTQ, assign the reads to sgRNAs, and export the counts to a csv file. All
    sgRNA sequences not found in the reference file (non-perfect matches) are
    written to a separate csv file (npcounts).

    Parameters
    ----------
    in_fastq : str or path
        String or path to the FASTQ file to be processed.
    in_ref : str or path
        String or path to the reference file. in_ref must have column headers,
        with 'sgRNA_seq' as the header for the column with the sgRNA sequences.
    KEY_INTERVAL : tuple, default (10,80)
        Tuple of (KEY_START, KEY_END) that defines the KEY_REGION. Denotes the
        substring within the read to search for the KEY.
    KEY : str, default 'CGAAACACCG'
        Upstream sequence that identifies the position of the sgRNA when used
        with DIR='FWD'. The default is the end of the hU6 promoter.
    KEY_REV : str, default 'GTTTTAGA'
        Downstream sequence that identifies the position of the sgRNA when used
        with DIR='REV'. The default is the start of the sgRNA scaffold sequence.
    DIR : {'FWD', 'REV'}, default 'FWD'
        The direction to identify the position of the sgRNA relative to the
        KEY sequence. 'FWD' is upstream of sgRNA, 'REV' is downstream of sgRNA.
    out_counts : str or path, default 'counts.csv'
        String or path for the output csv file with perfect sgRNA matches.
    out_np : str or path, default 'np_counts.csv'
        String or path for the output csv file with non-perfect sgRNA matches.
    out_stats : str or path, default 'stats.txt'
        String or path for the output txt file with the read counting statistics.
    """

    # STEP 1A: OPEN INPUT FILES FOR PROCESSING, CHECK FOR REQUIRED FORMATTING
    # look for 'sgRNA_seq' column, raise Exception if missing
    df_ref = pd.read_csv(in_ref, header=0) # explicit header = first row
    if 'sgRNA_seq' not in df_ref.columns.tolist():
        raise Exception('in_ref is missing column: sgRNA_seq')
    # look for other cols, raise Warning if suggested cols are missing
    list_headcols = ['sgRNA_ID', 'sgRNA_seq', 'Gene', 'cut_site_AA', 'Domain']
    if not all(col in df_ref.columns.tolist() for col in list_headcols):
        list_miss = [col for col in list_headcols if col not in df_ref.columns.tolist()]
        warnings.warn('Warning! in_ref is missing column(s) for downstream functions: ' + str(list_miss))
    # try opening input FASTQ, raise Exception if not possible
    try:
        handle = open(in_fastq)
    except:
        print('Error! Could not open the FASTQ file: %s' % in_fastq)
        return

    # STEP 1B: SET UP VARIABLES FOR SCRIPT
    # make dictionary to hold sgRNA counts - sgRNA_seq, count as k,v
    dict_perfects = {sgRNA:0 for sgRNA in df_ref['sgRNA_seq']}
    list_np = [] # placeholder list for non-perfect matches
    num_reads = 0 # total number of reads processed
    num_perfect_matches = 0 # count of reads with a perfect match to library
    num_np_matches = 0 # count of reads without a perfect match to library
    num_nokey = 0 # count of reads where key was not found
    KEY_START, KEY_END = KEY_INTERVAL[0], KEY_INTERVAL[1] # set the key interval

    # STEP 2: PROCESS FASTQ FILE READS AND ADD COUNTS TO DICT
    readiter = SeqIO.parse(handle, 'fastq') # process reads in fastq file
    # find sgRNA using FORWARD direction (default)
    if DIR == 'FWD':
        for record in readiter: # contains the seq and Qscore etc.
            num_reads += 1
            read_sequence = str.upper(str(record.seq))
            key_region = read_sequence[KEY_START:KEY_END]
            key_index = key_region.find(KEY)
            if key_index >= 0: # if key found
                start_index = key_index + KEY_START + len(KEY)
                guide = read_sequence[start_index:(start_index + 20)]
                if guide in dict_perfects:
                    dict_perfects[guide] += 1
                    num_perfect_matches += 1
                else:
                    num_np_matches += 1
                    list_np.append(guide)
            else:
                num_nokey += 1
    # find sgRNA using REVERSE direction
    elif DIR == 'REV':
        for record in readiter: # contains the seq and Qscore etc.
            num_reads += 1
            read_sequence = str.upper(str(record.seq))
            key_region = read_sequence[KEY_START:KEY_END]
            key_index = key_region.find(KEY_REV)
            if key_index >= 0: # if key found
                start_index = key_index + KEY_START
                guide = read_sequence[(start_index - 20):(start_index)]
                if guide in dict_perfects:
                    dict_perfects[guide] += 1
                    num_perfect_matches += 1
                else:
                    num_np_matches += 1
                    list_np.append(guide)
            else:
                num_nokey += 1
    else:
        raise Exception('ERROR! Specified direction is not valid')
    handle.close()

    # STEP 3: SORT DICTIONARIES AND GENERATE OUTPUT FILES
    # sort perf matches (A-Z) with guides,counts as k,v and output to csv
    df_perfects = pd.DataFrame(data=dict_perfects.items(), columns=['sgRNA_seq', 'reads'])
    df_perfects.sort_values(by='sgRNA_seq', inplace=True)
    df_perfects.to_csv(out_counts, index=False, header=False)
    # now sort non-perfect matches by frequency and output to csv
    dict_np = Counter(list_np) # use Counter to tally up np matches
    df_npmatches = pd.DataFrame(data=dict_np.items(), columns=['sgRNA_seq', 'reads'])
    df_npmatches.sort_values(by='reads', ascending=False, inplace=True)
    df_npmatches.to_csv(out_np, index=False)

    # STEP 4: CALCULATE STATS AND GENERATE STAT OUTPUT FILE
    # percentage of guides that matched perfectly
    pct_perfmatch = round(num_perfect_matches/float(num_perfect_matches + num_np_matches) * 100, 1)
    # percentage of undetected guides (no read counts)
    guides_with_reads = np.count_nonzero(list(dict_perfects.values()))
    guides_no_reads = len(dict_perfects) - guides_with_reads
    pct_no_reads = round(guides_no_reads/float(len(dict_perfects.values())) * 100, 1)
    # skew ratio of top 10% to bottom 10% of guide counts
    top_10 = np.percentile(list(dict_perfects.values()), 90)
    bottom_10 = np.percentile(list(dict_perfects.values()), 10)
    if top_10 != 0 and bottom_10 != 0:
        skew_ratio = top_10/bottom_10
    else:
        skew_ratio = 'Not enough perfect matches to determine skew ratio'
    # calculate the read coverage (reads processed / sgRNAs in library)
    num_guides = df_ref['sgRNA_seq'].shape[0]
    coverage = round(num_reads / num_guides, 1)
    # calculate the number of unmapped reads (num_nokey / total_reads)
    pct_unmapped = round((num_nokey / num_reads) * 100, 2)

    # write analysis statistics to statfile
    with open(out_stats, 'w') as statfile:
        statfile.write('Number of reads processed: ' + str(num_reads) + '\n')
        statfile.write('Number of reads where key was not found: ' + str(num_nokey) + '\n')
        statfile.write('Number of perfect guide matches: ' + str(num_perfect_matches) + '\n')
        statfile.write('Number of nonperfect guide matches: ' + str(num_np_matches) + '\n')
        statfile.write('Number of undetected guides: ' + str(guides_no_reads) + '\n')
        statfile.write('Percentage of unmapped reads (key not found): ' + str(pct_unmapped) + '\n')
        statfile.write('Percentage of guides that matched perfectly: ' + str(pct_perfmatch) + '\n')
        statfile.write('Percentage of undetected guides: ' + str(pct_no_reads) + '\n')
        statfile.write('Skew ratio of top 10% to bottom 10%: ' + str(skew_ratio) + '\n')
        statfile.write('Read coverage: ' + str(coverage))
        statfile.close()

    print(str(in_fastq) + ' processed')
    return

#%% batch_count() - batch process FASTQ files with input csv file

def batch_count(in_batch, in_ref, dir_fastq='', dir_counts='', dir_np='',
                dir_stats='', **kwargs):
    """
    Perform count_reads() on a batch scale given a sample sheet.

    Batch process FASTQ files with a csv sample sheet (in_batch) and a reference
    file (in_ref). in_batch must include headers for sample ids ('sample_id'),
    FASTQ file names ('fastq_file'), and treatment conditions ('condition').

    Parameters
    ----------
    in_batch : str or path
        String or path to the batch sample sheet csv file. Must have column
        headers for sample ids ('sample_id'), FASTQ file names ('fastq_file'),
        and treatment conditions (e.g. drug, DMSO; 'condition').
    in_ref : str or path
        String or path to the reference file. in_ref must have column headers,
        with 'sgRNA_seq' as the header for the column with the sgRNA sequences.
    dir_fastq : str, default ''
        The subfolder containing the FASTQ files. The default is the current
        working directory.
    dir_counts : str, default ''
        The subfolder to export the sgRNA read count csv files. The default is
        the current working directory.
    dir_np : str, default ''
        The subfolder to export the non-perfect match csv files. The default is
        the current working directory.
    dir_stats : str, default ''
        The subfolder to export the read count statistics files. The default is
        the current working directory.
    **kwargs : key, value mappings in format x=y
        All other keyword arguments are passed to count_reads(). See the
        count_reads() documentation for more information. Other kwargs include:
        KEY_INTERVAL, DIR, KEY, KEY_REV.
    """

    batch_st = time.perf_counter()
    # define all the directory paths
    path = Path.cwd()
    list_dirs = [path / subdir for subdir in [dir_fastq, dir_counts, dir_np, dir_stats]]
    for subdir in list_dirs:
        Path.mkdir(subdir, exist_ok=True)

    # import batch csv and process samples with count_reads()
    df_batch = pd.read_csv(in_batch)
    list_reqcols = ['sample_id', 'fastq_file', 'condition']
    list_batchcols = df_batch.columns.tolist()
    if not all(col in list_batchcols for col in list_reqcols):
        list_miss = [col for col in list_reqcols if col not in list_batchcols]
        raise Exception('Error! in_batch is missing column(s): ' + str(list_miss))

    # perform batch processing
    for row in df_batch.itertuples():
        t_start = time.perf_counter()
        fastq = list_dirs[0] / row.fastq_file
        counts = list_dirs[1] / (row.sample_id + '_counts.csv')
        np = list_dirs[2] / (row.sample_id + '_npcounts.csv')
        stats = list_dirs[3] / (row.sample_id + '_stats.txt')
        count_reads(in_fastq=fastq, in_ref=in_ref, out_counts=counts,
                    out_np=np, out_stats=stats, **kwargs)
        t_end = time.perf_counter()
        print(row.sample_id + ' processed in %.2f sec' % (t_end - t_start))

    batch_end = time.perf_counter()
    print('Batch count completed in %.2f min' % ((batch_end - batch_st) / 60))
    return

#%% batch_process() - batch process count_reads output (merging, log2/t0 norm)

def batch_process(in_batch, in_ref, merge_stats=True, dir_counts='', dir_stats='',
                  in_counts=None, in_stats=None, save='all', out_folder='',
                  out_prefix='', return_df=None):
    """
    Batch merging and pre-processing (log2/t0) of samples given a sample sheet.

    Batch processing of read counts from count_reads using a csv sample sheet
    (in_batch) and a reference file (in_ref). Aggregates raw reads, performs
    log2 and t0 normalization, averages reps per condition for t0 normalized
    values, and exports to csv. Also merges and exports the stat files as csv.

    Parameters
    ----------
    in_batch : str or path
        String or path to the batch sample sheet csv file. Must have column
        headers for sample ids ('sample_id'), FASTQ file names ('fastq_file'),
        and treatment conditions (e.g. drug, DMSO; 'condition').
    in_ref : str or path
        String or path to the reference file. in_ref must have column headers,
        with 'sgRNA_seq' as the header for the column with the sgRNA sequences.
    merge_stats : bool, default True
        Whether to merge the read counts statistics files.
    dir_counts : str, default ''
        The subfolder containing the read counts csv files. The default is
        the current working directory.
    dir_stats : str, default ''
        The subfolder containing the read counts stats files. The default is
        the current working directory.
    in_counts : list of tup in format ('sample_id', 'file name'), default None
        List of tuples of (sample_id, file name) for the samples
        in in_batch. The default is None, which assumes the default naming
        scheme from batch_count ('sample_id' + '_counts.csv' = 'KN-0_counts.csv').
        If your read counts files do not follow the default naming scheme,
        then use in_counts to map sample_ids to your read count file names.
    in_stats : list of tup in format ('sample_id', 'file name'), default None
        List of tuples of (sample_id, file name) for the samples
        in in_batch. The default is None, which assumes the default naming
        scheme from batch_count ('sample_id' + '_stats.txt' = 'KN-0_stats.txt').
        If your stats files do not follow the default naming scheme, then use
        in_stats to map sample_ids to your read count stats file names.
    save : {'all', None, ['reads', 'log2', 't0', 'conds', 'stats']}, default 'all'
        Choose files for export to csv. The default is 'all', which is the
        aggregated read counts ('reads'), log2 normalized values ('log2'), and
        t0 normalized values ('t0'). You may also enter any combination of
        'reads', 'log2', 't0') as a list of strings to choose which ones to
        save ('all' is equivalent to a list of all three). None will not export
        any files to csv.
    out_folder : str, default ''
        Name of the subfolder to save output files. The default is the current
        working directory.
    out_prefix : str, default ''
        File prefix for the output csv files. The prefix should contain an
        underscore.
    return_df : {None, 'reads', 'log2', 't0', 'conds', 'stats'}, default None
        Whether to return a dataframe at function end. The default is None,
        which returns nothing. However, you can return the reads, log2 norm,
        t0 norm, averaged reps by condition, or stats dataframes by calling
        'reads', 'log2', 't0', 'conds', or 'stats', respectively.
    """

    # import ref files and define variables/paths
    path = Path.cwd()
    df_ref = pd.read_csv(in_ref)
    if 'sgRNA_seq' not in df_ref.columns.tolist():
        raise Exception('in_ref is missing column: sgRNA_seq')
    df_batch = pd.read_csv(in_batch)
    list_reqcols = ['sample_id', 'fastq_file', 'condition']
    list_batchcols = df_batch.columns.tolist()
    if not all(col in list_batchcols for col in list_reqcols):
        list_miss = [col for col in list_reqcols if col not in list_batchcols]
        raise Exception('Error! in_batch is missing column(s): ' + str(list_miss))
    if 't0' not in df_batch['condition'].tolist():
        raise Exception('t0 condition not found in the in_batch file')
    # defaults to cwd if subdir == ''
    counts_path = path / dir_counts
    stats_path = path / dir_stats
    if in_counts is None:
        df_batch['counts_files'] = df_batch['sample_id'] + '_counts.csv'
    else:
        df_temp = pd.DataFrame(in_counts, columns=['sample_id', 'counts_files'])
        df_batch = df_batch.merge(df_temp, on='sample_id', how='left')
    if in_stats is None:
        df_batch['stats_files'] = df_batch['sample_id'] + '_stats.txt'
    else:
        df_temp = pd.DataFrame(in_stats, columns=['sample_id', 'stats_files'])
        df_batch = df_batch.merge(df_temp, on='sample_id', how='left')

    # import csv files and generate dfs for raw reads and log2 norm
    df_reads, df_log2 = df_ref.copy(), df_ref.copy()
    for row in df_batch.itertuples():
        file = counts_path / row.counts_files
        df_temp = pd.read_csv(file, names=['sgRNA_seq', row.sample_id])
        # merge on sgRNA_seq to aggregate columns
        df_reads = pd.merge(df_reads, df_temp, on='sgRNA_seq')
        # perform log2 normalization (brian/broad method)
        total_reads = df_reads[row.sample_id].sum()
        df_log2[row.sample_id] = df_reads[row.sample_id].apply(lambda x: np.log2((x * 1000000 / total_reads) + 1))

    # perform t0 normalization
    df_t0 = df_ref.copy()
    t0 = df_batch.loc[df_batch['condition'] == 't0']['sample_id']
    if t0.shape[0] != 1:
        raise Exception('Only a single t0 sample is allowed')
    t0 = t0[0]
    for row in df_batch.itertuples():
        df_t0[row.sample_id] = df_log2[row.sample_id].sub(df_log2[t0])
    df_t0.drop(columns=t0, inplace=True) # drop the t0 col

    # average replicates by condition
    list_conds = df_batch['condition'].unique().tolist()
    list_conds.remove('t0')
    df_conds = df_ref.copy()
    for cond in list_conds:
        reps = df_batch.loc[df_batch['condition'] == cond]['sample_id'].tolist()
        if len(reps) > 1:
            df_conds[cond] = df_t0[reps].mean(axis=1)
        elif len(reps) == 1:
            df_conds[cond] = df_t0[reps]
        else:
            raise Exception('Error! Invalid number of replicates')

    # merge statistics files
    if merge_stats:
        df_stats = pd.DataFrame(columns=['parameters'])
        for row in df_batch.itertuples():
            file = stats_path / row.stats_files
            df_temp = pd.read_csv(file, sep=': ', engine='python', names=['parameters', row.sample_id])
            df_stats = pd.merge(df_stats, df_temp, on='parameters', how='outer')

    # export files and return dataframes if necessary
    outpath = path / out_folder
    Path.mkdir(outpath, exist_ok=True)
    # dictionary to map kws to dfs and output file names
    dict_df = {'reads': (df_reads, out_prefix + 'reads.csv'),
               'log2': (df_log2, out_prefix + 'log2.csv'),
               't0': (df_t0, out_prefix + 't0_reps.csv'),
               'conds': (df_conds, out_prefix + 't0_conds.csv')}
    if merge_stats:
        dict_df.update({'stats': (df_stats, out_prefix + 'stats.csv')})
    # determine which files to export
    if save == 'all':
        save = ['reads','log2','t0', 'conds', 'stats']
    if isinstance(save, list):
        for key in save:
            dict_df[key][0].to_csv(outpath / dict_df[key][1], index=False)
    elif save is None:
        pass
    else:
        warnings.warn('Invalid value for save. No files exported')
    # determine df to return
    print('Batch processing completed')
    if return_df in dict_df.keys():
        return dict_df[return_df][0]
    elif return_df is None:
        return
    else:
        print('Invalid value for return_df. No dataframe returned')
        return