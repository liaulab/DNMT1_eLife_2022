# -*- coding: utf-8 -*-
"""
@author: kevin

Process FASTQ files and calculate normalized sgRNA counts for replicates and
conditions, which are used to calculate downstream sgRNA scores.


"""
#%% IMPORT PACKAGES

import csv, itertools, os, re, sys, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd

from DNMT1_Figure1_screen_analysis_functions import (batch_count, batch_process)

#%% pre-processing the DAC CRISPR scanning data (FASTQ files) to get t0-normalized replicates

path = os.getcwd() # path of the current working directory
in_ref = 'DNMT1_UHRF1_Input_Reference.csv' # library reference info
in_batch = 'DNMT1_UHRF1_Input_Batch.csv' # batch processing file

df_ref = pd.read_csv(in_ref)
df_batch = pd.read_csv(in_batch)

# COUNT READS AND PROCESS FASTQ FILES
batch_count(in_batch=in_batch, in_ref=in_ref,
            dir_fastq='FASTQ_files',
            dir_counts='DNMT1_screen_processing/counts',
            dir_np='DNMT1_screen_processing/npcounts',
            dir_stats='DNMT1_screen_processing/stats')

# PREPROCESSING (MERGING FILES, LOG2 NORM, T0 NORM, REP)
batch_process(in_batch=in_batch, in_ref=in_ref,
              out_prefix='DNMT1_screen_', out_folder='DNMT1_screen_processing',
              dir_counts='DNMT1_screen_processing/counts',
              dir_stats='DNMT1_screen_processing/stats')


#%% calculating 'resistance scores'
### drug treatment conditions are normalized to DMSO and neg ctrls --> resistance score
### also includes a neg. ctrl-normalization == ('norm')
### annotate gRNAs as enriched (if >mean+2sd of neg ctrls) or not enriched
### DAC == decitabine; GSKi == GSK3484862

# defining common variables for downstream processing (e.g. list of col headers)
list_refcols = ['sgRNA_ID', 'sgRNA_seq', 'Gene', 'cut_site_NT', 'cut_site_AA', 'Domain']
list_condcols = ['DAC_56d', 'DMSO_DAC_56d', 'GSKi_42d', 'DMSO_GSKi_42d']


### import rep-avg, t0-norm LFCs ('t0_conds') and clean up
in_lfc = pd.read_csv('DNMT1_screen_processing/DNMT1_screen_data_t0_conds.csv')
# re-sort sgRNAs by gene and cut_site_AA
in_lfc['sort'] = np.where(in_lfc['Gene'].str.contains('CTRL'), 'CTRL', in_lfc['Gene'])
in_lfc = in_lfc.sort_values(by=['sort','cut_site_AA','sgRNA_ID']).reset_index(drop=True).drop(columns='sort')


### normalize drug treatment LFCs to DMSO LFCs ("resistance score" aka "_res")
for drug, vehicle in [('DAC_56d', 'DMSO_DAC_56d'), ('GSKi_42d', 'DMSO_GSKi_42d')]:
    in_lfc[drug + '_res'] = in_lfc[drug] - in_lfc[vehicle]

### calculate stats for the neg ctrl gRNAs (to normalize LFCs/identify enriched gRNAs)
df_ctrl = in_lfc.loc[in_lfc['Gene'] == 'NEG_CTRL'][in_lfc.columns[6:]].describe().transpose().reset_index().rename(columns={'index':'cond'})
df_ctrl['mean2sd'] = df_ctrl['mean'] + (df_ctrl['std'] * 2)

### normalize LFCs to mean of negative control gRNAs (aka "_norm")
for x in in_lfc.columns[6:]:
    in_lfc[x + '_norm'] = in_lfc[x] - df_ctrl.loc[df_ctrl['cond'] == x]['mean'].values[0]

### annotate enriched gRNAs (mean+2sd) for each cond
# because already subtracted neg ctrl mean, look for sgRNAs > 2sd of neg ctrls
in_lfc['enrich_DAC'] = np.where(in_lfc['DAC_56d_res_norm'] >= df_ctrl.loc[df_ctrl['cond'] == 'DAC_56d_res', 'std'].values[0] * 2, 'enrich', 'not enrich')
in_lfc['enrich_GSKi'] = np.where(in_lfc['GSKi_42d_res_norm'] >= df_ctrl.loc[df_ctrl['cond'] == 'GSKi_42d_res', 'std'].values[0] * 2, 'enrich', 'not enrich')

# EXPORT in_lfc as 'master' (contains all scores (non-norm, DMSO-norm, ctrl-norm, etc.)
# in_lfc.to_csv('./DNMT1_screen_processing/DNMT1_screen_master_v1.csv', index=False)

### reduce cols to ctrl-norm data and rename to and also separate by gene --> df_lfc/df_dnmt/df_uhrf
df_lfc = in_lfc[list_refcols + ['DAC_56d_res_norm','GSKi_42d_res_norm','enrich_DAC','enrich_GSKi']].copy()
df_lfc = df_lfc.rename(columns={'DAC_56d_res_norm':'DAC_res', 'GSKi_42d_res_norm':'GSKi_res'})
# define the gene-specific dfs --> df_dnmt/df_uhrf
df_dnmt = df_lfc[df_lfc['Gene'] == 'DNMT1'].copy().reset_index(drop=True)
df_uhrf = df_lfc[df_lfc['Gene'] == 'UHRF1'].copy().reset_index(drop=True)
# EXPORT the reduced cols df_lfc as 'reduced' and also the gene-specific dfs
# df_lfc.to_csv('./DNMT1_screen_processing/DNMT1_screen_reduced_v1.csv', index=False)
# df_dnmt.to_csv('./DNMT1_screen_processing/DNMT1_screen_reduced_dnmt1_v1.csv', index=False)
# df_uhrf.to_csv('./DNMT1_screen_processing/DNMT1_screen_reduced_uhrf1_v1.csv', index=False)
