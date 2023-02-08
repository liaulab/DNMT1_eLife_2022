# -*- coding: utf-8 -*-
"""
@author: kevin

Code for the data analysis and processing in Figure 3
"""
#%% import packages

import numpy as np
import pandas as pd

import scipy.stats as stats
import scipy.spatial.distance as dist
import scipy.interpolate as interp
import scipy.optimize as sp_opt
import scipy.cluster as sp_cl
import scipy.special as sp

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import QuantileTransformer

from Bio.Seq import Seq

from DNMT1_Figures3_5_analysis_functions import (batch_analyze, batch_compare, agg_comps)
from DNMT1_Figures3_5_analysis_functions import (indelphi_process, get_allele_cumsum_v2)

#%% pre-processing the CRISPResso2 output data (run once)

# defining variables
in_ref = 'DNMT1_sgRNA_genotyping_reference.csv'
in_batch = 'DNMT1_sgRNA_genotyping_batchfile.batch'

df_ref = pd.read_csv(in_ref)
df_batch = pd.read_csv(in_batch, sep='\t')

list_refcols = ['sample','cond','label','sgRNA_ID','sgRNA_seq','Gene','Domain','cut_site_AA','DAC_res','cluster_id']
list_samples = [x for x in df_ref['sample_name']]
list_comparisons = [x for x in df_ref['oligo_id'].unique()]
list_comptups = [(x + '_DAC', x + '_DMSO') for x in list_comparisons]
list_compdirs = ['CRISPRessoCompare_on_' + x for x in list_comparisons]
list_outfiles = [x + '_comparisons.csv' for x in list_comparisons]

### run the batch allele analysis
df_stats = batch_analyze(list_samples, in_ref, in_batch,
                         out_folder='DNMT1_sgRNA_genotyping_analysis/translated_alleles',
                         out_stats='DNMT1_sgRNA_genotyping_batch_analysis_stats.csv')

### run the batch allele comparison
out_comps = batch_compare(list_comptups, in_ref, in_batch,
                          list_crispresso_dirs=list_compdirs, list_outfiles=list_outfiles,
                          dir_tlns='DNMT1_sgRNA_genotyping_analysis/translated_alleles',
                          out_folder='DNMT1_sgRNA_genotyping_analysis/comparisons',
                          out_stats='DNMT1_sgRNA_genotyping_batch_comparison_stats.csv',
                          out_mismatches='DNMT1_sgRNA_genotyping_batch_comparison_mismatches.csv')

# merge alleles w/ <=2 point muts (e.g., PCR/seq errors, SNPs), WT alleles to collapse size
# isolate all alleles w/ freq >=0.1% and recalculate absolute/adjusted (excl. WT) pct_reads
dict_sgrnas = {} # dict to hold aggregated/filtered genotype dfs for each sgRNA
for x in list_comparisons:
    # merging and collapsing alleles
    in_comp = pd.read_csv('./DNMT1_sgRNA_genotyping_analysis/comparisons/%s_comparisons.csv' % x, index_col=0)
    df_agg = agg_comps(infile=in_comp, clean_anno=True, keep_col_names=False)
    temp_groupby = df_agg.groupby(by='mut_type').agg({k:'sum' for k in ['n_reads_s1','n_reads_s2','pct_reads_s1','pct_reads_s2','n_alleles']})
    wt_row = df_agg[df_agg['mut_type'] == 'wild-type'].copy()
    if wt_row['pct_reads_s1'].idxmax() == wt_row['pct_reads_s2'].idxmax():
        wt_row = wt_row.loc[wt_row['pct_reads_s1'].idxmax()].copy()
    elif wt_row['pct_reads_s1'].idxmax() != wt_row['pct_reads_s2'].idxmax():
        raise Exception('WT allele freq differs for %s' % x)
    wt_row[['n_reads_s1','n_reads_s2','pct_reads_s1','pct_reads_s2','n_alleles']] = temp_groupby.loc['wild-type', ['n_reads_s1','n_reads_s2','pct_reads_s1','pct_reads_s2','n_alleles']]
    wt_row['LFC'] = np.log2((wt_row['pct_reads_s1'] + 0.1) / (wt_row['pct_reads_s2'] + 0.1))
    df_agg2 = df_agg[df_agg['mut_type'] != 'wild-type'].copy()
    df_agg2 = df_agg2.append(wt_row, ignore_index=True).sort_values(by='pct_reads_s1', ascending=False).reset_index(drop=True)
    # export the intermediate aggregated df
    df_agg2.to_csv('./DNMT1_sgRNA_genotyping_analysis/aggregated_comparisons/%s_aggregated.csv' % x, index=False)
    # now isolate >=0.1% alleles --> df_filt
    df_filt = df_agg2[(df_agg2['pct_reads_s1'] >= 0.1) | (df_agg2['pct_reads_s2'] >= 0.1)].copy()
    df_filt = df_filt.reset_index().rename(columns={'index':'old_idx'})
    # rank alleles by abs_freq (including WT) and adj_freq (excl. WT)
    df_filt['abs_rk1'] = df_filt['pct_reads_s1'].rank(ascending=False, method='min')
    df_filt['abs_rk2'] = df_filt['pct_reads_s2'].rank(ascending=False, method='min')
    df_filt['adj_rk1'] = df_filt['adj_freq_s1'].rank(ascending=False, method='min')
    df_filt['adj_rk2'] = df_filt['adj_freq_s2'].rank(ascending=False, method='min')
    # re-organize cols
    df_filt = df_filt.reindex(columns=['aln_seq', 'ref_seq', 'n_del', 'n_ins', 'n_mut', 'indel', 'mut_type', 'mut_type2', 'prot_aln', 'prot_ref', 'LFC',
                                       'pct_reads_s1', 'pct_reads_s2', 'abs_rk1', 'abs_rk2', 'adj_freq_s1', 'adj_freq_s2', 'adj_rk1', 'adj_rk2', 'old_idx'])
    # recalculate the abs/adj pct_reads for the >0.1% alleles
    df_filt[['pct_reads_s1','pct_reads_s2']] = df_filt[['pct_reads_s1','pct_reads_s2']].apply(lambda x: x / x.sum() * 100, axis=0)
    df_filt_nowt = df_filt[df_filt['mut_type'] != 'wild-type'].copy()
    df_filt_nowt[['adj_freq_s1','adj_freq_s2']] = df_filt_nowt[['pct_reads_s1','pct_reads_s2']].apply(lambda x: x / x.sum() * 100, axis=0)
    df_filt[['adj_freq_s1','adj_freq_s2']] = df_filt_nowt[['adj_freq_s1','adj_freq_s2']]
    df_filt['LFC'] = np.log2((df_filt['pct_reads_s1'] + 0.1) / (df_filt['pct_reads_s2'] + 0.1))
    dict_sgrnas[x] = df_filt.copy() # add df to dict_sgrnas
    # export the filtered alleles w/ recalculated frequencies
    df_filt.to_csv('./DNMT1_sgRNA_genotyping_analysis/filtered_min_0.1_pct/%s_agg_filtered_min0.1.csv' % x, index=False)

#%% import/process raw inDelphi data for the tested sgRNAs (1st time only, import afterwards)

# clean up the DNMT1/UHRF1 sgRNA inDelphi gene mode output summary and add information for processing
# run once; import cleaned file afterwards
dict_dphi = {'Gene symbol':'Gene','Exon number':'Exon','gRNA':'sgRNA_seq','Local context':'indelphi_seq','gRNA strand w.r.t. exon strand':'sgRNA_dir', "Dist. to 5' end":'dist_5p',
             "Dist. to 3' end":'dist_3p','Precision':'Precision','Frameshift (%)':'pct_fs','Frame +0 (%)':'pct_if','Frame +1 (%)':'pct_+1','Frame +2 (%)':'pct_+2','Exp. indel len':'avg_indel'}
df_dphi = pd.read_csv('./inDelphi_outputs/inDelphi_gene_summary_DNMT1.csv')[dict_dphi.keys()].rename(columns=dict_dphi)
df_dphi = df_dphi.merge(pd.read_csv('./inDelphi_outputs/inDelphi_gene_summary_UHRF1.csv')[dict_dphi.keys()].rename(columns=dict_dphi), how='outer')
df_dphi = df_dphi[df_dphi['sgRNA_seq'].isin(list_comparisons)].copy()
df_dphi = df_dphi.merge(df_ref[['oligo_id','label','sgRNA_ID','sgRNA_seq','Gene','Domain','cut_site_AA','DAC_res','cluster_id', 'CDS_frame']].drop_duplicates().rename(columns={'oligo_id':'sample'}), how='left', on='sgRNA_seq')
df_dphi = df_dphi.reindex(columns=['sample','sgRNA_ID','sgRNA_seq','sgRNA_dir','indelphi_seq','Gene','Exon','Domain','cut_site_AA','CDS_frame', 'cluster_id','label',
                                   'dist_5p','dist_3p','Precision','pct_fs','pct_if','pct_+1','pct_+2','avg_indel'])
df_dphi = df_dphi.sort_values(by=['Gene','cut_site_AA']).reset_index(drop=True)
df_dphi = df_dphi.merge(df_batch[['guide_seq','amplicon_seq','coding_seq']].copy().rename(columns={'guide_seq':'sgRNA_seq'}).drop_duplicates(), how='left', on='sgRNA_seq')

### adding information for processing

# dphi_wdw_seq == +/- 60 nt around cut_site (IN SAME DIRECTION AS sgRNA_dir)
# dphi_cds_seq == CDS seq within the dphi_wdw_seq range (IN SAME DIRECTION as dphi_wdw_seq/sgRNA_dir)
# crispresso_wdw_seq = +/- 30 nt around cut_site
# CDS_frame_dphi == # of nt until first full codon IN SAME DIRECTION AS dphi_wdw_seq
# that means if dphi_wdw_seq == reverse wrt CDS_seq, refers to 3' exon nts
df_dphi[['dphi_wdw_seq','dphi_cds_seq','crispresso_wdw_seq','CDS_frame_dphi']] = np.nan
for x_num,x in df_dphi.iterrows():
    if x['sgRNA_dir'] == '-':
        x_rev = True
        x_seq = str(Seq(x['sgRNA_seq']).reverse_complement())
        if x_seq not in x['amplicon_seq']:
            raise Exception('gRNA sequence not found')
        idx_cut = x['amplicon_seq'].find(x_seq) + 3
    else:
        x_rev = False
        x_seq = x['sgRNA_seq']
        if x_seq not in x['amplicon_seq']:
            raise Exception('gRNA sequence not found')
        idx_cut = x['amplicon_seq'].find(x_seq) + 17
    df_dphi.loc[df_dphi['sample'] == x['sample'], 'crispresso_wdw_seq'] = x['amplicon_seq'][idx_cut-30:idx_cut+30]
    if not x_rev:
        dphi_wdw_seq = x['amplicon_seq'][idx_cut-60:idx_cut+60]
        idxs_cds = (x['amplicon_seq'].find(x['coding_seq']), x['amplicon_seq'].find(x['coding_seq']) + len(x['coding_seq']))
        idxs_dphi = (x['amplicon_seq'].find(dphi_wdw_seq), x['amplicon_seq'].find(dphi_wdw_seq) + len(dphi_wdw_seq))
        idxs_dphi_wdw_cds = (max(idxs_cds[0], idxs_dphi[0]), min(idxs_cds[1], idxs_dphi[1]))
        dphi_wdw_cds_seq = x['amplicon_seq'][idxs_dphi_wdw_cds[0]:idxs_dphi_wdw_cds[1]]
        cds_l = int(x['CDS_frame'])
        if idxs_dphi_wdw_cds[0] > idxs_cds[0]:
            cds_l = cds_l - ((idxs_dphi_wdw_cds[0] - idxs_cds[0]) % 3)
            if cds_l < 0:
                cds_l = cds_l + 3
        df_dphi.loc[df_dphi['sample'] == x['sample'], 'dphi_wdw_seq'] = x['amplicon_seq'][idx_cut-60:idx_cut+60]
        df_dphi.loc[df_dphi['sample'] == x['sample'], 'dphi_cds_seq'] = dphi_wdw_cds_seq
        df_dphi.loc[df_dphi['sample'] == x['sample'], 'CDS_frame_dphi'] = cds_l
    elif x_rev:
        dphi_wdw_seq = str(Seq(x['amplicon_seq'][idx_cut-60:idx_cut+60]).reverse_complement())
        idxs_cds = (x['amplicon_seq'].find(x['coding_seq']), x['amplicon_seq'].find(x['coding_seq']) + len(x['coding_seq']))
        idxs_dphi = (x['amplicon_seq'].find(str(Seq(dphi_wdw_seq).reverse_complement())), x['amplicon_seq'].find(str(Seq(dphi_wdw_seq).reverse_complement())) + len(dphi_wdw_seq))
        idxs_dphi_wdw_cds = (max(idxs_cds[0], idxs_dphi[0]), min(idxs_cds[1], idxs_dphi[1]))
        dphi_wdw_cds_seq = x['amplicon_seq'][idxs_dphi_wdw_cds[0]:idxs_dphi_wdw_cds[1]]
        cds_l = int(x['CDS_frame'])
        if idxs_dphi_wdw_cds[0] > idxs_cds[0]:
            cds_l = cds_l - ((idxs_dphi_wdw_cds[0] - idxs_cds[0]) % 3)
            if cds_l < 0:
                cds_l = cds_l + 3
        df_dphi.loc[df_dphi['sample'] == x['sample'], 'dphi_wdw_seq'] = dphi_wdw_seq
        df_dphi.loc[df_dphi['sample'] == x['sample'], 'dphi_cds_seq'] = str(Seq(dphi_wdw_cds_seq).reverse_complement())
        df_dphi.loc[df_dphi['sample'] == x['sample'], 'CDS_frame_dphi'] = len(dphi_wdw_cds_seq[cds_l:]) % 3
# rearrange columns and export df_dphi as ref file
df_dphi = df_dphi.reindex(columns=['sample', 'sgRNA_ID', 'sgRNA_seq', 'sgRNA_dir', 'Gene', 'Exon', 'Domain', 'cut_site_AA', 'cluster_id',
                                   'label', 'dist_5p', 'dist_3p', 'Precision', 'pct_fs', 'pct_if', 'pct_+1', 'pct_+2', 'avg_indel', 'indelphi_seq',
                                   'dphi_wdw_seq', 'dphi_cds_seq', 'amplicon_seq', 'coding_seq', 'crispresso_wdw_seq', 'CDS_frame', 'CDS_frame_dphi'])
df_dphi.to_csv('./inDelphi_outputs/inDelphi_gene_summary_master_reference.csv', index=False)

# import raw inDelphi outputs for DNMT1/UHRF1 sgRNAs and process with indelphi_process()
# run once, then import after
df_dphi = pd.read_csv('./inDelphi_outputs/inDelphi_gene_summary_master_reference.csv')
dict_dphi = {}
for x in df_dphi['sample']:
    df_input = pd.read_csv('./inDelphi_outputs/inDelphi_output_%s.csv' % x, index_col=0)
    if df_dphi[df_dphi['sample'] == x]['sgRNA_dir'].values[0] == '-':
        x_rev = True
    else:
        x_rev = False
    temp_output = indelphi_process(sample=x, in_indelphi=df_input, in_ref=df_dphi, reverse=x_rev)
    temp_output = temp_output.fillna(value={k:temp_output['mut_type'] for k in ['prot_aln_cds', 'prot_ref_cds', 'prot_aln_wdw', 'prot_ref_wdw']})
    temp_output['mut_type'] = np.where(temp_output['prot_aln_wdw'].str.contains('\*', regex=True), 'nonsense', temp_output['mut_type'])
    temp_output = temp_output.rename(columns={'aln_wdw':'aln_seq','ref_wdw':'ref_seq','prot_aln_wdw':'prot_aln','prot_ref_wdw':'prot_ref'})
    temp_output['mut_type2'] = np.where(temp_output['mut_type'].str.contains('wild-type|in-frame'), temp_output['mut_type'], 'loss-of-function')
    temp_output.to_csv('./inDelphi_outputs/indelphi_processed/%s_processed.csv' % x, index=False)

### import processed inDelphi data (store in dict_dphi) and calculate predicted % IF/LOF
df_dphi[['pred_if_adj','pred_lof_adj']] = np.nan
dict_dphi = {}
for x in df_dphi['sample']:
    temp = pd.read_csv('./inDelphi_outputs/indelphi_processed/%s_processed.csv' % x).drop(columns=['prot_aln_cds','prot_ref_cds'])
    dict_dphi[x] = temp.copy()
    temp_dict = {'pred_if_adj': temp.groupby('mut_type2')['pred_freq'].sum()['in-frame'],
                 'pred_lof_adj': temp.groupby('mut_type2')['pred_freq'].sum()['loss-of-function']}
    df_dphi.loc[df_dphi['sample'] == x, ['pred_if_adj','pred_lof_adj']] = df_dphi.fillna(temp_dict)

#%% calculate allele type distributions with the aggregated/filtered genotypes
### 3 categories: wt, if (in-frame), lof (loss-of-function)
### calculate absolute % wt/if/lof in addition to relative % if/lof for dac/dmso treatment
### relative freq. == relative to total % of edited reads (excluding wt)

df_muts = df_ref[['oligo_id','label','sgRNA_ID','sgRNA_seq','Gene','Domain','cut_site_AA','DAC_res','cluster_id']].copy().rename(columns={'oligo_id':'sample'})
df_muts[['wt_dac', 'wt_dmso', 'if_dac', 'if_dmso', 'lof_dac', 'lof_dmso', 'rel_if_dac', 'rel_if_dmso', 'rel_lof_dac', 'rel_lof_dmso']] = np.nan

for x in df_muts['sample']:
    temp = dict_sgrnas[x].copy()
    temp_dict = dict(wt_dac=temp[temp['mut_type2'] == 'wild-type']['pct_reads_s1'].sum(),
                     wt_dmso=temp[temp['mut_type2'] == 'wild-type']['pct_reads_s2'].sum(),
                     if_dac=temp[temp['mut_type2'] == 'in-frame']['pct_reads_s1'].sum(),
                     if_dmso=temp[temp['mut_type2'] == 'in-frame']['pct_reads_s2'].sum(),
                     lof_dac=temp[temp['mut_type2'] == 'loss-of-function']['pct_reads_s1'].sum(),
                     lof_dmso=temp[temp['mut_type2'] == 'loss-of-function']['pct_reads_s2'].sum(),
                     rel_if_dac=temp[temp['mut_type2'] == 'in-frame']['adj_freq_s1'].sum(),
                     rel_if_dmso=temp[temp['mut_type2'] == 'in-frame']['adj_freq_s2'].sum(),
                     rel_lof_dac=temp[temp['mut_type2'] == 'loss-of-function']['adj_freq_s1'].sum(),
                     rel_lof_dmso=temp[temp['mut_type2'] == 'loss-of-function']['adj_freq_s2'].sum())
    df_muts.loc[df_muts['sample'] == x, temp_dict.keys()] = df_muts.fillna(temp_dict)

#%% Supp. Fig. 3 -- isolate enriched in-frame cluster 1 variants (>1% freq in DAC and >2 LFC)

df_cl1_vars = pd.DataFrame(columns=['sample'])
for x in df_muts[df_muts['cluster_id'] == 1]['sample']:
    temp_df = dict_sgrnas[x].copy()
    temp_df['sample'] = x
    temp_enrich = temp_df[temp_df['mut_type2'] == 'in-frame'].copy()
    temp_enrich = temp_enrich[(temp_enrich['pct_reads_s1'] >= 1) & (temp_enrich['LFC'] >= 2)].copy()
    df_cl1_vars = pd.concat([df_cl1_vars, temp_enrich], ignore_index=True)
df_cl1_vars['aa_indel'] = df_cl1_vars['indel'].div(3)
df_cl1_vars['if_mut_type'] = np.where(df_cl1_vars['aa_indel'] > 0, 'insertion', np.where(df_cl1_vars['aa_indel'] < 0, 'deletion', 'missense'))

#%% calculate all metrics and pre-processing for PCA and k-means clustering analysis

# df to hold all the pca metrics
df_pca_set = df_muts.copy()
# z-score transform the DAC resistance scores
df_pca_set['DAC_res'] = stats.zscore(df_pca_set['DAC_res'])

### calculate log2 fold-change of % WT/IF and rel% IF
df_pca_set['wt_lfc'] = np.log2(df_pca_set['wt_dac'] / df_pca_set['wt_dmso'])
df_pca_set['if_lfc'] = np.log2(df_pca_set['if_dac'] / df_pca_set['if_dmso'])
df_pca_set['rel_if_lfc'] = np.log2(df_pca_set['rel_if_dac'] / df_pca_set['rel_if_dmso'])

### calculate Kullback-Leibler divergence for all alleles w/ >0.1% freq in DAC or DMSO, add pseudocount of 0.01%
df_pca_set['kld_all'] = np.nan
for x in df_pca_set['sample']:
    temp_df = dict_sgrnas[x].copy()
    # convert to fraction and add pseudocount of 0.01%
    temp_df[['pct_reads_s1','pct_reads_s2']] = temp_df[['pct_reads_s1','pct_reads_s2']].apply(lambda x: x / x.sum(), axis=0) + (0.01 / 100)
    df_pca_set.loc[df_pca_set['sample'] == x, 'kld_all'] = (sp.rel_entr(temp_df['pct_reads_s1'], temp_df['pct_reads_s2']).sum() + sp.rel_entr(temp_df['pct_reads_s2'], temp_df['pct_reads_s1']).sum())

### calculate Gini coefficients for all alleles ('g1_dac') and edited alleles ('g2_dac') in DAC
df_pca_set[['g1_dac', 'g2_dac']] = np.nan
df_cdf_all, df_cdf_edit = pd.DataFrame(columns=['allele_id']), pd.DataFrame(columns=['allele_id'])
df_cdf_all.loc[0], df_cdf_edit.loc[0] = 0, 0
for x in df_pca_set['sample']:
    # generate CDF for all alleles >=0.1% freq in DAC
    temp_all = dict_sgrnas[x].copy()
    temp_all[temp_all['pct_reads_s1'] >= 0.1].copy().sort_values(by=['pct_reads_s1','pct_reads_s2'], ascending=False).reset_index(drop=True)
    temp_all['pct_s1'] = temp_all['pct_reads_s1'] / temp_all['pct_reads_s1'].sum()
    df_cdf_all = df_cdf_all.merge(get_allele_cumsum_v2(temp_all, 'pct_s1', out_col=x + '_dac'), on='allele_id', how='outer')
    # generate CDF for edited (IF/LOF) alleles >=0.1% in DAC
    temp_edit = dict_sgrnas[x].copy()
    temp_edit = temp_edit[(temp_edit['pct_reads_s1'] >= 0.1) & (temp_edit['mut_type'] != 'wild-type')].copy().sort_values(by=['pct_reads_s1','pct_reads_s2'], ascending=False).reset_index(drop=True)
    temp_edit['pct_s1'] = temp_edit['pct_reads_s1'] / temp_edit['pct_reads_s1'].sum()
    df_cdf_edit = df_cdf_edit.merge(get_allele_cumsum_v2(temp_edit, 'pct_s1', out_col=x + '_dac'), on='allele_id', how='outer')
# clean up allele CDF dataframes
df_cdf_all.loc[0], df_cdf_edit.loc[0] = 0, 0
df_cdf_all['allele_id'], df_cdf_edit['allele_id'] = df_cdf_all['allele_id'].astype(int), df_cdf_edit['allele_id'].astype(int)
# calculate Gini coefficients w/ CDF dataframes
for x in df_pca_set['sample']:
    df_pca_set.loc[df_pca_set['sample'] == x, 'g1_dac'] = np.trapz(df_cdf_all[x + '_dac'].dropna()) / len(df_cdf_all[x + '_dac'].dropna())
    df_pca_set.loc[df_pca_set['sample'] == x, 'g2_dac'] = np.trapz(df_cdf_edit[x + '_dac'].dropna()) / len(df_cdf_edit[x + '_dac'].dropna())

### calculate Pearson correlations between DAC vs. DMSO for all alleles
df_pca_set['r_s1s2_all'] = np.nan
for x in df_pca_set['sample']:
    temp_df = dict_sgrnas[x].copy()
    df_pca_set.loc[df_pca_set['sample'] == x, 'r_s1s2_all'] = stats.pearsonr(temp_df['pct_reads_s1'], temp_df['pct_reads_s2'])[0]

### calculate log-odds(% IF/% LOF) and (% IF/% LOF) odds ratios for DAC/DMSO/inDelphi
df_odds = df_muts[list_refcols + ['wt_dac', 'wt_dmso', 'if_dac', 'if_dmso', 'lof_dac', 'lof_dmso', 'rel_if_dac', 'rel_if_dmso', 'rel_lof_dac', 'rel_lof_dmso']].copy()
df_odds = df_odds.merge(df_dphi[['sample','Precision','pred_if_adj','pred_lof_adj']], how='left', on='sample')
# calculate % edited reads (abs % IF + abs % LOF)
df_odds['pct_edit_dac'] = df_odds['if_dac'] + df_odds['lof_dac']
df_odds['pct_edit_dmso'] = df_odds['if_dmso'] + df_odds['lof_dmso']
# log2 odds (rel_if / rel_lof) for DAC and DMSO -- 'in-frame vs. loss-of-function odds'
df_odds['dac_logodds'] = np.log2(df_odds['rel_if_dac'] / df_odds['rel_lof_dac'])
df_odds['pred_logodds'] = np.log2(df_odds['pred_if_adj'] / df_odds['pred_lof_adj'])
# log2(rel % IF / rel % LOF) odds ratio for DAC vs. DMSO -- in-frame vs. loss-of-function odds ratio
df_odds['dac_dmso_or'] = np.log2((df_odds['rel_if_dac'] / df_odds['rel_lof_dac']) / (df_odds['rel_if_dmso'] / df_odds['rel_lof_dmso']))
df_odds['dac_pred_or'] = np.log2((df_odds['rel_if_dac'] / df_odds['rel_lof_dac']) / (df_odds['pred_if_adj'] / df_odds['pred_lof_adj']))
# log2 (% edit/% wt) odds ratio for DAC vs. DMSO -- edited vs. wild-type odds ratio
df_odds['edit_dac_dmso_or'] = np.log2((df_odds['pct_edit_dac'] / df_odds['wt_dac']) / (df_odds['pct_edit_dmso'] / df_odds['wt_dmso']))
df_pca_set = df_pca_set.merge(df_odds[['sample','dac_logodds','dac_dmso_or','edit_dac_dmso_or']], how='left', on='sample')

#%% PCA and k-means clustering analysis

# refcols
list_idvars = ['sample', 'Gene', 'Domain', 'cut_site_AA', 'cluster_id']
# PCA feature set
list_pca = ['DAC_res', 'wt_dac', 'if_dac', 'rel_if_dac', 'wt_lfc', 'if_lfc', 'rel_if_lfc',
            'kld_all', 'g1_dac', 'g2_dac', 'r_s1s2_all', 'dac_logodds', 'dac_dmso_or', 'edit_dac_dmso_or']


### perform PCA with df_pca1 == DNMT1 sgRNAs only
# preprocess with quantile transformation
df_pca1 = df_pca_set[df_pca_set['Gene'] == 'DNMT1'][list_idvars + list_pca].copy().reset_index(drop=True)
df_pca1[list_pca] = QuantileTransformer(n_quantiles=len(df_pca1)).fit_transform(df_pca1[list_pca])
# PCA and K-means clustering on DNMT1 sgRNAs
df_pca_dnmt = df_pca_set[df_pca_set['Gene'] == 'DNMT1'][list_idvars].copy().reset_index(drop=True)
pca_model = PCA(n_components=10)
pca_fit = pca_model.fit_transform(df_pca1[list_pca])
pca_out = df_pca1[list_idvars].join(pd.DataFrame(pca_fit))
pca_model_km = KMeans(n_clusters=2, n_init=1000)
pca_model_km.fit(pca_out[np.arange(pca_model.n_components_)])
pca_out['kclus'] = pca_model_km.labels_
df_pca_dnmt[['pc1','pc2']] = pca_out[[0,1]]
df_pca_dnmt['km_clus'] = pca_out['kclus']
df_pca_dnmt['km_dnmt'] = np.where(df_pca_dnmt['km_clus'] == 0, 'Cluster K1', 'Cluster K2')
# join k-means clusters back to df_muts
df_muts = df_muts.merge(df_pca_dnmt[['sample','km_dnmt']], how='left', on='sample')


### perform PCA with df_pca2 == DNMT1 + UHRF1 sgRNAs
# preprocess with quantile transformation
df_pca2 = df_pca_set[list_idvars + list_pca].copy().reset_index(drop=True)
df_pca2[list_pca] = QuantileTransformer(n_quantiles=len(df_pca2)).fit_transform(df_pca2[list_pca])
# PCA and K-means clustering on DNMT1+UHRF1 sgRNAs
df_pca_all = df_pca_set[list_idvars].copy().reset_index(drop=True)
pca_model = PCA(n_components=10)
pca_fit = pca_model.fit_transform(df_pca2[list_pca])
pca_out = df_pca2[list_idvars].join(pd.DataFrame(pca_fit))
pca_model_km = KMeans(n_clusters=2, n_init=1000)
pca_model_km.fit(pca_out[np.arange(pca_model.n_components_)])
pca_out['kclus'] = pca_model_km.labels_
df_pca_all[['pc1','pc2']] = pca_out[[0,1]]
df_pca_all['km_clus'] = pca_out['kclus']
# for all sgRNAs, Cluster K1 == 'drug-divergent', Cluster K2 == 'other'
df_pca_all['km_all'] = np.where(df_pca_all['km_clus'] == 0, 'Cluster K1', 'Cluster K2')
# join k-means clusters back to df_muts
df_muts = df_muts.merge(df_pca_all[['sample','km_all']], how='left', on='sample')

#%% import and process the sgRNA genotyping data for the timecourse expt in Fig. 3

# setting variables
in_ref = 'DNMT1_sgRNA_timecourse_reference.csv'
in_batch = 'DNMT1_sgRNA_timecourse_batchfile.batch'
out_folder = 'KN10180_analysis_v1_221031'

df_ref = pd.read_csv(in_ref)
df_batch = pd.read_csv(in_batch, sep='\t')


batch_analyze(list_samples=df_ref['sample_name'], in_ref=in_ref, in_batch=in_batch,
              out_folder='DNMT1_sgRNA_timecourse/translated_alleles',
              out_stats='DNMT1_sgRNA_timecourse_batch_analysis_stat.csv',
              list_outfiles=df_ref['sample_name'] + '_tln.csv')

dict_sgrnas2 = {}
df_mut_stats = pd.DataFrame(index=['wild-type','in-frame','loss-of-function'])
# clean analyze_alleles() output csvs
for x in df_ref['sample_name']:
    temp = pd.read_csv('./DNMT1_sgRNA_timecourse/translated_alleles/' + x + '_tln.csv', index_col=0)
    temp['mut_type'] = temp['mut_type'].str.replace('likely ', '', regex=False) # remove likely tag
    # merge 1 nt subs w/ no ins/del (prob. pcr/seq errors) into WT
    temp1a = temp[(temp['mut_type'].str.contains('wild-type|in-frame'))].copy() # first isolate WT/IF alleles
    temp1a = temp1a[(temp1a['n_del'] == 0) & (temp1a['n_ins'] == 0) & (temp1a['n_mut'] <= 1)].copy() # 0 ins/del, 1 nt subs
    # assign aggregated stats to WT then drop aggregated alleles and re-sort
    temp.loc[temp1a.index[0], ['n_reads','pct_reads','n_alleles']] = [temp1a[x].sum() for x in ['n_reads','pct_reads','n_alleles']]
    temp = temp.drop(index=temp1a.index[1:]).sort_values(by=['pct_reads'], ascending=False).reset_index(drop=True)
    # merge alleles w/ prot_aln == wt prot into WT (silent or muts outside window) and drop
    seq_wt = temp.loc[temp['mut_type'] == 'wild-type', 'prot_ref'].values[0]
    temp1b = temp.loc[((temp['prot_aln'] == seq_wt) & (temp['mut_type'] == 'in-frame')) | (temp['mut_type'] == 'wild-type')].copy()
    temp.loc[temp1b.index[0], ['n_reads','pct_reads','n_alleles']] = [temp1b[x].sum() for x in ['n_reads','pct_reads','n_alleles']]
    temp = temp.drop(index=temp1b.index[1:]).sort_values(by=['pct_reads'], ascending=False).reset_index(drop=True)
    # label nonsense alleles and then classify WT/IF/LOF
    temp['mut_type'] = np.where(temp['prot_aln'].str.contains('\*', regex=True), 'nonsense', temp['mut_type'])
    temp['mut_type2'] = np.where(temp['mut_type'] == 'wild-type', 'wild-type',
                                 np.where(temp['mut_type'] == 'in-frame', 'in-frame', 'loss-of-function'))
    # drop alleles w/ freq less than 0.1% and recalculate new frequencies
    temp2 = temp[temp['pct_reads'] >= 0.1].copy()
    temp2['pct_reads_adj'] = temp2['n_reads'] / temp2['n_reads'].sum() * 100
    # export to csv
    temp2.to_csv('./DNMT1_sgRNA_timecourse/aggregated_alleles_min0.1/' + x + '_agg.csv', index=False)
    dict_sgrnas2[x] = temp2.copy()
    # calculate WT/IF/LOF stats using all alleles (df_muts1) vs. <0.1% alleles (df_muts2)
    df_mut_stats = df_mut_stats.merge(temp.groupby(by='mut_type2').agg({'pct_reads_adj':'sum'}).rename(columns={'pct_reads_adj':x}), how='outer', left_index=True, right_index=True)
# transpose/rename df_mut_stats, fillna
df_mut_stats = df_mut_stats.transpose().reset_index().rename(columns={'index':'sample_name','wild-type':'wt', 'in-frame':'if', 'loss-of-function': 'lof'}).fillna(0)
# calculate in-frame log2 fold-change
df_mut_stats['lfc_if'] = np.log2(df_mut_stats.apply(lambda x: ((x['if'] + 0.1) / (df_mut_stats.loc[(df_mut_stats['label'] == x['label']) & (df_mut_stats['time'] == 0)]['if'] + 0.1)).values[0], axis=1, result_type='expand'))
# export
df_mut_stats.to_csv('./DNMT1_sgRNA_timecourse/DNMT1_sgRNA_timecourse_mut_type_breakdown.csv', index=False)