
import jpy_tools.overlapTools
# importlib.reload(jpy_tools.overlapTools)
from jpy_tools.overlapTools import GeneOverlapBase
import scipy
import tqdm
from typing import Literal, Union, List, Dict, Optional
# from typing_extensions import Literal

from jpy_tools.otherTools import F
from jpy_tools.soExt import Axhline, Axvline, Axline, KdeDensityColor

from scipy.stats import mannwhitneyu
from scipy.interpolate import interpn
from scipy.stats import gaussian_kde
from scipy.stats import pearsonr
import math

import pandas as pd
import seaborn.objects as so
import seaborn as sns
import numpy as np

from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

'''
Author            : KVAVP, YMBOIBP and AIMKMBO
Date              : 2024-01-01
LastEditTime      : 2024-08-20

Draw xyplot
'''


def addDensityToDf(df:pd.DataFrame, x:str, y:str, group=None, bins=20):
    if group:
        df = df.groupby(group, as_index=False).apply(lambda df: addDensityToDf(df, x=x, y=y, bins=bins)).reset_index(level=0, drop=True)
    elif bins is None:
        x = df[x].values
        y = df[y].values
        xy = np.vstack([x,y])
        z = gaussian_kde(xy)(xy)
        df['temp_density'] = z
        df = df.sort_values('temp_density')
    else:
        x = df[x].values
        y = df[y].values
        data , x_e, y_e = np.histogram2d( x, y, bins = bins, density = True )
        z = interpn( ( 0.5*(x_e[1:] + x_e[:-1]) , 0.5*(y_e[1:]+y_e[:-1]) ) , data , np.vstack([x,y]).T , method = "splinef2d", bounds_error = False)
        z[np.where(np.isnan(z))] = 0.0
        df['temp_density'] = z
        df = df.sort_values('temp_density',)
    return df

class CompareTermination(object):

    def __init__(self):
        # self.wtName = wtName
        # df_wt = df_wt.copy()
        # dt_trnaInfo = df_trnaInfo.set_index('Gene')['Category'].to_dict()
        # df_wt['tRNA'] = df_wt['gene_id'].map(lambda _: dt_trnaInfo.get(_, 'None'))
        # self.df_wt = df_wt

        self.dtDf_mock = {}
        self.dtDf_compare = {}
        self.dtPvalueInfo = {}
        self.dt_compareMockRelation = {}
        self.dt_cutoff = {}
        self.overlapInfo = {}
        self.dt_genes = {}
        self.dt_geneCoverage = {}

    def __repr__(self) -> str:
        # wtInfo = f'WT: {self.wtName}: {self.df_wt.shape[0]}'
        mockInfo = '\n\t'.join([f'{k}: {v.shape[0]}' for k, v in self.dtDf_mock.items()])
        compareInfo = '\n\t'.join([f'{k}: {v.shape[0]}' for k, v in self.dtDf_compare.items()])
        compareRelation = '\n\t'.join([f'{k}: {v}' for k, v in self.dt_compareMockRelation.items()])
        pvalueInfo = '\n\t'.join([f'{k}: {v.shape[0]}' for k, v in self.dtPvalueInfo.items()])
        return f'{mockInfo}\ncompare:\n{compareInfo}\n{compareRelation}\npvalue:\n{pvalueInfo}'
    
    def addMockData(self, name: str, df_mock: pd.DataFrame):
        df_mock = df_mock.copy()
        # df_trnaInfo = self.df_trnaInfo.copy()
        # dt_trnaInfo = df_trnaInfo.set_index('Gene')['Category'].to_dict()
        # df_mock['tRNA'] = df_mock['gene_id'].map(lambda _: dt_trnaInfo.get(_, 'None'))
        self.dtDf_mock[name] = df_mock
    
    def addGenesCategory(self, key:str, dt_genes: Dict[str, str], ls_order:Optional[List]=None):
        # dt_genes : {'g1': 'cate1', 'g2': 'cate2', 'g3': 'cate1'}
        self.dt_genes[key] = dt_genes.copy()
        if ls_order is None:
            ls_order = sorted(set(dt_genes.values()))
        self.dt_genes[key + '_order'] = [*ls_order, 'Others']

    def addComparedData(self, name:str, df_compare:pd.DataFrame, usedMock:Union[str, List[str]]):
        df_compare = df_compare.copy()
        self.dtDf_compare[name] = df_compare.copy()
        if isinstance(usedMock, str):
            pass
        elif isinstance(usedMock, list):
            usedMock = tuple(usedMock)
            lsDf = []
            for mock in list(usedMock):
                assert mock in self.dtDf_mock, f'{mock} not in {list(self.dtDf_mock.keys())}'
                # merge all mock dataframe
                df_mock = self.dtDf_mock[mock]
                lsDf.append(df_mock)
                # print(df_mock.shape)
            # only keep intersection
            ls_intersect = [set(df['gene_id'].tolist()) for df in lsDf]
            ls_intersect = list(set.intersection(*ls_intersect))
            # print(len(ls_intersect))
            for i, (mock, df) in enumerate(zip(list(usedMock), lsDf)):
                df = df.set_index('gene_id').loc[ls_intersect]
                df.columns = [f"{x}_{mock}" for x in df.columns]
                lsDf[i] = df
            df_mock = pd.concat(lsDf, axis=1)
            # add multiindex
            self.dtDf_mock[usedMock] = df_mock.reset_index()

        else:
            raise ValueError(f'{usedMock} not supported')
        self.dt_compareMockRelation[name] = usedMock


    def getPvalueUseUtest(self, name:Union[None, str, List[str]]=None):
        from scipy.stats import mannwhitneyu
        if name is None:
            name = list(self.dtDf_compare.keys())
        elif isinstance(name, str):
            name = [name]
        ls_name = name
        for x in ls_name:
            assert x in self.dtDf_compare, f'{x} not in {list(self.dtDf_compare.keys())}'
        
        for x in tqdm.tqdm(ls_name):
            df_compare = self.dtDf_compare[x].copy()
            compareUsedMock = self.dt_compareMockRelation[x]
            if isinstance(compareUsedMock, str):
                df_wt = self.dtDf_mock[compareUsedMock]
                usedGene = set(df_compare['gene_id']) & set(df_wt['gene_id'])
                df_res = df_wt[df_wt['gene_id'].isin(usedGene)]
                df_res = df_res.merge(df_compare, on='gene_id', suffixes=('_wt', '_compare'))
                ls_pvalue = df_res.apply(lambda x: mannwhitneyu(
                    x[f'data_wt'], x[f'data_compare'])[1], axis=1)
                df_res['pvalue'] = ls_pvalue
                df_res['-log10P'] = -np.log10(df_res['pvalue'])
                self.dtPvalueInfo[x] = df_res
            elif isinstance(compareUsedMock, tuple):
                df_wt = self.dtDf_mock[compareUsedMock]
                usedGene = set(df_compare['gene_id']) & set(df_wt['gene_id'])
                # import pdb; pdb.set_trace()
                df_res = df_wt[df_wt['gene_id'].isin(usedGene)]
                df_compare.columns = [f"{x}_compare" if x != 'gene_id' else 'gene_id' for x in df_compare.columns]
                df_res = df_res.merge(df_compare, on='gene_id')
                for mock in compareUsedMock:
                    ls_pvalue = df_res.apply(lambda x: mannwhitneyu(
                        x[f'data_compare'], x[f'data_{mock}'])[1], axis=1)
                    df_res[f'pvalue_{mock}'] = ls_pvalue
                    df_res[f'-log10P_{mock}'] = -np.log10(df_res[f'pvalue_{mock}'])
                df_res['pvalue'] = df_res[[f'pvalue_{mock}' for mock in compareUsedMock]].max(axis=1)
                df_res['-log10P'] = -np.log10(df_res['pvalue'])
                self.dtPvalueInfo[x] = df_res

    def getTerminationLengthDifferenceAndFc(self, name:Union[None, str, List[str]]=None):
        if name is None:
            name = list(self.dtPvalueInfo.keys())
        elif isinstance(name, str):
            name = [name]
        ls_name = name
        for x in ls_name:
            assert x in self.dtPvalueInfo, f'{x} not in {list(self.dtPvalueInfo.keys())}'
        
        for x in ls_name:
            df_res = self.dtPvalueInfo[x].copy()
            compareUsedMock = self.dt_compareMockRelation[x]
            if isinstance(compareUsedMock, str):
                df_res['median_wt'] = df_res['data_wt'].map(np.median)
            elif isinstance(compareUsedMock, tuple):
                df_res['median_wt'] = df_res[[f'data_{mock}' for mock in compareUsedMock]].applymap(np.median).mean(axis=1)
            df_res['median_compare'] = df_res['data_compare'].map(np.median)
            df_res['difference'] = df_res['median_compare'] - df_res['median_wt']
            df_res['log2FC'] = np.log2(df_res['median_compare'] / df_res['median_wt'])
            self.dtPvalueInfo[x] = df_res
    
    def setCutoff(self, name:Union[None, str, List[str]]=None, difference:Optional[float]=None, fc:Optional[float]=None, pvalue:Optional[float]=None):
        if name is None:
            name = list(self.dtPvalueInfo.keys())
        elif isinstance(name, str):
            name = [name]
        ls_name = name

        for name in ls_name:
            df_res = self.dtPvalueInfo[name].copy()
            if difference is None:
                ls_long = df_res['gene_id'].tolist()
                ls_short = df_res['gene_id'].tolist()
            else:
                ls_long = df_res[df_res['difference'] >= difference]['gene_id'].tolist()
                ls_short = df_res[df_res['difference'] <= -difference]['gene_id'].tolist()

            if fc is None:
                pass
            else:
                _df = df_res.query('gene_id in @ls_long')
                ls_long = _df[_df['log2FC'] >= fc]['gene_id'].tolist()

                _df = df_res.query('gene_id in @ls_short')
                ls_short = _df[_df['log2FC'] <= -fc]['gene_id'].tolist()

            if pvalue is None:
                pass
            else:
                _df = df_res.query('gene_id in @ls_long')
                ls_long = _df[_df['pvalue'] <= pvalue]['gene_id'].tolist()
                _df = df_res.query('gene_id in @ls_short')
                ls_short = _df[_df['pvalue'] <= pvalue]['gene_id'].tolist()

            df_res = df_res.assign(significant=np.select(
                [df_res['gene_id'].isin(ls_long), df_res['gene_id'].isin(ls_short)],
                ['Up', 'Down'], 'NS'
            ))
            self.dtPvalueInfo[name] = df_res
            self.dt_cutoff[name] = dict(difference=difference, fc=fc, pvalue=pvalue)
    
    def scatterPlot(self, name: Union[str, List[str]], x, y, color='density', cutoff=True):
        if isinstance(name, str):
            ls_name = [name]
            onlyOneSample = True
        else:
            onlyOneSample = False
        
        lsDf = []
        ls_name = name
        for name in ls_name:
            df = self.dtPvalueInfo[name].copy()
            if (color == 'density') | (color == 'temp_density'):
                df = addDensityToDf(df, x=x, y=y, bins=None)
                color = 'temp_density'
                colorTitle = 'Density'
            elif color == 'significant':
                colorTitle = 'significant'
                df = df.sort_values('significant')
            elif color in self.dt_genes:
                colorTitle = color
                df[color] = df['gene_id'].map(lambda _: self.dt_genes[color].get(_, 'Others'))
                ls_order = self.dt_genes[color + '_order']
                df[color] = pd.Categorical(df[color], categories=ls_order, ordered=True)
                df = df.sort_values(color, ascending=False)
            else:
                raise ValueError(f'{color} not supported')
            df = df.assign(name=name)
            lsDf.append(df)
            
        df = pd.concat(lsDf).reset_index(drop=True)
        p = (
            so.Plot(df, x=x, y=y, color=color)
            .facet(col='name', wrap=4, order=ls_name)
            .add(so.Dot(pointsize=5, alpha=0.6))
            .label(color=colorTitle)
        )

        # if color == 'tRNA':
        #     p = p.scale(color={"None": "#c9c9c9", "convergent": "#f6a6a6", 'tandem': '#86b1d2'})
        if color == 'significant':
            p = p.scale(color=so.Nominal({"NS": "#d2d2d2", "Up": '#F84042' , 'Down': '#56A0F9'}, ['NS','Up','Down']))
        if cutoff:
            if isinstance(cutoff, bool):
                if x in ['difference', 'log2FC']:
                    cutoffX = {
                        'difference': 'difference',
                        'log2FC': 'fc',
                    }[x]
                    cutoff = self.dt_cutoff[name][cutoffX]
            elif isinstance(cutoff, (int, float)):
                pass

            if cutoff is not None:
                p = p.add(
                        Axvline(x=cutoff, linestyle='--', color='lightgrey'), legend=False, x=None, y=None, color=None, data={}
                    ).add(
                        Axvline(x=-cutoff, linestyle='--', color='lightgrey'), legend=False, x=None, y=None, color=None, data={}
                    )
            cutoffP = self.dt_cutoff[name]['pvalue']
            if cutoffP is not None:
                if y == '-log10P':
                    p = p.add(
                            Axhline(y=-np.log10(cutoffP), linestyle='--', color='lightgrey'), legend=False, x=None, y=None, color=None, data={}
                            )
        p = p.label(x=x.capitalize())
        if y == '-log10P':
            p = p.label(y='-log$_{10}$P-value')
        
        if onlyOneSample:
            p = p.facet(col=None, wrap=None)

        return p

    def getOverlapInfo(self, category:Literal['Up', 'Down'], names=None, force=False, diagS=1, diagQ=1):
        from itertools import product
        ### 断言函数，如果 category 不在['Up', 'Down']中的话，就报错
        assert category in ['Up', 'Down']
        if names is None:
            names = list(self.dtPvalueInfo.keys())
        if category in self.overlapInfo:
            if not force:
                print(f'{category} already exists')
                return self.overlapInfo[category]
        lsDf_allGenes = []
        for name in names:
            df = self.dtPvalueInfo[name].assign(name=name)
            lsDf_allGenes.append(df)
        df_allGenes = pd.concat(lsDf_allGenes).reset_index(drop=True)
        go = GeneOverlapBase(df_allGenes, 'gene_id', 'name', 'significant', categoryTarget=category)
        go.getOverlapInfo(names, diagS=diagS, diagQ=diagQ)
        self.overlapInfo[category] = go
        return go
    
    def overlapDot(self, category, *args, **dt_args):
        go = self.overlapInfo[category]
        h = go.dotplot(*args, **dt_args)
        return h

    def overlapTile(self, category, *args, **dt_args):
        go = self.overlapInfo[category]
        h = go.tileplot(*args, **dt_args)
        return h
    
    def addElongationBam(self, name, bamPath):
        if hasattr(self, 'dt_sampleElongationBam'):
            pass
        else:
            self.dt_sampleElongationBam = {}

        self.dt_sampleElongationBam[name] = bamPath

    def getAllGeneCoverage(self, name, polyaSites, bins=1, before=200, after=1000, excludeChr={'Pt', 'Mt'}, force=False, threads=64):
        import metaplot
        if isinstance(name, str):
            ls_name = [name]
        else:
            ls_name = name
        for sample in ls_name:
            treatment = sample
            runTreatment = True
            if force:
                pass
            else:
                if treatment in self.dt_geneCoverage:
                    runTreatment = False
            
            if runTreatment:
                df_treatment = metaplot.bam_reference_point(
                    self.dt_sampleElongationBam[treatment], polyaSites, bins=bins, before=before, after=after, 
                    min_counts=1, exclude_chr=excludeChr, return_raw=True, threads=64
                )
                self.dt_geneCoverage[treatment] = df_treatment
            
            
            wt = self.dt_compareMockRelation[sample]
            if isinstance(wt, str):
                ls_wt = [wt]
            else:
                ls_wt = wt
            for wt in ls_wt:
                runWt = True
                if force:
                    pass
                else:
                    if wt in self.dt_geneCoverage:
                        runWt = False

                if runWt:
                    df_wt = metaplot.bam_reference_point(
                        self.dt_sampleElongationBam[wt], polyaSites, bins=bins, before=before, after=after, 
                        min_counts=1, exclude_chr=excludeChr, return_raw=True, threads=64
                    )
                    self.dt_geneCoverage[wt] = df_wt
    
    def getGeneCoverage(self, name, ls_genes:Union[None, str, List[str]]=None, minCounts=15, scaleMethod:Literal['max', 'pas', 'none']='max'):
        df_coverage = self.dt_geneCoverage[name].copy()
        if ls_genes is None:
            ls_genes = df_coverage.index
        if isinstance(ls_genes, str):
            ls_genes = list(set(self.dt_genes[ls_genes].keys()))
            _ls = [x for x in ls_genes if x not in df_coverage.index.tolist()]
            logger.info(f"Not found {len(_ls)}/{len(ls_genes)}: {','.join(_ls)}")
            ls_genes = [x for x in ls_genes if x not in _ls]
            
            
        df_coverage = df_coverage.loc[ls_genes].query("N > @minCounts").drop(columns=['N'])
        sr_coverage = df_coverage.apply(lambda _: _/_.sum(), axis=1).mean(0)
        ### Do not ask the normalized method, we should keep the same pipeline wit mowp ( •̀ ω •́ )✧
        if scaleMethod == 'none':
            sr_coverage = sr_coverage
        elif scaleMethod == 'max':
            sr_coverage = sr_coverage / sr_coverage.max()
        elif scaleMethod == 'pas':
            sr_coverage = sr_coverage / sr_coverage.loc[0]
        else:
            raise ValueError(f'{scaleMethod} not supported')
        
        return sr_coverage
        
    def getCoverageForMetaplot(self, name, dt_genes:Dict[str, Union[None, str, List[str]]], minCounts=15, scaleMethod=Literal['max', 'pas', 'none']):
        if isinstance(name, str):
            ls_name = [name]
        else:
            ls_name = name
        
        ls_res = []
        for name in ls_name:
            for genesetName, ls_genes in dt_genes.items():
                res = self.getGeneCoverage(name, ls_genes, minCounts, scaleMethod)
                res['sample'] = name
                res['genesetName'] = genesetName
                ls_res.append(res)
        df_res = pd.concat(ls_res, axis=1).T
        df_res = df_res.melt(['sample', 'genesetName'], value_name='Cov', var_name='Pos')
        p = (
            so.Plot(df_res, x='Pos', y='Cov', color='sample', linestyle='genesetName')
            .add(so.Line())
            .label(color='Sample', linestyle='Gene set')
        )
        return df_res, p
    
def convert_word(strin:str, step:int = 1):
    """secret key + 1"""
    def num_add1(n:int, step):
        if n + step <= 25:
            return n + step
        else:
            return n + step - 25 - 1
    
    alp = "abcdefghijkimnopqrstuvwxyz".upper()
    alp_num = {x:i for i, x in enumerate(list(alp))}
    num_alp = {i:x for i, x in enumerate(list(alp))}
    
    data = ""
    for x in list(strin.upper()):
        if x not in alp_num:
            data += x
        else:
            n = alp_num[x]
            n_add = num_add1(n, step)
            data += num_alp[n_add]
    
    return data

def merge_data_draw_xy(
    data1: pd.DataFrame,
    data2: pd.DataFrame,
    on: str = 'gene_id',
    key: str = 'data',
    data1_label: str = None,
    data2_label: str = None,
    data1_font_style: str = None,
    data2_font_style: str = None,
    s: int = 6,
    p_values: float = .001,
    figsize: tuple = (3, 3),
    xlim: tuple = (0, 1000),
    ylim: tuple = (0, 1000)
):
    '''
    Plot scatterplot with p_values
    Args:
        data1, data2: input DataFrame with raw data
        on: Column or index level names to join on
        key: Column names of the raw data
        s: dot size
        p_values: threshold of the significant

    Return:
        data: merge Dataframe
        ax: pre-existing axes for the plot
    '''
    from scipy.stats import mannwhitneyu

    data = pd.merge(data1, data2, on=on)
    data.dropna(inplace=True)

    data['p_values'] = data.apply(lambda x: mannwhitneyu(
        x[f'{key}_x'], x[f'{key}_y'])[1], axis=1)
    data['data_x'] = data[f'{key}_x'].map(np.median)
    data['data_y'] = data[f'{key}_y'].map(np.median)
    for index, row in data.iterrows():
        if row['data_x'] < row['data_y'] and row['p_values'] < 0.001:
            data.loc[index,'color'] =2
        elif row['data_x'] > row['data_y'] and row['p_values'] < 0.001:
            data.loc[index,'color'] =1
        else :
            data.loc[index,'color'] =0
    plt.scatter(
        x='data_x',
        y='data_y',
        data=data.query('p_values > @p_values'),
        s=s,
        c='#D2D2D2',
        alpha=.4
    )

    plt.scatter(
        x='data_x',
        y='data_y',
        data=data.query('p_values < @p_values and data_y > data_x'),
        s=s,
        c='#56A0F9',
        alpha=.4
    )

    plt.scatter(
        x='data_x',
        y='data_y',
        data=data.query('p_values < @p_values and data_y < data_x'),
        s=s,
        c='#FF6666',
        alpha=.7
    )
    up = data.query('p_values < @p_values and data_y < data_x')
    down = data.query('p_values < @p_values and data_y > data_x')
    
    #NS = len(data) - len(up) - len(down)
    #plt.text(50, 950,f'NS = {NS}', fontsize=10, ha='left')
    
    plt.text(50, 950,f'Up = {len(up)}', fontsize=10, ha='left',c='#FF6666')
    plt.text(50, 850,f'Down = {len(down)}', fontsize=10, ha='left',c='#56A0F9')
    plt.plot(xlim, ylim, ls='--', color='#555555')
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xlabel(data1_label, style=data1_font_style)
    plt.xticks(np.linspace(xlim[0], xlim[1], 5, dtype=int), fontsize=10)
    plt.yticks(np.linspace(ylim[0], ylim[1], 5, dtype=int), fontsize=10)

    ax = plt.gca()
    ax.patch.set_visible(False)

    return data, ax


def twoD_density_xyplot(data,
                        control_name,
                        mutant_name,
                        marker_point:list = []):

    y = data['ratio_y']
    x = data['ratio_x']
    
    correlation, p_value = pearsonr(x, y)
    
    xy = np.vstack([x,y])
    z = gaussian_kde(xy)(xy)
    idx = z.argsort()
    x, y, z = x.iloc[idx], y.iloc[idx], z[idx]

    fig, ax = plt.subplots(figsize=(4,4))

    plt.xlabel(mutant_name)
    plt.ylabel(control_name)

    plt.plot([0, 1], [0, 1], ls='--', color='#555555')
    if marker_point != []:
        ax.plot(marker_point[0], marker_point[1], marker='*',color='red',markersize=8)
        
    xticks = np.array([0.0, 0.2,0.4, 0.6, 0.8,1.0])
    
    plt.xticks(xticks)
    scatter = ax.scatter(x, y, marker='o', c=z, s=5, 
                         cmap="RdYlBu_r",
                         label='LST')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = fig.colorbar(scatter, cax=cax, label='Density')

    # plt.text(-20, 9, 
    #          f'Pearson\'s R = {round(correlation, 3)}', 
    #          fontsize=10, 
    #          ha='left',
    #          c='black')

    sns.despine(top=True, right=True)
    
    return ax


def twoD_density_xyplot_v2(data,
                           control_name,
                           mutant_name,
                           marker_point:list = []):

    x = -1 * data['move']
    y = -1 * data['log2FoldChange']
    xy = np.vstack([x,y])
    z = gaussian_kde(xy)(xy)
    idx = z.argsort()
    x, y, z = x.iloc[idx], y.iloc[idx], z[idx]

    fig, ax = plt.subplots(figsize=(4,4))
    plt.xlabel('change in termination window')
    plt.ylabel('change in cleavage ratio')
    plt.title(f'{mutant_name} vs {control_name}')
    ax.axvline(0, ls='--', color='#555555')
    ax.axhline(0, ls='--', color='#555555')
    if marker_point != []:
        ax.plot(marker_point[0], marker_point[1], marker='*',color='red',markersize=8)

    plt.ylim(-3, 3)
    plt.xlim(-300, 300)

    xticks = np.array([-300, -150, 0, 150, 300])
    plt.xticks(xticks)
    scatter = ax.scatter(x, y,
                         marker='o',
                         c = z,
                         s=5,
                         cmap="RdYlBu_r",
                         label='LST')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = fig.colorbar(scatter, cax=cax, label='Density')
    sns.despine(top=True, right=True)
    
    return ax
