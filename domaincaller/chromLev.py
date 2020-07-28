# -*- coding: utf-8 -*-
"""
Created on Tue May 31 15:47:34 2016

@author: wxt
"""

from __future__ import division
import copy, collections
import numpy as np
from scipy import sparse

np.seterr(divide = "ignore")

class Chrom(object):
    
    minsize = 5

    def __init__(self, chrom, res, hicdata):

        self.chrom = chrom
        self.res = res
        self.chromLen = hicdata.shape[0]
        self.hmm = None

        x, y = hicdata.nonzero()
        mask = x < y
        x, y = x[mask], y[mask]
        mat_ = hicdata[x, y]
        if isinstance(mat_, np.matrix):
            IF = np.array(mat_).ravel()
        else:
            # mat_ is a sparse matrix
            IF = np.array(mat_.todense()).ravel()

        IF[np.isnan(IF)] = 0
        self.IF =IF
        self.x, self.y = x, y

        del hicdata


    def calDI(self, window=2000000):
        """
        Calculate DI for each bin.
        """
        ws = window // self.res
        # Perform filtering according to window size
        mask = self.y - self.x <= ws
        x, y = self.x[mask], self.y[mask]
        idata = self.IF[mask]
        
        Len = y.max() + 1
        # Downstream
        downs = np.bincount(x, weights = idata)
        # Upstream
        ups = np.bincount(y, weights = idata)
        ## Correct for length
        cdowns = np.zeros(Len)
        cdowns[:downs.size] = downs
        cups = np.zeros(Len)
        cups[(Len-ups.size):] = ups
        ## Formula
        numerator = cdowns - cups
        denominator_1 = cdowns + cups
        denominator_2 = numerator.copy()
        denominator_1[denominator_1==0] = 1
        denominator_2[denominator_2==0] = 1
        
        self.DIs = numerator**3 / np.abs(denominator_2) / denominator_1

    def splitChrom(self, DIs):
        
        # minregion and maxgaplen are set intuitively
        maxgaplen = max(100000 // self.res, 5)
        minregion = maxgaplen * 2

        valid_pos = np.where(DIs != 0)[0]
        gapsizes = valid_pos[1:] - valid_pos[:-1]
        endsIdx = np.where(gapsizes > (maxgaplen + 1))[0]
        startsIdx = endsIdx + 1

        chromRegions = {}
        for i in range(startsIdx.size - 1):
            start = valid_pos[startsIdx[i]]
            end = valid_pos[endsIdx[i + 1]] + 1
            if end - start > minregion:
                chromRegions[(start, end)] = DIs[start:end]

        if startsIdx.size > 0:
            start = valid_pos[startsIdx[-1]]
            end = valid_pos[-1] + 1
            if end - start > minregion:
                chromRegions[(start, end)] = DIs[start:end]
            start = valid_pos[0]
            end = valid_pos[endsIdx[0]] + 1
            if end - start > minregion:
                chromRegions[(start, end)] = DIs[start:end]

        if not startsIdx.size:
            if valid_pos.size > 0:
                start = valid_pos[0]
                end = valid_pos[-1]
                if end - start > minregion:
                    chromRegions[(start, end)] = DIs[start:end]
        
        self.regionDIs = chromRegions

    def viterbi(self, seq):
        """
        Find the most likely hidden state series using the viterbi algorithm.
        
        Parameters
        ----------
        seq : 1-D numbpy ndarray, float
            Adaptive DI array for any region.
        
        Returns
        -------
        path : list
            List of hidden state labels. Has the same length as the input
            *seq*.
        
        """
        path = [int(s.name) for i, s in self.hmm.viterbi(seq)[1][1:-1]]

        return path

    def _getBounds(self, path, junctions=['30']):
        
        pathseq = ''.join(map(str, path))
        pieces = [pathseq]
        for junc in junctions:
            gen = []
            for seq in pieces:
                tmp = seq.split(junc)
                if len(tmp) == 1:
                    gen.extend(tmp)
                else:
                    gen.append(tmp[0]+junc[0])
                    for s in tmp[1:-1]:
                        gen.append(junc[1]+s+junc[0])
                    gen.append(junc[1]+tmp[-1])

            pieces = gen
            
        bounds = np.r_[0, np.r_[list(map(len, pieces))]].cumsum()

        return bounds
        
    def pipe(self, seq, start):
       
        # bin-level domain (not base-pair-level domain!)
        bounds = self._getBounds(self.viterbi(seq), junctions=['30'])
        pairs = [[bounds[i], bounds[i+1]] for i in range(len(bounds)-1)]
        domains = []
        for b in pairs:
            # start, end
            tmp = [b[0]+start, b[1]+start]
            domains.append(tmp)

        return domains

    def minCore(self, regionDIs):
        
        tmpDomains = {}
        for region in sorted(regionDIs):
            seq = regionDIs[region]
            domains = self.pipe(seq, region[0])
            cr = (region[0]*self.res, region[1]*self.res)
            tmpDomains[cr] = []
            for domain in domains:
                domain[0] = domain[0] * self.res
                domain[1] = domain[1] * self.res
                tmpDomains[cr].append(domain)
        
        minDomains = self._orifilter(tmpDomains)

        return minDomains

    def getDomainList(self, byregion):
        """
        Combine by-region domains into a single list.
        
        Parameters
        ----------
        byregion : dict
            The keys are tuples representing gap-free regions of the chromosome,
            and the values are corresponding identified domain lists.
        
        Returns
        -------
        DomainList : list
            A merged domain list of all regions
        """
        DomainList = []
        for region in sorted(byregion):
            DomainList.extend(byregion[region])

        return DomainList

    def _orifilter(self, oriDomains):
        """
        Perform size filtering on the input domain lists.
        
        Parameters
        ----------
        oriDomains : dict
            The keys are tuples representing gap-free regions of the chromosome,
            and the values are corresponding identified domain lists. Start
            and end of the domain should be in base-pair unit.
        
        Returns
        -------
        filtered : dict
            Pairs of gap-free regions and corresponding filtered domain lists.
        """
        filtered = {}
        for region in oriDomains:
            tmplist = []
            for d in oriDomains[region]:
                if d[1] - d[0] >= (self.minsize*self.res):
                    tmplist.append(d)
            if len(tmplist):
                filtered[region] = tmplist
        return filtered
    

    def callDomains(self, model, window=2000000):
        
        self.hmm = model
        self.calDI(window=window)
        self.splitChrom(self.DIs)
        minDomains = self.minCore(self.regionDIs)
        self.domains = self.getDomainList(minDomains)
    
    

