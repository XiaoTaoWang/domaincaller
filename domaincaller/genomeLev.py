import cooler, logging
from domaincaller.chromLev import Chrom
import numpy as np
from pomegranate import NormalDistribution, HiddenMarkovModel, GeneralMixtureModel, State

log = logging.getLogger(__name__)

class Genome(object):

    def __init__(self, uri, balance_type='weight', window=2000000, exclude=[]):
        
        lib = cooler.Cooler(uri)
        res = lib.binsize
        seqs = []
        log.debug('Calculating DIs for each chromosome ...')
        self.chroms = []
        for c in lib.chromnames:
            if c in exclude:
                continue
            log.debug('Chrom {0} ...'.format(c))
            self.chroms.append(c)
            tdata = lib.matrix(balance=balance_type, sparse=True).fetch(c).tocsr()
            work = Chrom(c, res, tdata)
            work.calDI(window=window)
            work.splitChrom(work.DIs)
            for r in work.regionDIs:
                withzeros = work.regionDIs[r]
                nozeros = withzeros[withzeros!=0]
                if nozeros.size > 20:
                    seqs.append(nozeros)
        
        self.training_data = seqs
        self.hic = lib
    
    def oriHMMParams(self, numdists=3):
        """
        Set initial parameters for the Hidden Markov Model (HMM).
        
        """
        # GMM emissions
        # 3 Hidden States:
        # 0--downstream, 1--no bias, 2--upstream
        if numdists==1:
            dists = [NormalDistribution(-2.5, 7.5), NormalDistribution(0, 7.5), NormalDistribution(2.5, 7.5)]
        else:
            var = 7.5 / (numdists - 1)
            means = [[], [], []]
            for i in range(numdists):
                means[0].append(i * 7.5 / ( numdists - 1 ) + 2.5)
                means[1].append(i * 7.5 * (-1)**i / ( numdists - 1 ))
                means[2].append(-i * 7.5 / ( numdists - 1 ) - 2.5)

            dists = []
            for i, m in enumerate(means):
                tmp = []
                for j in m:
                    tmp.append(NormalDistribution(j, var))
                mixture = GeneralMixtureModel(tmp)
                dists.append(mixture)

        # transition matrix
        A = [[0.34, 0.33, 0.33],
            [0.33, 0.34, 0.33],
            [0.33, 0.33, 0.34]]
        starts = np.ones(3) / 3

        hmm = HiddenMarkovModel.from_matrix(A, dists, starts, state_names=['0', '1', '2'], name='mixture{0}'.format(numdists))
        
        return hmm
        

