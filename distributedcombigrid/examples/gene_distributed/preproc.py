from __future__ import print_function
#path to python interface
from future import standard_library
standard_library.install_aliases()
from builtins import str
from builtins import range

from configparser import SafeConfigParser
import collections
from subprocess import call

######## python interface stuff - copy/pasted from https://gitlab.lrz.de/sparse_grids/gene_python_interface_clean

import numpy as np
import itertools as it
import abc
import logging
import random
from argparse import ArgumentError

class ActiveSetBase(object):
    __metaclass__ = abc.ABCMeta
    def __init__(self):
        self.log = logging.getLogger(__name__)
        self.log.debug('Created ActiveSet Object')
       
    @abc.abstractmethod
    def getActiveSet(self, *args):
        pass

       
class ActiveSetTiltedPlane(ActiveSetBase):
    def __init__(self, planeNormalVector, anchorVector):
        super(ActiveSetTiltedPlane, self).__init__()
        self.n = np.array(planeNormalVector)
        self.anchor = np.array(anchorVector)
        self.setRoundingMethod(np.floor)
        self.lmax=np.ones(len(self.n))*20
    
    def setRoundingMethod(self, method):
        self.roundingMethod = method
    
    def getIntersectionPointsWithCoordinateAxes(self, currentDim):
        indexSet = range(currentDim) + range(currentDim + 1, len(self.n))
        ret= np.sum(self.n[indexSet] * (self.anchor[indexSet] - 1.)) / self.n[currentDim] + self.anchor[currentDim]
        return ret
    
    def getLmax(self):
        erg = []
        for i in range(len(self.n)):
            erg.append(int(np.ceil(self.getIntersectionPointsWithCoordinateAxes(i))))
        return tuple(erg)
        
    def getIntersectionPoints(self, currentDim):
        a = np.array(self.anchor)
        n = np.array(self.n)
        dim = currentDim
        lmax = self.getLmax()
        
        dimIndices = range(dim) + range(dim + 1, len(lmax))
        maxInd = np.array(lmax)[dimIndices]
        coords = [range(1, l + 1) for l in maxInd]
        tuples = list(it.product(*coords))
        
        erg = {}
        for i in range(len(tuples)):
            t = tuples[i]
            s = np.array(t)
            ind = tuple(s - np.ones(len(s), dtype=int))
            erg[t] = np.sum((a[dimIndices] - s) * n[dimIndices]) / (n[dim] * 1.0) + a[dim]
#             erg[t] = int(np.floor(erg[t]))
            
        ergSet = set()
        for e in erg:
            eList = list(e)
            listAll = eList[:dim] + [erg[e]] + eList[dim:]
            if (np.array(listAll) >= 0).all():
                ergSet.add(tuple(listAll))
        return ergSet

    def getActiveSet(self):
        erg = set()
        for i in range(len(self.n)):
            s = self.getIntersectionPoints(i)
            for grid in s:
                g = (self.roundingMethod(grid)).astype(int)
                if (g >= 1).all():
                    erg.add(tuple(g))
        return erg
            
class RootActiveSet(ActiveSetBase):
    def __init__(self,lmax,lmin):
        super(ActiveSetBase,self).__init__()
        self.lmin = lmin
        self.lmax = lmax

    def getActiveSet(self, *args):
        d = len(self.lmax)
        erg = set()
        for i in range(d):
            buf = list(self.lmin)
            buf[i]=self.lmax[i]
            erg.add(tuple(buf))
        erg.add(self.lmin)
        return erg

class ClassicDiagonalActiveSet(ActiveSetBase):
    def __init__(self,lmax,lmin=None,diagonalIndex=0):
        super(ClassicDiagonalActiveSet, self).__init__()
        self.lmax = lmax
        if lmin==None:
            self.lmin = [1 for i in range(len(self.lmax))]
        else:
            self.lmin = lmin
        self.setDiagonalIndex(diagonalIndex)
        
    def setDiagonalIndex(self,i):
        self.diagonalIndex = i

    def getMinLevelSum(self):
        return self.getLevelMinima().sum()
#         return self.getMaxLevel()-1+len(self.lmax)
    
    def getMaxLevel(self):
        return np.max(self.lmax)
    
    def getLevelMinima(self):
        maxInd = np.argmax(np.array(self.lmax)-np.array(self.lmin))
        lm = np.array(self.lmin)
        lm[maxInd]=self.lmax[maxInd]
        return lm
        
    
    def getActiveSet(self):
        listOfRanges=[range(0,self.lmax[i]+1) for i in range(len(self.lmax))]
        listOfAllGrids = list(it.product(*listOfRanges))
        s = set()
        levelSum = self.getMinLevelSum()+self.diagonalIndex
        for grid in listOfAllGrids:
            if (np.sum(grid)==levelSum and (np.array(grid)>=np.array(self.lmin)).all()):
                s.add(grid)
                
        return s
    
    def getExtraGrids(self,numExtraDiags):
        listOfRanges=[range(1,self.lmax[i]+1) for i in range(len(self.lmax))]
        listOfAllGrids = list(it.product(*listOfRanges))
        diff = np.array(self.lmax)-np.array(self.lmin)
        dim = len(self.lmax)-len(np.where(diff == 0)[0])
        lmin_eff = np.zeros(dim)
        lmax_eff = np.zeros(dim)
        j = 0
        for i in range(len(self.lmin)):
            if diff[i] != 0:
                lmin_eff[j] = self.lmin[i]
                lmax_eff[j] = self.lmax[i]
                j+=1

        s = set()
        for q in range(dim,dim+numExtraDiags):
            levelSum = np.array(lmax_eff).max()-np.array(lmin_eff).max()+np.array(self.lmin).sum()-q
#             print self.getMinLevelSum(),self.getMaxLevel().max(), np.array(self.lmin).max(), np.array(self.lmin).sum(),q,levelSum
            for grid in listOfAllGrids:
                if (np.sum(grid)==levelSum and (np.array(grid)>=np.array(self.lmin)).all()):
                    s.add(grid)
        return s
    
    def getEffLmin(self):
        diff = np.array(self.lmax)-np.array(self.lmin)
        activeInds = np.where(diff != 0)[0]
        lmin = np.array(tuple(self.lmin[i] for i in activeInds))
        lmax = np.array(tuple(self.lmax[i] for i in activeInds))

#         For lmin with extra 1's:
        erg = np.ones(len(self.lmin))
        ergTmp = lmax - np.min(lmax - lmin)*np.ones(len(lmax))
        for i in range(len(activeInds)):
            erg[activeInds[i]] = ergTmp[i]
        return  tuple(map(int,erg))

        # For lmin without extra 1's:
#         erg = lmax - np.min(lmax - lmin)*np.ones(len(lmax))
#         return  tuple(map(int,erg))
    
class ThinnedDiagonalActiveSet(ClassicDiagonalActiveSet):
    def __init__(self,lmax,thinningFactor=None,thinningNumber=None,lmin=None, diagonalIndex=0):
        super(ThinnedDiagonalActiveSet, self).__init__(lmax,lmin,diagonalIndex)
        if thinningFactor==None and thinningNumber==None:
            raise ArgumentError('Please fix a thinningNumber or thinningFactor')
        if thinningFactor!=None and thinningNumber!=None:
            raise ArgumentError('Please fix either thinningNumber or thinningFactor, not both')
        
        if thinningFactor!=None:
            self.thinningFactor = thinningFactor
            self.getActiveSet = self.getActiveSetThinningFactor
        else:
            self.thinningNumber = thinningNumber
            self.getActiveSet = self.getActiveSetFixedNumber
        
    
    def isSetValid(self,activeSet):
        ##get maxima
        it=iter(activeSet)
        maxInd=list(it.next())
        for s in activeSet:
            for i in range(len(s)):
                if s[i]>maxInd[i]:
                    maxInd[i]=s[i]
        for el,l in zip(maxInd,self.lmax):
            if el!=l:
                return False
        return True
    
    def removeNGrids(self,s,nGrids):
        if nGrids>len(s)-len(self.lmax):
            raise ArgumentError('too high thinning, cannot remove more grids than are there')
        counter=0
        while(1):
            counter+=1
            randomSample =set( random.sample(s,len(s)-nGrids))
            if self.isSetValid(randomSample):
                self.log.debug(str(counter)+' draws where required to find the active Set')
                return randomSample
    
    def getActiveSetFixedNumber(self):
        classicSet = ClassicDiagonalActiveSet.getActiveSet(self)
        return self.removeNGrids(classicSet, self.thinningNumber)
                
        
    def getActiveSetThinningFactor(self):
        classicSet = ClassicDiagonalActiveSet.getActiveSet(self)
        nGrids = len(classicSet)
        rounds = nGrids-int(round(nGrids*self.thinningFactor))
        return self.removeNGrids(classicSet, rounds)
                        

def showActiveSet3D(listOfsets):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    colors = it.cycle(['black','red','blue','green'])
    
    for sets,color in zip(listOfsets,colors):
        s = np.array(list(sets))
    
        ax.scatter(s[:,0],s[:,1],s[:,2],c=color)
    
    plt.show()


import abc,logging
#from reportlab.graphics.shapes import NotImplementedError
import numpy as np
import itertools as it
from collections import OrderedDict

class combinationSchemeBase(object):
    '''
    base class for combination scheme
    '''
    def __init__(self):
        self.log=logging.getLogger(__name__)
        
    def getMaxTupel(self):
        d = self.getCombinationDictionary()
        maxLevel = None
        for k in d.keys():
            if maxLevel==None:
                maxLevel=list(k)
            else:
                for i in range(self.getDim()):
                    if k[i]>maxLevel[i]:
                        maxLevel[i]=k[i]
        return tuple(maxLevel)

    @abc.abstractmethod
    def getCombinationDictionary(self):
        return

    @abc.abstractmethod
    def getDim(self):
        return
    

class combinationSchemeArbitrary(combinationSchemeBase):
    def __init__(self,activeSet):
        combinationSchemeBase.__init__(self)
        self.activeSet = activeSet
        self.updateDict()
        self.log.info('created combischeme with \n'+str(self.getCombinationDictionary()).replace(', (','\n('))
        
    def getDownSet(self,l):
        subs = map(lambda x:range(0, x + 1), l)
        downSet = it.product(*subs)
        return downSet
    
    def getUnifiedDownSet(self):
        s= set([])
        for a in self.activeSet:
            # print a,(self.getBoundedDownSet(a))
            s=s.union((self.getBoundedDownSet(a)))
            self.log.info(s)
        self.log.info(s)
        return s
    
    def getBoundedDownSet(self,l):
        lmin=self.getLmin()
        subs = map(lambda x,y:range(y, x + 1),l,lmin)
        downSet = it.product(*subs)
#         self.log.info(list(downSet))
        return list(downSet)
        
    def getLmin(self):
        erg=None
        for s in self.activeSet:
            if erg==None:
                erg=list(s)
            else:
                for i in range(len(s)):
                    if s[i]<erg[i]:
                        erg[i]=s[i]
        return tuple(erg)
    
    def updateDict(self):
        self.dictOfScheme={}
        for l in self.activeSet:
            self.dictOfScheme[l] = 1
            
        dictOfSubspaces={}
        
        for l in self.dictOfScheme:
            for subspace in self.getDownSet(l):
                if subspace in dictOfSubspaces:
                    dictOfSubspaces[subspace] += 1
                else:
                    dictOfSubspaces[subspace] = 1
                    
        # self.log.debug(str(dictOfSubspaces))

        # # remove subspaces which are too much
        while(set(dictOfSubspaces.values()) != set([1])):
            
            for subspace in dictOfSubspaces:
                currentCount = dictOfSubspaces[subspace]
                if currentCount != 1:
                    diff = currentCount - 1
     
                    if subspace in self.dictOfScheme:
                        
                        self.dictOfScheme[subspace] -= diff
                        if self.dictOfScheme[subspace] == 0:
                            del self.dictOfScheme[subspace]
                            
                    else:
                        self.dictOfScheme[subspace] = -diff
                    
                         
                    for l in self.getDownSet(subspace):
                        dictOfSubspaces[l] -= diff

        self.log.debug('created scheme '+str(self.dictOfScheme))
        
    def getCombinationDictionary(self):
        return self.dictOfScheme
    
    def getDim(self):
        return len(list(self.activeSet)[0])
    
    def removeSubspace(self,subspace):
        if subspace in self.dictOfScheme:
            del self.dictOfScheme[subspace]
            self.log.debug('subspace '+str(subspace)+' removed')
        else:
            raise AssertionError('The subspace does not exist in the scheme.')
    
    def removeZeroSubspaces(self):
        combiDict = self.getCombinationDictionary()
        for key, val in combiDict.items():
            if val == 0:
                del self.dictOfScheme[key]
    
    def getNearestNeighbors(self):
        dictOfNeighbors = {}
        keys = self.dictOfScheme.keys()
        keys_search = self.dictOfScheme.keys()
        while keys:
            key = keys.pop()
            d = np.inf
            for key_n in keys_search:
                d_new = sum(abs(np.array(key) - np.array(key_n)))
                if d_new < d and d_new > 0:
                    d = d_new
                    key_nearest = key_n
                    dictOfNeighbors[key] = key_n
            if key_nearest in keys:
                keys.remove(key_nearest)
        return dictOfNeighbors

    def getKNearestNeighbors(self,K):
        if K == 1:
            return self.getNearestNeighbors()
        dictOfNeighbors = {}
        keys = self.dictOfScheme.keys()
        while keys:
            key = keys.pop()
            keys_search = self.dictOfScheme.keys()
            for k in range(K):
                d = np.inf
                for key_n in keys_search:
                    d_new = sum(abs(np.array(key) - np.array(key_n)))
                    if d_new < d and d_new > 0:
                        d = d_new
                        key_nearest = key_n
#                if key_nearest in keys:
#                    keys.remove(key_nearest)
                if k == 0:
                    dictOfNeighbors[key] = [key_nearest]
                else:
                    dictOfNeighbors[key].append(key_nearest)
                keys_search.remove(key_nearest)

        return dictOfNeighbors

    def hierSubs(self,lmin,lmax):
        list_ranges = [range(lmin[0],lmax[0]+1),range(lmin[1],lmax[1]+1)]
        return it.product(*list_ranges)

    def keysPerLevel(self,lmin,lmax):
        dictOfKeysPerLevel = {}
        for key in self.hierSubs(lmin,lmax):
            l1 = sum(key)
            if not l1 in dictOfKeysPerLevel:
                dictOfKeysPerLevel[l1] = []
            dictOfKeysPerLevel[l1].append(key)
        return dictOfKeysPerLevel

    def getLargerSubspaces(self,grid,unified=True):
        if unified:
            downSet = self.getUnifiedDownSet()
        else:
            downSet = self.dictOfScheme.keys()
        largerSubspaces = []
        dim = len(grid)
        for level in downSet:
            isLarger = True
            for i in range(dim):
                if level[i] < grid[i]:
                    isLarger = False
                    break
            if isLarger:
                largerSubspaces.append(level)
        return largerSubspaces

class combinationSchemeFaultTolerant(combinationSchemeArbitrary):
    def __init__(self,activeSet):
        combinationSchemeArbitrary.__init__(self,activeSet.getActiveSet())
        self.activeSet = activeSet.getActiveSet()
        self.updateDict()
        self.numExtraLayers = 2
        self.extendSchemeExtraLayers(activeSet, self.numExtraLayers)
        self.tmpVec = []
        self.log.info('created combischeme with \n'+str(self.getCombinationDictionary()).replace(', (','\n('))

    def extendSchemeExtraLayers(self,activeSet,numExtraLayers):
        extraGrids = activeSet.getExtraGrids(numExtraLayers)
        for grid in extraGrids:
            self.dictOfScheme[grid] = 0
        
    def getHierarchicalCoefficients(self):
        downSet = self.getUnifiedDownSet()
        w = {}
        for grid in downSet:
            w[grid] = 0
            largerSubspaces = self.getLargerSubspaces(grid)
            for largerSpace in largerSubspaces:
                if largerSpace in self.dictOfScheme:
                    w[grid] += self.dictOfScheme[largerSpace] 
        return w
    
    def evalGCP(self,w):
        Q = 0
        for wi in w:
            Q+= 4**(-sum(wi))*w[wi]
        return Q
    
    def updateFaults(self,faults):
        maxLevelSum = sum(list(self.activeSet)[0]) #all grids in the active set have the same levelSum
        noRecomputationLayers = 2
        removeInds = []
        for i in range(len(faults)):
            if sum(faults[i]) <= maxLevelSum - noRecomputationLayers:
                removeInds.append(i)
        newFaults = list(faults)
        for i in range(len(removeInds)):
            del newFaults[removeInds[i]-i]
        return newFaults

    def divideFaults(self,faults):
        maxLevelSum = sum(list(self.activeSet)[0])
        qZeroFaults = []
        qOneFaults = []
        for fault in faults:
            l1 = sum(fault)
            if l1 == maxLevelSum:
                qZeroFaults.append(fault)
            else:
                qOneFaults.append(fault)
        return qZeroFaults, qOneFaults
    
    def partitionFaults(self,faults, w):
        isPartitioned = {}
        partitionedFaults = []
        if len(faults) == 1:
            isPartitioned[faults[0]] = True
            return isPartitioned
        
        neighborsDict = {}
        for fault in faults:
            isPartitioned[fault] = False
            nbrList = self.getNeighbors(fault, w)
            s = set()
            if len(nbrList) == 0:
                s.add(tuple(fault))
                neighborsDict[fault] = s
            else:
                nbrList.append(fault)
                nbrList = set(nbrList)
                neighborsDict[fault] = nbrList

        for fault in faults:
            if not isPartitioned[fault]:
                isPartitioned[fault] = True
                currentNbrs = neighborsDict[fault]
                self.findCurrentPartition(fault, currentNbrs,faults,
                                     neighborsDict,isPartitioned,partitionedFaults)
        return partitionedFaults
                
    def findCurrentPartition(self,fault,currentNbrs,faults,neighborsDict,isPartitioned,partitionedFaults):
        self.tmpVec.append(fault)
        for flt in faults:
            if not isPartitioned[flt]:
                interSet = currentNbrs.intersection(neighborsDict[flt])
                if len(interSet) != 0:
                    isPartitioned[flt] = True
                    unionSet = currentNbrs.union(neighborsDict[flt])
                    currentNbr = unionSet
                    self.findCurrentPartition(flt,currentNbr,faults,
                                            neighborsDict,isPartitioned,partitionedFaults)
        if len(self.tmpVec) != 0:
            partitionedFaults.append(self.tmpVec)
        self.tmpVec = []
        return partitionedFaults
    
    def generateCasesGCP(self,faults):
        w = OrderedDict()
        for grid in self.getUnifiedDownSet():
            w[grid] = 1
            
        qZeroFaults, qOneFaults = self.divideFaults(faults)
        
        for fault in qZeroFaults:
            del w[fault]
        
        partitionedFaults = self.partitionFaults(faults, w)
        
        allW = []
        allWDicts = []
        if len(faults) == 1:
            oneW = self.generateOneCaseGCP(faults[0], w)
            for case,numCase in zip(oneW,range(len(oneW))):
                wTmp = w.copy()
                for grid in case:
                    wTmp[grid] = oneW[numCase][grid]
                allWDicts.append(wTmp)
            return allWDicts
        else:
            for part in partitionedFaults:
                currentWs = []
                result = []
                onePartW = self.generateOnePartitionCasesGCP(part,w)
                for case in onePartW:
                    currentWs.append(case.values())
                if len(allW) == 0:
                    allW = currentWs
                else:
                    for pairs in it.product(allW,currentWs):
                        result.append(list(np.prod(np.array(pairs),axis=0)))
                    allW = result

        for case,numCase in zip(allW,range(len(allW))):
            wTmp = w.copy()
            for grid,j in zip(w,range(len(w))):
                wTmp[grid] = allW[numCase][j]
            allWDicts.append(wTmp)
        return allWDicts
    
    def generateOnePartitionCasesGCP(self,faults,w):
        qZeroFaults, qOneFaults = self.divideFaults(faults)

        allW = []
        if len(qOneFaults) == 0:
            allW.append(w.copy())
            return allW
        
        numFaults = len(qOneFaults)
        casesGCP = {}
        indexList = []
        for fault in qOneFaults:
            casesGCP[fault] = self.generateOneCaseGCP(fault, w)
            lenCase = len(casesGCP[fault])
            indexList.append(range(lenCase))

        allW = []
        canBeChanged = {}
        for wi in w:
            canBeChanged[wi] = True
        for index in it.product(*indexList):
            tmpW = w.copy()
            changedGrids = []
            for i,fault in zip(range(numFaults),qOneFaults):
                success = True
                currentW = casesGCP[fault][index[i]]
                for cw in currentW:
                    if currentW[cw] != tmpW[cw]:
                        if canBeChanged[cw]:
                            tmpW[cw] = currentW[cw]
                            canBeChanged[cw] = False
                            changedGrids.append(cw)
                        else:
                            success = False
                            break
                    else:
                        canBeChanged[cw] = False
                        changedGrids.append(cw)
                if not success:
                    break
            if success:
                allW.append(tmpW)
            for grid in changedGrids:
                canBeChanged[grid] = True
        
        return allW
    
    def getNeighbors(self,grid,allGrids):
        neighbors = []
        for k in range(self.getDim()):
            nbr = list(grid)
            nbr[k] +=1
            nbr = tuple(nbr)
            if nbr in allGrids:
                neighbors.append(nbr)
        return neighbors
    
    def generateOneCaseGCP(self,fault,w):

        cases = []
        neighbors = self.getNeighbors(fault, w)
        
        # case w_i = 0
        caseGCP = {}
        caseGCP[fault] = 0
        for nbr in neighbors:
            caseGCP[nbr] = 0
        cases.append(caseGCP)
        # cases k=1,...,d
        caseGCP = {}
        caseGCP[fault] = 1
        for nbr in neighbors:
            caseGCP[nbr] = 0
        for nbr in neighbors:
            caseGCP[nbr] = 1
            cases.append(caseGCP.copy())
            caseGCP[nbr] = 0      
        return cases
    
    def chooseBestGCP(self,allW):
        if len(allW) == 1:
            return allW[0]
        Q_max = -np.inf
        for w in allW:
            Q = self.evalGCP(w)
            if Q > Q_max:
                Q_max = Q
                W_max = w
        return W_max
        
    def buildGCPmatrix(self,w,faults):
        maxLevelSum = sum(list(self.activeSet)[0])
        qZeroFaults = []
        for fault in faults:
            l1 = sum(fault)
            if l1 == maxLevelSum:
                qZeroFaults.append(fault)
        indsDict = {}
        i = 0
        for w_i in w:
            indsDict[w_i] = i
            i+=1
        M = np.matlib.identity(len(w))
        dictOfSchemeFaults = set(self.getUnifiedDownSet()).difference(set(qZeroFaults))
        for w_i in w:
            largerGrids = set(self.getLargerSubspaces(w_i)).intersection(dictOfSchemeFaults)
            for grid in largerGrids :
                M[indsDict[w_i],indsDict[grid]] = 1
        return M
    
    def solveGCP(self,M,w):
        return np.linalg.solve(M, w)
    
    def recoverSchemeGCP(self,faults):
        
        # faults below layer two will e recalculated
        newFaults = self.updateFaults(faults)
        
        # if all faults occur below the second diagonal, return these
        if len(newFaults) == 0:
            return faults
        # generate all the w coeff. vectors that lead to 
        # a scheme ommitting the faults 
        allW = self.generateCasesGCP(newFaults)
        
        # out of those, choose the best one
        wBest = self.chooseBestGCP(allW)
        
        # calculate the c coeffs: first generate
        # the coefficient matrix
        M = self.buildGCPmatrix(wBest,faults)
        
        # solve for c (combination coeffs)
        c = self.solveGCP(M,wBest.values())
        
        # calculate new maxLevelSum
        maxLevelSum = 0
        self.activeSet = set()
        for grid,i in zip(wBest,range(len(c))):
            l1 = sum(grid)
            if l1 > maxLevelSum and c[i] != 0:
                maxLevelSum = l1
   
        # update the combischeme
        for grid,i in zip(wBest,range(len(c))):
            l1 = sum(grid)
            if l1==maxLevelSum and c[i] == 1:
                self.activeSet.add(grid)
        
        self.updateDict()
        
        # check that no q=0,1 faults appear on the scheme (this should always hold!)
        # return grids to be recomputed
        if len(set(self.dictOfScheme.keys()).intersection(set(newFaults))) == 0:
            return set(self.dictOfScheme.keys()).intersection(set(faults))
        else:
            raise ValueError
    
def showScheme3D(schemeobject,coords=[0,1,2]):    
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    scheme = schemeobject.getCombinationDictionary()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    col = ['blue','red','black','green']
    colors = it.cycle(col)
    
    for s in scheme:
        if scheme[s]>0:
            c = col[0]
        else:
            c = col[1]  
        ax.scatter(s[coords[0]],s[coords[1]],s[coords[2]],s=abs(scheme[s])*300.0,c=c)
    
    plt.show()

########################################################################################

#SGPP Directory set by Scons
SGPP_LIB="/home/pollinta/Desktop/combi/lib/sgpp"
print ("SGPP_LIB =", SGPP_LIB)
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))
TASK_LIB= str(dir_path) + "/lib"
print ("TASK_LIB =", TASK_LIB)
 
# some static definitions
spcfile = 'spaces.dat'
        
# read parameter file
parser = SafeConfigParser()
parser.read('ctparam')

config = collections.namedtuple('Config', 'lmin lmax ntimesteps_total dt_max ntimesteps_combi basename executable mpi startscript ngroup nprocs shat kymin lx numFaults combitime')

config.lmin = [int(x) for x in parser.get('ct','lmin').split()]
config.lmax = [int(x) for x in parser.get('ct','lmax').split()]
config.leval = [int(x) for x in parser.get('ct','leval').split()]
config.p = [int(x) for x in parser.get('ct','p').split()]
config.ntimesteps_total = int( parser.get('application', 'nsteps') )
config.dt_max = float( parser.get('application', 'dt') )
config.basename = parser.get('preproc','basename')
config.executable = parser.get('preproc','executable')
config.mpi = parser.get('preproc','mpi')
config.startscript = parser.get('preproc','startscript')
config.ngroup = int( parser.get('manager','ngroup') )
config.nprocs = int( parser.get('manager','nprocs') )
config.istep_omega = 10
#config.shat = parser.get('application','shat') 
config.kymin = float(parser.get('application','kymin')) 
config.lx = parser.get('application','lx') 
config.numspecies = parser.get('application','numspecies') 
config.local = parser.get('application','GENE_local') 
config.nonlinear = parser.get('application','GENE_nonlinear') 
config.numFaults = int( parser.get('faults', 'num_faults'))
if config.local == "T" :
    config.shat = parser.get('application','shat') 
config.combitime = float(parser.get('application','combitime'))
# if command line options given overwrite config options
'''
import sys
if( len( sys.argv ) > 1 ):
    assert( len( sys.argv ) == 5 )
    
    config.ntimesteps_total = int( sys.argv[1] )
    config.dt_max = float( sys.argv[2] )
    config.ntimesteps_combi = int( sys.argv[3] )
    config.ntimesteps_ev_calc = int( sys.argv[4] )
    
    #update config
    parser.set( 'ct', 'ntimesteps_total', str( config.ntimesteps_total ) )
    parser.set( 'ct', 'dt_max', str( config.dt_max ) )
    parser.set( 'ct', 'ntimesteps_combi', str( config.ntimesteps_combi ) )
    parser.set( 'ct', 'ntimesteps_ev_calc', str( config.ntimesteps_ev_calc ) )
    cfgfile = open('./ctparam','w')
    parser.write(cfgfile)
    cfgfile.close()
'''

# some sanity checks
lmin = config.lmin
lmax = config.lmax
leval = config.leval
print(lmax)
if( not (len(lmin) == len(lmax) == len(leval) ) ):
    raise ValueError('config: lmin,lmax,leval not of same size')

for i in range(0,len(lmin)):
    if lmin[i] < 0 or lmax[i] < 0 or leval[i] < 0:
        raise ValueError('config: lmin,lmax,leval may not be negative')
    
    if lmin[i] > lmax[i]:
        raise ValueError('config: lmin may not be larger than lmax') 
    
'''
if( not (config.ntimesteps_total%config.ntimesteps_combi == 0) ):
    raise ValueError('config: ntimesteps_total must be a multiple of ntimesteps_combi')
    
if( config.ntimesteps_combi > config.ntimesteps_ev_calc ):
    if( not (config.ntimesteps_combi%config.ntimesteps_ev_calc == 0) ):
        raise ValueError('config: ntimesteps_combi must be a multiple of ntimesteps_ev_calc')
else:
    if( not (config.ntimesteps_total%config.ntimesteps_combi == 0) ):
        raise ValueError('config: ntimesteps_ev_calc must be a multiple of ntimesteps_combi')
'''
            
# create base folder
call(["cp","-r","template",config.basename])

# set ct specific parameters in base folder
p = config.p
px = p[0]
py = p[1]
pz = p[2]
pv = p[3]
pw = p[4]
ps = p[5]

parfile = './' + config.basename + "/parameters"
with open(parfile,'r') as pfilein:
    pin = pfilein.read()    
pout = pin.replace('$ngroup',str(config.ngroup))
pout = pout.replace('$nprocs',str(config.nprocs))
pout = pout.replace('$istep_omega', str(config.istep_omega))
pout = pout.replace('$ps',str(ps),1)
pout = pout.replace('$pv',str(pv),1)
pout = pout.replace('$pw',str(pw),1)
pout = pout.replace('$px',str(px),1)
pout = pout.replace('$py',str(py),1)
pout = pout.replace('$pz',str(pz),1)
with open(parfile,'w') as pfileout:
    pfileout.write(pout)

# create combischeme
# note that the level vector in the python vector is stored in reverse order
lminp = lmin[::-1]
lmaxp = lmax[::-1]
if(config.numFaults == 0):
    activeSet = ClassicDiagonalActiveSet(lmaxp,lminp)
    scheme = combinationSchemeArbitrary(activeSet.getActiveSet())
else:
    factory = ClassicDiagonalActiveSet(lmaxp,lminp,0)
    activeSet = factory.getActiveSet()
    scheme = combinationSchemeFaultTolerant(factory)
# detect number of simulation steps
#nsteps = config.ntimesteps_combi if config.ntimesteps_combi <= config.ntimesteps_ev_calc else config.ntimesteps_ev_calc

# loop over scheme
id = 0
spaces = ''
print(len(scheme.getCombinationDictionary()))
print(scheme.getCombinationDictionary())

for l in scheme.getCombinationDictionary():
    # note that the level vector in the python vector is stored in reverse order
    print(l)
    l0 = l[5]
    l1 = l[4]
    l2 = l[3]
    l3 = l[2]
    l4 = l[1]
    l5 = l[0]
     
    # append subspaces entry
    spaces  += str(id) + " " +  str(l0) + ' ' + str(l1) + ' ' + str(l2) \
            + ' ' + str(l3) + ' ' + str(l4) + ' ' + str(l5) + ' ' \
            + str(scheme.getCombinationDictionary()[l]) + "\n"
             
    # copy template folder
    call(["cp","-r","./template",'./' + config.basename + str(id)])
    
    # set ct specific parameters
    parfile = './' + config.basename + str(id) + "/parameters"
    with open(parfile,'r') as pfilein:
        pin = pfilein.read()
    
    pout = pin.replace('$nx0',str(2**l0),1)
    if config.nonlinear == "T" :
        pout = pout.replace('$nky0',str(2**l1),1)
    else:
        pout = pout.replace('$nky0',str(2**l1-1),1)
    pout = pout.replace('$nz0',str(2**l2),1)
    pout = pout.replace('$nv0',str(2**l3),1)
    pout = pout.replace('$nw0',str(2**l4),1)
    pout = pout.replace('$nspec',str(config.numspecies),1)
    pout = pout.replace('$GENE_local',str(config.local),1)
    pout = pout.replace('$GENE_nonlinear',str(config.nonlinear),1)

    pout = pout.replace('$ps',str(ps),1)
    pout = pout.replace('$pv',str(pv),1)
    pout = pout.replace('$pw',str(pw),1)
    pout = pout.replace('$px',str(px),1)
    pout = pout.replace('$py',str(py),1)
    pout = pout.replace('$pz',str(pz),1)

    pout = pout.replace('$ngroup',str(config.ngroup))
    pout = pout.replace('$nprocs',str(config.nprocs))
    
    pout = pout.replace('$ntimesteps_combi', str(config.ntimesteps_total))
    pout = pout.replace('$istep_omega', str(config.istep_omega))
    pout = pout.replace('$dt_max', str(config.dt_max))
    if config.local == "T" :
        pout = pout.replace('$shat', str(config.shat))
    print (2**(lmax[1]-l1)*config.kymin)
    pout = pout.replace('$kymin', str(2**(lmax[1]-l1)*config.kymin))
    pout = pout.replace('$lx', str(config.lx))

    pout = pout.replace('$read_cp','F')
    pout = pout.replace('$write_cp','T')
    pout = pout.replace('$combitime',str(config.combitime)) 
    with open(parfile,'w') as pfileout:
        pfileout.write(pout)
    
    # update counter
    id += 1
                            
# print spaces file
with open('./' + config.basename + '/' + spcfile,'w') as sfile:
    sfile.write(spaces)

# link manager executable to base folder
call(["ln","-s","../manager",'./' + config.basename + '/manager'])

# copy param file to base folder 
call(["cp","./ctparam",'./' + config.basename + '/'])

# create start script in base folder
scmd = "export LD_LIBRARY_PATH=" + SGPP_LIB + ":" + TASK_LIB + ":/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH\n"
#scmd = "source xprtld.bat\n"
scmd += config.mpi
scmd += " -n " + str(config.nprocs*config.ngroup) + ' ' + config.executable
scmd += " : "
scmd += " -n 1" + " ./manager"

with open( config.basename + "/" + config.startscript,'w') as sfile:
    sfile.write(scmd)
