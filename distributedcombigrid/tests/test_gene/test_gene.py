#!/usr/bin/env python3
# this is a script to semi-automatedly compare the results obtained by gene-only with 
# the results we get for a combi-scheme with just one grid, that is the same as before.
#
# to achieve that, one must first make sure that all the prerequisites are built and  
# symlinked properly. That entails: an independent GENE version with COMBI=yes and WITH_COMBI=gene (link into a folder prob_compare_ct),
# another GENE version to be used with the combi framework (WITH_COMBI=framework) (link into template/), 
# the distributedcombigrid module, and the gene_distributed example (manager and errorCalc should be linked).

from subprocess import call
import os
import fileinput
import re
from configparser import SafeConfigParser
import collections
import glob

#https://stackoverflow.com/questions/431684/how-do-i-change-directory-cd-in-python/13197763#13197763
class cd:
    """Context manager for changing the current working directory"""
    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)


# read parameter file
parser = SafeConfigParser()
parser.read('ctparam')

config = collections.namedtuple('Config', 'nsteps ncombi')
config.nsteps = int( parser.get('application', 'nsteps') )
config.ncombi = int( parser.get('ct', 'ncombi') )

#make sure we're using an unchanged parameters file (may have been overwritten by combi framework)
call(["git", "checkout", "ginstance0/parameters"])
#multiply ntimesteps with ncombi
with fileinput.FileInput("ginstance0/parameters", inplace=True, backup='.bak') as file:
    for line in file:
        if line.startswith("ntimesteps"):
            line = "ntimesteps = " + str(config.nsteps*config.ncombi) + '\n'
        print(line, end='')

#run the pure-gene problem in its folder
with cd ("./prob_compare_ct"):
    #call gene no matter how the executable is called
    gene_files = glob.glob("./gene_*")
    # print(gene_files[0])
    call(["mpirun.mpich","-n","4",gene_files[0]])

#call combi framework
call(["./run.sh"])

#compare results with errorCalc
print('call(["./errorCalc","fg","plot2.dat0","prob_compare_ct/checkpoint","errorCalcTestGene","auto"])')
call(["./errorCalc","fg","plot2.dat0","prob_compare_ct/checkpoint","errorCalcTestGene","auto"])

# on the current setup, we see an error of 3.37935e-15 only.
