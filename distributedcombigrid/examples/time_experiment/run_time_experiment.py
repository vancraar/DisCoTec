import os
from subprocess import call
import math
import sys
from ConfigParser import SafeConfigParser

mpicmd = "mpirun.mpich"
procpernode = 16

def setctparam( parfile, parGroup, parName, value ):
    # read parameter file
    parser = SafeConfigParser()
    parser.read( parfile )
    
     #update config
    parser.set( parGroup, parName, str(value) )
    
    cfgfile = open( parfile,'w')
    parser.write(cfgfile)
    cfgfile.close()
    

def getctparam( parfile, parGroup, parName ):
    # read parameter file
    parser = SafeConfigParser()
    parser.read( parfile )
    
    return parser.get( parGroup, parName )

def create_ref_experiment( basename, parfile ):
    # compute total number of processes
    ngroups = getctparam( parfile, 'manager', 'ngroup')
    nprocs = getctparam( parfile, 'manager', 'nprocs' )
    numproc = int(ngroups) * int(nprocs) + 1
    
    # make file name   
    filename = basename + "_ref"
    
    dirname = filename.strip('_')
    
    print dirname
    
    # create dir and go into
    path = "./" + dirname 
    os.mkdir( path )
    os.chdir( path )
    
    # create link to executable
    call( ["ln","-s","../" + execname,execname] )
    
    #write ctparam
    call( ["cp", "../ctparam", "."] )
    lmax = getctparam( "./ctparam", 'ct', 'lmax' )
    setctparam( "./ctparam", 'ct', 'lmin', lmax )
    setctparam( "./ctparam", 'faults', 'num_faults', 0 )
    
    #copy run.sh
    call( ["cp", "../run.sh", "."] )
    
    #run experiment
    call( ['./run.sh'] )
    
       # leave dir
    os.chdir( '..' )
    
    # return path to solution file
    return filename + "/solution.fg"
    
    
def create_ct_experiment( basename, parfile, pargroup, parname, value ):
    # compute total number of processes
    ngroups = getctparam( parfile, 'manager', 'ngroup')
    nprocs = getctparam( parfile, 'manager', 'nprocs' )
    numproc = int(ngroups) * int(nprocs) + 1
    
    # make file name   
    filename = basename + "_" + parname + "_" + value
    
    dirname = filename.strip('_')
    
    print dirname
    
    # create dir and go into
    path = "./" + dirname 
    os.mkdir( path )
    os.chdir( path )
    
    # create link to executable
    call( ["ln","-s","../" + execname,execname] )
    
    #write ctparam
    call( ["cp", "../ctparam", "."] )
    setctparam( "./ctparam", pargroup, parname, value )
    
    #copy run.sh
    call( ["cp", "../run.sh", "."] )
    
    #run experiment
    call( ['./run.sh'] )
    
       # leave dir
    os.chdir( '..' )
    
    # return path to solution file
    return filename + "/solution.fg"
    
    
if __name__ == "__main__":
    if len(sys.argv) < 4:
        print( "use: ./run_time_experiment basename executable ctparam" )
        exit(0)
    
    #read filenames from args
    basename = sys.argv[1]
    execname = sys.argv[2] 
    parfile = sys.argv[3] 
    
    sol_file_ref = create_ref_experiment( basename, parfile )

    pargroup = getctparam( "./ctparam", 'experiment', 'pargroup' )
    parname = getctparam( "./ctparam", 'experiment', 'parname' )
    values = getctparam( "./ctparam", 'experiment', 'values' )
    
    for val in values.split():
        sol_file_exp = create_ct_experiment( basename, parfile, pargroup, parname, val )
        
        call( ['./errorCalc',sol_file_ref,sol_file_exp,'err.out'] )
        
        