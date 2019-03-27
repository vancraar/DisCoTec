umask 002

#module load fftw/3.3.4.7
#module load cray-petsc-complex/3.7.6.2
#module load numlib/hlrs_SLEPc/3.7.4

module load python-site/2.7

mype=$(echo $PE_ENV| tr [:upper:] [:lower:])
module switch PrgEnv-$mype PrgEnv-intel
#module switch PrgEnv-$mype PrgEnv-gnu
#module load gcc

module load cray-fftw
module load cray-hdf5-parallel
module load cray-petsc-complex-64/3.8.4.0
module load cray-tpsl-64

SCRIPT=$(readlink -f "$0")
# Absolute path this script is in
SCRIPTPATH=$(dirname "$SCRIPT")
export SLEPC_DIR=$(readlink -f "$SCRIPTPATH/../soft/slepc/3.8.3/INTEL-64")
echo "slepc in $SLEPC_DIR"
export SLEPC_LIB="-Wl,-rpath,$SLEPC_DIR/lib -L$SLEPC_DIR/lib -lslepc  -L/opt/cray/pe/petsc/3.8.4.0/complex/INTEL/16.0/haswell/lib -L/opt/cray/pe/hdf5-parallel/1.10.2.0/INTEL/16.0/lib -lcraypetsc_intel_complex"
module unload perftools-base/7.0.5
module load perftools-base
#module load perftools-lite
#module load perftools
#module load perftools-lite-events

#set boost last, such that version 1.54 is not loaded again
module load tools/boost/1.66.0
module switch PrgEnv-gnu PrgEnv-intel

export BOOST_DIR=$BOOST_ROOT
#export BOOST_DIR=/lustre/cray/ws8/ws/ipvpolli-EXAHD/boost_1_68_0_intel
export BOOST_LIB=$BOOST_DIR/lib
export CRAYPE_LINK_TYPE=dynamic
#export MACHINE=hazelhen

# advised by the Cray optimization guy: Good Cray MPI environment settings USE THEM !!!!!
# General setup information :
#export MPICH_VERSION_DISPLAY=1
#export MPICH_ENV_DISPLAY=1
#export MPICH_CPUMASK_DISPLAY=1 # uncomment if output to large.
#export MPICH_RANK_REORDER_DISPLAY=1 # 

# If using MPI-IO (parallel NetCDF or parallel HDF5) :
#export MPICH_MPIIO_AGGREGATOR_PLACEMENT_DISPLAY=1
#export MPICH_MPIIO_HINTS_DISPLAY=1
#export MPICH_MPIIO_TIMERS=1 
#export MPICH_MPIIO_STATS=1 # or 2

module list

#export
