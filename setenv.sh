umask 002

#module load fftw/3.3.4.7
#module load cray-petsc-complex/3.7.6.2
#module load numlib/hlrs_SLEPc/3.7.4

module load python-site/2.7

#module switch PrgEnv-cray PrgEnv-intel
#module switch PrgEnv-gnu PrgEnv-intel
module switch PrgEnv-cray PrgEnv-gnu
module switch PrgEnv-intel PrgEnv-gnu
#module load gcc

module load cray-fftw

module load cray-petsc-complex/3.8.4.0
export SLEPC_DIR="/lustre/cray/ws8/ws/ipvpolli-EXAHD/soft/slepc/3.8.3/INTEL/"

#set boost last, such that version 1.54 is not loaded again
module load tools/boost/1.66.0
export LIBRARY_PATH=/opt/hlrs/tools/boost/1.66.0/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/hlrs/tools/boost/1.66.0/lib:$LIBRARY_PATH

export CRAYPE_LINK_TYPE=dynamic
#export MACHINE=hazelhen

module list

#export
