#make
module load python-site/3.6
module unload python-site/2.7
umask 002

module load tools/boost/1.66.0
module load cray-fftw
module load gcc
export CRAYPE_LINK_TYPE=dynamic
export LD_LIBRARY_PATH=/lustre/cray/ws8/ws/ipvpolli-EXAHD/combi/glpk/lib:/lustre/cray/ws8/ws/ipvpolli-EXAHD/combi/lib/sgpp/:/lustre/cray/ws8/ws/ipvpolli-EXAHD/combi/distributedcombi/examples/gene_distributed/lib:/opt/hlrs/tools/boost/1.66.0/lib:$LD_LIBRARY_PATH
#module list
#env
#python -c "import sys; print(sys.path)"
rm -r ginstance*
rm out/*
echo "done removing"
python3 preproc.py
echo "ran python3 preproc.py"
cd ginstance
source start.bat
cd ..
