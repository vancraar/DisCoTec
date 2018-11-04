source ../../../setenv.sh

module unload python-site/2.7
module load python-site/3.6

#env
#python -c "import sys; print(sys.path)"
export PYTHONPATH=/lustre/cray/ws8/ws/ipvpolli-EXAHD/gene_python_interface_clean/src:/lustre/cray/ws8/ws/ipvpolli-EXAHD/gene_python_interface_clean/src/tools:$PYTHONPATH

export LD_LIBRARY_PATH=/lustre/cray/ws8/ws/ipvpolli-EXAHD/combi/glpk/lib:/lustre/cray/ws8/ws/ipvpolli-EXAHD/combi/lib/sgpp:/lustre/cray/ws8/ws/ipvpolli-EXAHD/combi/distributedcombigrid/examples/gene_distributed/lib:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

rm -r ginstance*
rm out/*
echo "done removing"
python3 preproc.py
echo "ran python3 preproc.py"
cd ginstance
source start_hazelhen.sh
#source start.bat
cd ..
