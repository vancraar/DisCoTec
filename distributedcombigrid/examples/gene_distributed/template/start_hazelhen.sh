export LD_LIBRARY_PATH=/lustre/cray/ws8/ws/ipvpolli-EXAHD/combi/glpk/lib:/lustre/cray/ws8/ws/ipvpolli-EXAHD/combi/lib/sgpp:/lustre/cray/ws8/ws/ipvpolli-EXAHD/combi/distributedcombigrid/examples/gene_distributed/lib:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
aprun -n 1366 -N 24 ./gene_new_machine :  -n 1 ./manager
