source setenv.sh

scons -c FC=gfort && rm -r .scon*

./compile.sh

cd ./distributedcombigrid/examples/gene_distributed
make clean
make -j8
cd -

cd gene-dev-exahd
make distclean
./compile.sh
cd -
