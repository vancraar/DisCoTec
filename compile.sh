source setenv.sh

#export LD_LIBRARY_PATH=/lustre/cray/ws8/ws/ipvpolli-EXAHD/combi/lib/sgpp:/lustre/cray/ws8/ws/ipvpolli-EXAHD/hlrs-tools/boost_1_58_0/stage/lib/:$LD_LIBRARY_PATH

echo $LIBRARY_PATH

scons -j 16 SG_ALL=0 SG_DISTRIBUTEDCOMBIGRID=1 VERBOSE=1 RUN_BOOST_TESTS=0 COMPILE_BOOST_TESTS=0 RUN_CPPLINT=0 BUILD_STATICLIB=0 CC=cc FC=ftn CXX=CC OPT=1 TIMING=0 LIBPATH=/opt/hlrs/tools/boost/1.66.0/lib #DEBUG_OUTPUT=1


