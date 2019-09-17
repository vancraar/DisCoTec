What is this project about?
---------------------------
This project contains the third level implementation of the
__distributed combigrid module__. 

Important:
----------
Please do use feature branches in development.

Guidelines
---------
*  Please develop new features in a new branch and then create a merge request 
to notify other users about the changes. So everyone can check whether there are 
side-effects to other branches.
* Of course, you are still very welcome to directly fix obvious bugs in the 
master branch.
* Before merging new features to the master branch, please make sure that they
are sufficiently commented. 
* Although the distributed combigrid module is independent of the other modules
in SG++, it will remain a part of this project. To ensure compability please
make sure that you only change files in the distributedcombigrid folder. 
* In the future the automated testing and code style checking on commits that is 
done in SG++ will also be done for this project.

Installation instructions:
--------------------------
Use the provided compile.sh for compilation. The script also contains compilation
commands for the most common scenarios.
Or run
```
scons -j4 SG_JAVA=0 COMPILE_BOOST_TESTS=1 SG_ALL=0 SG_DISTRIBUTEDCOMBIGRID=1 VERBOSE=1 RUN_BOOST_TESTS=0 RUN_CPPLINT=0 BUILD_STATICLIB=0 COMPILER=mpich OPT=1
``` 
and for debugging, add
```
CPPFLAGS=-g
``` 

There are gene versions as submodules: a linear one in the gene_mgr folder, and 
a nonlinear one in gene-non-linear. To get them, you need access to their 
respective repos. Then go into the folder and

```
git submodule init
git submodule update
```
and then you just hope you can `make` it by adapting the local library flags ;)


Building the SimpleAmqpClient C++ Library
-----------------------------------------

The library building instructions are described in the official git repo
(https://github.com/alanxz/SimpleAmqpClient). However we need to make some
adjustments since installing the dependency rabbitmq-c
(https://github.com/alanxz/rabbitmq-c) systemwide is not possible.

The Building steps are the following:

1. Clone and build rabbitmq-c as described in the git repo. This requires
   Openssl librarys (libopenssl-dev for ubuntu systems))
2. Clone the SimpleAmqpClient Library, create a build folder in the main
   directory and enter it. (`mkdir build && cd build`)
4. Run `cmake ..` once. This will throw errors since it cannot find the
   rabbitmq-c (maybe also boost libs and headers (note: boost-chrono library is
   a seperate package under ubuntu libboost-chrono-dev)). If you don't want to
   or can't (i.e. on HLRS) use the systemwide installation, run cmake in the
   build directory as follows:
   `BOOST_ROOT=/path/to/boost/installation/ cmake ..`
3. Use `ccmake ..` to set the corrseponding paths
   (use c to save your configuration). For rabbitmq-c the includes are under
   rabbitmq-c/librabbitmq and the library is under
   rabbitmq-c/build/librabbitmq/librabbitmq.so
4. Use `cmake ..` again, this time there should not be any error.
5. Use `make` to build the library.

4. (Optional) The tests are not build by default. You have to clone gtest from
   https://github.com/abseil/googletest/tree/ec44c6c1675c25b9827aacd08c02433cccde7780
   into SimpleAmqpClient/third-party/ and pass `-DENABLE_TESTING=ON` to cmake.
   But I haven't yet figured out how to successfully run the tests.

On Hazel Hen
------------

# Prerequisites

* load modules: PrgEnv-gnu, scons, python 2.7

* Check loaded modules with:
  `module list`

* E.g. to change the default cray compiler to gnu:
  `module switch PrgEnv-cray PrgEnv-gnu`

* Allow dynamic linking:
  `export CRAYPE_LINK_TYPE=dynamic`

# Compilation

Unfortunately, the boost module on Hazel Hen does not work with our code any more.
Hence, you have to install boost yourself, e.g. use version >= 1.58
Do not forget to set boost paths in SConfigure, i.e. BOOST_INCLUDE_PATH, 
BOOST_LIBRARY_PATH and during compilation of the SimpleAmqpClient library.

Compile by first adapt the compile.sh script to use the HLRS specific command.

(the linking of the boost tests might fail. however this is not a problem, the
sg++ libraries should be there)

# Execution

Let's assume we want to run the example under
combi/distributedcombigrid/examples/distributed_third_level/ and distribute our
combischeme to HLRS and helium, while the third-level-manager is running on the
HLRS login node eslogin002 at port 9999. The following steps are necessary:

* Since data connections to the HLRS are not possible without using ssh tunnels,
  we set them up in advance. Run
  `ssh -L  9999:localhost:9999 username@eslogin002.hww.de` from helium to login
  into HLRS while creating an ssh tunnel.

* Adjust the parameter files on HLRS and helium to fit the simulation.
  Use hostname=eslogin002 on HLRS and hostname=localhost on helium. Set
  dataport=9999 on both systems.

* Run the third-level-manger

* Start the simulation. The example directory holds a separate run file for the HLRS. 
