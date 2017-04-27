/*
 * MPISetup.cpp
 *
 *  Created on: Jan 23, 2013
 *      Author: mh
 *
 *  Partially copied from the pe Physics Engine class MPISystem
 */

#include "sgpp/distributedcombigrid/mpi/MPISystem.hpp"
#include "sgpp/distributedcombigrid/utils/Stats.hpp"
#include "sgpp/distributedcombigrid/manager/ProcessGroupManager.hpp"

#include <iostream>

namespace{
  using namespace combigrid;

  std::string getMinMaxAvg( RankType r, int size, std::string timerName, bool isTimer,
                            MPI_Comm comm ){
    double value;
/*    if( isTimer )
      value = theStatsContainer()->getDuration( timerName );
    else
      value = theStatsContainer()->getValue( timerName );
*/
    double min, max, sum;
    MPI_Reduce( &value, &min, 1, MPI_DOUBLE, MPI_MIN, r, comm );
    MPI_Reduce( &value, &max, 1, MPI_DOUBLE, MPI_MAX, r, comm );
    MPI_Reduce( &value, &sum, 1, MPI_DOUBLE, MPI_SUM, r, comm );

    double avg = sum / static_cast<double>( size );

    std::stringstream ss;
    ss << timerName << "\t\t" << min << "\t" << max << "\t" << avg;

    return ss.str();
  }
} // anonymous namespace


namespace combigrid {



/*!\brief Constructor for the MPISystem class.
 //
 // \exception std::runtime_error MPI was not initialized
 //
 // Constructor for the MPI System. The default global communicator and local communicator is MPI_COMM_WORLD.
 // The total number of MPI processes and the rank of the MPI process in is determined from
 // the communicator. The default root process has rank zero.
 */
MPISystem::MPISystem() :
    initialized_(false),
    worldComm_(MPI_COMM_NULL),
    globalComm_(MPI_COMM_NULL),
    localComm_(MPI_COMM_NULL),
    globalReduceComm_(MPI_COMM_NULL),
    worldCommFT_(nullptr),
    globalCommFT_(nullptr),
    spareCommFT_(nullptr),
    localCommFT_(nullptr),
    globalReduceCommFT_(nullptr),
    worldRank_(MPI_UNDEFINED),
    globalRank_(MPI_UNDEFINED),
    localRank_(MPI_UNDEFINED),
    globalReduceRank_(MPI_UNDEFINED),
    managerRank_(MPI_UNDEFINED),
    managerRankWorld_(MPI_UNDEFINED),
    masterRank_(MPI_UNDEFINED),
    reusableRanks_(std::vector<RankType>(0))
{
  // check if MPI was initialized (e.g. by MPI_Init or similar)
  int mpiInitialized(0);
  MPI_Initialized(&mpiInitialized);
  assert( mpiInitialized && "MPI is not initialized! Call MPI_Init first." );
}

/*!\brief Destructor for the MPISystem class.
 */
MPISystem::~MPISystem() {
  // todo: the fault tolerant communicator are initialized with new -> delete
}


void MPISystem::init( size_t ngroup, size_t nprocs ){
  assert( !initialized_ && "MPISystem already initialized!" );

  ngroup_ = ngroup;
  nprocs_ = nprocs;

  worldComm_ = MPI_COMM_WORLD;

  /* init worldComm
   * the manager has highest rank here
   */
  int worldSize;
  MPI_Comm_size( worldComm_, &worldSize );
  assert( worldSize == int(ngroup_ * nprocs_ + 1) );

  MPI_Comm_rank( worldComm_, &worldRank_ );
  managerRankWorld_ = worldSize - 1;
  managerRankFT_ = worldSize - 1;

  //managerRankWorld_ = 0;
  if( ENABLE_FT ){
    worldCommFT_ = simft::Sim_FT_MPI_COMM_WORLD;
    spareCommFT_ = simft::Sim_FT_MPI_COMM_WORLD;
  }

  /* init localComm
   * lcomm is the local communicator of its own process group for each worker process.
   * for manager, lcomm is a group which contains only manager process and can be ignored
   */
  initLocalComm();

  /* create global communicator which contains only the manager and the master
   * process of each process group
   * the master processes of the process groups are the processes which have
   * rank 0 in lcomm
   * this communicator is used for communication between master processes of the
   * process groups and the manager and the master processes to each other
   */
  initGlobalComm();

  initGlobalReduceCommm();

  initialized_ = true;
}


/*  here the local communicator has already been created by the application */
void MPISystem::init( size_t ngroup, size_t nprocs, CommunicatorType lcomm ){
  assert( !initialized_ && "MPISystem already initialized!" );

  ngroup_ = ngroup;
  nprocs_ = nprocs;

  worldComm_ = MPI_COMM_WORLD;

  /* init worldComm
   * the manager has highest rank here
   */
  int worldSize;
  MPI_Comm_size( worldComm_, &worldSize );
  assert( worldSize == int(ngroup_ * nprocs_ + 1) );

  MPI_Comm_rank( worldComm_, &worldRank_ );
  managerRankWorld_ = worldSize - 1;
  managerRankFT_ = worldSize - 1;

  if( ENABLE_FT ){
      worldCommFT_ = simft::Sim_FT_MPI_COMM_WORLD;
      spareCommFT_ = simft::Sim_FT_MPI_COMM_WORLD;
  }
  //std::cout << "Global rank of root is" << worldCommFT_->Root_Rank << "\n";
  //worldCommFT_->Root_Rank = worldSize - 1;
  /* init localComm
   * lcomm is the local communicator of its own process group for each worker process.
   * for manager, lcomm is a group which contains only manager process and can be ignored
   */
  // manager is not supposed to have a localComm
  if( worldRank_ == managerRankWorld_ )
    localComm_ = MPI_COMM_NULL;
  else{
    localComm_ = lcomm;

    // todo: think through which side effects changing the master rank would have
    // in principle this does not have to be 0
    const int masterRank = 0;

    int localSize;
    MPI_Comm_size( localComm_, &localSize );
    assert( masterRank < localSize );

    masterRank_ = masterRank;

    MPI_Comm_rank( localComm_, &localRank_ );
  }

   if(ENABLE_FT){
     if( localComm_ != MPI_COMM_NULL){
         createCommFT( &localCommFT_, localComm_ );
     }
   }


  /* create global communicator which contains only the manager and the master
   * process of each process group
   * the master processes of the process groups are the processes which have
   * rank 0 in lcomm
   * this communicator is used for communication between master processes of the
   * process groups and the manager and the master processes to each other
   */
  initGlobalComm();

  initGlobalReduceCommm();

  initialized_ = true;
}


void MPISystem::initLocalComm(){
  int color = worldRank_ / int(nprocs_);
  int key = worldRank_ - color * int(nprocs_);
  MPI_Comm_split( worldComm_, color, key, &localComm_ );

  /* set group number in Stats. this is necessary for postprocessing */
  Stats::setAttribute("group", std::to_string(color));

  // manager is not supposed to have a localComm
  if( worldRank_ == managerRankWorld_ )
    localComm_ = MPI_COMM_NULL;
  else{
    // todo: think through which side effects changing the master rank would have
    // in principle this does not have to be 0
    const int masterRank = 0;

    int localSize;
    MPI_Comm_size( localComm_, &localSize );
    assert( masterRank < localSize );

    masterRank_ = masterRank;

    MPI_Comm_rank( localComm_, &localRank_ );
  }


  if(ENABLE_FT){
    if( localComm_ != MPI_COMM_NULL){
      createCommFT( &localCommFT_, localComm_ );
    }
  }
}


void MPISystem::initGlobalComm(){
  MPI_Group worldGroup;
  MPI_Comm_group( worldComm_, &worldGroup);

  std::vector<int> ranks( ngroup_ + 1 );
  for (size_t i = 0; i < ngroup_; i++) {
    ranks[i] = int( i * nprocs_ );
  }
  ranks.back() = managerRankWorld_;

  MPI_Group globalGroup;
  MPI_Group_incl( worldGroup, int( ranks.size() ), &ranks[0], &globalGroup );

  MPI_Comm_create( worldComm_, globalGroup, &globalComm_ );

  if( globalComm_ != MPI_COMM_NULL ) {
    int globalSize;
    MPI_Comm_size( globalComm_, &globalSize );

    managerRank_ = globalSize - 1;
    MPI_Comm_rank( globalComm_, &globalRank_ );
  }

  if( ENABLE_FT ){
    if( globalComm_ != MPI_COMM_NULL){
      createCommFT( &globalCommFT_, globalComm_ );
    }
  }
  /* mark master processes and manager process in Stats. this is necessary for postprocessing */
  Stats::setAttribute("group_manager", std::to_string(globalComm_ != MPI_COMM_NULL));
}

void MPISystem::initGlobalReduceCommm() {
  // create communicator which only contains workers
  MPI_Comm workerComm;
  {
    int color = ( worldRank_ != managerRankWorld_ ) ? 1 : 0;
    int key = (worldRank_ != managerRankWorld_ ) ? worldRank_ : 0;
    MPI_Comm_split( worldComm_, color, key, &workerComm);
  }

  if( worldRank_ != managerRankWorld_ ) {
    int workerID;
    MPI_Comm_rank(workerComm, &workerID);

    MPI_Comm globalReduceComm;
    int color = workerID % int(nprocs_);
    int key = workerID / int(nprocs_);
    MPI_Comm_split(workerComm, color, key, &globalReduceComm);

    globalReduceComm_ = globalReduceComm;

    if( ENABLE_FT ){
      createCommFT( &globalReduceCommFT_, globalReduceComm_ );
    }
  }
}

void MPISystem::createCommFT( simft::Sim_FT_MPI_Comm* commFT, CommunicatorType comm ){
  *commFT = new simft::Sim_FT_MPI_Comm_struct;
  (*commFT)->c_comm = comm;
  simft::Sim_FT_Initialize_new_comm( commFT, true );
}

std::vector<RankType> MPISystem::getFailedRanks( int numFailedProcs ){ //world comm still contains failed ranks at this point!
  std::vector<RankType> failedProcs(numFailedProcs);
  for(int i=0; i< numFailedProcs; i++){
    int failedRank;
     MPI_Recv(&failedRank, 1, MPI_INT, MPI_ANY_SOURCE, FT_FAILED_RANK_TAG, theMPISystem()->getWorldComm(), MPI_STATUS_IGNORE);
     failedProcs[i] = failedRank;
  }
  return failedProcs;
}

std::vector<RankType> MPISystem::getReusableRanks( int remainingProcs ){ //toDo matching send
  std::vector<RankType> reusableProcs(remainingProcs);
  for(int i=0; i< remainingProcs; i++){
    int reusableRank;
     MPI_Recv(&reusableRank, 1, MPI_INT, MPI_ANY_SOURCE, FT_REUSABLE_RANK_TAG, theMPISystem()->getSpareCommFT()->c_comm, MPI_STATUS_IGNORE);
     reusableProcs[i] = reusableRank;
  }
  return reusableProcs;
}

bool MPISystem::sendRankIds(std::vector<RankType>& failedRanks, std::vector<RankType>& reusableRanks ){ //toDo matching send
  std::vector<MPI_Request> requests(failedRanks.size());
  for(int i=0; i< failedRanks.size(); i++){
     MPI_Isend(&failedRanks[i], 1, MPI_INT, reusableRanks[i], FT_NEW_RANK_TAG, theMPISystem()->getSpareCommFT()->c_comm, &requests[i]);
  }
  bool success = true; //so far no time out failing possible
  int numTimeouts = 0; // counts number of ranks that do not answer in time
  for(int i=0; i< failedRanks.size(); i++){
    MPI_Wait(&requests[i],MPI_STATUS_IGNORE); //toDo implement timeout
  }
  int lastIndex = failedRanks.size() + numTimeouts; //needs to be increased in case of ranks with time-out
  if(success){
    reusableRanks.erase(reusableRanks.begin(), reusableRanks.begin() + lastIndex);
  }
  return false; //so far failing not implemented
}

void MPISystem::sendRecoveryStatus(bool failedRecovery, std::vector<RankType>& newReusableRanks ){ //toDo matching send
  std::vector<MPI_Request> requests(newReusableRanks.size());
  for(int i=0; i< newReusableRanks.size(); i++){
     MPI_Isend(&failedRecovery, 1, MPI::BOOL, newReusableRanks[i], FT_RECOVERY_STATUS_TAG, theMPISystem()->getSpareCommFT()->c_comm, &requests[i]);
  }
  for(int i=0; i< newReusableRanks.size(); i++){
    MPI_Wait(&requests[i],MPI_STATUS_IGNORE); //toDo implement timeout
  }
}

void MPISystem::sendExcludeSignal(std::vector<RankType>& reusableRanks){
  std::vector<MPI_Request> requests(reusableRanks.size());

  for(int i = 0; i< reusableRanks.size(); i++){
    MPI_Isend(&reusableRanks[i],0,MPI_INT,managerRankFT_,FT_EXCLUDE_TAG,theMPISystem()->getSpareCommFT()->c_comm, &requests[i]);
  }

  for(int i = 0; i< reusableRanks.size(); i++){
    MPI_Wait(&requests[i],MPI_STATUS_IGNORE); //toDo implement timeout
  }
}

void MPISystem::sendShrinkSignal(std::vector<RankType>& reusableRanks){
  std::vector<MPI_Request> requests(reusableRanks.size());

  for(int i = 0; i< reusableRanks.size(); i++){
    MPI_Isend(&reusableRanks[i],0,MPI_INT,managerRankFT_,FT_SHRINK_TAG,theMPISystem()->getSpareCommFT()->c_comm, &requests[i]);
  }

  for(int i = 0; i< reusableRanks.size(); i++){
    MPI_Wait(&requests[i],MPI_STATUS_IGNORE); //toDo implement timeout
  }
}

void MPISystem::sendReusableSignal(){
  MPI_Send(&worldRank_,1,MPI_INT,managerRankFT_,FT_REUSABLE_RANK_TAG,theMPISystem()->getSpareCommFT()->c_comm);
}

void MPISystem::sendFailedSignal(){ //worldComm still includes failed ranks at this point!
  MPI_Request sendReq;
  MPI_Isend(&worldRank_,1,MPI_INT,managerRankWorld_,FT_FAILED_RANK_TAG,theMPISystem()->getWorldComm(), &sendReq);
}

bool MPISystem::receiveRecoverStatus(){
  bool recoveryState;

  //receive recovery state
  MPI_Recv(&recoveryState, 1, MPI::BOOL, managerRankFT_, FT_RECOVERY_STATUS_TAG, theMPISystem()->getSpareCommFT()->c_comm, MPI_STATUS_IGNORE);
  return recoveryState;


}
/**
 * Routine that handles waiting for idle procs that can be reused.
 * Processors stay in this routine until the can be reused + they contribute in all shrink and split operations.
 */
void MPISystem::waitForReuse(){
  int excludeFlag = 0;
  int newRankFlag = 0;
  int shrinkFlag = 0;
  MPI_Status shrinkStatus;
  MPI_Status excludeStatus;
  MPI_Status newRankStatus;
  while(true){
    MPI_Iprobe(managerRankFT_, FT_NEW_RANK_TAG, theMPISystem()->getSpareCommFT()->c_comm, &newRankFlag, &newRankStatus);
    if(newRankFlag){
      int rankID;
      //receive rank ID
      MPI_Recv(&rankID, 1, MPI_INT, managerRankFT_, FT_NEW_RANK_TAG, theMPISystem()->getSpareCommFT()->c_comm, MPI_STATUS_IGNORE);

      newRankFlag = 0;
      //receive recovery state to check if recovery really succeeded
      bool recoveryStatus;
      MPI_Recv(&recoveryStatus, 1, MPI::BOOL, managerRankFT_, FT_RECOVERY_STATUS_TAG, theMPISystem()->getSpareCommFT()->c_comm, MPI_STATUS_IGNORE);
      if(recoveryStatus){
        worldRank_=rankID;
        return;
      }
      //otherwise wait for recovery again
    }
    MPI_Iprobe(managerRankFT_, FT_EXCLUDE_TAG, theMPISystem()->getSpareCommFT()->c_comm, &excludeFlag, &excludeStatus);
    if(excludeFlag){
       int excludeData;
       excludeFlag = 0;
       //receive exclude data ; so far nothing there
       MPI_Recv(&excludeData, 0, MPI_INT, managerRankFT_, FT_EXCLUDE_TAG, theMPISystem()->getSpareCommFT()->c_comm, MPI_STATUS_IGNORE);
       //perform split with color 0
       int color = 0;
       int key;
       MPI_Comm_size(worldComm_,&key);
       MPI_Comm_split( spareCommFT_->c_comm, color, key, &worldComm_ );
    }
    MPI_Iprobe(managerRankFT_, FT_SHRINK_TAG, theMPISystem()->getSpareCommFT()->c_comm, &shrinkFlag, &shrinkStatus);
    if(shrinkFlag){
       shrinkFlag = 0;
       int shrinkData; // not used so far
       //receive exclude data ; so far nothing there
       MPI_Recv(&shrinkData, 0, MPI_INT, managerRankFT_, FT_SHRINK_TAG, theMPISystem()->getSpareCommFT()->c_comm, MPI_STATUS_IGNORE);
       //perform shrink
       simft::Sim_FT_MPI_Comm newSpareCommFT;
       MPI_Comm_shrink( theMPISystem()->getSpareCommFT(), &newSpareCommFT ); //remove dead processors from spareComm(worldComm + reusable ranks)
       createCommFT( &spareCommFT_, newSpareCommFT->c_comm ); //removes dead processors from worldComm
       //adjust manger rank in spareComm as it has changed during shrink
       int ftCommSize;
       MPI_Comm_size(spareCommFT_->c_comm, &ftCommSize );
       managerRankFT_= ftCommSize - 1;
    }
  }
}


bool MPISystem::recoverCommunicators( bool groupAlive, std::vector< std::shared_ptr< ProcessGroupManager >> failedGroups ){ //toDo fix multiple failed groups
  assert( ENABLE_FT && "this funtion is only availabe if FT enabled!" );
  //std::cout << "start recovery \n";
  // revoke commmworld
  //theStatsContainer()->setTimerStart("recoverComm-revoke");
  //WORLD_MANAGER_EXCLUSIVE_SECTION{
   //MPI_Comm_revoke( theMPISystem()->getWorldCommFT() );
  //}
  //theStatsContainer()->setTimerStop("recoverComm-revoke");
  //std::cout << "revoked MPI comm \n";
  // shrink world
  //theStatsContainer()->setTimerStart("recoverComm-shrink");
  simft::Sim_FT_MPI_Comm newSpareCommFT;
  simft::Sim_FT_MPI_Comm newWorldCommFT;

  //indicate shrink to reusable ranks
  sendShrinkSignal(reusableRanks_);
  //shrink of all processors including reusable ones
  MPI_Comm_shrink( theMPISystem()->getSpareCommFT(), &newSpareCommFT ); //remove dead processors from spareComm(worldComm + reusable ranks)
  //shrink of all active processors
  createCommFT( &spareCommFT_, newSpareCommFT->c_comm ); //removes dead processors from worldComm
  //adjust manger rank in spareComm as it has changed durin shrink
  int ftCommSize;
  MPI_Comm_size(spareCommFT_->c_comm, &ftCommSize );
  managerRankFT_= ftCommSize - 1;

  MPI_Comm_shrink( theMPISystem()->getWorldCommFT(), &newWorldCommFT); //remove dead processors from current worldComm
  bool failedRecovery = true;
  WORLD_MANAGER_EXCLUSIVE_SECTION{ //get failed ranks
    int sizeNew,sizeOld,sizeSpare;
    sizeSpare = 0;
    MPI_Comm_size(theMPISystem()->getWorldComm(),&sizeOld);
    MPI_Comm_size(newWorldCommFT->c_comm,&sizeNew);
    MPI_Comm_size(spareCommFT_->c_comm,&sizeSpare);
    int numFailedRanks = sizeOld-sizeNew;
    std::vector<RankType> failedRanks = getFailedRanks(numFailedRanks);
    std::vector<RankType> newReusableRanks = getReusableRanks(nprocs_ - numFailedRanks);
    //toDO reusableRanks might be outdated due to new failures there
    bool enoughSpareProcs = sizeSpare - sizeNew > numFailedRanks;
    bool failedSendingRankIds = false;
    if(enoughSpareProcs){ //send failed ranks to reusable procs so they can set their worldRank accordingly
      failedSendingRankIds = sendRankIds(failedRanks,reusableRanks_);  //check with timeout if reusable ranks are still available;
      //otherwise fail recovery of broken processgroup
      //remove failed spare procs from reusableRanks_
      if(!failedSendingRankIds){
        failedRecovery = false;
      }
    }
    if(!enoughSpareProcs or failedSendingRankIds){
      failedRecovery = true;
    }
    sendRecoveryStatus(failedRecovery, newReusableRanks);
    //order shrink with color 0 to reusable ranks as they need to be excluded for this recovery process
    sendExcludeSignal(reusableRanks_);
    if(!enoughSpareProcs or failedSendingRankIds){ // add in case of failure newreusable ranks to vector
      reusableRanks_.insert(reusableRanks_.end(), newReusableRanks.begin(), newReusableRanks.end());
    }
  }
  int color;
  if(!groupAlive){ //check if group was recovered and mark as reusable
    sendReusableSignal();
    bool recovered = receiveRecoverStatus(); //check if exclude signal
    if(!recovered){
      color = 0;
      int key = worldRank_;
      MPI_Comm_split( spareCommFT_->c_comm, color, key, &worldComm_ );
      waitForReuse(); //does only leave this routine when reused later //participates in future shrinks
      color = 1;
    }
  }
  //theStatsContainer()->setTimerStop("recoverComm-shrink");
  //std::cout << "shrinked communicator \n";
  // split off alive procs. this will be the new WorldComm
  // processes of dead groups set color to MPI_UNDEFINED. in this case
  // MPI_Comm_split returns MPI_COMMM_NULL
  color = 1; //all processors at this point belong to alive or recovered process groups
  int key = worldRank_;
  MPI_Comm_split(spareCommFT_->c_comm, color, key, &worldComm_ );
  /*color = (!groupAlive) ? 1 : MPI_UNDEFINED; //all alive procs from dead process groups
  WORLD_MANAGER_EXCLUSIVE_SECTION{
     color = 1;
  }
  int key = worldRank_;
  MPI_Comm_split( newCommWorld, color, key, &spareComm_ );*/

  // early exit for dead procs; not existing anymore
  //if( worldComm_ == MPI_COMM_NULL)
  //  return;

  // todo: remove
  // output new commWorld
  {
    int newRank, newSize;
    MPI_Comm_rank( worldComm_, &newRank );
    MPI_Comm_size( worldComm_, &newSize );

    if( newRank == 0 )
      std::cout << "new WorldComm:" << std::endl;
    for( auto r=0; r < newSize; ++r ){
        if( newRank == r ){
          std::cout << "rank " << theMPISystem()->getWorldRank()
                    << " new rank " << newRank
                    << " new size " << newSize
                    << std::endl;
        }
        MPI_Barrier( worldComm_ );
    }
  }

  int worldSize;
  MPI_Comm_size( worldComm_, &worldSize );
  assert( (worldSize - 1) % nprocs_ == 0 );
  ngroup_ = (worldSize - 1) / nprocs_;

  MPI_Comm_rank( worldComm_, &worldRank_ );
  managerRankWorld_ = worldSize - 1;


  if( worldComm_ != MPI_COMM_NULL ){
    createCommFT( &worldCommFT_, worldComm_ );
  }
  if(!groupAlive){
    //toDo:init local comm?
    initLocalComm();
  }
  //initLocalComm();

  initGlobalComm();

  initGlobalReduceCommm();

  /* print stats */
  /* outdated
   int ngroup( theMPISystem()->getNumGroups() );
   int nprocs( theMPISystem()->getNumProcs() );
   std::string t_revoke = getMinMaxAvg( theMPISystem()->getManagerRankWorld(),
                                        ngroup*nprocs+1,
                                      "recoverComm-revoke", true,
                                      theMPISystem()->getWorldComm() );
   std::string t_shrink = getMinMaxAvg( theMPISystem()->getManagerRankWorld(),
                                        ngroup*nprocs+1,
                                      "recoverComm-shrink", true,
                                      theMPISystem()->getWorldComm() );
   WORLD_MANAGER_EXCLUSIVE_SECTION{ std::cout << t_revoke << std::endl; }
   WORLD_MANAGER_EXCLUSIVE_SECTION{ std::cout << t_shrink << std::endl; }*/

   //toDo return fixed process group IDs

  return failedRecovery;
}

} // namespace combigrid


