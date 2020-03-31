#include "sgpp/distributedcombigrid/manager/ProcessGroupManager.hpp"
#include "sgpp/distributedcombigrid/manager/CombiParameters.hpp"
#include "sgpp/distributedcombigrid/mpi/MPIUtils.hpp"
#include "sgpp/distributedcombigrid/mpi_fault_simulator/MPI-FT.h"

namespace combigrid {
ProcessGroupManager::ProcessGroupManager(RankType pgroupRootID)
    : pgroupRootID_(pgroupRootID),
      status_(PROCESS_GROUP_WAIT),
      statusRequest_(MPI_Request()),
      statusRequestFT_(nullptr) {}

bool ProcessGroupManager::runfirst(Task* t) {
  return storeTaskReferenceAndSendTaskToProcessGroup(t, RUN_FIRST);
}

bool ProcessGroupManager::storeTaskReferenceAndSendTaskToProcessGroup(Task* t, SignalType signal) {
  // first check status
  // tying to add a task to a busy group is an invalid operation
  // and should be avoided
  if (status_ != PROCESS_GROUP_WAIT) return false;

  storeTaskReference(t);
  return sendTaskToProcessGroup(t, signal);
}

void ProcessGroupManager::storeTaskReference(Task* t) {
  // add task to list of tasks managed by this pgroup
  tasks_.push_back(t);
}

bool ProcessGroupManager::sendTaskToProcessGroup(Task* t, SignalType signal) {
  // send signal to pgroup
  sendSignalToProcessGroup(signal);

  // send task
  Task::send(&t, pgroupRootID_, theMPISystem()->getGlobalComm());

  setProcessGroupBusyAndReceive();

  // only return true if task successfully sent to pgroup
  return true;
}

void ProcessGroupManager::sendSignalAndReceive(SignalType signal) {
  sendSignalToProcessGroup(signal);
  setProcessGroupBusyAndReceive();
}

void ProcessGroupManager::sendSignalToProcessGroup(SignalType signal) {
  MPI_Send(&signal, 1, MPI_INT, pgroupRootID_, signalTag, theMPISystem()->getGlobalComm());
}

void ProcessGroupManager::sendSignalToProcess(SignalType signal, RankType rank) { //TODO send only to process in this pgroup
  MPI_Send(&signal, 1, MPI_INT, rank, signalTag, theMPISystem()->getGlobalComm());
}

inline void ProcessGroupManager::setProcessGroupBusyAndReceive() {
  // set status
  status_ = PROCESS_GROUP_BUSY;

  // start non-blocking MPI_IRecv to receive status
  recvStatus();
}

bool ProcessGroupManager::runnext() {
  // first check status
  // trying to send a command to a busy group is an invalid operation
  // and should be avoided
  assert(status_ == PROCESS_GROUP_WAIT);

  if (tasks_.size() == 0) return false;

  sendSignalAndReceive(RUN_NEXT);

  return true;
}

bool ProcessGroupManager::exit() {
  // can only send exit signal when in wait state
  if (status_ != PROCESS_GROUP_WAIT) return false;

  sendSignalAndReceive(EXIT);
  return true;
}

bool ProcessGroupManager::combine() {
  // can only send sync signal when in wait state
  assert(status_ == PROCESS_GROUP_WAIT);

  sendSignalAndReceive(COMBINE);

  return true;
}

bool ProcessGroupManager::initDsgus() {
  // can only send sync signal when in wait state
  assert(status_ == PROCESS_GROUP_WAIT);

  sendSignalAndReceive(INIT_DSGUS);

  return true;
}

bool ProcessGroupManager::combineThirdLevel(const ThirdLevelUtils& thirdLevel,
                                            CombiParameters& params,
                                            bool isSendingFirst) {
  // can only send sync signal when in wait state
  assert(status_ == PROCESS_GROUP_WAIT);

  sendSignalAndReceive(COMBINE_THIRD_LEVEL);

  if (isSendingFirst)
    exchangeDsgus(thirdLevel, params, true);
  else
    exchangeDsgus(thirdLevel, params, false);

  return true;
}


/*
 * Differs from third level reduce since we have enough memory space to collect
 * the subspace sizes from all dsgs of all procs in the third level pg in a single
 * MPI_Gather call.
 */
bool ProcessGroupManager::reduceLocalAndRemoteSubspaceSizes(const ThirdLevelUtils& thirdLevel,
                                       CombiParameters& params,
                                       bool isSendingFirst)
{
  // tell workers to perform reduce
  sendSignalAndReceive(REDUCE_SUBSPACE_SIZES_TL);

  // prepare buffers
  std::unique_ptr<uint64_t[]> sendBuff;
  std::unique_ptr<uint64_t[]> recvBuff;
  size_t buffSize;
  std::vector<int> numSubspacesPerWorker;

  // gather subspace sizes from workers
  collectSubspaceSizes(thirdLevel, params, sendBuff, buffSize, numSubspacesPerWorker);
  recvBuff.reset(new uint64_t[buffSize]);

  /* TODO can be easily parallelized by removing the condition and call send and
     receive in a separate thread*/
  if (isSendingFirst) {
    // send subspace sizes to remote
    thirdLevel.sendData(sendBuff.get(), buffSize);
    // receive remote subspace sizes
    thirdLevel.recvData(recvBuff.get(), buffSize);
  } else {
    // receive remote subspace sizes
    thirdLevel.recvData(recvBuff.get(), buffSize);
    // send subspace sizes to remote
    thirdLevel.sendData(sendBuff.get(), buffSize);
  }

  // perform max reduce
  for (size_t i = 0; i < buffSize; ++i)
    sendBuff[i] = std::max(sendBuff[i], recvBuff[i]);

  // set accumulated dsgu sizes per worker
  dsguDataSizePerWorker_.resize(numSubspacesPerWorker.size());
  uint64_t* sizePtr = sendBuff.get();
  for (size_t w = 0; w < numSubspacesPerWorker.size(); ++w) {
    int sum = 0;
    for (int ss = 0; ss < numSubspacesPerWorker[w]; ++ss) {
      sum += (int) *(sizePtr++);
    }
    dsguDataSizePerWorker_[w] = sum;
  }

  // scatter data back to workers
  distributeSubspaceSizes(thirdLevel, params, sendBuff, buffSize, numSubspacesPerWorker);
  return true;
}


void ProcessGroupManager::exchangeDsgus(const ThirdLevelUtils& thirdLevel,
                                             CombiParameters& params,
                                             bool isSendingFirst) {
  const std::vector<CommunicatorType>& thirdLevelComms = theMPISystem()->getThirdLevelComms();
  assert(theMPISystem()->getNumGroups() == thirdLevelComms.size() &&
      "initialisation of third level communicator failed");
  const CommunicatorType& comm = thirdLevelComms[params.getThirdLevelPG()];

  // exchange dsgus
  MPI_Datatype dataType = getMPIDatatype(abstraction::getabstractionDataType<CombiDataType>());
  IndexType numGrids = params.getNumGrids();
  std::unique_ptr<CombiDataType[]> dsguData;
  for (IndexType g = 0; g < numGrids; g++) {
    for (RankType p = 0; p < (RankType) theMPISystem()->getNumProcs(); p++) {
      // we assume here that all dsgus have the same size otherwise size collection must change
      size_t dsguSize = (size_t) (dsguDataSizePerWorker_[(size_t)p]/numGrids);
      assert(dsguSize < INT_MAX && "dsgu is larger than what we can send in a "
                                   "single mpi call");
      // recv dsgu from worker
      dsguData.reset(new CombiDataType[dsguSize]);
      MPI_Recv(dsguData.get(), (int) dsguSize, dataType, p, MPI_ANY_TAG, comm, MPI_STATUS_IGNORE);

      if (isSendingFirst) {
        // send dsgu to remote
        thirdLevel.sendData(dsguData.get(), dsguSize);
        // recv combined dsgu from remote
        thirdLevel.recvData(dsguData.get(), dsguSize);
        // send to worker
        MPI_Send(dsguData.get(), (int) dsguSize, dataType, p, 0, comm);
      } else {
        // recv and combine dsgu from remote
        thirdLevel.recvAndAddToData(dsguData.get(), dsguSize);
        // send combined solution to remote
        thirdLevel.sendData(dsguData.get(), dsguSize);
        // send to worker
        MPI_Send(dsguData.get(), (int) dsguSize, dataType, p, 0, comm);
      }
    }
  }
}


bool ProcessGroupManager::collectSubspaceSizes(const ThirdLevelUtils& thirdLevel,
                                               CombiParameters& params,
                                               std::unique_ptr<uint64_t[]>& buff,
                                               size_t& buffSize,
                                               std::vector<int>& numSubspacesPerWorker) {

  // prepare args of MPI_Gather
  const CommunicatorType& comm =
    theMPISystem()->getThirdLevelComms()[(size_t)pgroupRootID_];
  size_t nprocs = theMPISystem()->getNumProcs();
  std::vector<int> recvCounts(nprocs + 1); // includes master
  RankType thirdLevelManagerRank = theMPISystem()->getThirdLevelManagerRank();
  int dummy = 0;

  // gather number of subspaces in all dsgus per worker for upcoming MPI_Gatherv
  // for now all workers should have the same number of subspaces
  MPI_Gather(&dummy, 1, MPI_INT, recvCounts.data(), (int) 1,
             MPI_INT, thirdLevelManagerRank, comm);

  buffSize = std::accumulate(recvCounts.begin(),
                             recvCounts.end(), 0U);

  std::unique_ptr<size_t[]> mdBuff(new size_t[buffSize]); // size_t is machine dependent
  assert(buffSize < INT_MAX && "bufSize is larger than what we can send in a "
                               "single mpi call");

  // prepare displacements for MPI_Gatherv
  std::vector<int> displacements(nprocs + 1); // includes master
  int disp = 0;
  for (size_t i = 0; i < displacements.size(); ++i) {
    displacements[i] = disp;
    disp += recvCounts[i];
  }
  // perform gather of subspace sizes
  MPI_Datatype dataType = getMPIDatatype(
                        abstraction::getabstractionDataType<size_t>());
  MPI_Gatherv(&dummy, 0, dataType, mdBuff.get(), recvCounts.data(),
      displacements.data(), dataType, thirdLevelManagerRank, comm);

  // remove master
  numSubspacesPerWorker = recvCounts;
  numSubspacesPerWorker.pop_back();

  // create machine independent buffer
  buff.reset(new size_t[buffSize]);
  for (size_t i = 0; i < buffSize; ++i)
    buff[i] = static_cast<uint64_t>(mdBuff[i]);

  return true;
}


bool ProcessGroupManager::distributeSubspaceSizes(const ThirdLevelUtils& thirdLevel,
                                                  CombiParameters& params,
                                                  const std::unique_ptr<uint64_t[]>& buff,
                                                  size_t buffSize,
                                                  const std::vector<int>& numSubspacesPerWorker) {
  // prepare args of MPI_Scatterv
  const CommunicatorType& comm =
    theMPISystem()->getThirdLevelComms()[(size_t)pgroupRootID_];
  RankType thirdLevelManagerRank = theMPISystem()->getThirdLevelManagerRank();
  std::vector<int> sendCounts(numSubspacesPerWorker);
  sendCounts.push_back(0); // append manager
  std::vector<int> displacements(sendCounts.size());
  int disp = 0;
  for (size_t i = 0; i < displacements.size(); ++i) {
    displacements[i] = disp;
    disp += sendCounts[i];
  }

  // create machine dependent buffer
  std::unique_ptr<size_t[]> mdBuff(new size_t[buffSize]);
  for (size_t i = 0; i < buffSize; ++i)
    mdBuff[i] = static_cast<size_t>(buff[i]);

  // perform scatter of subspace sizes
  MPI_Datatype dataType = getMPIDatatype(
                        abstraction::getabstractionDataType<size_t>());
  MPI_Scatterv(mdBuff.get(), sendCounts.data(), displacements.data(), dataType,
               nullptr, 0, dataType, thirdLevelManagerRank, comm);

  return true;
}

bool ProcessGroupManager::combineLocalAndGlobal() {
  // can only send sync signal when in wait state
  assert(status_ == PROCESS_GROUP_WAIT);

  sendSignalAndReceive(COMBINE_LOCAL_AND_GLOBAL);
  return true;
}

bool ProcessGroupManager::waitForThirdLevelCombiResult() {
  // can only send sync signal when in wait state
  assert(status_ == PROCESS_GROUP_WAIT);

  sendSignalAndReceive(WAIT_FOR_TL_COMBI_RESULT);
  return true;
}

bool ProcessGroupManager::waitForThirdLevelSizeUpdate() {
  // can only send sync signal when in wait state
  assert(status_ == PROCESS_GROUP_WAIT);

  sendSignalAndReceive(WAIT_FOR_TL_SIZE_UPDATE);
  return true;
}

bool ProcessGroupManager::updateCombiParameters(CombiParameters& params) {
  // can only send sync signal when in wait state
  assert(status_ == PROCESS_GROUP_WAIT);

  sendSignalToProcessGroup(UPDATE_COMBI_PARAMETERS);

  // send combiparameters
  // std::cout << "sending class \n";
  MPIUtils::sendClass(&params, pgroupRootID_, theMPISystem()->getGlobalComm());

  setProcessGroupBusyAndReceive();
  // std::cout << "manager received status \n";
  return true;
}


bool ProcessGroupManager::addTask(Task* t) {
  return storeTaskReferenceAndSendTaskToProcessGroup(t, ADD_TASK);
}

bool ProcessGroupManager::refreshTask(Task* t) {
  // first check status
  // tying to add a task to a busy group is an invalid operation
  // and should be avoided
  if (status_ != PROCESS_GROUP_WAIT) {
    std::cout << "refreshing failed! \n";
    return false;
  }

  sendSignalToProcessGroup(ADD_TASK);

  // send task
  Task::send(&t, pgroupRootID_, theMPISystem()->getGlobalComm());

  setProcessGroupBusyAndReceive();

  // only return true if task successfully send to pgroup
  return true;
}

bool ProcessGroupManager::resetTasksWorker() {
  // first check status
  // tying to reset tasks of a busy group is an invalid operation
  // and should be avoided
  if (status_ != PROCESS_GROUP_WAIT) {
    assert(false);
    // return false;
  }

  // add task to list of tasks managed by this pgroup
  // tasks_.clear(); we do not clear group manager tasks

  sendSignalAndReceive(RESET_TASKS);

  return true;
}

bool ProcessGroupManager::recompute(Task* t) {
  storeTaskReferenceAndSendTaskToProcessGroup(t, RECOMPUTE);

  // only return true if task successfully send to pgroup
  return true;
}

bool ProcessGroupManager::parallelEval(const LevelVector& leval, std::string& filename) {
  // can only send sync signal when in wait state, so check first
  assert(status_ == PROCESS_GROUP_WAIT);

  sendSignalToProcessGroup(PARALLEL_EVAL);

  // send levelvector
  std::vector<int> tmp(leval.begin(), leval.end());
  MPI_Send(&tmp[0], static_cast<int>(tmp.size()), MPI_INT, pgroupRootID_, 0,
           theMPISystem()->getGlobalComm());

  // send filename
  MPIUtils::sendClass(&filename, pgroupRootID_, theMPISystem()->getGlobalComm());

  setProcessGroupBusyAndReceive();

  return true;
}

void ProcessGroupManager::recvStatus() {
  // start non-blocking call to receive status
  if (ENABLE_FT) {
    simft::Sim_FT_MPI_Irecv(&status_, 1, MPI_INT, pgroupRootID_, statusTag,
                            theMPISystem()->getGlobalCommFT(), &statusRequestFT_);
  } else {
    MPI_Irecv(&status_, 1, MPI_INT, pgroupRootID_, statusTag, theMPISystem()->getGlobalComm(),
              &statusRequest_);
  }
}

bool ProcessGroupManager::recoverCommunicators() {
  assert(status_ == PROCESS_GROUP_WAIT);

  sendSignalToProcessGroup(RECOVER_COMM);

  return true;
}

bool ProcessGroupManager::rescheduleAddTask(Task *task) {
  return storeTaskReferenceAndSendTaskToProcessGroup(task, RESCHEDULE_ADD_TASK);
}

Task *ProcessGroupManager::rescheduleRemoveTask(const LevelVector &lvlVec) {
  for (std::vector<Task *>::size_type i = 0; i < this->tasks_.size(); ++i) {
    Task *currentTask = this->tasks_[i];
    if (currentTask->getLevelVector() == lvlVec) { 
      // if the task has been found send remove signal and return the task
      Task *removedTask;
      auto taskID = currentTask->getID();
      sendSignalToProcessGroup(RESCHEDULE_REMOVE_TASK);
      MPI_Send(&taskID, 1, MPI_INT, this->pgroupRootID_, 0, theMPISystem()->getGlobalComm());
      Task::receive(&removedTask, this->pgroupRootID_, theMPISystem()->getGlobalComm());
      setProcessGroupBusyAndReceive();

      tasks_.erase(tasks_.begin() + i);
      delete currentTask;


      return removedTask;
    }
  }
  return nullptr;
}

bool ProcessGroupManager::writeCombigridsToVTKPlotFile() {
  // can only send sync signal when in wait state
  assert(status_ == PROCESS_GROUP_WAIT);

  sendSignalAndReceive(WRITE_DFGS_TO_VTK);
  return true;
}

} /* namespace combigrid */
