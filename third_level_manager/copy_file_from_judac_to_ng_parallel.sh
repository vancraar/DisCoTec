#!/bin/bash
# variation of copy_file_from_judac_to_ng.sh where the transfer is done in parallel over multiple nodes. Start these script on login05, login06 and login07.
# The script will automatically determine the host and the part of the file to transfer.
# For further dokumentation see also copy_file_from_judac_to_ng.sh

# execute this script on skx-arch.supermuc.lrz.de
# as described on https://doku.lrz.de/display/PUBLIC/Data+Transfer+Options+on+SuperMUC-NG#DataTransferOptionsonSuperMUCNG-UNICOREFileTransfer(UFTP)

module load uftp-client


# kill all processes from this script if the script is terminated
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT

ID=${ID:=1}
PATHLRZ=${PATHLRZ:=//hppfs/work/pn36xu/di39qun2/widely/widely_measurements/widely_10x512/}
PATHJUDAC=${PATHJUDAC:=/p/scratch/widediscotecjsc/widely/widely_measurements/widely_1x512/}
FILEJUDAC=${FILEJUDAC:=${PATHJUDAC}/dsg_${ID}_step*_0}
JUDAC=https://uftp.fz-juelich.de:9112/UFTP_Auth/rest/auth/JUDAC
USERJUDAC=${USERJUDAC:=pollinger2}
JUDAC_ID=${JUDAC_ID:=~/.uftp/id_uftp_to_jsc}
num_hosts=3

TOKEN_TRANSFER_BACKWARD=${FILEJUDAC:0:-2}_token.txt
TOKEN_STOP=uftp_transfer_stop.txt

PROCS=${PROCS:=4}
THREADS_PER_PROC=${THREADS_PER_PROC:=12}
STREAMS=${STREAMS:=3}

echo "$FILEJUDAC"
echo "$TOKEN_TRANSFER_BACKWARD"

step=0

# loop that is infinite until $TOKEN_STOP is created
while true
do
    FILEJUDAC=${PATHJUDAC}/dsg_${ID}_step${step}_0
    TOKEN_TRANSFER_BACKWARD=${FILEJUDAC:0:-2}_token.txt
    # test if the file is there by trying to copy it (maybe there is a better way?)
    # the part after `>` is to ignore the potential (error) output
    uftp cp -i $JUDAC_ID -u $USERJUDAC $JUDAC:$TOKEN_TRANSFER_BACKWARD /dev/null > /dev/null 2>&1
    cp_status=$?
    pids=( )
    if (exit $cp_status); then
        # ./copy_hawk_to_sng.sh
        # try to copy the file
        FILEJUDAC_INSTANCE=$(echo $TOKEN_TRANSFER_BACKWARD)
        FILEJUDAC_INSTANCE=${FILEJUDAC_INSTANCE:0:-10}_0
        echo copy $FILEJUDAC_INSTANCE from hawk to NG
        size=$(uftp ls -i $JUDAC_ID -u $USERJUDAC  $JUDAC:$FILEJUDAC_INSTANCE | awk '{print $2 }')
        local_size=$((size/num_hosts))
        host=$(hostname)
        if [ $host == "login05" ]; then
            local_start=0
        elif [ $host == "login06" ]; then
            local_start=$((local_size))
        elif [ $host == "login07" ]; then
            local_start=$((local_size*2))
        fi
        startblock=$local_start
        endblock=$((local_start+(local_size/PROCS)))
        starttime=`date +%s`
        for ((i=1; i<=PROCS; i++)); do
          echo "Block: $startblock-$endblock of $size"
          uftp cp  -B "${startblock}-${endblock}-p" -i $JUDAC_ID -u $USERJUDAC $JUDAC:$FILEJUDAC_INSTANCE $PATHLRZ/ &
          pids+=($!)
          startblock=$((endblock))
          if [ $i -eq $((PROCS-1)) ]; then
              if [ $host ==  "login07" ]; then
                        endblock=$((size))
              else
                endblock=$((local_start+local_size))
              fi
          else
              endblock=$((endblock+local_size/PROCS))
          fi
        done
        # wait for all processes to finish
        for pid in "${pids[@]}"; do
            wait $pid || exit
        done
        touch dsg_${ID}_step${step}_0_${host}
        if [ $host == "login05" ]; then
            while [ ! -f dsg_${ID}_step${step}_0_login05 ] || [ ! -f dsg_${ID}_step${step}_0_login06 ] || [ ! -f dsg_${ID}_step${step}_0_login07 ]; do
                sleep 1
            done
            rm dsg_${ID}_step${step}_0_login05
            rm dsg_${ID}_step${step}_0_login06
            rm dsg_${ID}_step${step}_0_login07
            uftp cp -i $JUDAC_ID -u $USERJUDAC $JUDAC:$TOKEN_TRANSFER_BACKWARD $PATHLRZ/
            uftp rm --quiet -i $JUDAC_ID -u $USERJUDAC "$JUDAC:$TOKEN_TRANSFER_BACKWARD"
        fi

        endtime=`date +%s`
        echo "copied $local_size bites from file $FILEJUDAC_INSTANCE in `expr $endtime - $starttime` seconds."
        throughput=$( echo "scale=4;($local_size/1024/1024)/(($endtime-$starttime))" | bc )
        throughput_bits=$( echo "scale=4;($throughput*8)" | bc )
        echo "Local average throughput: $throughput MB/s; $throughput_bits Mbit/s"
        if [ $host == "login05" ]; then
          throughput=$( echo "scale=4;($size/1024/1024)/(($endtime-$starttime))" | bc )
          throughput_bits=$( echo "scale=4;($throughput*8)" | bc )
          echo "Approx total average throughput: $throughput MB/s; $throughput_bits Mbit/s"
          uftp rm -q -i $JUDAC_ID -u $USERJUDAC $JUDAC:$FILEJUDAC_INSTANCE
        fi
        date
        step=$((step+1))
    fi
    if test -f "$PATHLRZ/$TOKEN_STOP"; then
        break
    fi
    sleep 1
done
