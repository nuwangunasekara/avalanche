print_usage()
{
  echo "Usage: $0 <app_path> <exp_path> <conda_path>"
  echo "e.g:   $0 /Scratch/ng98/CL/avalanche_nuwan_fork/exp_scripts/train_pool.py /Scratch/ng98/CL/results/ /Scratch/ng98/CL/conda"
  echo "e.g:   $0 ~/Desktop/avalanche_nuwan_fork/exp_scripts/train_pool.py ~/Desktop/CL/results/ /Users/ng98/miniconda3/envs/avalanche-dev-env"
}
cuda='0'
clean_dir="TRUE"
clean_dir="FALSE"

dataset=(LED_a RotatedMNIST RotatedCIFAR10 CORe50)
dataset=(RotatedMNIST RotatedCIFAR10 CORe50)
dataset=(RotatedMNIST RotatedCIFAR10)
strategy=(LwF EWC GDumb TrainPool)

mini_batch_size='10'

tp_pool_type='6CNN'
tp_number_of_nns_to_train='6'
tp_predict_methods_array=('ONE_CLASS' 'ONE_CLASS_end' 'MAJORITY_VOTE' 'RANDOM' 'NAIVE_BAYES' 'NAIVE_BAYES_end' 'TASK_ID_KNOWN')

model='SimpleCNN'
optimizer='Adam'
l_rate='0.0005'

app_path="/Users/ng98/Desktop/avalanche_test/train_pool.py"
base_dir="/Users/ng98/Desktop/avalanche_test/exp"

if [ $# -lt 3 ]; then
  print_usage
  exit 1
fi

app_path=$1
base_dir=$2

echo "Activating conda environment $3"
eval "$(conda shell.bash hook)"
conda init bash
conda activate "$3"
conda env list

log_dir="${base_dir}/logs"
log_file_name=''

if [ -d "${log_dir}" ]; then
  if [ "${clean_dir}" == "TRUE" ]; then
    echo "Remove directory ${log_dir}"
    rm -rf ${log_dir}
    echo "Create log directory structure"
    mkdir -p "${log_dir}/"{tb_data,txt_logs,exp_logs,csv_data}
  fi
else
  echo "Create log directory structure"
  mkdir -p "${log_dir}/"{tb_data,txt_logs,exp_logs,csv_data}
fi


for (( j=0; j<${#dataset[@]}; j++ ))
do
  for (( i=0; i<${#strategy[@]}; i++ ))
  do
    if [ "${strategy[$i]}" = "TrainPool" ] ; then
      tp_predict_methods=("${tp_predict_methods_array[@]}")
    else
      tp_predict_methods=('OTHER')
    fi
    for (( k=0; k<${#tp_predict_methods[@]}; k++ ))
    do
      command_args="--base_dir $base_dir --dataset ${dataset[$j]} --strategy ${strategy[$i]} --minibatch_size ${mini_batch_size} --cuda ${cuda}"
          log_file_name="${dataset[$j]}_${strategy[$i]}_mb_${mini_batch_size}"
      case ${strategy[$i]} in
        LwF)
          command_args="${command_args} --module ${model} --optimizer ${optimizer} --lr ${l_rate} --hs 1024"
          log_file_name="${log_file_name}_${model}"
          ;;
        EWC)
          command_args="${command_args} --module ${model} --optimizer ${optimizer} --lr ${l_rate} --hs 1024"
          log_file_name="${log_file_name}_${model}"
          ;;
        GDumb)
          command_args="${command_args} --module ${model} --optimizer ${optimizer} --lr ${l_rate} --hs 1024 --mem_buff_size 1000"
          log_file_name="${log_file_name}_${model}_b1000"
          ;;
        TrainPool)
          tp_predict_method=${tp_predict_methods[$k]}
          tp_p_method="${tp_predict_method}"
          tp_train_p_type='--no-train_task_predictor_at_the_end'
          case ${tp_predict_method} in
            ONE_CLASS)
              ;;
            ONE_CLASS_end)
              tp_train_p_type='--train_task_predictor_at_the_end'
              tp_p_method='ONE_CLASS'
              ;;
            NAIVE_BAYES)
              ;;
            NAIVE_BAYES_end)
              tp_train_p_type='--train_task_predictor_at_the_end'
              tp_p_method='NAIVE_BAYES'
              ;;
            *)
              ;;
          esac
          command_args="${command_args} --module MultiMLP --pool_type ${tp_pool_type} --number_of_mpls_to_train ${tp_number_of_nns_to_train} --skip_back_prop_threshold 0.0 --task_detector_type ${tp_p_method} ${tp_train_p_type}"
          log_file_name="${log_file_name}_TP_${tp_pool_type}_${tp_number_of_nns_to_train}_${tp_predict_method}"
          ;;
        *)
          command_args=""
          log_file_name=""
          ;;
      esac

      command_args="${command_args} --log_file_name ${log_file_name}"

      if [ -n "$command_args" ] ; then
        full_log_file="${log_dir}/exp_logs/$log_file_name"
        echo "python $app_path $command_args &>${full_log_file}"
        echo "Log file: $full_log_file"
        time python $app_path $command_args &>$full_log_file &
      fi

          if [ -z $! ]; then
        task_failed=1
      else
        PID=$!
  #      echo -e "PID=$PID : $exp_cmd \n"
        echo -e "PID=$PID : \n"
        sleep 5

        while [ $(grep -m 1 -c 'Experiment completed' $full_log_file ) -lt 1 ];
        do
          sleep 10
          if ! ps -p $PID &>/dev/null;
          then
            task_failed=1
            break
          esle
            echo -ne "Waiting for exp with $PID to finish\r"
          fi
        done

        echo "Child processors of PID $PID----------------------"
        # This is process id, parameter passed by user
        ppid=$PID

        if [ -z $ppid ] ; then
           echo "No PID given."
        fi

        child_process_count=1
        while true
        do
          FORLOOP=FALSE
          # Get all the child process id
          for c_pid in `ps -ef| awk '$3 == '$ppid' { print $2 }'`
          do
            if [ $c_pid -ne $SCRIPT_PID ] ; then
              child_pid[$child_process_count]=$c_pid
              child_process_count=$((child_process_count + 1))
              ppid=$c_pid
              FORLOOP=TRUE
            else
              echo "Skip adding PID $SCRIPT_PID"
            fi
          done
          if [ "$FORLOOP" = "FALSE" ] ; then
             child_process_count=$((child_process_count - 1))
             ## We want to kill child process id first and then parent id's
             while [ $child_process_count -ne 0 ]
             do
               echo "killing ${child_pid[$child_process_count]}"
               kill -9 "${child_pid[$child_process_count]}" >/dev/null
               child_process_count=$((child_process_count - 1))
             done
           break
          fi
        done
        echo "Child processors of PID $PID----------------------"
        echo -e "killing PID $PID\n"
        kill $PID
      fi

    done
  done
done
