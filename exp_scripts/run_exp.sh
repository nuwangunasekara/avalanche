print_usage()
{
  echo "Usage: $0 <app_path> <exp_path> <conda_path>"
  echo "e.g:   $0 /Scratch/ng98/CL/avalanche_nuwan_fork/exp_scripts/train_pool.py /Scratch/ng98/CL/results/ /Scratch/ng98/CL/conda"
  echo "e.g:   $0 ~/Desktop/avalanche_nuwan_fork/exp_scripts/train_pool.py ~/Desktop/CL/results/ /Users/ng98/miniconda3/envs/avalanche-dev-env"
}
cuda='0'
clean_dir="TRUE"
#clean_dir="FALSE"

dataset=(LED_a RotatedMNIST RotatedCIFAR10 CORe50 CLStream51)
dataset=(CORe50 RotatedMNIST RotatedCIFAR10 )
#dataset=(CORe50)
strategy=(LwF EWC GDumb ER TrainPool)
strategy=(TrainPool)

mini_batch_size='16'

tp_pool_type='6CNN'
tp_number_of_nns_to_train='6'
tp_predict_methods_array=('ONE_CLASS' 'ONE_CLASS_end' 'MAJORITY_VOTE' 'RANDOM' 'NAIVE_BAYES' 'NAIVE_BAYES_end' 'TASK_ID_KNOWN' 'HT')
tp_predict_methods_array=('HT')
#tp_reset_tp='reset'
tp_reset_tp='no_reset'
#tp_use_one_class_probas='no_use_probas'
tp_use_one_class_probas='use_probas'
#tp_use_weights_from_task_detectors='no_use_weights'
tp_use_weights_from_task_detectors='use_weights'
tp_auto_detect_tasks='no_detect'
#tp_auto_detect_tasks='detect'
# DO_NOT_NOT_TRAIN_TASK_PREDICTOR_AT_THE_END, WITH_ACCUMULATED_INSTANCES, WITH_ACCUMULATED_LEARNED_FEATURES, WITH_ACCUMULATED_STATIC_FEATURES
train_task_predictor_at_the_end_default='DO_NOT_NOT_TRAIN_TASK_PREDICTOR_AT_THE_END'

# exp/scripts/train_pool.py sets internal variable use_static_f_ex to True if train_task_predictor_at_the_end_default='WITH_ACCUMULATED_STATIC_FEATURES'
# But it is best to set it externally via --use_static_f_ex
tp_use_static_f_ex='no-use_static_f_ex'
tp_use_static_f_ex='use_static_f_ex'

tp_train_nn_using_ex_static_f='no-train_nn_using_ex_static_f'
#tp_train_nn_using_ex_static_f='train_nn_using_ex_static_f'

tp_train_only_the_best_nn='no-train_only_the_best_nn'
#tp_train_only_the_best_nn='train_only_the_best_nn'

model='SimpleCNN'
model='CNN4'
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
        ER)
          command_args="${command_args} --module ${model} --optimizer ${optimizer} --lr ${l_rate} --hs 1024 --mem_buff_size 1000"
          log_file_name="${log_file_name}_${model}_b1000"
          ;;
        TrainPool)
          tp_predict_method=${tp_predict_methods[$k]}
          tp_p_method="${tp_predict_method}"
          tp_train_task_p_at_end_type="${train_task_predictor_at_the_end_default}"
          case ${tp_predict_method} in
            ONE_CLASS)
              ;;
            ONE_CLASS_end)
#              tp_train_task_p_at_end_type="${train_task_predictor_at_the_end}"
              tp_p_method='ONE_CLASS'
              ;;
            NAIVE_BAYES)
              ;;
            NAIVE_BAYES_end)
#              tp_train_task_p_at_end_type="${train_task_predictor_at_the_end}"
              tp_p_method='NAIVE_BAYES'
              ;;
            *)
              ;;
          esac

          if [ "${tp_reset_tp}" == "reset" ]; then
            tp_reset_tp_cmd='--reset_training_pool'
          else
            tp_reset_tp_cmd='--no-reset_training_pool'
          fi

          if [ "${tp_use_one_class_probas}" == "use_probas" ]; then
            tp_use_one_class_probas_cmd='--use_one_class_probas'
          else
            tp_use_one_class_probas_cmd='--no-use_one_class_probas'
          fi

          if [ "${tp_use_weights_from_task_detectors}" == "use_weights" ]; then
            tp_use_weights_from_task_detectors_cmd='--use_weights_from_task_detectors'
          else
            tp_use_weights_from_task_detectors_cmd='--no-use_weights_from_task_detectors'
          fi

          if [ "${tp_auto_detect_tasks}" == "detect" ]; then
            tp_auto_detect_tasks_cmd='--auto_detect_tasks'
          else
            tp_auto_detect_tasks_cmd='--no-auto_detect_tasks'
          fi

          if [ "${tp_use_static_f_ex}" == "use_static_f_ex" ]; then
            tp_use_static_f_ex_cmd='--use_static_f_ex'
          else
            tp_use_static_f_ex_cmd='--no-use_static_f_ex'
          fi

          if [ "${tp_train_nn_using_ex_static_f}" == "train_nn_using_ex_static_f" ]; then
            tp_train_nn_using_ex_static_f_cmd='--train_nn_using_ex_static_f'
          else
            tp_train_nn_using_ex_static_f_cmd='--no-train_nn_using_ex_static_f'
          fi

          if [ "${tp_train_only_the_best_nn}" == "train_only_the_best_nn" ]; then
            tp_train_only_the_best_nn_cmd='--train_only_the_best_nn'
          else
            tp_train_only_the_best_nn_cmd='--no-train_only_the_best_nn'
          fi

          command_args="${command_args} --module MultiMLP --pool_type ${tp_pool_type} --number_of_mpls_to_train ${tp_number_of_nns_to_train} --skip_back_prop_threshold 0.0 --task_detector_type ${tp_p_method} ${tp_reset_tp_cmd} ${tp_use_one_class_probas_cmd} ${tp_use_weights_from_task_detectors_cmd} ${tp_auto_detect_tasks_cmd} --train_task_predictor_at_the_end ${tp_train_task_p_at_end_type} ${tp_use_static_f_ex_cmd} ${tp_train_nn_using_ex_static_f_cmd} ${tp_train_only_the_best_nn_cmd}"
          log_file_name="${log_file_name}_TP_${tp_pool_type}_${tp_number_of_nns_to_train}_${tp_predict_method}_${tp_reset_tp}_${tp_use_one_class_probas}_${tp_use_weights_from_task_detectors}_${tp_auto_detect_tasks}_${tp_train_task_p_at_end_type}_${tp_use_static_f_ex}_${tp_train_nn_using_ex_static_f}_${tp_train_only_the_best_nn}"
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
