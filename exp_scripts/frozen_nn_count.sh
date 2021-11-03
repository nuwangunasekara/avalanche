d="$1"
dataset=(CORe50 RotatedCIFAR10 RotatedMNIST)


for (( i=0; i<${#dataset[@]}; i++ ))
do
  ds="${dataset[$i]}"
  avg_nn=`echo "scale=2; $(find $d -name "*_nn*" | grep "${ds}" |wc -l) / 3" | bc`
  echo "$(basename $d),${ds},$avg_nn"
done
