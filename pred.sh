args=$@
for arg in $args; do
    eval "$arg"
done

echo "config:    ${config:=configs/roberta.yaml}"
echo "path:      ${path:=exp/ctc.roberta/model}"
echo "data:      ${data:=data/conll14.test}"
echo "pred:      ${pred:=$path.conll14.test.pred}"
echo "input:     ${input:=data/conll14.test.input}"
echo "errant:    ${errant:=data/conll14.test.errant.m2}"
echo "devices:   ${devices:=0}"
echo "batch:     ${batch:=10000}"
echo "beam:      ${beam:=12}"
echo "iteration: ${iteration:=2}"

(set -x; python -u run.py predict -d $devices -c $config -p $path --data $data --pred $pred --batch-size=$batch --beam-size=$beam --iteration $iteration
CUDA_VISIBLE_DEVICES=$devices python recover.py --hyp $pred -o $pred.out -i $input -p $path -m 62)

if ! conda env list | grep -q "^py27"; then
    echo "Creating the py27 environment..."; conda create -n py27 -y python=2.7
fi

source ~/anaconda3/etc/profile.d/conda.sh
conda activate py27
python tools/m2scorer/scripts/m2scorer.py -v $pred.out data/conll14.test.m2 > $pred.m2scorer.log
tail -n 9 $pred.m2scorer.log
conda deactivate