# nohup bash train.sh > log 2>&1 &
args=$@
for arg in $args; do
    eval "$arg"
done

echo "seed:        ${seed:=1}"
echo "bert:        ${bert:=roberta-large}"
echo "lr1:         ${lr1:=5e-5}"
echo "lr2:         ${lr2:=5e-6}"
echo "lr3:         ${lr3:=1e-6}"
echo "rate1:       ${rate1:=10}"
echo "rate2:       ${rate2:=10}"
echo "rate3:       ${rate3:=10}"
echo "upsampling:  ${upsampling:=4}"
echo "batch:       ${batch:=100000}"
echo "epochs1:     ${epochs1:=64}"
echo "epochs2:     ${epochs2:=64}"
echo "epochs3:     ${epochs3:=64}"
echo "warmup1:     ${warmup1:=1000}"
echo "warmup2:     ${warmup2:=0}"
echo "warmup3:     ${warmup3:=0}"
echo "glat:        ${glat:=1}"
echo "update:      ${update:=5}"
echo "devices:     ${devices:=0,1,2,3,4,5,6,7}"
echo "config:      ${config:=configs/roberta.yaml}"
echo "path:        ${path:=exp/ctc.roberta}"

code=$path.code
mkdir -p $path.code
cp run.py $code/
cp -r ctc $code/
cp -r 3rdparty $code/
printf "Current commits:\n$(git log -1 --oneline)\n3rd parties:\n"
cd 3rdparty/parser/ && printf "parser\n$(git log -1 --oneline)\n" && cd ../..

for stage in 1 2 3; do
    mkdir -p $path/stage$stage
    var="lr$stage";     lr=${!var}
    var="rate$stage";   rate=${!var}
    var="warmup$stage"; warmup=${!var}
    var="epochs$stage"; epochs=${!var}
    current="$path/stage$stage/model.lr$lr.rate$rate.upsampling$upsampling.batch$batch.epochs$epochs.warmup$warmup.glat$glat.seed$seed"

    if [ $stage -eq 1 ]; then
        train=data/clang8.train
        (set -x
        python -u run.py train -b -s $seed -d $devices -c $config -p $current --lr=$lr  --lr-rate=$rate  --upsampling=$upsampling --batch-size=$batch --epochs=$epochs  --warmup-steps=$warmup  --glat=$glat --update-steps=$update --encoder=bert --bert=$bert --train $train --eval-tgt --cache --amp
        )
    else
        if [ $stage -eq 2 ]; then
            train=data/error_coded.train
        else
            train=data/wi_locness.train
        fi
        (set -x
        cp $prev $current
        python -u run.py train -s $seed -d $devices -c $config -p $current --lr=$lr  --lr-rate=$rate  --upsampling=$upsampling --batch-size=$batch --epochs=$epochs  --warmup-steps=$warmup  --glat=$glat --update-steps=$update --encoder=bert --bert=$bert --train $train --eval-tgt --cache --amp
        )
    fi
    bash pred.sh path=$current
    prev=$current
done
