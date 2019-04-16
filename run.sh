for model in vgg16 vgg16_bn
do
    echo "python main.py  --arch=$model  --save-dir=save_$model |& tee -a log_$model"
    python main.py  --arch=$model  --save-dir=save_$model |& tee -a log_$model
done

for model in vgg16 vgg16_bn
do
    echo "python main.py  --arch=$model --half --save-dir=save_half_$model |& tee -a log_half_$model"
    python main.py  --arch=$model --half --save-dir=save_half_$model |& tee -a log_half_$model
done