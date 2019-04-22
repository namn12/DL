for model in vgg16_bn
do
    echo "python main.py  --arch=$model  --save-dir=save_$model |& tee -a log_$model"
    python main.py  --arch=$model  --save-dir=save_$model |& tee -a log_$model
done

#|& is the pipe that puts the output and log information into the log file