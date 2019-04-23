for model in vgg16_bn #make this the list of models
do
    echo "python main.py  --arch=$model  --save-dir=save_$model |& tee -a log_$model" #print the string on cmd of which command is being executed
    python main.py  --arch=$model  --save-dir=save_$model |& tee -a log_$model #the |& tee redirects console output to a log file
done

#|& is the pipe that puts the output and log information into the log file
#tee keeps the pipe open
#$give variable name to arch and such
#echo executes a command
#for do done are standard for loop syntax