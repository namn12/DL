'''
Plots the topk precision of assembled networks
Modified from https://github.com/chengyangfu/pytorch-vgg-cifar10
'''

import matplotlib
import numpy as np
import matplotlib.pyplot as plt

def load_log(file):
    '''Reads log file top to bottom and extract the strings of prec@1 values for figure'''
    results = [] #stores all the precisions
    with open(file) as f: #opens the whole txt/log file
        for line in f: #iterate line by line
            if '* Prec@1' in line: #this is the average accuracy of the whole validation set of batches 
                line = line.split() #splits each token in the string
                results.append(float(line[-1])) #append the last token in the line as a float
    return results

shows = {} 
shows['vgg16_bn'] = load_log('log_vgg16_bn') #only have one model in this case


for key in sorted(shows.keys()): #return lables of dictionary, not values
    epochs = np.arange(1, 1+len(shows[key])) #each key corresponds to each architerctures, which each have an...
#...array of precisions as value

    plt.plot(epochs, shows[key], label='{}:\n max Prec@1: {}'.format(key, np.max(shows[key])))
    
#plt.legend(shows.keys(), loc='upper left')
plt.rcParams['figure.figsize'] = (12,9)
lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.xlabel('Number of Epcohs')
plt.ylabel('Top 1 Precision')
plt.savefig('vgg16_Prec@1.png', bbox_extra_artists=(lgd,), bbox_inches='tight')
