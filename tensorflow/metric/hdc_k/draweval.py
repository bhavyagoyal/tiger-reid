import sys
import os
import math
import argparse
import cPickle
import numpy as np
import pprint

import matplotlib.pyplot as plt
#matplotlib.use('Agg')  # to avoid X display error


#styles = ['b-', 'b:', 'g-', 'g:', 'r-', 'r:', 'c-', 'c:', 'm-', 'm:', 'y-', 'y:', 'k-', 'k:']
base_colors = ['blue', 'green', 'red', 'darkcyan', 'magenta',
            'gold', 'black', 'tomato', 'gray', 'olive',
            'indigo', '#CCCCCC'] * 10
#styles = [':', '-', '--', '-.'] * len(base_colors)
#colors = [ base_colors[i//4] for i in range(len(base_colors)*4) ] 
styles = [':', '-'] * len(base_colors)
colors = [ base_colors[i//2] for i in range(len(base_colors)*2) ] 

methods = [
    ('EX1 train',  'ex1/save/evalc_train.dmp'), ('EX1 test',  'ex1/save/evalc_test.dmp') , 
    ('EX2 train',  'ex2/save/evalc_train.dmp'), ('EX2 test',  'ex2/save/evalc_test.dmp') , 
    ('EX3 train',  'ex3/save/evalc_train.dmp'), ('EX3 test',  'ex3/save/evalc_test.dmp') , 
    ('EX4 train',  'ex4/save/evalc_train.dmp'), ('EX4 test',  'ex4/save/evalc_test.dmp') , 
    ('EX5 train',  'ex5/save/evalc_train.dmp'), ('EX5 test',  'ex5/save/evalc_test.dmp') , 
    ('EX6 train',  'ex6/save/evalc_train.dmp'), ('EX6 test',  'ex6/save/evalc_test.dmp') , 
    ('EX7 train',  'ex7/save/evalc_train.dmp'), ('EX7 test',  'ex7/save/evalc_test.dmp') , 
    ('EX8 train',  'ex8/save/evalc_train.dmp'), ('EX8 test',  'ex8/save/evalc_test.dmp') , 
    ('EX9 train',  'ex9/save/evalc_train.dmp'), ('EX9 test',  'ex9/save/evalc_test.dmp') , 
    ('EX10 train', 'ex10/save/evalc_train.dmp'),('EX10 test', 'ex10/save/evalc_test.dmp') , 
    ('EX11 train', 'ex11/save/evalc_train.dmp'),('EX11 test', 'ex11/save/evalc_test.dmp') , 
    ('EX12 train', 'ex12/save/evalc_train.dmp'),('EX12 test', 'ex12/save/evalc_test.dmp') , 
    ]


def load(methods):
    # db is a dictionary which contains all retrieval ranks at each iteration.
    # ex) { (0: [3,2,7,...] ), (100: [3,2,7,...] ), ... }
    return [ (name, cPickle.load(open(path))) for name,path in methods if os.path.exists(path) ]       

def draw_rank(result_db, skipzero=True, ncol=1):
    print "Drawing 'Average rank vs. Iteration' ..."
    print "Best values"
    fig = plt.figure()
    for s, (name, db) in enumerate(result_db):
        x, y = zip( *sorted( [ (i[0], i[1].mean()) for i in db.iteritems() ] ) )
        if skipzero:
            x, y =x[1:], y[1:]
        best_idx = np.argmin(np.array(y))
        print "'%s'\tat\t%d\t%f"%( name, x[best_idx], y[best_idx] )
        plt.plot( x,y, styles[s], color=colors[s], label=name)

    plt.title('Average rank vs. Iteration')
    plt.ylabel('Average rank')
    plt.xlabel('Iteration')
    if ncol == 2 :
        handles, labels = plt.gca().get_legend_handles_labels()
        handles = handles[::2] + handles[1::2]
        labels = labels[::2] + labels[1::2]
        plt.gca().legend(handles, labels, loc='lower right', ncol=ncol, columnspacing=1, frameon=True ) 
    else:
        plt.legend(loc='lower right', ncol=ncol, columnspacing=1, frameon=True ) 
    plt.grid() 
    plt.gcf().set_facecolor('white') 
    return fig

def draw_acc(result_db, topk, skipzero=True, ncol=1):
    print "Drawing 'Top %d accuracy vs. Iteration' ..."%topk
    print "Best values"
    fig = plt.figure()
    bvals = []
    for s, (name, db) in enumerate(result_db):
        x, y = zip( *sorted( [ (i[0], (i[1]<=topk).mean()) for i in db.iteritems() ] ) )
        if skipzero:
            x, y =x[1:], y[1:]
        best_idx = np.argmax(np.array(y))
        print "'%s'\tat\t%d\t%f"%( name, x[best_idx], y[best_idx] )
        bvals.append( (x[best_idx], y[best_idx]) )
        plt.plot( x,y, styles[s], color=colors[s], label=name)

    ano = lambda i: plt.annotate( "%0.4f"%bvals[i][1], bvals[i], xytext=(2, 2), textcoords='offset points', color=colors[i])
    [ ano(i) for i in range(len(bvals)) ]
    X, Y = zip(*bvals)
    plt.scatter( X, Y, c =colors, alpha=0.5, marker='.' )
    plt.title('Top %d accuracy vs. Iteration'%topk)
    plt.ylabel('Top %d accuracy'%topk)
    plt.xlabel('Iteration')
    if ncol == 2 :
        handles, labels = plt.gca().get_legend_handles_labels()
        handles = handles[::2] + handles[1::2]
        labels = labels[::2] + labels[1::2]
        plt.gca().legend(handles, labels, loc='lower right', ncol=ncol, columnspacing=1, frameon=True ) 
    else:
        plt.legend(loc='lower right', ncol=ncol, columnspacing=1, frameon=True ) 
    plt.grid() 
    plt.gcf().set_facecolor('white') 
    return fig

def draw_topk(result_db, topk_iter, topk, skipzero=True, ncol=1):
    print "Drawing 'Accuracy vs. Top K' at %s iteration..." % topk_iter
    fig = plt.figure()
    for s, (name, db) in enumerate(result_db):
        if topk_iter == 'last':
            pos = max(db.iterkeys())
        elif topk_iter == 'best':
            x, y = zip( *sorted( [ (i[0], (i[1]<=topk).mean()) for i in db.iteritems() ] ) )
            if skipzero:
                x, y =x[1:], y[1:]
            best_idx = np.argmax(np.array(y))
            pos = x[best_idx]
        else:
            pos = eval(topk_iter)
        y= [ (db[pos] <= k).mean() for k in xrange(1,101) ] 
        plt.plot(xrange(1,101), y, styles[s], color=colors[s], label=name)

    plt.title('Accuracy vs. Top K at %s iteration'%topk_iter)
    plt.ylabel('Accuracy')
    plt.xlabel('Top K')
    plt.xscale('log')
    plt.xticks( [1,2,3,4,5,10,100], [1,2,3,4,5,10,100] )
    if ncol == 2:
        handles, labels = plt.gca().get_legend_handles_labels()
        handles = handles[::2] + handles[1::2]
        labels = labels[::2] + labels[1::2]
        plt.gca().legend(handles, labels, loc='lower right', ncol=ncol, frameon=True ) 
    else:
        plt.legend(loc='lower right', ncol=ncol, columnspacing=1 , frameon=True ) 
    plt.grid()
    plt.gcf().set_facecolor('white')  
    return fig

def save_all_figs(save_plot):
    if not save_plot:
        return
    if os.path.exists(save_plot):
        if raw_input( "'%s' already exists. Overwrite (y/N)? "%(save_plot) ) != 'y':
            return

    all_figs = list(map(plt.figure, plt.get_fignums()))
    #all_axes = [ (f, f.get_axes()) for f in  all_figs ]
    print "saving the plot to '%s'"%(save_plot)
    cPickle.dump( all_figs, open(save_plot,"wb") , -1 )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m','--methods', help='python list of (name,path)')
    parser.add_argument('-k','--topk', help='top k for accuracy graph', type=int, default=1)
    parser.add_argument('-p','--topk_iter', help='iteration at which the top k is drawn, default=best', default='best')
    parser.add_argument('-a','--draw_all', help='draw all three graphs', action='store_true')
    parser.add_argument('-z','--draw_zero', help='draw zero iteration', action='store_true')
    parser.add_argument('-c','--legend_ncol', help='# of columns in legend', type=int, default=1 )
    parser.add_argument('-s','--save_plot', help='file to save the plot' )
    args = parser.parse_args()

    args.methods = eval(args.methods) if args.methods else methods
    print '********************************************************************'
    print '\n'.join([ str((name,path)) for name,path in args.methods if os.path.exists(path) ])
    print '********************************************************************'

    result_db = load(args.methods)
    draw_acc(result_db, args.topk, not args.draw_zero, args.legend_ncol)
    if args.draw_all :
        draw_topk(result_db, args.topk_iter, args.topk, not args.draw_zero, args.legend_ncol)
        draw_rank(result_db, not args.draw_zero, args.legend_ncol)

    plt.rcParams["savefig.directory"] =os.getcwd()

    save_all_figs( args.save_plot)

    plt.show()

if __name__ == "__main__":
    main()
