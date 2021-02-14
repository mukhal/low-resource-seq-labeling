

f= open('sents.txt')



f_out=open('sents.unlabeled', 'w')


for l in f:
    for w in l.split():
        f_out.write(' '.join([w, 'U']) + '\n')

    f_out.write('\n')




