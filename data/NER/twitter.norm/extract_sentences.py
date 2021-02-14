
split = 'valid'

f = open('{}.txt'.format(split))

cur_sent=[]
cur_lbl=[]

all_sents=[]
all_labels=[]

for line in f:
    print(line)

    if line.strip():
        k = line.split()
        cur_sent.append(k[0].strip())
        cur_lbl.append(k[-1].strip())
    
    else :
        all_sents.append(cur_sent)
        all_labels.append(cur_lbl)

        cur_sent=[]
        cur_lbl=[]


f1=open('{}_sents.txt'.format(split), 'w+')
f2=open('{}_labels.txt'.format(split), 'w+')

for s, l in zip(all_sents, all_labels):

    s=' '.join(s)
    l = ' '.join(l)
    f1.write(s.strip() + '\n')
    f2.write(l.strip() + '\n')




