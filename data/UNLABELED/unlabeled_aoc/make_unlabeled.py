
f = open('sents.txt')

word_per_line = True

f_out = open('sents.unlbl', 'w')




if word_per_line:

    for l in f:

        if l.strip():
            w, t = l.split()[0], l.split()[-1]
            f_out.write(w + ' ' + 'U' + '\n')
        else:
            f_out.write(l)


