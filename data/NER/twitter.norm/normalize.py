from araNorm import araNorm

norm = araNorm()

f = open('test.txt', encoding='utf-8')
f_out = open('test.txt.norm', 'w+', encoding='utf-8')


for line in f:
    if not line.strip():
        f_out.write('\n')
        continue
    elif '.' in line:
        f_out.write(line)
    
    else: 
        w, t = line.split()

        w = norm.run(w, False, 1)
        if w.strip():
            f_out.write(' '.join([w, t]) + '\n')

