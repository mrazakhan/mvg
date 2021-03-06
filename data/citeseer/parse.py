#!/usr/bin/python

fpfx = 'citeseer'
idmap = {}
haslabel = {}
hasword = {}
labels = {'Agents':1,'AI':2,'DB':3,'IR':4,'ML':5,'HCI':6}

f = open('{}.content'.format(fpfx),'r')
line = f.readline()
id = 0
while line != '':
	row = line.strip().split('\t')
	id += 1
	idmap[row[0]] = id
	lab = row[len(row)-1]
	haslabel[id] = labels[lab]
	hasword[id] = {}
	for i in range(1,len(row)-1):
		if row[i] == '1':
			hasword[id][i] = 1
	line = f.readline()
f.close()

f = open('{}.labels'.format(fpfx),'w')
for id in haslabel:
	f.write('{}\t{}\n'.format(id,haslabel[id]))
f.close()

f = open('{}.words'.format(fpfx),'w')
for id in hasword:
	f.write('{}\t'.format(id))
	for w in hasword[id]:
		f.write('{}:1 '.format(w))
	f.write('\n')
f.close()

fi = open('{}.cites'.format(fpfx),'r')
fo = open('{}.links'.format(fpfx),'w')
fo2 = open('{}.cites2'.format(fpfx),'w')
line = fi.readline()
while line != '':
	row = line.strip().split('\t')
	if row[0] in idmap and row[1] in idmap:
		id1 = idmap[row[0]]
		id2 = idmap[row[1]]
		if id1 != id2:
			fo.write('{}\t{}\n'.format(id1,id2))
			fo2.write('{}\t{}\n'.format(row[0],row[1]))
	line = fi.readline()
fi.close()
fo.close()
fo2.close()
f = open('{}.ids'.format(fpfx),'w')
for origid in sorted(idmap):
	f.write('{}\t{}\n'.format(origid,idmap[origid]))
f.close()
