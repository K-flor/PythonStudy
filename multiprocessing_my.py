
# Multiprocessing

from multiprocessing import Pool, Process
import multiprocessing as mp
import os

def foo(x):
    info('function abount '+str(x))
    print(x+100)

def info(title):
    print(title)
    print('module name :', __name__)
    print('parent process :', os.getppid())
    print('process id :', os.getpid())

if __name__ == '__main__' :
	info('main')
	prss = [Process(target=foo, args= (x,)) for x in range(4,10)]
	for p in prss :
		p.start()
	
	for p in prss :
		p.join()
	


'''
	info('main')
    p = Process(target=foo, args=[3])
    p.start()
    p.join()
	info('main')
	pools = Pool(10)
	pools.map(foo, [1,2,3,4,5,6,7,8,9,10])
	pools.close()
	pools.join()
'''