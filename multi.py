from math import ceil,floor

def karatsuba(x,y):
	if(x<10 and y<10):
		return x*y

	n=max(len(str(x)),len(str(y)))
	mid=int(ceil(float(n)/2))	

	xa=int(floor(x/10**mid))
	xb=int(x%10**mid)

	yc=int(floor(y/10**mid))
	yd=int(y%10**mid)

	left=karatsuba(xa,yc)
	right=karatsuba(xb,yd)
	midterm=karatsuba(xa+xb,yc+yd)-left-right

	# return (10**2*mid)*left+(10**mid)*midterm+right
	return int((10**2*mid)*left+(10**(mid))*midterm+right)

print(karatsuba(121,362))	

#print(karatsuba(3141592653589793238462643383279502884197169399375105820974944592,2718281828459045235360287471352662497757247093699959574966967627))






