######################## NAIVE Solution #####################
def isPrime(n):
    if n==1:
        return False
    i=2
    while i*i<=n:
        if(n%i==0):
            return False
        i+=1
    return True

def allPrime(n):
    for i in range(1,n+1):
        if(isPrime(i)):
            print(i, end="-")
############################## Better Solution #########################################

def seive(n):
    if n==1:
        return
    
    isPrime=[True]*(n+1)
    i=2

    while i<=n:                                                   #Only choosing numbers bigger than i as numbers smaller than that are already covered when I was smaller
        if isPrime[i]:                                              #All values are set as 'True' i.e. prime. i=2 is indeed prime, so we start from there
            print(i, end=" ")                                        #printing in ascending values of i
            for j in range (i*i,n+1,i):                             #Starting from i multiple of i because all smaller multiples are covered by higher multiples of smaller values or i
                isPrime[j]=False                                    #For eg of the last statement : starting from 3x3 because 3x2 is covered by 2x3 already
        i+=1





if __name__=="__main__":
    num=int(input("Enter the number here : "))
    allPrime(num)
    #seive(num)
   