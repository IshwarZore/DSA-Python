#Modification of Prime

def Fact(n):
    
    if(n==1):
        print(n)
        return

    i=1
    while i*i<n:
        if(n%i==0):
            print (i)
            if(n//i!=i):
                print(n//i)
        i+=1


def FactAsc(n):
    
    i=1
    while i*i<n:
        if(n%i==0):
            print (i)
        i+=1

    while i>=1:
        if(n%i==0):
            if(n//i!=(i-1)):                #Because it gets repeated otherwise
                print (n//i)
        i-=1



if __name__=="__main__":
    num=int(input("Enter the number here : "))
    Fact(num)           #All Factors
 #   FactAsc(num)       #All Factors in ascending order