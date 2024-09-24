def max(l,n):

    if not l:
        return None
    else :
        max=l[0]
        for i in l:
            if i>max:
                max=i
        
        return max

def secMax(l,n):

    if not l:
        return None
    else :
        max=l[0]
        sec=None

        for i in l:
            if i>max:
                sec=max
                max=i
            elif i!=max:
                if sec==None or i>sec:
                    sec=i
        
        print('Second Max is : ',sec)


if __name__ == "__main__":
    n=int(input("Enter the number : "))
    l=[]
    for i in range(n):
        l.append(int(input("Enter next int : ")))
    
    #print(max(l,n))
    secMax(l,n)