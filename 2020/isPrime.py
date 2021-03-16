def isPrime(num):
    if num == 2: return True
    else:
        for i in range(2,num):
            if num % i == 0: return False
        return True

print([i for i in range(1,200) if isPrime(i)])
    
list2 = [191,193,197,199,221,223,227,229,391,397,401,403]
print([i for i in list2 if isPrime(i)])