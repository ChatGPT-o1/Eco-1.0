from decimal import Decimal as D
print('45', D('7284')*D('59.6'), D(7284*60), D('7284')*D('0.4'), D(437040)-D('2913.6'))
print('46', 3508*4219, 3508*4000, 3508*219, 14032000+768252)
print('47', D('12.75')*D('1846.4'), D('1846.4')*12, D('1846.4')*D('0.75'), D('22156.8')+D('1384.8'))
print('48', D(9063)*D('0.0825'), D(9063)*D('0.08'), D(9063)*D('0.0025'), D('725.04')+D('22.6575'))
print('10', D(7365)*D('48.27')); print('11', 6482*3917); print('12', D('1284.5')*D('3067.25')); print('13', D(3847)*D('1.609344')); print('14', 58219*7406)
try:
    import sympy
except ImportError:
    sympy=None
def isprime(n):
    if sympy: return sympy.isprime(n)
    # Miller-Rabin deterministic for < 3.3e24
    if n<2: return False
    for p in [2,3,5,7,11,13,17,19,23,29,31,37,41]:
        if n%p==0: return n==p
    d=n-1; s=0
    while d%2==0: d//=2; s+=1
    for a in [2,3,5,7,11,13,17,19,23,29,31,37,41]:
        x=pow(a,d,n)
        if x in (1,n-1): continue
        for _ in range(s-1):
            x=pow(x,2,n)
            if x==n-1: break
        else: return False
    return True
for n in [4360258552782731,6396154357928470559,2405459884486899763,949597301950332195533,4225174690031441409749]:
    small=[p for p in [2,3,5,7,11,13,17,19,23,29,31,37,41,43,47] if n%p==0]
    print(n, 'digits',len(str(n)),'digsum',sum(map(int,str(n))),'digsum%3',sum(map(int,str(n)))%3,'small factors',small,'prime',isprime(n))
print('sympy available', sympy is not None)
